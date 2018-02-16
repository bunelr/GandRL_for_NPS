import itertools
import torch
import torch.autograd as autograd
from torch.autograd import Variable


def do_supervised_minibatch(model,
                            # Source
                            inp_grids, out_grids,
                            # Target
                            in_tgt_seq, in_tgt_seq_list, out_tgt_seq,
                            # Criterion
                            criterion):

    # Get the log probability of each token in the ground truth sequence of tokens.
    decoder_logit, _ = model(inp_grids, out_grids, in_tgt_seq, in_tgt_seq_list)

    nb_predictions = torch.numel(out_tgt_seq.data)
    # criterion is a weighted CrossEntropyLoss. The weights are used to not penalize
    # the padding prediction used to make the batch of the appropriate size.
    loss = criterion(
        decoder_logit.contiguous().view(nb_predictions, decoder_logit.size(2)),
        out_tgt_seq.view(nb_predictions)
    )

    # Do the backward pass over the loss
    loss.backward()

    # Return the value of the loss over the minibatch for monitoring
    return loss.data[0]

def do_syntax_weighted_minibatch(model,
                                 # Source
                                 inp_grids, out_grids,
                                 # Target
                                 in_tgt_seq, in_tgt_seq_list, out_tgt_seq,
                                 # Criterion
                                 criterion,
                                 # Beta trades off between CrossEntropyLoss
                                 # and SyntaxLoss
                                 beta):

    # Get the log probability of each token in the ground truth sequence of tokens,
    # for both the IO based model and the syntax checker.
    decoder_logit, syntax_logit = model(inp_grids, out_grids,
                                        in_tgt_seq, in_tgt_seq_list)
    # {decoder,syntax}_logit: batch_size x seq_len x vocab_size

    # The criterion is the same as in `do_supervised_minibatch`
    nb_predictions = torch.numel(out_tgt_seq.data)
    ce_loss = criterion(
        decoder_logit.contiguous().view(nb_predictions, decoder_logit.size(2)),
        out_tgt_seq.view(nb_predictions)
    )

    # A syntax loss is also computed, penalizing any masking of tokens that we
    # know are valid, given that they are in the ground truth sequence.
    syntax_loss = -syntax_logit.gather(2, out_tgt_seq.unsqueeze(2)).sum()

    loss = ce_loss + beta*syntax_loss
    # Do the backward pass over the loss
    loss.backward()

    # Return the value of the loss over the minibatch for monitoring
    return loss.data[0]

def do_rl_minibatch(model,
                    # Source
                    inp_grids, out_grids,
                    # Target
                    envs,
                    # Config
                    tgt_start_idx, tgt_end_idx, max_len,
                    nb_rollouts):

    # Samples `nb_rollouts` samples from the decoding model.
    rolls = model.sample_model(inp_grids, out_grids,
                               tgt_start_idx, tgt_end_idx, max_len,
                               nb_rollouts, vol=False)
    for roll, env in zip(rolls, envs):
        # Assign the rewards for each sample
        roll.assign_rewards(env, [])

    # Evaluate the performance on the minibatch
    batch_reward = sum(roll.dep_reward for roll in rolls)

    # Get all variables and all gradients from all the rolls
    variables, grad_variables = zip(*batch_rolls_reinforce(rolls))

    # For each of the sampling probability, we know their gradients.
    # See https://arxiv.org/abs/1506.05254 for what we are doing,
    # simply using the probability of the choice made, times the reward of all successors.
    autograd.backward(variables, grad_variables)

    # Return the value of the loss/reward over the minibatch for convergence
    # monitoring.
    return batch_reward

def do_rl_minibatch_two_steps(model,
                              # Source
                              inp_grids, out_grids,
                              # Target
                              envs,
                              # Config
                              tgt_start_idx, tgt_end_idx, pad_idx, max_len,
                              nb_rollouts, rl_inner_batch):
    '''
    This is an alternative to do simple expected reward maximization.
    The problem with the previous method of `do_rl_minibatch` is that it is
    memory demanding, as due to all the sampling steps / bookkeeping, the graph
    becomes large / complex. It's entirely possible that future version of pytorch
    will fix this but this has proven quite useful.

    The idea is to first sample the `nb_rollouts` samples, doing all the process with
    Volatile=True, so that no graph needs to be held.
    Once the samples have been sampled, we re-evaluate them through the
    `score_multiple_decs` functions that returns the log probabilitise of several decodings.
    This has the disadvantage that we don't make use of shared elements in the sequences,
    but that way, the decoding graph is much simpler (scoring of the whole decoded sequence
    at once vs. one timestep at a time to allow proper input feeding.)
    '''
    use_cuda = inp_grids.is_cuda
    tt = torch.cuda if use_cuda else torch
    rolls = model.sample_model(inp_grids, out_grids,
                               tgt_start_idx, tgt_end_idx, max_len,
                               nb_rollouts, vol=True)
    for roll, env in zip(rolls, envs):
        # Assign the rewards for each sample
        roll.assign_rewards(env, [])
    batch_reward = sum(roll.dep_reward for roll in rolls)

    for start_pos in range(0, len(rolls), rl_inner_batch):
        roll_ib = rolls[start_pos: start_pos + rl_inner_batch]
        to_score = []
        nb_cand_per_sp = []
        rews = []
        for roll in roll_ib:
            nb_cand = 0
            for trajectory, multiplicity, _, rew in roll.yield_final_trajectories():
                to_score.append(trajectory)
                rews.append(multiplicity*rew)
                nb_cand += 1
            nb_cand_per_sp.append(nb_cand)

        in_tgt_seqs = []
        lines = [[tgt_start_idx] + line  for line in to_score]
        lens = [len(line) for line in lines]
        ib_max_len = max(lens)
        inp_lines = [
            line[:ib_max_len-1] + [pad_idx] * (ib_max_len - len(line[:ib_max_len-1])-1) for line in lines
        ]
        out_lines = [
            line[1:] + [pad_idx] * (ib_max_len - len(line)) for line in lines
        ]
        in_tgt_seq = Variable(torch.LongTensor(inp_lines))
        out_tgt_seq = Variable(torch.LongTensor(out_lines))
        if use_cuda:
            in_tgt_seq, out_tgt_seq = in_tgt_seq.cuda(), out_tgt_seq.cuda()
        out_care_mask = (out_tgt_seq != pad_idx)

        inner_batch_in_grids = inp_grids.narrow(0, start_pos, rl_inner_batch)
        inner_batch_out_grids = out_grids.narrow(0, start_pos, rl_inner_batch)

        tkns_lpb = model.score_multiple_decs(inner_batch_in_grids,
                                             inner_batch_out_grids,
                                             in_tgt_seq, inp_lines,
                                             out_tgt_seq, nb_cand_per_sp)
        # tkns_lpb contains the log probability of each choice that was taken

        tkns_pb = tkns_lpb.exp()
        # tkns_lpb contains the probability of each choice that was taken

        # The gradient on the probability of a multinomial choice is
        # 1/p * (rewards after)
        tkns_invpb = tkns_pb.data.reciprocal()

        # The gradient on the probability will get multiplied by the reward We
        # also take advantage of the fact that we have a mask to avoid putting
        # gradients on padding
        rews_tensor = tt.FloatTensor(rews).unsqueeze(1).expand_as(tkns_invpb)
        reinforce_grad = torch.mul(-rews_tensor, out_care_mask.data.float())
        torch.mul(reinforce_grad, tkns_invpb, out=reinforce_grad)

        tkns_pb.backward(reinforce_grad)

    return batch_reward


def do_beam_rl(model,
               # source
               inp_grids, out_grids, targets,
               # Target
               envs, reward_comb_fun,
               # Config
               tgt_start_idx, tgt_end_idx, pad_idx,
               max_len, beam_size, rl_inner_batch, rl_use_ref):
    '''
    Rather than doing an actual expected reward,
    evaluate the most likely programs using a beam search (with `beam_sample`)
    If `rl_use_ref` is set, include the reference program in the search.
    Similarly to `do_rl_minibatch_two_steps`, first decode the programs as Volatile,
    then score them.
    '''
    batch_reward = 0
    use_cuda = inp_grids.is_cuda
    tt = torch.cuda if use_cuda else torch
    vol_inp_grids = Variable(inp_grids.data, volatile=True)
    vol_out_grids = Variable(out_grids.data, volatile=True)
    # Get the programs from the beam search
    decoded = model.beam_sample(vol_inp_grids, vol_out_grids,
                                tgt_start_idx, tgt_end_idx, max_len,
                                beam_size, beam_size)

    # For each element in the batch, get the version of the log proba that can use autograd.
    for start_pos in range(0, len(decoded), rl_inner_batch):
        to_score = decoded[start_pos: start_pos + rl_inner_batch]
        scorers = envs[start_pos: start_pos + rl_inner_batch]
        # Eventually add the reference program
        if rl_use_ref:
            references = [target for target in targets[start_pos: start_pos + rl_inner_batch]]
            for ref, candidates_to_score in zip(references, to_score):
                for _, predded in candidates_to_score:
                    if ref == predded:
                        break
                else:
                    candidates_to_score.append((None, ref)) # Don't know its lpb

        # Build the inputs to be scored
        nb_cand_per_sp = [len(candidates) for candidates in to_score]
        in_tgt_seqs = []
        preds  = [pred for lp, pred in itertools.chain(*to_score)]
        lines = [[tgt_start_idx] + line  for line in preds]
        lens = [len(line) for line in lines]
        ib_max_len = max(lens)

        inp_lines = [
            line[:ib_max_len-1] + [pad_idx] * (ib_max_len - len(line[:ib_max_len-1])-1) for line in lines
        ]
        out_lines = [
            line[1:] + [pad_idx] * (ib_max_len - len(line)) for line in lines
        ]
        in_tgt_seq = Variable(torch.LongTensor(inp_lines))
        out_tgt_seq = Variable(torch.LongTensor(out_lines))
        if use_cuda:
            in_tgt_seq, out_tgt_seq = in_tgt_seq.cuda(), out_tgt_seq.cuda()
        out_care_mask = (out_tgt_seq != pad_idx)

        inner_batch_in_grids = inp_grids.narrow(0, start_pos, len(to_score))
        inner_batch_out_grids = out_grids.narrow(0, start_pos, len(to_score))

        # Get the scores for the programs we decoded.
        seq_lpb_var = model.score_multiple_decs(inner_batch_in_grids,
                                                inner_batch_out_grids,
                                                in_tgt_seq, inp_lines,
                                                out_tgt_seq, nb_cand_per_sp)
        lpb_var = torch.mul(seq_lpb_var, out_care_mask.float()).sum(1)

        # Compute the reward that were obtained by each of the sampled programs
        per_sp_reward = []
        for env, all_decs in zip(scorers, to_score):
            sp_rewards = []
            for (lpb, dec) in all_decs:
                sp_rewards.append(env.step_reward(dec, True))
            per_sp_reward.append(sp_rewards)

        per_sp_lpb = []
        start = 0
        for nb_cand in nb_cand_per_sp:
            per_sp_lpb.append(lpb_var.narrow(0, start, nb_cand))
            start += nb_cand

        # Use the reward combination function to get our loss on the minibatch
        # (See `reinforce.py`, possible choices are RenormExpected and the BagExpected)
        inner_batch_reward = 0
        for pred_lpbs, pred_rewards in zip(per_sp_lpb, per_sp_reward):
            inner_batch_reward += reward_comb_fun(pred_lpbs, pred_rewards)

        # We put a minus sign here because we want to maximize the reward.
        (-inner_batch_reward).backward()

        batch_reward += inner_batch_reward.data[0]

    return batch_reward


def batch_rolls_reinforce(rolls):
    for roll in rolls:
        for var, grad in roll.yield_var_and_grad():
            if grad is None:
                assert var.requires_grad is False
            else:
                yield var, grad
