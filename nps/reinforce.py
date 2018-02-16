# External imports
import torch
import numpy as np
import torch.nn.functional as F

from torch.autograd import Variable


class Rolls(object):

    def __init__(self, action, proba, multiplicity, depth):
        self.successor = {}
        # The action that this node in the tree corresponds to
        self.action = action  # -> what was the sample taken here
        self.proba = proba  # -> Variable containing the log proba of
                            # taking this action
        self.multi_of_this = multiplicity  # -> How many times was this
                                              # prefix (until this point of
                                              # the sequence) seen
        self.depth = depth  # -> How far along are we in the sequence

        # Has no successor to this sample
        self.is_final = True

        # This is the reward that would be obtained by doing this prefix once.
        # This is only to use for bookkeeping.
        self.own_reward = 0
        # The one to use to compute gradients is the following.

        # This contains `self.own_reward * self.multi_of_this` + sum of all the
        # dependents self.dep_reward
        self.dep_reward = 0

    def expand_samples(self, trajectory, end_multiplicity, end_proba):
        '''
        The assumption here is that all but the last steps of the trajectory
        have already been created.
        '''
        assert(len(trajectory) > 0)

        pick = trajectory[0]
        if pick in self.successor:
            self.successor[pick].expand_samples(trajectory[1:],
                                                end_multiplicity,
                                                end_proba)
        else:
            # We add a successor so we are necessarily not final anymore
            self.is_final = False
            # We don't expand the samples by several steps at a time so verify
            # that we are done
            assert(len(trajectory) == 1)
            self.successor[pick] = Rolls(pick, end_proba,
                                         end_multiplicity,
                                         self.depth + 1)

    def yield_final_trajectories(self):
        '''
        Yields 3-tuples:
        -> Trajectory
        -> Multiplicity of this trajectory
        -> Proba of this trajectory
        -> Final reward of this trajectory
        '''
        if self.is_final:
            yield [], self.multi_of_this, self.proba, self.own_reward
        else:
            for key, succ in self.successor.items():
                for final_traj, multi, proba_suffix, reward \
                    in succ.yield_final_trajectories():
                    yield ([key] + final_traj,
                           multi,
                           self.proba * proba_suffix,
                           reward)

    def yield_var_and_grad(self):
        '''
        Yields 2-tuples:
        -> Proba: Variable correponding to the proba of this last choice
        -> Grad: Gradients for each of those variables
        '''
        for succ in self.successor.values():
            for var, grad in succ.yield_var_and_grad():
                yield var, grad
        yield self.proba, self.reinforce_gradient()

    def assign_rewards(self, reward_assigner, trace):
        '''
        Using the `reward_assigner` scorer, go depth first to assign the
        reward at each timestep, and then collect back all the "depending
        rewards"
        '''
        if self.depth == -1:
            # This is the root from which all the samples come from, ignore
            pass
        else:
            # Assign to this step its own reward
            self.own_reward = reward_assigner.step_reward(trace,
                                                          self.is_final)

        # Assign their own score to each of the successor
        for next_step, succ in self.successor.items():
            new_trace = trace + [next_step]
            succ.assign_rewards(reward_assigner, new_trace)

        # If this is a final node, there is no successor, so I can already
        # compute the dep-reward.
        if self.is_final:
            self.dep_reward = self.multi_of_this * self.own_reward
        else:
            # On the other hand, all my child nodes have already computed their
            # dep_reward so I can collect them to compute mine
            self.dep_reward = self.multi_of_this * self.own_reward
            for succ in self.successor.values():
                self.dep_reward += succ.dep_reward

    def reinforce_gradient(self):
        '''
        At each decision, compute a reinforce gradient estimate to the
        parameter of the probability that was sampled from.
        '''
        if self.depth == -1:
            return None
        else:
            # We haven't put in a baseline so just ignore this
            baselined_reward = self.dep_reward
            grad_value = baselined_reward / (1e-6 + self.proba.data)

            # We return a negative here because we want to maximize the rewards
            # And the pytorch optimizers try to minimize them, so this put them
            # in agreement
            return -grad_value


class Environment(object):

    def __init__(self, reward_norm, environment_data):
        '''
        reward_norm: float -> Value of the reward for correct answer
        environment_data: anything -> Data/Ground Truth to use for the reward evaluation


        To create different types of reward, subclass it and modify the
        `should_skip_reward` and `reward_value` function.
        '''
        self.reward_norm = reward_norm
        self.environment_data = environment_data

    def step_reward(self, trace, is_final):
        '''
        trace: List[int] -> all prediction of the sample to score.
        is_final: bool -> Is the sample finished.
        '''
        if self.should_skip_reward(trace, is_final):
            return 0
        else:
            return self.reward_value(trace, is_final)

    def should_skip_reward(self, trace, is_final):
        raise NotImplementedError

    def reward_value(self, trace, is_final):
        raise NotImplementedError


class MultiIO01(Environment):
    '''
    This only gives rewards at the end of the prediction.
    +1 if the two programs lead to the same final state.
    -1 if the two programs lead to different outputs
    '''

    def __init__(self, reward_norm,
                 target_program, input_worlds, output_worlds, simulator):
        '''
        reward_norm: float
        input_grids, output_grids: Reference IO for the synthesis
        '''
        super(MultiIO01, self).__init__(reward_norm,
                                        (target_program,
                                         input_worlds,
                                         output_worlds,
                                         simulator))
        self.target_program = target_program
        self.input_worlds = input_worlds
        self.output_worlds = output_worlds
        self.simulator = simulator

        # Make sure that the reference program works for the IO given
        parse_success, ref_prog = self.simulator.get_prog_ast(self.target_program)
        assert(parse_success)
        self.correct_reference = True
        self.ref_actions_taken = 1
        for inp_world, out_world in zip(self.input_worlds, self.output_worlds):
            res_emu = self.simulator.run_prog(ref_prog, inp_world)

            self.correct_reference = self.correct_reference and (res_emu.status == 'OK')
            self.correct_reference = self.correct_reference and (not res_emu.crashed)
            self.correct_reference = self.correct_reference and (out_world == res_emu.outgrid)
            self.ref_actions_taken = max(self.ref_actions_taken, len(res_emu.actions))

    def should_skip_reward(self, trace, is_final):
        return (not is_final)

    def reward_value(self, trace, is_final):
        if (not self.correct_reference):
            # There is some problem with the data because the reference program
            # crashed. Ignore it.
            return 0
        rew = 0
        parse_success, cand_prog = self.simulator.get_prog_ast(trace)
        if not parse_success:
            # Program is not syntactically correct
            rew = -self.reward_norm
        else:
            for inp_world, out_world in zip(self.input_worlds, self.output_worlds):
                res_emu = self.simulator.run_prog(cand_prog, inp_world)
                if res_emu.status != 'OK' or res_emu.crashed:
                    # Crashed or failed the simulator
                    # Set the reward to negative and stop looking
                    rew = -self.reward_norm
                    break
                elif res_emu.outgrid != out_world:
                    # Generated a wrong state
                    # Set the reward to negative and stop looking
                    rew = -self.reward_norm
                    break
                else:
                    rew = self.reward_norm
        return rew

class PerfRewardMul(MultiIO01):
    '''
    This only gives rewards at the end of the prediction.
    +val if the two programs lead to the same final state.
    - 1 if the two programs lead to different outputs

    val is a value depending on the numbers of steps taken to measure how many
    steps it took to run the program.
    This is a ratio comparing the number of steps of the reference program,
    vs. the number of steps of the sampled program.
    '''
    def reward_value(self, trace, is_final):
        if (not self.correct_reference):
            # There is some problem with the data because the reference program
            # crashed. Ignore it.
            return 0
        rew = 0
        parse_success, cand_prog = self.simulator.get_prog_ast(trace)
        if not parse_success:
            # Program is not syntactically correct
            rew = -len(self.input_worlds) * self.reward_norm
        else:
            for inp_world, out_world in zip(self.input_worlds, self.output_worlds):
                res_emu = self.simulator.run_prog(cand_prog, inp_world)
                if res_emu.status != 'OK' or res_emu.crashed:
                    # Crashed or failed the simulator
                    rew = -self.reward_norm
                    break
                elif res_emu.outgrid != out_world:
                    # Generated a wrong state
                    rew = -self.reward_norm
                    break
                else:
                    # We are correct
                    # Get a reward corresponding to the ratio between
                    # the number of actions taken by the reference program,
                    # and the number of actions taken by the proposed program.
                    rew += self.reward_norm * ((self.ref_actions_taken) / float(1.0 + len(res_emu.actions)))
        return rew

class PerfRewardDiff(MultiIO01):
    '''
    This only gives rewards at the end of the prediction.
    +val if the two programs lead to the same final state.
    - 1 if the two programs lead to different outputs

    val is a value depending on the numbers of steps taken to measure how many
    steps it took to run the program.
    This is a constant value, minus a penalty for each step taken.
    '''

    def reward_value(self, trace, is_final):
        if (not self.correct_reference):
            # There is some problem with the data because the reference program
            # crashed. Ignore it.
            return 0
        rew = 0
        parse_success, cand_prog = self.simulator.get_prog_ast(trace)
        if not parse_success:
            # Program is not syntactically correct
            rew = -len(self.input_worlds) * self.reward_norm
        else:
            for inp_world, out_world in zip(self.input_worlds, self.output_worlds):
                res_emu = self.simulator.run_prog(cand_prog, inp_world)
                if res_emu.status != 'OK' or res_emu.crashed:
                    # Crashed or failed the simulator
                    rew = -self.reward_norm
                    break
                elif res_emu.outgrid != out_world:
                    # Generated a wrong state
                    rew = -self.reward_norm
                    break
                else:
                    # We are correct.
                    # Get a positive reward,
                    # minus a penalty proportional to the number of actions
                    # necessary to accomplish the task.
                    rew += self.reward_norm * (1 - (len(res_emu.actions)/100.0))
        return rew

def expected_rew_renorm(prediction_lpbs, prediction_reward_list):
    '''
    Simplest Reward Combination Function

    Takes as input:
    `prediction_lpbs`: The log probabilities of each sampled programs
    `prediction_reward_list`: The reward associated with each of these
                              sampled programs.

    Returns the expected reward under the (renormalized so that it sums to 1)
    probability distribution defined by prediction_lbps.
    '''
    # # Method 1:
    # pbs = prediction_lpbs.exp()
    # pb_sum = pbs.sum()
    # pbs = pbs.div(pb_sum.expand_as(pbs))

    # Method 2:
    prediction_pbs = F.softmax(prediction_lpbs, dim=0)

    if prediction_pbs.is_cuda:
        prediction_reward = torch.cuda.FloatTensor(prediction_reward_list)
    else:
        prediction_reward = torch.FloatTensor(prediction_reward_list)
    prediction_reward = Variable(prediction_reward, requires_grad=False)

    return torch.dot(prediction_pbs, prediction_reward)


def n_samples_expected_genrew(nb_samples_in_bag):
    '''
    Generates a Reward Combination Function
    based on sampling with replacement `nb_samples_in_bag` programs from the
    renormalized probability distribution and keeping the one with the best
    reward.

    This DOESN'T assume that the reward are either +1 or -1
    '''
    def fun(prediction_lpbs, prediction_reward_list):
        '''
        Takes as input:
        `prediction_lpbs`: The log probabilities of each sampled programs
        `prediction_reward_list`: The reward associated with each of these
                                  sampled programs.

        Returns the expected reward when you sample with replacement
        `nb_samples_in_bag` programs from the (renormalized) probability
        distribution defined by prediction_lbps and keep the best reward
        out of those `nb_samples_in_bag`.
        '''
        prediction_pbs = F.softmax(prediction_lpbs, dim=0)

        tt = torch.cuda if prediction_pbs.is_cuda else torch
        ## Get the probability associated with each possible reward value.

        ## Get the possible reward values
        unique_rewards = np.unique(prediction_reward_list)
        per_reward_proba = Variable(tt.FloatTensor(unique_rewards.shape[0]))
        # Note: np.unique returns its results sorted

        ## Get the probability associated with each reward value
        rewards_tensor = Variable(tt.FloatTensor(prediction_reward_list),
                                  requires_grad=False)
        for idx, rew in enumerate(unique_rewards):
            this_rew_mask = (rewards_tensor == rew)
            # Sum the proba of all possible paths leading to the same reward,
            per_reward_proba[idx] = torch.dot(prediction_pbs,
                                              this_rew_mask.float())

        if len(unique_rewards) > 1:
            # Now we have all the different rewards in `unique_rewards`
            # and their associated probability in `per_reward_proba`

            # We now compute the probability of getting a reward "less or equal"
            leq_reward_proba = per_reward_proba.cumsum(0)

            # Compute the probability that we have obtained something "less or
            # equal" when sampling `nb_samples_in_bag` times.
            leqbag_reward_proba = leq_reward_proba.pow(nb_samples_in_bag)

            # Now compute the probability that we have obtained the reward exactly
            # This is simply the leq proba minus the leq proba of the previous reward
            to_mod = leqbag_reward_proba.narrow(0, 1, leqbag_reward_proba.size(0)-1)
            to_sub = leqbag_reward_proba.narrow(0, 0, leqbag_reward_proba.size(0)-1)
            bag_reward_proba = torch.cat([
                leqbag_reward_proba.narrow(0,0,1),
                to_mod - to_sub
            ])

            var_unique_rewards = Variable(torch.from_numpy(unique_rewards).float(),
                                          requires_grad=False)
            if prediction_lpbs.is_cuda:
                var_unique_rewards = var_unique_rewards.cuda()
            expected_bag_rew = torch.dot(var_unique_rewards, bag_reward_proba)
        else:
            # There is only one reward
            expected_bag_rew = per_reward_proba * unique_rewards[0]
        return expected_bag_rew

    return fun


def n_samples_expected_1m1rew(nb_samples_in_bag):
    '''
    Generates a Reward Combination Function
    based on sampling with replacement `nb_samples_in_bag` programs from the
    renormalized probability distribution and keeping the one with the best
    reward.

    This is similar to n_samples_expected_genrew, except that this version
    works only under the assumption that all rewards are either 1 or minus1.
    '''
    def fun(prediction_lpbs, prediction_reward_list):
        '''
        Takes as input:
        `prediction_lpbs`: The log probabilities of each sampled programs
        `prediction_reward_list`: The reward associated with each of these
                                  sampled programs, assumed to be 1 or -1

        Returns the expected reward when you sample with replacement
        `nb_samples_in_bag` programs from the (renormalized) probability
        distribution defined by prediction_lbps and keep the best reward
        out of those `nb_samples_in_bag`.
        '''
        prediction_pbs = F.softmax(prediction_lpbs)

        if prediction_pbs.is_cuda:
            prediction_reward = torch.cuda.FloatTensor(prediction_reward_list)
        else:
            prediction_reward = torch.FloatTensor(prediction_reward_list)
        prediction_reward = Variable(prediction_reward, requires_grad=False)

        negs_mask = (prediction_reward == -1)
        prob_negs = prediction_pbs.masked_select(negs_mask)
        prob_of_neg_rew_per_sp = prob_negs.sum()

        prob_of_neg_rew_for_bag = prob_of_neg_rew_per_sp.pow(nb_samples_in_bag)
        prob_of_pos_rew_for_bag = 1 - prob_of_neg_rew_for_bag

        expected_bag_rew = prob_of_pos_rew_for_bag - prob_of_neg_rew_per_sp
        return expected_bag_rew

    return fun



RewardCombinationFun = {
    "RenormExpected": expected_rew_renorm
}
for bag_size in [5, 50]:
    rob_key_name = str(bag_size) + "BagExpected"
    RewardCombinationFun[rob_key_name] = n_samples_expected_genrew(bag_size)
    key_name = str(bag_size) + "1m1BagExpected"
    RewardCombinationFun[key_name] = n_samples_expected_1m1rew(bag_size)

EnvironmentClasses = {
    "BlackBoxGeneralization": MultiIO01,
    "BlackBoxConsistency": MultiIO01,
    "PerfRewardMul": PerfRewardMul,
    "PerfRewardDiff": PerfRewardDiff
}
