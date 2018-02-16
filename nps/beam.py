import torch
from torch.autograd import Variable
import heapq
import math
import random

class Beam(object):
    '''
    Object that we used to keep track of our candidates while we run a beam search.
    Usage can be found in `network.py`, by looking at the `beam_sample` methods.

    The object is created, one for each of the sample for which decoding
    is necessary. This will hold the best decoding found, as well as the
    most likely candidates that still need to be expanded.
    - At each time step, the `advance` method is called with the log
      probability of each possible following token, for each of the
      current candidate in the beam.
    - All finishing sequences (containing `out_end`)are considered
      and attempted to be added to the `done_seq` list containing the best
      finished candidates.
    - For the other possibles new tokens, we keep the `nb_beams` best, keeping
      track of the token and what was the index of the ray in the beam that
      lead to it.
    - The `get_next_input` can then be called to return
      what should the next input to be fed to the decoder be
    - The `get_parent_beams` can be called to return
      what previously decoded part each new token corresponds to
      (in order to pick up the appropriate decoder state to use.)
      Those two functions share their indexing scheme.
    - When no further improvement is possible (We already have the `k_best` we
      want and no unfinished candidates still holds enough probability to be
      extended into one of the `k_best` (proba is only decreasing)), the `done`
      flag is set.
    - The results of the beam search can be obtained by calling `get_sampled`
    '''
    def __init__(self, nb_beams, k_best,
                 out_start, out_end,
                 use_cuda):
        '''
        nb_beams: int - How many ray to propagate at each time step.
        k_best  : int - How many decoded strings to return.
        out_start:  Index of the first token that all decoded sequences start with
        out_end  :  Index of the token that signifies that the sequence is finished.
        use_cuda :  Whether or not operation should be done on the GPU or brought back
                    to the CPU.
        '''
        self.nb_beams = nb_beams
        self.k_best = k_best
        self.out_start = out_start
        self.out_end = out_end
        self.done = False
        self.use_cuda = use_cuda

        self.tt = torch.cuda if use_cuda else torch

        # Contains tuple of (logscore, sequence)
        self.done_seq = []

        # Score for each of the beam
        self.scores = self.tt.FloatTensor(self.nb_beams).zero_()

        # Input at time step (timestep, beam) -> input_idx
        self.ts_input_for_beam = [[out_start]]
        # Which beam is the parent to this one
        self.parentBeam = []

        # What do to for the next timestep
        self.next_beam_input = None
        self.next_beam_input_list = None
        self.parent_beam_idxs = None

    def get_next_input(self):
        return Variable(self.next_beam_input, volatile=True), self.next_beam_input_list

    def get_parent_beams(self):
        return Variable(self.parent_beam_idxs, volatile=True)

    def advance(self, wordLprobas):
        '''
        wordLprobas: (beam_size x words), log probability for each beam,
        for each word
        '''
        numWords = wordLprobas.size(1)
        numExpandWords = numWords - 1


        #########################################
        # Evaluate all finishing possibilities. #
        #########################################
        if len(self.parentBeam) > 0:
            stop_lps = wordLprobas.select(1, self.out_end)
            stopped_beam_lps = self.scores + stop_lps

            for idx, beam_lp in enumerate(stopped_beam_lps):
                if beam_lp == -float('inf'):
                    # Avoid clearly wrong solutions
                    continue
                beam_idx = idx
                seq = [self.out_end]
                parent_step_idx = -1
                prev_input = self.ts_input_for_beam[parent_step_idx][beam_idx]
                while prev_input != self.out_start:
                    seq.append(prev_input)
                    beam_idx = self.parentBeam[parent_step_idx][beam_idx]
                    parent_step_idx -= 1
                    prev_input = self.ts_input_for_beam[parent_step_idx][beam_idx]
                seq.reverse()
                seq_rep = (beam_lp, seq)
                if (len(self.done_seq) < self.k_best):
                    # The heap is not yet full, just append
                    heapq.heappush(self.done_seq, seq_rep)
                else:
                    # We already have the correct number of elements so we will
                    # stay at this size
                    heapq.heappushpop(self.done_seq, seq_rep)

        ########################################
        # Evaluate all expansion possibilities #
        ########################################
        expand_wordLprobas = torch.cat([wordLprobas.narrow(1, 0, self.out_end),
                                        wordLprobas.narrow(1, self.out_end+1,
                                                           numWords - (self.out_end+1))],
                                       1)
        if len(self.parentBeam)>0:
            prev_score = self.scores.unsqueeze(1).expand_as(expand_wordLprobas)
            ext_beam_score = expand_wordLprobas + prev_score
        else:
            # All the beams were the same, we can just use the first one
            ext_beam_score = expand_wordLprobas[0]

        flat_ext_beam_score = ext_beam_score.view(-1)
        nb_cont = flat_ext_beam_score.size(0)
        if self.nb_beams < nb_cont:
            bestScores, bestScoresId = flat_ext_beam_score.topk(self.nb_beams,
                                                                0,
                                                                True, False)
        else:
            bestScores = flat_ext_beam_score
            bestScoresId = torch.arange(0, nb_cont, 1 ).long()

        if bestScores.min() == -float('inf'):
            to_keep = (bestScores != -float('inf'))
            bestScores = torch.masked_select(bestScores, to_keep)
            bestScoresId = torch.masked_select(bestScoresId, to_keep)
        # Because we flattened the beam x expandword array, we need to reidentify
        prevBeam = bestScoresId / numExpandWords
        next_input = bestScoresId - prevBeam * numExpandWords
        self.scores = bestScores
        self.parent_beam_idxs = prevBeam

        adjust = (next_input >= self.out_end).long()
        self.next_beam_input = next_input + adjust

        parent_idxs = self.parent_beam_idxs.cpu().numpy().tolist()
        next_ts_beam_input = self.next_beam_input.cpu().numpy().tolist()
        self.next_beam_input_list = next_ts_beam_input
        self.parentBeam.append(parent_idxs)
        self.ts_input_for_beam.append(next_ts_beam_input)

        if (len(self.done_seq) == self.k_best):
            best_potential_to_cont = self.scores.max()
            if self.done_seq[0][0] > best_potential_to_cont:
                # There is no potential for improvement in the remaining
                # beams over the ones we already collected
                self.done = True
        return self.done

    def get_sampled(self):
        return heapq.nlargest(self.k_best, self.done_seq)
