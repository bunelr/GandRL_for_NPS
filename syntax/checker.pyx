import torch

DEF STATE_MANDATORY_NEXT = 0
DEF STATE_ACT_NEXT = 1
DEF STATE_CSTE_NEXT = 2
DEF STATE_BOOL_NEXT = 3
DEF STATE_POSTCOND_OPEN_PAREN = 4


open_paren_token = ["m(", "c(", "r(", "w(", "i(", "e("]
close_paren_token = ["m)", "c)", "r)", "w)", "i)", "e)"]
flow_leads = ["REPEAT", "WHILE", "IF", "IFELSE"]
flow_need_bool = ["WHILE", "IF", "IFELSE"]
acts = ["move", "turnLeft", "turnRight", "pickMarker", "putMarker"]
bool_check = ["markersPresent", "noMarkersPresent", "leftIsClear", "rightIsClear", "frontIsClear"]
next_is_act = ["i(", "e(", "r(", "m(", "w("]
postcond_open_paren = ["i(", "w("]
possible_mandatories = ["<s>", "DEF", "run", "c)", "ELSE", "<pad>"] + open_paren_token



cdef class CheckerState:
    cdef short state
    cdef short next_mandatory
    cdef bint[16] i_need_else_stack
    cdef short i_need_else_stack_pos
    cdef short[16] to_close_stack
    cdef short to_close_stack_pos
    cdef short c_deep
    cdef short next_actblock_open

    def __init__(self, short state, short next_mandatory,
                 short i_need_else_stack_pos, short to_close_stack_pos,
                 short c_deep, short next_actblock_open):
        self.state = state
        self.next_mandatory = next_mandatory
        self.i_need_else_stack_pos = i_need_else_stack_pos
        self.to_close_stack_pos = to_close_stack_pos
        self.c_deep = c_deep
        self.next_actblock_open = next_actblock_open

    def __copy__(self):
        new_state = CheckerState(self.state, self.next_mandatory,
                                 self.i_need_else_stack_pos, self.to_close_stack_pos,
                                 self.c_deep, self.next_actblock_open)
        for i in range(0, self.i_need_else_stack_pos+1):
            new_state.i_need_else_stack[i] = self.i_need_else_stack[i]
        for i in range(0, self.to_close_stack_pos+1):
            new_state.to_close_stack[i] = self.to_close_stack[i]
        return new_state

    cdef push_closeparen_to_stack(self, short close_paren):
        self.to_close_stack_pos += 1
        self.to_close_stack[self.to_close_stack_pos] = close_paren

    cdef short pop_close_paren(self):
        to_ret = self.to_close_stack[self.to_close_stack_pos]
        self.to_close_stack_pos -= 1
        return to_ret

    cdef short paren_to_close(self):
        return self.to_close_stack[self.to_close_stack_pos]

    cdef make_next_mandatory(self, short next_mandatory):
        self.state = STATE_MANDATORY_NEXT
        self.next_mandatory = next_mandatory

    cdef make_bool_next(self):
        self.state = STATE_BOOL_NEXT
        self.c_deep += 1

    cdef make_act_next(self):
        self.state = STATE_ACT_NEXT

    cdef close_cond_paren(self):
        self.c_deep -= 1
        if self.c_deep == 0:
            self.state = STATE_POSTCOND_OPEN_PAREN
        else:
            self.state = STATE_MANDATORY_NEXT
            # The mandatory next should already be "c)"

    cdef push_needelse_stack(self, bint need_else):
        self.i_need_else_stack_pos += 1
        self.i_need_else_stack[self.i_need_else_stack_pos] = need_else

    cdef bint pop_needelse_stack(self):
        to_ret = self.i_need_else_stack[self.i_need_else_stack_pos]
        self.i_need_else_stack_pos -= 1
        return to_ret

    cdef set_next_actblock(self, short next_actblock):
        self.next_actblock_open = next_actblock

    cdef make_next_cste(self):
        self.state = STATE_CSTE_NEXT


cdef class SyntaxVocabulary:
    cdef public short start_tkn
    cdef public short def_tkn
    cdef public short run_tkn
    cdef public short m_open_tkn
    cdef public short m_close_tkn
    cdef public short else_tkn
    cdef public short e_open_tkn
    cdef public short c_open_tkn
    cdef public short c_close_tkn
    cdef public short i_open_tkn
    cdef public short i_close_tkn
    cdef public short while_tkn
    cdef public short w_open_tkn
    cdef public short repeat_tkn
    cdef public short r_open_tkn
    cdef public short not_tkn
    cdef public short pad_tkn

    def __init__(self, short start_tkn, short def_tkn, short run_tkn,
                 short m_open_tkn, short m_close_tkn,
                 short else_tkn, short e_open_tkn,
                 short c_open_tkn, short c_close_tkn,
                 short i_open_tkn, short i_close_tkn,
                 short while_tkn, short w_open_tkn,
                 short repeat_tkn, short r_open_tkn,
                 short not_tkn, short pad_tkn):
        self.start_tkn = start_tkn
        self.def_tkn = def_tkn
        self.run_tkn = run_tkn
        self.m_open_tkn = m_open_tkn
        self.m_close_tkn = m_close_tkn
        self.else_tkn = else_tkn
        self.e_open_tkn = e_open_tkn
        self.c_open_tkn = c_open_tkn
        self.c_close_tkn = c_close_tkn
        self.i_open_tkn = i_open_tkn
        self.i_close_tkn = i_close_tkn
        self.while_tkn = while_tkn
        self.w_open_tkn = w_open_tkn
        self.repeat_tkn = repeat_tkn
        self.r_open_tkn = r_open_tkn
        self.not_tkn = not_tkn
        self.pad_tkn = pad_tkn

class PySyntaxChecker:

    def __init__(self, dict T2I, bint use_cuda):

        self.use_cuda = use_cuda
        self.open_parens = set([T2I[op] for op in open_paren_token])
        self.close_parens = set([T2I[op] for op in close_paren_token])
        self.if_statements = set([T2I[tkn] for tkn in ["IF", "IFELSE"]])
        self.op2cl = {}
        for op, cl in zip(open_paren_token, close_paren_token):
            self.op2cl[T2I[op]] = T2I[cl]
        self.need_else = {T2I["IF"]: False,
                          T2I["IFELSE"]: True}
        self.flow_lead = set([T2I[flow_lead_tkn] for flow_lead_tkn in flow_leads])
        self.effect_acts = set([T2I[act_tkn] for act_tkn in acts])
        self.act_acceptable = self.effect_acts | self.flow_lead | self.close_parens
        self.flow_needs_bool = set([T2I[flow_tkn] for flow_tkn in flow_need_bool])
        self.postcond_open_paren = set([T2I[op] for op in postcond_open_paren])
        self.range_cste = set([idx for tkn, idx in T2I.items() if tkn.startswith("R=")])
        self.bool_checks = set([T2I[bcheck] for bcheck in bool_check])

        self.vocab = SyntaxVocabulary(T2I["<s>"],  T2I["DEF"],  T2I["run"],
                                      T2I["m("], T2I["m)"], T2I["ELSE"], T2I["e("],
                                      T2I["c("], T2I["c)"], T2I["i("], T2I["i)"],
                                      T2I["WHILE"], T2I["w("], T2I["REPEAT"], T2I["r("],
                                      T2I["not"], T2I["<pad>"])
        tt = torch.cuda if use_cuda else torch
        self.vocab_size = len(T2I)
        self.mandatories_mask = {}
        for mand_tkn in possible_mandatories:
            mask = tt.ByteTensor(1,1,self.vocab_size).fill_(1)
            mask[0,0,T2I[mand_tkn]] = 0
            self.mandatories_mask[T2I[mand_tkn]] = mask
        self.act_next_masks = {}
        for close_tkn in self.close_parens:
            mask = tt.ByteTensor(1,1,self.vocab_size).fill_(1)
            mask[0,0,close_tkn] = 0
            for effect_idx in self.effect_acts:
                mask[0,0,effect_idx] = 0
            for flowlead_idx in self.flow_lead:
                mask[0,0,flowlead_idx] = 0
            self.act_next_masks[close_tkn] = mask
        self.range_mask = tt.ByteTensor(1,1,self.vocab_size).fill_(1)
        for ridx in self.range_cste:
            self.range_mask[0,0,ridx] = 0
        self.boolnext_mask = tt.ByteTensor(1,1,self.vocab_size).fill_(1)
        for bcheck_idx in self.bool_checks:
            self.boolnext_mask[0,0,bcheck_idx] = 0
        self.boolnext_mask[0,0,self.vocab.not_tkn] = 0
        self.postcond_open_paren_masks = {}
        for tkn in self.postcond_open_paren:
            mask = tt.ByteTensor(1,1,self.vocab_size).fill_(1)
            mask[0,0,tkn] = 0
            self.postcond_open_paren_masks[tkn] = mask



    def forward(self, CheckerState state, short new_idx):
        # Whatever happens, if we open a paren, it needs to be closed
        if new_idx in self.open_parens:
            state.push_closeparen_to_stack(self.op2cl[new_idx])
        if new_idx in self.close_parens:
            paren_to_end = state.pop_close_paren()
            assert(new_idx == paren_to_end)

        if state.state == STATE_MANDATORY_NEXT:
            assert(new_idx == state.next_mandatory)
            if new_idx == self.vocab.start_tkn:
                state.make_next_mandatory(self.vocab.def_tkn)
            elif new_idx == self.vocab.def_tkn:
                state.make_next_mandatory(self.vocab.run_tkn)
            elif new_idx == self.vocab.run_tkn:
                state.make_next_mandatory(self.vocab.m_open_tkn)
            elif new_idx == self.vocab.else_tkn:
                state.make_next_mandatory(self.vocab.e_open_tkn)
            elif new_idx in self.open_parens:
                if new_idx == self.vocab.c_open_tkn:
                    state.make_bool_next()
                else:
                    state.make_act_next()
            elif new_idx == self.vocab.c_close_tkn:
                state.close_cond_paren()
            elif new_idx == self.vocab.pad_tkn:
                # Should this be at the top?
                # Keep the state in mandatory next, targetting <pad>
                # Once you go <pad>, you never go back.
                pass
            else:
                raise NotImplementedError

        elif state.state == STATE_ACT_NEXT:
            assert(new_idx in self.act_acceptable)

            if new_idx in self.flow_needs_bool:
                state.make_next_mandatory(self.vocab.c_open_tkn)
                # If we open one of the IF statements, we need to keep track if
                # it's one with a else statement or not
                if new_idx in self.if_statements:
                    state.push_needelse_stack(self.need_else[new_idx])
                    state.set_next_actblock(self.vocab.i_open_tkn)
                elif new_idx == self.vocab.while_tkn:
                    state.set_next_actblock(self.vocab.w_open_tkn)
                else:
                    raise NotImplementedError
            elif new_idx == self.vocab.repeat_tkn:
                state.make_next_cste()
            elif new_idx in self.effect_acts:
                pass
            elif new_idx in self.close_parens:
                if new_idx == self.vocab.i_close_tkn:
                    need_else = state.pop_needelse_stack()
                    if need_else:
                        state.make_next_mandatory(self.vocab.else_tkn)
                    else:
                        state.make_act_next()
                elif new_idx == self.vocab.m_close_tkn:
                    state.make_next_mandatory(self.vocab.pad_tkn)
                else:
                    state.make_act_next()
            else:
                raise NotImplementedError

        elif state.state == STATE_CSTE_NEXT:
            assert(new_idx in self.range_cste)
            state.make_next_mandatory(self.vocab.r_open_tkn)

        elif state.state == STATE_BOOL_NEXT:
            if new_idx in self.bool_checks:
                state.make_next_mandatory(self.vocab.c_close_tkn)
            elif new_idx == self.vocab.not_tkn:
                state.make_next_mandatory(self.vocab.c_open_tkn)
            else:
                raise NotImplementedError

        elif state.state == STATE_POSTCOND_OPEN_PAREN:
            assert(new_idx in self.postcond_open_paren)
            assert(new_idx == state.next_actblock_open)
            state.make_act_next()

        else:
            raise NotImplementedError

    def allowed_tokens(self, CheckerState state):
        if state.state == STATE_MANDATORY_NEXT:
            # Only one possible token follows
            return self.mandatories_mask[state.next_mandatory]
        elif state.state == STATE_ACT_NEXT:
            # Either an action, a control flow statement or a closing of an open-paren
            return self.act_next_masks[state.paren_to_close()]
        elif state.state == STATE_CSTE_NEXT:
            return self.range_mask
        elif state.state == STATE_BOOL_NEXT:
            return self.boolnext_mask
        elif state.state == STATE_POSTCOND_OPEN_PAREN:
            return self.postcond_open_paren_masks[state.next_actblock_open]

    def get_sequence_mask(self, CheckerState state, list inp_sequence):
        if len(inp_sequence) == 1:
            self.forward(state, inp_sequence[0])
            return self.allowed_tokens(state)
        else:
            tt = torch.cuda if self.use_cuda else torch
            mask_infeasible_list = []
            mask_infeasible = tt.ByteTensor(1, 1, self.vocab_size)
            for stp_idx, inp in enumerate(inp_sequence):
                self.forward(state, inp)
                mask_infeasible_list.append(self.allowed_tokens(state))
            torch.cat(mask_infeasible_list, 1, out=mask_infeasible)
            return mask_infeasible

    def get_initial_checker_state(self):
        return CheckerState(STATE_MANDATORY_NEXT, self.vocab.start_tkn,
                            -1, -1, 0, -1)
