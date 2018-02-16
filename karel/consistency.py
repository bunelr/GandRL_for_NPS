from karel.ast import Ast
from karel.ast_converter import AstParser, AstParseException
from karel.fast_emulator import FastEmulator

class Simulator(object):

    def __init__(self, idx_to_token_vocab):
        super(Simulator, self).__init__()
        self.idx_to_token = idx_to_token_vocab
        self.ast_parser = AstParser()
        self.emulator = FastEmulator(max_ticks=200)

    def tkn_prog_from_idx(self, prg_idxs):
        return [self.idx_to_token[idx] for idx in prg_idxs]

    def get_prog_ast(self, prg_idxs):
        prg_tkns = self.tkn_prog_from_idx(prg_idxs)
        try:
            prg_ast_json = self.ast_parser.parse(prg_tkns)
        except AstParseException as e:
            return False, None
        prog_ast = Ast(prg_ast_json)
        return True, prog_ast

    def run_prog(self, prog_ast, inp_grid):
        emu_result = self.emulator.emulate(prog_ast, inp_grid)
        return emu_result
