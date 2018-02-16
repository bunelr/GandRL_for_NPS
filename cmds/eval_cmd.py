#!/usr/bin/env python
import argparse

from nps.evaluate import evaluate_model, add_beam_size_arg, add_eval_args
from nps.utils import add_common_arg

parser = argparse.ArgumentParser(description='Evaluate a Seq2Seq model on type prediction')

# What we need to run
parser.add_argument("--model_weights", type=str,
                    default="exps/fake_run/Weights/weights_1.model",
                    help="Weights of the model to evaluate")
parser.add_argument("--vocabulary", type=str,
                    default="data/1m_6ex_karel/new_vocab.vocab",
                    help="Vocabulary of the trained model")
parser.add_argument("--dataset", type=str,
                    default="data/1m_6ex_karel/val.json",
                    help="Dataset to evaluate against")
parser.add_argument("--output_path", type=str,
                    default="exps/fake_run/val_.txt",
                    help="Where to dump the result")
parser.add_argument("--dump_programs", action="store_true")

add_beam_size_arg(parser)
add_eval_args(parser)
add_common_arg(parser)

args = parser.parse_args()

evaluate_model(args.model_weights,
               args.vocabulary,
               args.dataset,
               args.eval_nb_ios,
               args.val_nb_samples,
               args.use_grammar,
               args.output_path,
               args.beam_size,
               args.top_k,
               args.eval_batch_size,
               args.use_cuda,
               args.dump_programs)
