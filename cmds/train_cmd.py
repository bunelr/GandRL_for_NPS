#!/usr/bin/env python
import argparse

from nps.train import train_seq2seq_model, add_train_cli_args, TrainSignal
from nps.network import add_model_cli_args
from nps.utils import add_common_arg, s2intL


parser = argparse.ArgumentParser(
    description='Train a Seq2Seq model on type prediction.')

# Experiment parameters
add_model_cli_args(parser)
add_train_cli_args(parser)
add_common_arg(parser)
args = parser.parse_args()

train_seq2seq_model(args.signal, args.nb_ios, args.nb_epochs, args.optim_alg,
                    args.batch_size, args.learning_rate, args.use_grammar, args.beta, args.val_frequency, 

                    args.kernel_size, s2intL(args.conv_stack), s2intL(args.fc_stack),
                    args.tgt_embedding_size, args.lstm_hidden_size, args.nb_lstm_layers,
                    args.learn_syntax,

                    args.environment, args.reward_comb, args.nb_rollouts,
                    args.rl_beam, args.rl_inner_batch, args.rl_use_ref,

                    args.train_file, args.val_file, args.vocab, args.nb_samples, args.init_weights,

                    args.result_folder, vars(args),

                    args.use_cuda, args.log_frequency)
