# Leveraging Grammar and Reinforcement Learning for Neural Program Synthesis

This repository contains the code used for the paper [Leveraging Grammar and
Reinforcement Learning for Neural Program
Synthesis](https://openreview.net/forum?id=H1Xw62kRZ).

## Requirements
We recommend installing this code into a virtual environment. In order to run
the code, you first need to install pytorch, following the instructions from
[the pytorch website](http://pytorch.org/). Once this is done, you can install
this package and its dependencies by running:

```bash
pip install cython
python setup.py install
```

The experiments in the original paper were run using the dataset found at [the
Karel dataset webpage](https://msr-redmond.github.io/karel-dataset/). We
recommend you download and extract them into the `./data** directory.

## Commands
The code can be interacted with using two commands: `train_cmd.py` to perform
training of a model and `eval_cmd.py` to perform testing. This section
introduces the possible option, you can also use `--help` to see what is
available.

### Train
* `--kernel_size`, `--conv_stack`, `--fc_stack`, `--tgt_embedding_size`,
  `--lstm_hidden_size`, `--nb_lstm_layers` are flags to specify the architecture
  of the model to learn. See `nps/network.py` to see how they are used.
  `--nb_ios` specifies how many of the IO pairs should be used as inputs to the
  encoder (note that due to the architecture, even a model trained with `x` IO
  can be used to do prediction, even if a different number of IOs is available
  at test time).
* `--use_grammar` makes the model use the handwritten syntax checker, found in
  `syntax/checker.pyx`. `--learn_syntax` adds a Syntax LSTM to the model that
  attempts to learn a syntax checker, jointly with the rest of the model. The
  importance of this objective is controlled by the `--beta` parameter.
* `--signal` allows to choose the loss, between `supervised`, `rl` and
  `beam_rl`. Supervised attempts to reproduce the ground truth program, while
  `rl` and `beam_rl` try to maximize expected rewards. What rewards are used is
  specified using the `--environment` argument (it can be Consistency to
  evaluate coherence of the programs with the observed IO grids, Generalization
  to also take into account the held out pair, or Perf to additionally include
  consideration about number of steps taken.) In the case where the beam search
  approximation is used, it is also possible to specify a Reward Combination
  Function using `--reward_comb`. The default one is `RenormExpected` but the
  "bag of samples" version can be used by choosing `X1m1BagExpected` for 1/-1
  rewards or `XBagExpected` for the general case. In order to be able to fit
  experiments in a single GPU, you may need to adjust `--nb_rollouts` (how many
  samples are taken from the model to estimate a gradient when using `rl`) or
  `--rl_beam` (the size of the beam search when using `beam_rl`). There is also
  the `--rl_inner_batch` option that splits the computation of a batch into
  several minibatches that are separately evaluated before doing a gradient
  step.
* `--optim_alg` chooses the optimization algorithm used, `--batch_size` allows
  to choose the size of the mini batches. `--learning_rate` adjusts the learning
  rate. `--init_weights` can be used to specify a '.model' file from which to
  load weights.
* `--train_file` specify the json file where to look for the training samples
  and `--val_file` indicates a validation set. The validation set is used to
  keep track of the best model seen so far, so as to perform early stopping. The
  `--vocab` file is there to give a correspondence between tokens and indices in
  the learned predictions. Setting `--nb_samples` allows to train on only part
  of the dataset (0, the default, trains on the whole dataset.).
  `--result_folder` allows to indicate where the results of the experiment
  should be stored. Changing `--val_frequency` allows to evaluate accuracy on
  the validation set less frequently. 
* Specify `--use_cuda` to run everything on a GPU. You can use the
  `CUDA_VISIBLE_DEVICES` to run on a specific GPU.


```bash
# Train a simple supervised model, using the handcoded syntax checker
train_cmd.py --kernel_size 3 \
             --conv_stack "64,64,64" \
             --fc_stack "512" \
             --tgt_embedding_size 256 \
             --lstm_hidden_size 256 \
             --nb_lstm_layers 2 \
             \
             --signal supervised \
             --nb_ios 5 \
             --nb_epochs 100 \
             --optim_alg Adam \
             --batch_size 128 \
             --learning_rate 1e-4 \
             \
             --train_file data/1m_6ex_karel/train.json \
             --val_file data/1m_6ex_karel/val.json \
             --vocab data/1m_6ex_karel/new_vocab.vocab \
             --result_folder exps/supervised_use_grammar \
             \
             --use_grammar \
             \
             --use_cuda
             
# Train a supervised model, learning the grammar at the same time
train_cmd.py --kernel_size 3 \
             --conv_stack "64,64,64" \
             --fc_stack "512" \
             --tgt_embedding_size 256 \
             --lstm_hidden_size 256 \
             --nb_lstm_layers 2 \
             \
             --signal supervised \
             --nb_ios 5 \
             --nb_epochs 100 \
             --optim_alg Adam \
             --batch_size 128 \
             --learning_rate 1e-4 \
             --beta 1e-5 \
             \
             --train_file data/1m_6ex_karel/train.json \
             --val_file data/1m_6ex_karel/val.json \
             --vocab data/1m_6ex_karel/new_vocab.vocab \
             --result_folder exps/supervised_learn_grammar \
             \
             --learn_syntax \
             \
             --use_cuda
             
# Use a pretrained model, to fine-tune it using simple Reinforce
# Change the --environment flag if you want to use a reward including performance.
train_cmd.py  --signal rl \
              --environment BlackBoxGeneralization \
              --nb_rollouts 100 \
              \
              --init_weights exps/supervised_use_grammar/Weights/best.model \
              --nb_epochs 5 \
              --optim_alg Adam \
              --learning_rate 1e-5 \
              --batch_size 16 \
              \
              --train_file data/1m_6ex_karel/train.json \
              --val_file data/1m_6ex_karel/val.json \
              --vocab data/1m_6ex_karel/new_vocab.vocab \
              --result_folder exps/reinforce_finetune \
              \
              --use_grammar \
              \
              --use_cuda
              

# Use a pretrained model, fine-tune it using BS Expected reward
# Change the --environment flag if you want to use a reward including performance.
# Change the --reward_comb flag if you want to use one of the "bag of samples" loss
# Remove the --rl_use_ref flag if you don't want to make use of the known ground truth in 
# the bag.
train_cmd.py  --signal beam_rl \
              --environment BlackBoxGeneralization \
              --reward_comb RenormExpected \
              --rl_inner_batch 8 \
              --rl_use_ref \
              --rl_beam 64 \
              \
              --init_weights exps/supervised_use_grammar/Weights/best.model \
              --nb_epochs 5 \
              --optim_alg Adam \
              --learning_rate 1e-5 \
              --batch_size 16 \
              \
              --train_file data/1m_6ex_karel/train.json \
              --val_file data/1m_6ex_karel/val.json \
              --vocab data/1m_6ex_karel/new_vocab.vocab \
              --result_folder exps/beamrl_finetune \
              \
              --use_grammar \
              \
              --use_cuda
             
```

### Evaluation
The evaluation command is fairly similar. Any flags non-specified has the same
role as for the `train_cmd.py` command. The relevant file is `nps/evaluate.py`.

* `--model_weights` should point to the model to evaluate.
* `--dataset` should point to the json file containing the dataset you want to
  evaluate against.
* `--output_path` points to where the results should be written. This should be
  a prefix for all the names of the files that will be generated, followed 
* `--dump_programs` can be used to investigate by dumping the programs returned
  by the model.
* `--eval_nb_ios` is analogous to `--nb_ios` during training, how many IO pairs
  should be used as input to the model.
* `--val_nb_samples` is analogous to `--nb_samples`, can be used to do
  evaluation on only part of the dataset.
* `--eval_batch_size` specifies the batch size to use during decoding. This
  doesn't affect accuracies and batching operations only allows to go faster.
* `--beam_size` controls the size of the beam search to run when decoding the
  programs and `--top_k` should be the largest integer for which the accuracies
  should be computed.

This will generate a set of files. If `--dump_programs` is passed, the `--top_k`
most likely programs for each element of the dataset will be dumped, with their
rank and their log-probability in the `generated` subfolder. This will also
include the reference program, under the name `target`. 

The values at various ranks are reported in the generated files. `exactmatch`
corresponds to exactly reproducing the input, `semantic` corresponds to
generating a program being correct on the observed IOs, `fullgeneralize` means
generating a program correct on the observed AND held-out IOs. `syntax` simply
indicates that the program was synctatically correct. If the file,
`semantic_top3.txt` contains the number 75.00, this means that for 75.00% of the
samples, one of the top 3 programs according to the model will be semantically
correct on the observed samples.

```bash
# Evaluate a trained model on the validation set, dumping programs to allow for debugging.
eval_cmd.py --model_weights exps/supervised_use_grammar/Weights/best.model \
            \
            --vocabulary data/1m_6ex_karel/new_vocab.vocab \
            --dataset data/1m_6ex_karel/val.json \
            --eval_nb_ios 5 \
            --eval_batch_size 8 \
            --output_path exps/supervised_use_grammar/Results/ValidationSet_ \
            \
            --beam_size 64 \
            --top_k 10 \
            --dump_programs \
            --use_grammar \
            \
            --use_cuda

# Evaluate a trained model on the test set
eval_cmd.py --model_weights exps/beamrl_finetune/Weights/best.model \
            \
            --vocabulary data/1m_6ex_karel/new_vocab.vocab \
            --dataset data/1m_6ex_karel/test.json \
            --eval_nb_ios 5 \
            --eval_batch_size 8 \
            --output_path exps/beamrl_finetune/Results/TestSet_ \
            \
            --beam_size 64 \
            --top_k 10 \
            --use_grammar \
            \
            --use_cuda

```

## Citation
 If you use this code in your research, consider citing:

```
@Article{Bunel2018,
  author        = {Bunel, Rudy and Hausknecht, Matthew and Devlin, Jacob and Singh, Rishabh and Kohli, Pushmeet},
  title        =  {Leveraging Grammar and Reinforcement Learning for Neural Program Synthesis},
  journal      = {ICLR},
  year         = {2018},
}
```
