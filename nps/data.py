# External imports
import json
import random
import torch
import os
from tqdm import tqdm
from torch.autograd import Variable
from karel.world import World
from itertools import chain

IMG_FEAT = 5184
IMG_DIM = 18
IMG_SIZE = torch.Size((16, IMG_DIM, IMG_DIM))


def translate(seq,
              vocab):
    return [vocab[str(elt)] for elt in seq]


def load_input_file(path_to_dataset, path_to_vocab):
    '''
    path_to_dataset: File containing the data
    path_to_vocab: File containing the vocabulary
    '''
    tgt_tkn2idx = {
        '<pad>': 0,
    }
    next_id = 1
    with open(path_to_vocab, 'r') as vocab_file:
        for line in vocab_file.readlines():
            tgt_tkn2idx[line.strip()] = next_id
            next_id += 1
    tgt_idx2tkn = {}
    for tkn, idx in tgt_tkn2idx.items():
        tgt_idx2tkn[idx] = tkn

    vocab = {"idx2tkn": tgt_idx2tkn,
             "tkn2idx": tgt_tkn2idx}

    path_to_ds_cache = path_to_dataset.replace('.json', '.thdump')
    if os.path.exists(path_to_ds_cache):
        dataset = torch.load(path_to_ds_cache)
    else:
        with open(path_to_dataset, 'r') as dataset_file:
            srcs = []
            tgts = []
            current_ios = []
            for sample_str in tqdm(dataset_file.readlines()):
                sample_data = json.loads(sample_str)

                # Get the target program
                tgt_program_tkn = sample_data['program_tokens']

                tgt_program_idces = translate(tgt_program_tkn, tgt_tkn2idx)
                current_ios = []

                for example in sample_data['examples']:
                    inp_grid_coord = []
                    inp_grid_val = []
                    inp_grid_str = example['inpgrid_tensor']
                    for coord_str in inp_grid_str.split():
                        idx, val = coord_str.split(':')
                        inp_grid_coord.append(int(idx))
                        assert(float(val)==1.0)
                    inp_grid = torch.ShortTensor(inp_grid_coord)

                    out_grid_coord = []
                    out_grid_val = []
                    out_grid_str = example['outgrid_tensor']
                    for coord_str in out_grid_str.split():
                        idx, val = coord_str.split(':')
                        out_grid_coord.append(int(idx))
                        assert(float(val)==1.0)
                    out_grid = torch.ShortTensor(out_grid_coord)

                    current_ios.append((inp_grid, out_grid))

                srcs.append(current_ios)
                tgts.append(tgt_program_idces)

        dataset = {"sources": srcs,
                   "targets": tgts}
        torch.save(dataset, path_to_ds_cache)

    return dataset, vocab


def shuffle_dataset(dataset, batch_size, randomize=True):
    '''
    We are going to group together samples that have a similar length, to speed up training
    batch_size is passed so that we can align the groups
    '''
    pairs = list(zip(dataset["sources"], dataset["targets"]))
    bucket_fun = lambda x: len(x[1]) / 5
    pairs.sort(key=bucket_fun, reverse=True)
    grouped_pairs = [pairs[pos: pos + batch_size]
                     for pos in xrange(0,len(pairs), batch_size)]
    if randomize:
        to_shuffle = grouped_pairs[:-1]
        random.shuffle(to_shuffle)
        grouped_pairs[:-1] = to_shuffle
    pairs = chain.from_iterable(grouped_pairs)
    in_seqs, out_seqs = zip(*pairs)
    return {
        "sources": in_seqs,
        "targets": out_seqs
    }

def grid_desc_to_tensor(grid_desc):
    grid = torch.Tensor(IMG_FEAT).fill_(0)
    grid.index_fill_(0, grid_desc.long(), 1)
    grid = grid.view(IMG_SIZE)
    return grid


def get_minibatch(dataset, sp_idx, batch_size,
                  start_idx, end_idx, pad_idx,
                  nb_ios, shuffle=True, volatile_vars=False):
    """Prepare minibatch."""

    # Prepare the grids
    grid_descriptions = dataset["sources"][sp_idx:sp_idx+batch_size]
    inp_grids = []
    out_grids = []
    inp_worlds= []
    out_worlds= []
    inp_test_worlds = []
    out_test_worlds = []
    for sample in grid_descriptions:
        if shuffle:
            random.shuffle(sample)
        sample_inp_grids = []
        sample_out_grids = []
        sample_inp_worlds = []
        sample_out_worlds = []
        sample_test_inp_worlds = []
        sample_test_out_worlds = []
        for inp_grid_desc, out_grid_desc in sample[:nb_ios]:

            # Do the inp_grid
            inp_grid = grid_desc_to_tensor(inp_grid_desc)
            # Do the out_grid
            out_grid = grid_desc_to_tensor(out_grid_desc)

            sample_inp_grids.append(inp_grid)
            sample_out_grids.append(out_grid)
            sample_inp_worlds.append(World.fromPytorchTensor(inp_grid))
            sample_out_worlds.append(World.fromPytorchTensor(out_grid))
        for inp_grid_desc, out_grid_desc in sample[nb_ios:]:
            # Do the inp_grid
            inp_grid = grid_desc_to_tensor(inp_grid_desc)
            # Do the out_grid
            out_grid = grid_desc_to_tensor(out_grid_desc)
            sample_test_inp_worlds.append(World.fromPytorchTensor(inp_grid))
            sample_test_out_worlds.append(World.fromPytorchTensor(out_grid))

        sample_inp_grids = torch.stack(sample_inp_grids, 0)
        sample_out_grids = torch.stack(sample_out_grids, 0)
        inp_grids.append(sample_inp_grids)
        out_grids.append(sample_out_grids)
        inp_worlds.append(sample_inp_worlds)
        out_worlds.append(sample_out_worlds)
        inp_test_worlds.append(sample_test_inp_worlds)
        out_test_worlds.append(sample_test_out_worlds)
    inp_grids = Variable(torch.stack(inp_grids, 0), volatile=volatile_vars)
    out_grids = Variable(torch.stack(out_grids, 0), volatile=volatile_vars)

    # Prepare the target sequences
    targets = dataset["targets"][sp_idx:sp_idx+batch_size]

    lines = [
        [start_idx] + line for line in targets
    ]
    lens = [len(line) for line in lines]
    max_len = max(lens)

    # Drop the last element, it should be the <end> symbol for all of them
    # padding for all of them
    input_lines = [
        line[:max_len-1] + [pad_idx] * (max_len - len(line[:max_len-1])-1) for line in lines
    ]
    # Drop the first element, should always be the <start> symbol. This makes
    # everything shifted by one compared to the input_lines
    output_lines = [
        line[1:] + [pad_idx] * (max_len - len(line)) for line in lines
    ]

    in_tgt_seq = Variable(torch.LongTensor(input_lines), volatile=volatile_vars)
    out_tgt_seq = Variable(torch.LongTensor(output_lines), volatile=volatile_vars)

    return inp_grids, out_grids, in_tgt_seq, input_lines, out_tgt_seq, \
        inp_worlds, out_worlds, targets, inp_test_worlds, out_test_worlds
