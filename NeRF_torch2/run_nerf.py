import os, sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from run_nerf_helpers import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(0)


def batchify(fn, chunk):
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i: i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], dim=-1)

    output_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(output_flat, list(inputs.shape[:-1]) + [output_flat.shape[-1]])
    return outputs


if __name__ == '__main__':
    pass