import torch


def encode_position(x, pos_encode_dims):
    positions = []

    for i in range(pos_encode_dims):
        for fn in [torch.sin, torch.cos]:
            positions.append(fn(2.0 ** i * x))

    return torch.cat(tensors=positions, dim=-1)