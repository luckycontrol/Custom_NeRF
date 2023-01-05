import torch
from NeRF_torch import *


def render_rgb_depth(model, rays_flat, t_vals, H, W, rand=True, train=True):
    if train:
        predictions = model(rays_flat)
    else:
        predictions = model.predict(rays_flat)

    predictions = torch.reshape(predictions, shape=(BATCH_SIZE, H, W, NUM_SAMPLES, 4))

    rgb = torch.sigmoid(predictions[..., :-1])
    sigma_a = torch.nn.ReLU(predictions[..., :-1])

    delta = t_vals[..., 1:] - t_vals[..., :-1]

    if rand:
        delta = torch.cat(
            [delta, torch.broadcast_to(torch.tensor(1e10), size=(BATCH_SIZE, H, W, 1))], dim=-1
        )
        alpha = 1.0 - torch.exp(-sigma_a * delta)
    else:
        delta = torch.cat(
            [delta, torch.broadcast_to(torch.tensor(1e10), size=(BATCH_SIZE, 1))], dim=-1
        )
        alpha = 1.0 - torch.exp(-sigma_a * delta[:, None, None, :])

    exp_term = 1.0 - alpha
    epsilon = 1e-10
    transmittance = torch.cumprod(exp_term + epsilon, dim=-1)
    weights = alpha * transmittance
    rgb = torch.sum(weights[..., None] * rgb, dim=-2)

    if rand:
        depth_map = torch.sum(weights * t_vals, dim=-1)
    else:
        depth_map = torch.sum(weights * t_vals[:, None, None], dim=-1)

    return rgb, depth_map
