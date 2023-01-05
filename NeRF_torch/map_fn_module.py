import torch
from NeRF_torch import *


def map_fn(H, W, focal, pose):

    ray_origins, ray_directions = get_rays(H, W, focal, pose)
    rays_flat, t_vals = render_flat_rays(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        near=2.0,
        far=6.0,
        num_samples=NUM_SAMPLES,
        rand=True
    )

    return rays_flat, t_vals
