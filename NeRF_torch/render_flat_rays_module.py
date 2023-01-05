import torch
from NeRF_torch import encode_position


def render_flat_rays(ray_origins, ray_directions, near, far, num_samples, rand=False):
    t_vals = torch.linspace(near, far, num_samples)

    if rand:
        shape = list(ray_origins.shape[:-1]) + [num_samples]
        noise = torch.rand(shape) * (far - near) / num_samples
        t_vals = t_vals + noise

    rays = ray_origins[..., None, :] + (
        ray_directions[..., None, :] * t_vals[..., None]
    )

    rays_flat = torch.reshape(rays, [-1, 3])
    rays_flat = encode_position(rays_flat)
    return rays_flat, t_vals