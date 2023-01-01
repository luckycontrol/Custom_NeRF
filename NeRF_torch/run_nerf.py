import torch

from load_data import MyDataset, file_name

dataset = MyDataset(file_name)


def encode_position(x):
    dim = 10
    positions = []

    for i in range(dim):
        for fn in [torch.sin, torch.cos]:
            positions.append(fn(2.0 ** i * x))

    return torch.concat(positions, dim=1)


def get_rays(height, width, focal, pose):
    i, j = torch.meshgrid(
        torch.range(width, dtype=torch.float32),
        torch.range(height, dtype=torch.float32),
        indexing="xy"
    )

    transformed_i = (i - width * 0.5) / focal
    transformed_j = (j - height * 0.5) / focal

    directions = torch.stack([transformed_i, -transformed_j, torch.ones_like(i)], dim=2)

    camera_matrix = pose[:3, :3]
    height_width_focal = pose[:3, -1]

    transformed_dirs = directions[..., None, :]
    camera_dirs = transformed_dirs * camera_matrix
    ray_directions = torch.sum(camera_dirs, dim=-1)
    ray_origins = torch.broadcast_to(height_width_focal, ray_directions.shape)

    return ray_origins, ray_directions


def render_flat_rays(ray_origins, ray_directions, near, far, num_samples, rand=False):
    t_vals = torch.linspace(near, far, num_samples)

    if rand:
        shape = list(ray_origins.shape[:-1]) + [num_samples]
        noise = torch.rand(size=shape) * (far - near) / num_samples
        t_vals = t_vals + noise

    rays = ray_origins[..., None, :] + (
            ray_directions[..., None, :] * t_vals[..., None]
    )
    rays_flat = torch.reshape(rays, [-1, 3])
    rays_flat = encode_position(rays_flat)
    return rays_flat, t_vals
