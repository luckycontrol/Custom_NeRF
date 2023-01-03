import os
import glob
import imageio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader, Dataset

from nerf import NeRF

BATCH_SIZE = 5
NUM_SAMPLES = 32
POS_ENCODE_DIMS = 16
EPOCHS = 20


def encode_position(x):
    positions = []

    for i in range(POS_ENCODE_DIMS):
        for fn in [torch.sin, torch.cos]:
            positions.append(fn(2.0 ** i * x))

    return torch.cat(tensors=positions, dim=-1)


def get_rays(height, width, focal, pose):
    i, j = torch.meshgrid(
        torch.arange(width, dtype=torch.float32),
        torch.arange(height, dtype=torch.float32),
        indexing="xy"
    )

    transformed_i = (i - width * 0.5) / focal
    transformed_j = (j - height * 0.5) / focal

    # (x, y, z)형태 만들기
    directions = torch.stack([transformed_i, -transformed_j, -torch.ones_like(i)], dim=-1)

    camera_matrix = pose[:3, :3]
    camera_pos = pose[:3, -1]

    transformed_dirs = directions[..., None, :]

    # directions * camera_matrix = world coordinate
    camera_dirs = transformed_dirs * camera_matrix

    ray_directions = torch.sum(input=camera_dirs, dim=-1)
    ray_origins = torch.broadcast_to(camera_pos, size=ray_directions.size())

    return ray_origins, ray_directions


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


def map_fn(pose):
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


def get_nerf_model(num_layers, num_pos):
    return NeRF(num_layers, num_pos)


def render_rgb_depth(model, rays_flat, t_vals, rand=True, train=True):
    if train:
        predictions = model(rays_flat)
    else:
        predictions = model.predict(rays_flat)

    predictions = torch.reshape(predictions, shape=(BATCH_SIZE, H, W, NUM_SAMPLES, 4))

    rgb = torch.sigmoid(predictions[..., :-1])
    sigma_a = torch.nn.ReLU(predictions[..., :-1])

    delta = t_vals[..., 1:] - t_vals[..., :-1]




if __name__ == '__main__':
    file_name = "tiny_nerf_data.npz"
    url = "https://people.eecs.berkeley.edu/~bmild/nerf/tiny_nerf_data.npz"
    if not os.path.exists(file_name):
        data_path = torch.hub.download_url_to_file(url, file_name)
    else:
        data_path = file_name

    data = np.load(data_path)
    images = data["images"]
    num_images, H, W, _ = images.shape
    focal, poses = data["focal"], data["poses"]

    split_index = int(num_images * 0.8)

    train_imgs = images[:split_index]
    val_imgs = images[split_index:]
    train_poses = poses[:split_index]
    val_poses = poses[split_index:]

    train_imgs = torch.from_numpy(train_imgs)
    val_imgs = torch.from_numpy(val_imgs)
    train_poses = torch.from_numpy(train_poses)
    val_poses = torch.from_numpy(val_poses)

    o, d = get_rays(H, W, focal, train_poses[0])

    r, t = render_flat_rays(o, d, 1, 6, 10, True)


