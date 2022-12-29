import tensorflow as tf

tf.random.set_seed(42)

import os
import glob
import imageio
import numpy as np
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 5
NUM_SAMPLES = 32
POS_ENCODE_DIMS = 16
EPOCHS = 20

def encode_position(x):
    positions = [x]
    for i in range(POS_ENCODE_DIMS)
        for fn in [tf.sin, tf.cos]:
            positions.append(fn(2.0 ** i * x))

    return tf.concat(positions, axis=-1)


def get_rays(height, width, focal, pose):
    i, j = tf.meshgrid(
        tf.range(width, dtype=tf.float32),
        tf.range(height, dtype=tf.float32),
        indexing="xy",
    )

    transformed_i = (i - width * 0.5) / focal
    transformed_j = (j - height * 0.5) / focal

    directions = tf.stack([transformed_i, -transformed_j, -tf.ones_like(i)], axis=-1)

    camera_matrix = pose[:3, :3]
    height_width_focal = pose[:3, -1]

    transformed_dirs = directions[..., None, :]
    camera_dirs = transformed_dirs * camera_matrix
    ray_directions = tf.reduce_sum(camera_dirs, axis=-1)
    ray_origins = tf.broadcast_to(height_width_focal, tf.shape(ray_directions))

    return (ray_origins, ray_directions)


def render_flat_rays(ray_origins, ray_directions, near, far, num_samples, rand=False):
    t_vals = tf.linspace(near, far, num_samples)

    if rand:
        shape = list(ray_origins.shape[:-1]) + [num_samples]
        noise = tf.random.uniform(shape=shape) * (far - near) / num_samples
        t_vals = t_vals + noise

    rays = ray_origins[..., None, :] + (
        ray_directions[..., None, :] * t_vals[..., None]
    )

    rays_flat = tf.reshape(rays, [-1, 3])
    rays_flat = encode_position(rays_flat)
    return (rays_flat, t_vals)