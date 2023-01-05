import os
import numpy as np
import torch
from torch.utils.data import Dataset

from NeRF_torch import map_fn


class CustomDataset(Dataset):
    def __init__(self, is_train=True):
        file_name = "tiny_nerf_data.npz"
        url = "https://people.eecs.berkeley.edu/~bmild/nerf/tiny_nerf_data.npz"
        if not os.path.exists(file_name):
            data_path = torch.hub.download_url_to_file(url, file_name)
        else:
            data_path = file_name

        self.data = np.load(data_path)
        self.images = self.data["images"]
        self.num_images, self.H, self.W, _ = self.images.shape
        self.focal, self.poses = self.data["focal"], self.data["poses"]

        split_index = int(self.num_images * 0.8)
        if is_train:
            self.images = self.images[:split_index]
            self.num_images = int(self.num_images * 0.8)
            self.poses = self.poses[:split_index]
        else:
            self.images = self.images[split_index:]
            self.num_images = int(self.num_images * 0.2)
            self.poses = self.poses[split_index:]

    def __len__(self):
        pass

    def __getitem__(self, index):
        image = torch.from_numpy(self.images[index])
        pose = torch.from_numpy(self.poses[index])

        ray_flat, t_vals = map_fn(self.H, self.W, self.focal, pose)
        print(ray_flat)

