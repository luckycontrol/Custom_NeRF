import os
import numpy as np
import torch
from torch.utils.data import DataLoader

file_name = "tiny_nerf_data_npz"

l = os.listdir()
if file_name not in l:
    url = "https://people.eecs.berkeley.edu/~bmild/nerf/tiny_nerf_data.npz"
    torch.hub.download_url_to_file(url, file_name)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        self.data = np.load(file_path)

        self.images = self.data["images"]
        self.num_images, self.H, self.W = self.images.shape[0], self.images.shape[1], self.images.shape[2]
        self.poses, self.focal = self.data["poses"], self.data["focal"]

    def __len__(self):
        return len(self.data["images"])

    def __getitem__(self, index):
        pass


