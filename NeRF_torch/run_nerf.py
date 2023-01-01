import os
import glob
import imageio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torchvision

BATCH_SIZE = 5
NUM_SAMPLES = 32
POS_ENCODE_DIMS = 16
EPOCHS = 20

file_name = "tiny_nerf_data.npz"
url = "https://people.eecs.berkeley.edu/~bmild/nerf/tiny_nerf_data.npz"
if not os.path.exists(file_name):
    data = data = torch.hub.download_url_to_file(url, file_name)

data = np.load(file_name)
images = data["images"]
im_shape = images.shape
(num_images, H, W, _) = images.shape
poses = data["poses"]
focal = data["focal"]

print(focal)