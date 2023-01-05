import os
import copy
import imageio
from tqdm import tqdm
import matplotlib.pyplot as plt

from NeRF_torch import CustomDataset


def train(model, inputs, optimizer, loss, epoch):
    best_model = copy.deepcopy(model)
    best_psnr = 0.

    for i in range(epoch):
        print(f'Epoch: {i + 1}/{epoch}')
        print('-' * 10)

        for phase in ['train', 'eval']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_psnr = 0.
            running_loss = 0.


if __name__ == '__main__':
    dataset = CustomDataset(is_train=True)
    dataset[0]

