import torch.nn as nn

class NeRF(nn.Module):
    def __init__(self, num_layers, num_pos):
        super(NeRF, self).__init__()
        self.num_layers = num_layers
        self.num_pos = num_pos
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Linear(num_pos, 64))
            if i % 4 == 0 and i > 0:
                self.layers.append(nn.Linear(num_pos, num_pos))
        self.out = nn.Linear(64, 4)


    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i % 4 == 0 and i > 0:
                x = x + self.num_pos
        return self.out(x)

def get_nerf_model(num_layers, num_pos):
    return NeRF(num_layers=num_layers, num_pos=num_pos)