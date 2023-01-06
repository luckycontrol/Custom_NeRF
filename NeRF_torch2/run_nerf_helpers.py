import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
norm28bit = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fn = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fn.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_band = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_band = torch.linspace(2., 2.**max_freq, steps=N_freqs)

        for freq in freq_band:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fn.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fn = embed_fn
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fn], dim=-1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos]
    }

    embedder = Embedder(**embed_kwargs)
    embed = lambda x, embedder: embedder.embed(x)
    return embed, embedder.out_dim


class NeRF(nn.Module):
    # input_ch: 입력 데이터 중 위치값(x, y, z)
    # input_ch_views: 입력 데이터 중 방향값(세타, 피)

    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D - 1)]
        )

        self.views_linears = nn.ModuleList(
            [nn.Linear(input_ch_views + W, W // 2)]
        )

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)

        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](input_pts)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = self.output_linear([rgb, alpha], -1)

        else:
            outputs = self.output_linear(h)

        return outputs


def get_rays(H, W, K, c2w):
    # H: Height
    # W: Width
    # K: Camara intrinsic matrix(3x3)
    #       [[focal_length_x, skew           , principal_point_x],
    #        [0             , focal_length_y , principal_point_y],
    #        [0             , 0              , 1                ]]
    # c2w: Camera extrinsic matrix(4x4)
    #       [[R11, R12, R13, t1],
    #        [R21, R22, R23, t2],
    #        [R31, R32, R33, t3],
    #        [0  , 0  , 0  , 1 ]

    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W, dtype=torch.float32),
        torch.linspace(0, H - 1, H, dtype=torch.float32),
        indexing="xy"
    )

    i = i.t()
    j = j.t()

    # Coordinate Normalize: (pixel - principal_point) / focal_length
    dirs = torch.stack([(i - K[0][2])/K[0][0], -(j - K[1][2])/K[1][1], -torch.ones_like(i)], dim=-1)
    # Rotate ray direction from camera coordinate to world coordinate.
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
    # Translate camera coordinate's camera position to world coordinate's camera position
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]



if __name__ == '__main__':
    data = np.load('./tiny_nerf_data.npz')
    images, poses, focal = data["images"], data["poses"], data["focal"]

    num_images, H, W, _ = images.shape

    pose = torch.from_numpy(poses[0])
    K = np.array([
        [focal, 0, W / 2],
        [0, focal, H / 2],
        [0, 0, 1]
    ])

    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W, dtype=torch.float32),
        torch.linspace(0, H - 1, H, dtype=torch.float32),
        indexing="xy"
    )

    i.t()
    j.t()

    camera_dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], dim=-1)
    rays_d = torch.sum(camera_dirs[..., np.newaxis, :] * pose[:3, :3], dim=-1)
    rays_o = pose[:3, -1].expand(rays_d.shape)
