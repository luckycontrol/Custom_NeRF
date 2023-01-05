import torch


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
