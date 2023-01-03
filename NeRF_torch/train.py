import torch
import torch.nn as nn

class Train(nn.Module):
    def __init__(self, nerf_model):
        super().__init__()
        self.nerf_model = nerf_model
        self.loss_fn = None
        self.optimizer = None
        self.loss_tracker = nn.Metric(name="loss")
        self.psnr_metric = nn.Metric(name="psnr")

    def forward(self, images, rays):
        rays_flat, t_vals = rays
        rgb, _ = render_rgb_depth(
            model=self.nerf_model, rays_flat=rays_flat, t_vals=t_vals, rand=True
        )
        return rgb

    def train_step(self, images, rays):
        self.optimizer.zero_grad()

        preds = self.forward(images, rays)
        loss = self.loss_fn(images, preds)

        loss.backward()
        self.optimizer.step()

        psnr = torch.image.psnr(images, preds, max_val=1.0)

        # Compute our own metrics
        self.loss_tracker.update(loss.item())
        self.psnr_metric.update(psnr.item())
        return {"loss": self.loss_tracker.result(), "psnr": self.psnr_metric.result()}
