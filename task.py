import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

import lpips

from disc import discriminator

d_loss = lambda logits_real, logits_fake: F.softplus(
    torch.stack([-logits_real, logits_fake])
).mean()


class BaseAE(pl.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        lpips_net: str = "vgg",
        mse_weight: float = 1.0,
        lpips_weight: float = 1.0,
        aux_weight: float = 1.0,
    ):
        super().__init__()
        self.net = net
        self.lpips = lpips.LPIPS(net=lpips_net)
        self.lpips.eval().requires_grad_(False)
        self.weights = {
            "mse": mse_weight,
            "lpips": lpips_weight,
            "aux": aux_weight,
        }

    def handle_reals(self, reals):
        rc, aux = self.net.forward(reals)
        mse = F.mse_loss(rc, reals)
        lpips = self.lpips.forward(rc, reals).mean()
        losses = {"mse": mse, "lpips": lpips, "aux": aux}
        loss = sum(v * self.weights[k] for k, v in losses.items())
        losses["loss"] = loss
        return rc, losses

    def step(self, batch, prefix: str):
        reals = batch
        rc, losses = self.handle_reals(reals)
        log_dict = {f"{prefix}/{k}": v.item() for k, v in losses.items()}
        self.log_dict(log_dict)
        return {"x": reals, "xrec": rc, **losses}

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "valid")


class BaseAEGAN(pl.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        lpips_net: str = "vgg",
        disc_dim: int = 64,
        disc_depth: int = 3,
        mse_weight: float = 1.0,
        lpips_weight: float = 1.0,
        aux_weight: float = 1.0,
        disc_weight: float = 1.0,
    ):
        super().__init__()
        self.net = net
        self.disc = discriminator(dim=disc_dim, depth=disc_depth, grad_scale=-1.0,)
        self.lpips = lpips.LPIPS(net=lpips_net)
        self.lpips.eval().requires_grad_(False)
        self.weights = {
            "mse": mse_weight,
            "lpips": lpips_weight,
            "aux": aux_weight,
            "disc": disc_weight,
        }

    def handle_reals(self, reals):
        rc, aux = self.net.forward(reals)
        mse = F.mse_loss(rc, reals)
        lpips = self.lpips.forward(rc, reals).mean()
        l_real = self.disc(reals)
        l_fake = self.disc(rc)
        disc_loss = d_loss(l_real, l_fake)

        losses = {"mse": mse, "lpips": lpips, "aux": aux, "disc": disc_loss}
        loss = sum(v * self.weights[k] for k, v in losses.items())
        losses["loss"] = loss
        return rc, losses

    def step(self, batch, prefix: str):
        reals = batch
        rc, losses = self.handle_reals(reals)
        log_dict = {f"{prefix}/{k}": v.item() for k, v in losses.items()}
        self.log_dict(log_dict)
        return {"x": reals, "xrec": rc, **losses}

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "valid")
