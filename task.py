import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

import lpips


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
        self.weights = {
            "mse": mse_weight,
            "lpips": lpips_weight,
            "aux": aux_weight,
        }

    def handle_batch(self, batch):
        reals = batch
        rc, aux = self.net.forward(reals)
        mse = F.mse_loss(rc, reals)
        lpips = self.lpips.forward(rc, reals).mean()
        losses = {"mse": mse, "lpips": lpips, "aux": aux}
        loss = sum(v * self.weights[k] for k, v in losses.items())
        losses["loss"] = loss
        return rc, losses

    def step(self, batch, prefix: str):
        rc, losses = self.handle_batch(batch)
        log_dict = {f"{prefix}/{k}": v.item() for k, v in losses.items()}
        self.log_dict(log_dict)
        return {"x": reals, "xrec": rc, **losses}

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "valid")
