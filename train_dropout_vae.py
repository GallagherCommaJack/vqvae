import itertools
from functools import partial
from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops import rearrange
from pl_bolts.datamodules import CIFAR10DataModule
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.cli import LightningCLI
from torch import nn
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import ReduceLROnPlateau

from loader import ImageFolderDataModule
from net import Decoder, Encoder
from task import Metrics


class TrainDropoutAE(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        p_dropout: float = 16.0,
        p_quant: float = 0.0,
        aux_weight: float = 1e-2,
        log_images_every: int = 100,
        shared_hparams: Dict[str, Any] = {},
        encoder_hparams: Dict[str, Any] = {},
        decoder_hparams: Dict[str, Any] = {},
        metrics: Dict[str, Any] = {},
        loss_type: str = "l2",
    ):
        super().__init__()
        if loss_type == "l1":
            self.var_mean = False
            self.loss = nn.L1Loss()
        elif loss_type == "l2":
            self.var_mean = False
            self.loss = nn.MSELoss()
        elif loss_type == "nll":
            self.var_mean = True
            self.loss = nn.GaussianNLLLoss()
        else:
            raise ValueError(loss_type)
        self.lr = lr
        self.p_dropout = p_dropout if p_dropout <= 1.0 else 1 / p_dropout
        self.p_quant = p_quant
        self.aux_weight = aux_weight
        self.loss_type = loss_type
        self.log_images_every = log_images_every

        encoder_hparams = shared_hparams | encoder_hparams
        decoder_hparams = shared_hparams | decoder_hparams
        self.encoder = Encoder(**encoder_hparams)
        self.decoder = Decoder(**decoder_hparams)
        self.metrics = Metrics(**metrics)

        self.save_hyperparameters()

    def handle_reals(self, batch, prefix: str = ""):
        reals, classes = batch
        eout = self.encoder(reals)
        ld = {}
        loss = 0.0
        if isinstance(eout, tuple):
            y, q, ix, l_aux = eout
            ld["codebook"] = l_aux
            quant_mask = torch.rand(y.size(0), device=y.device) < self.p_quant
            z = torch.where(quant_mask[:, None, None, None], q, y)
        else:
            l_aux = None
            z = eout
            zg = self.all_gather(z)
            zg = rearrange(zg, "world batch c h w -> c (world batch h w)").contiguous()
            skew = zg.pow(3).mean(dim=1).abs().mean()
            kurtosis = zg.pow(4).mean()
            ld["train/loss/skew"] = skew
            ld["train/loss/kurtosis"] = kurtosis
            l_aux = skew.abs() + kurtosis

        loss = loss + self.aux_weight * l_aux

        if self.training:
            b, c, _, _ = z.size()
            feat_mask = torch.rand((b, c), device=z.device) < self.p_dropout
            z = torch.where(feat_mask[:, :, None, None], z, torch.zeros_like(z))
        # z = F.group_norm(z, 1)
        rc = self.decoder(z)
        if self.var_mean:
            mean, logvar = rc.chunk(2, dim=1)
            rc = mean
            ae_loss = self.loss(mean, reals, logvar.exp().clip(1e-3, 1.0))
        else:
            ae_loss = self.loss(rc, reals)
        ld[f"{prefix}/loss/{self.loss_type}"] = ae_loss
        loss = loss + ae_loss
        ld[f"{prefix}/loss/total"] = loss
        return reals, rc, loss, ld

    def step(self, prefix, batch, batch_idx):
        reals, rc, loss, ld = self.handle_reals(batch, prefix)
        if self.logger is not None:
            self.log_dict(ld, prog_bar=True)
            if batch_idx % self.log_images_every == 0:
                imgs = rearrange(
                    self.all_gather(torch.stack([reals.detach(), rc.detach()])),
                    "world item batch channel height width -> (world batch item) channel height width",
                ).clip(0, 1)
                grid = make_grid(imgs, padding=2, nrow=16)
                try:
                    self.logger.log_image(
                        key=f"{prefix}/demo", images=[grid], caption=["demo grid"]
                    )
                except NotImplementedError:
                    pass
        return {"reals": reals.detach(), "fakes": rc.detach(), "loss": loss}

    def training_step(self, batch, batch_idx):
        return self.step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        d = self.step("valid", batch, batch_idx)
        reals, fakes = d["reals"], d["fakes"]
        self.metrics.update(reals, fakes)
        return d

    def on_validation_epoch_end(self) -> None:
        self.log_dict(
            {f"valid/{k}": v.detach() for k, v in self.metrics.compute().items()},
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx):
        return self.step("test", batch, batch_idx)

    def configure_optimizers(self):
        params = itertools.chain(self.encoder.parameters(), self.decoder.parameters())
        opt = torch.optim.AdamW(params, self.lr, eps=1e-5)
        return dict(
            optimizer=opt,
            lr_scheduler=dict(
                scheduler=ReduceLROnPlateau(opt, patience=5),
                interval="epoch",
                frequency=10,
                monitor="valid/loss/total",
            ),
        )

    def forward(self, *args):
        return self.step("", *args)


logger = pl.loggers.WandbLogger(project="dropout-vqvae", log_model=True)

trainer_defaults = dict(logger=logger)
cli = LightningCLI(
    TrainDropoutAE,
    CIFAR10DataModule,
    run=False,
    save_config_callback=None,
    trainer_defaults=trainer_defaults,
)

logger.watch(cli.model, log_freq=500)

# cli.trainer.tune(cli.model, cli.datamodule)
cli.trainer.fit(cli.model, cli.datamodule)
