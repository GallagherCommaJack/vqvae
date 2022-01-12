import itertools
import math
from copy import deepcopy
from functools import partial
from timeit import default_timer as timer
from typing import Any, Dict

import lpips
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from einops import rearrange
from torch import nn
from torchmetrics import FID, IS, PSNR, SSIM
from torchvision.utils import make_grid

from disc import discriminator

d_loss = lambda logits_real, logits_fake: F.softplus(
    torch.stack([-logits_real, logits_fake])
).mean()


def sqrtm(m):
    u, s, v = torch.linalg.svd(m)
    return u @ torch.diag(s.relu().sqrt()) @ v


torchmetrics.image.fid.sqrtm = sqrtm


class Metrics(nn.Module):
    def __init__(
        self, use_fid=True, use_is=True, use_psnr=True, use_ssim=False,
    ):
        super().__init__()
        self.fid = FID() if use_fid else None
        self.inc = IS() if use_is else None
        self.psnr = PSNR() if use_psnr else None
        self.ssim = SSIM() if use_ssim else None

    def update(self, reals, rc):
        x = reals.add(1).div(2).clip(0, 1)
        y = rc.add(1).div(2).clip(0, 1)
        x_int8 = x.mul(255).to(dtype=torch.uint8)
        y_int8 = y.mul(255).to(dtype=torch.uint8)

        if self.fid:
            self.fid.update(x_int8, True)
            self.fid.update(y_int8, False)
        if self.inc:
            self.inc.update(y_int8)
        if self.psnr:
            self.psnr.update(y, x)
        if self.ssim:
            self.ssim.update(y, x)

    def compute(self):
        ld = {}
        if self.fid:
            ld["FID"] = self.fid.compute()
            self.fid.real_features.clear()
            self.fid.fake_features.clear()
        if self.inc:
            uncond, cond = self.inc.compute()
            self.inc.features.clear()
            ld["IS/uncond"] = uncond
            ld["IS/cond"] = cond
        if self.psnr:
            ld["PSNR"] = self.psnr.compute()
            self.psnr.sum_squared_error.fill_(0.0)
            self.psnr.total.fill_(0)
        if self.ssim:
            ld["SSIM"] = self.ssim.compute()
            self.ssim.preds.clear()
            self.ssim.target.clear()
        return ld


class BaseAE(pl.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        lpips_net: str = "vgg",
        mse_weight: float = 1.0,
        lpips_weight: float = 1.0,
        aux_weight: float = 1.0,
        p_quant: float = 0.5,
        seed: int = 42,
        lr: float = 1e-4,
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
        self.p_quant = p_quant
        self.metrics = Metrics(
            use_fid=True, use_is=True, use_ssim=False, use_psnr=True,
        )
        self.metrics.eval().requires_grad_(False)
        self.rng = torch.quasirandom.SobolEngine(dimension=1, scramble=True, seed=seed)
        self.lr = lr

    def handle_reals(self, reals):
        quant_mask = (
            self.rng.draw(reals.size(0))[:, 0].to(device=self.device) < self.p_quant
        )
        rc, aux = self.net(reals, quant_mask=quant_mask)
        mse = F.mse_loss(rc, reals)
        lpips = self.lpips(rc, reals).mean()
        losses = {"mse": mse, "lpips": lpips, "aux": aux}
        loss = sum(v * self.weights[k] for k, v in losses.items())
        losses["loss"] = loss
        return rc, losses

    def step(self, batch, prefix: str):
        reals = batch
        rc, losses = self.handle_reals(reals)
        log_dict = {f"{prefix}/{k}": v.item() for k, v in losses.items()}
        self.log_dict(log_dict, prog_bar=True)
        return {"x": reals.detach(), "xrec": rc.detach(), **losses}

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        d = self.step(batch, "valid")
        self.metrics.update(d["x"], d["xrec"])
        return d

    def on_validation_epoch_end(self) -> None:
        self.log_dict(
            self.metrics.compute(), prog_bar=True,
        )

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return opt


class BaseAEGAN(BaseAE):
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
        p_quant: float = 0.5,
        seed: int = 42,
        lr: float = 1e-4,
    ):
        super().__init__(
            net=net,
            lpips_net=lpips_net,
            mse_weight=mse_weight,
            lpips_weight=lpips_weight,
            aux_weight=aux_weight,
            p_quant=p_quant,
            seed=seed,
            lr=lr,
        )
        self.disc = discriminator(dim=disc_dim, depth=disc_depth, grad_scale=-1.0,)
        self.weights["disc"] = disc_weight

    def handle_reals(self, reals):
        rc, losses = super().handle_reals(reals)
        if self.training:
            l_real = self.disc(reals)
            l_fake = self.disc(rc)
            disc_loss = d_loss(l_real, l_fake)
            losses["loss"] = losses["loss"] + disc_loss * self.weights["disc"]
            losses["disc"] = disc_loss
        return rc, losses


from diffusion_utils.utils import (ema_update, get_alphas_sigmas,
                                   get_ddpm_schedule)

from net import Decoder, DiffusionAE, DiffusionDecoder, Encoder


class HybridDiffusionAE(pl.LightningModule):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        diffusion_decoder: DiffusionDecoder,
        metrics: Metrics,
        codebook_weight: float = 1.0,
        vae_weight: float = 1.0,
        diffusion_weight: float = 1.0,
        p_quant: float = 0.5,
        seed: int = 42,
        ema_init_steps=12000,
        ema_decay=0.999,
        lr=1e-4,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.diffusion_decoder = diffusion_decoder
        self.diffusion_decoder_ema = (
            deepcopy(diffusion_decoder).eval().requires_grad_(False)
        )

        weight_sum = codebook_weight + vae_weight + diffusion_weight
        self.weights = {
            "codebook": codebook_weight / weight_sum,
            "vae": vae_weight / weight_sum,
            "diffusion": diffusion_weight / weight_sum,
        }

        self.p_quant = p_quant
        self.rng = torch.quasirandom.SobolEngine(dimension=1, scramble=True, seed=seed)
        self.metrics = metrics.eval().requires_grad_(False)
        self.ema_init_steps = ema_init_steps
        self.ema_decay = ema_decay
        self.lr = lr
        self.save_hyperparameters()

    def noise_reals(self, reals):
        # Draw uniformly distributed continuous timesteps
        t = self.rng.draw(reals.shape[0])[:, 0].to(self.device)

        # Calculate the noise schedule parameters for those timesteps
        alphas, sigmas = torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

        # Combine the ground truth images and the noise
        alphas = alphas[:, None, None, None]
        sigmas = sigmas[:, None, None, None]
        noise = torch.randn_like(reals)
        noised_reals = reals * alphas + noise * sigmas
        targets = noise * alphas - reals * sigmas
        return t, noised_reals, targets

    def handle_reals(self, reals):
        t, noised_reals, v_targets = self.noise_reals(reals)
        quant_mask = (self.rng.draw(reals.size(0))[:, 0] < self.p_quant).to(
            device=self.device
        )

        y, q, _, l_aux = self.encoder(reals)
        z = torch.where(quant_mask[:, None, None, None], q, y)
        rc = self.decoder(z)

        if self.training:
            v = self.diffusion_decoder(z, noised_reals, t)
        else:
            v = self.diffusion_decoder_ema(z, noised_reals, t)

        losses = {
            "codebook": l_aux,
            "vae": F.mse_loss(rc, reals),
            "diffusion": F.mse_loss(v, v_targets),
        }

        return rc, v, losses, y, q

    def step(self, batch, prefix: str = "", ddim_steps=None):
        reals = batch
        rc, v, losses, y, q = self.handle_reals(reals)
        log_dict = {f"{prefix}/loss/{k}": v.item() for k, v in losses.items()}
        self.log_dict(log_dict, prog_bar=True)
        loss = sum(self.weights[k] * v for k, v in losses.items())
        out = {"x": reals.detach(), "xrec": rc.detach(), "v": v.detach(), "loss": loss}
        if ddim_steps:
            out["xrec_ddim"] = self.diffusion_decoder_ema.ddim_sample(
                y, num_steps=ddim_steps
            )
        return out

    def forward(self, reals, *args):
        return self.step(reals)

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        d = self.step(
            batch,
            "valid",
            ddim_steps=32,
            # ddim_steps_quant=32 if self.p_quant > 0.01 else None,
        )
        self.metrics.update(d["x"], d["xrec_ddim"])
        should_log = self.logger is not None and not isinstance(
            self.logger, pl.loggers.base.DummyLogger
        )
        if should_log:
            it = d["x"].add(1).div(2).clamp(0, 1)
            rc = d["xrec"].add(1).div(2).clamp(0, 1)
            ddim = d["xrec_ddim"].add(1).div(2).clamp(0, 1)
            it, rc, ddim = rearrange(
                self.all_gather(torch.stack([it, rc, ddim])),
                "world item batch channel height width -> item (world batch) channel height width",
            )
            it, rc, ddim = map(partial(make_grid, padding=2, nrow=8), [it, rc, ddim])
            try:
                self.logger.log_image(
                    key="val/demo",
                    images=[it, rc, ddim],
                    caption=["input", "reconstruction", "ddim"],
                )
            except NotImplementedError:
                pass

        return d

    def on_before_zero_grad(self, *args, **kwargs):
        decay = (
            0.95 if self.trainer.global_step < self.ema_init_steps else self.ema_decay
        )
        ema_update(self.diffusion_decoder, self.diffusion_decoder_ema, decay)

    def configure_optimizers(self):
        params = itertools.chain(
            self.encoder.parameters(),
            self.decoder.parameters(),
            self.diffusion_decoder.parameters(),
        )
        opt = torch.optim.AdamW(params, self.lr)
        return opt


class TrainDiffusionAE(pl.LightningModule):
    def __init__(
        self,
        ae: Dict[str, Any] = {},
        metrics: Dict[str, bool] = {},
        codebook_weight: float = 1.0,
        diffusion_weight: float = 1.0,
        p_quant: float = 0.5,
        seed: int = 42,
        ema_init_steps=12000,
        ema_decay=0.999,
        lr=1e-4,
    ):
        super().__init__()

        self.ae = DiffusionAE(**ae)
        self.ae_ema = deepcopy(self.ae).eval().requires_grad_(False)

        weight_sum = codebook_weight + diffusion_weight
        self.weights = {
            "codebook": codebook_weight / weight_sum,
            "diffusion": diffusion_weight / weight_sum,
        }

        self.p_quant = p_quant
        self.rng = torch.quasirandom.SobolEngine(dimension=1, scramble=True, seed=seed)
        self.metrics = Metrics(**metrics).eval().requires_grad_(False)
        self.ema_init_steps = ema_init_steps
        self.ema_decay = ema_decay
        self.lr = lr
        self.save_hyperparameters()

    def noise_reals(self, reals):
        # Draw uniformly distributed continuous timesteps
        t = self.rng.draw(reals.shape[0])[:, 0].to(self.device)

        # Calculate the noise schedule parameters for those timesteps
        alphas, sigmas = torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

        # Combine the ground truth images and the noise
        alphas = alphas[:, None, None, None]
        sigmas = sigmas[:, None, None, None]
        noise = torch.randn_like(reals)
        noised_reals = reals * alphas + noise * sigmas
        targets = noise * alphas - reals * sigmas
        return t, noised_reals, targets

    def handle_reals(self, reals):
        t, noised_reals, v_targets = self.noise_reals(reals)
        quant_mask = (self.rng.draw(reals.size(0))[:, 0] < self.p_quant).to(
            device=self.device
        )
        ae = self.ae if self.training else self.ae_ema

        # ac_dtype = torch.bfloat16 if self.training else torch.float16
        # with torch.cuda.amp.autocast(dtype=ac_dtype):
        y, q, _, l_aux = ae.encoder(reals)
        z = torch.where(quant_mask[:, None, None, None], q, y)
        v = ae.decoder(z, noised_reals, t)

        losses = {
            "codebook": l_aux,
            "diffusion": F.mse_loss(v, v_targets),
        }

        return v, losses, y, q

    def step(self, batch, prefix: str = "", ddim_steps=None):
        reals = batch
        v, losses, y, q = self.handle_reals(reals)
        log_dict = {f"{prefix}/loss/{k}": v.item() for k, v in losses.items()}
        self.log_dict(log_dict, prog_bar=True)
        loss = sum(self.weights[k] * v for k, v in losses.items())
        out = {"x": reals.detach(), "v": v.detach(), "loss": loss}
        if ddim_steps:
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    out["xrec"] = self.ae_ema.decoder.ddim_sample(
                        y, num_steps=ddim_steps
                    )
        return out

    def forward(self, reals, *args):
        return self.step(reals)

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        d = self.step(batch, "valid", ddim_steps=32)
        self.metrics.update(d["x"], d["xrec"])
        should_log = self.logger is not None and not isinstance(
            self.logger, pl.loggers.base.DummyLogger
        )
        if should_log:
            it = d["x"].add(1).div(2).clamp(0, 1)
            ddim = d["xrec"].add(1).div(2).clamp(0, 1)
            it, ddim = rearrange(
                self.all_gather(torch.stack([it, ddim])),
                "world item batch channel height width -> item (world batch) channel height width",
            )
            it, ddim = map(partial(make_grid, padding=2, nrow=8), [it, ddim])
            try:
                self.logger.log_image(
                    key="val/demo", images=[it, ddim], caption=["input", "ddim"],
                )
            except NotImplementedError:
                pass

        return d

    def on_before_zero_grad(self, *args, **kwargs):
        decay = (
            0.95 if self.trainer.global_step < self.ema_init_steps else self.ema_decay
        )
        ema_update(self.ae, self.ae_ema, decay)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.ae.parameters(), self.lr, eps=1e-5)
        return opt

