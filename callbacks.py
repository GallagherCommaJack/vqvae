from typing import Any, Dict, List, Optional, Tuple
from PIL.Image import Image

import torch
import pytorch_lightning as pl
from pytorch_lightning import Callback

from torchmetrics import FID, IS, PSNR, SSIM

from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.utilities.distributed import rank_zero_only
import torch.nn.functional as F
from torchvision.utils import make_grid
import wandb


class ReconstructedImageLogger(Callback):
    def __init__(
        self,
        every_n_steps: int = 1000,
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = True,
        norm_range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: int = 0,
        use_wandb: bool = False,
        multi_optim=False,
    ) -> None:
        """
        Args:
            num_samples: Number of images displayed in the grid. Default: ``3``.
            nrow: Number of images displayed in each row of the grid.
                The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
            padding: Amount of padding. Default: ``2``.
            normalize: If ``True``, shift the image to the range (0, 1),
                by the min and max values specified by :attr:`range`. Default: ``True``.
            norm_range: Tuple (min, max) where min and max are numbers,
                then these numbers are used to normalize the image. By default, min and max
                are computed from the tensor.
            scale_each: If ``True``, scale each image in the batch of
                images separately rather than the (min, max) over all images. Default: ``False``.
            pad_value: Value for the padded pixels. Default: ``0``.
        """
        super().__init__()
        self.every_n_steps = every_n_steps
        self.nrow = nrow
        self.padding = padding
        self.normalize = normalize
        self.norm_range = norm_range
        self.scale_each = scale_each
        self.pad_value = pad_value
        self.multi_optim = multi_optim
        self.use_wandb = use_wandb

    def mk_grid(self, x):
        x = x.add(1).div(2).clamp(0, 1)
        x_grid = make_grid(
            tensor=x,
            nrow=self.nrow,
            padding=self.padding,
            pad_value=self.pad_value,
            # normalize=self.normalize,
            # value_range=self.norm_range,
            # scale_each=self.scale_each,
        )
        return x_grid

    def log_grids(self, prefix, trainer, x_grid, xrec_grid, quant_grid):
        diff_grid = x_grid.sub(xrec_grid).abs()
        qdiff_grid = x_grid.sub(quant_grid).abs()
        qerr_grid = xrec_grid.sub(quant_grid).abs()
        if self.use_wandb:
            trainer.logger.log_image(
                key=f"{prefix}/demo",
                images=[
                    x_grid,
                    xrec_grid,
                    quant_grid,
                    diff_grid,
                    qdiff_grid,
                    qerr_grid,
                ],
                caption=[
                    "input",
                    "reconstruction/continuous",
                    "reconstruction/quantized",
                    "diff/continuous",
                    "diff/quantized",
                    "diff/quantization",
                ],
            )
        else:
            trainer.logger.experiment.add_image(
                f"{prefix}/input", x_grid, global_step=trainer.global_step
            )
            trainer.logger.experiment.add_image(
                f"{prefix}/reconstruction/continuous",
                xrec_grid,
                global_step=trainer.global_step,
            )
            trainer.logger.experiment.add_image(
                f"{prefix}/reconstruction/quantized",
                quant_grid,
                global_step=trainer.global_step,
            )
            trainer.logger.experiment.add_image(
                f"{prefix}/diff/continuous", diff_grid, global_step=trainer.global_step,
            )
            trainer.logger.experiment.add_image(
                f"{prefix}/diff/quantized", qdiff_grid, global_step=trainer.global_step,
            )
            trainer.logger.experiment.add_image(
                f"{prefix}/diff/quantization",
                qerr_grid,
                global_step=trainer.global_step,
            )

    def handle_batch(self, module, prefix, batch, trainer):
        if trainer.global_step % self.every_n_steps == 0:
            was_training = module.training
            module.eval()
            reals = batch
            with torch.no_grad():
                y, q, *_ = module.encoder(reals)
                rc = module.decoder(y)
                rc_q = module.decoder(q)
            grids = map(self.mk_grid, [reals, rc, rc_q])
            self.log_grids(prefix, trainer, *grids)
            if was_training:
                module.train()

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.handle_batch(pl_module, "train", batch, trainer)

    @rank_zero_only
    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.handle_batch(pl_module, "valid", batch, trainer)


class PerceptualMetrics(pl.Callback):
    def __init__(
        self,
        use_fid=True,
        use_is=True,
        use_psnr=True,
        use_ssim=True,
        every_n_train_steps=None,
        every_n_valid_steps=1,
    ):
        super().__init__()
        self.fid = FID() if use_fid else None
        self.inc = IS() if use_is else None
        self.psnr = PSNR() if use_psnr else None
        self.ssim = SSIM() if use_ssim else None
        self.every_n_train_steps = every_n_train_steps
        self.every_n_valid_steps = every_n_valid_steps

    def update(self, outputs):
        x = outputs["x"].add(1).div(2).clip(0, 1)
        y = outputs["xrec"].add(1).div(2).clip(0, 1)
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

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if trainer.global_step % self.every_n_valid_steps == 0:
            self.update(outputs)

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        if self.every_n_train_steps:
            if trainer.global_step % self.every_n_train_steps == 0:
                self.update(outputs)

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        ld = {}
        if self.fid:
            ld["FID"] = self.fid.compute()
        if self.inc:
            uncond, cond = self.inc.compute()
            ld["IS/uncond"] = uncond
            ld["IS/cond"] = cond
        if self.psnr:
            ld["PSNR"] = self.psnr.compute()
        if self.ssim:
            ld["SSIM"] = self.ssim.compute()
        pl_module.log_dict(ld, prog_bar=True)

