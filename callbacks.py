from typing import Any, Dict, List, Optional, Tuple

import torch
import pytorch_lightning as pl
from pytorch_lightning import Callback

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
        # x = x.add(1).div(2).clamp(0, 1)
        x_grid = make_grid(
            tensor=x,
            nrow=self.nrow,
            padding=self.padding,
            normalize=self.normalize,
            value_range=self.norm_range,
            scale_each=self.scale_each,
            pad_value=self.pad_value,
        )
        return x_grid

    def log_grids(self, prefix, x_grid, xrec_grid, trainer):
        if self.use_wandb:
            trainer.logger.experiment.log(
                {
                    f"{prefix}/input": wandb.Image(x_grid),
                    f"{prefix}/reconstruction": wandb.Image(xrec_grid),
                    "global_step": trainer.global_step,
                }
            )
        else:
            x_title = f"{prefix}/input"
            trainer.logger.experiment.add_image(
                x_title, x_grid, global_step=trainer.global_step
            )
            xrec_title = f"{prefix}/reconstruction"
            trainer.logger.experiment.add_image(
                xrec_title, xrec_grid, global_step=trainer.global_step
            )

    def handle_batch(self, prefix, outputs, trainer):
        if trainer.global_step % self.every_n_steps == 0:
            output = outputs[0] if self.multi_optim else outputs
            grids = [self.mk_grid(output[k]) for k in ["x", "xrec"]]
            self.log_grids(prefix, *grids, trainer)

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
        self.handle_batch("train", outputs, trainer)

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
        self.handle_batch("valid", outputs, trainer)
