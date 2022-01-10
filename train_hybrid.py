from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.cli import LightningCLI
from torch.nn.modules.batchnorm import SyncBatchNorm
import pytorch_lightning as pl

from callbacks import ReconstructedImageLogger
from loader import ImageFolderDataModule
from task import HybridDiffusionAE

logger = pl.loggers.WandbLogger(project="diffusion-vqvae", log_model=True)
checkpoint_callback = ModelCheckpoint(
    "/data2/vqvae-ckpt/hybrid/latest/",
    save_top_k=10,
    save_last=True,
    monitor="valid/loss/diffusion",
    filename="{epoch}-{FID:.2f}",
)
logger_callback = ReconstructedImageLogger(every_n_steps=100, use_wandb=True)

trainer_defaults = dict(logger=logger)
cli = LightningCLI(
    HybridDiffusionAE,
    ImageFolderDataModule,
    run=False,
    save_config_callback=None,
    trainer_defaults=trainer_defaults,
)
cli.logger = logger
cli.trainer.callbacks.extend(
    [
        checkpoint_callback,
        # logger_callback,
    ]
)

if cli.trainer.gpus and cli.trainer.gpus > 1:
    SyncBatchNorm.convert_sync_batchnorm(cli.model)

# cli.trainer.tune(cli.model, cli.datamodule)
cli.trainer.fit(cli.model, cli.datamodule)

