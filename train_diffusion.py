from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.cli import LightningCLI
import pytorch_lightning as pl

from loader import ImageFolderDataModule
from task import TrainDiffusionAE

logger = pl.loggers.WandbLogger(project="diffusion-vqvae", log_model=True)
checkpoint_callback = ModelCheckpoint(
    "/data2/vqvae-ckpt/diffusion/latest/",
    save_top_k=10,
    save_last=True,
    monitor="valid/loss/diffusion",
    filename="epoch={epoch}-FID={FID:.2E}-mse={valid/loss/diffusion:.2E}",
    auto_insert_metric_name=False,
)

trainer_defaults = dict(logger=logger)
cli = LightningCLI(
    TrainDiffusionAE,
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

logger.watch(cli.model, log_freq=500)

# cli.trainer.tune(cli.model, cli.datamodule)
cli.trainer.fit(cli.model, cli.datamodule)

