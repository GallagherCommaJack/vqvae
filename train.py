from datetime import timedelta

from pytorch_lightning.accelerators import accelerator
from task import *
from net import *
from loader import *
from callbacks import *

from torch import optim


class WNetAE(BaseAE):
    def __init__(
        self,
        dim: int,
        f: int = 8,
        u_net_stages: int = 2,
        num_blocks: int = 2,
        mults: Sequence[int] = None,
        ff_mult: int = 2,
        in_ch: int = 3,
        pe_dim: int = 8,
        bits: int = 14,
        codebook_dim: int = 16,
        use_ddp: bool = False,
        lpips_net: str = "vgg",
        mse_weight: float = 1.0,
        lpips_weight: float = 1.0,
        aux_weight: float = 1.0,
    ):
        net = WNet(
            dim,
            f=f,
            u_net_stages=u_net_stages,
            num_blocks=num_blocks,
            mults=mults,
            ff_mult=ff_mult,
            in_ch=in_ch,
            pe_dim=pe_dim,
            bits=bits,
            codebook_dim=codebook_dim,
            use_ddp=use_ddp,
        )
        super().__init__(
            net=net,
            lpips_net=lpips_net,
            mse_weight=mse_weight,
            lpips_weight=lpips_weight,
            aux_weight=aux_weight,
        )
        self.save_hyperparameters()

    def configure_optimizers(self):
        opt = optim.AdamW(self.net.parameters(), lr=1e-4)
        # sched = optim.lr_scheduler.ReduceLROnPlateau(
        #     opt, factor=0.1, patience=10_000, threshold=1e-3, min_lr=1e-8,
        # )
        # config = {
        #     "scheduler": sched,
        #     "interval": "step",
        #     "frequency": 1,
        #     "monitor": "train/loss_ema",
        # }
        return opt
        # return {"optimizer": opt, "lr_scheduler": config}


def main():
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning import Trainer
    from pytorch_lightning.plugins import DDPPlugin
    from torch.utils.data import DataLoader

    model = WNetAE(
        32,
        f=16,
        bits=16,
        num_blocks=2,
        mults=[1, 2, 2, 4, 4],
        u_net_stages=1,
        mse_weight=1.0,
        lpips_weight=0.1,
        aux_weight=0.1,
    )
    dset = ImageFolder("/data1/DALLE-datasets/general/cc12", "ixs_filtered.txt", 256, "images")
    loader = DataLoader(
        dset,
        batch_size=48,
        shuffle=False,
        num_workers=128,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_skip_error(dset),
    )

    checkpoint_callback = ModelCheckpoint(
        "/data2/vqvae-ckpt/run1",
        train_time_interval=timedelta(hours=1),
        save_last=True,
    )
    logger_callback = ReconstructedImageLogger(every_n_steps=100)

    trainer = Trainer(
        # accelerator="ddp",
        gpus=8,
        callbacks=[checkpoint_callback, logger_callback],
        plugins=DDPPlugin(find_unused_parameters=False),
        # fast_dev_run=True,
        terminate_on_nan=True,
        benchmark=True,
        log_every_n_steps=5,
    )

    trainer.fit(model, loader)


if __name__ == "__main__":
    main()

