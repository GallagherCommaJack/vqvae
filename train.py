from datetime import timedelta

from torch.utils.data import DataLoader
from pytorch_lightning.accelerators import accelerator
from task import *
from net import *
from loader import *
from callbacks import *

from torch import optim


def noop(*args, **kwargs):
    pass


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
        use_fft: bool = False,
        use_depthwise: bool = False,
        use_quant: bool = True,
        lpips_net: str = "vgg",
        mse_weight: float = 1.0,
        lpips_weight: float = 1.0,
        aux_weight: float = 1.0,
        epochs: int = None,
        steps_per_epoch: int = None,
        batch_size: int = 16,
    ):
        self.batch_size = batch_size
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
            use_fft=use_fft,
            use_depthwise=use_depthwise,
            use_quant=use_quant,
        )
        self.save_hyperparameters()
        super().__init__(
            net=net,
            lpips_net=lpips_net,
            mse_weight=mse_weight,
            lpips_weight=lpips_weight,
            aux_weight=aux_weight,
        )
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch

    def encode(self, x):
        return self.net.encoder(x)

    def decode(self, z):
        return self.net.decoder(z)

    def toggle_quant(self):
        self.net.use_quant = not self.net.use_quant
        return self.net.use_quant

    def enable_quant(self):
        self.net.use_quant = True

    def disable_quant(self):
        self.net.use_quant = False

    def enable_ddp(self):
        for c in self.net.encoder.codebooks.layers:
            c.all_reduce_fn = torch.distributed.all_reduce

    def disable_ddp(self):
        for c in self.net.encoder.codebooks.layers:
            c.all_reduce_fn = noop
    
    def reset_codebook(self):
        for c in self.net.encoder.codebooks.layers:
            c.initted.data.fill_(False)

    def forward(self, x):
        return self.net.forward(x)

    def configure_optimizers(self):
        opt = optim.AdamW(self.net.parameters(), lr=1e-4)
        if self.epochs and self.steps_per_epoch:
            sched = optim.lr_scheduler.OneCycleLR(
                opt, 5e-4, epochs=self.epochs, steps_per_epoch=self.steps_per_epoch,
            )
            config = {
                "scheduler": sched,
                "interval": "step",
                "frequency": 1,
            }
            return {"optimizer": opt, "lr_scheduler": config}
        return opt

    def train_dataloader(self):
        if not hasattr(self, "dset"):
            self.dset = ImageFolder(
                "/data1/DALLE-datasets/general/cc12", "ixs_filtered.txt", 256, "images"
            )
        loader = DataLoader(
            self.dset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=128,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_skip_error(self.dset),
        )
        return loader


def main():
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning import Trainer
    from pytorch_lightning.plugins import DDPPlugin

    batch_size=48
    gpus=8

    dset = ImageFolder(
        "/data1/DALLE-datasets/general/cc12", "ixs_filtered.txt", 256, "images"
    )
    loader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=128,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_skip_error(dset),
    )
    epochs = 4
    steps_per_epoch = len(loader) // gpus
    print(steps_per_epoch)

    model = WNetAE(
        dim=32,
        f=16,
        bits=64,
        codebook_dim=32,
        num_blocks=2,
        # mults=[1, 2, 2, 4, 4],
        u_net_stages=0,
        mse_weight=1.0,
        lpips_weight=0.1,
        aux_weight=1.0,
        ff_mult=2,
        use_fft=False,
        use_depthwise=False,
        use_ddp=True,
        use_quant=False,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
    )

    checkpoint_callback = ModelCheckpoint(
        "/data2/vqvae-ckpt/run_f16_noquant",
        train_time_interval=timedelta(hours=1),
        save_last=True,
    )
    logger_callback = ReconstructedImageLogger(every_n_steps=100)

    trainer = Trainer(
        # accelerator="ddp",
        gpus=gpus,
        callbacks=[checkpoint_callback, logger_callback],
        plugins=DDPPlugin(find_unused_parameters=False),
        # fast_dev_run=10,
        terminate_on_nan=True,
        benchmark=True,
        log_every_n_steps=5,
        max_epochs=epochs,
        auto_scale_batch_size="binsearch",
    )

    # print("init done, finding batch size")
    # model.disable_ddp()
    # trainer.tune(model)

    print(f"training with batch size {model.batch_size}")
    # model.enable_ddp()
    # model.reset_codebook()
    trainer.fit(model, loader)

    trainer.save_checkpoint("noquant_f16.ckpt")


if __name__ == "__main__":
    main()

