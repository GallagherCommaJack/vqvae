from datetime import timedelta
from torch.nn.modules.batchnorm import SyncBatchNorm

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
        num_blocks: int = 2,
        mults: Sequence[int] = None,
        ff_mult: int = 2,
        in_ch: int = 3,
        pe_dim: int = 8,
        num_codes: int = 2,
        num_codebooks: int = 14,
        codebook_dim: int = 16,
        use_ddp: bool = False,
        use_quant: bool = True,
        lpips_net: str = "vgg",
        mse_weight: float = 1.0,
        lpips_weight: float = 1.0,
        aux_weight: float = 1.0,
        p_quant: float = 0.5,
        epochs: int = None,
        steps_per_epoch: int = None,
        batch_size: int = 16,
        use_attn: bool = True,
        lr: float = 1e-4,
    ):
        self.batch_size = batch_size
        net = VQVAE(
            dim,
            f=f,
            num_blocks=num_blocks,
            mults=mults,
            ff_mult=ff_mult,
            in_ch=in_ch,
            pe_dim=pe_dim,
            num_codes=num_codes,
            num_codebooks=num_codebooks,
            codebook_dim=codebook_dim,
            use_ddp=use_ddp,
            use_quant=use_quant,
            use_attn=use_attn,
        )
        self.save_hyperparameters()
        super().__init__(
            net=net,
            lpips_net=lpips_net,
            mse_weight=mse_weight,
            lpips_weight=lpips_weight,
            aux_weight=aux_weight,
            p_quant=p_quant,
        )
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.lr = lr

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
        opt = optim.AdamW(self.parameters(), lr=self.lr)
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


class WNetAEGAN(BaseAEGAN):
    def __init__(
        self,
        dim: int,
        f: int = 8,
        num_blocks: int = 2,
        mults: Sequence[int] = None,
        ff_mult: int = 2,
        in_ch: int = 3,
        pe_dim: int = 8,
        num_codes: int = 2,
        num_codebooks: int = 14,
        codebook_dim: int = 16,
        use_ddp: bool = False,
        use_quant: bool = True,
        lpips_net: str = "vgg",
        mse_weight: float = 1.0,
        lpips_weight: float = 1.0,
        aux_weight: float = 1.0,
        disc_weight: float = 1.0,
        p_quant: float = 0.5,
        epochs: int = None,
        steps_per_epoch: int = None,
        batch_size: int = 16,
        use_attn: bool = True,
        lr: float = 1e-4,
    ):
        self.batch_size = batch_size
        net = VQVAE(
            dim,
            f=f,
            num_blocks=num_blocks,
            mults=mults,
            ff_mult=ff_mult,
            in_ch=in_ch,
            pe_dim=pe_dim,
            num_codes=num_codes,
            num_codebooks=num_codebooks,
            codebook_dim=codebook_dim,
            use_ddp=use_ddp,
            use_quant=use_quant,
            use_attn=use_attn,
        )
        self.save_hyperparameters()
        super().__init__(
            net=net,
            lpips_net=lpips_net,
            mse_weight=mse_weight,
            lpips_weight=lpips_weight,
            aux_weight=aux_weight,
            disc_weight=disc_weight,
            p_quant=p_quant,
        )
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.lr = lr

    # variable bitrate dropout
    # ignoring for now
    # def handle_reals(self, reals):
    #     z, q, i, aux = self.encode(reals)
    #     aux = aux + F.mse_loss(z, q.detach())
    #     rc = self.decode(z)

    #     b, bits, h, w = i.shape
    #     dropout = torch.randint(bits // 4, bits, [b], device=i.device)
    #     mask = torch.arange(0, bits, 1, device=i.device)
    #     mask = mask[None, :] <= dropout[:, None]
    #     # mask.shape == [b,bits]

    #     embs = torch.stack(
    #         [layer._codebook.embed for layer in self.net.encoder.codebooks.layers]
    #     )
    #     # embs.shape = [bits,2,d]

    #     oh = F.one_hot(i).to(dtype=embs.dtype)
    #     # oh.shape = [b,bits,h,w,2]
    #     q_masked = torch.einsum("bc,bchwn,cnd->bdhw", mask, oh, embs)

    #     # replace grad
    #     q_masked = z + (q_masked - z).detach()
    #     rc_q = self.decode(q_masked)

    #     mse = F.mse_loss(rc, reals) + F.mse_loss(rc_q, reals)
    #     lpips = (
    #         self.lpips.forward(rc, reals).mean()
    #         + self.lpips.forward(rc_q, reals).mean()
    #     )
    #     l_real = self.disc.forward(reals)
    #     l_fake = self.disc.forward(rc)
    #     l_fake_q = self.disc.forward(rc_q)
    #     disc_loss = d_loss(l_real, l_fake) + d_loss(l_real, l_fake_q)

    #     losses = {"mse": mse, "lpips": lpips, "aux": aux, "disc": disc_loss}
    #     loss = sum(v * self.weights[k] for k, v in losses.items())
    #     losses["loss"] = loss
    #     return rc, losses

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
        opt = optim.AdamW(self.parameters(), lr=self.lr)
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


def main():
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning import Trainer
    from pytorch_lightning.plugins import DDPPlugin

    batch_size = 64
    epochs = None
    mode = "train"
    if mode == "test1":
        gpus = 1
        fast_dev_run = 10
    elif mode == "test2":
        gpus = 2
        fast_dev_run = 10
    elif mode == "train":
        gpus = 8
        fast_dev_run = False

    dset = ImageFolder(
        "/data1/DALLE-datasets/general/cc12", "ixs_filtered.txt", 256, "images"
    )
    # do 100 batches in validation mode
    vsize = 64 * 8 * 100
    tsize = len(dset) - vsize
    valid_ds, train_ds = D.random_split(
        dset, [vsize, tsize], torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=128,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=128,
        pin_memory=True,
        drop_last=True,
    )

    hparams = dict(
        lr=1e-4,
        dim=32,
        use_attn=False,
        f=8,
        num_codes=2,
        num_codebooks=64,
        codebook_dim=12,
        num_blocks=4,
        mse_weight=1.0,
        lpips_weight=0.0,
        aux_weight=1e-2,
        # disc_weight=1e-1,
        p_quant=0.0,
        ff_mult=4,
        use_ddp=gpus > 1,
    )
    idx = 0
    f, dim, num_codes, num_codebooks, use_attn = [
        hparams[k]
        for k in ["f", "codebook_dim", "num_codes", "num_codebooks", "use_attn"]
    ]
    path_base = Path("/data2/vqvae-ckpt/")
    attn = "attn" if use_attn else "noattn"
    ext = f"vae_f{f}_d{dim}_v{num_codes}_n{num_codebooks}_{attn}"
    ckpt_path = path_base / f"{ext}_{idx}"
    if idx > 0:
        resume_path = path_base / f"{ext}_{idx-1}" / "last.ckpt"
    else:
        resume_path = None

    if resume_path:
        model = WNetAE.load_from_checkpoint(resume_path, strict=False, **hparams,)
    else:
        model = WNetAE(**hparams)

    if gpus > 1:
        model = SyncBatchNorm.convert_sync_batchnorm(model)

    checkpoint_callback = ModelCheckpoint(
        ckpt_path,
        save_top_k=10,
        save_last=True,
        monitor="FID",
        filename="{epoch}-{FID:.2f}",
    )
    logger_callback = ReconstructedImageLogger(every_n_steps=100)

    trainer = Trainer(
        accelerator="ddp",
        gpus=gpus,
        callbacks=[checkpoint_callback, logger_callback],
        plugins=DDPPlugin(find_unused_parameters=False),
        fast_dev_run=fast_dev_run,
        # logger=False,
        # terminate_on_nan=True,
        # benchmark=True,
        log_every_n_steps=5,
        max_epochs=epochs,
        # auto_lr_find=True,
        # auto_scale_batch_size="binsearch",
        resume_from_checkpoint=resume_path,
        val_check_interval=0.1,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="value",
        stochastic_weight_avg=True,
    )

    # print("init done, finding batch size")
    # model.disable_ddp()
    # trainer.tune(model, loader)

    # print(f"training with batch size {model.batch_size}")
    # model.enable_ddp()
    # model.reset_codebook()
    trainer.fit(model, train_loader, valid_loader)


if __name__ == "__main__":
    main()

