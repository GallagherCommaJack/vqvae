import math
import numpy as np
from typing import Sequence
import torch
from torch import nn

from diffusion_utils.models import Block, unet
from diffusion_utils.pos_emb import FourierPosEmb

from vector_quantize_pytorch import ResidualVQ
from einops import rearrange

class Encoder(nn.Module):
    def __init__(
        self,
        dim: int,
        f: int = 8,
        u_net_stages: int = 1,
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
    ):
        super().__init__()
        down_stages = int(math.log2(f))
        stages = down_stages + u_net_stages
        if mults is None:
            mults = [2 ** (i + 1) for i in range(stages)]
        elif len(mults) < stages:
            mults = mults + [mults[-1] for _ in range(stages - len(mults))]
        elif len(mults) > stages:
            mults = mults[:stages]
        mults = [1] + mults
        dims = dim * np.array(mults)

        self.pe = FourierPosEmb(pe_dim)

        self.project_in = nn.Sequential(
            nn.Conv2d(in_ch + pe_dim, dim, 3, padding=1), nn.LeakyReLU(inplace=True),
        )
        down_blocks = []
        for d_in, d_out in zip(dims[:down_stages], dims[1 : down_stages + 1]):
            down_blocks.append(
                nn.Sequential(
                    Block(
                        d_in,
                        num_blocks,
                        rotary_emb=False,
                        ff_mult=ff_mult,
                        conv_fft=use_fft,
                        use_depthwise=use_depthwise,
                    ),
                    nn.Conv2d(d_in, d_out // 4, 3, padding=1),
                    nn.PixelUnshuffle(2),
                )
            )
        self.down = nn.Sequential(*down_blocks)

        # self.u_net = unet(
        #     dim=dims[down_stages],
        #     channels=dims[down_stages],
        #     num_blocks=num_blocks,
        #     mults=mults[down_stages + 1 :],
        #     stages=u_net_stages,
        #     input_res=4096,  # hack to make it use channel attention at all resolutions
        #     rotary_emb=False,
        #     ff_mult=ff_mult,
        #     conv_fft=use_fft,
        #     use_depthwise=use_depthwise,
        # )

        self.quant_conv = nn.Sequential(
            nn.Conv2d(dims[down_stages], codebook_dim, 1, bias=False),
            nn.BatchNorm2d(codebook_dim, affine=False),
        )

        self.codebooks = ResidualVQ(
            # dim=dims[down_stages],
            # dim=dim,
            dim=codebook_dim,
            # codebook_dim=codebook_dim,
            num_quantizers=bits,
            codebook_size=2,
            kmeans_init=True,
            decay=0.99,
            commitment_weight=1e-2,
            # orthogonal_reg_weight=10.0,
            # use_cosine_sim=True,
            accept_image_fmap=True,
            sync_codebook=use_ddp,
        )

    def forward(self, x):
        y = torch.cat([x, self.pe(x)], dim=1)
        y = self.project_in(y)
        y = self.down(y)
        # y = self.u_net([y, None])
        y = self.quant_conv(y)
        qs, ixs, ls = self.codebooks(y)
        ixs = rearrange(ixs, "c b h w -> b c h w")
        loss = torch.mean(ls)
        return y, qs, ixs, loss


class Decoder(nn.Module):
    def __init__(
        self,
        dim: int,
        f: int = 8,
        u_net_stages: int = 2,
        num_blocks: int = 2,
        mults: Sequence[int] = None,
        ff_mult: int = 2,
        out_ch: int = 3,
        pe_dim: int = 8,
        codebook_dim: int = 16,
        use_fft: bool = True,
        use_depthwise: bool = False,
    ):
        super().__init__()
        down_stages = int(math.log2(f))
        stages = down_stages + u_net_stages
        if mults is None:
            mults = [2 ** (i + 1) for i in range(stages)]
        elif len(mults) < stages:
            mults = mults + [mults[-1] for _ in range(stages - len(mults))]
        elif len(mults) > stages:
            mults = mults[:stages]
        mults = [1] + mults
        dims = dim * np.array(mults)

        self.pe = FourierPosEmb(pe_dim)
        self.quant_conv = nn.Conv2d(codebook_dim + pe_dim, dims[down_stages], 1)

        # self.u_net = unet(
        #     dim=dims[down_stages],
        #     channels=dims[down_stages],
        #     num_blocks=num_blocks,
        #     mults=mults[down_stages + 1 :],
        #     stages=u_net_stages,
        #     input_res=4096,  # hack to make it use channel attention at all resolutions
        #     rotary_emb=False,
        #     ff_mult=ff_mult,
        #     conv_fft=use_fft,
        #     heads=8,
        #     use_depthwise=use_depthwise,
        # )

        up_blocks = []
        for d_out, d_in in zip(dims[:down_stages], dims[1 : down_stages + 1]):
            up_blocks.append(
                nn.Sequential(
                    Block(
                        d_in,
                        num_blocks,
                        rotary_emb=False,
                        ff_mult=ff_mult,
                        conv_fft=use_fft,
                        use_depthwise=use_depthwise,
                    ),
                    nn.Conv2d(d_in, d_out * 4, 3, padding=1),
                    nn.PixelShuffle(2),
                )
            )
        self.up = nn.Sequential(*reversed(up_blocks))
        self.conv_out = nn.Conv2d(dim, out_ch, 3, padding=1)

    def forward(self, qs):
        y = torch.cat([qs, self.pe(qs)], dim=1)
        y = self.quant_conv(y)
        # y = self.u_net([y, None])
        y = self.up(y)
        y = self.conv_out(y)
        return y


class WNet(nn.Module):
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
    ):
        super().__init__()
        self.encoder = Encoder(
            dim,
            f=f,
            u_net_stages=u_net_stages,
            num_blocks=num_blocks,
            mults=mults,
            ff_mult=ff_mult,
            in_ch=in_ch,
            pe_dim=pe_dim,
            codebook_dim=codebook_dim,
            bits=bits,
            use_ddp=use_ddp,
            use_fft=use_fft,
            use_depthwise=use_depthwise,
        )
        self.decoder = Decoder(
            dim,
            f=f,
            u_net_stages=u_net_stages,
            num_blocks=num_blocks,
            mults=mults,
            ff_mult=ff_mult,
            out_ch=in_ch,
            pe_dim=pe_dim,
            codebook_dim=codebook_dim,
            use_fft=use_fft,
            use_depthwise=use_depthwise,
        )
        self.use_quant = use_quant

    def forward(self, x, do_log=True):
        y, qs, ixs, ls = self.encoder(x)
        if self.use_quant:
            rc = self.decoder(qs)
        else:
            rc = self.decoder(y)
        return rc, ls

