from functools import partial
import math
import numpy as np
from typing import Optional, Sequence

from einops import rearrange
from einops.layers.torch import Rearrange
import torch
from torch import nn
import torch.nn.functional as F
from vector_quantize_pytorch import ResidualVQ


class FourierFeatures(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        assert out_features % 2 == 0
        self.map = nn.Linear(in_features, out_features // 2, bias=False)
        with torch.no_grad():
            self.map.weight.mul_(2 * math.pi)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        f = self.map(input)
        return torch.cat([f.cos(), f.sin()], dim=-1)


class FourierPosEmb(nn.Module):
    def __init__(self, out_features: int):
        super().__init__()
        self.map = nn.Conv2d(2, out_features // 2, 1, bias=False)
        with torch.no_grad():
            self.map.weight.mul_(2 * math.pi)

    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        h: Optional[int] = None,
        w: Optional[int] = None,
    ) -> torch.Tensor:
        if x is not None:
            b, _, h, w = x.shape
            dtype, device = x.dtype, x.device
        else:
            if h is None and w is not None:
                h = w
            elif w is None and h is not None:
                w = h
            else:
                raise ValueError(h, w, "one of h or w must not be None")
            device = self.map.weight.device
            dtype = self.map.weight.dtype

        h_axis = torch.linspace(-1, 1, h, device=device, dtype=dtype)
        w_axis = torch.linspace(-1, 1, w, device=device, dtype=dtype)
        grid = torch.cat(
            torch.broadcast_tensors(
                h_axis[None, None, :, None], w_axis[None, None, None, :],
            ),
            dim=1,
        )
        f = self.map(grid)
        f = torch.cat([f.cos(), f.sin()], dim=1)
        f = f.broadcast_to(b, -1, h, w)
        return f


def factor_int(n):
    val = math.ceil(math.sqrt(n))
    val2 = int(n / val)
    while val2 * val != float(n):
        val -= 1
        val2 = int(n / val)
    return val, val2


class FF(nn.Module):
    def __init__(self, dim: int, ff_mult: int):
        super().__init__()
        inner = [
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim * ff_mult, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * ff_mult, dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(dim),
        ]
        with torch.no_grad():
            inner[-1].weight.fill_(1e-1)
        self.inner = nn.Sequential(*inner)

    def forward(self, x):
        return x + self.inner(x)


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        (input,) = ctx.saved_tensors
        return (
            grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0),
            None,
            None,
        )


clamp_with_grad = ClampWithGrad.apply


def clamp_exp(
    t: torch.Tensor, low: float = math.log(1e-2), high: float = math.log(100),
):
    return clamp_with_grad(t, low, high).exp()


class ScaledCosineAttention(nn.Module):
    def __init__(self, n_head, d_head, scale=None, split_head=True):
        super().__init__()
        scale = scale if scale else math.log(d_head ** -0.5)
        self.softmax_scale = nn.Parameter(torch.full([n_head, 1, 1], scale))
        self.split_head = (
            Rearrange("b s (h d) -> b h s d", h=n_head) if split_head else nn.Identity()
        )
        self.unsplit_head = Rearrange("b h s d -> b s (h d)")

    def forward(self, q, k, v, v_kq=None):
        if v_kq is not None:
            q, k, v, v_kq = map(self.split_head, (q, k, v, v_kq))
        else:
            q, k, v = map(self.split_head, (q, k, v))

        q, k = map(partial(F.normalize, dim=-1), (q, k))
        sim = torch.einsum("bhid,bhjd->bhij", q, k) * clamp_exp(self.softmax_scale)
        qkv = torch.einsum("bhij,bhjd->bhid", sim.softmax(dim=-1), v)
        qkv = self.unsplit_head(qkv)
        if v_kq is not None:
            vkq = torch.einsum("bhij,bhid->bhjd", sim.softmax(dim=-1), v_kq)
            vkq = self.unsplit_head(vkq)
            return qkv, vkq
        else:
            return qkv


class ChannelAttention(nn.Module):
    def __init__(
        self, dim, heads=8,
    ):
        super().__init__()
        self.heads = heads
        inner_dim = dim
        assert inner_dim % heads == 0
        dim_head = inner_dim // heads
        self.norm_in = nn.BatchNorm2d(dim, affine=False)
        self.attn = ScaledCosineAttention(heads, dim_head)
        self.proj_in = nn.Sequential(
            nn.Conv2d(dim, inner_dim * 3, 3, padding=1),
            Rearrange("b c x y -> b c (x y)"),
        )
        self.proj_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim * 2),
        )
        with torch.no_grad():
            self.proj_out[-1].weight.fill_(1e-3)
            self.proj_out[-1].bias.copy_(torch.cat([torch.ones(dim), torch.zeros(dim)]))

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.norm_in(x)
        q, k, v = self.proj_in(x).chunk(3, dim=1)
        attn = rearrange(self.attn(q, k, v), "b c (h w) -> b c h w", h=h, w=w)
        scales, shifts = self.proj_out(attn).chunk(2, dim=1)
        return shifts.addcmul(x, scales)


class Block(nn.Module):
    def __init__(
        self, dim: int, num_blocks: int = 2, ff_mult: int = 4, use_attn: bool = True,
    ):
        super().__init__()

        ffs = [FF(dim, ff_mult) for _ in range(num_blocks)]
        if use_attn:
            h, d = factor_int(dim)
            attns = [ChannelAttention(dim, heads=h) for _ in range(num_blocks)]
            blocks = [nn.Sequential(ff, a) for ff, a in zip(ffs, attns)]
        else:
            blocks = ffs
        self.inner = nn.Sequential(*blocks)

        self.pre_norm = nn.BatchNorm2d(dim)
        self.post_norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        x = self.pre_norm(x)
        x = self.inner(x)
        x = self.post_norm(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        dim: int,
        f: int = 8,
        num_blocks: int = 2,
        mults: Sequence[int] = None,
        ff_mult: int = 2,
        in_ch: int = 3,
        pe_dim: int = 8,
        bits: int = 14,
        codebook_dim: int = 16,
        use_ddp: bool = False,
        use_attn: bool = True,
    ):
        super().__init__()
        stages = int(math.log2(f))
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
            nn.Conv2d(in_ch + pe_dim, dim, 3, padding=1), nn.ReLU(inplace=True),
        )
        downs = []
        for d_in, d_out in zip(dims[:stages], dims[1 : stages + 1]):
            downs.append(
                nn.Sequential(
                    Block(d_in, num_blocks, ff_mult=ff_mult, use_attn=use_attn,),
                    nn.Conv2d(d_in, d_out // 4, 3, padding=1),
                    nn.PixelUnshuffle(2),
                )
            )
        self.down = nn.Sequential(*downs)

        self.quant_conv = nn.Sequential(
            nn.Conv2d(dims[stages], codebook_dim, 1, bias=False),
            nn.BatchNorm2d(codebook_dim, affine=False),
        )

        self.codebooks = ResidualVQ(
            # dim=dims[down_stages],
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
        y = self.quant_conv(y)
        qs, ixs, ls = self.codebooks(y)
        ixs = rearrange(ixs, "c b h w -> b c h w")
        l_quant = torch.mean(ls)
        return y, qs, ixs, l_quant


class Decoder(nn.Module):
    def __init__(
        self,
        dim: int,
        f: int = 8,
        num_blocks: int = 2,
        mults: Sequence[int] = None,
        ff_mult: int = 2,
        out_ch: int = 3,
        pe_dim: int = 8,
        codebook_dim: int = 16,
        use_attn: bool = True,
    ):
        super().__init__()
        stages = int(math.log2(f))
        if mults is None:
            mults = [2 ** (i + 1) for i in range(stages)]
        elif len(mults) < stages:
            mults = mults + [mults[-1] for _ in range(stages - len(mults))]
        elif len(mults) > stages:
            mults = mults[:stages]
        mults = [1] + mults
        dims = dim * np.array(mults)

        self.pe = FourierPosEmb(pe_dim)
        self.quant_conv = nn.Conv2d(codebook_dim + pe_dim, dims[stages], 1)

        up_blocks = []
        for d_out, d_in in zip(dims[:stages], dims[1 : stages + 1]):
            up_blocks.append(
                nn.Sequential(
                    Block(d_in, num_blocks, ff_mult=ff_mult, use_attn=use_attn),
                    nn.Conv2d(d_in, d_out * 4, 3, padding=1),
                    nn.PixelShuffle(2),
                )
            )
        self.up = nn.Sequential(*reversed(up_blocks))
        self.conv_out = nn.Conv2d(dim, out_ch, 3, padding=1)

    def forward(self, qs):
        y = torch.cat([qs, self.pe(qs)], dim=1)
        y = self.quant_conv(y)
        y = self.up(y)
        y = self.conv_out(y)
        return y


class VQVAE(nn.Module):
    def __init__(
        self,
        dim: int,
        f: int = 8,
        num_blocks: int = 2,
        mults: Sequence[int] = None,
        ff_mult: int = 2,
        in_ch: int = 3,
        pe_dim: int = 8,
        bits: int = 14,
        codebook_dim: int = 16,
        use_ddp: bool = False,
        use_quant: bool = True,
        use_attn: bool = True,
    ):
        super().__init__()
        self.encoder = Encoder(
            dim,
            f=f,
            num_blocks=num_blocks,
            mults=mults,
            ff_mult=ff_mult,
            in_ch=in_ch,
            pe_dim=pe_dim,
            codebook_dim=codebook_dim,
            bits=bits,
            use_ddp=use_ddp,
            use_attn=use_attn,
        )
        self.decoder = Decoder(
            dim,
            f=f,
            num_blocks=num_blocks,
            mults=mults,
            ff_mult=ff_mult,
            out_ch=in_ch,
            pe_dim=pe_dim,
            codebook_dim=codebook_dim,
            use_attn=use_attn,
        )

        self.use_quant = use_quant

    def forward(self, x, quant_mask=None):
        y, qs, ixs, l_quant = self.encoder(x)
        if quant_mask is not None:
            z = torch.where(quant_mask[:, None, None, None], qs, y)
        elif self.use_quant:
            z = qs
        else:
            z = y
        rc = self.decoder(z)
        return rc, l_quant

