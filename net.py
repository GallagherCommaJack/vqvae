import math
from functools import partial
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from diffusion_utils.norm import LayerNorm
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
from torch.nn.modules import padding
from tqdm import tqdm
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
    def __init__(self, dim: int, ff_mult: int, use_batchnorm: bool = True):
        super().__init__()
        inner = [
            nn.SyncBatchNorm(dim) if use_batchnorm else LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim * ff_mult, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * ff_mult, dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.SyncBatchNorm(dim) if use_batchnorm else LayerNorm(dim),
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
        self, dim, heads=8, use_batchnorm=True,
    ):
        super().__init__()
        self.heads = heads
        inner_dim = dim
        assert inner_dim % heads == 0
        dim_head = inner_dim // heads
        self.norm_in = (
            nn.SyncBatchNorm(dim, affine=False) if use_batchnorm else LayerNorm(dim)
        )
        self.attn = ScaledCosineAttention(heads, dim_head)
        self.proj_in = nn.Sequential(
            nn.Conv2d(dim, inner_dim * 3, 3, padding=1),
            Rearrange("b c x y -> b c (x y)"),
        )
        self.proj_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim * 2, 3, padding=1, bias=False),
            nn.SyncBatchNorm(dim * 2) if use_batchnorm else nn.GroupNorm(2, dim * 2),
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
        self,
        dim: int,
        num_blocks: int = 2,
        ff_mult: int = 4,
        use_attn: bool = True,
        use_batchnorm: bool = True,
    ):
        super().__init__()

        ffs = [FF(dim, ff_mult, use_batchnorm=use_batchnorm) for _ in range(num_blocks)]
        if use_attn:
            h, d = factor_int(dim)
            attns = [
                ChannelAttention(dim, heads=h, use_batchnorm=use_batchnorm)
                for _ in range(num_blocks)
            ]
            blocks = [nn.Sequential(ff, a) for ff, a in zip(ffs, attns)]
        else:
            blocks = ffs
        self.inner = nn.Sequential(*blocks)

        self.pre_norm = nn.SyncBatchNorm(dim) if use_batchnorm else LayerNorm(dim)
        self.post_norm = nn.SyncBatchNorm(dim) if use_batchnorm else LayerNorm(dim)

    def forward(self, x):
        x = self.pre_norm(x)
        x = self.inner(x)
        x = self.post_norm(x)
        return x


class Downsampler(nn.Sequential):
    def __init__(self, ch_in: int, ch_out: int, factor: int = 2):
        super().__init__(
            nn.Conv2d(ch_in, ch_out // factor ** 2, 3, padding=1),
            nn.PixelUnshuffle(factor),
        )


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
        num_codes: int = 2,
        num_codebooks: int = 14,
        codebook_dim: int = 16,
        use_ddp: bool = False,
        use_attn: bool = True,
        use_batchnorm: bool = True,
        use_codes: bool = True,
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
                    Downsampler(d_in, d_out, 2),
                    Block(
                        d_out,
                        num_blocks,
                        ff_mult=ff_mult,
                        use_attn=use_attn,
                        use_batchnorm=use_batchnorm,
                    ),
                )
            )
        self.down = nn.Sequential(*downs)

        self.quant_conv = nn.Sequential(
            nn.Conv2d(dims[stages], codebook_dim, 1, bias=False),
            nn.SyncBatchNorm(codebook_dim, affine=False)
            if use_batchnorm
            else nn.GroupNorm(1, codebook_dim, affine=False),
        )

        if use_codes:
            self.codebooks = ResidualVQ(
                dim=codebook_dim,
                num_quantizers=num_codebooks,
                codebook_size=num_codes,
                kmeans_init=True,
                decay=0.99,
                commitment_weight=1.0,
                accept_image_fmap=True,
                sync_codebook=use_ddp,
            )
        else:
            self.codebooks = None

    def forward(self, x):
        y = torch.cat([x, self.pe(x)], dim=1)
        y = self.project_in(y)
        y = self.down(y)
        if self.codebooks is None:
            return self.quant_conv(y)
        else:
            with torch.cuda.amp.autocast(False):
                y = y.to(dtype=torch.float32)
                y = self.quant_conv(y)
                qs, ixs, ls = self.codebooks(y)
            ixs = rearrange(ixs, "c b h w -> b c h w")
            l_quant = torch.mean(ls)
            return y, qs, ixs, l_quant


class Upsampler(nn.Sequential):
    def __init__(self, ch_in: int, ch_out: int, factor: int = 2):
        super().__init__(
            nn.Conv2d(ch_in, ch_out * factor ** 2, 3, padding=1),
            nn.PixelShuffle(factor),
        )


class SkipUpsampler(Upsampler):
    def __init__(self, ch_in: int, ch_out: int, factor: int = 2):
        super().__init__(ch_in * 2, ch_out, factor)


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
        use_batchnorm: bool = True,
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
                    Block(
                        d_in,
                        num_blocks,
                        ff_mult=ff_mult,
                        use_attn=use_attn,
                        use_batchnorm=use_batchnorm,
                    ),
                    Upsampler(d_in, d_out, 2),
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
        num_codes: int = 2,
        num_codebooks: int = 14,
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
            num_codes=num_codes,
            num_codebooks=num_codebooks,
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


def compute_channel_change_mat(io_ratio):
    base = torch.eye(1)
    if io_ratio < 1:
        # reduce channels
        c_in = int(1 / io_ratio)
        cmat = repeat(base * io_ratio, "i1 i2 -> i1 (i2 m)", m=c_in)
    else:
        c_out = int(io_ratio)
        cmat = repeat(base, "i1 i2 -> (i1 m) i2", m=c_out)
    return cmat


class Downsample2d(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        dkern = torch.tensor([[1, 3, 3, 1]]) / 8
        dkern = dkern.T @ dkern
        cmat = compute_channel_change_mat(c_out / c_in)
        weight = torch.einsum("hw,oi->oihw", dkern, cmat)
        self.register_buffer("weight", weight)

    def forward(self, x):
        n, c, h, w = x.shape
        o, i, _, _ = self.weight.shape
        groups = c // i
        x = rearrange(x, "b (g c) h w -> (b g) c h w", g=groups)
        x = F.conv2d(x, self.weight, padding=1, stride=2)
        x = rearrange(x, "(b g) c h w -> b (g c) h w", g=groups)
        return x


class Upsample2d(nn.Module):
    def __init__(self, c_in, c_out):
        # todo: figure out how to do this w/a fused kernel
        super().__init__()
        cmat = compute_channel_change_mat(c_out / c_in)
        self.register_buffer("weight", cmat[:, :, None, None])
        self.upsampler = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        n, c, h, w = x.shape
        o, i, _, _ = self.weight.shape
        groups = c // i
        x = rearrange(x, "b (g c) h w -> (b g) c h w", g=groups)
        x = F.conv2d(x, self.weight)
        x = rearrange(x, "(b g) c h w -> b (g c) h w", g=groups)
        return self.upsampler(x)


class DiffusionDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        f: int = 8,
        num_blocks: int = 2,
        up_mults: Sequence[int] = None,
        ff_mult: int = 2,
        out_ch: int = 3,
        pe_dim: int = 8,
        codebook_dim: int = 16,
        decoder_use_attn: bool = False,
        u_net_use_attn: bool = True,
        u_net_stages: int = 4,
        u_net_ff_mult: int = None,
        u_net_mults: Sequence[int] = None,
        num_diffusion_blocks: int = None,
        time_emb_dim: int = 4,
    ):
        super().__init__()
        self.f = f
        self.out_ch = out_ch
        num_diffusion_blocks = (
            num_diffusion_blocks if num_diffusion_blocks else num_blocks
        )
        u_net_ff_mult = u_net_ff_mult if u_net_ff_mult else ff_mult
        self.dec_up = Decoder(
            dim,
            f=f,
            num_blocks=num_blocks,
            mults=up_mults,
            ff_mult=ff_mult,
            out_ch=dim,
            pe_dim=pe_dim,
            codebook_dim=codebook_dim + time_emb_dim,
            use_attn=decoder_use_attn,
            use_batchnorm=False,
        )

        self.time_emb = FourierFeatures(1, time_emb_dim)
        self.pos_emb = FourierPosEmb(pe_dim)
        self.conv_in = nn.Conv2d(3 + dim + time_emb_dim + pe_dim, dim, 3, padding=1)

        if u_net_mults is None:
            u_net_mults = [2 ** (i + 1) for i in range(u_net_stages)]
        elif len(u_net_mults) < u_net_stages:
            u_net_mults = u_net_mults + [
                u_net_mults[-1] for _ in range(u_net_stages - len(u_net_mults))
            ]
        elif len(u_net_mults) > u_net_stages:
            u_net_mults = u_net_mults[:u_net_stages]
        u_net_mults = [1] + u_net_mults
        dims = dim * np.array(u_net_mults)

        self.down_stages = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Downsample2d(d_in, d_out),
                        Block(
                            d_out,
                            num_blocks=num_diffusion_blocks,
                            ff_mult=u_net_ff_mult,
                            use_attn=u_net_use_attn,
                            use_batchnorm=False,
                        ),
                    ]
                )
                for d_in, d_out in zip(dims[:-1], dims[1:])
            ]
        )
        updims = list(reversed(dims))
        self.up_stages = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Block(
                            d_in,
                            num_blocks=num_diffusion_blocks,
                            ff_mult=u_net_ff_mult,
                            use_attn=u_net_use_attn,
                            use_batchnorm=False,
                        ),
                        Upsample2d(d_in, d_out),
                    ]
                )
                for d_in, d_out in zip(updims[:-1], updims[1:])
            ]
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(dim * 2 + time_emb_dim, dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, out_ch, 3, padding=1),
        )

    def forward(self, z, noised_real, t):
        _, _, zh, zw = z.shape
        _, _, h, w = noised_real.shape
        te = self.time_emb(t[:, None])[:, :, None, None]
        zt = torch.cat([z, te.expand(-1, -1, zh, zw)], dim=1)
        zu = self.dec_up(zt)
        pe = self.pos_emb(noised_real)
        with torch.cuda.amp.autocast(False):
            zu = zu.to(dtype=noised_real.dtype)
            te = te.to(dtype=noised_real.dtype)
            pe = pe.to(dtype=noised_real.dtype)
            x = torch.cat([zu, te.expand(-1, -1, h, w), pe, noised_real], dim=1)
            x = self.conv_in(x)

        skips = []

        for down, block in self.down_stages:
            x = down(x)
            skips.append(x)
            x = block(x)

        for block, up in self.up_stages:
            x = block(x)
            skip = skips.pop()
            x = (x + skip) / 2
            x = up(x)

        with torch.cuda.amp.autocast(False):
            zu = zu.to(dtype=torch.float32)
            te = te.to(dtype=torch.float32)
            x = x.to(dtype=torch.float32)
            return self.conv_out(torch.cat([zu, te.expand(-1, -1, h, w), x], dim=1))

    def ddim_sample(
        self, z, start: Tuple[float, torch.Tensor] = None, num_steps: int = 16
    ):
        b, _, hz, wz = z.size()
        h, w = hz * self.f, wz * self.f
        if start is not None:
            x, t = start
        else:
            x = torch.randn([b, self.out_ch, h, w], device=z.device)
            t = torch.ones(b, device=z.device)
        step_weights = torch.linspace(0.0, 1.0, num_steps, device=z.device)
        ts = torch.lerp(t[None, :], torch.zeros_like(t)[None, :], step_weights[:, None])
        dists = ts[:-1] - ts[1:]
        deltas = math.pi / 2 * dists

        for t, delta in zip(ts[:-1], tqdm(deltas)):
            v = self(z, x, t)
            x = (
                x * torch.cos(delta)[:, None, None, None]
                - v * torch.sin(delta)[:, None, None, None]
            )

        return x


class DiffusionAE(nn.Module):
    def __init__(
        self,
        dim: int = 32,
        f: int = 8,
        num_blocks: int = 2,
        mults: Sequence[int] = None,
        down_mults: Sequence[int] = None,
        up_mults: Sequence[int] = None,
        ff_mult: int = 2,
        out_ch: int = 3,
        pe_dim: int = 8,
        num_codes: int = 2,
        num_codebooks: int = 14,
        codebook_dim: int = 16,
        use_ddp: bool = True,
        encoder_use_attn: bool = False,
        decoder_use_attn: bool = False,
        u_net_use_attn: bool = False,
        u_net_stages: int = 4,
        u_net_ff_mult: int = None,
        u_net_mults: Sequence[int] = None,
        num_diffusion_blocks: int = None,
        time_emb_dim: int = 4,
    ):
        super().__init__()
        mults = (
            mults
            if mults
            else up_mults
            if up_mults
            else down_mults
            if down_mults
            else u_net_mults
            if u_net_mults
            else None
        )
        if down_mults is None:
            down_mults = mults
        if up_mults is None:
            up_mults = mults
        if u_net_mults is None:
            u_net_mults = mults
        self.encoder = Encoder(
            dim,
            f=f,
            num_blocks=num_blocks,
            mults=down_mults,
            ff_mult=ff_mult,
            in_ch=out_ch,
            pe_dim=pe_dim,
            num_codes=num_codes,
            num_codebooks=num_codebooks,
            codebook_dim=codebook_dim,
            use_ddp=use_ddp,
            use_attn=encoder_use_attn,
            use_batchnorm=False,
        )
        self.decoder = DiffusionDecoder(
            dim,
            f=f,
            num_blocks=num_blocks,
            up_mults=up_mults,
            ff_mult=ff_mult,
            out_ch=out_ch,
            pe_dim=pe_dim,
            codebook_dim=codebook_dim,
            decoder_use_attn=decoder_use_attn,
            u_net_use_attn=u_net_use_attn,
            u_net_stages=u_net_stages,
            u_net_ff_mult=u_net_ff_mult,
            u_net_mults=u_net_mults,
            num_diffusion_blocks=num_diffusion_blocks,
            time_emb_dim=time_emb_dim,
        )
