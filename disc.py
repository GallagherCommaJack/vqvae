import torch
from torch import nn
from torch.nn.modules.activation import LeakyReLU


class ScaleGrad(nn.Identity):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def backward(self, ctx, grad_in):
        return grad_in * self.scale


def downsampler(c_in, c_out=None, factor=2):
    c_out = c_out if c_out else c_in
    return nn.Sequential(
        nn.Conv2d(c_in, c_out // factor ** 2, 3, padding=1), nn.PixelUnshuffle(factor)
    )


def upsampler(c_in, c_out=None, factor=2):
    c_out = c_out if c_out else c_in
    return nn.Sequential(
        nn.Conv2d(c_in, c_out * factor ** 2, 3, padding=1), nn.PixelShuffle(factor)
    )


def ff(dim: int, mult: int = 2):
    return [
        nn.Conv2d(dim, dim * mult, 3, padding=1, bias=False),
        nn.BatchNorm2d(dim * mult),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(dim * mult, dim, 3, padding=1),
    ]


def discriminator(input_ch=3, dim=64, depth=3, grad_scale=None):
    if grad_scale:
        blocks = [ScaleGrad(grad_scale)]
    else:
        blocks = []
    blocks += [downsampler(input_ch, dim), nn.BatchNorm2d(dim)]
    mults = [2 ** i for i in range(depth + 1)]
    dims = [dim * min(mult, 8) for mult in mults]
    act = lambda: nn.LeakyReLU(0.2, inplace=True)

    for d_in, d_out in zip(dims, dims[1:]):
        blocks += [downsampler(d_in, d_out), nn.BatchNorm2d(d_out), act()]

    blocks += ff(dims[-1])
    blocks.append(nn.Conv2d(dims[-1], 1))

    return nn.Sequential(*blocks)
