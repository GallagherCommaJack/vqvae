# %%
from vector_quantize_pytorch import ResidualVQ, VectorQuantize
from diffusion_utils import *
import torch
import pytorch_lightning as pl
from net import WNet

# %%
net = WNet(16, mults=[2], num_blocks=2).cuda()


# %%
# print([type(m) for m in net.encoder.u_net])
print(torch.cat([torch.flatten(p) for p in net.parameters()]).size(0) / 1e6)

# %%
x = torch.randn([8, 3, 256, 256]).cuda()
y, l = net.forward(x)
print(y.shape, l.shape)

# %%
from task import BaseAE

lm = BaseAE(net).cuda()
rc, losses = lm.handle_batch(x)

print(rc.allclose(y), {k: v.shape for k, v in losses.items()})