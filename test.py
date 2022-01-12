import torch
from net import Downsample2d, Upsample2d
t = torch.randn([3,32,14,14])
d = Downsample2d(1,2)
u = Upsample2d(2, 1)
t_ = d(t)
b, c, h, w = t.shape
b_, c_, h_, w_ = t_.shape
assert b_ == b
assert c_ == c * 2
assert h_ == h // 2
assert w_ == w // 2

t_ = u(t)
b_, c_, h_, w_ = t_.shape
assert b_ == b
assert c_ == c // 2
assert h_ == h * 2
assert w_ == w * 2

t__ = d(u(t))
assert t.shape == t__.shape
print('max error: ', t.sub(t__).abs().max())

print('success')
