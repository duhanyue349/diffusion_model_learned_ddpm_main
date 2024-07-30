import torch
import numpy as np
from functools import partial
# def generate_linear_schedule(T, low, high):
#     return np.linspace(low, high, T)
#
# beta=generate_linear_schedule(500,2,9)
# alphas=1-beta
# alphas_cumprod = np.cumprod(alphas)
# to_torch = partial(torch.tensor, dtype=torch.float32)
# a=to_torch(alphas_cumprod)
# t = torch.randint(0, 6, (3,))
# print(t)
# out = a.gather(-1, t)
# print(out)
# x=torch.randn(3,3,224,224)
# x_shape=x.shape
# print(x_shape)
# print(len(x_shape))
# def extract(a, t, x_shape):
#     b, *_ = t.shape
#     out = a.gather(-1, t)
#     return out.reshape(b, *((1,) * (len(x_shape) - 1)))
# y=extract(a,t,x_shape)
# print(y)
for t in range(6 - 1, -1, -1):
    t_batch = torch.tensor([t], ).repeat(3)
    print(t_batch)