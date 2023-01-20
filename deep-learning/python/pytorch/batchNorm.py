import torch
import torch.nn as nn

x = torch.Tensor(10000,3).normal_()
print(x.dim())
print(x.size(dim=0))
print(x.size(dim=1))
x = x * torch.Tensor([2., 5., 10.]) + torch.Tensor([-10., 25, 3.])
print(x.mean(0))
print(x.std(0))

bn = nn.BatchNorm1d(3)
with torch.no_grad():
    bn.bias.copy_(torch.tensor([2., 4., 8.]))
    bn.weight.copy_(torch.tensor([1., 2., 3.]))
y = bn(x)
print("y normalization (not data dependent)")
print(y.mean(0))
