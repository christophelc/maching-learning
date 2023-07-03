import torch
import torch.nn as nn

x= torch.full((3,5), 1.0).requires_grad_()
print(x)
dropout = torch.nn.Dropout(p = 0.75)
y = dropout(x)
print(y)

#norm according to dim 1 (for each row)
#p=2 means usual distance
l = y.norm(p=2, dim=1).sum()
print(y.norm(p=2, dim=1))
#l=y.norm(p=2, dim=1).sum()
l.backward()
print(x.grad)
print(4/2.8284)
