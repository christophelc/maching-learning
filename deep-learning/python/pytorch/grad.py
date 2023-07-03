import torch
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]=""

x=torch.tensor([1.,2.,2.]).requires_grad_()
l=x.norm()
print(l)
g=torch.autograd.grad(l, (x,))
print(g)

######
x=torch.tensor([1.,2.,2.]).requires_grad_()
l=x.norm()
l.backward()
print(x.grad)

######
print()
x=torch.tensor([1.,2.,2.]).requires_grad_()
phi=x.pow(2).sum()
g1=torch.autograd.grad(phi,x,create_graph=True)
print(g1)
psi=g1[0][0].exp()-g1[0][2].exp()
g2=torch.autograd.grad(psi,x)
print(g2)
