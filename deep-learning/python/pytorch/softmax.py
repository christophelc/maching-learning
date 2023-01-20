import torch
import torch.nn as nn
from torch.nn import Linear
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pyplot as plt

class Softmax(nn.Module):
    def __init__(self,in_size,out_size):
        super(Softmax, self).__init__()
        self.linear = nn.Linear(in_size, out_size)

    def forward(self,x):
        out = self.linear(x)
        return out

#parameters intial values
W=list(model.parameters())[0]
b=list(model.parameters())[1]
print('W', W.size())
print('b', b.size())

#see https://pytorch.org/vision/stable/auto_examples/plot_visualization_utils.html
def showW(imgs):
    square = 28
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.reshape(square,square).detach()
        axs[0, i].imshow(img)
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


train_dataset = dsets.MNIST(root='./data', train=True,download=True,transform=transforms.ToTensor())
validation_dataset = dsets.MNIST(root='./data', train=False,download=True,transform=transforms.ToTensor())

torch.manual_seed(1)
model=Softmax(2,3)
x=torch.tensor([[1.0, 2.0]])
z=model(x)
_,yhat = z.max(1)
print(yhat)
print("====")

x=torch.tensor([[1.0, 1.0],[1.0, 2.0], [1.0, 3.0]])
z=model(x)
print(z)
_,yhat = z.max(1)
print(yhat)

input_dim = 28*28
output_dim = 10
model=Softmax(input_dim, output_dim)
#show initial W parameters as image
showW(W)

criterion=nn.CrossEntropyLoss()
optimiser = optim.SGD(model.parameters(), lr=0.01)
n_epochs=100
accuracy_list = []
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100)
N_test = len(validation_dataset)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000)

for epoch in range(n_epochs):
    for x,y in train_loader:
        optimiser.zero_grad()
        z = model(x, view(-1, 28*28))
        loss = criterion(z, y)
        loss.backward()
        optimiser.step()

        correct = 0
        for x_test, y_test in validation_loader:
            _, yhat = torch.max(z.data, 1)
            correct = correct + (yhat == y_test).sum().item()
        accuracy = correct / N_test
        accuracy_list.append(accuracy)
