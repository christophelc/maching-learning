import torch
import numpy as np

data = [[1, 2],[3, 4]]

#Directly from data
x_data = torch.tensor(data)

#From a NumPy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

#From another tensor
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

#With random or constant values
# shape is a tuple of tensor dimensions
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

#Attributes of a Tensor
tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

#Operations on Tensors
print(f"GPU available: {torch.cuda.is_available()} \n")
# We move our tensor to the GPU if available
if torch.cuda.is_available():
  tensor = tensor.to('cuda')

#tensor operations examples
#cross product on the same device  
print(tensor)
print(torch.cross(tensor, torch.ones_like(tensor).to("cuda")))

print(ones_tensor)
print(torch.cross(ones_tensor, ones_tensor))

#mesh grid
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])
grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
print(f"grid_x: \n {grid_x} \n")
print(f"grid_y: \n {grid_y} \n")
import matplotlib.pyplot as plt
xs = torch.linspace(-5, 5, steps=100)
ys = torch.linspace(-5, 5, steps=100)
x, y = torch.meshgrid(xs, ys, indexing='xy')
z = torch.sin(torch.sqrt(x * x + y * y))
ax = plt.axes(projection='3d')
ax.plot_surface(x.numpy(), y.numpy(), z.numpy())
plt.show()

#Standard numpy-like indexing and slicing:
tensor = torch.ones(4, 4)
print('First row: ',tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])
tensor[:,1] = 0
print(tensor)

# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
print(f"y1 = \n {y1} \n")
print(f"y2 = \n {y1} \n")
print(torch.matmul(tensor, tensor.T, out=y3))

# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
print(f"z1 = \n {z1} \n")
print(f"z2 = \n {z1} \n")
print(torch.mul(tensor, tensor, out=z3))

#Single-element tensors
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

#In-place operations
print(tensor, "\n")
tensor.add_(5)
print(tensor)

#Bridge with NumPy
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

#NumPy array to Tensor
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")

