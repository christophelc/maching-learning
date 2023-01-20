import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

colors=["red", "green"]
X, y = make_blobs(n_samples = 100, n_features = 2, centers = 2, random_state = 0)
y = y.reshape((y.shape[0], 1))

print("dimension of X:", X.shape)
print("dimension of Y:", y.shape)

plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.get_cmap('bwr',2))
plt.xlabel("x1", fontweight='bold')
plt.ylabel("x2", fontweight='bold')
cbar = plt.colorbar(orientation='vertical', pad = 0.1)
cbar.set_label(label='y', size = 2)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(["y = 0", "y = 1"])
plt.show()

#A=np.array([4,6,5])
#B=A*2
#print(B)
