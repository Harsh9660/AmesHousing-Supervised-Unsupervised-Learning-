import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X,y = make_blobs(n_samples= 500, n_features = 2, centers= 3, random_state=23)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.grid(True)
plt.scatter(X[:,0], X[:,1], c=y)
plt.show()

k = 3
clusters = {i: [] for i in range(k)}
np.random.seed(23)

for idx in range(len(k)):
    center = 2*(2*np.random.random((X.shape[1],)) - 1)
    points = []
    clusters = {
        'clusters': center,
        'points': points
    }

    clusters[idx] = clusters
clusters