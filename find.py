from sklearn.neighbors import BallTree
import numpy as np
from collections import defaultdict
import time
import random
import stumpy
from sklearn.neighbors import NearestNeighbors

dataset = 'BinaryHeartbeat'

prototypes = np.load('.data/data_center/prototypes.npz')
time_series = np.load('.data/raw/')


dim = 32
num_sketch = 1024
num_subsequences = 16384
# for validtion
X1 = np.random.rand(num_subsequences, dim) #subsequences
X2 = np.random.rand(num_sketch, dim) #sketch

tree = BallTree(X2, leaf_size=40)

t_1_start = time.time()


dist, ind = tree.query(X1, k=1)


labels = ind.flatten()



x2_as_label_set = set(labels)

x2_to_x1s = defaultdict(list)
for i, label in enumerate(labels):
    x2_to_x1s[label].append(i)


x2_to_nearest_x1 = dict()

t_1_4 = time.time()

t_1_have_all = 0
t_1_no_all = 0
t_1_no_need = 0
for idx in range(X2.shape[0]):
    if idx in x2_to_x1s:
        x1_indices = x2_to_x1s[idx]
        x1_vectors = X1[x1_indices]
        dists = np.linalg.norm(x1_vectors - X2[idx], axis=1)
        nearest_x1_idx = x1_indices[np.argmin(dists)]
        x2_to_nearest_x1[idx] = nearest_x1_idx

    else:

        labeled_x2s = np.array(list(x2_as_label_set))
        dists = np.linalg.norm(X2[labeled_x2s] - X2[idx], axis=1)
        nearest_labeled_x2 = labeled_x2s[np.argmin(dists)]


        x1_indices = x2_to_x1s[nearest_labeled_x2]
        x1_vectors = X1[x1_indices]
        dists2 = np.linalg.norm(x1_vectors - X2[idx], axis=1)
        nearest_x1_idx = x1_indices[np.argmin(dists2)]
        x2_to_nearest_x1[idx] = nearest_x1_idx



