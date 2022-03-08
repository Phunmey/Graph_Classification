import random
from time import time

import numpy as np
from grakel.datasets import fetch_dataset
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from sklearn.model_selection import train_test_split

start = time()
bzr = fetch_dataset("ENZYMES", verbose=True, produce_labels_nodes=True)
G, y = bzr.data, bzr.target

H = []
for i in G:
    h = i[0]
    H.append(list(h))

u = sum(H, [])
v = sum(u, ())
r = set(v)

labels = np.arange(min(y), max(y) + 1)
# y1 = label_binarize(y, classes=labels)
# n_classes = len(labels)

g_train, g_test, y_train, y_test = train_test_split(G, y, test_size=0.2, random_state=random.randint(0, 100))

wl = WeisfeilerLehman(base_graph_kernel=VertexHistogram, normalize=True, n_jobs=1, n_iter=5)
k_train = wl.fit_transform(g_train)
k_test = wl.transform(g_test)
