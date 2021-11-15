from time import time

from grakel.datasets import fetch_dataset
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

start = time()
bzr = fetch_dataset("BZR", verbose=True, produce_labels_nodes=True)
G, y = bzr.data, bzr.target

# labels = np.arange(min(y), max(y)+1)
# y1 = label_binarize(y, classes=labels)
# n_classes = len(labels)

g_train, g_test, y_train, y_test = train_test_split(G, y, test_size=0.2, random_state=42)

wl = WeisfeilerLehman(base_graph_kernel=VertexHistogram, normalize=True, n_jobs=1, n_iter=5)
k_train = wl.fit_transform(g_train)
k_test = wl.transform(g_test)

# classifier = OneVsRestClassifier(RandomForestClassifier(max_depth=2, random_state=42))
# rfc = classifier.fit(k_train, y_train)
rfc = RandomForestClassifier(max_depth=5, random_state=42).fit(k_train, y_train)
test_pred = rfc.predict(k_test)
# print([test_pred, len(test_pred)])
rfc_probs = rfc.predict_proba(k_test)[:, 1]
# print([rfc_probs, len(rfc_probs)])

acc = accuracy_score(y_test, test_pred)
print(acc)
roc = roc_auc_score(y_test, rfc_probs)
print(roc)
