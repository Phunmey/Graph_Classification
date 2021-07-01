import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from grakel.datasets import fetch_dataset
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from time import time


def Standardkernel():
    start = time()
    DATA = fetch_dataset("PROTEINS", verbose=True)
    G, y = DATA.data, DATA.target

    G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.2, random_state=42)

    gk = WeisfeilerLehman(n_iter=4, base_graph_kernel=VertexHistogram, normalize=True)
    K_train = gk.fit_transform(G_train)
    K_test = gk.transform(G_test)

    RFC_pred = RandomForestClassifier()
    RFC_pred.fit(K_train, y_train)
    y_pred = RFC_pred.predict(K_test)

    # PREDICTION PROBABILITIES
    r_probs = [0 for _ in range(len(y_test))]  # worst case scenario
    rf_probs = RFC_pred.predict_proba(K_test)
    RFC_probs = rf_probs[:, 1]  # keep the probabilities of positive outcomes
    r_auc = roc_auc_score(y_test, r_probs)
    RFC_auc = roc_auc_score(y_test, RFC_probs)

    print('Chance prediction: AUROC = %.3f' % (r_auc))
    print('RFC: AUROC = %.3f' % (RFC_auc))
    print(accuracy_score(y_test, y_pred))
    print(f'Time taken to run:{time() - start} seconds')

 #   r_fpr, r_tpr, _ = roc_curve(y_test, r_probs)
 #   RFC_fpr, RFC_tpr, _ = roc_curve(y_test, RFC_probs)


if __name__ == '__main__':
    Standardkernel()


