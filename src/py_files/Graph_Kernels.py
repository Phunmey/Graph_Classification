import numpy as np
import pandas as pd
import random
from datetime import datetime
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from igraph import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from time import time


def standardGraphFile(dataset, file, data_path, iter):
    start = time()

    df_edges = pd.read_csv(data_path + "/" + dataset + "/" + dataset + "_A.txt", header=None)  # import edge data
    df_edges.columns = ['from', 'to']
    unique_nodes = ((df_edges['from'].append(df_edges['to'])).unique()).tolist()  # list all nodes in the dataset
    print("Graph edges are loaded")
    node_list = np.arange(min(unique_nodes), max(unique_nodes) + 1)  # unique_nodes + missing_nodes
    node_list.sort()
    csv = pd.read_csv(data_path + "/" + dataset + "/" + dataset + "_graph_indicator.txt", header=None)
    print("Graph indicators are loaded")
    csv.columns = ["ID"]
    graph_indicators = (csv["ID"].values.astype(int))
    read_csv = pd.read_csv(data_path + "/" + dataset + "/" + dataset + "_graph_labels.txt", header=None)
    read_csv.columns = ["ID"]
    graph_labels = (read_csv["ID"].values.astype(int))
    print("Graph labels are loaded")
    random_nodelabels = [5] * len(node_list)
    nodes_dict = dict(zip(node_list, random_nodelabels))
    unique_graph_indicator = np.arange(min(graph_indicators),
                                       max(graph_indicators) + 1)  # list unique graph ids

    DATA = []
    for i in unique_graph_indicator:
        graphid = i
        graphid_loc = [index for index, element in enumerate(graph_indicators) if element == graphid]
        edges_loc = df_edges[df_edges['from'].isin(graphid_loc)]
        edges_loc_asset = (edges_loc.to_records(index=False)).tolist()
        nodes_aslist = ((edges_loc['from'].append(edges_loc['to'])).unique()).tolist()
        ext = {k: nodes_dict[k] for k in nodes_aslist if k in nodes_dict}
        edges_nodes = [edges_loc_asset, ext]
        DATA.append(edges_nodes)

    random.seed(123)

    G_train, G_test, y_train, y_test = train_test_split(DATA, graph_labels, test_size=0.2, random_state=42)

    WL = WeisfeilerLehman(n_iter=5, base_graph_kernel=VertexHistogram, verbose=True, normalize=True)
    train_data = WL.fit_transform(G_train)
    test_data = WL.transform(G_test)

    t2 = time()
    time_taken = t2 - start

    # RandomForest hyper-parameters tuning
    max_features = ['auto', 'sqrt']
    n_estimators = [int(a) for a in np.linspace(start=300, stop=300, num=1)]
    max_depth = [int(b) for b in np.linspace(start=5, stop=6, num=2)]
    min_samples_split = [2, 5]
    min_samples_leaf = [2]
    bootstrap = [True, False]
    num_cv = 10
    gridlength = len(n_estimators) * len(max_depth) * len(min_samples_leaf) * len(min_samples_split) * num_cv
    print(str(gridlength) + " RFs will be created in the grid search.")
    Param_Grid = dict(max_features=max_features, n_estimators=n_estimators, max_depth=max_depth,
                      min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, bootstrap=bootstrap)

    print(dataset + " training started at", datetime.now().strftime("%H:%M:%S"))
    rfc = RandomForestClassifier(n_jobs=10)
    grid = GridSearchCV(estimator=rfc, param_grid=Param_Grid, cv=num_cv, n_jobs=10)
    grid.fit(train_data, y_train)
    param_choose = grid.best_params_

    if len(set(y_test)) > 2:  # multiclass case
        print(dataset + " requires multi class RF")
        forest = RandomForestClassifier(**param_choose, random_state=1, verbose=1).fit(train_data, y_train)
        y_pred = forest.predict(test_data)
        y_preda = forest.predict_proba(test_data)
        auc = roc_auc_score(y_test, y_preda, multi_class="ovr", average="macro")
        accuracy = accuracy_score(y_test, y_pred)
        conf_mat = confusion_matrix(y_test, y_pred)
    else:  # binary case
        rfc_pred = RandomForestClassifier(**param_choose, random_state=1, verbose=1).fit(train_data, y_train)
        test_pred = rfc_pred.predict(test_data)
        auc = roc_auc_score(y_test, rfc_pred.predict_proba(test_data)[:, 1])
        accuracy = accuracy_score(y_test, test_pred)
        conf_mat = confusion_matrix(y_test, test_pred)
    print(dataset + " accuracy is " + str(accuracy) + ", AUC is " + str(auc))
    t3 = time()
    print(f'Ripser took {time_taken} seconds, training took {t3 - t2} seconds')
    flat_conf_mat = (str(conf_mat.flatten(order='C')))[1:-1]
    file.write(dataset + "\t" + str(time_taken) + "\t" + str(t3 -   t2) +
               "\t" + str(accuracy) + "\t" + str(auc) +
               "\t" + str(iter) + "\t" + str(flat_conf_mat) + "\n")
    file.flush()


if __name__ == '__main__':
    data_path = sys.argv[1]  # dataset path on computer
    datasets = ('ENZYMES', 'BZR', 'MUTAG', 'PROTEINS', 'DHFR', 'NCI1', 'COX2', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K')
    outputFile = "../results/" + 'kernelresults.csv'
    file = open(outputFile, 'w')
    for dataset in datasets:
        for iter_ in (2, 3, 4):
            for duplication in np.arange(5):
                standardGraphFile(dataset, file, data_path, iter=iter_)
    file.close()
