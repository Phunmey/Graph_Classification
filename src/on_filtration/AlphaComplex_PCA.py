import random
from datetime import datetime
from time import time

import gudhi as gd
import numpy as np
import pandas as pd
from igraph import *
from numpy import inf
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV


def standardGraphFile(dataset, file, data_path, iter, step):
    start = time()

    df_edges = pd.read_csv(data_path + "/" + dataset + "/" + dataset + "_A.txt", header=None)  # import edge data
    df_edges.columns = ['from', 'to']
    unique_nodes = ((df_edges['from'].append(df_edges['to'])).unique()).tolist()
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
    unique_graph_indicator = np.arange(min(graph_indicators),
                                       max(graph_indicators) + 1)  # list unique graph ids

    random.seed(42)
    train_set, test_set, y_train, y_test = train_test_split(unique_graph_indicator, graph_labels, train_size=0.8,
                                                            test_size=0.2)  # split the dataset to train and test sets

    train_bet = []
    for i in train_set:
        train_graph_id = i
        train_id_loc = [index for index, element in enumerate(graph_indicators) if
                        element == train_graph_id]  # list the index of the graphid locations
        train_graph_edges = df_edges[df_edges['from'].isin(train_id_loc)]
        train_graph = Graph.TupleList(train_graph_edges.itertuples(index=False), directed=False, weights=True)
        train_dist_matrix = np.asarray(Graph.shortest_paths_dijkstra(train_graph))
        train_dist_matrix[train_dist_matrix == inf] = 0
        sym_matrix = np.matmul(train_dist_matrix, train_dist_matrix.T)  # obtain a symmetric matrix
        [min1, max1] = [np.nanmin(sym_matrix), np.nanmax(sym_matrix[sym_matrix != np.inf])]
        norm_dist_matrix = sym_matrix / max1  # normalized distancematrix
        train_dim_reduction = PCA(n_components=3).fit_transform(norm_dist_matrix)
        train_alpha_complex = gd.AlphaComplex(points=train_dim_reduction)
        train_simplex_tree = train_alpha_complex.create_simplex_tree()
        train_diagrams = np.asarray(train_simplex_tree.persistence(), dtype='object')
        # gd.plot_persistence_diagram(diagrams)
        # plt.show()

        # splitting the dimension into 0 and 1
        train_persist_0 = train_diagrams[0]
        train_persist_1 = train_diagrams[1]

        # obtain betti numbers for the unique dimensions
        threshold = np.arange(0, np.nanmax(norm_dist_matrix) + step, step)
        train_betti_0 = []
        train_betti_1 = []
        for j in threshold:
            b_0 = 0
            for k in train_persist_0:
                if k[0] <= j and k[1] > j:
                    b_0 = b_0 + 1
            train_betti_0.append(b_0)

            b_1 = 0
            for l in train_persist_1:
                if l[0] <= j and l[1] > j:
                    b_1 = b_1 + 1
            train_betti_1.append(b_1)

        train_bet.append(train_betti_0 + train_betti_1)  # concatenate betti numbers

    test_bet = []
    for w in test_set:
        test_graph_id = w
        test_id_loc = [index for index, element in enumerate(graph_indicators) if
                       element == test_graph_id]  # list the index of the graph_id locations
        test_graph_edges = df_edges[
            df_edges['from'].isin(test_id_loc)]  # obtain edges that with source node equal to test_id_loc
        test_graph = Graph.TupleList(test_graph_edges.itertuples(index=False), directed=False, weights=True)
        test_dist_matrix = np.asarray(Graph.shortest_paths_dijkstra(test_graph))  # obtain the distance matrix
        test_dist_matrix[test_dist_matrix == inf] = 0  # replace inf with zero
        test_sym_matrix = np.matmul(test_dist_matrix, test_dist_matrix.T)  # obtain a symmetric matrix
        [min2, max2] = [np.nanmin(test_sym_matrix), np.nanmax(test_sym_matrix[test_sym_matrix != np.inf])]
        testnorm_dist_matrix = test_sym_matrix / max2  # normalized distancematrix
        test_dim_reduction = PCA(n_components=3).fit_transform(testnorm_dist_matrix)
        test_alpha_complex = gd.AlphaComplex(points=test_dim_reduction)  # initialize alpha complex
        test_simplex_tree = test_alpha_complex.create_simplex_tree()  # creating a simplex tree
        test_diagrams = np.asarray(test_simplex_tree.persistence(),
                                   dtype='object')  # run AlphaComplex filtartion on the normalized distance matrix
        # gd.plot_persistence_diagram(diagrams)
        # plt.show()

        # splitting the dimension into 0 and 1
        test_persisted_0 = test_diagrams[0]
        test_persisted_1 = test_diagrams[1]

        test_betti_0 = []
        test_betti_1 = []
        for q in threshold:
            b_0 = 0
            for h in test_persisted_0:
                if h[0] <= q and h[1] > q:
                    b_0 = b_0 + 1
            test_betti_0.append(b_0)

            b_1 = 0
            for y in test_persisted_1:
                if y[0] <= q and y[1] > q:
                    b_1 = b_1 + 1
            test_betti_1.append(b_1)

        test_bet.append(test_betti_0 + test_betti_1)  # concatenate betti numbers

    train_data = pd.DataFrame(train_bet)
    test_data = pd.DataFrame(test_bet)
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
    file.write(dataset + "\t" + str(time_taken) + "\t" + str(t3 - t2) +
               "\t" + str(accuracy) + "\t" + str(auc) +
               "\t" + str(iter) + "\t" + str(step) + "\t" + str(flat_conf_mat) + "\n")
    file.flush()


if __name__ == '__main__':
    data_path = sys.argv[1]  # dataset path on computer
    datasets = (
        'ENZYMES', 'BZR', 'MUTAG', 'DD', 'PROTEINS', 'DHFR', 'NCI1', 'COX2', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K')
    outputFile = "../results/" + 'AlphaPCAresult.csv'
    file = open(outputFile, 'w')
    for dataset in datasets:
        for iter_ in (2, 3, 4):
            for step_ in (0.05, 0.1, 0.15, 0.2, 0.5):
                for duplication in np.arange(5):
                    standardGraphFile(dataset, file, data_path, iter=iter_, step=step_)
    file.close()
