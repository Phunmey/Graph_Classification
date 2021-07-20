import numpy as np
from ripser import ripser
import pandas as pd
from igraph import *
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from numpy import inf
from time import time
from Helper_Functions import *


def standard_graph_file(dataset):
    start = time()
    edge_data = pd.read_csv(config['data_path']+  config['name'] + "_A.txt", header=None)
    edge_data.columns = ['from', 'to']
    graph_labels = pd.read_csv(config['data_path'] + config['name'] + "_graph_indicator.txt", header=None)
    edge_labels = pd.read_csv(config['data_path'] + config['name'] + "_graph_labels.txt", header=None)
    grapher = sum(graph_labels.values.tolist(), [])
    data = list(set(grapher))  # counting unique graph ids

    training_set, test_set = train_test_split(data, train_size=config['train_size'], test_size=config['test_size'])

    # BETTI NUMBERS
    ################
    train_bet = []
    for i in training_set:
        norm_distmat = get_norm_dist_mat(i, graph_labels, edge_data)
        [m, M] = [np.nanmin(norm_distmat), np.nanmax(norm_distmat)]
        diagrams = ripser(norm_distmat, thresh=config['thresh'], maxdim=config['maxdim'], distance_matrix=True)['dgms']
        betti_graph_labels = get_betti_graph_labels(diagrams, M, data, edge_labels, i)
        train_bet.append(betti_graph_labels)

    test_bet = []
    for w in test_set:
        normtest_distmat = get_norm_dist_mat(w, graph_labels, edge_data)
        testdiagrams = ripser(normtest_distmat, thresh=config['thresh'], maxdim=config['maxdim'], distance_matrix=True)['dgms']
        betti_graph_labels_test = get_betti_graph_labels(testdiagrams, M, data, edge_labels, w)
        test_bet.append(betti_graph_labels_test)

    train_data = pd.DataFrame(train_bet)
    test_data = pd.DataFrame(test_bet)

    train_features = train_data.iloc[:, :-1].values
    train_labels = train_data.iloc[:, -1].values
    test_features = test_data.iloc[:, :-1].values
    test_labels = test_data.iloc[:, -1].values


    # RANDOM FOREST HYPERPARAMETERS TUNING
    ######################################
    max_features = ['auto', 'sqrt']
    n_estimators = [int(a) for a in np.linspace(start=10, stop=100, num=10)]
    max_depth = [int(b) for b in np.linspace(start=2, stop=10, num=5)]
    #min_samples_split = [2, 5, 10]
    #min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    param_grid = dict(max_features=max_features, n_estimators=n_estimators, max_depth=max_depth,
                      min_samples_leaf=config['min_samples_leaf'], min_samples_split=config['min_samples_split'], bootstrap=bootstrap)

    rfc = RandomForestClassifier()
    grid = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=config['cv'], n_jobs=1)
    grid.fit(train_features, train_labels)  # FIRSTMM_DB dataset doesn't work with this
    param_choose = grid.best_params_

    rfc_pred = RandomForestClassifier(**param_choose, random_state=1).fit(train_features, train_labels)
    test_pred = rfc_pred.predict(test_features)


    # PREDICTION PROBABILITIES
    ##########################
    r_probs = [0 for _ in range(len(test_labels))]  # worst case scenario
    # predict the class probabilities for K_test and keep the positive outcomes
    rfc_probs = (rfc_pred.predict_proba(test_features))[:, 1]
    # Compute area under the receiver operating characteristic (ROC) curve for worst case scenario
    r_auc = roc_auc_score(test_labels, r_probs, multi_class=config['multi_class'])
    # Compute area under the receiver operating characteristic curve for RandomForest # problem with ENZYME here
    rfc_auc = roc_auc_score(test_labels, rfc_probs)

    tsv_writer.writerow([config['name'], 'NA', '%.3f' % r_auc, '%.3f' % rfc_auc, accuracy_score(test_labels, test_pred),
                         time() - start])  # if you want more output you must include here and update column names

    r_fpr, r_tpr, thresholds = roc_curve(test_labels, r_probs, config['pos_label'])
    rfc_fpr, rfc_tpr, thresholds = roc_curve(test_labels, rfc_probs, config['pos_label'])  # compute ROC

    plt.figure(figsize=(3, 3), dpi=100)
    plt.plot(r_fpr, r_tpr, marker='.', label='Chance prediction (AUROC= %.3f)' % r_auc)
    plt.plot(rfc_fpr, rfc_tpr, linestyle='-', label='RFC (AUROC= %.3f)' % rfc_auc)
    plt.title('ROC Plot')  # title
    plt.xlabel('False Positive Rate')  # x-axis label
    plt.ylabel('True Positive Rate')  # y-axis label
    plt.savefig(config['graph_dir'] + config['name'] + ".png")  # save the plot
    #plt.legend()  # show legend
    #plt.show()  # show plot


def get_norm_dist_mat(id, labels, edges):
    graph_nodes = labels[labels.iloc[:, 0] == id].index.tolist()
    graph_edges = edges[edges.index.isin(graph_nodes)]
    graph = Graph.TupleList(graph_edges.itertuples(index=False), directed=False, weights=True)
    distmat = np.asarray(Graph.shortest_paths_dijkstra(graph))
    distmat[distmat == inf] = 0
    [mi, ma] = [np.nanmin(distmat), np.nanmax(distmat)]
    norm_distmat = distmat / ma
    return norm_distmat


def get_betti_graph_labels(persistence_diagram, M, data, edgelabels, id):
    step = 0.05
    eps = np.arange(0, M + step, step)
    # splitting the dimension into 0, 1 and 2
    h_0 = persistence_diagram[0]
    h_1 = persistence_diagram[1]
    h_2 = persistence_diagram[2]

    bb_0 = []
    bb_1 = []
    bb_2 = []

    for q in eps:
        b_0 = 0
        for h in h_0:
            if h[0] <= q and h[1] > q:
                b_0 = b_0 + 1
        bb_0.append(b_0)

        b_1 = 0
        for y in h_1:
            if y[0] <= q and y[1] > q:
                b_1 = b_1 + 1
        bb_1.append(b_1)

        b_2 = 0
        for e in h_2:
            if e[0] <= q and e[1] > q:
                b_2 = b_2 + 1
        bb_2.append(b_2)

    # concatenate betti numbers
    betti_numbers = bb_0 + bb_1 + bb_2
    # obtain the locations of the graphid
    graph_id_location = data.index(id)
    # extract the corresponding graphlabels of the graphid
    graph_labels_location = (edgelabels.values[graph_id_location]).tolist()
    # save betti numbers with graph labels
    betti_graph_labels = betti_numbers + graph_labels_location
    return betti_graph_labels


if __name__ == '__main__':
    configs = get_configs('Ripser_RFC')
    out_file = open("../results/Ripser_RFC/Ripser_RFC_output.tsv", 'wt')  # opens output file
    tsv_writer = csv.writer(out_file, delimiter="\t")
    tsv_writer.writerow(["dataset", "kernel", "Random_prediction_(AUROC)", "RFC_(AUROC)", "accuracy_score", "run_time"])
    # runs standardGraphFile for all datasets
    for config in configs:
        print("PROCESSING DATA SET: " + config['name'])
        standard_graph_file(config)
