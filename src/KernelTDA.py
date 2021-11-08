import random
import shutil
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from igraph import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from datetime import datetime


def standardGraphFile(dataset, file, datapath, h_filt, iter):
    start = time()
    edges_asdf = pd.read_csv(datapath + "/" + dataset + "/" + dataset + "_A.txt", header=None)  # import edge data
    edges_asdf.columns = ['from', 'to']  # import the graphindicators#import graphlabels  # counting unique graph ids
    unique_nodes = ((edges_asdf['from'].append(edges_asdf['to'])).unique()).tolist()
    print(dataset + " graph edges are loaded")
    node_list = np.arange(min(unique_nodes), max(unique_nodes) + 1);  # unique_nodes + missing_nodes
    node_list.sort()
    csv = pd.read_csv(datapath + "/" + dataset + "/" + dataset + "_graph_indicator.txt", header=None)
    print(dataset + " graph indicators are loaded")
    csv.columns = ["ID"]
    graphindicator_aslist = ((csv["ID"].values.astype(int)))
    read_csv = pd.read_csv(datapath + "/" + dataset + "/" + dataset + "_graph_labels.txt", header=None)
    read_csv.columns = ["ID"]
    graphlabels_aslist = ((read_csv["ID"].values.astype(int)))
    print(dataset + " graph labels are loaded")
    random_nodelabels = [5] * len(node_list)
    random_dict = list(dict(zip(node_list, random_nodelabels)).items())
    unique_graphindicator = np.arange(min(graphindicator_aslist),
                                      max(graphindicator_aslist) + 1)  # list unique graphids 100

    random.seed(27)

    progress = len(unique_graphindicator)
    total_degree = {}
    id_max = []
    id_min = []
    print(dataset + " has " + str(progress) + " graphs.")
    for graphid1 in unique_graphindicator:
        if graphid1 % (progress / 100) == 0:
            print(str(graphid1) + "/" + str(progress) + " completed")
        graphid_loc1 = [index for index, element in enumerate(graphindicator_aslist) if
                        element == graphid1]  # list the index of the graphid locations
        edges_loc1 = edges_asdf[edges_asdf.index.isin(graphid_loc1)]  # obtain edges that corresponds to these locations
        a_graph1 = Graph.TupleList(edges_loc1.itertuples(index=False), directed=False, weights=True)
        activation_values = np.asarray(a_graph1.degree())  # obtain node degrees
        #activation_values = [int(i) for i in np.asarray((a_graph1.betweenness()))] #obtain betweenness
        id_max.append(max(activation_values))
        id_min.append(min(activation_values))
        for i in activation_values:
            total_degree[i] = total_degree.get(i, 0) + 1

    plt.bar(total_degree.keys(), total_degree.values(), 1, color='b')
    plt.xticks(np.arange(min(id_min), max(id_max) + 1))
    plt.xlabel('Degrees')
    plt.ylabel('Fraction of nodes')  # obtained by dividing the node count of the filtration by the data node count
    plt.title(dataset)
    # plt.show()
    # plt.yscale("log")
    plt.savefig("../results/" + dataset + "DegreeStats.png")
    print(dataset + " degree computations are completed.")
    max_activation = max(id_max)
    #max_activation=np.percentile(id_max, 90)# new line for using betweenness
    min_activation = min(id_min)
    if h_filt:
        max_activation = int(max_activation / 2)
    print(dataset + " filtration will run from " + str(min_activation) + " to " + str(max_activation))
    diag_matrix = []
    for graphid in unique_graphindicator:
        if graphid % (progress / 10) == 0:
            print(str(graphid) + "/" + str(progress) + " graphs completed")

        graphid_loc = [index for index, element in enumerate(graphindicator_aslist) if
                       element == graphid]  # list the index of the graphid locations
        edges_loc = edges_asdf[edges_asdf.index.isin(graphid_loc)]  # obtain edges that corresponds to these locations
        nodedict_loc = dict([random_dict[pos] for pos in graphid_loc])

        a_graph = Graph.TupleList(edges_loc.itertuples(index=False), directed=False, weights=True)
        activation_values = np.asarray(a_graph.degree())
        activation_values =[int(i) for i in np.asarray((a_graph.betweenness()))]
        wl_data = [[] for j in range(max_activation - min_activation + 1)]

        for deg in np.arange(min_activation, max_activation + 1):
            deg_loc = (np.where(activation_values <= deg))[0]  # obtain indices where a degree is the maxdegree
            sub_graph = a_graph.subgraph(deg_loc)  # construct subgraphs from original graph using the indices
            subname = sub_graph.vs["name"]  # the subgraph vertex names
            subdict = [(k, v) for k, v in nodedict_loc.items() for k in
                       subname]  # list nodes and nodelabels as dict
            subedges = sub_graph.get_edgelist()  # obtain subgraph edges
            if subedges != []:
                subname_ids = set([x for y in subedges for x in y])  # obtain unique node indices for subgraph
                # subnodes = [subname[pos] for pos in subname_ids]  # corresponding vertex names
                dict_node = dict([subdict[u] for u in subname_ids])  # obtain corresponding dictionaries of indices
                index_dict = dict(
                    zip(subname_ids, list(dict_node.values())))  # replace the dict keys with node indices
                nodes_concat = [subedges, index_dict]
                wl_data[deg - min_activation].extend(nodes_concat)

        for e in wl_data:
            if e == []:
                e.extend([[(-1, -1)], {-1: -1}])
            #  e.extend([[(0, 0)], {0: 0}])

        wl = WeisfeilerLehman(n_iter=iter, base_graph_kernel=VertexHistogram, normalize=True)
        wl_transform = wl.fit_transform(wl_data)
        upper_diag = wl_transform[np.triu_indices(len(wl_transform), k=1)]
        diag_matrix.append(upper_diag)
    RFC_input = pd.DataFrame(diag_matrix)
    t2 = time()
    time_taken = t2 - start
    random.seed(42)
    G_train, G_test, y_train, y_test = train_test_split(RFC_input, graphlabels_aslist, test_size=0.2,
                                                        random_state=42)

    # hyperparameter tuning
    max_features = ['auto', 'sqrt']
    n_estimators = [int(a) for a in np.linspace(start=200, stop=300, num=2)]
    max_depth = [int(b) for b in np.linspace(start=4, stop=6, num=3)]
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    num_cv = 10
    gridlength = len(n_estimators) * len(max_depth) * len(min_samples_leaf) * len(min_samples_split) * num_cv
    print(str(gridlength) + " RFs will be created in the grid search.")
    Param_Grid = dict(max_features=max_features, n_estimators=n_estimators, max_depth=max_depth,
                      min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, bootstrap=bootstrap)
    # Param_Grid= dict(max_features = max_features, n_estimators = n_estimators)
    print(dataset + " training started at", datetime.now().strftime("%H:%M:%S"))
    RFC = RandomForestClassifier(n_jobs=10)
    grid = GridSearchCV(estimator=RFC, param_grid=Param_Grid, cv=num_cv, n_jobs=10)
    grid.fit(G_train, y_train)
    param_choose = grid.best_params_

    RFC_pred = RandomForestClassifier(**param_choose, random_state=1, verbose=1).fit(G_train, y_train)
    test_pred = RFC_pred.predict(G_test)
    auc = roc_auc_score(y_test, RFC_pred.predict_proba(G_test)[:, 1])
    accuracy = accuracy_score(y_test, test_pred)
    print(dataset + " accuracy is " + str(accuracy) + ", AUC is " + str(auc))
    t3 = time()
    print(f'Kernels took {time_taken} seconds, training took {t3 - t2} seconds')
    file.write(dataset + "\t" + str(time_taken) +
               "\t" + str(accuracy) + "\t" + str(auc) +
               "\t" + str(iter) + "\t" + str(h_filt))
    file.flush()


if __name__ == '__main__':
    datasets = ('PROTEINS', 'BZR', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K',
                'ENZYMES', 'FIRSTMM_DB', 'COX2', 'DHFR')
    outputFile = "../results/" + 'kernelTDAResults.txt'
    shutil.copy(outputFile, '../results/latestresultbackupkernelTDA.txt')
    output_file = open(outputFile, 'w')

    datapath = "C:/data"  # dataset path on computer
    for dataset_name in datasets:
        standardGraphFile(dataset_name, output_file, datapath, h_filt=True, iter=5)
        standardGraphFile(dataset_name, output_file, datapath, h_filt=False, iter=5)
    output_file.close()

# TODO:
# use reddit-12k, reddit-5k dataset

