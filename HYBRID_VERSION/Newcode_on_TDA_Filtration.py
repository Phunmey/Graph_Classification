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


def standardGraphFile(dataset, file, datapath):

    start = time()
    df_edges = pd.read_csv(datapath + "/" + dataset + "/" + dataset + "_A.txt", header=None)  # import edge data
    df_edges.columns = ['from', 'to']  # import the graphindicators#import graphlabels  # counting unique graph ids
    unique_nodes = ((df_edges['from'].append(df_edges['to'])).unique()).tolist()
    node_list = np.arange(min(unique_nodes), max(unique_nodes) + 1)  #list of nodes
    csv = pd.read_csv(datapath + "/" + dataset + "/" + dataset + "_graph_indicator.txt", header=None)
    csv.columns = ["ID"]
    graph_indicators = (csv["ID"].values.astype(int))
    read_csv = pd.read_csv(datapath + "/" + dataset + "/" + dataset + "_graph_labels.txt", header=None)
    read_csv.columns = ["ID"]
    graph_labels = (read_csv["ID"].values.astype(int))
    random_nodelabels = [5] * len(node_list)
    random_dict = list(dict(zip(node_list, random_nodelabels)).items())
    unique_graphindicator = np.arange(min(graph_indicators), max(graph_indicators) + 1)  # list unique graphids 100

    random.seed(27)

    id_max = []
    id_min = []
    total_degree = []
    progress = len(unique_graphindicator)
    for graphid1 in unique_graphindicator:
        if graphid1 % (progress / 10) == 0:
            print(str(graphid1) + "/" + str(progress) + " completed")
        graphid_loc1 = [index for index, element in enumerate(graph_indicators) if
                        element == graphid1]  # list the index of the graphid locations
        edges_loc1 = df_edges[df_edges.index.isin(graphid_loc1)]  # obtain edges that corresponds to these locations
        a_graph1 = Graph.TupleList(edges_loc1.itertuples(index=False), directed=False, weights=True)
        deg_calc1 = np.asarray(a_graph1.degree())  # obtain node degrees
        id_max.append(max(deg_calc1))
        id_min.append(min(deg_calc1))
        total_degree.extend(deg_calc1)

    x_axis = sorted(set(total_degree))
    y_axis = [list(total_degree).count(v) / float(len(total_degree)) for v in
              x_axis]  # count each node and divide by the count of all nodes
    plt.bar(x_axis, y_axis)
    plt.xticks(np.arange(min(id_min), max(id_max) + 1))
    plt.xlabel('Degrees')
    plt.ylabel('Fraction of nodes')  # obtained by dividing the node count of the filtration by the data node count
    plt.title(dataset)
    # plt.show()
    # plt.yscale("log")
    plt.savefig("../results/" + dataset + "DegreeStats.png")
    print(dataset + " degree computations are completed.")
    max_degree = max(id_max)
    min_degree = min(id_min)
    print(dataset + " filtration will run from " + str(min_degree) + " to " + str(max_degree))
    diag_matrix = []
    for graphid in unique_graphindicator:
        if graphid % (progress / 10) == 0:
            print(str(graphid) + "/" + str(progress) + " completed")

        graphid_loc = [index for index, element in enumerate(graph_indicators) if
                       element == graphid]  # list the index of the graphid locations
        edges_loc = df_edges[df_edges.index.isin(graphid_loc)]  # obtain edges that corresponds to these locations
        nodedict_loc = dict([random_dict[pos] for pos in graphid_loc])
      #  edges_loc_asset = (edges_loc.to_records(index=False)).tolist()  # convert edges to a set
       # nodes_aslist = (
        #    (edges_loc['from'].append(edges_loc['to'])).unique()).tolist()  # list nodes that makeup the edges
        a_graph = Graph.TupleList(edges_loc.itertuples(index=False), directed=False, weights=True)
        deg_calc = np.asarray(a_graph.degree())

        wl_data = [[] for j in range(max_degree - min_degree + 1)]

        for deg in np.arange(min_degree, max_degree + 1):
            if deg in deg_calc:
                deg_loc = (np.where(deg_calc <= deg))[0]  # obtain indices where a degree is the maxdegree
                sub_graph = a_graph.subgraph(deg_loc)  # construct subgraphs from original graph using the indices
                subname = sub_graph.vs["name"]  # the subgraph vertex names
                subdict = [(k, v) for k, v in nodedict_loc.items() for k in subname]  # list nodes and nodelabels as dict
                subedges = sub_graph.get_edgelist()  # obtain subgraph edges
                if subedges != []:
                    subname_ids = set([x for y in subedges for x in y])  # obtain unique node indices for subgraph
                 #   subnodes = [subname[pos] for pos in subname_ids]  # corresponding vertex names
                    dict_node = dict([subdict[u] for u in subname_ids])  # obtain corresponding dictionaries of indices
                    index_dict = dict(zip(subname_ids, list(dict_node.values())))  # replace the dict keys with node indices
                    nodes_concat = [subedges, index_dict]
                    wl_data[deg - min_degree].extend(nodes_concat)

        for e in wl_data:
            if e == []:
                e.extend([[(-1, -1)], {-1: -1}])
            #  e.extend([[(0, 0)], {0: 0}])

        wl = WeisfeilerLehman(n_iter=5, base_graph_kernel=VertexHistogram, normalize=True)
        wl_transform = wl.fit_transform(wl_data)
        upper_diag = wl_transform[np.triu_indices(len(wl_transform), k=1)]
        diag_matrix.append(upper_diag)
    RFC_input = pd.DataFrame(diag_matrix)
    time_taken = time() - start

    random.seed(27)
    g_train, g_test, y_train, y_test = train_test_split(RFC_input, graph_labels, test_size=0.2,
                                                        random_state=42)

    # hyperparameter tuning
    max_features = ['auto', 'sqrt']
    n_estimators = [int(a) for a in np.linspace(start=300, stop=500, num=100)]
    max_depth = [int(b) for b in np.linspace(start=2, stop=10, num=5)]
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    Param_Grid = dict(max_features=max_features, n_estimators=n_estimators, max_depth=max_depth,
                      min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, bootstrap=bootstrap)

    RFC = RandomForestClassifier()
    grid = GridSearchCV(estimator=RFC, param_grid=Param_Grid, cv=10, n_jobs=10)
    grid.fit(g_train, y_train)
    param_choose = grid.best_params_

    rfc_pred = RandomForestClassifier(**param_choose, random_state=42).fit(g_train, y_train)
    test_pred = rfc_pred.predict(g_test)
    predc = roc_auc_score(y_test, rfc_pred.predict_proba(g_test)[:, 1])
    score = accuracy_score(y_test, test_pred)
    print(dataset + " accuracy is " + str(score))
    print(f'Time taken to run:{time_taken} seconds')
    file.write(dataset + "\t" + str(time_taken) + "\t" + str(score) + "\t" + str(predc) + "\r\n")



if __name__ == '__main__':
    datasets = ('BZR','PROTEINS','ENZYMES','FIRSTMM_DB','COX2','DHFR', 'MUTAG', 'REDDIT-MULTI-5K','REDDIT-MULTI-12K')
    outputFile = "../results/" + 'kernelTDAResults.txt'
    shutil.copy(outputFile, '../results/latestresultbackupkernelTDA.txt')
    file = open(outputFile, 'w')
   # Append 'hello' at the end of file
    datapath = "C:/Users/Mary/Documents/PhD/DATASET"  # dataset path on computer
    for dataset in datasets:
        standardGraphFile(dataset, file, datapath)
    file.close()

