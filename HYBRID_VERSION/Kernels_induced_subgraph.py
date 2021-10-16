import random
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from time import time
import matplotlib.pyplot as plt
import pandas as pd
from igraph import *


# UNUSED IMPORT, MAYBE USED LATER
# from tqdm import tqdm


def standard_graph_file(dataset):
    start = time()

    unique_graph_indicator, graph_indicator_list, edges_asdf, random_dict, graph_labels_list = read_data(dataset)

    id_max, id_min, total_degree = get_degree_value(unique_graph_indicator, graph_indicator_list, edges_asdf)

    plot_degree_distribution(total_degree, id_min, id_max)

    max_degree = max(id_max)
    min_degree = min(id_min)

    diag_matrix = []

    for i in unique_graph_indicator:
        graph_id = i
        graph_id_loc = [index for index, element in enumerate(graph_indicator_list) if
                        element == graph_id]  # list the index of the graph_id locations
        edges_loc = edges_asdf[edges_asdf.index.isin(graph_id_loc)]  # obtain edges that corresponds to these locations
        node_dict_loc = dict([random_dict[pos] for pos in graph_id_loc])
        # edges_loc_asset = (edges_loc.to_records(index=False)).tolist()  # convert edges to a set
        # node_list = (
        #     (edges_loc['from'].append(edges_loc['to'])).unique()).tolist()  # list nodes that makeup the edges
        a_graph = Graph.TupleList(edges_loc.itertuples(index=False), directed=False, weights=True)
        deg_calc = np.asarray(a_graph.degree())

        wl_data = [[] for _ in range(max_degree)]

        for deg in np.arange(min_degree, max_degree + 1):
            if deg in deg_calc:
                deg_loc = (np.where(deg_calc <= deg))[0]  # obtain indices where a degree is the max_degree
                sub_graph = a_graph.subgraph(deg_loc)  # construct sub_graphs from original graph using the indices
                sub_name = sub_graph.vs["name"]  # the subgraph vertex names
                sub_dict = [(k, v) for k, v in node_dict_loc.items() for k in
                            sub_name]  # list nodes and node_labels as dict
                sub_edges = sub_graph.get_edgelist()  # obtain subgraph edges

                if len(sub_edges) != 0:
                    sub_name_ids = set([x for y in sub_edges for x in y])  # obtain unique node indices for subgraph
                    # sub_nodes = [sub_name[pos] for pos in sub_name_ids]  # corresponding vertex names
                    dict_node = dict(
                        [sub_dict[u] for u in sub_name_ids])  # obtain corresponding dictionaries of indices
                    index_dict = dict(
                        zip(sub_name_ids, list(dict_node.values())))  # replace the dict keys with node indices
                    nodes_concat = [sub_edges, index_dict]
                    wl_data[deg - min_degree].extend(nodes_concat)

        for e in wl_data:
            if len(e) == 0:
                e.extend([[(-1, -1)], {-1: -1}])
            #  e.extend([[(0, 0)], {0: 0}])

        wl = WeisfeilerLehman(n_iter=5, base_graph_kernel=VertexHistogram, normalize=True)
        wl_transform = wl.fit_transform(wl_data)
        upper_diag = wl_transform[np.triu_indices(len(wl_transform), k=1)]
        diag_matrix.append(upper_diag)
    rfc_input = pd.DataFrame(diag_matrix)

    random.seed(42)
    g_train, g_test, y_train, y_test = train_test_split(rfc_input, graph_labels_list, test_size=0.2,
                                                        random_state=42)

    # hyper-parameter tuning
    max_features = ['auto', 'sqrt']
    n_estimators = [int(a) for a in np.linspace(start=10, stop=100, num=10)]
    max_depth = [int(b) for b in np.linspace(start=2, stop=10, num=5)]
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    param_grid = dict(max_features=max_features, n_estimators=n_estimators, max_depth=max_depth,
                      min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, bootstrap=bootstrap)
    # param_grid= dict(max_features = max_features, n_estimators = n_estimators)

    rfc = RandomForestClassifier()
    grid = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=10, n_jobs=1)
    grid.fit(g_train, y_train)
    param_choose = grid.best_params_

    rfc_pred = RandomForestClassifier(**param_choose, random_state=1).fit(g_train, y_train)
    test_pred = rfc_pred.predict(g_test)

    print(accuracy_score(y_test, test_pred))
    print(f'Time taken to run:{time() - start} seconds')


def read_data(dataset):
    data_path = "/project/6058757/taiwo/dataset"  # dataset path on computer
    edges_asdf = pd.read_csv(data_path + "/" + dataset + "/" + dataset + "_A.txt", header=None)  # import edge data
    edges_asdf.columns = ['from', 'to']  # import the graphindicators#import graphlabels  # counting unique graph ids
    unique_nodes = ((edges_asdf['from'].append(edges_asdf['to'])).unique()).tolist()
    missing_nodes = [x for x in range(unique_nodes[0], unique_nodes[-1] + 1) if
                     x not in unique_nodes]  # find the missing nodes
    node_list = unique_nodes + missing_nodes
    node_list.sort()
    graph_indicator_list = sum((pd.read_csv(data_path + "/" + dataset + "/" + dataset + "_graph_indicator.txt",
                                            header=None).values.tolist()), [])
    graph_labels_list = sum(
        (pd.read_csv(data_path + "/" + dataset + "/" + dataset + "_graph_labels.txt", header=None).values.tolist()), [])
    # node_labels_list = sum(
    # (pd.read_csv(data_path + "/" + dataset + "/" + dataset + "_node_labels.txt", header=None).values.tolist()), [])
    random_node_labels = [5] * len(node_list)
    random_dict = list(dict(zip(node_list, random_node_labels)).items())
    # nodes_dict = list(dict(zip(node_list, node_labels_list)).items())  # makes nodes and their labels as a dict
    unique_graph_indicator = list(set(graph_indicator_list))  # list unique graphids 100

    random.seed(42)

    return unique_graph_indicator, graph_indicator_list, edges_asdf, random_dict, graph_labels_list


def get_degree_value(unique_graph_indicator, graph_indicator_list, edges_asdf):
    id_max = []
    id_min = []
    total_degree = []

    for j in unique_graph_indicator:
        graph_id1 = j
        graph_id_loc1 = [index for index, element in enumerate(graph_indicator_list) if
                         element == graph_id1]  # list the index of the graph_id locations
        edges_loc1 = edges_asdf[
            edges_asdf.index.isin(graph_id_loc1)]  # obtain edges that corresponds to these locations
        a_graph1 = Graph.TupleList(edges_loc1.itertuples(index=False), directed=False, weights=True)
        deg_calc1 = np.asarray(a_graph1.degree())  # obtain node degrees
        id_max.append(max(deg_calc1))
        id_min.append(min(deg_calc1))
        total_degree.extend(deg_calc1)

    return id_max, id_min, total_degree


def plot_degree_distribution(total_degree, id_min, id_max):
    x_axis = sorted(set(total_degree))
    y_axis = [list(total_degree).count(v) / float(len(total_degree)) for v in
              x_axis]  # count each node and divide by the count of all nodes
    plt.bar(x_axis, y_axis)
    plt.xticks(np.arange(min(id_min), max(id_max) + 1))
    plt.xlabel('Degrees')
    plt.ylabel('Fraction of nodes')  # obtained by dividing the node count of the filtration by the data node count
    plt.title('REDDIT-MULTI-5K')
    # plt.yscale("log")
    plt.savefig("../results/Graph_Distribution/REDDIT-MULTI-5K.png")


if __name__ == '__main__':
    data_set = 'REDDIT-MULTI-5K'
    standard_graph_file(data_set)
