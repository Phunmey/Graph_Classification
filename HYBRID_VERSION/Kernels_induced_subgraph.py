import random
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc
from grakel.datasets import fetch_dataset
from grakel.kernels import WeisfeilerLehman, VertexHistogram, WeisfeilerLehmanOptimalAssignment
from time import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
from pandas import DataFrame
import igraph as ig
from igraph import *
from itertools import chain
from operator import itemgetter
from numpy import count_nonzero
from scipy.sparse import csr_matrix
from scipy.optimize import curve_fit
from tqdm import tqdm


def standardGraphFile(dataset):
    start = time()
    datapath = "/project/6058757/taiwo/dataset"  # dataset path on computer
    edges_asdf = pd.read_csv(datapath + "/" + dataset + "/" + dataset + "_A.txt", header=None)  # import edge data
    edges_asdf.columns = ['from', 'to']  # import the graphindicators#import graphlabels  # counting unique graph ids
    unique_nodes = ((edges_asdf['from'].append(edges_asdf['to'])).unique()).tolist()
    missing_nodes = [x for x in range(unique_nodes[0], unique_nodes[-1] + 1) if
                     x not in unique_nodes]  # find the missing nodes
    node_list = unique_nodes + missing_nodes
    node_list.sort()
    graphindicator_aslist = sum((pd.read_csv(datapath + "/" + dataset + "/" + dataset + "_graph_indicator.txt",
                                             header=None).values.tolist()), [])
    graphlabels_aslist = sum(
        (pd.read_csv(datapath + "/" + dataset + "/" + dataset + "_graph_labels.txt", header=None).values.tolist()), [])
   # nodelabels_aslist = sum(
        #(pd.read_csv(datapath + "/" + dataset + "/" + dataset + "_node_labels.txt", header=None).values.tolist()), [])
    random_nodelabels = [5] * len(node_list)
    random_dict = list(dict(zip(node_list, random_nodelabels)).items())
    #nodes_dict = list(dict(zip(node_list, nodelabels_aslist)).items())  # makes nodes and their labels as a dict
    unique_graphindicator = list(set(graphindicator_aslist))  # list unique graphids 100

    random.seed(42)

    id_max = []
    id_min = []
    total_degree = []
    for j in unique_graphindicator:
        graphid1 = j
        graphid_loc1 = [index for index, element in enumerate(graphindicator_aslist) if
                        element == graphid1]  # list the index of the graphid locations
        edges_loc1 = edges_asdf[edges_asdf.index.isin(graphid_loc1)]  # obtain edges that corresponds to these locations
        a_graph1 = Graph.TupleList(edges_loc1.itertuples(index=False), directed=False, weights=True)
        deg_calc1 = np.asarray(a_graph1.degree()) #obtain node degrees
        id_max.append(max(deg_calc1))
        id_min.append(min(deg_calc1))
        total_degree.extend(deg_calc1)

    x_axis = sorted(set(total_degree))
    y_axis = [list(total_degree).count(v) / float(a_graph1.vcount()) for v in x_axis]  # count each node and divide by the count of all nodes
    #plt.plot(deg_value, histogram)
    plt.bar(x_axis, y_axis)
    plt.xticks(np.arange(min(id_min), max(id_max)+1))
    plt.xlabel('Degrees')
    plt.ylabel('Fraction of nodes')  # obtained by dividing the node count of the filtration by the data node count
    plt.title('REDDIT-MULTI-5K')
    #plt.yscale("log")
    plt.savefig("C:/Users/Mary/Documents/TDA_codes/Graph_Distribution/REDDIT-MULTI-5K.png")

    max_degree = max(id_max)
    min_degree = min(id_min)
    diag_matrix = []
    for i in unique_graphindicator:
        graphid = i
        graphid_loc = [index for index, element in enumerate(graphindicator_aslist) if
                        element == graphid]  # list the index of the graphid locations
        edges_loc = edges_asdf[edges_asdf.index.isin(graphid_loc)]  # obtain edges that corresponds to these locations
        nodedict_loc = dict([random_dict[pos] for pos in graphid_loc])
        edges_loc_asset = (edges_loc.to_records(index=False)).tolist()  # convert edges to a set
        nodes_aslist = (
            (edges_loc['from'].append(edges_loc['to'])).unique()).tolist()  # list nodes that makeup the edges
        a_graph = Graph.TupleList(edges_loc.itertuples(index=False), directed=False, weights=True)
        deg_calc = np.asarray(a_graph.degree())

        wl_data = [[] for j in range(max_degree)]
        for deg in np.arange(min_degree, max_degree + 1):
            if deg in deg_calc:
                deg_loc = (np.where(deg_calc <= deg))[0]  # obtain indices where a degree is the maxdegree
                sub_graph = a_graph.subgraph(deg_loc) #construct subgraphs from original graph using the indices
                subname = sub_graph.vs["name"] #the subgraph vertex names
                subdict = [(k,v) for k,v in nodedict_loc.items() for k in subname] #list nodes and nodelabels as dict
                subedges = sub_graph.get_edgelist() #obtain subgraph edges
                if subedges != []:
                    subname_ids = set([x for y in subedges for x in y]) #obtain unique node indices for subgraph
                    subnodes = [subname[pos] for pos in subname_ids] #corresponding vertex names
                    dict_node = dict([subdict[u] for u in subname_ids]) #obtain corresponding dictionaries of indices
                    index_dict = dict(zip(subname_ids, list(dict_node.values()))) #replace the dict keys with node indices
                    nodes_concat = [subedges, index_dict]
                    wl_data[deg - min_degree].extend(nodes_concat)

        for e in wl_data:
            if e == []:
                e.extend([[(-1,-1)], {-1:-1}])
              #  e.extend([[(0, 0)], {0: 0}])

        wl = WeisfeilerLehman(n_iter=5, base_graph_kernel=VertexHistogram, normalize=True)
        wl_transform = wl.fit_transform(wl_data)
        upper_diag = wl_transform[np.triu_indices(len(wl_transform), k=1)]
        diag_matrix.append(upper_diag)
    RFC_input = pd.DataFrame(diag_matrix)

    random.seed(42)
    G_train, G_test, y_train, y_test = train_test_split(RFC_input, graphlabels_aslist, test_size=0.2,
                                                        random_state=42)

    # hyperparameter tuning
    max_features = ['auto', 'sqrt']
    n_estimators = [int(a) for a in np.linspace(start=10, stop=100, num=10)]
    max_depth = [int(b) for b in np.linspace(start=2, stop=10, num=5)]
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    Param_Grid = dict(max_features=max_features, n_estimators=n_estimators, max_depth=max_depth,
                      min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, bootstrap=bootstrap)
    # Param_Grid= dict(max_features = max_features, n_estimators = n_estimators)

    RFC = RandomForestClassifier()
    grid = GridSearchCV(estimator=RFC, param_grid=Param_Grid, cv=10, n_jobs=1)
    grid.fit(G_train, y_train)
    param_choose = grid.best_params_

    RFC_pred = RandomForestClassifier(**param_choose, random_state=1).fit(G_train, y_train)
    Test_pred = RFC_pred.predict(G_test)

    print(accuracy_score(y_test, Test_pred))
    print(f'Time taken to run:{time() - start} seconds')


if __name__ == '__main__':
    dataset = 'REDDIT-MULTI-5K'
    standardGraphFile(dataset)

#TODO:
#use reddit-12k, reddit-5k dataset
#see papers with code to see if there are any weighted graph classification datasets
#use min and max degrees separately on each dataset
#use the induced_subgraph function of igraph to construct your subgraphs
#from the resulting WL kernel matrix of each graph, extract the upper diagonal elements and flatten into a vector
#use the result above diagonal for your classifier
#obtain the maximum degree possible for all the graphs in the dataset (this will serve as the limit for epsilon)

#if max(a_graph_degree) > max_degree:
 #   max_degree = max(a_graph_degree)
#if min(a_graph_degree) < min_degree:
 #   min_degree = min(a_graph_degree)
