import numpy as np
from ripser import ripser
from persim import plot_diagrams
import pandas as pd
from pandas import DataFrame
from igraph import *
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.manifold import MDS, TSNE
from numpy import inf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import math
from time import time
from grakel.kernels import WeisfeilerLehman, VertexHistogram


def standardGraphFile(dataset):
    start = time()
    datapath = "C:/Users/Mary/Documents/TDA_codes"
    edges_asdf = pd.read_csv(datapath + "/" + dataset + "/" + dataset + "_A.txt", header=None)
    edges_asdf.columns = ['from', 'to']
    unique_nodes = ((edges_asdf['from'].append(edges_asdf['to'])).unique()).tolist()
    missing_nodes = [x for x in range(unique_nodes[0], unique_nodes[-1] + 1) if
                     x not in unique_nodes]  # find the missing nodes
    node_list = unique_nodes + missing_nodes
    node_list.sort()
    graphindicator_aslist = sum((pd.read_csv(datapath + "/" + dataset + "/" + dataset + "_graph_indicator.txt",
                                             header=None).values.tolist()), [])
    graphlabels_aslist = sum(
        (pd.read_csv(datapath + "/" + dataset + "/" + dataset + "_graph_labels.txt", header=None).values.tolist()), [])
    nodelabels_aslist = sum(
        (pd.read_csv(datapath + "/" + dataset + "/" + dataset + "_node_labels.txt", header=None).values.tolist()), [])
    nodes_dict = dict(zip(node_list, nodelabels_aslist))
    unique_graphindicator = list(set(graphindicator_aslist))


    DATA = []
    for i in unique_graphindicator:
        graphid = i
        graphid_loc = [index for index, element in enumerate(graphindicator_aslist) if element == graphid]
        edges_loc = edges_asdf[edges_asdf.index.isin(graphid_loc)]
        edges_loc_asset = (edges_loc.to_records(index=False)).tolist()
        nodes_aslist = ((edges_loc['from'].append(edges_loc['to'])).unique()).tolist()
        ext = {k: nodes_dict[k] for k in nodes_aslist if k in nodes_dict}
        edges_nodes = [edges_loc_asset, ext]
        DATA.append(edges_nodes)

    WL = WeisfeilerLehman(n_iter=5, base_graph_kernel=VertexHistogram, verbose=True, normalize=True)
    K_train = WL.fit_transform(DATA)

    G_train, G_test, y_train, y_test = train_test_split(K_train, graphlabels_aslist, test_size=0.2, random_state=42)

    #K_test = WL.transform(G_test)


    [m, M] = [np.nanmin(G_train), np.nanmax(G_train)]
    diagrams = ripser(G_train, thresh=0.5, maxdim=2)['dgms']
   # plot_diagrams(diagrams, title="Persistence Diagrams showing H_0 and H_1", show=True)

    H_0 = diagrams[0]
    H_1 = diagrams[1]
    H_2 = diagrams[2]

    # obtain betti numbers for the unique dimensions
    step = 0.05
    eps = np.arange(0, M + step, step)
    BB_0 = [];
    BB_1 = []
    BB_2 = []
    for j in eps:
        B_0 = 0
        for k in H_0:
            if k[0] <= j and k[1] > j:
                B_0 = B_0 + 1
        BB_0.append(B_0)

        B_1 = 0
        for l in H_1:
            if l[0] <= j and l[1] > j:
                B_1 = B_1 + 1
        BB_1.append(B_1)

        B_2 = 0
        for x in H_2:
            if x[0] <= j and x[1] > j:
                B_2 = B_2 + 1
        BB_2.append(B_2)

    Betti_train = BB_0 + BB_1 + BB_2  # concatenate betti numbers

    [ma, Ma] = [np.nanmin(G_test), np.nanmax(G_test)]
    testdiagrams = ripser(G_test, thresh=0.5, maxdim=2)['dgms']
    #plot_diagrams(testdiagrams, title="Persistence Diagrams showing H_0 and H_1", show=True)

    # splitting the dimension into 0, 1 and 2
    Htest_0 = testdiagrams[0]
    Htest_1 = testdiagrams[1]
    Htest_2 = testdiagrams[2]

    bb_0 = []
    bb_1 = []
    bb_2 = []
    for q in eps:
        b_0 = 0
        for h in Htest_0:
            if h[0] <= q and h[1] > q:
                b_0 = b_0 + 1
        bb_0.append(b_0)

        b_1 = 0
        for y in Htest_1:
            if y[0] <= q and y[1] > q:
                b_1 = b_1 + 1
        bb_1.append(b_1)

        b_2 = 0
        for e in Htest_2:
            if e[0] <= q and e[1] > q:
                b_2 = b_2 + 1
        bb_2.append(b_2)

    Betti_test = bb_0 + bb_1 + bb_2  # concatenate betti numbers

    RFC_pred = RandomForestClassifier().fit(Betti_train, y_train)
    y_pred = RFC_pred.predict(Betti_test)

    print(accuracy_score(y_test, y_pred))
    print(f'Time taken to run:{time() - start} seconds')

I want to merge TDA with Kernels.

if __name__ == '__main__':
    dataset = 'BZR'
    standardGraphFile(dataset)
