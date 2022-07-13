import igraph
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from igraph import *
from persim import plot_diagrams
from ripser import ripser
from sklearn.model_selection import train_test_split


def BZR(dataset):
    edge_path = pd.read_csv(data_path + "/" + dataset + "/" + dataset + "_A.txt", header=None)
    edge_path.columns = ['from', 'to']
    indicator_path = pd.read_csv(data_path + "/" + dataset + "/" + dataset + "_graph_indicator.txt", header=None)
    indicator_path.columns = ["ID"]
    graph_indicators = (indicator_path["ID"].values.astype(int))
    label_path = pd.read_csv(data_path + "/" + dataset + "/" + dataset + "_graph_labels.txt", header=None)
    label_path.columns = ["ID"]
    graph_labels = (label_path["ID"].values.astype(int))

    unique_graph_indicator = np.arange(min(graph_indicators),max(graph_indicators) + 1)  # list unique graph ids

    random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split(unique_graph_indicator, graph_labels, train_size=0.8,
                                                            test_size=0.2)  # split the dataset to train and test sets

    return X_train, X_test, y_train, y_test, graph_indicators, edge_path, graph_labels, unique_graph_indicator

def first_step_0(X_train, graph_indicators, edge_path, dataset):
    dgms=[]
    for i in X_train[0:5]: #random.sample(range(50, 400), 5)
        graph_id = i
        id_location = [index for index, element in enumerate(graph_indicators) if element == graph_id] #list the index of the graph_id locations
        graph_edges = edge_path[edge_path['from'].isin(id_location)]
        create_traingraph = Graph.TupleList(graph_edges.itertuples(index=False), directed=False, weights=True)
        igraph.plot(create_traingraph, ("C:/Code/src/XTDA/graph_and_persistence/graphs/" + dataset + "_" + str(i) + "_graph.png"))
        train_distance_matrix = np.asarray(Graph.shortest_paths_dijkstra(create_traingraph))
        train_normalize = train_distance_matrix / np.nanmax(train_distance_matrix[train_distance_matrix != np.inf])
        train_diagrams = ripser(train_normalize, thresh=1, maxdim=1, distance_matrix=True)[
            'dgms']  # maximum homology dimension computed i.e H_0, H_1 for maxdim=1. thresh is maximum distance considered when constructing filtration
        dgms.append(train_diagrams[0])
        plot_diagrams(train_diagrams, show=False)
        plt.savefig("C:/Code/src/XTDA/graph_and_persistence/persistence/" + dataset + "_" + str(i) + "persistence_plot.png")
        plt.clf()
    print(dataset + "has finished, move to the next")

    # remove_infinity = lambda barcode: np.array([bars for bars in barcode if bars[1] != np.inf])
    # dgms = list(map(remove_infinity, dgms))
    #
    # for j in np.arange(len(dgms)):
    #     if (dgms[j]).size == 0:
    #         dgms[j] = np.array([[0, 0], [0, 0]])
    #
    # ES_train_0 = gd.representations.Entropy().fit_transform(dgms)
    # ES_train_0 = [x for xs in ES_train_0 for x in xs]





def main():
    X_train, X_test, y_train, y_test, graph_indicators, edge_path, graph_labels, unique_graph_indicator = BZR(dataset)
    first_step_0(X_train, graph_indicators, edge_path, dataset)

if __name__ == '__main__':
    data_path = sys.argv[1]  # dataset path on computer
    data_list = ('ENZYMES', 'BZR', 'MUTAG', 'PROTEINS', 'DHFR', 'NCI1', 'COX2', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K')
    for dataset in data_list:
        main()
