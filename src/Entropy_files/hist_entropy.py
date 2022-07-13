import gudhi as gd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import warnings
from igraph import *
from ripser import ripser
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=RuntimeWarning)

random.seed(42)


def read_csv(dataset):
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

    X_train, X_test, y_train, y_test = train_test_split(unique_graph_indicator, graph_labels, test_size=0.2,
                                                        random_state=100)


    return X_train, X_test, y_train, y_test, graph_indicators, df_edges


def train_ES(X_train, graph_indicators, df_edges, y_train):  # this is for the train test
    dgms = []
    for i in X_train:
        graph_id = i
        id_location = [index for index, element in enumerate(graph_indicators) if
                       element == graph_id]  # list the index of the graph_id locations
        graph_edges = df_edges[df_edges['from'].isin(id_location)]
        create_traingraph = Graph.TupleList(graph_edges.itertuples(index=False), directed=False, weights=True)
        train_distance_matrix = np.asarray(Graph.shortest_paths_dijkstra(create_traingraph))
        train_normalize = train_distance_matrix / np.nanmax(train_distance_matrix[train_distance_matrix != np.inf])
        train_diagrams = ripser(train_normalize, thresh= 1, maxdim=1, distance_matrix=True)['dgms']  # maximum homology dimension computed i.e H_0, H_1 for maxdim=1. thresh is maximum distance considered when constructing filtration
        dgms.append(train_diagrams[1])

    remove_infinity = lambda barcode: np.array([bars for bars in barcode if bars[1] != np.inf])
    dgms = list(map(remove_infinity, dgms))

    for j in np.arange(len(dgms)):
        if (dgms[j]).size == 0:
            dgms[j] = np.array([[0,0], [0,0]])

    ES_train = gd.representations.Entropy().fit_transform(dgms)
    ES_train = [x for xs in ES_train for x in xs]

    return ES_train

def second_ES(X_test, graph_indicators, df_edges, ES_train):
    dgms_test = []
    for j in X_test:
        graph_id = j
        id_location = [index for index, element in enumerate(graph_indicators) if
                       element == graph_id]  # list the index of the graph_id locations
        graph_edges = df_edges[df_edges['from'].isin(id_location)]
        create_testgraph = Graph.TupleList(graph_edges.itertuples(index=False), directed=False, weights=True)
        test_distance_matrix = np.asarray(Graph.shortest_paths_dijkstra(create_testgraph))
        test_normalize = test_distance_matrix / np.nanmax(test_distance_matrix[test_distance_matrix != np.inf])
        test_diagrams = ripser(test_normalize, thresh=1, maxdim=1, distance_matrix=True)[
            'dgms']  # using thresh=1 because the largest number in the matrix is 1
        dgms_test.append(test_diagrams[1])

    remove_infinity = lambda barcode: np.array([bars for bars in barcode if bars[1] != np.inf])
    dgms_test = list(map(remove_infinity, dgms_test))

    for j in np.arange(len(dgms_test)):
        if (dgms_test[j]).size == 0:
            dgms_test[j] = np.array([[0,0], [0,0]])

    ES_test = gd.representations.Entropy().fit_transform(dgms_test)
    ES_test = [x for xs in ES_test for x in xs]

    entropy = ES_train + ES_test

    df_entropy = pd.DataFrame(entropy)
    df_entropy.to_csv("C:/Code/src/Entropy_files/" + dataset + "_entropy.csv", index=False)

    return entropy

def entropy_plot(entropy):
    entropy_array = np.array(entropy)[np.isfinite(np.array(entropy))]
   # bins = np.linspace(np.min(entropy), np.max(entropy)+1, num=5)
    plt.hist(entropy_array, ec='black')  #ec means edgecolor
    plt.title(dataset)
    plt.xlabel('entropy_values')
    plt.ylabel('entropy_count')
    plt.legend(loc='upper right')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlim(left=0)
    plt.show()
   # plt.savefig("C:/Code/src/Entropy_files/entropy_figures/" + dataset + "entropy.png")
   # plt.clf()
   # print(dataset + " entropy computations are completed.")

# def result_csv(dataset, entropy):
#     file.write(dataset + "\n" + str(entropy) + "\n")
#
#     file.flush()


def main():
    X_train, X_test, y_train, y_test, graph_indicators, df_edges = read_csv(dataset)
    ES_train = train_ES(X_train, graph_indicators, df_edges, y_train)
    entropy = second_ES(X_test, graph_indicators, df_edges, ES_train)
    entropy_plot(entropy)
   # result_csv(dataset, entropy)


if __name__ == '__main__':
    data_path = sys.argv[1]  # dataset path on computer
    data_list = ('ENZYMES', 'BZR', 'MUTAG', 'PROTEINS', 'DHFR', 'NCI1', 'COX2', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K' )
    # outputFile = "C:/Code/src/Entropy_files/" + 'results.csv'
    # file = open(outputFile, 'w')
    for dataset in data_list:
        main()

   # file.close()


