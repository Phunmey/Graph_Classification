import gudhi as gd
import numpy as np
import pandas as pd
import random
from igraph import *
from sklearn.model_selection import train_test_split
from time import time

random.seed(42)

def read_csv(dataset):
    start1 = time()

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

    X_train, X_test, y_train, y_test = train_test_split(unique_graph_indicator, graph_labels, test_size=0.2, random_state=100)

    t1 = time()
    read_split_time = t1-start1

    return X_train, X_test, y_train, y_test, graph_indicators, df_edges, graph_labels

def ripser_train(X_train, X_test, thresh, graph_indicators, df_edges):  #this is for the train test
    start2 = time()
    train_betti =[]
    for i in X_train:
        graph_id = i
        id_location = [index for index, element in enumerate(graph_indicators) if element == graph_id] #list the index of the graph_id locations
        graph_edges = (df_edges[df_edges['from'].isin(id_location)]).values.tolist()
        #create_traingraph = Graph.TupleList(graph_edges.itertuples(index=False), directed=False, weights=False)
        create_sim = gd.RipsComplex(graph_edges, max_edge_length=thresh)
        simplex_tree = create_sim.create_simplex_tree(max_dimension=1)
        get_filt = tuple(simplex_tree.get_filtration())

    return graph_edges

def main():
    X_train, X_test, y_train, y_test, graph_indicators, df_edges, graph_labels = read_csv(dataset)
    graph_edges = ripser_train(X_train, X_test, thresh, graph_indicators, df_edges)


if __name__ == '__main__':
    data_path = sys.argv[1] #dataset path on computer
    data_list = ('MUTAG', 'ENZYMES')
    outputFile = "../../../results/" + 'Ripsimplex.csv'
    file = open(outputFile, 'w')
    for dataset in data_list:
        for thresh in [1]:
            for duplication in np.arange(5):
                main()
    file.close()
