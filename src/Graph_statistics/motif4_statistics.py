import numpy as np
import pandas as pd
import random
from igraph import *


def standardGraphFile(dataset, data_path):
    df_edges = pd.read_csv(data_path + "/" + dataset + "/" + dataset + "_A.txt", header=None)  # import edge data
    df_edges.columns = ['from', 'to']
    unique_nodes = ((df_edges['from'].append(df_edges['to'])).unique()).tolist()
    node_list = np.arange(min(unique_nodes), max(unique_nodes) + 1)  # unique_nodes + missing_nodes
    node_list.sort()
    csv = pd.read_csv(data_path + "/" + dataset + "/" + dataset + "_graph_indicator.txt", header=None)
    csv.columns = ["ID"]
    graph_indicators = (csv["ID"].values.astype(int))
    read_csv = pd.read_csv(data_path + "/" + dataset + "/" + dataset + "_graph_labels.txt", header=None)
    read_csv.columns = ["ID"]
    graph_labels = (read_csv["ID"].values.astype(int))
    unique_graph_indicator = np.arange(min(graph_indicators),
                                       max(graph_indicators) + 1)  # list unique graph ids

    random.seed(42)

    cliques = []
    motifs = []

    for i in unique_graph_indicator:
        train_graph_id = i
        train_id_loc = [index for index, element in enumerate(graph_indicators) if
                        element == train_graph_id]  # list the index of the graphid locations
        train_graph_edges = df_edges[
            df_edges['from'].isin(train_id_loc)]  # obtain the edges with source node as train_graph_id
        train_graph = Graph.TupleList(train_graph_edges.itertuples(index=False), directed=False,
                                      weights=True)  # obtain the graph

        clique_count = train_graph.clique_number()  # obtain clique count
        motifs_count = train_graph.motifs_randesu(size=4)  # obtain motif count

        cliques.append(clique_count)
        motifs.append(motifs_count)

    new_motif = sum(motifs, [])
    mean_count = [np.nanmean(new_motif)]
    std_dev = [np.nanstd(new_motif)]

    df = pd.DataFrame(data=zip(mean_count, std_dev), columns=['Average', 'Standard_deviation'])

    df.insert(0, 'dataset', dataset)

    return (df)

if __name__ == '__main__':
    data_path = sys.argv[1]  # dataset path on computer
    datasets = ('ENZYMES', 'BZR', 'MUTAG', 'PROTEINS', 'DHFR', 'NCI1', 'COX2', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K')
    df1 = []
    for dataset in datasets:
        func = standardGraphFile(dataset, data_path)
        df1.append(func)
    df2 = pd.concat(df1)

    df2.to_csv("C:/Code/src/Graph_statistics/statistics__.csv", index=False)


