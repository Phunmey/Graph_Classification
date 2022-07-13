import gudhi as gd
import igraph
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from igraph import *
from numpy import inf
from sklearn.decomposition import PCA

# Set a random seed for reproducibility
random.seed(0)

def create_graph():
    dgms = []
   # for i in range(10):
        # Generate Erdos Renyi graphs based on probability in igraph
    G = Graph.Erdos_Renyi(n=30, m=32, directed=False, loops=False)
    igraph.plot(G, "C:/Code/src/XTDA/Toy_example/erdos_graph.png")
   # plt.savefig("C:/Code/src/XTDA/Toy_example/erdos_graph.png")

    edgelist = []
    for i in G.es:
        obtain_edges = i.tuple #extract edges as tuples
        edgelist.append(obtain_edges)
   # print(len(edgelist))
    graph_edges = pd.DataFrame(edgelist)
    create_traingraph = Graph.TupleList(graph_edges.itertuples(index=False), directed=False, weights=True)
    train_distance_matrix = np.asarray(Graph.shortest_paths_dijkstra(create_traingraph))
    train_distance_matrix[train_distance_matrix == inf] = 0
    sym_matrix = np.matmul(train_distance_matrix, train_distance_matrix.T)  # obtain a symmetric matrix
    [min1, max1] = [np.nanmin(sym_matrix), np.nanmax(sym_matrix[sym_matrix != np.inf])]
    norm_dist_matrix = sym_matrix / max1  # normalized distancematrix
    train_dim_reduction = PCA(n_components=3).fit_transform(norm_dist_matrix)
    train_alpha_complex = gd.AlphaComplex(points=train_dim_reduction)
    train_simplex_tree = train_alpha_complex.create_simplex_tree()
    train_diagrams = np.asarray(train_simplex_tree.persistence(), dtype='object')
    # print("betti_numbers()=", train_simplex_tree.betti_numbers())
    gd.plot_persistence_diagram(train_diagrams)
    plt.savefig("C:/Code/src/XTDA/Toy_example/" + "persistence_plot.png")


    # splitting the dimension into 0 and 1
    train_persist_0 = train_diagrams[:, 1][np.where(train_diagrams[:, 0] == 0)]
    train_persist_1 = train_diagrams[:, 1][np.where(train_diagrams[:, 0] == 1)]
    train_0 = np.array([list(x) for x in train_persist_0])
    train_1 = np.array([list(y) for y in train_persist_1])
    merge_array = dgms.append(train_1)
      #  [train_0, train_1]

    remove_infinity = lambda barcode: np.array([bars for bars in barcode if bars[1] != np.inf])
    # apply this operator to all diagrams.
    dgms = list(map(remove_infinity, dgms))
    sample_range = np.linspace(0, 1, 10)

    ES = gd.representations.Entropy().fit_transform(dgms)
    print(ES)

    return


if __name__ == '__main__':
    create_graph()



# G = nx.erdos_renyi_graph(25, 0.75)
# nx.draw(G, with_labels=True)
# plt.show()