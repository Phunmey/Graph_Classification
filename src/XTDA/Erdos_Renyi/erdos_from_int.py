import math
import matplotlib.pyplot as plt
import networkx as nx
import random
from networkx.algorithms import average_clustering
from networkx.generators.classic import empty_graph


# -------------------------------------------------------------------------
#  Erdos Renyi Random Graph
# -------------------------------------------------------------------------
def fast_gnp_random_graph(n, p, seed=None):
    G = empty_graph(n)
    G.name = "fast_gnp_random_graph(%s,%s)" % (n, p)

    if not seed is None:
        random.seed(seed)

    v = 1  # Nodes in graph are from 0,n-1 (this is the second node index).
    w = -1
    lp = math.log(1.0 - p)

    while v < n:
        lr = math.log(1.0 - random.random())
        w = w + 1 + int(lr / lp)
        while w >= v and v < n:
            w = w - v
            v = v + 1
        if v < n:
            G.add_edge(v, w)
    return G


# accepting input from the user
nodes = int(input("please input the number of the nodes   "))
probability = float(input("please input the number of the probability   "))
# drawing the graph of erdos
Erdős = fast_gnp_random_graph(nodes, probability)
t = nx.info(Erdős)
# showing number of nodes, edges and Average degree of erdos graph
print("Erdos Graph info: ", t)

# checking if the graph is directed or undirected
print("is erdős graph directed?", (nx.is_directed(Erdős)))

# diameter(G, e=None)
# degree_centrality(G)
# For average degree: #_edges * 2 / #_nodes;
# print(average_clustering(Erdős))

# compute avarage clustering for the graphs
print("Average clustering of Erdos graph is:  ", average_clustering(Erdős))

print("Transitivity of Erdos is: ",nx.transitivity(Erdős))

# compute clusterings of the graphs
# print("clustering of Erdos graph is  ", clustering(Erdős))

# compute Degree_centrality for nodes
# print("Degree_centrality of Erdos graph is:  ", de50gree_centrality(Erdős))


# compute Diameter of the Graphs
# print("Diameter of Erdos graph is:  ", diameter(Erdős))
# print ("diameter of erdos is ",nx.diameter(Erdős))

# Drowing Erdos grpah acording to the users input
nx.draw(Erdős, with_labels=True)
plt.show()


