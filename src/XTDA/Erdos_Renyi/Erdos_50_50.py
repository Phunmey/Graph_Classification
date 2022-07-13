import matplotlib.pyplot as plt
import networkx as nx
import random


def display_graph(G, i, ne): #ne is the list of edges while i is a single node
    pos = nx.circular_layout(G)
    if i == '' and ne == '':
        new_node = []
        rest_nodes = G.nodes()
        new_edges = []
        rest_edges = G.edges()
    elif i == '':
        rest_nodes = G.nodes()
        new_edges = ne
        rest_edges = list(set(G.edges()) - set(new_edges) - set([(b,a) for (a,b) in new_edges]))

    #nx.draw_networkx_nodes(G, pos, nodelist=new_node, node_color='g')
    nx.draw_networkx_nodes(G, pos, nodelist=rest_nodes, node_color='r')
    nx.draw_networkx_edges(G, pos, edgelist=new_edges, edge_color='g', style='-')
    nx.draw_networkx_edges(G, pos, edgelist=rest_edges, edge_color='r')
    plt.show()

def erdos_renyi(G, p):
    for i in G.nodes():
        for j in G.nodes():
            if i != j:
                r = random.random()
                if r <= p:
                    G.add_edge(i,j)
                    ne = [(i,j)]
                    display_graph(G, '', ne)
                else:
                    ne = []
                    display_graph(G, '', ne)
                    continue



def main():
    #Take n, i.e total number of nodes, from the user
    n = int(input('Enter the value of n')) #gives a string, so convert to integer
    #Take p, i.e the value of probability from the user
    p = float(input('Enter the value of p'))
    #create an empty graph. Add nodes to it
    G = nx.Graph()
    G.add_nodes_from([i for i in range(n)])
    #Add edges to he graph randomly
    display_graph(G, '', '')
    erdos_renyi(G, p)

main()