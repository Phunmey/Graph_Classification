import numpy as np
from ripser import ripser
from persim import plot_diagrams
import pandas as pd
from pandas import DataFrame
from igraph import *
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
#import gudhi as gd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.manifold import MDS, TSNE
from numpy import inf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import math
from plotly import graph_objs as go
from time import time
#from grakel.datasets import fetch_dataset
from grakel.kernels import WeisfeilerLehman, VertexHistogram


def standardGraphFile(dataset):
    start = time()
    datapath = "C:/Users/Mary/Documents/TDA_codes"
    edges_asdf = pd.read_csv(datapath + "/" + dataset + "/" + dataset + "_A.txt", header=None)
    edges_asdf.columns = ['from', 'to']
    graphindicator_asdf = pd.read_csv(datapath + "/" + dataset + "/" + dataset + "_graph_indicator.txt",
                              header=None)
    graphlabels_asdf = pd.read_csv(datapath + "/" + dataset + "/" + dataset + "_graph_labels.txt", header=None)
    nodelabels_asdf = pd.read_csv(datapath + "/" + dataset + "/" + dataset + "_node_labels.txt", header=None)
    graphindicator_aslist = sum(graphindicator_asdf.values.tolist(), [])
    unique_graphindicator= list(set(graphindicator_aslist))

    DATA = []
    for i in unique_graphindicator:
        graphid = i
        graphid_loc = graphindicator_asdf[graphindicator_asdf.iloc[:, 0] == graphid].index.tolist()
        edges_loc = edges_asdf[edges_asdf.index.isin(graphid_loc)]
        edges_loc_asset = set((edges_loc.to_records(index=False)).tolist())
        nodes_asdf = (edges_loc['from']).to_frame()
        nodes_loc = (nodelabels_asdf[nodelabels_asdf.index.isin(graphid_loc)])
        nodes_asdict = dict((pd.concat([nodes_asdf, nodes_loc], axis=1)).values.tolist())
        #empty_dict = dict()
        edges_nodes = [edges_loc_asset, nodes_asdict]
        DATA.append(edges_nodes)

    labels = np.asarray(sum(graphlabels_asdf.values.tolist(),[]))
    G_train, G_test, y_train, y_test = train_test_split(DATA, labels, test_size=0.2, random_state=42)

    gk = WeisfeilerLehman(n_iter=4, base_graph_kernel=VertexHistogram, normalize=True)
    K_train = gk.fit_transform(G_train)
    K_test = gk.transform(G_test)

    RFC = RandomForestClassifier()
    grid = GridSearchCV(estimator=RFC, param_grid=Param_Grid, cv=2, n_jobs=1)
   # grid = RandomizedSearchCV(estimator=RFC, param_distributions=Param_Grid, cv=2, n_jobs=4)
    grid.fit(Train_features, Train_labels)
    param_choose = grid.best_params_
   # print(param_choose)


    RFC_pred = RandomForestClassifier(**param_choose, random_state=1).fit(Train_features, Train_labels)
    Test_pred = RFC_pred.predict(Test_features)

    print(accuracy_score(Test_labels, Test_pred))
    print(f'Time taken to run:{time() - start} seconds')


if __name__ == '__main__':
    dataset = 'PROTEINS'
    standardGraphFile(dataset)

    #TO DO:
    #GridsearchCV documentation and output tab separated files for Emma
    #Steven: Cross validation presentation on Monday
    #Poupak will present on feature scaling
    #We need a config file for the project