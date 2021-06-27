import numpy as np
from ripser import ripser
from persim import plot_diagrams
import pandas as pd
from pandas import DataFrame
from igraph import *
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import gudhi as gd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.manifold import MDS, TSNE
from numpy import inf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import math
from plotly import graph_objs as go
from time import time


# prepare file for printing
outputFile = open('Ripser_RFC_w_dim_scaling_output.txt', 'w') # can rename


def standardGraphFile(dataset):
    start = time()
    datapath = "C:/Users/Mary/Documents/TDA_codes"
    edgedata = pd.read_csv(datapath + "/" + dataset + "/" + dataset + "_A.txt", header=None)
    edgedata.columns = ['from', 'to']
    graphlabels = pd.read_csv(datapath + "/" + dataset + "/" + dataset + "_graph_indicator.txt",
                              header=None)
    edgelabels = pd.read_csv(datapath + "/" + dataset + "/" + dataset + "_graph_labels.txt", header=None)
    grapher = sum(graphlabels.values.tolist(), [])
    data = list(set(grapher))  # counting unique graph ids
   #data = shuffle(graph1)


    Training_set, Test_set = train_test_split(data, train_size=0.8, test_size=0.2)

    Train_Bet = []
    for i in Training_set:
        graphId = i
        graphNodes = graphlabels[graphlabels.iloc[:, 0] == graphId].index.tolist()
        graphEdges = edgedata[edgedata.index.isin(graphNodes)]
        graph = Graph.TupleList(graphEdges.itertuples(index=False), directed=False, weights=True)
        distmat = np.asarray(Graph.shortest_paths_dijkstra(graph))
      #  fig, ax = plt.subplots()
       # plot(graph, target = ax)
       # plt.show()
       # [mg, Mg] = [np.nanmin(distmat), np.nanmax(distmat[distmat != np.inf])]
        distmat[distmat == inf] = 0
       # mds_embedding = MDS(n_components=3, dissimilarity='precomputed').fit_transform(distmat)
       # tsne_embedding = TSNE(n_components=3).fit_transform(distmat)
       # pca_embedding = PCA(n_components=3).fit_transform(distmat)
      #  [mi, ma] = [np.nanmin(tsne_embedding), np.nanmax(tsne_embedding)]
      #  norm_embedding = tsne_embedding/ma
        [mi, ma] = [np.nanmin(distmat), np.nanmax(distmat)]
        norm_distmat = distmat / ma
        [m, M] = [np.nanmin(norm_distmat), np.nanmax(norm_distmat)]
        diagrams = ripser(norm_distmat, thresh=0.5, maxdim=2, distance_matrix=True)['dgms']

        # splitting the dimension into 0, 1 and 2
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

        Betti_numbers = BB_0 + BB_1 + BB_2 # concatenate betti numbers
       # Betti_graphid = [Betti_numbers, i]  # save Betti numbers with the graphid

        graphidlocation = data.index(i)  # obtain the locations of the graphid
        graphlabelslocation = (edgelabels.values[graphidlocation]).tolist()  # extract the corresponding graphlabels of the graphid
        Betti_graphlabels = Betti_numbers + graphlabelslocation  # save betti numbers with graph labels
        Train_Bet.append(Betti_graphlabels)

    Test_Bet = []
    for w in Test_set:
        testgraphId = w
        testgraphNodes = graphlabels[graphlabels.iloc[:, 0] == testgraphId].index.tolist()
        testgraphEdges = edgedata[edgedata.index.isin(testgraphNodes)]
        testgraph = Graph.TupleList(testgraphEdges.itertuples(index=False), directed=False, weights=True)
        testdistmat = np.asarray(Graph.shortest_paths_dijkstra(testgraph))
      #  [ng, Ng] = [np.nanmin(testdistmat), np.nanmax(testdistmat[testdistmat != np.inf])]
        testdistmat[testdistmat == inf] = 0
       # mdstest_embedding = MDS(n_components=3, dissimilarity='precomputed').fit_transform(testdistmat)
       # tsnetest_embedding = TSNE(n_components=3).fit_transform(testdistmat)
       # pcatest_embedding = PCA(n_components=3).fit_transform(testdistmat)
       # [ni, na] = [np.nanmin(tsnetest_embedding), np.nanmax(tsnetest_embedding)]
       # normtest_embedding = tsnetest_embedding/na
        [ni, na] = [np.nanmin(testdistmat), np.nanmax(testdistmat)]
        normtest_distmat = testdistmat/na
      #  [n, N] = [np.nanmin(normtest_embedding), np.nanmax(normtest_embedding)]
        testdiagrams = ripser(normtest_distmat, thresh=0.5, maxdim=2, distance_matrix=True)['dgms']


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

        Betti_numbers_test = bb_0 + bb_1 + bb_2  # concatenate betti numbers

        graphidlocation_test = data.index(w)  # obtain the locations of the graphid
        graphlabelslocation_test = (edgelabels.values[graphidlocation_test]).tolist()  # extract the corresponding graphlabels of the graphid
        Betti_graphlabels_test = Betti_numbers_test + graphlabelslocation_test  # save betti numbers with graph labels
        Test_Bet.append(Betti_graphlabels_test)

    Train_Data = pd.DataFrame(Train_Bet)
    Test_Data = pd.DataFrame(Test_Bet)

    Train_features = Train_Data.iloc[:, :-1].values
    Train_labels = Train_Data.iloc[:, -1].values
    Test_features = Test_Data.iloc[:, :-1].values
    Test_labels = Test_Data.iloc[:, -1].values

    # RandomForest hyperparameters tuning
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
    grid = GridSearchCV(estimator=RFC, param_grid=Param_Grid, cv=2, n_jobs=1)
   # grid = RandomizedSearchCV(estimator=RFC, param_distributions=Param_Grid, cv=2, n_jobs=4)
    grid.fit(Train_features, Train_labels)
    param_choose = grid.best_params_
   # print(param_choose, file=outputFile)



    RFC_pred = RandomForestClassifier(**param_choose, random_state=1).fit(Train_features, Train_labels)
    Test_pred = RFC_pred.predict(Test_features)

    print(accuracy_score(Test_labels, Test_pred), file=outputFile)
    print(f'Time taken to run:{time() - start} seconds', file=outputFile)

    outputFile.close()  # close output file

if __name__ == '__main__':
    # runs standardGraphFile for all datasets
    sets = ['BZR', 'COX2', 'DHFR', 'ENZYMES', 'FIRSTMM_DB', 'FRANKENSTEIN', 'PROTEINS']
    for dataset in sets:
        print(dataset, file=outputFile)
        standardGraphFile(dataset)