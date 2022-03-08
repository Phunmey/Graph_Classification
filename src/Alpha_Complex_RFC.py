import gudhi as gd
import numpy as np
import pandas as pd
from igraph import *
from numpy import inf
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import MDS
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def standardGraphFile(dataset):
    datapath = "../data"
    edgedata = pd.read_csv(datapath + "/" + dataset + "/" + dataset + "_A.txt", header=None)
    edgedata.columns = ['from', 'to']
    graphlabels = pd.read_csv(datapath + "/" + dataset + "/" + dataset + "_graph_indicator.txt",
                              header=None)
    edgelabels = pd.read_csv(datapath + "/" + dataset + "/" + dataset + "_graph_labels.txt", header=None)
    grapher = sum(graphlabels.values.tolist(), [])
    data = list(set(grapher))  # counting unique graph ids
   #data = shuffle(graph1)

    Accuracy_results = []
    for num in np.arange(0, 10):
        Training_set, Test_set = train_test_split(data, train_size=0.8, test_size=0.2)

        Train_Bet = []
        for i in Training_set:
            graphId = i
            graphNodes = graphlabels[graphlabels.iloc[:, 0] == graphId].index.tolist()
            graphEdges = edgedata[edgedata['from'].isin(graphNodes)]
            graph = Graph.TupleList(graphEdges.itertuples(index=False), directed=True, weights=True)
            # plot(graph, target=None, bbox=(-20, -20, 800, 800)) #show the graph
            distmat = np.asarray(Graph.shortest_paths_dijkstra(graph))
            distmat[distmat == inf] = 0#distancematrix
            symmetric_distmat = np.matmul(distmat, distmat.T)
            [mi, ma] = [np.nanmin(symmetric_distmat), np.nanmax(symmetric_distmat[symmetric_distmat != np.inf])]
            normdistmat = symmetric_distmat / ma
            [m, M] = [np.nanmin(normdistmat), np.nanmax(normdistmat)]
            embedding = MDS(n_components=3, dissimilarity='precomputed')
            normdistmat_transformed = embedding.fit_transform(normdistmat)
            #gg = normdistmat_transformed.shape
           # fig = plt.figure()
          #  plt.scatter(normdistmat_transformed[:,0], normdistmat_transformed[:,1], label = 'MDS')
            Alpha_complex = gd.AlphaComplex(points = normdistmat_transformed)
            simplex_tree = Alpha_complex.create_simplex_tree()
            diagrams = np.asarray(simplex_tree.persistence(), dtype= 'object')
           # gd.plot_persistence_diagram(diagrams)
           # plt.show()

            # splitting the dimension into 0, 1 and 2
            H_0 = diagrams[diagrams[:, 0] == 0, :]
            H_1 = diagrams[diagrams[:, 0] == 1, :]
            H_2 = diagrams[diagrams[:, 0] == 2, :]

            # obtain betti numbers for the unique dimensions
            step = 0.1
            eps = np.arange(0, M + step, step)
            BB_0 = [];
            BB_1 = []
            BB_2 = []
            for j in eps:
                B_0 = 0
                for k in H_0[:, 1]:
                    if k[0] <= j and k[1] > j:
                        B_0 = B_0 + 1
                BB_0.append(B_0)

                B_1 = 0
                for l in H_1[:, 1]:
                    if l[0] <= j and l[1] > j:
                        B_1 = B_1 + 1
                BB_1.append(B_1)

                B_2 = 0
                for x in H_2[:, 1]:
                    if x[0] <= j and x[1] > j:
                        B_2 = B_2 + 1
                BB_2.append(B_2)

            # print([len(BB_0), len(BB_1), len(eps)])
            Betti_numbers = np.array(BB_0 + BB_1 + BB_2)  # concatenate betti numbers
            Betti_graphid = [Betti_numbers, i]  # save Betti numbers with the graphid

            graphidlocation = data.index(i)  # obtain the locations of the graphid
            graphlabelslocation = edgelabels.values[graphidlocation]  # extract the corresponding graphlabels of the graphid
            Betti_graphlabels = np.concatenate((Betti_numbers, graphlabelslocation))  # save betti numbers with graph labels
            Train_Bet.append(Betti_graphlabels)

        Test_Bet = []
        for w in Test_set:
            testgraphId = w
            testgraphNodes = graphlabels[graphlabels.iloc[:, 0] == testgraphId].index.tolist()
            testgraphEdges = edgedata[edgedata['from'].isin(testgraphNodes)]
            testgraph = Graph.TupleList(testgraphEdges.itertuples(index=False), directed=True, weights=True)
            testdistmat = np.asarray(Graph.shortest_paths_dijkstra(testgraph))
            testdistmat[testdistmat == inf] = 0  # distancematrix
            symmetric_testdistmat = np.matmul(testdistmat, testdistmat.T)
            [ni, na] = [np.nanmin(symmetric_testdistmat), np.nanmax(symmetric_testdistmat[symmetric_testdistmat != np.inf])]
            norm_testdistmat = symmetric_testdistmat / na
            [n, N] = [np.nanmin(norm_testdistmat), np.nanmax(norm_testdistmat)]
            test_embedding = MDS(n_components=3, dissimilarity='precomputed')
            norm_testdistmattransformed = embedding.fit_transform(norm_testdistmat)
            # gg = normtestdistmat_transformed.shape
            # fig = plt.figure()
            #  plt.scatter(nomr_testdistmattransformed[:,0], norm_testdistmattransformed[:,1], label = 'MDS')
            Alphatest_complex = gd.AlphaComplex(points=norm_testdistmattransformed)
            simplex_testtree = Alphatest_complex.create_simplex_tree()
            testdiagrams = np.asarray(simplex_testtree.persistence(), dtype= 'object')
           # gd.plot_persistence_diagram(testdiagrams)

            # splitting the dimension into 0, 1 and 2
            Htest_0 = testdiagrams[testdiagrams[:, 0] == 0, :]
            Htest_1 = testdiagrams[testdiagrams[:, 0] == 1, :]
            Htest_2 = testdiagrams[testdiagrams[:, 0] == 2, :]

            bb_0 = []
            bb_1 = []
            bb_2 = []
            for q in eps:
                b_0 = 0
                for h in Htest_0[:, 1]:
                    if h[0] <= q and h[1] > q:
                        b_0 = b_0 + 1
                bb_0.append(b_0)

                b_1 = 0
                for y in Htest_1[:, 1]:
                    if y[0] <= q and y[1] > q:
                        b_1 = b_1 + 1
                bb_1.append(b_1)

                b_2 = 0
                for e in Htest_2[:, 1]:
                    if e[0] <= q and e[1] > q:
                        b_2 = b_2 + 1
                bb_2.append(b_2)

            Betti_numbers_test = np.array(bb_0 + bb_1 + bb_2)  # concatenate betti numbers
            Betti_graphid_test = [Betti_numbers_test, w]  # save Betti numbers with the graphid

            graphidlocation_test = data.index(w)  # obtain the locations of the graphid
            graphlabelslocation_test = edgelabels.values[graphidlocation_test]  # extract the corresponding graphlabels of the graphid
            Betti_graphlabels_test = np.concatenate((Betti_numbers_test, graphlabelslocation_test))  # save betti numbers with graph labels
            Test_Bet.append(Betti_graphlabels_test)

        Train_Data = np.array(Train_Bet)
        Test_Data = np.array(Test_Bet)
        Train_features = Train_Data[:, 0:len(Betti_numbers)]
        Train_labels = Train_Data[:, len(Betti_numbers)]
        Test_features = Test_Data[:, 0:len(Betti_numbers_test)]
        Test_labels = Test_Data[:, len(Betti_numbers_test)]

        RFC = RandomForestClassifier(n_estimators=100, random_state=42)
        RFC.fit(Train_features, Train_labels)
        Test_pred = RFC.predict(Test_features)

        #print(confusion_matrix(Test_labels, Test_pred))
       # print(classification_report(Test_labels, Test_pred))
        #print(accuracy_score(Test_labels, Test_pred))
        Accurate = accuracy_score(Test_labels, Test_pred)
        Accuracy_results.append(Accurate)
    #print(Accuracy_results)
    print('Average accuracy score is', np.mean(Accuracy_results))


if __name__ == '__main__':
    dataset = 'ENZYMES'
    standardGraphFile(dataset)
