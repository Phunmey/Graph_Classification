import gudhi as gd
import numpy as np
import pandas as pd
import random
from datetime import datetime
from igraph import *
from numpy import inf
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
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
                                       max(graph_indicators) + 1)  # list unique graph ids in a dataset

    X_train, X_test, y_train, y_test = train_test_split(unique_graph_indicator, graph_labels, test_size=0.2,
                                                        random_state=42)

    t1 = time()
    read_split_time = t1 - start1

    return X_train, X_test, y_train, y_test, graph_indicators, df_edges, graph_labels


def alpha_train(X_train, thresh, graph_indicators, df_edges, step_size):  # this is for the train data
    start2 = time()
    train_betti = []
    graph_density = []
    graph_diameter = []
    clustering_coeff = []
    spectral_gap = []
    assortativity_ = []
    cliques = []
    motifs = []
    components = []
    for i in X_train:
        graph_id = i
        id_location = [index for index, element in enumerate(graph_indicators) if
                       element == graph_id]  # list the index of the graph_id locations
        graph_edges = df_edges[df_edges['from'].isin(id_location)]
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
        #   plot_diagrams(train_diagrams, title= "train persistence diagrams showing H_0 and H_1",  show=True) #using thresh=1 because the largest number in the matrix is 1

        # splitting the dimension into 0 and 1
        train_persist_0 = train_diagrams[:, 1][np.where(train_diagrams[:, 0] == 0)]
        train_persist_1 = train_diagrams[:, 1][np.where(train_diagrams[:, 0] == 1)]

        # obtain betti numbers for the unique dimensions
        train_betti_0 = []
        train_betti_1 = []

        for eps in np.linspace(0, thresh, step_size):
            b_0 = 0
            for k in train_persist_0:
                if k[0] <= eps and k[1] > eps:
                    b_0 = b_0 + 1
            train_betti_0.append(b_0)

            b_1 = 0
            for l in train_persist_1:
                if l[0] <= eps and l[1] > eps:
                    b_1 = b_1 + 1
            train_betti_1.append(b_1)

        train_betti.append(train_betti_0 + train_betti_1)  # concatenate betti numbers

        Density = create_traingraph.density()  # obtain density
        Diameter = create_traingraph.diameter()  # obtain diameter
        cluster_coeff = create_traingraph.transitivity_avglocal_undirected()  # obtain transitivity
        laplacian = create_traingraph.laplacian()  # obtain laplacian matrix
        laplace_eigenvalue = np.linalg.eig(laplacian)
        sort_eigenvalue = sorted(np.real(laplace_eigenvalue[0]), reverse=True)
        spectral = sort_eigenvalue[0] - sort_eigenvalue[1]  # obtain spectral gap
        assortativity = create_traingraph.assortativity_degree()  # obtain assortativity
        clique_count = create_traingraph.clique_number()  # obtain clique count
        motifs_count = create_traingraph.motifs_randesu(size=3)  # obtain motif count
        count_components = len(create_traingraph.clusters())  # obtain count components

        graph_density.append(Density)
        graph_diameter.append(Diameter)
        clustering_coeff.append(cluster_coeff)
        spectral_gap.append(spectral)
        assortativity_.append(assortativity)
        cliques.append(clique_count)
        motifs.append(motifs_count)
        components.append(count_components)

    df1 = pd.DataFrame(train_betti)
    df2 = pd.DataFrame(motifs)
    df3 = pd.DataFrame(list(zip(graph_density, graph_diameter, clustering_coeff, spectral_gap, assortativity_, cliques, components)))
    train_data = pd.concat([df1, df2, df3], axis=1, ignore_index=True)
    train_data = train_data.fillna(0)

    t2 = time()
    train_time = t2 - start2

    return train_data, train_time


def alpha_test(X_test, thresh, graph_indicators, df_edges, step_size, train_time):  # this is for the train test
    start3 = time()
    test_betti = []
    test_graph_density = []
    test_graph_diameter = []
    test_clustering_coeff = []
    test_spectral_gap = []
    test_assortativity_ = []
    test_cliques = []
    test_motifs = []
    test_components = []
    for j in X_test:
        graph_id = j
        id_location = [index for index, element in enumerate(graph_indicators) if
                       element == graph_id]  # list the index of the graph_id locations
        graph_edges = df_edges[df_edges['from'].isin(id_location)]
        create_testgraph = Graph.TupleList(graph_edges.itertuples(index=False), directed=False, weights=True)
        test_distance_matrix = np.asarray(Graph.shortest_paths_dijkstra(create_testgraph))
        test_distance_matrix[test_distance_matrix == inf] = 0  # replace inf with zero
        test_sym_matrix = np.matmul(test_distance_matrix, test_distance_matrix.T)  # obtain a symmetric matrix
        [min2, max2] = [np.nanmin(test_sym_matrix), np.nanmax(test_sym_matrix[test_sym_matrix != np.inf])]
        testnorm_dist_matrix = test_sym_matrix / max2  # normalized distancematrix
        test_dim_reduction = PCA(n_components=3).fit_transform(testnorm_dist_matrix)
        test_alpha_complex = gd.AlphaComplex(points=test_dim_reduction)  # initialize alpha complex
        test_simplex_tree = test_alpha_complex.create_simplex_tree()  # creating a simplex tree
        test_diagrams = np.asarray(test_simplex_tree.persistence(),
                                   dtype='object')  # run AlphaComplex filtration on the normalized distance matrix
        # plt.clf()
        # plot_diagrams(test_diagrams, title= "test persistence diagrams showing H_0 and H_1",  show=True)

        # splitting the dimension into 0 and 1
        test_persist_0 = test_diagrams[:, 1][np.where(test_diagrams[:, 0] == 0)]
        test_persist_1 = test_diagrams[:, 1][np.where(test_diagrams[:, 0] == 1)]

        # obtain betti numbers for the unique dimensions
        test_betti_0 = []
        test_betti_1 = []

        for eps in np.linspace(0, thresh, step_size):
            b_0 = 0
            for k in test_persist_0:
                if k[0] <= eps and k[1] > eps:
                    b_0 = b_0 + 1
            test_betti_0.append(b_0)

            b_1 = 0
            for l in test_persist_1:
                if l[0] <= eps and l[1] > eps:
                    b_1 = b_1 + 1
            test_betti_1.append(b_1)

        test_betti.append(test_betti_0 + test_betti_1)  # concatenate betti numbers

        Density = create_testgraph.density()  # obtain density
        Diameter = create_testgraph.diameter()  # obtain diameter
        cluster_coeff = create_testgraph.transitivity_avglocal_undirected()  # obtain transitivity
        laplacian = create_testgraph.laplacian()  # obtain laplacian matrix
        laplace_eigenvalue = np.linalg.eig(laplacian)
        sort_eigenvalue = sorted(np.real(laplace_eigenvalue[0]), reverse=True)
        spectral = sort_eigenvalue[0] - sort_eigenvalue[1]  # obtain spectral gap
        assortativity = create_testgraph.assortativity_degree()  # obtain assortativity
        clique_count = create_testgraph.clique_number()  # obtain clique count
        motifs_count = create_testgraph.motifs_randesu(size=3)  # obtain motif count
        count_components = len(create_testgraph.clusters())  # obtain count components

        test_graph_density.append(Density)
        test_graph_diameter.append(Diameter)
        test_clustering_coeff.append(cluster_coeff)
        test_spectral_gap.append(spectral)
        test_assortativity_.append(assortativity)
        test_cliques.append(clique_count)
        test_motifs.append(motifs_count)
        test_components.append(count_components)

    df1_ = pd.DataFrame(test_betti)
    df2_ = pd.DataFrame(test_motifs)
    df3_ = pd.DataFrame(
        list(zip(test_graph_density, test_graph_diameter, test_clustering_coeff, test_spectral_gap, test_assortativity_, test_cliques, test_components)))
    test_data = pd.concat([df1_, df2_, df3_], axis=1, ignore_index=True)
    test_data = test_data.fillna(0)

    t3 = time()
    test_time = t3 - start3

    alpha_time = train_time + test_time

    return test_data, alpha_time


def tuning_hyperparameter():
    start4 = time()
    max_features = ['auto', 'sqrt']
    n_estimators = [int(a) for a in np.linspace(start=300, stop=300, num=1)]
    max_depth = [int(b) for b in np.linspace(start=5, stop=6, num=2)]
    min_samples_split = [2, 5]
    min_samples_leaf = [2]
    bootstrap = [True, False]
    num_cv = 10
    gridlength = len(n_estimators) * len(max_depth) * len(min_samples_leaf) * len(min_samples_split) * num_cv
    print(str(gridlength) + " RFs will be created in the grid search.")
    Param_Grid = dict(max_features=max_features, n_estimators=n_estimators, max_depth=max_depth,
                      min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, bootstrap=bootstrap)

    t4 = time()
    tuning_time = t4 - start4

    return Param_Grid, num_cv


def random_forest(dataset, Param_Grid, train_data, test_data, y_train, y_test, alpha_time, thresh, num_cv, step_size):
    print(dataset + " training started at", datetime.now().strftime("%H:%M:%S"))
    start5 = time()
    rfc = RandomForestClassifier(n_jobs=10)
    grid = GridSearchCV(estimator=rfc, param_grid=Param_Grid, cv=num_cv, n_jobs=10)
    grid.fit(train_data, y_train)
    param_choose = grid.best_params_
    if len(set(y_test)) > 2:  # multiclass case
        print(dataset + " requires multi class RF")
        forest = RandomForestClassifier(**param_choose, random_state=1, verbose=1).fit(train_data, y_train)
        y_pred = forest.predict(test_data)
        y_preda = forest.predict_proba(test_data)
        auc = roc_auc_score(y_test, y_preda, multi_class="ovr", average="macro")
        accuracy = accuracy_score(y_test, y_pred)
        conf_mat = confusion_matrix(y_test, y_pred)
    else:  # binary case
        rfc_pred = RandomForestClassifier(**param_choose, random_state=1, verbose=1).fit(train_data, y_train)
        test_pred = rfc_pred.predict(test_data)
        auc = roc_auc_score(y_test, rfc_pred.predict_proba(test_data)[:, 1])
        accuracy = accuracy_score(y_test, test_pred)
        conf_mat = confusion_matrix(y_test, test_pred)
    print(dataset + " accuracy is " + str(accuracy) + ", AUC is " + str(auc))

    t5 = time()
    training_time = t5 - start5

    print(f'Ripser took {alpha_time} seconds, training took {training_time} seconds')
    flat_conf_mat = (str(conf_mat.flatten(order='C')))[
                    1:-1]  # flatten confusion matrix into a single row while removing the [ ]
    file.write(dataset + "\t" + str(alpha_time) + "\t" + str(training_time) +
               "\t" + str(accuracy) + "\t" + str(auc) +
               "\t" + str(thresh) + "\t" + str(step_size) + "\t" + str(flat_conf_mat) + "\n")

    file.flush()


def main():
    X_train, X_test, y_train, y_test, graph_indicators, df_edges, graph_labels = read_csv(dataset)
    train_data, train_time = alpha_train(X_train, thresh, graph_indicators, df_edges, step_size)
    test_data, alpha_time = alpha_test(X_test, thresh, graph_indicators, df_edges, step_size, train_time)
    Param_Grid, num_cv = tuning_hyperparameter()
    random_forest(dataset, Param_Grid, train_data, test_data, y_train, y_test, alpha_time, thresh, num_cv, step_size)


if __name__ == '__main__':
    data_path = sys.argv[1]  # dataset path on computer
    data_list = ('ENZYMES', 'BZR')
    outputFile = "../../../results/" + 'Alpha01_graphstats.csv'
    file = open(outputFile, 'w')
    for dataset in data_list:
        for thresh in [1]:
            for step_size in [3, 5, 10, 20, 50]:  # we will consider step sizes [3,5,10,20,50] for epsilon
                for duplication in np.arange(5):
                    main()
    file.close()

#
