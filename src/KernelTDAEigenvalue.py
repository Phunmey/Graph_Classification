import random
from datetime import datetime
from time import time

import numpy as np
import pandas as pd
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from igraph import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV


def standardGraphFile(dataset, file, datapath, h_filt, iter, filtration, max_allowed_filtration):
    start = time()
    edges_asdf = pd.read_csv(datapath + "/" + dataset + "/" + dataset + "_A.txt", header=None)  # import edge data
    edges_asdf.columns = ['from', 'to']
    unique_nodes = ((edges_asdf['from'].append(edges_asdf['to'])).unique()).tolist()
    print(dataset + " graph edges are loaded")
    node_list = np.arange(min(unique_nodes), max(unique_nodes) + 1);  # unique_nodes + missing_nodes
    node_list.sort()
    csv = pd.read_csv(datapath + "/" + dataset + "/" + dataset + "_graph_indicator.txt", header=None)
    print(dataset + " graph indicators are loaded")
    csv.columns = ["ID"]
    graphindicator_aslist = ((csv["ID"].values.astype(int)))
    read_csv = pd.read_csv(datapath + "/" + dataset + "/" + dataset + "_graph_labels.txt", header=None)
    read_csv.columns = ["ID"]
    graphlabels_aslist = ((read_csv["ID"].values.astype(int)))
    print(dataset + " graph labels are loaded")

    random_nodelabels = [5] * len(node_list)
    random_dict = list(dict(zip(node_list, random_nodelabels)).items())
    unique_graphindicator = np.arange(min(graphindicator_aslist),
                                      max(graphindicator_aslist) + 1)  # list unique graphids 100

    random.seed(123)

    progress = len(unique_graphindicator)
    total_degree = {}
    node_degree_max = []  # list of the node degree maximums
    node_degree_min = []  # list of the node degree minimums
    print(dataset + " has " + str(progress) + " graphs.")
    activation_discovery(dataset, edges_asdf, graphindicator_aslist, node_degree_max, node_degree_min, progress,
                         total_degree,
                         unique_graphindicator)
    max_activation = max(node_degree_max)  # obtain the maximum of the degree maximums
    min_activation = min(node_degree_min)  # obtain the minimum of the degree minimums
    # if (max_activation-min_activation) > max_allowed_filtration:
    #     max_activation = 500
    # max_activation = int(np.percentile(node_degree_max, 90))
    print(dataset + " max activation was " + str(max(node_degree_max)) + ", we will use " + str(max_activation))
    print(dataset + " min activation was " + str(min(node_degree_min)))

    filtr_range = filtration_discovery(dataset, filtration, h_filt, max_activation, max_allowed_filtration,
                                       min_activation)
    feature_matrix = []
    for graphid in unique_graphindicator:
        kernelize_graph(feature_matrix, edges_asdf, filtr_range, filtration, graphid, graphindicator_aslist, iter,
                        progress, random_dict)
    rfc_input = pd.DataFrame(feature_matrix)
    print(dataset + " has a feature matrix of " + str(rfc_input.shape))
    t2 = time()
    time_taken = t2 - start
    random.seed(42)
    g_train, g_test, y_train, y_test = train_test_split(rfc_input, graphlabels_aslist, test_size=0.2,
                                                        random_state=random.randint(0, 100))

    # hyperparameter tuning
    Param_Grid, num_cv = rf_preprocess()

    # start training
    print(dataset + " training started at", datetime.now().strftime("%H:%M:%S"))

    accuracy, auc, conf_mat = train_test_rf(Param_Grid, dataset, g_test, g_train, num_cv, y_test, y_train)
    print(dataset + " accuracy is " + str(accuracy) + ", AUC is " + str(auc))
    t3 = time()
    print(f'Kernels took {time_taken} seconds, training took {t3 - t2} seconds')
    flat_conf_mat = (str(conf_mat.flatten(order='C')))[1:-1]
    file.write(dataset + "\t" + str(time_taken) + "\t" + str(t3 - t2) +
               "\t" + str(accuracy) + "\t" + str(auc) + "\t" + str(iter) + "\t" + str(h_filt) + "\t" +
               str(flat_conf_mat) + "\n")
    file.flush()


def activation_discovery(dataset, edges_asdf, graphindicator_aslist, node_degree_max, node_degree_min, progress,
                         total_degree,
                         unique_graphindicator):
    for graphid1 in unique_graphindicator:
        if graphid1 % (progress / 100) == 0:
            print(str(graphid1) + "/" + str(progress) + " completed")
        graphid_loc1 = [index for index, element in enumerate(graphindicator_aslist) if
                        element == graphid1]  # list the index of the graphid locations
        edges_loc1 = edges_asdf[edges_asdf.index.isin(graphid_loc1)]  # obtain edges that corresponds to these locations
        a_graph1 = Graph.TupleList(edges_loc1.itertuples(index=False), directed=False, weights=True)
        activation_values = np.asarray(a_graph1.degree())  # obtain node degrees
        # activation_values = [int(i) for i in np.asarray((a_graph1.betweenness()))] #obtain betweenness
        node_degree_max.append(max(activation_values))
        node_degree_min.append(min(activation_values))

        for i in activation_values:
            total_degree[i] = total_degree.get(i, 0) + 1
    # plt.bar(total_degree.keys(), total_degree.values(), 1, color='b')
    # plt.xticks(np.arange(min(node_degree_min), max(node_degree_max) + 1))
    # #plt.yscale()
    # plt.xlabel('Degrees')
    # plt.ylabel('Fraction of nodes')  # obtained by dividing the node count of the filtration by the data node count
    # plt.title(dataset)
    # # plt.show()
    # plt.savefig("/home/taiwo/projects/def-cakcora/taiwo/results/" + dataset + "DegreeStats.png")
    print(dataset + " degree computations are completed.")


def filtration_discovery(dataset, filtration, h_filt, max_activation, max_allowed_filtration, min_activation):
    if filtration == "sublevel":
        if h_filt:
            activation2 = int(max_activation / 2) + 1
            if (activation2 - min_activation) > max_allowed_filtration:
                filtr_range = np.unique(
                    np.linspace(start=min_activation, stop=activation2, dtype=int, num=max_allowed_filtration))
            else:
                filtr_range = np.arange(min_activation, activation2)
        else:
            if (max_activation - min_activation) > max_allowed_filtration:
                filtr_range = np.unique(
                    np.linspace(start=min_activation, stop=max_activation + 1, dtype=int, num=max_allowed_filtration))
            else:
                filtr_range = np.arange(min_activation, max_activation + 1)
    else:
        if h_filt:
            activation3 = int(max_activation / 2) - 1
            if (max_activation - activation3) > max_allowed_filtration:
                filtr_range = np.flip(np.unique(
                    np.linspace(start=max_activation, stop=activation3, dtype=int, num=max_allowed_filtration)))
            else:
                filtr_range = np.arange(max_activation, activation3, -1)
        else:
            if (max_activation - min_activation) > max_allowed_filtration:
                filtr_range = np.flip(np.unique(
                    np.linspace(start=max_activation, stop=min_activation, dtype=int, num=max_allowed_filtration)))
            else:
                filtr_range = np.arange(max_activation, min_activation - 1, -1)
    print(
        dataset + " filtration will run from " + str(filtr_range[0]) + " to " + str(filtr_range[len(filtr_range) - 1]))
    return filtr_range


def kernelize_graph(feature_matrix, edges_asdf, filtr_range, filtration, graphid, graphindicator_aslist, iter, progress,
                    random_dict):
    if graphid % (progress / 10) == 0:
        print(str(graphid) + "/" + str(progress) + " graphs completed")
    graphid_loc = [index for index, element in enumerate(graphindicator_aslist) if
                   element == graphid]  # list the index of the graphid locations
    edges_loc = edges_asdf[edges_asdf.index.isin(graphid_loc)]  # obtain edges that corresponds to these locations
    nodedict_loc = dict([random_dict[pos] for pos in graphid_loc])
    a_graph = Graph.TupleList(edges_loc.itertuples(index=False), directed=False, weights=True)
    activation_values = np.asarray(a_graph.degree())
    # activation_values =[int(i) for i in np.asarray((a_graph.betweenness()))]
    wl_data = [[] for j in range(0, len(filtr_range))]
    for indx, deg in enumerate(filtr_range, start=0):
        if filtration == "sublevel":
            deg_loc = (np.where(activation_values <= deg))[0]
        else:
            deg_loc = (np.where(activation_values >= deg))[0]
        sub_graph = a_graph.subgraph(deg_loc)  # construct subgraphs from original graph using the indices
        subname = sub_graph.vs["name"]  # the subgraph vertex names
        subdict = [(k, v) for k, v in nodedict_loc.items() for k in
                   subname]  # list nodes and nodelabels as dict
        subedges = sub_graph.get_edgelist()  # obtain subgraph edges
        if subedges != []:
            subname_ids = set([x for y in subedges for x in y])  # obtain unique node indices for subgraph
            # subnodes = [subname[pos] for pos in subname_ids]  # corresponding vertex names
            dict_node = dict([subdict[u] for u in subname_ids])  # obtain corresponding dictionaries of indices
            index_dict = dict(
                zip(subname_ids, list(dict_node.values())))  # replace the dict keys with node indices
            nodes_concat = [subedges, index_dict]
            # wl_data[deg - min_activation].extend(nodes_concat)
            wl_data[indx].extend(nodes_concat)
    for e in wl_data:
        if e == []:
            e.extend([[(1, 2)], {1: 5, 2: 5}])
            # Approach 2: e.extend([[(-1, -1)], {-1: -1}])
            # Approach 3: e.extend([[(0, 0)], {0: 0}])
    wl = WeisfeilerLehman(n_iter=iter, base_graph_kernel=VertexHistogram, normalize=True)
    wl_transform = wl.fit_transform(wl_data)
    eigen_value, eigen_vector = np.linalg.eig(wl_transform)
    # compute_mean = np.mean(eigen_value, axis=0)
    # upper_diag = wl_transform[np.triu_indices(len(wl_transform), k=1)]
    feature_matrix.append(np.real(eigen_value))


def rf_preprocess():
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
    return Param_Grid, num_cv


def train_test_rf(Param_Grid, dataset, g_test, g_train, num_cv, y_test, y_train):
    rfc = RandomForestClassifier(n_jobs=10)
    grid = GridSearchCV(estimator=rfc, param_grid=Param_Grid, cv=num_cv, n_jobs=10)
    grid.fit(g_train, y_train)
    param_choose = grid.best_params_
    if len(set(y_test)) > 2:  # multiclass case
        print(dataset + " requires multi class RF")
        forest = RandomForestClassifier(**param_choose, random_state=1, verbose=1).fit(g_train, y_train)
        y_pred = forest.predict(g_test)
        y_preda = forest.predict_proba(g_test)
        # print(pd.crosstab(y_test, y_pred))
        auc = roc_auc_score(y_test, y_preda, multi_class="ovr", average="macro")
        # auc_random = roc_auc_score(y_test, r_prob, multi_class="ovr")
        accuracy = accuracy_score(y_test, y_pred)
        conf_mat = confusion_matrix(y_test, y_pred)
    # print(conf_mat)
    else:  # binary case
        rfc_pred = RandomForestClassifier(**param_choose, random_state=1, verbose=1).fit(g_train, y_train)
        test_pred = rfc_pred.predict(g_test)
        auc = roc_auc_score(y_test, rfc_pred.predict_proba(g_test)[:, 1])
        accuracy = accuracy_score(y_test, test_pred)
        conf_mat = confusion_matrix(y_test, test_pred)
    return accuracy, auc, conf_mat


if __name__ == '__main__':
    datapath = sys.argv[1]  # dataset path on computer such as  "C:/data"
    datasets = (
        "ENZYMES", "REDDIT-MULTI-5K", "REDDIT-MULTI-12K", 'BZR', 'MUTAG', 'DD', 'PROTEINS', 'DHFR', 'NCI1', 'COX2')
    outputFile = "../results/" + 'Eigenvalueresults.csv'
    output_file = open(outputFile, 'w')
    for dataset_name in datasets:
        for filtr_type in ('sublevel'):
            for iter_ in (2, 3, 4):
                for duplication in range(5):
                    standardGraphFile(dataset_name, output_file, datapath, h_filt=False, iter=iter_,
                                      filtration=filtr_type, max_allowed_filtration=10)
    output_file.close()
