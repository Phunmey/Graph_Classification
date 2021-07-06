import numpy as np
from ripser import ripser
from persim import plot_diagrams
import pandas as pd
from pandas import DataFrame
from igraph import *
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc
from plotly import graph_objs as go
from time import time
from grakel.kernels import WeisfeilerLehman, VertexHistogram


def standardGraphFile(dataset):
    start = time()
    edges_asdf = get_read_csv(dataset, "_A.txt")
    edges_asdf.columns = ['from', 'to']
    unique_nodes = ((edges_asdf['from'].append(edges_asdf['to'])).unique()).tolist()
    missing_nodes = [x for x in range(unique_nodes[0], unique_nodes[-1] + 1) if
                     x not in unique_nodes]  # find the missing nodes
    node_list = unique_nodes + missing_nodes
    node_list.sort()
    graphindicator_aslist = get_csv_value_sum(dataset, "_graph_indicator.txt")
    graphlabels_aslist = get_csv_value_sum(dataset, "_graph_labels.txt")
    nodelabels_aslist = get_csv_value_sum(dataset, "_node_labels.txt")
    nodes_dict = dict(zip(node_list, nodelabels_aslist))
    unique_graphindicator = list(set(graphindicator_aslist))

    DATA = []
    for i in unique_graphindicator:
        graphid = i
        graphid_loc = [index for index, element in enumerate(graphindicator_aslist) if element == graphid]
        edges_loc = edges_asdf[edges_asdf.index.isin(graphid_loc)]
        edges_loc_asset = (edges_loc.to_records(index=False)).tolist()
        nodes_aslist = ((edges_loc['from'].append(edges_loc['to'])).unique()).tolist()
        ext = {k: nodes_dict[k] for k in nodes_aslist if k in nodes_dict}
        # empty_dict = dict()
        edges_nodes = [edges_loc_asset, ext]
        DATA.append(edges_nodes)

    G_train, G_test, y_train, y_test = train_test_split(DATA, graphlabels_aslist, test_size=0.2, random_state=42)

    WL = WeisfeilerLehman(n_iter=4, base_graph_kernel=VertexHistogram, normalize=True)
    K_train = WL.fit_transform(G_train)
    K_test = WL.transform(G_test)

    RFC_pred = RandomForestClassifier().fit(K_train, y_train)
    y_pred = RFC_pred.predict(K_test)  # predict the class for K_test

    # PREDICTION PROBABILITIES
    r_probs = [0 for _ in range(len(y_test))]  # worst case scenario
    RFC_probs = (RFC_pred.predict_proba(K_test))[:,
                1]  # predict the class prob. for K_test and keep the positive outcomes
    r_auc = roc_auc_score(y_test, r_probs)
    RFC_auc = roc_auc_score(y_test, RFC_probs)  # Compute AUROC scores

    print('Random prediction: AUROC = %.3f' % (r_auc))
    print('RFC: AUROC = %.3f' % (RFC_auc))
    print(accuracy_score(y_test, y_pred))
    print(f'Time taken to run:{time() - start} seconds')

    plot_roc_curve(y_test, r_probs, RFC_probs, r_auc, RFC_auc)


def plot_roc_curve(y_test, r_probs, RFC_probs, r_auc, RFC_auc):
    #PLOTTING THE ROC_CURVE
    r_fpr, r_tpr, thresholds = roc_curve(y_test, r_probs, pos_label=2)
    RFC_fpr, RFC_tpr, thresholds = roc_curve(y_test, RFC_probs, pos_label=2)  # compute ROC
    # RFAC_auc = auc(RFC_fpr, RFC_tpr)

    plt.figure(figsize=(4, 4), dpi=100)
    plt.plot(r_fpr, r_tpr, marker='.', label='Chance prediction (AUROC= %.3f)' % r_auc)
    plt.plot(RFC_fpr, RFC_tpr, linestyle='-', label='RFC (AUROC= %.3f)' % RFC_auc)
    plt.title('ROC Plot')  # title
    plt.xlabel('False Positive Rate')  # x-axis label
    plt.ylabel('True Positive Rate')  # y-axis label
    plt.legend()  # show legend
    plt.show()  # show plot


def get_read_csv (dataset, extension):
    path = "../data"
    return pd.read_csv(path + "/" + dataset + "/" + dataset + extension, header=None)


def get_csv_value_sum(dataset, extension):
    return sum((get_read_csv(dataset, extension).values.tolist()), [])


if __name__ == '__main__':
    dataset = 'PROTEINS'
    standardGraphFile(dataset)

    # TO DO:
    # GridsearchCV documentation and output tab separated files for Emma
    # Steven: Cross validation presentation on Monday
    # Poupak will present on feature scaling
    # We need a config file for the project
