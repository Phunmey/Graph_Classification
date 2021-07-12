import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc
from time import time
from grakel.kernels import WeisfeilerLehman, VertexHistogram, WeisfeilerLehmanOptimalAssignment, ShortestPath
import csv

"""
UNUSED IMPORTS:
    import numpy as np
    from ripser import ripser
    from persim import plot_diagrams
    from pandas import DataFrame
    from igraph import *
    from matplotlib.pyplot import figure
    from plotly import graph_objs as go
"""


def standard_graph_file(data_set, tsv_file):
    start = time()
    data, graph_labels = get_data_and_labels(data_set)
    perform_rfc(data_set, data, graph_labels, start, tsv_file)


def get_data_and_labels(data_set):
    edges = get_read_csv(data_set, "_A.txt")
    edges.columns = ['from', 'to']
    unique_nodes = ((edges['from'].append(edges['to'])).unique()).tolist()
    missing_nodes = [x for x in range(unique_nodes[0], unique_nodes[-1] + 1) if
                     x not in unique_nodes]  # find the missing nodes
    nodes = unique_nodes + missing_nodes
    nodes.sort()
    graph_indicators = get_csv_value_sum(data_set, "_graph_indicator.txt")
    graph_labels = get_csv_value_sum(data_set, "_graph_labels.txt")
    node_labels = get_csv_value_sum(data_set, "_node_labels.txt")
    nodes_dict = dict(zip(nodes, node_labels))
    unique_graph_indicator = list(set(graph_indicators))

    data = []
    for i in unique_graph_indicator:
        graph_id = i
        graph_id_loc = [index for index, element in enumerate(graph_indicators) if
                        element == graph_id]
        edges_loc = edges[edges.index.isin(graph_id_loc)]
        edges_loc_asset = (edges_loc.to_records(index=False)).tolist()
        node_list = ((edges_loc['from'].append(edges_loc['to'])).unique()).tolist()
        ext = {k: nodes_dict[k] for k in node_list if k in nodes_dict}
        edges_nodes = [edges_loc_asset, ext]
        data.append(edges_nodes)

    return data, graph_labels


def get_read_csv(data_set, extension):
    path = "../data"
    return pd.read_csv(path + "/" + data_set + "/" + data_set + extension, header=None)


def get_csv_value_sum(data_set, extension):
    return sum((get_read_csv(data_set, extension).values.tolist()), [])


def perform_rfc(data_set, data, graph_labels, start, tsv_file):
    g_train, g_test, y_train, y_test = train_test_split(data, graph_labels,
                                                        test_size=0.2, random_state=42)

    w_lehman = WeisfeilerLehman(n_iter=4, base_graph_kernel=VertexHistogram, normalize=True)
    k_train = w_lehman.fit_transform(g_train)
    k_test = w_lehman.transform(g_test)

    rfc_pred = RandomForestClassifier().fit(k_train, y_train)
    y_pred = rfc_pred.predict(k_test)  # predict the class for k_test

    # PREDICTION PROBABILITIES
    r_prob = [0 for _ in range(len(y_test))]  # worst case scenario
    # predict the class prob. for k_test and keep the positive outcomes
    rfc_prob = (rfc_pred.predict_proba(k_test))#[:, 1]
    # r_auc = roc_auc_score(y_test, r_prob, multi_class='ovr')
    rfc_auc = roc_auc_score(y_test, rfc_prob, multi_class='ovr')  # Compute AUROC scores
    acc_score = accuracy_score(y_test, y_pred)

    # print('Random prediction: AUROC = %.3f' % r_auc)
    print('RFC: AUROC = %.3f' % rfc_auc)
    print("Accuracy score:", acc_score)
    print(f'Time taken to run: {time() - start} seconds')

    # write_tsv(data_set, r_auc, rfc_auc, acc_score, start, tsv_file)
    write_tsv(data_set, rfc_auc, acc_score, start, tsv_file)
    # plot_roc_curve(data_set, y_test, r_prob, rfc_prob, r_auc, rfc_auc)
    plot_roc_curve(data_set, y_test, r_prob, rfc_prob, rfc_auc)


# def write_tsv(data_set, r_auc, rfc_auc, acc_score, start, tsv_file):
def write_tsv(data_set, rfc_auc, acc_score, start, tsv_file):
    # if you want more output you must include here and update column names
    # tsv_file.writerow([data_set, '%.3f' % r_auc, '%.3f' % rfc_auc, acc_score,
    tsv_file.writerow([data_set, '%.3f' % rfc_auc, acc_score,
                       time() - start])


# def plot_roc_curve(data_set, y_test, r_prob, rfc_prob, r_auc, rfc_auc):
def plot_roc_curve(data_set, y_test, r_prob, rfc_prob, rfc_auc):
    # PLOTTING THE ROC_CURVE
    # r_fpr, r_tpr, thresholds = roc_curve(y_test, r_prob, pos_label=2)
    # rfc_fpr, rfc_tpr, thresholds = roc_curve(y_test, rfc_prob, pos_label=2)  # compute ROC
    rfc_fpr, rfc_tpr, thresholds = roc_curve(y_test, rfc_prob)  # compute ROC
    # rfc_fpr, rfc_tpr, thresholds = roc_curve(y_test, rfc_prob, pos_label=6)  # compute ROC
    # rfac_auc = auc(rfc_fpr, rfc_tpr)

    plt.figure(figsize=(4, 4), dpi=100)
    # plt.plot(r_fpr, r_tpr, marker='.', label='Chance prediction (AUROC= %.3f)' % r_auc)
    plt.plot(rfc_fpr, rfc_tpr, linestyle='-', label='RFC (AUROC= %.3f)' % rfc_auc)
    plt.title('ROC Plot')  # title
    plt.xlabel('False Positive Rate')  # x-axis label
    plt.ylabel('True Positive Rate')  # y-axis label
    plt.legend()  # show legend
    plt.savefig("../results/Graph_Kernels/plots/" + data_set + ".png")  # save the plot
    plt.show()  # show plot


def get_tsv_writer(file):
    tsv_file = csv.writer(file, delimiter='\t')  # makes output file into a tsv

    # column names: if you want more output you must create a column name here
    tsv_file.writerow(['dataset', 'Random_prediction_(AUROC)', 'RFC_(AUROC)',
                       'accuracy_score', 'run_time'])

    return tsv_file


if __name__ == '__main__':
    # removed "ENZYMES" and "FIRSTMM_DB" as they both have an error appear
    # datasets = ["BZR", "COX2", "DHFR", "PROTEINS"]
    datasets = ["ENZYMES"]

    output_file = open("../results/Graph_Kernels/Graph_Kernels_output.tsv", "wt")
    tsv_writer = get_tsv_writer(output_file)

    for dataset in datasets:
        print("Working on: " + dataset)
        standard_graph_file(dataset, tsv_writer)
        print()

    output_file.close()
    print("End of processing.")

    # TO DO:
    # GridsearchCV documentation and output tab separated files for Emma
    # Steven: Cross validation presentation on Monday
    # Poupak will present on feature scaling
    # We need a config file for the project
