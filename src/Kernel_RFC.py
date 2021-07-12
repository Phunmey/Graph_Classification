from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from Helper_Functions import *
from src.Kernels import *


def standard_graph_file(data_set, tsv_file):
    start = time()
    data, graph_labels = get_data_and_labels(data_set)

    kernels = get_kernels()

    for kernel in kernels:
        perform_rfc(data_set, kernel, data, graph_labels, start, tsv_file)


def get_kernels():
    # add more kernel instances here
    w_lehman = WLehman()

    # add them to the list to return here
    kernels = [w_lehman]

    return kernels


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


def perform_rfc(data_set, kernel, data, graph_labels, start, tsv_file):
    g_train, g_test, y_train, y_test = train_test_split(data, graph_labels,
                                                        test_size=0.2, random_state=42)

    k_train = kernel.get_k_train(g_train)
    k_test = kernel.get_k_test(g_test)

    rfc_pred = RandomForestClassifier().fit(k_train, y_train)
    y_pred = rfc_pred.predict(k_test)  # predict the class for k_test

    # PREDICTION PROBABILITIES
    r_prob = [0 for _ in range(len(y_test))]  # worst case scenario

    # predict the class prob. for k_test and keep the positive outcomes

    """
    # NEW VERSION from monday meeting
    rfc_prob = (rfc_pred.predict_proba(k_test))#[:, 1]
    r_auc = roc_auc_score(y_test, r_prob, multi_class='ovr')
    rfc_auc = roc_auc_score(y_test, rfc_prob, multi_class='ovr')  # Compute AUROC scores
    """

    # OLD VERSION from before monday meeting
    rfc_prob = (rfc_pred.predict_proba(k_test))[:, 1]
    r_auc = roc_auc_score(y_test, r_prob)
    rfc_auc = roc_auc_score(y_test, rfc_prob)  # Compute AUROC scores

    acc_score = accuracy_score(y_test, y_pred)

    print("\tUsing Kernel:", kernel.get_name())
    print("\tRandom prediction: AUROC = %.3f" % r_auc)
    print("\tRFC: AUROC = %.3f" % rfc_auc)
    print("\tAccuracy score:", acc_score)
    print(f"\tTime taken to run: {time() - start} seconds\n")

    """
    # NEW VERSION from monday meeting
    # write_tsv(data_set, rfc_auc, acc_score, start, tsv_file)
    # plot_roc_curve(data_set, y_test, r_prob, rfc_prob, rfc_auc)
    """

    write_tsv(data_set, kernel.get_name(), r_auc, rfc_auc, acc_score, start, tsv_file)
    plot_roc_curve(data_set, y_test, r_prob, rfc_prob, r_auc, rfc_auc)


if __name__ == '__main__':
    # removed "ENZYMES" and "FIRSTMM_DB" as they both have an error appear
    # datasets = ["BZR", "COX2", "DHFR", "PROTEINS"]  # initial datasets
    # datasets = ["DD", "NCI1", "REDDIT-BINARY"]  # new datasets
    # datasets = ["ENZYMES"]  # run against the file with errors
    datasets = ["PROTEINS"]  # run against the file that's good

    output_file = open("../results/Kernel_RFC/Kernel_RFC_output.tsv", "wt")
    tsv_writer = get_tsv_writer(output_file)

    for dataset in datasets:
        print("Working on dataset: " + dataset)
        standard_graph_file(dataset, tsv_writer)
        print()

    output_file.close()
    print("End of processing.")

    # TO DO:
    # GridsearchCV documentation and output tab separated files for Emma
    # Steven: Cross validation presentation on Monday
    # Poupak will present on feature scaling
    # We need a config file for the project
