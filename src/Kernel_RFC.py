from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from Helper_Functions import *
from src.Kernels import *
from time import time


def standard_graph_file(config):
    data, graph_labels, data_time = get_data_and_labels(config)
    kernels = get_kernels(config)

    print("\tPerforming random forest classification.")
    for kernel in kernels:
        perform_rfc(config, kernel, data, graph_labels, data_time)

    # creates a plot with a line for each kernel for this dataset / config
    plot_roc_curve(config)


def get_kernels(config):
    # add more kernel instances here
    w_lehman = WLehman()
    w_lehman_optimal = WLehmanOptimal()
    shortest_path = ShortPath()

    # add them to the list to return here
    kernels = [w_lehman, w_lehman_optimal, shortest_path]

    # adds a list of all kernel names to this dataset / config for plotting
    config["plot_list"] = [name.get_name() for name in kernels]

    return kernels


def get_data_and_labels(config):
    start = time()
    print("\tGetting data and labels.")

    edges = get_read_csv(config, "_A.txt")
    edges.columns = ["from", "to"]
    unique_nodes = ((edges["from"].append(edges["to"])).unique()).tolist()
    missing_nodes = [x for x in range(unique_nodes[0], unique_nodes[-1] + 1) if
                     x not in unique_nodes]  # find the missing nodes
    nodes = unique_nodes + missing_nodes
    nodes.sort()
    graph_indicators = get_csv_value_sum(config, "_graph_indicator.txt")
    graph_labels = get_csv_value_sum(config, "_graph_labels.txt")
    node_labels = get_csv_value_sum(config, "_node_labels.txt")
    nodes_dict = dict(zip(nodes, node_labels))
    unique_graph_indicator = list(set(graph_indicators))

    data = []
    for i in unique_graph_indicator:
        graph_id = i
        graph_id_loc = [index for index, element in enumerate(graph_indicators) if
                        element == graph_id]
        edges_loc = edges[edges.index.isin(graph_id_loc)]
        edges_loc_asset = (edges_loc.to_records(index=False)).tolist()
        node_list = ((edges_loc["from"].append(edges_loc["to"])).unique()).tolist()
        ext = {k: nodes_dict[k] for k in node_list if k in nodes_dict}
        edges_nodes = [edges_loc_asset, ext]
        data.append(edges_nodes)

    data_time = time() - start
    print(f"\t\tTime taken: {data_time} seconds\n")

    return data, graph_labels, data_time


def perform_rfc(config, kernel, data, graph_labels, data_time):
    start = time()
    print("\t\tUsing Kernel:", kernel.get_name())

    g_train, g_test, y_train, y_test = train_test_split(data, graph_labels,
                                                        test_size=config["test_size"],
                                                        random_state=config["random_state"])

    k_train = kernel.get_k_train(g_train)
    k_test = kernel.get_k_test(g_test)

    rfc_pred = RandomForestClassifier().fit(k_train, y_train)
    y_pred = rfc_pred.predict(k_test)  # predict the class for k_test

    # PREDICTION PROBABILITIES
    r_prob = [0 for _ in range(len(y_test))]  # worst case scenario

    # predict the class prob. for k_test and keep the positive outcomes
    rfc_prob = (rfc_pred.predict_proba(k_test))[:, 1]
    r_auc = roc_auc_score(y_test, r_prob, multi_class=config["multi_class"])
    rfc_auc = roc_auc_score(y_test, rfc_prob, multi_class=config["multi_class"])  # Compute AUROC scores

    acc_score = accuracy_score(y_test, y_pred)
    rfc_time = time() - start
    total_time = rfc_time + data_time

    print("\t\t\tRandom prediction: AUROC = %.3f" % r_auc)
    print("\t\t\tRFC: AUROC = %.3f" % rfc_auc)
    print("\t\t\tAccuracy score:", acc_score)
    print(f"\t\t\tTime taken to run RFC: {rfc_time} seconds")
    print(f"\t\t\tTotal time: {total_time} seconds\n")

    write_tsv(config, kernel.get_name(), r_auc, rfc_auc, acc_score, total_time)
    add_roc_info(config, kernel.get_name(), y_test, r_prob, rfc_prob, r_auc, rfc_auc)


if __name__ == "__main__":
    script_name = "Kernel_RFC"

    configs = get_configs(script_name)
    output_file = open(configs[0]["out_tsv_file"], "wt")
    tsv_writer = get_tsv_writer(output_file)

    # Change the following lines to whatever dataset you want to test,
    # or comment them out if you want to run all datasets.
    # Skipping ENZYMES and FIRSTMM_DB due to error messages
    configs = [configs[0],
               configs[1],
               configs[2],
               configs[3],
               configs[6],
               configs[7]]

    for conf in configs:
        print("Working on dataset: " + conf["name"])
        conf["tsv_file"] = tsv_writer
        standard_graph_file(conf)
        print()

    output_file.close()
    print("End of processing.")
