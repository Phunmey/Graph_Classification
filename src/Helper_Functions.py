import pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import roc_curve
import json


def get_read_csv(config, extension):
    return pd.read_csv(config["data_path"] + "/" + config["name"] + extension,
                       header=None)


def get_csv_value_sum(config, extension):
    return sum((get_read_csv(config, extension).values.tolist()), [])


def get_tsv_writer(file):
    tsv_file = csv.writer(file, delimiter="\t")  # makes output file into a tsv

    # column names: if you want more output you must create a column name here
    tsv_file.writerow(["dataset", "kernel", "Random_prediction_(AUROC)",
                       "RFC_(AUROC)", "accuracy_score", "run_time"])

    return tsv_file


# def write_tsv(data_set, rfc_auc, acc_score, start, tsv_file):
def write_tsv(config, kernel_name, r_auc, rfc_auc, acc_score, total_time):
    # if you want more output you must include here and update column names
    """
    # NEW VERSION from monday meeting
    tsv_file.writerow([data_set, "%.3f" % rfc_auc, acc_score, time() - start])
    """

    # OLD VERSION from before monday meeting
    config["tsv_file"].writerow([config["name"], kernel_name, "%.3f" % r_auc,
                                 "%.3f" % rfc_auc, acc_score, total_time])


# def plot_roc_curve(data_set, y_test, r_prob, rfc_prob, rfc_auc):
def plot_roc_curve(config, y_test, r_prob, rfc_prob, r_auc, rfc_auc):
    # PLOTTING THE ROC_CURVE
    """
    # NEW VERSION from monday meeting
    # rfc_fpr, rfc_tpr, thresholds = roc_curve(y_test, rfc_prob)  # compute ROC
    # rfc_fpr, rfc_tpr, thresholds = roc_curve(y_test, rfc_prob, pos_label=6)  # compute ROC
    """

    # OLD VERSION from before monday meeting
    r_fpr, r_tpr, thresholds = roc_curve(y_test, r_prob, pos_label=config["pos_label"])
    rfc_fpr, rfc_tpr, thresholds = roc_curve(y_test, rfc_prob, pos_label=config["pos_label"])  # compute ROC

    # rfac_auc = auc(rfc_fpr, rfc_tpr)

    plt.figure(figsize=(config["fig_size"], config["fig_size"]), dpi=config["dpi"])
    plt.plot(r_fpr, r_tpr, marker=".", label="Chance prediction (AUROC= %.3f)" % r_auc)
    plt.plot(rfc_fpr, rfc_tpr, linestyle="-", label="RFC (AUROC= %.3f)" % rfc_auc)
    plt.title("ROC Plot")  # title
    plt.xlabel("False Positive Rate")  # x-axis label
    plt.ylabel("True Positive Rate")  # y-axis label
    plt.legend()  # show legend
    plt.savefig(config["graph_path"] + ".png")  # save the plot
    plt.show()  # show plot


def get_configs(script_name):
    """
    gets and consolidates configs for each dataset
    return: list of config dictionaries
    credit: Jon Brownell and Simon Powell
    """
    config_file = "../config/config.json"  # relative path to config file
    with open(config_file, 'rt') as f:
        config_full = json.load(f)

    global_conf = config_full['global']
    datasets = config_full['dataset']
    default = config_full['default']  # default configs for datasets
    script = config_full[script_name]
    # config dictionaries for each dataset: conf comes after default so it will replace duplicate keys
    configs = [{'name': name, **global_conf, **default, **script, **conf} for name, conf in datasets.items()]

    for c in configs:
        c['filename'] = c['name'].replace(' ', '_').lower()  # clean filename
        c['graph_path'] = f'{c["graph_dir"]}{c["filename"]}'

    return configs
