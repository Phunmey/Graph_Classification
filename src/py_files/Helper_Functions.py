import pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import roc_curve
import json


def get_read_csv(config, extension):
    return pd.read_csv(config["data_path"] + config["name"] + extension, header=None)


def get_csv_value_sum(config, extension):
    return sum((get_read_csv(config, extension).values.tolist()), [])


def get_tsv_writer(file):
    tsv_file = csv.writer(file, delimiter="\t")  # makes output file into a tsv

    # column names: if you want more output you must create a column name here
    tsv_file.writerow(["dataset", "kernel", "Random_prediction_(AUROC)",
                       "RFC_(AUROC)", "accuracy_score", "run_time"])

    return tsv_file


def write_tsv(config, kernel_name, r_auc, rfc_auc, acc_score, total_time):
    # if you want more output you must include here and update column names
    config["tsv_file"].writerow([config["name"], kernel_name, "%.3f" % r_auc,
                                 "%.3f" % rfc_auc, acc_score, total_time])


def add_roc_info(config, plot_name, y_test, r_prob, rfc_prob, r_auc, rfc_auc):
    # adds roc curve information for future plotting of multiple lines in one graph
    config["y_test"] = y_test
    config["r_prob"] = r_prob
    config["r_auc"] = r_auc

    rfc_fpr, rfc_tpr, thresholds = roc_curve(y_test, rfc_prob, pos_label=config["pos_label"])
    config[plot_name + "/rfc_fpr"] = rfc_fpr
    config[plot_name + "/rfc_tpr"] = rfc_tpr
    config[plot_name + "/rfc_auc"] = rfc_auc


def plot_roc_curve(config):
    # PLOTTING THE ROC_CURVE
    # rfac_auc = auc(rfc_fpr, rfc_tpr)
    plt.figure(figsize=(config["fig_size"], config["fig_size"]), dpi=config["dpi"])
    plt.title("ROC Plot")  # title
    plt.xlabel("False Positive Rate")  # x-axis label
    plt.ylabel("True Positive Rate")  # y-axis label

    # calculate the chance prediction line and plot it
    r_fpr, r_tpr, thresholds = roc_curve(config["y_test"], config["r_prob"],
                                         pos_label=config["pos_label"])
    plt.plot(r_fpr, r_tpr, marker=".", label="Chance prediction (AUROC= %.3f)"
                                             % config["r_auc"])

    # plots a separate line for every item in config["plot_list"]
    if len(config["plot_list"]) != 0:
        for plot in config["plot_list"]:
            rfc_fpr = config[plot + "/rfc_fpr"]
            rfc_tpr = config[plot + "/rfc_tpr"]
            plt.plot(rfc_fpr, rfc_tpr, linestyle="-", label=plot + " RFC (AUROC= %.3f)"
                     % config[plot + "/rfc_auc"])
    else:
        plt.plot(config["rfc_fpr"], config["rfc_tpr"], linestyle="-",
                 label="RFC (AUROC= %.3f)" % config["rfc_auc"])

    plt.legend(loc='best')  # show legend, place it in the "best" location
    plt.savefig(config["graph_path"] + ".png")  # save the plot
    plt.show()  # show plot


def get_configs(script_name):
    """
    gets and consolidates configs for each dataset
    return: list of config dictionaries
    credit: Jon Brownell and Simon Powell
    """
    config_file = "../../config/config.json"  # relative path to config file
    with open(config_file, 'rt') as f:
        config_full = json.load(f)

    datasets = config_full['dataset']
    default = config_full['default']  # default configs for datasets
    script = config_full[script_name]  # which script specific configurations to run against
    # config dictionaries for each dataset: conf comes after default so it will replace duplicate keys
    configs = [{'name': name, **default, **script, **conf} for name, conf in datasets.items()]

    for c in configs:
        c['filename'] = c['name'].replace(' ', '_').lower()  # clean filename
        c['graph_path'] = f'{c["graph_dir"]}{c["filename"]}'

    return configs
