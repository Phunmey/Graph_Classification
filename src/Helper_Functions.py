import pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import roc_curve


def get_read_csv(data_set, extension):
    path = "../data"
    return pd.read_csv(path + "/" + data_set + "/" + data_set + extension, header=None)


def get_csv_value_sum(data_set, extension):
    return sum((get_read_csv(data_set, extension).values.tolist()), [])


def get_tsv_writer(file):
    tsv_file = csv.writer(file, delimiter="\t")  # makes output file into a tsv

    # column names: if you want more output you must create a column name here
    tsv_file.writerow(["dataset", "kernel", "Random_prediction_(AUROC)",
                       "RFC_(AUROC)", "accuracy_score", "run_time"])

    return tsv_file


# def write_tsv(data_set, rfc_auc, acc_score, start, tsv_file):
def write_tsv(data_set, kernel_name, r_auc, rfc_auc, acc_score, total_time, tsv_file):
    # if you want more output you must include here and update column names
    """
    # NEW VERSION from monday meeting
    tsv_file.writerow([data_set, "%.3f" % rfc_auc, acc_score, time() - start])
    """

    # OLD VERSION from before monday meeting
    tsv_file.writerow([data_set, kernel_name, "%.3f" % r_auc, "%.3f" % rfc_auc,
                       acc_score, total_time])


# def plot_roc_curve(data_set, y_test, r_prob, rfc_prob, rfc_auc):
def plot_roc_curve(data_set, y_test, r_prob, rfc_prob, r_auc, rfc_auc, folder_name):
    # PLOTTING THE ROC_CURVE
    """
    # NEW VERSION from monday meeting
    # rfc_fpr, rfc_tpr, thresholds = roc_curve(y_test, rfc_prob)  # compute ROC
    # rfc_fpr, rfc_tpr, thresholds = roc_curve(y_test, rfc_prob, pos_label=6)  # compute ROC
    """

    # OLD VERSION from before monday meeting
    r_fpr, r_tpr, thresholds = roc_curve(y_test, r_prob, pos_label=2)
    rfc_fpr, rfc_tpr, thresholds = roc_curve(y_test, rfc_prob, pos_label=2)  # compute ROC

    # rfac_auc = auc(rfc_fpr, rfc_tpr)

    plt.figure(figsize=(4, 4), dpi=100)
    plt.plot(r_fpr, r_tpr, marker=".", label="Chance prediction (AUROC= %.3f)" % r_auc)
    plt.plot(rfc_fpr, rfc_tpr, linestyle="-", label="RFC (AUROC= %.3f)" % rfc_auc)
    plt.title("ROC Plot")  # title
    plt.xlabel("False Positive Rate")  # x-axis label
    plt.ylabel("True Positive Rate")  # y-axis label
    plt.legend()  # show legend
    plt.savefig("../results/" + folder_name + "/plots/" + data_set + ".png")  # save the plot
    plt.show()  # show plot
