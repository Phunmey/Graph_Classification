import numpy as np
import pandas as pd
import sys


def my_sort(datapath):
    columns = ['dataset', 'ripser_time', 'training_time', 'accuracy', 'auc', 'thresh', 'step_size', 'flat_conf_mat']
    read_data = pd.read_csv(datapath + "/" + "AlphacomplexPCA.csv", sep='\t', names=columns)
    sort_by = read_data.groupby(['dataset','step_size']).agg([np.mean, np.std])

    return read_data


def main():
    read_data = my_sort(datapath)


if __name__ == '__main__':
    datapath = sys.argv[1]
    main()