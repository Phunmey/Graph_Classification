import gudhi as gd
import matplotlib.pyplot as plt
import numpy as np
from gudhi.representations import DiagramSelector
from gudhi.representations import Landscape, PersistenceImage
from matplotlib import gridspec
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from datasets.utils import create_multiple_circles


def creating_circles():
    N_sets_train = 900  # Number of train point clouds
    N_sets_test = 300  # Number of test  point clouds
    N_points = 600  # Point cloud cardinality
    N_noise = 200  # Number of corrupted points

    data_train,label_train = create_multiple_circles(N_sets_train, N_points, noisy=0, N_noise=N_noise)
    clean_data_test,clean_label_test = create_multiple_circles(N_sets_test,  N_points, noisy=0, N_noise=N_noise)
    noisy_data_test,noisy_label_test = create_multiple_circles(N_sets_test,  N_points, noisy=1, N_noise=N_noise)

    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(3, 3, width_ratios=[1,1,1], wspace=0.0, hspace=0.0)
    for i in range(3):
        for j in range(3):
            ax = plt.subplot(gs[i,j])
            ax.scatter(clean_data_test[3*i+j][:,0], clean_data_test[3*i+j][:,1], s=3)
            plt.xticks([])
            plt.yticks([])
    plt.savefig('C:/Code/expes_synth_ucr/expes/figures/' + 'clean_gudhi_pc.png', bbox_inches='tight')

    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(3, 3, width_ratios=[1, 1, 1], wspace=0.0, hspace=0.0)
    for i in range(3):
        for j in range(3):
            ax = plt.subplot(gs[i, j])
            ax.scatter(noisy_data_test[3 * i + j][:, 0], noisy_data_test[3 * i + j][:, 1], s=3)
            plt.xticks([])
            plt.yticks([])
    plt.savefig('C:/Code/expes_synth_ucr/expes/figures/' + 'noisy_gudhi_pc.png', bbox_inches='tight')

    return data_train, label_train, clean_label_test, noisy_label_test, clean_data_test, noisy_data_test

def pre_process(data_train, label_train, clean_label_test, noisy_label_test): #saving the circle numbers as proper labels
    le = LabelEncoder().fit(label_train)
    label_classif_train = le.transform(label_train)
    clean_label_classif_test = le.transform(clean_label_test)
    noisy_label_classif_test = le.transform(noisy_label_test)

    ds = [pairwise_distances(X).flatten() for X in data_train[:30]]
    maxd = np.max(np.concatenate(ds))

    return maxd

def simp_complex(data_train, maxd, clean_data_test, noisy_data_test):
    PD_train = []
    for X in tqdm(data_train):
        st = gd.AlphaComplex(points=X).create_simplex_tree(max_alpha_square=maxd)
        st.persistence()
        #dgmX = np.asarray(st.persistence(2), dtype='object') #used to plot the persistence diagrams
        dg = st.persistence_intervals_in_dimension(1)
        if len(dg) == 0:
            dg = np.empty([0, 2])
        PD_train.append(dg)
        # gd.plot_persistence_diagrams(dgmX)
        # plt.show()

    clean_PD_test = []
    for X in tqdm(clean_data_test):
        st = gd.AlphaComplex(points=X).create_simplex_tree(max_alpha_square=maxd)
        st.persistence()
        dg = st.persistence_intervals_in_dimension(1)
        if len(dg) == 0:
            dg = np.empty([0, 2])
        clean_PD_test.append(dg)


    noisy_PD_test = []
    for X in tqdm(noisy_data_test):
        st = gd.AlphaComplex(points=X).create_simplex_tree(max_alpha_square=maxd)
        st.persistence()
        dg = st.persistence_intervals_in_dimension(1)
        if len(dg) == 0:
            dg = np.empty([0, 2])
        noisy_PD_test.append(dg)

    return PD_train, clean_PD_test, noisy_PD_test

def diag_select(PD_train, clean_PD_test, noisy_PD_test):
    PVs_train, clean_PVs_test, noisy_PVs_test, PVs_params = [], [], [], []
    pds_train = DiagramSelector(use=True).fit_transform(PD_train)
    clean_pds_test = DiagramSelector(use=True).fit_transform(clean_PD_test)
    noisy_pds_test = DiagramSelector(use=True).fit_transform(noisy_PD_test)

    vpdtr = np.vstack(pds_train)
    pers = vpdtr[:, 1] - vpdtr[:, 0]
    bps_pairs = pairwise_distances(np.hstack([vpdtr[:, 0:1], vpdtr[:, 1:2] - vpdtr[:, 0:1]])[:200]).flatten()
    ppers = bps_pairs[np.argwhere(bps_pairs > 1e-5).ravel()]
    sigma = np.quantile(ppers, .2)
    im_bnds = [np.quantile(vpdtr[:, 0], 0.), np.quantile(vpdtr[:, 0], 1.), np.quantile(pers, 0.), np.quantile(pers, 1.)]
    sp_bnds = [np.quantile(vpdtr[:, 0], 0.), np.quantile(vpdtr[:, 1], 1.)]

    return sigma, im_bnds, pds_train, clean_pds_test, noisy_pds_test, sp_bnds

def vectorization(sigma, im_bnds, pds_train, clean_pds_test, noisy_pds_test, sp_bnds):
    PI_params = {'bandwidth': sigma, 'weight': lambda x: 10 * np.tanh(x[1]),
                 'resolution': [50, 50], 'im_range': im_bnds}
    PI_train = PersistenceImage(**PI_params).transform(pds_train)
    clean_PI_test = PersistenceImage(**PI_params).transform(clean_pds_test)
    noisy_PI_test = PersistenceImage(**PI_params).transform(noisy_pds_test)
    MPI = np.max(PI_train)
    PI_train /= MPI
    clean_PI_test /= MPI
    noisy_PI_test /= MPI

    PL_params = {'num_landscapes': 5, 'resolution': 300, 'sample_range': sp_bnds}
    PL_train = Landscape(**PL_params).transform(pds_train)
    clean_PL_test = Landscape(**PL_params).transform(clean_pds_test)
    noisy_PL_test = Landscape(**PL_params).transform(noisy_pds_test)
    MPL = np.max(PL_train)
    PL_train /= MPL
    clean_PL_test /= MPL
    noisy_PL_test /= MPL

    return

def main():
    data_train, label_train, clean_label_test, noisy_label_test, clean_data_test, noisy_data_test = creating_circles()
    maxd = pre_process(data_train, label_train, clean_label_test, noisy_label_test)
    PD_train, clean_PD_test, noisy_PD_test = simp_complex(data_train, maxd, clean_data_test, noisy_data_test)
    sigma, im_bnds, pds_train, clean_pds_test, noisy_pds_test, sp_bnds = diag_select(PD_train, clean_PD_test, noisy_PD_test)
    vectorization(sigma, im_bnds, pds_train, clean_pds_test, noisy_pds_test, sp_bnds)



if __name__ == '__main__':
    main()
