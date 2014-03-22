__author__ = 'brandonkelly'

__author__ = 'brandonkelly'

import numpy as np
import cPickle
import os
import matplotlib.pyplot as plt
import glob
from react import REACT2D
import triangle
from scipy.misc import bytescale
from sklearn.cross_decomposition import CCA
import pandas as pd
from dct_to_lda import remove_outliers

base_dir = os.environ['HOME'] + '/Projects/Kaggle/galaxy_zoo/'
dct_dir = base_dir + 'data/react/'
plot_dir = base_dir + 'plots/'

doshow = False
verbose = True
do_parallel = False


def plot_cca_projections(X_cca, n_components=7):

    labels = []
    for i in range(n_components):
        labels.append('CCA ' + str(i+1))

    fig = triangle.corner(X_cca[:, :n_components], labels=labels)
    return fig


def make_cca_images(cca, shape, dct_idx=None):

    n_components = cca.x_weights_.shape[1]
    U = REACT2D.build_dct(shape[0], shape[1], 50)

    if dct_idx is not None:
        U = U[:, dct_idx]
    dct_idx = np.arange(2499)

    cca_images = np.empty((n_components, shape[0], shape[1], 3))

    cca_images[:, :, :, 0] = \
        cca.components_[:, dct_idx].dot(U.T).reshape((n_components, shape[0], shape[1]))
    cca_images[:, :, :, 1] = \
        cca.components_[:, dct_idx + len(dct_idx)].dot(U.T).reshape((n_components, shape[0], shape[1]))
    cca_images[:, :, :, 2] = \
        cca.components_[:, dct_idx + 2*len(dct_idx)].dot(U.T).reshape((n_components, shape[0], shape[1]))

    ncca_rows = 3
    ncca_cols = 3
    nplots = 2

    cca_idx = 0
    for plot in range(nplots):
        idx = 1
        plt.clf()
        for row in range(ncca_rows):
            for col in range(ncca_cols):
                print row, col, idx
                plt.subplot(ncca_rows, ncca_cols, idx)
                plt.imshow(bytescale(cca_images[cca_idx, :, :, :]))
                plt.title('CCA ' + str(cca_idx + 1))
                plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
                plt.tick_params(axis='y', which='both', bottom='off', top='off', labelbottom='off')
                idx += 1
                cca_idx += 1
        plt.savefig(plot_dir + 'CCA_Images_' + str(plot + 1) + '.png')
        if doshow:
            plt.show()


if __name__ == "__main__":

    # find which galaxies we have a full dct for
    files_0 = glob.glob(dct_dir + '*_0_dct.pickle')
    files_1 = glob.glob(dct_dir + '*_1_dct.pickle')
    files_2 = glob.glob(dct_dir + '*_2_dct.pickle')

    galaxy_ids_0 = set([f.split('/')[-1].split('_')[0] for f in files_0])
    galaxy_ids_1 = set([f.split('/')[-1].split('_')[0] for f in files_1])
    galaxy_ids_2 = set([f.split('/')[-1].split('_')[0] for f in files_2])

    galaxy_ids = galaxy_ids_0 & galaxy_ids_1 & galaxy_ids_2
    if verbose:
        print "Found", len(galaxy_ids), "galaxies."

    # load the training labels
    if verbose:
        print 'Loading training labels and removing test data...'
    y = pd.read_csv(base_dir + 'data/training_solutions_rev1.csv').set_index('GalaxyID')

    # find which galaxy_ids correspond to the training set
    train_set = []
    train_ids = []
    for idx, gal_id in enumerate(galaxy_ids):
        if int(gal_id) in y.index:
            train_set.append(idx)
            train_ids.append(int(gal_id))
    train_set = np.array(train_set)

    # make sure indices align for X and y
    y = y.ix[train_ids]

    print 'Found', len(train_set), 'galaxies in training set.'

    if verbose:
        print 'Loading the data...'
    X = np.load(base_dir + 'data/DCT_array_all.npy')[train_set, :].astype(np.float32)

    zero_idx = np.where(np.all(X == 0, axis=1))[0]  # remove columns with all zeros
    if len(zero_idx) > 0:
        if verbose:
            print 'Detected the following columns as having all zeros and removed them:'
            print zero_idx
        X = np.delete(X, zero_idx, axis=1)
        dct_idx = np.where(np.all(X[:, :2500] != 0, axis=1))[0]
    else:
        dct_idx = None

    # remove outliers
    X, good_idx = remove_outliers(X, 6.0)
    y = y.ix[y.index[good_idx]]

    # only keep unique values



    # do CCA
    cca = CCA(n_components=len(y.columns), copy=False)
    X_cca, y_cca = cca.fit_transform(X, y)

    # make plots
    make_cca_images(cca, (100, 100), dct_idx=dct_idx)
    fig = plot_cca_projections(X_cca)
    fig.savefig(plot_dir + 'CCA_dist_no_outliers.png')
    if doshow:
        plt.show()

    print 'Saving the transformed values...'
    np.save(base_dir + 'data/LDA_training_transform', X_cca)