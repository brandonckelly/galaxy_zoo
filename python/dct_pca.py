__author__ = 'brandonkelly'

import numpy as np
import cPickle
import os
import matplotlib.pyplot as plt
import multiprocessing
import glob
from react import REACT2D
import triangle
from scipy.misc import bytescale
from scipy import linalg
from sklearn.decomposition import RandomizedPCA

base_dir = os.environ['HOME'] + '/Projects/Kaggle/galaxy_zoo/'
dct_dir = base_dir + 'data/react/'
plot_dir = base_dir + 'plots/'

npca = 500
doshow = False
verbose = True
do_parallel = False
load_X = False
load_pca = False
ncoefs = 2500


def build_dct_array(galaxy_ids, ncoefs):

    X = np.empty((len(galaxy_ids), ncoefs * 3))
    print 'Loading data for source'
    for i, gal_id in enumerate(galaxy_ids):
        print i + 1
        dct_coefs = []
        for band in range(3):
            image_file = open(dct_dir + gal_id + '_' + str(band) + '_dct.pickle', 'rb')
            dct = cPickle.load(image_file)
            image_file.close()
            if len(dct.coefs) < ncoefs:
                nzeros = ncoefs - len(dct.coefs)
                dct.coefs = np.append(dct.coefs, np.zeros(nzeros))
            dct_coefs.append(dct.coefs[:ncoefs])

        X[i, :] = np.hstack(dct_coefs)

    return X


def make_pc_images(pca, shape):

    U = REACT2D.build_dct(shape[0], shape[1], 50)

    pca_images = np.empty((npca, shape[0], shape[1], 3))

    pca_images[:, :, :, 0] = pca.components_[:, :ncoefs].dot(U.T[:ncoefs, :]).reshape((npca, shape[0], shape[1]))
    pca_images[:, :, :, 1] = pca.components_[:, ncoefs:2*ncoefs].dot(U.T[:ncoefs, :]).reshape((npca, shape[0], shape[1]))
    pca_images[:, :, :, 2] = pca.components_[:, 2*ncoefs:].dot(U.T[:ncoefs, :]).reshape((npca, shape[0], shape[1]))

    npca_rows = 3
    npca_cols = 3
    nplots = 2

    pca_idx = 0
    for plot in range(nplots):
        idx = 1
        plt.clf()
        for row in range(npca_rows):
            for col in range(npca_cols):
                print row, col, idx
                plt.subplot(npca_rows, npca_cols, idx)
                plt.imshow(bytescale(pca_images[pca_idx, :, :, :]))
                plt.title('PC ' + str(pca_idx + 1))
                plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
                plt.tick_params(axis='y', which='both', bottom='off', top='off', labelbottom='off')
                idx += 1
                pca_idx += 1
        plt.savefig(plot_dir + 'PC_Images_' + str(plot + 1) + '.png')
        if doshow:
            plt.show()


def plot_pc_projections(X_pca, npca=5):

    labels = []
    for i in range(npca):
        labels.append('PC ' + str(i+1))

    fig = triangle.corner(X_pca[:, :npca], labels=labels)
    return fig


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

    if load_X:
        if verbose:
            print 'Loading the data...'
        X = np.load(base_dir + 'data/DCT_array_all.npy')
    else:
        X = build_dct_array(galaxy_ids, ncoefs)
        np.save(base_dir + 'data/DCT_array_all', X)

    # find the outliers
    if verbose:
        print 'Finding the outliers...'
    row_norm = np.linalg.norm(X - np.median(X, axis=0), axis=1)
    mad = np.median(np.abs(row_norm - np.median(row_norm)))
    robsig = 1.48 * mad
    zscore = np.abs(row_norm - np.median(row_norm)) / robsig
    notout = np.where(zscore < 10)[0]
    out = np.where(zscore > 10)[0]
    print 'Found', X.shape[0] - len(notout), 'outliers'
    X = X[notout, :]
    galaxy_ids = np.array(list(galaxy_ids))
    if verbose:
        print galaxy_ids[out]

    if load_pca:
        print 'Loading PCA...'
        rpca = cPickle.load(open(base_dir + 'data/DCT_PCA.pickle', 'rb'))
        X = rpca.transform(X)
    else:
        # do the PCA
        if verbose:
            print 'Doing PCA...'
        rpca = RandomizedPCA(n_components=npca)
        X = rpca.fit_transform(X)
        rpca.galaxy_ids = galaxy_ids[notout]
        cPickle.dump(rpca, open(base_dir + 'data/DCT_PCA.pickle', 'wb'))

    plt.plot(rpca.explained_variance_ratio_.cumsum())
    plt.ylabel('Cumulative Fractional Explained Variance')
    plt.xlabel('Number of Components')
    plt.savefig(plot_dir + 'explained_variance.png')
    if doshow:
        plt.show()

    # now plot after removing outliers
    fig = plot_pc_projections(X, npca=6)
    fig.savefig(plot_dir + 'PC_dist_no_outliers.png')
    if doshow:
        plt.show()

    if verbose:
        print 'Making Images of PCs...'
    make_pc_images(rpca, (100, 100))