__author__ = 'brandonkelly'

import numpy as np
from sklearn.decomposition import RandomizedPCA
import cPickle
import os
import matplotlib.pyplot as plt
import multiprocessing
import glob

base_dir = os.environ['HOME'] + '/Projects/Kaggle/galaxy_zoo/'
dct_dir = base_dir + 'data/react/'
plot_dir = base_dir + 'plots/'

npca = 1000
doshow = False
verbose = False
do_parallel = False


def build_dct_array(galaxy_ids):

    X = np.empty((len(galaxy_ids), 7500))
    print 'Loading data for'
    for i, gal_id in enumerate(galaxy_ids):
        print gal_id
        dct_coefs = []
        for band in range(3):
            dct = cPickle.load(open(dct_dir + gal_id + '_' + str(band) + '_dct.pickle', 'rb'))
            dct_coefs.append(dct.coefs)

        X[i, :] = np.hstack(dct_coefs)

    return X


def make_pc_images(evect, dims):
    pass


if __name__ == "__main__":

    # find which galaxies we have a full dct for
    files_0 = glob.glob(dct_dir + '*_0_dct.pickle')
    files_1 = glob.glob(dct_dir + '*_1_dct.pickle')
    files_2 = glob.glob(dct_dir + '*_2_dct.pickle')

    galaxy_ids_0 = set([f.split('/')[-1].split('_')[0] for f in files_0])
    galaxy_ids_1 = set([f.split('/')[-1].split('_')[0] for f in files_1])
    galaxy_ids_2 = set([f.split('/')[-1].split('_')[0] for f in files_2])

    galaxy_ids = galaxy_ids_0 & galaxy_ids_1 & galaxy_ids_2

    # do the PCA
    X = build_dct_array(galaxy_ids)

    pca = RandomizedPCA(npca)

    X_pca = pca.fit_transform(X)

