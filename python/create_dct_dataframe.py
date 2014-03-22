__author__ = 'brandonkelly'

import numpy as np
import pandas as pd
import os
import glob
import cPickle

base_dir = os.environ['HOME'] + '/Projects/Kaggle/galaxy_zoo/'
dct_dir = base_dir + 'data/react/'
plot_dir = base_dir + 'plots/'

npca = 500
verbose = True
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

    return X.astype(np.float32)


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

    X = build_dct_array(galaxy_ids, ncoefs)
    colnames = []
    for b in range(3):
        for i in xrange(ncoefs):
            colnames.append('DCT ' + str(b) + '.' + str(i))
    df = pd.DataFrame(data=X, index=galaxy_ids, columns=colnames)

    df.to_hdf(base_dir + 'data/DCT_coefs.h5', key='df')