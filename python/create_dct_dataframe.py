__author__ = 'brandonkelly'

import numpy as np
import pandas as pd
import os
import glob
import cPickle
import multiprocessing

base_dir = os.environ['HOME'] + '/Projects/Kaggle/galaxy_zoo/'
dct_dir = base_dir + 'data/react/'
plot_dir = base_dir + 'plots/'

npca = 500
verbose = True
ncoefs = 40 * 40


def build_dct_array(galaxy_ids):

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

    n_jobs = multiprocessing.cpu_count() - 1
    pool = multiprocessing.Pool(n_jobs)
    pool.map(int, range(n_jobs))

    # find which ones we've already done
    already_done1 = glob.glob(plot_dir + '*_0.png')
    already_done2 = glob.glob(plot_dir + '*_1.png')
    already_done3 = glob.glob(plot_dir + '*_2.png')

    already_done1 = set([s.split('/')[-1].split('_')[0] for s in already_done1])
    already_done2 = set([s.split('/')[-1].split('_')[0] for s in already_done2])
    already_done3 = set([s.split('/')[-1].split('_')[0] for s in already_done3])

    already_done = already_done1 & already_done2 & already_done3
    print 'Found', len(already_done), 'galaxies.'

    galaxy_id = list(already_done)

    nchunks = len(galaxy_id) / n_jobs
    id_list = []
    integer_ids = []
    for j in range(n_jobs - 1):
        id_list.append(galaxy_id[j * nchunks:(j+1)*nchunks])
        integer_ids.extend(np.array(id_list[-1]).astype(np.int))
    id_list.append(galaxy_id[nchunks * (n_jobs-1):])
    integer_ids.extend(np.array(id_list[-1]))

    assert(len(np.unique(integer_ids)) == len(galaxy_id))

    output = pool.map(build_dct_array, id_list)

    X = np.vstack(output)

    cPickle.dump((galaxy_id, X), open(base_dir + 'data/DCT_values.pickle', 'wb'))

    colnames = []
    for b in range(3):
        for i in xrange(ncoefs):
            colnames.append('DCT ' + str(b) + '.' + str(i))
    df = pd.DataFrame(data=X, index=np.array(galaxy_id).astype(np.int), columns=colnames)

    df.to_hdf(base_dir + 'data/DCT_coefs.h5', key='df')