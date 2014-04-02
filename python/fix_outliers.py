__author__ = 'brandonkelly'

import numpy as np
import os
import pandas as pd
import glob
from make_feature_dataframe import make_gaussfit_features
from extract_postage_stamp import extract_gal_image
import cPickle
from galaxies_to_dct import do_dct_transform
import multiprocessing
import matplotlib.pyplot as plt

base_dir = os.environ['HOME'] + '/Projects/Kaggle/galaxy_zoo/'
data_dir = base_dir + 'data/'
test_dir = data_dir + 'images_test_rev1/'
train_dir = data_dir + 'images_training_rev1/'
dct_dir = data_dir + 'react/'
ann_dir = data_dir + 'nnets/'
plot_dir = base_dir + 'plots/'

verbose = True
do_parallel = True

test_files = glob.glob(test_dir + '*jpg')
train_files = glob.glob(train_dir + '*jpg')

rpca = cPickle.load(open(base_dir + 'data/DCT_PCA.pickle', 'rb'))
npcs = 200

questions = range(1, 12)


def get_lda(question):
    if verbose:
        print 'Loading LDA transform for question'
        print question, '...'
    lda = cPickle.load(open(base_dir + 'data/DCT_LDA_' + str(question) + '.pickle', 'rb'))
    return lda

lda_list = map(get_lda, questions)


def rerun_pipeline(galaxy_id):

    # steps: extract_gal_image, run DCT transform, project onto PCA, project onto LDA, compute gaussian features
    if verbose:
        print 'Rerunning pipeline for', galaxy_id

    # find where this JPG image is
    train_name = train_dir + str(galaxy_id) + '.jpg'
    test_name = test_dir + str(galaxy_id) + '.jpg'
    if train_name in train_files:
        file_dir = train_dir
        file_name = train_name
    elif test_name in test_files:
        file_dir = test_dir
        file_name = test_name
    else:
        err_msg = 'Cannot find file for', galaxy_id
        print 'Cannot find file for', galaxy_id
        return

    err_msg = extract_gal_image(file_name)

    do_dct_transform((str(galaxy_id), file_dir))

    dct_coefs = []
    ncoefs = 2500
    for band in range(3):
        image_file = open(dct_dir + str(galaxy_id) + '_' + str(band) + '_dct.pickle', 'rb')
        dct = cPickle.load(image_file)
        image_file.close()
        if len(dct.coefs) < ncoefs:
            nzeros = ncoefs - len(dct.coefs)
            dct.coefs = np.append(dct.coefs, np.zeros(nzeros))
        dct_coefs.append(dct.coefs[:ncoefs])

    dct_coefs = np.hstack(dct_coefs)
    dct_coefs -= rpca.mean_

    print 'Getting PCA and LDA transformed coefficients...'

    pc_coefs = np.dot(rpca.components_[:npcs], dct_coefs)

    dct_coefs += rpca.mean_
    # now add LDA directions
    lda_coefs = []
    for question in questions:
        lda = lda_list[question-1]
        ncoefs = lda.components_.shape[1] / 3
        dct_idx = np.asarray([np.arange(ncoefs), 2500 + np.arange(ncoefs), 5000 + np.arange(ncoefs)]).ravel()
        if question == 1:
            dct_coefs = dct_coefs[dct_idx]
        lda_proj = lda.components_.dot(dct_coefs)
        lda_coefs.append(lda_proj)

    lda_coefs = np.hstack(lda_coefs)

    # now add gaussian features
    gfeatures = make_gaussfit_features(galaxy_id)

    return pc_coefs, lda_coefs, gfeatures


if __name__ == "__main__":

    njobs = multiprocessing.cpu_count() - 1

    pool = multiprocessing.Pool(njobs)
    pool.map(int, range(njobs))

    # load the data for the features
    df = pd.read_hdf(base_dir + 'data/galaxy_features.h5', 'df')

    assert np.all(np.isfinite(df.values))

    # find the outliers
    pc_names = []
    for c in df.columns:
        if 'PC' in c:
            pc_names.append(c)

    if verbose:
        print 'Finding the outliers...'
    X = df[pc_names].values
    thresh = 6.0

    row_norm = np.linalg.norm(X - np.median(X, axis=0), axis=1)
    mad = np.median(np.abs(row_norm - np.median(row_norm)))
    robsig = 1.48 * mad
    zscore = np.abs(row_norm - np.median(row_norm)) / robsig
    out = np.where(zscore > thresh)[0]
    print 'Found', len(out), 'outliers out of', len(df), 'galaxies.'

    outliers = df.index[out]
    if do_parallel:
        features = pool.map(rerun_pipeline, outliers)
    else:
        features = map(rerun_pipeline, outliers)

    lda_names = []
    for c in df.columns:
        if 'LDA' in c:
            lda_names.append(c)

    gauss_labels = ['GalaxyCentDist', 'GalaxyMajor', 'GalaxyAratio', 'GalaxyFlux',
                    'GaussMahDist_1', 'GaussMajor_1', 'GaussAratio_1', 'GaussFlux_1',
                    'GaussMahDist_2', 'GaussMajor_2', 'GaussAratio_2', 'GaussFlux_2',
                    'GaussMahDist_3', 'GaussMajor_3', 'GaussAratio_3', 'GaussFlux_3',
                    'GaussMahDist_4', 'GaussMajor_4', 'GaussAratio_4', 'GaussFlux_4']

    print len(features)

    for i, out in enumerate(outliers):
        print i, out, len(features[i]), features[i][0].shape, features[i][1].shape, features[i][2].shape
        df[pc_names].ix[out] = features[i][0]
        df[lda_names].ix[out] = features[i][1]
        df[gauss_labels].ix[out] = features[i][2]

    df.to_hdf(base_dir + 'data/galaxy_features.h5', 'df')

    X = df[pc_names].values
    thresh = 6.0

    row_norm = np.linalg.norm(X - np.median(X, axis=0), axis=1)
    mad = np.median(np.abs(row_norm - np.median(row_norm)))
    robsig = 1.48 * mad
    zscore_new = np.abs(row_norm - np.median(row_norm)) / robsig

    plt.plot(zscore, 'r.', ms=2, label='old')
    plt.plot(zscore_new, 'b.', ms=2, label='new')
    plt.ylabel('PC z-score')
    plt.legend(loc='best')
    plt.show()