__author__ = 'brandonkelly'

import pandas as pd
import numpy as np
import cPickle
import os
import glob
from PIL import Image
import multiprocessing
from scipy import linalg
import datetime

base_dir = os.environ['HOME'] + '/Projects/Kaggle/galaxy_zoo/'
plot_dir = base_dir + 'plots/'
dct_dir = base_dir + 'data/react/'
test_dir = base_dir + 'data/images_test_rev1/'
train_dir = base_dir + 'data/images_training_rev1/'

doshow = False
verbose = False
do_parallel = True
npcs = 200


def logit(x):
    return np.log(x / (1.0 - x))


def get_central_pixel_colors(galaxy_id):
    if verbose:
        print galaxy_id
    # find which directory galaxy is in
    train_file = base_dir + 'data/images_training_rev1/' + str(galaxy_id) + '.jpg'
    test_file = base_dir + 'data/images_test_rev1/' + str(galaxy_id) + '.jpg'
    if os.path.isfile(train_file):
        file = train_file
    else:
        file = test_file

    im = np.array(Image.open(file)).astype(float)
    ndim = im.shape

    blue = np.log(im[ndim[0]/2, ndim[0]/2, 0]) - np.log(im[ndim[0]/2, ndim[1]/2, 1])
    green = np.log(im[ndim[0]/2, ndim[0]/2, 1]) - np.log(im[ndim[0]/2, ndim[1]/2, 2])
    red = np.log(im[ndim[0]/2, ndim[0]/2, 0]) - np.log(im[ndim[0]/2, ndim[1]/2, 2])

    return blue, green, red


def make_gaussfit_features(galaxy_id):
    if verbose:
        print galaxy_id
    gfit = pd.read_csv(base_dir + 'data/gauss_fit/transfer/' + str(galaxy_id) + '_gauss_params.csv').set_index('GaussianID')

    # first get features for gaussian corresponding to galaxy
    if 'Band' in gfit.columns:
        gfit = gfit[gfit['Band'] == 2]
    gal = gfit.ix[0]
    gal_major = np.log(gal['amajor'])
    gal_aratio = logit(gal['aminor'] / gfit.ix[0]['amajor'])
    gal_cent_distance = 0.5 * np.log((gal['xcent'] - 212.0) ** 2 + (gal['ycent'] - 212.0) ** 2)
    gal_flux = np.log(2.0 * np.pi * gal['amplitude'] * gal['amajor'] * gal['aminor'])

    features = np.asarray([gal_cent_distance, gal_major, gal_aratio, gal_flux])

    # now get features corresponding to gaussian for other modes
    gal_centroid = np.array([gal['xcent'], gal['ycent']])
    gal_covar = np.zeros((2, 2))
    gal_covar[0, 0] = gal['xsigma'] ** 2
    gal_covar[1, 1] = gal['ysigma'] ** 2
    gal_covar[0, 1] = gal['rho'] * np.sqrt(gal_covar[0, 0] * gal_covar[1, 1])
    gal_covar[1, 0] = gal_covar[0, 1]
    mah_distance = 1000.0 * np.ones(4)
    for i in range(1, len(gfit)):
        # first get mahalonobis distance and sort by this
        this_centroid = np.array([gfit.ix[i]['xcent'], gfit.ix[i]['ycent']])
        centdiff = this_centroid - gal_centroid
        covar = np.zeros((2, 2))
        covar[0, 0] = gfit.ix[i]['xsigma'] ** 2
        covar[1, 1] = gfit.ix[i]['ysigma'] ** 2
        covar[0, 1] = gfit.ix[i]['rho'] * np.sqrt(covar[0, 0] * covar[1, 1])
        covar[1, 0] = covar[0, 1]
        mah_distance[i-1] = np.sqrt(np.sum(centdiff * np.dot(linalg.inv(covar + gal_covar), centdiff)))

    assert np.all(np.isfinite(mah_distance))

    s_idx = mah_distance.argsort()
    for idx in s_idx:
        if idx in gfit.index:
            this_major = np.log(gfit.ix[idx]['amajor'])
            this_aratio = logit(gfit.ix[idx]['aminor'] / gfit.ix[idx]['amajor'])
            this_flux = np.log(2.0 * np.pi * gfit.ix[idx]['amplitude'] *
                               gfit.ix[idx]['amajor'] * gfit.ix[idx]['aminor'])
        else:
            this_major = -9999.0
            this_aratio = -9999.0
            this_flux = -9999.0
        these_features = np.asarray([np.log(mah_distance[idx]), this_major, this_aratio, this_flux])
        features = np.append(features, these_features)

    return features

if __name__ == "__main__":
    pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
    pool.map(int, range(multiprocessing.cpu_count()-1))

    # find which galaxies we have a full dct for
    files_0 = glob.glob(dct_dir + '*_0_dct.pickle')
    files_1 = glob.glob(dct_dir + '*_1_dct.pickle')
    files_2 = glob.glob(dct_dir + '*_2_dct.pickle')

    gfiles = glob.glob(base_dir + 'data/gauss_fit/transfer/*.csv')

    galaxy_ids_0 = set([f.split('/')[-1].split('_')[0] for f in files_0])
    galaxy_ids_1 = set([f.split('/')[-1].split('_')[0] for f in files_1])
    galaxy_ids_2 = set([f.split('/')[-1].split('_')[0] for f in files_2])
    galaxy_ids_3 = set([f.split('/')[-1].split('_')[0] for f in gfiles])

    galaxy_ids = galaxy_ids_0 & galaxy_ids_1 & galaxy_ids_2 & galaxy_ids_3
    del galaxy_ids_0, galaxy_ids_1, galaxy_ids_2, galaxy_ids_3

    if verbose:
        print "Found", len(galaxy_ids), "galaxies."

    # get galaxy ids
    test_gals = glob.glob(test_dir + '*.jpg')
    test_ids = set([f.split('/')[-1].split('.')[0] for f in test_gals])
    train_gals = glob.glob(train_dir + '*.jpg')
    train_ids = set([f.split('/')[-1].split('.')[0] for f in train_gals])

    galaxy_ids_ref = test_ids | train_ids

    if len(galaxy_ids_ref - galaxy_ids) != 0:
        print 'Missing data for the following galaxies:'
        print galaxy_ids_ref - galaxy_ids
        exit()

    galaxy_ids = list(galaxy_ids)

    # load the DCT coefficients
    if verbose:
        print 'Loading DCT coefficients...'
    dct_coefs = np.load(base_dir + 'data/DCT_array_all.npy').astype(np.float32)

    # first add principal components
    if verbose:
        print 'Doing PC Transform for chunk...'
    rpca = cPickle.load(open(base_dir + 'data/DCT_PCA.pickle', 'rb'))
    dct_coefs -= rpca.mean_
    # do PCA transform in chunks of 10,000 data points to save memory
    ndata = dct_coefs.shape[0]
    chunk_first = 0
    chunk_last = 10000
    X = np.empty((ndata, npcs), dtype=np.float32)
    chunk = 1
    while chunk_last < ndata:
        if verbose:
            print chunk
        X[chunk_first:chunk_last] = np.dot(dct_coefs[chunk_first:chunk_last], rpca.components_[:npcs].T)
        chunk_first = chunk_last
        chunk_last += 10000
        chunk += 1
    # add the mean back in before doing LDA
    dct_coefs += rpca.mean_
    del rpca
    nfeatures = X.shape[1]
    pc_labels = ['PC ' + str(i + 1) for i in range(npcs)]

    # now add LDA directions
    questions = range(1, 12)
    lda_labels = []
    if verbose:
        print 'Doing LDA transform for question'
    for question in questions:
        if verbose:
           print question, '...'
        lda = cPickle.load(open(base_dir + 'data/DCT_LDA_' + str(question) + '.pickle', 'rb'))
        ncoefs = lda.components_.shape[1] / 3
        dct_idx = np.asarray([np.arange(ncoefs), 2500 + np.arange(ncoefs), 5000 + np.arange(ncoefs)]).ravel()
        if question == 1:
            dct_coefs = dct_coefs[:, dct_idx]
        n_components = lda.components_.shape[0]
        X = np.append(X, dct_coefs.dot(lda.components_.T), axis=1)
        lda_labels.extend(['LDA ' + str(question) + '.' + str(i + 1) for i in range(n_components)])
    del lda
    del dct_coefs

    # add the central pixel colors
    if verbose:
        print 'Getting central pixel color...'
    if do_parallel:
        colors = pool.map(get_central_pixel_colors, galaxy_ids)
    else:
        colors = map(get_central_pixel_colors, galaxy_ids)
    colors = np.asarray(colors)
    X = np.append(X, colors, axis=1)
    color_labels = ['blue', 'green', 'red']

    # finally, add the gaussian fit parameters
    if verbose:
        print 'Getting gaussian parameters...'
    if do_parallel:
        gauss_features = pool.map(make_gaussfit_features, galaxy_ids)
    else:
        gauss_features = map(make_gaussfit_features, galaxy_ids)
    X = np.append(X, gauss_features, axis=1)
    gauss_labels = ['GalaxyCentDist', 'GalaxyMajor', 'GalaxyAratio', 'GalaxyFlux',
                    'GaussMahDist_1', 'GaussMajor_1', 'GaussAratio_1', 'GaussFlux_1',
                    'GaussMahDist_2', 'GaussMajor_2', 'GaussAratio_2', 'GaussFlux_2',
                    'GaussMahDist_3', 'GaussMajor_3', 'GaussAratio_3', 'GaussFlux_3',
                    'GaussMahDist_4', 'GaussMajor_4', 'GaussAratio_4', 'GaussFlux_4']

    # now construct the dataframe
    labels = pc_labels
    # labels.extend(lda_labels)
    labels.extend(color_labels)
    labels.extend(gauss_labels)
    if verbose:
        print 'Creating and saving the dataframe...'
    X = pd.DataFrame(data=X, index=galaxy_ids, columns=labels)
    X.to_hdf(base_dir + 'data/galaxy_features.h5', 'df')