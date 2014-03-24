__author__ = 'brandonkelly'

import pandas as pd
import numpy as np
import cPickle
import os
import glob
from PIL import Image
import multiprocessing
from scipy import linalg

base_dir = os.environ['HOME'] + '/Projects/Kaggle/galaxy_zoo/'
plot_dir = base_dir + 'plots/'
dct_dir = base_dir + 'data/react/'
training_dir = base_dir + 'data/images_training_rev1/'

doshow = False
verbose = True
do_parallel = True
npcs = 200


def logit(x):
    return np.log(x / (1.0 - x))


def get_central_pixel_colors(galaxy_id):
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
    gfit = pd.read_csv(base_dir + 'data/gauss_fit/' + str(galaxy_id) + '_gauss_params.csv').set_index('GaussianID')

    # first get features for gaussian corresponding to galaxy
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
    mah_distance = np.zeros(4)
    for i in range(1, 5):
        # first get mahalonobis distance and sort by this
        this_centroid = np.array([gfit.ix[i]['xcent'], gfit.ix[i]['ycent']])
        centdiff = this_centroid - gal_centroid
        covar = np.zeros((2, 2))
        covar[0, 0] = gfit.ix[i]['xsigma'] ** 2
        covar[1, 1] = gfit.ix[i]['ysigma'] ** 2
        covar[0, 1] = gfit.ix[i]['rho'] * np.sqrt(covar[0, 0] * covar[1, 1])
        covar[1, 0] = gal_covar[0, 1]
        mah_distance[i] = np.sqrt(np.sum(centdiff * np.dot(linalg.inv(covar + gal_covar), centdiff)))

    assert np.all(np.isfinite(mah_distance))

    s_idx = mah_distance.argsort()
    for idx in s_idx:
        this_major = np.log(gfit.ix[idx]['amajor'])
        this_aratio = logit(gfit.ix[idx]['aminor'] / gfit.ix[idx]['amajor'])
        this_flux = np.log(2.0 * np.pi * gfit.ix[idx]['amplitude'] * gfit.ix[idx]['amajor'] * gfit.ix[idx]['aminor'])
        these_features = np.asarray([np.log(mah_distance), this_major, this_aratio, this_flux])
        features = np.append(features, these_features)

    return features

if __name__ == "__main__":
    pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
    pool.map(int, range(multiprocessing.cpu_count()-1))

    # find which galaxies we have a full dct for
    files_0 = glob.glob(dct_dir + '*_0_dct.pickle')
    files_1 = glob.glob(dct_dir + '*_1_dct.pickle')
    files_2 = glob.glob(dct_dir + '*_2_dct.pickle')

    galaxy_ids_0 = set([f.split('/')[-1].split('_')[0] for f in files_0])
    galaxy_ids_1 = set([f.split('/')[-1].split('_')[0] for f in files_1])
    galaxy_ids_2 = set([f.split('/')[-1].split('_')[0] for f in files_2])

    galaxy_ids = list(galaxy_ids_0 & galaxy_ids_1 & galaxy_ids_2)
    if verbose:
        print "Found", len(galaxy_ids), "galaxies."

    # load the DCT coefficients
    dct_coefs = np.load(base_dir + 'data/DCT_array_all.npy').astype(np.float32)

    # first add principal components
    rpca = cPickle.load(open(base_dir + 'data/DCT_PCA.pickle', 'rb'))
    X = rpca.transform(dct_coefs)[:, :npcs]
    nfeatures = X.shape[1]

    # now add LDA directions
    lda = cPickle.load(open(base_dir + 'data/DCT_LDA.pickle', 'rb'))
    X = np.append(X, lda.transform(dct_coefs), axis=1)
    nlda = X.shape[1] - nfeatures
    nfeatures = X.shape[1]

    # add the CCA directions
    cca = cPickle.load(open(base_dir + 'data/DCT_CCA.pickle', 'rb'))
    X = np.append(X, cca.transform(dct_coefs), axis=1)
    ncca = X.shape[1] - nfeatures
    nfeatures = X.shape[1]

    # add the central pixel colors
    colors = pool.map(get_central_pixel_colors, galaxy_ids)
    colors = np.asarray(colors)
    X = np.append(X, colors, axis=1)

    # finally, add the gaussian fit parameters
    gauss_features = pool.map(make_gaussfit_features, galaxy_ids)
    X = np.append(X, gauss_features, axis=1)

    # now construct the dataframe
    pc_labels = ['PC ' + str(i + 1) for i in range(npcs)]
    lda_labels = ['LDA ' + str(i + 1) for i in range(nlda)]
    cca_labels = ['CCA ' + str(i + 1) for i in range(ncca)]
    color_labels = ['blue', 'green', 'red']
    gauss_labels = ['GalaxyCentDist', 'GalaxyMajor', 'GalaxyAratio', 'GalaxyFlux',
                    'GaussMahDist_1', 'GaussMajor_1', 'GaussAratio_1', 'GaussFlux_1',
                    'GaussMahDist_2', 'GaussMajor_2', 'GaussAratio_2', 'GaussFlux_2',
                    'GaussMahDist_3', 'GaussMajor_3', 'GaussAratio_3', 'GaussFlux_3',
                    'GaussMahDist_4', 'GaussMajor_4', 'GaussAratio_4', 'GaussFlux_4']

    labels = pc_labels
    labels.extend(lda_labels)
    labels.extend(cca_labels)
    labels.extend(color_labels)
    labels.extend(gauss_labels)

    X = pd.DataFrame(data=X, index=galaxy_ids, columns=labels)
    X.to_hdf(base_dir + 'data/galaxy_features.h5', 'df')