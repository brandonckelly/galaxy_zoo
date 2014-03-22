__author__ = 'brandonkelly'

import numpy as np
import cPickle
import os
import matplotlib.pyplot as plt
import glob
from react import REACT2D
import triangle
from scipy.misc import bytescale
from probabilistic_lda import ProbabilisticLDA
import pandas as pd
from multiclass_triangle_plot import multiclass_triangle

base_dir = os.environ['HOME'] + '/Projects/Kaggle/galaxy_zoo/'
dct_dir = base_dir + 'data/react/'
plot_dir = base_dir + 'plots/'

doshow = False
verbose = True
do_parallel = False


def plot_lda_projections(X_lda, probs):

    classes = probs.argmax(axis=1)

    n_components = X_lda.shape[1]
    labels = []
    for i in range(n_components):
        labels.append('LDA ' + str(i+1))

    fig = multiclass_triangle(X_lda, classes, labels=labels, verbose=verbose)
    return fig


def make_lda_images(lda, shape, question, dct_idx=None):

    n_components = lda.components_.shape[0]
    U = REACT2D.build_dct(shape[0], shape[1], 50)

    if dct_idx is not None:
        dct_idx = np.arange(2500)
    dct_idx = np.arange(2499)
    U = U[:, dct_idx]

    lda_images = np.empty((n_components, shape[0], shape[1], 3))

    # lda.components_.shape = (ncomponents, nfeatures)
    print 'LDA components shape:', lda.components_.shape
    lda_images[:, :, :, 0] = \
        lda.components_[:, dct_idx].dot(U.T).reshape((n_components, shape[0], shape[1]))
    lda_images[:, :, :, 1] = \
        lda.components_[:, dct_idx + len(dct_idx)].dot(U.T).reshape((n_components, shape[0], shape[1]))
    lda_images[:, :, :, 2] = \
        lda.components_[:, dct_idx + 2*len(dct_idx)].dot(U.T).reshape((n_components, shape[0], shape[1]))

    for i in range(n_components):
        plt.clf()
        plt.imshow(bytescale(lda_images[i, :, :, :]))
        question_idx = str(int(question)) + '-' + str(i+1)
        plt.title('LDA ' + question_idx)
        plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        plt.tick_params(axis='y', which='both', bottom='off', top='off', labelbottom='off')
        plt.savefig(plot_dir + 'LDA_Images_' + question_idx + '.png')
        if doshow:
            plt.show()


def lda_transform(X, y, question):

    print 'Doing LDA for question', question, '...'
    lda = ProbabilisticLDA()
    lda.question = question
    X_lda = lda.fit_transform(X, y)

    cPickle.dump(lda, open(base_dir + 'data/DCT_LDA_' + str(question) + '.pickle', 'wb'))

    return X_lda, lda


def remove_outliers(X, thresh=10.0):
    # find the outliers
    if verbose:
        print 'Finding the outliers...'
    row_norm = np.linalg.norm(X - np.median(X, axis=0), axis=1)
    mad = np.median(np.abs(row_norm - np.median(row_norm)))
    robsig = 1.48 * mad
    zscore = np.abs(row_norm - np.median(row_norm)) / robsig
    notout = np.where(zscore < thresh)[0]
    print 'Found', X.shape[0] - len(notout), 'outliers'
    return X[notout, :], notout


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

    zero_idx = np.where(np.all(X == 0, axis=0))[0]  # remove columns with all zeros
    if len(zero_idx) > 0:
        if verbose:
            print 'Detected the following columns as having all zeros and removed them:'
            print zero_idx
        dct_idx = np.where(np.all(X[:, :2500] != 0, axis=0))[0]
        X = np.delete(X, zero_idx, axis=1)
    else:
        dct_idx = None

    # remove outliers
    X, good_idx = remove_outliers(X, thresh=6.0)
    y = y.ix[y.index[good_idx]]

    questions = range(1, 12)

    # do LDA for each question
    for question in questions:
        q_cols = []
        for col in y.columns:
            # find columns corresponding to this question
            if str(question) == col.split('Class')[1].split('.')[0]:
                q_cols.append(col)
        print 'Found columns', q_cols
        probs = y[q_cols].values.copy()
        norm = probs.sum(axis=1)
        probs /= norm[:, np.newaxis]
        assert np.all(np.isfinite(X[norm > 0]))
        assert np.all(np.isfinite(probs[norm > 0]))
         # don't include gals that never made it to this node
        X_lda_q, lda = lda_transform(X[norm > 0], probs[norm > 0], question)
        # make plots
        fig = plot_lda_projections(X_lda_q, probs[norm > 0])
        fig.savefig(plot_dir + 'LDA_dist_no_outliers_' + str(question) + '.png')
        if doshow:
            plt.show()
        make_lda_images(lda, (100, 100), question, dct_idx=dct_idx)

        if question == 1:
            X_lda = X_lda_q
        else:
            X_lda_q = lda.transform(X)  # need to transform all of the gals, not just those in this node
            X_lda = np.hstack((X_lda, X_lda_q))

    print 'Saving the transformed values...'
    np.save(base_dir + 'data/LDA_training_transform', X_lda)