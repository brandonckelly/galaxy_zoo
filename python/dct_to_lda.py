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


base_dir = os.environ['HOME'] + '/Projects/Kaggle/galaxy_zoo/'
dct_dir = base_dir + 'data/react/'
plot_dir = base_dir + 'plots/'

doshow = False
verbose = True
do_parallel = False


def plot_lda_projections(X_lda):

    n_components = X_lda.shape[1]
    labels = []
    for i in range(n_components):
        labels.append('LDA ' + str(i+1))

    fig = triangle.corner(X_lda[:, :n_components], labels=labels)
    return fig


def make_lda_images(lda, shape, question):

    n_components = lda.shape[1]
    U = REACT2D.build_dct(shape[0], shape[1], 50)

    lda_images = np.empty((n_components, shape[0], shape[1], 3))

    lda_images[:, :, :, 0] = lda.components_.dot(U.T).reshape((n_components, shape[0], shape[1]))
    lda_images[:, :, :, 1] = lda.components_.dot(U.T).reshape((n_components, shape[0], shape[1]))
    lda_images[:, :, :, 2] = lda.components_.dot(U.T).reshape((n_components, shape[0], shape[1]))

    for i in range(n_components):
        plt.clf()
        plt.imshow(bytescale(lda_images[i, :, :, :]))
        question_idx = str(int(question)) + '.' + str(i+1)
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

    make_lda_images(lda, (100, 100), question)
    fig = plot_lda_projections(X_lda)
    fig.savefig(plot_dir + 'LDA_dist_no_outliers_' + str(question) + '.png')
    if doshow:
        plt.show()

    return X_lda


def remove_outliers(X):
    # find the outliers
    if verbose:
        print 'Finding the outliers...'
    row_norm = np.linalg.norm(X - np.median(X, axis=0), axis=1)
    mad = np.median(np.abs(row_norm - np.median(row_norm)))
    robsig = 1.48 * mad
    zscore = np.abs(row_norm - np.median(row_norm)) / robsig
    notout = np.where(zscore < 10)[0]
    print 'Found', X.shape[0] - len(notout), 'outliers'
    return X[notout, :]


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


    if verbose:
        print 'Loading the data...'
    X = np.load(base_dir + 'data/DCT_array_all.npy')

    X = remove_outliers(X)

    questions = range(1, 12)

    y = pd.read_csv(base_dir + 'data/training_solutions_rev1.csv').set_index('GalaxyID')
    # do LDA for each question
    for question in questions:
        q_cols = []
        for col in y.columns:
            # find columns corresponding to this question
            if str(question) == col.split('Class')[1].split('.')[0]:
                q_cols.append(col)
        print 'Found columns', q_cols
        X_lda_q = lda_transform(X, y[q_cols].values, question)
        if question == 1:
            X_lda = X_lda_q
        else:
            X_lda = np.hstack((X_lda, X_lda_q))

    print 'Saving the transformed values...'
    np.save(base_dir + 'data/LDA_training_transform', X_lda)