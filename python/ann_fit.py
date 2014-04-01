#!/usr/bin/env python

__author__ = 'brandonkelly'

import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import pandas as pd
import climate
import theanets
import glob
from randomforest_fit import get_err
from make_prediction_file import write_rf_predictions
from dct_to_lda import remove_outliers

base_dir = os.environ['HOME'] + '/Projects/Kaggle/galaxy_zoo/'
data_dir = base_dir + 'data/'
dct_dir = data_dir + 'react/'
ann_dir = data_dir + 'nnets/'
plot_dir = base_dir + 'plots/'

doshow = False
verbose = True

climate.enable_default_logging()


def clean_features(df):

    for color in ['blue', 'green', 'red']:
        df[color].ix[df[color] == -9999] = df[color].median()

    df['GalaxyCentDist'].ix[df['GalaxyCentDist'] == -9999] = -0.5

    # standardize inputs
    mad = (df - df.median()).abs().median()
    df -= df.median()
    df /= 1.5 * mad

    return df


def train_ann(X, y, l2_reg, l1_reg=0.0):

    train_set_x, valid_set_x, train_set_y, valid_set_y = train_test_split(X, y, random_state=1234)

    n_hidden = 1000
    layers = (X.shape[1], n_hidden, y.shape[1])
    experiment = theanets.Experiment(theanets.Regressor, layers=layers, train_batches=100, weight_l2=l2_reg,
                                     hidden_l2=l2_reg, weight_l1=l1_reg, hidden_l1=l1_reg)

    # experiment.add_dataset('train', (train_set_x, train_set_y))
    # experiment.add_dataset('valid', (valid_set_x, valid_set_y))

    experiment.run(train=(train_set_x, train_set_y), valid=(valid_set_x, valid_set_y))

    return experiment.network


if __name__ == "__main__":

    # load the training labels
    if verbose:
        print 'Loading training labels...'
    y = pd.read_csv(base_dir + 'data/training_solutions_rev1.csv').set_index('GalaxyID')

    if not np.all(np.isfinite(y)):
        print 'Error! Non-finite training solutions detected.'
        exit()

    print 'Found', len(y), 'galaxies with training labels.'

    # load the training data for the features
    df = pd.read_hdf(base_dir + 'data/galaxy_features.h5', 'df')

    df = df[df.columns[:-16]]  # omit features on nearby objects

    if len(y.index - df.index) > 0:
        print 'Error! Missing training data in feature dataframe.'
        exit()

    files = glob.glob(base_dir + 'data/images_test_rev1/*.jpg')
    test_set = [int(f.split('/')[-1].split('.')[0]) for f in files]

    assert np.all(np.isfinite(df.values))

    print 'Found', len(test_set), 'galaxies with test labels.'

    train_set = y.index

    print 'Cleaning the data...'
    df = clean_features(df)

    # remove outliers
    pc_names = []
    for c in df.columns:
        if 'PC' in c:
            pc_names.append(c)

    X, good_idx = remove_outliers(df.ix[train_set][pc_names].values, thresh=6.0)
    df_train = df.ix[train_set[good_idx]]
    y = y.ix[train_set[good_idx]]

    if not np.all(np.isfinite(df.ix[train_set])):
        print 'Error! Non-finite feature values detected in training set.'
    if not np.all(np.isfinite(df.ix[test_set])):
        print 'Error! Non-finite feature values detected in test set.'

    unique_cols = ['Class1.1', 'Class1.3', 'Class2.1', 'Class3.1', 'Class4.1', 'Class5.1', 'Class5.2', 'Class5.4',
                   'Class6.1', 'Class7.1', 'Class7.3', 'Class8.1', 'Class8.2', 'Class8.3', 'Class8.4', 'Class8.6',
                   'Class8.7', 'Class9.2', 'Class9.3', 'Class10.2', 'Class10.3', 'Class11.1', 'Class11.2',
                   'Class11.3', 'Class11.4', 'Class11.5']

    t1 = time.clock()

    train_set, valid_set = train_test_split(df_train.index, random_state=1234)

    print 'Using a training set of', len(train_set), 'and a validation set of', len(valid_set)

    l2_regs = [0.0005, 0.0007, 0.0009, 0.0011, 0.0013, 0.0015]
    valerr = []

    for l2reg in l2_regs:

        ann_id = 'SGD_L2-' + str(l2reg) + '_arch-1000.pickle'

        print 'Training the ANN...'
        ann = train_ann(df_train.ix[train_set].values, y[unique_cols].ix[train_set].values, l2reg)

        t2 = time.clock()

        print 'Took', (t2 - t1) / 60.0, 'minutes.'

        ann.save(ann_dir + 'ANN_' + ann_id + '.pickle')

        # first get training error
        yhat = ann.predict(df_train.ix[train_set].values)
        yhat[yhat < 0] = 0.0
        yhat[yhat > 1] = 1.0
        yhat = pd.DataFrame(data=yhat, index=y.ix[train_set].index, columns=unique_cols)

        train_err = get_err(y.ix[train_set], yhat)

        print 'Training error:', np.sqrt(np.mean(train_err.values ** 2))

        # now get validation error
        yhat = ann.predict(df_train.ix[valid_set].values)
        yhat[yhat < 0] = 0.0
        yhat[yhat > 1] = 1.0
        yhat = pd.DataFrame(data=yhat, index=y.ix[valid_set].index, columns=unique_cols)

        valid_err = get_err(y.ix[valid_set], yhat)

        print 'Validation error:', np.sqrt(np.mean(valid_err.values ** 2))

        yfit = ann.predict(df.ix[test_set].values)
        write_rf_predictions(yfit, test_set, ann_id)

    plt.plot(l2_regs, valerr, lw=2)
    plt.ylabel('Validation Error')
    plt.xlabel('L2 Regularization')
    plt.title(ann_id.split('.pickle'[0]))
    plt.savefig(ann_id.split('.pickle')[0] + '.png')
    plt.show()