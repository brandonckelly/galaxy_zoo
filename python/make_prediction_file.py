__author__ = 'brandonkelly'

import pandas as pd
import numpy as np
import os
import cPickle
import glob
import sys
import theanets
import theano.tensor as T


base_dir = os.environ['HOME'] + '/Projects/Kaggle/galaxy_zoo/'
plot_dir = base_dir + 'plots/'
dct_dir = base_dir + 'data/react/'
training_dir = base_dir + 'data/images_training_rev1/'
gbt_dir = base_dir + 'data/gbt/'

do_nnets = True
do_gbr = True

unique_cols = ['Class1.1', 'Class1.3', 'Class2.1', 'Class3.1', 'Class4.1', 'Class5.1', 'Class5.2', 'Class5.4',
               'Class6.1', 'Class7.1', 'Class7.3', 'Class8.1', 'Class8.2', 'Class8.3', 'Class8.4', 'Class8.6',
               'Class8.7', 'Class9.2', 'Class9.3', 'Class10.2', 'Class10.3', 'Class11.1', 'Class11.2',
               'Class11.3', 'Class11.4', 'Class11.5']


def write_predictions(y_predict, base_name):

    norm = y_predict[['Class1.1', 'Class1.2', 'Class1.3']].sum(axis=1)
    y_predict['Class1.1'] /= norm
    y_predict['Class1.2'] /= norm
    y_predict['Class1.3'] /= norm
    zeros = (norm == 0)
    y_predict['Class1.1'][zeros] = 0
    y_predict['Class1.2'][zeros] = 0
    y_predict['Class1.3'][zeros] = 0

    norm = y_predict[['Class2.1', 'Class2.2']].sum(axis=1)
    y_predict['Class2.1'] /= norm / y_predict['Class1.2']
    y_predict['Class2.2'] /= norm / y_predict['Class1.2']
    zeros = np.logical_or(norm == 0, y_predict['Class1.2'] == 0)
    y_predict['Class2.1'][zeros] = 0
    y_predict['Class2.2'][zeros] = 0

    norm = y_predict[['Class3.1', 'Class3.2']].sum(axis=1)
    y_predict['Class3.1'] /= norm / y_predict['Class2.2']
    y_predict['Class3.2'] /= norm / y_predict['Class2.2']
    zeros = np.logical_or(norm == 0, y_predict['Class2.2'] == 0)
    y_predict['Class3.1'][zeros] = 0
    y_predict['Class3.2'][zeros] = 0

    norm = y_predict[['Class4.1', 'Class4.2']].sum(axis=1)
    y_predict['Class4.1'] /= norm / y_predict['Class2.2']
    y_predict['Class4.2'] /= norm / y_predict['Class2.2']
    zeros = np.logical_or(norm == 0, y_predict['Class2.2'] == 0)
    y_predict['Class4.1'][zeros] = 0
    y_predict['Class4.2'][zeros] = 0

    norm = y_predict[['Class5.1', 'Class5.2', 'Class5.3', 'Class5.4']].sum(axis=1)
    y_predict['Class5.1'] /= norm / y_predict['Class2.2']
    y_predict['Class5.2'] /= norm / y_predict['Class2.2']
    y_predict['Class5.3'] /= norm / y_predict['Class2.2']
    y_predict['Class5.4'] /= norm / y_predict['Class2.2']
    zeros = np.logical_or(norm == 0, y_predict['Class2.2'] == 0)
    y_predict['Class5.1'][zeros] = 0
    y_predict['Class5.2'][zeros] = 0
    y_predict['Class5.3'][zeros] = 0
    y_predict['Class5.4'][zeros] = 0

    norm = y_predict[['Class6.1', 'Class6.2']].sum(axis=1)
    y_predict['Class6.1'] /= norm
    y_predict['Class6.2'] /= norm
    zeros = (norm == 0)
    y_predict['Class6.1'][zeros] = 0
    y_predict['Class6.2'][zeros] = 0

    norm = y_predict[['Class7.1', 'Class7.2', 'Class7.3']].sum(axis=1)
    y_predict['Class7.1'] /= norm / y_predict['Class1.1']
    y_predict['Class7.2'] /= norm / y_predict['Class1.1']
    y_predict['Class7.3'] /= norm / y_predict['Class1.1']
    zeros = np.logical_or(norm == 0, y_predict['Class1.1'] == 0)
    y_predict['Class7.1'][zeros] = 0
    y_predict['Class7.2'][zeros] = 0
    y_predict['Class7.3'][zeros] = 0

    norm = y_predict[['Class8.1', 'Class8.2', 'Class8.3', 'Class8.4',
                      'Class8.5', 'Class8.6', 'Class8.7']].sum(axis=1)
    y_predict['Class8.1'] /= norm / y_predict['Class6.1']
    y_predict['Class8.2'] /= norm / y_predict['Class6.1']
    y_predict['Class8.3'] /= norm / y_predict['Class6.1']
    y_predict['Class8.4'] /= norm / y_predict['Class6.1']
    y_predict['Class8.5'] /= norm / y_predict['Class6.1']
    y_predict['Class8.6'] /= norm / y_predict['Class6.1']
    y_predict['Class8.7'] /= norm / y_predict['Class6.1']
    zeros = np.logical_or(norm == 0, y_predict['Class6.1'] == 0)
    y_predict['Class8.1'][zeros] = 0
    y_predict['Class8.2'][zeros] = 0
    y_predict['Class8.3'][zeros] = 0
    y_predict['Class8.4'][zeros] = 0
    y_predict['Class8.5'][zeros] = 0
    y_predict['Class8.6'][zeros] = 0
    y_predict['Class8.7'][zeros] = 0

    norm = y_predict[['Class9.1', 'Class9.2', 'Class9.3']].sum(axis=1)
    y_predict['Class9.1'] /= norm / y_predict['Class2.1']
    y_predict['Class9.2'] /= norm / y_predict['Class2.1']
    y_predict['Class9.3'] /= norm / y_predict['Class2.1']
    zeros = np.logical_or(norm == 0, y_predict['Class2.1'] == 0)
    y_predict['Class9.1'][zeros] = 0
    y_predict['Class9.2'][zeros] = 0
    y_predict['Class9.3'][zeros] = 0

    norm = y_predict[['Class10.1', 'Class10.2', 'Class10.3']].sum(axis=1)
    y_predict['Class10.1'] /= norm / y_predict['Class4.1']
    y_predict['Class10.2'] /= norm / y_predict['Class4.1']
    y_predict['Class10.3'] /= norm / y_predict['Class4.1']
    zeros = np.logical_or(norm == 0, y_predict['Class4.1'] == 0)
    y_predict['Class10.1'][zeros] = 0
    y_predict['Class10.2'][zeros] = 0
    y_predict['Class10.3'][zeros] = 0

    norm = y_predict[['Class11.1', 'Class11.2', 'Class11.3', 'Class11.4',
                      'Class11.5', 'Class11.6']].sum(axis=1)
    y_predict['Class11.1'] /= norm / y_predict['Class4.1']
    y_predict['Class11.2'] /= norm / y_predict['Class4.1']
    y_predict['Class11.3'] /= norm / y_predict['Class4.1']
    y_predict['Class11.4'] /= norm / y_predict['Class4.1']
    y_predict['Class11.5'] /= norm / y_predict['Class4.1']
    y_predict['Class11.6'] /= norm / y_predict['Class4.1']
    zeros = np.logical_or(norm == 0, y_predict['Class4.1'] == 0)
    y_predict['Class11.1'][zeros] = 0
    y_predict['Class11.2'][zeros] = 0
    y_predict['Class11.3'][zeros] = 0
    y_predict['Class11.4'][zeros] = 0
    y_predict['Class11.5'][zeros] = 0
    y_predict['Class11.6'][zeros] = 0

    y_predict[y_predict > 1] = 1.0
    y_predict[y_predict < 0] = 0.0

    assert np.all(np.isfinite(y_predict))

    # dump to CSV file
    y_predict.index.name = 'GalaxyID'
    y_predict.to_csv(base_dir + 'data/' + base_name + '_predictions.csv')


def get_unique_cols_complement(y_predict):
    # calculate remaining classes using constraints
    y_predict['Class1.2'] = 1.0 - y_predict['Class1.1'] - y_predict['Class1.3']
    y_predict['Class2.2'] = y_predict['Class1.2'] - y_predict['Class2.1']
    y_predict['Class3.2'] = y_predict['Class2.2'] - y_predict['Class3.1']
    y_predict['Class4.2'] = y_predict['Class2.2'] - y_predict['Class4.1']
    y_predict['Class5.3'] = y_predict['Class2.2'] - y_predict['Class5.1'] - y_predict['Class5.2'] - \
                            y_predict['Class5.4']
    y_predict['Class6.2'] = 1.0 - y_predict['Class6.1']
    y_predict['Class7.2'] = y_predict['Class1.1'] - y_predict['Class7.1'] - y_predict['Class7.3']
    y_predict['Class8.5'] = y_predict['Class6.1'] - y_predict['Class8.1'] - y_predict['Class8.2'] - \
                            y_predict['Class8.3'] - y_predict['Class8.4'] - y_predict['Class8.6'] - \
                            y_predict['Class8.7']
    y_predict['Class9.1'] = y_predict['Class2.1'] - y_predict['Class9.2'] - y_predict['Class9.3']
    y_predict['Class10.1'] = y_predict['Class4.1'] - y_predict['Class10.2'] - y_predict['Class10.3']
    y_predict['Class11.6'] = y_predict['Class4.1'] - y_predict['Class11.1'] - y_predict['Class11.2'] - \
                             y_predict['Class11.3'] - y_predict['Class11.4'] - y_predict['Class11.5']

    return y_predict


def ann_predict(df, classes, do_unique=True):

    if do_unique:
        cols = unique_cols
    else:
        cols = classes

    print 'Predicting values for ANN...'

    def clean_features(df):

        for color in ['blue', 'green', 'red']:
            df[color].ix[df[color] == -9999] = df[color].median()

        df['GalaxyCentDist'].ix[df['GalaxyCentDist'] == -9999] = -0.5

        # standardize inputs
        mad = (df - df.median()).abs().median()
        df -= df.median()
        df /= 1.5 * mad

        return df

    df_clean = clean_features(df.copy())

    ann_files = ['ANN_SGD_L2-0.0_arch-1000-1000-1000_L1-0.0_learnrate0p01_trial11.pickle',
                 'ANN_SGD_L2-0.0_arch-1000-1000-1000_L1-0.0_learnrate0p01_trial22.pickle',
                 'ANN_SGD_L2-0.0_arch-1000-1000-1000_L1-0.0_learnrate0p01_trial12.pickle',
                 'ANN_SGD_L2-0.0_arch-1000-1000-1000_L1-0.0_learnrate0p01_trial31.pickle',
                 'ANN_SGD_L2-0.0_arch-1000-1000-1000_L1-0.0_learnrate0p01.pickle',
                 'ANN_SGD_L2-0.0_arch-1000-1000-1000_L1-0.0_learnrate0p01_trial21.pickle',
                 'ANN_SGD_L2-0.0_arch-1000-1000-1000_L1-0.0_learnrate0p01_trial32.pickle']

    ann_prediction = 0.0
    for f in ann_files:
        ann = theanets.Network((df.shape[1]-16, 1000, 1000, 1000, len(cols)), lambda z: T.maximum(0, z))
        print f
        ann.load(base_dir + 'data/nnets/' + f)
        ann_prediction += ann.predict(df_clean[df_clean.columns[:-16]].values) / len(ann_files)

    y_predict = pd.DataFrame(data=ann_prediction, index=df.index, columns=cols)
    y_predict.index.name = 'GalaxyID'
    y_predict[y_predict < 0] = 0.0
    y_predict[y_predict > 1] = 1.0

    if do_unique:
        y_predict = get_unique_cols_complement(y_predict)
        y_predict[y_predict < 0] = 0.0
        y_predict[y_predict > 1] = 1.0

    return y_predict


def gbr_predict(df, classes, do_unique=False):

    print 'Predicting values for GBR...'

    gbt_files = glob.glob(gbt_dir + '*depth6.pickle')

    if do_unique:
        cols = unique_cols
    else:
        cols = classes

    gbt_list = []
    for c in cols:
        for f in gbt_files:
            if c in f:
                gbt_list.append(f)

    y_predict = np.zeros((len(df), len(cols)))
    print 'Loading GBT pickle for'
    for j, gbt_file in enumerate(gbt_list):
        print gbt_file
        gbt = cPickle.load(open(gbt_file, 'rb'))
        y_predict[:, j] = gbt.predict(df.values)

    y_predict = pd.DataFrame(data=y_predict, index=df.index, columns=cols)
    y_predict.index.name = 'GalaxyID'
    y_predict[y_predict < 0] = 0.0
    y_predict[y_predict > 1] = 1.0

    if do_unique:
        y_predict = get_unique_cols_complement(y_predict)
        y_predict[y_predict < 0] = 0.0
        y_predict[y_predict > 1] = 1.0

    return y_predict


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print 'Syntax: python make_prediction_file.py base_name'
        exit()

    base_name = sys.argv[1]

    y = pd.read_csv(base_dir + 'data/training_solutions_rev1.csv').set_index('GalaxyID')
    classes = y.columns

    files = glob.glob(base_dir + 'data/images_test_rev1/*.jpg')
    test_ids = [int(f.split('/')[-1].split('.')[0]) for f in files]

    # load the test data for the features
    df = pd.read_hdf(base_dir + 'data/galaxy_features.h5', 'df')
    df = df.ix[test_ids]

    assert np.all(np.isfinite(df.values))

    print 'Found', len(test_ids), 'galaxies with test labels.'

    print 'Predicting the values...'

    if do_nnets:
        y_predict = ann_predict(df, classes, do_unique=False)
    if do_gbr:
        g_predict = gbr_predict(df, classes, do_unique=False)
        if do_nnets:
            assert y_predict.shape == g_predict.shape
            y_predict = 0.5 * (y_predict + g_predict)
        else:
            y_predict = g_predict

    assert np.all(np.isfinite(y_predict.values))

    write_predictions(y_predict, base_name)