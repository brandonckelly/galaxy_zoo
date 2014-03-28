__author__ = 'brandonkelly'

__author__ = 'brandonkelly'

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from sklearn.ensemble import GradientBoostingRegressor
import cPickle
import pandas as pd
import multiprocessing


base_dir = os.environ['HOME'] + '/Projects/Kaggle/galaxy_zoo/'
data_dir = base_dir + 'data/'
dct_dir = data_dir + 'react/'
gbt_dir = data_dir + 'gbt/'
plot_dir = base_dir + 'plots/'

doshow = False
verbose = True
use_constraints = False
do_parallel = True


def train_gbt(args):

    X, y, question, depth, feature_labels, ntrees = args

    # first find optimal number of trees
    if ntrees is None:
        ntrees = 1000
    if verbose:
        print 'Training gradient boosted tree for question', question, 'using', ntrees, 'trees.'
    gbt = GradientBoostingRegressor(n_estimators=ntrees, subsample=0.5, max_depth=depth)
    gbt.fit(X, y)
    oob_score = gbt.oob_improvement_.cumsum()

    if verbose:
        print 'Optimal number of trees is', oob_score.argmax() + 1, 'for question', question
    plt.clf()
    plt.plot(oob_score)
    plt.ylabel('OOB Score')
    plt.xlabel('Number of Trees')
    plt.title(question)
    plt.savefig(plot_dir + 'OOB_Score_GBT_' + question + '.png')
    if doshow:
        plt.show()
        plt.close()

    ntrees = oob_score.argmax() + 1
    gbt.n_estimators = ntrees
    gbt.fit(X, y)

    if verbose:
        print 'Pickling best GBT object...'
    cPickle.dump(gbt, open(gbt_dir + 'GBT_' + question + '_ntrees' + str(ntrees) + '_depth' + str(depth) +
                           '.pickle', 'wb'))

    # make feature importance plot
    fimp = gbt.feature_importances_
    fimp /= fimp.max()
    sidx = np.argsort(fimp)
    feature_labels = np.asarray(feature_labels)
    pos = np.arange(50) + 0.5
    plt.clf()
    plt.barh(pos, fimp[sidx[-50:]], align='center')
    plt.yticks(pos, feature_labels[sidx[-50:]])
    plt.xlabel("Relative Importance: Top 50")
    plt.savefig(plot_dir + 'feature_importance_GBT_' + question + '.png')
    if doshow:
        plt.show()

    return gbt


def write_predictions(gbt_list, df_test, questions, depth):
    y_predict = np.zeros((len(df_test), len(questions)))

    for j in xrange(len(questions)):
        y_predict[:, j] = gbt_list[j].predict(df_test.values)

    if use_constraints:
        # Unique predictions Correspond to these values. Need to calculate remaining classes.
        unique_cols = ['Class1.1', 'Class1.2', 'Class2.1', 'Class3.1', 'Class4.1', 'Class5.1', 'Class5.2', 'Class5.3',
                       'Class6.1', 'Class7.1', 'Class7.2', 'Class8.1', 'Class8.2', 'Class8.3', 'Class8.4', 'Class8.5',
                       'Class8.6', 'Class9.1', 'Class9.2', 'Class10.1', 'Class10.2', 'Class11.1', 'Class11.2',
                       'Class11.3', 'Class11.4', 'Class11.5']

        # store predictions in a CSV file
        y_predict = pd.DataFrame(data=y_predict, index=df_test.index, columns=unique_cols)

        # calculate remaining classes using constraints
        y_predict['Class1.3'] = 1.0 - y_predict['Class1.1'] - y_predict['Class1.2']
        y_predict['Class2.2'] = y_predict['Class1.2'] - y_predict['Class2.1']
        y_predict['Class3.2'] = y_predict['Class2.2'] - y_predict['Class3.1']
        y_predict['Class4.2'] = y_predict['Class2.2'] - y_predict['Class4.1']
        y_predict['Class5.4'] = y_predict['Class2.2'] - y_predict['Class5.1'] - y_predict['Class5.2'] - \
                                y_predict['Class5.3']
        y_predict['Class6.2'] = 1.0 - y_predict['Class6.1']
        y_predict['Class7.3'] = y_predict['Class1.1'] - y_predict['Class7.1'] - y_predict['Class7.2']
        y_predict['Class8.7'] = y_predict['Class6.1'] - y_predict['Class8.1'] - y_predict['Class8.2'] - \
                                y_predict['Class8.3'] - y_predict['Class8.4'] - y_predict['Class8.5'] - \
                                y_predict['Class8.6']
        y_predict['Class9.3'] = y_predict['Class2.1'] - y_predict['Class9.1'] - y_predict['Class9.2']
        y_predict['Class10.3'] = y_predict['Class4.1'] - y_predict['Class10.1'] - y_predict['Class10.2']
        y_predict['Class11.6'] = y_predict['Class4.1'] - y_predict['Class11.1'] - y_predict['Class11.2'] - \
                                 y_predict['Class11.3'] - y_predict['Class11.4'] - y_predict['Class11.5']

    else:
        y_predict = pd.DataFrame(data=y_predict, index=df_test.index, columns=questions)

    y_predict.index.name = 'GalaxyID'

    # dump to CSV file
    y_predict.to_csv(base_dir + 'data/GBT_predictions_depth' + str(depth) + '.csv')


if __name__ == "__main__":

    njobs = multiprocessing.cpu_count() - 1
    pool = multiprocessing.Pool(njobs)
    pool.map(int, range(njobs))

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
    if len(y.index - df.index) > 0:
        print 'Error! Missing training data in feature dataframe.'
        exit()

    train_set = y.index

    files = glob.glob(base_dir + 'data/images_test_rev1/*.jpg')
    test_set = [int(f.split('/')[-1].split('.')[0]) for f in files]
    assert np.all(np.isfinite(df.ix[test_set]))

    if not np.all(np.isfinite(df)):
        print 'Error! Non-finite feature values detected.'

    depths = [2, 4, 6, 8, 10]
    for depth in depths:
        args = []
        print 'Doing depth', depth
        for question in y.columns:
            args.append((df.ix[train_set].values, y[question], question, depth,
                         df.columns, None))
        if do_parallel:
            gbt_list = pool.map(train_gbt, args)
        else:
            gbt_list = map(train_gbt, args)
        print 'Writing predictions for depth', depth, '...'
        write_predictions(gbt_list, df.ix[test_set], y.columns, depth)
        del gbt_list  # free memory

