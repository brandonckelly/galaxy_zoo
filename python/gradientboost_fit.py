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

doshow = True
verbose = True


def train_gbt(args):

    X, y, question, depth, feature_labels, ntrees = args

    # first find optimal number of trees
    if ntrees is None:
        ntrees = 1000
    if verbose:
        print 'Training gradient boosted tree for question', question
    gbt = GradientBoostingRegressor(n_estimators=ntrees, subsample=0.5, max_depth=depth, verbose=verbose)
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

    ntrees = oob_score.argmax()
    gbt.n_estimators = ntrees
    gbt.fit(X, y)

    if verbose:
        print 'Pickling best RF object...'
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

    df = df.ix[y.index]

    if not np.all(np.isfinite(df)):
        print 'Error! Non-finite feature values detected.'

    depths = [2, 4, 6, 8, 10]
    for depth in depths:
        args = []
        for question in y.columns:
            args.append((df.values[:1000,:10], y[question].values[:1000], question, depth, df.columns[:10], None))
        pool.map(train_gbt, args)

