__author__ = 'brandonkelly'

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from sklearn.ensemble import RandomForestRegressor
import cPickle
import pandas as pd
from dct_to_lda import image_sanity_check

base_dir = os.environ['HOME'] + '/Projects/Kaggle/galaxy_zoo/'
data_dir = base_dir + 'data/'
dct_dir = data_dir + 'react/'
plot_dir = base_dir + 'plots/'

doshow = True
verbose = True
njobs = 1


def train_rf(df, y):

    # first find optimal number of trees
    ntrees = [10, 20, 40, 80, 160, 320]
    oob_rmse = np.zeros(len(ntrees))
    for i, nt in enumerate(ntrees):
        rf = RandomForestRegressor(max_features='sqrt', oob_score=True, n_estimators=nt, verbose=verbose, n_jobs=njobs)
        rf.fit(df.values, y)
        yhat_oob = rf.oob_prediction_
        oob_rmse[i] = np.sqrt(np.mean((yhat_oob - y) ** 2))

    if verbose:
        print '# of trees | OOB RMSE'
        for i in range(len(ntrees)):
            print ntrees[i], '    | ', oob_rmse[i]
        print ''
        print 'Optimal number of trees is', ntrees[oob_rmse.argmin()]

    plt.clf()
    plt.plot(ntrees, oob_rmse, '-o')
    plt.ylabel('OOB RMSE')
    plt.xlabel('Number of Trees')
    plt.savefig(plot_dir + 'OOB_RMSE_RF_ntrees.png')
    if doshow:
        plt.show()
        plt.close()

    ntrees = ntrees[oob_rmse.argmin()]

    # now find optimum value of m (features to consider in split)
    nfeatures = [5, 10, 20, 40, 80, 160]

    oob_rmse = np.zeros(len(nfeatures))
    best_rf = None
    best_rmse = 1e300
    for i, m in enumerate(nfeatures):
        rf = RandomForestRegressor(max_features=m, oob_score=True, n_estimators=ntrees, verbose=verbose, n_jobs=njobs)
        rf.fit(df.values, y)
        yhat_oob = rf.oob_prediction_
        oob_rmse[i] = np.sqrt(np.mean((yhat_oob - y) ** 2))
        if oob_rmse[i] < best_rmse:
            # save best RF so we don't need to recompute it later
            best_rf = rf
            best_rmse = oob_rmse[i]

    if verbose:
        print 'm features | OOB RMSE'
        for i in range(len(nfeatures)):
            print nfeatures[i], '    | ', oob_rmse[i]
        print ''
        print 'Optimal number of features for split is', nfeatures[oob_rmse.argmin()]

    plt.clf()
    plt.plot(nfeatures, oob_rmse, '-o')
    plt.ylabel('OOB RMSE')
    plt.xlabel('Number of Feature for Split Criteria')
    plt.savefig(plot_dir + 'OOB_RMSE_RF_mfeatures.png')
    if doshow:
        plt.show()
        plt.close()

    mfeatures = nfeatures[oob_rmse.argmin()]
    cPickle.dump(best_rf, open(data_dir + 'RF_regressor.pickle', 'wb'))

    # make feature importance plot
    fimp = best_rf.feature_importances_
    fimp /= fimp.max()
    sidx = np.argsort(fimp)
    feature_labels = np.array(df.columns)
    pos = np.arange(50) + 0.5
    plt.clf()
    plt.barh(pos, fimp[sidx[-50:]], align='center')
    plt.yticks(pos, feature_labels[sidx])
    plt.xlabel("Relative Importance: Top 50")
    plt.savefig(plot_dir + 'feature_importance_RF.png')
    if doshow:
        plt.show()


if __name__ == "__main__":

    # load the training labels
    if verbose:
        print 'Loading training labels...'
    y = pd.read_csv(base_dir + 'data/training_solutions_rev1.csv').set_index('GalaxyID')

    print 'Found', len(y), 'galaxies with training labels.'

    # load the training data for the features
    df = pd.read_hdf(base_dir + 'data/galaxy_features.h5', 'df')
    df = df.ix[y.index]

    train_rf(df, y)