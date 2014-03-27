__author__ = 'brandonkelly'

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import cPickle
import pandas as pd

base_dir = os.environ['HOME'] + '/Projects/Kaggle/galaxy_zoo/'
data_dir = base_dir + 'data/'
dct_dir = data_dir + 'react/'
plot_dir = base_dir + 'plots/'

doshow = True
verbose = True
njobs = 7
do_extratrees = False


def get_err(y, yfit):
    # calculate remaining classes using constraints
    yfit['Class1.3'] = 1.0 - yfit['Class1.1'] - yfit['Class1.2']
    yfit['Class2.2'] = yfit['Class1.2'] - yfit['Class2.1']
    yfit['Class3.2'] = yfit['Class2.2'] - yfit['Class3.1']
    yfit['Class4.2'] = yfit['Class2.2'] - yfit['Class4.1']
    yfit['Class5.4'] = yfit['Class2.2'] - yfit['Class5.1'] - yfit['Class5.2'] - \
                            yfit['Class5.3']
    yfit['Class6.2'] = 1.0 - yfit['Class6.1']
    yfit['Class7.3'] = yfit['Class1.1'] - yfit['Class7.1'] - yfit['Class7.2']
    yfit['Class8.7'] = yfit['Class6.1'] - yfit['Class8.1'] - yfit['Class8.2'] - \
                            yfit['Class8.3'] - yfit['Class8.4'] - yfit['Class8.5'] - \
                            yfit['Class8.6']
    yfit['Class9.3'] = yfit['Class2.1'] - yfit['Class9.1'] - yfit['Class9.2']
    yfit['Class10.3'] = yfit['Class4.1'] - yfit['Class10.1'] - yfit['Class10.2']
    yfit['Class11.6'] = yfit['Class4.1'] - yfit['Class11.1'] - yfit['Class11.2'] - \
                             yfit['Class11.3'] - yfit['Class11.4'] - yfit['Class11.5']

    err = y - yfit
    return err


def train_rf(df, y, ntrees=None, msplit=None):

    # Random Forest predictions Correspond to these values. Need to calculate the values for the remaining
    # classes using the summation constraints.
    unique_cols = ['Class1.1', 'Class1.2', 'Class2.1', 'Class3.1', 'Class4.1', 'Class5.1', 'Class5.2', 'Class5.3',
                   'Class6.1', 'Class7.1', 'Class7.2', 'Class8.1', 'Class8.2', 'Class8.3', 'Class8.4', 'Class8.5',
                   'Class8.6', 'Class9.1', 'Class9.2', 'Class10.1', 'Class10.2', 'Class11.1', 'Class11.2',
                   'Class11.3', 'Class11.4', 'Class11.5']

    y_unique = y[unique_cols]

    if do_extratrees:
        reg_str = 'ET'
    else:
        reg_str = 'RF'

    # first find optimal number of trees
    if ntrees is None:
        ntrees = [250, 300, 330, 360, 400]
        oob_rmse = np.zeros(len(ntrees))
        for i, nt in enumerate(ntrees):
            if verbose:
                print 'Training random forest with T =', nt, ', m = 80'
            if do_extratrees:
                rf = ExtraTreesRegressor(max_features=80, oob_score=True, n_estimators=nt, verbose=verbose,
                                         n_jobs=njobs, bootstrap=True)
            else:
                rf = RandomForestRegressor(max_features=80, oob_score=True, n_estimators=nt, verbose=verbose,
                                           n_jobs=njobs)
            rf.fit(df.values, y_unique.values)
            yhat_oob = pd.DataFrame(data=rf.oob_prediction_, index=y.index, columns=unique_cols)
            oob_rmse[i] = np.sqrt(np.mean(get_err(y, yhat_oob).values ** 2))

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
        plt.savefig(plot_dir + 'OOB_RMSE_' + reg_str + '_ntrees.png')
        if doshow:
            plt.show()
            plt.close()

        ntrees = ntrees[oob_rmse.argmin()]

    if msplit is None:
        # now find optimum value of m (features to consider in split)
        msplit = [60, 80, 100, 120, 140]

        oob_rmse = np.zeros(len(msplit))
        best_rmse = 1e300
        best_err = None
        for i, m in enumerate(msplit):
            if verbose:
                print 'Training random forest with T =', ntrees, ', m =', m
            if do_extratrees:
                rf = ExtraTreesRegressor(max_features=m, oob_score=True, n_estimators=ntrees, verbose=verbose,
                                         n_jobs=njobs, bootstrap=True)
            else:
                rf = RandomForestRegressor(max_features=m, oob_score=True, n_estimators=ntrees, verbose=verbose,
                                           n_jobs=njobs)
            rf.fit(df.values, y_unique.values)
            yhat_oob = pd.DataFrame(data=rf.oob_prediction_, index=y.index, columns=unique_cols)
            oob_err = get_err(y, yhat_oob)
            oob_rmse[i] = np.sqrt(np.mean(oob_err.values ** 2))

            if oob_rmse[i] < best_rmse:
                # save best RF so we don't need to recompute it later
                cPickle.dump(rf, open(data_dir + reg_str + '_regressor.pickle', 'wb'))
                best_rmse = oob_rmse[i]
                best_err = oob_err.values

        if verbose:
            print 'm features | OOB RMSE'
            for i in range(len(msplit)):
                print msplit[i], '    | ', oob_rmse[i]
            print ''
            print 'Optimal number of features for split is', msplit[oob_rmse.argmin()]

        plt.clf()
        plt.plot(msplit, oob_rmse, '-o')
        plt.ylabel('OOB RMSE')
        plt.xlabel('Number of Feature for Split Criteria')
        plt.savefig(plot_dir + 'OOB_RMSE_' + reg_str + '_mfeatures.png')
        if doshow:
            plt.show()
            plt.close()
        best_rf = cPickle.load(open(data_dir + reg_str + '_regressor.pickle', 'rb'))
    else:
        if do_extratrees:
            best_rf = ExtraTreesRegressor(max_features=msplit, oob_score=True, n_estimators=ntrees, verbose=verbose,
                                          n_jobs=njobs, bootstrap=True)
        else:
            best_rf = RandomForestRegressor(max_features=msplit, oob_score=True, n_estimators=ntrees, verbose=verbose,
                                            n_jobs=njobs)
        best_rf.fit(df.values, y_unique)
        yhat_oob = pd.DataFrame(data=best_rf.oob_prediction_, index=y.index, columns=unique_cols)
        best_err = get_err(y, yhat_oob)
        if verbose:
            print 'Pickling best RF object...'
        cPickle.dump(best_rf, open(data_dir + reg_str + '_regressor.pickle', 'wb'))

    # make feature importance plot
    fimp = best_rf.feature_importances_
    fimp /= fimp.max()
    sidx = np.argsort(fimp)
    feature_labels = np.array(df.columns)
    pos = np.arange(50) + 0.5
    plt.clf()
    plt.barh(pos, fimp[sidx[-50:]], align='center')
    plt.yticks(pos, feature_labels[sidx[-50:]])
    plt.xlabel("Relative Importance: Top 50")
    plt.savefig(plot_dir + 'feature_importance_' + reg_str + '.png')
    if doshow:
        plt.show()

    # plot histogram of the errors
    rmse_by_galaxy = np.sqrt(np.mean(best_err ** 2, axis=1))
    plt.hist(rmse_by_galaxy, bins=200, histtype='stepfilled')
    plt.xlabel('OOB RMSE')
    plt.savefig(plot_dir + 'OOB_RMSE_' + reg_str + '_histogram.png')
    if doshow:
        plt.show()

    # plot errors vs. PCA norm
    pc_cols = [c for c in df.columns if 'PC' in c]
    pca_norm = np.norm(df[pc_cols], axis=1)
    plt.loglog(pca_norm, rmse_by_galaxy, '.')
    plt.xlabel('PC Norm')
    plt.ylabel('OOB RMSE')
    plt.savefig(plot_dir + 'OOB_RMSE_vs_PC_Norm_' + reg_str + '.png')
    if doshow:
        plt.show()


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
    if len(y.index - df.index) > 0:
        print 'Error! Missing training data in feature dataframe.'
        exit()

    df = df.ix[y.index]

    if not np.all(np.isfinite(df)):
        print 'Error! Non-finite feature values detected.'

    train_rf(df, y, ntrees=400, msplit=140)