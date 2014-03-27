__author__ = 'brandonkelly'

import pandas as pd
import numpy as np
import os
import cPickle
import glob
import sys

base_dir = os.environ['HOME'] + '/Projects/Kaggle/galaxy_zoo/'
plot_dir = base_dir + 'plots/'
dct_dir = base_dir + 'data/react/'
training_dir = base_dir + 'data/images_training_rev1/'


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print 'Syntax: python make_prediction_file.py base_name'
        exit()

    base_name = sys.argv[1]

    files = glob.glob(base_dir + 'data/images_test_rev1/*.jpg')
    test_ids = [int(f.split('/')[-1].split('.')[0]) for f in files]

    # load the test data for the features
    df = pd.read_hdf(base_dir + 'data/galaxy_features.h5', 'df')
    df = df.ix[test_ids]

    assert np.all(np.isfinite(df.values))

    print 'Found', len(test_ids), 'galaxies with test labels.'
    print 'Loading the Regression object...'
    # load the random forest object
    rf = cPickle.load(open(base_dir + 'data/' + base_name + '.pickle', 'rb'))

    print 'Predicting the values...'
    y_predict = rf.predict(df.values)

    # Random Forest predictions Correspond to these values. Need to calculate remaining classes.
    unique_cols = ['Class1.1', 'Class1.2', 'Class2.1', 'Class3.1', 'Class4.1', 'Class5.1', 'Class5.2', 'Class5.3',
                   'Class6.1', 'Class7.1', 'Class7.2', 'Class8.1', 'Class8.2', 'Class8.3', 'Class8.4', 'Class8.5',
                   'Class8.6', 'Class9.1', 'Class9.2', 'Class10.1', 'Class10.2', 'Class11.1', 'Class11.2',
                   'Class11.3', 'Class11.4', 'Class11.5']

    # store predictions in a CSV file
    y_predict = pd.DataFrame(data=y_predict, index=test_ids, columns=unique_cols)

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

    # dump to CSV file
    y_predict.to_csv(base_dir + 'data/' + base_name + '_predictions.csv')