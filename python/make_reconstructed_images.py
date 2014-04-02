__author__ = 'brandonkelly'

import numpy as np
import matplotlib.pyplot as plt
import glob
from react import REACT2D
import cPickle
import os

base_dir = os.environ['HOME'] + '/Projects/Kaggle/galaxy_zoo/'
plot_dir = base_dir + 'plots/'
dct_dir = base_dir + 'data/react/'
test_dir = base_dir + 'data/images_test_rev1/'
train_dir = base_dir + 'data/images_training_rev1/'

doshow = True
verbose = True


# find which galaxies we have a full dct for
files_0 = glob.glob(dct_dir + '*_0_dct.pickle')
files_1 = glob.glob(dct_dir + '*_1_dct.pickle')
files_2 = glob.glob(dct_dir + '*_2_dct.pickle')

gfiles = glob.glob(base_dir + 'data/gauss_fit/transfer/*.csv')

galaxy_ids_0 = set([f.split('/')[-1].split('_')[0] for f in files_0])
galaxy_ids_1 = set([f.split('/')[-1].split('_')[0] for f in files_1])
galaxy_ids_2 = set([f.split('/')[-1].split('_')[0] for f in files_2])
galaxy_ids_3 = set([f.split('/')[-1].split('_')[0] for f in gfiles])

galaxy_ids = galaxy_ids_0 & galaxy_ids_1 & galaxy_ids_2 & galaxy_ids_3
del galaxy_ids_0, galaxy_ids_1, galaxy_ids_2, galaxy_ids_3

if verbose:
    print "Found", len(galaxy_ids), "galaxies."

# get galaxy ids
test_gals = glob.glob(test_dir + '*.jpg')
test_ids = set([f.split('/')[-1].split('.')[0] for f in test_gals])
train_gals = glob.glob(train_dir + '*.jpg')
train_ids = set([f.split('/')[-1].split('.')[0] for f in train_gals])

galaxy_ids_ref = test_ids | train_ids

if len(galaxy_ids_ref - galaxy_ids) != 0:
    print 'Missing data for the following galaxies:'
    print galaxy_ids_ref - galaxy_ids
    exit()

galaxy_ids = list(galaxy_ids)

# load the DCT coefficients
if verbose:
    print 'Loading DCT coefficients...'
dct_coefs = np.load(base_dir + 'data/DCT_array_all.npy').astype(np.float32)
