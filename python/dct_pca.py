__author__ = 'brandonkelly'

import numpy as np
from sklearn.decomposition import RandomizedPCA
import cPickle
import os
import matplotlib.pyplot as plt
import multiprocessing

base_dir = os.environ['HOME'] + '/Projects/Kaggle/galaxy_zoo/'
dct_dir = base_dir + 'data/react/'
plot_dir = base_dir + 'plots/'

npca = 1000
doshow = False
verbose = False
do_parallel = False


def make_pc_images(evect, dims):
    pass


if __name__ == "__main__":

    pca = RandomizedPCA(npca)