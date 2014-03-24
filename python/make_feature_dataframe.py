__author__ = 'brandonkelly'

import pandas as pd
import numpy as np
import cPickle
import os

base_dir = os.environ['HOME'] + '/Projects/Kaggle/galaxy_zoo/'
dct_dir = base_dir + 'data/react/'
plot_dir = base_dir + 'plots/'
training_dir = base_dir + 'data/images_training_rev1/'

doshow = False
verbose = True
do_parallel = False
