__author__ = 'brandonkelly'

import numpy as np
import matplotlib.pyplot as plt
from react import REACT2D
import os
import glob
import multiprocessing

base_dir = os.environ['HOME'] + '/Projects/Kaggle/galaxy_zoo/'
data_dir = base_dir + 'data/'
training_dir = data_dir + 'images_training_rev1/'
test_dir = data_dir + 'images_test_rev1/'
plot_dir = base_dir + 'plots/'

doshow = False
