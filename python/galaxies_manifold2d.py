__author__ = 'brandonkelly'

import numpy as np
from scipy.misc import bytescale
import matplotlib.pyplot as plt
import cPickle
from sklearn.decomposition import KernelPCA
import os

base_dir = os.environ['HOME'] + 'Projects/Kaggle/galaxy_zoo/'

