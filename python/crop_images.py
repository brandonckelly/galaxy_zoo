__author__ = 'brandonkelly'

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
import os
import glob
from skimage.feature import peak_local_max

base_dir = os.environ['HOME'] + '/Projects/Kaggle/galaxy_zoo/'
data_dir = base_dir + 'data/'
training_dir = data_dir + 'images_training_rev1/'
test_dir = data_dir + 'images_test_rev1/'

training_files = glob.glob(training_dir + '*.jpg')
doplot = True

for file in training_files:
    # load the JPEG image into a numpy array
    im = np.array(Image.open(file)).astype(float)
    ndim = im.shape

    # loop over images in each band
    for c in range(3):
        # find local maximum that is closest to center of image
        this_im = im[:, :, c]
        coords = peak_local_max(this_im, min_distance=20)
        distance = np.sqrt((coords[:, 0] - ndim[0] / 2) ** 2 + (coords[:, 1] - ndim[1] / 2) ** 2)
        lmax_idx = distance.argmin()
        centroid = coords[lmax_idx, :]  # coordinates of galaxy center

        # now find approximate galaxy radius by fitting a 2-d gaussian function


        # crop the image to 3-sigma and save it

        if doplot:
            # show the image