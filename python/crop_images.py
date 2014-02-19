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

print 'Doing file...'
for file in training_files[:10]:
    print file, '...'
    # load the JPEG image into a numpy array
    im = np.array(Image.open(file)).astype(float)
    ndim = im.shape

    # loop over images in each band
    for c in range(3):
        print '    ', c, '...'
        # find local maximum that is closest to center of image
        this_im = im[:, :, c]
        coords = peak_local_max(this_im, min_distance=20)
        distance = np.sqrt((coords[:, 0] - ndim[0] / 2) ** 2 + (coords[:, 1] - ndim[1] / 2) ** 2)
        lmax_idx = distance.argmin()
        centroid = coords[lmax_idx, :]  # coordinates of galaxy center

        # now find approximate galaxy radius by fitting a 2-d gaussian function
        xcent = np.arange(this_im.shape[0]) - centroid[0]
        ycent = np.arange(this_im.shape[1]) - centroid[1]
        Mxx = np.sum(xcent ** 2 * this_im[:, centroid[1]]) / np.sum(this_im[:, centroid[1]])
        Myy = np.sum(ycent ** 2 * this_im[centroid[0], :]) / np.sum(this_im[centroid[0], :])
        g2d_init = models.Gaussian2D(amplitude=1.0, x_mean=centroid[0], y_mean=centroid[1], y_stddev=np.sqrt(Myy),
                                     x_stddev=np.sqrt(Mxx))
        g2d_init.x_mean.fixed = True
        g2d_init.y_mean.fixed = True
        fitter = fitting.NonLinearLSQFitter()
        x, y = np.mgrid[:ndim[0], :ndim[1]]
        gauss_fit = fitter(g2d_init, x, y, this_im)

        # crop the image to 3-sigma and save it
        xrange = 3.0 * np.abs(gauss_fit.x_stddev)
        yrange = 3.0 * np.abs(gauss_fit.y_stddev)
        xmin = int(centroid[0] - xrange)
        if xmin < 0:
            xmin = 0
        xmax = int(centroid[0] + xrange)
        if xmax > ndim[0]:
            xmax = ndim[0]
        ymin = int(centroid[1] - yrange)
        if ymin < 0:
            ymin = 0
        ymax = int(centroid[1] + yrange)
        if ymax > ndim[1]:
            ymax = ndim[1]

        cropped_im = this_im[xmin:xmax, ymin:ymax]  # crop the image

        if doplot:
            # show the image
            plt.subplot(121)
            plt.imshow(this_im, cmap='hot')
            plt.title('Original')
            plt.subplot(122)
            plt.imshow(cropped_im, cmap='hot')
            plt.title('Cropped')
            plt.show()

        # finally, save the image