__author__ = 'brandonkelly'

import numpy as np
import matplotlib.pyplot as plt
from react import REACT2D
import os
import glob
import multiprocessing
import cPickle
import datetime

base_dir = os.environ['HOME'] + '/Projects/Kaggle/galaxy_zoo/'
data_dir = base_dir + 'data/'
training_dir = data_dir + 'images_training_rev1/'
test_dir = data_dir + 'images_test_rev1/'
plot_dir = base_dir + 'plots/'

doshow = False
image_dir = training_dir
max_order0 = 50
verbose = True


def do_dct_transform(args):

    galaxy_id = args
    # first get flux ratios (colors)
    total_flux = 0.0
    for band in range(3):
        image = np.load(image_dir + galaxy_id + '_' + str(band) + '.npy')
        total_flux += image.sum()

    # now do the DCT on each image
    for band in range(3):
        image = np.load(image_dir + galaxy_id + '_' + str(band) + '.npy')
        image /= total_flux  # normalize to flux over all 3 bands is unity
        border = np.hstack((image[:, 0], image[:, -1], image[0, 1:-1], image[-1, 1:-1]))
        sigsqr = np.median(np.abs(border - np.median(border))) ** 2
        max_order = min(min(image.shape), max_order0)

        if verbose:
            print 'Galaxy ID:', galaxy_id
            print 'Band:', band
            print 'Estimated noise level:', np.sqrt(sigsqr)
            print 'Noise relative to center:', np.sqrt(sigsqr) / image[image.shape[0]/2, image.shape[1]/2]
            print 'Image size:', image.shape

        smoother2d = REACT2D(max_order=max_order, method='monotone')
        ismooth = smoother2d.fit(image, sigsqr)

        plt.clf()
        plt.subplot(221)
        plt.imshow(image, cmap='hot')
        plt.title(str(galaxy_id) + ', ' + str(band))
        plt.subplot(222)
        plt.imshow(ismooth, cmap='hot')
        plt.title('REACT Fit')
        plt.subplot(223)
        plt.imshow(image - ismooth, cmap='hot')
        plt.title('Residual')
        plt.subplot(224)
        plt.plot(smoother2d.shrinkage_factors)
        plt.ylabel('Shrinkage Factor')
        plt.xlabel('Index of Basis Function')
        # plt.tight_layout()
        plt.savefig(plot_dir + galaxy_id + '_' + str(band) + '_DCT.png')
        plt.close()

        smoother2d.galaxy_id = galaxy_id
        smoother2d.band = band

        cPickle.dump(smoother2d, open(data_dir + 'react/' + galaxy_id + '_' + str(band) + '_dct.pickle', 'wb'))


if __name__ == "__main__":
    start_time = datetime.datetime.now()

    pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
    # warm up the pool
    pool.map(int, range(multiprocessing.cpu_count() - 1))

    # run on training set images
    files_0 = glob.glob(image_dir + '*_0.npy')
    files_1 = glob.glob(image_dir + '*_1.npy')
    files_2 = glob.glob(image_dir + '*_2.npy')

    galaxy_ids_0 = set([f.split('/')[-1].split('_')[0] for f in files_0])
    galaxy_ids_1 = set([f.split('/')[-1].split('_')[0] for f in files_1])
    galaxy_ids_2 = set([f.split('/')[-1].split('_')[0] for f in files_2])

    galaxy_ids = galaxy_ids_0 & galaxy_ids_1 & galaxy_ids_2
    galaxy_ids = list(galaxy_ids)
    galaxy_ids = galaxy_ids[:2]

    do_parallel = False
    if not do_parallel:
        do_dct_transform(galaxy_ids[0])
        # map(do_dct_transform, galaxy_ids)
    else:
        pool.map(do_dct_transform, galaxy_ids)

    end_time = datetime.datetime.now()

    tdiff = end_time - start_time
    tdiff = tdiff.total_seconds()
    print 'Did', len(galaxy_ids), 'galaxies in', tdiff / 60.0 / 60.0, 'hours.'