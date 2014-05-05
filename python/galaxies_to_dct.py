__author__ = 'brandonkelly'

import numpy as np
import matplotlib.pyplot as plt
from react import REACT2D
import os
import glob
import multiprocessing
import cPickle
import datetime
import cProfile
import pstats

base_dir = os.environ['HOME'] + '/Projects/Kaggle/galaxy_zoo/'
data_dir = base_dir + 'data/'
training_dir = data_dir + 'images_training_rev1/'
test_dir = data_dir + 'images_test_rev1/'
plot_dir = base_dir + 'plots/'

doshow = False
image_dir = test_dir
max_order0 = 40
ncoefs = max_order0 * max_order0
verbose = False
do_parallel = True


def do_dct_transform(args):

    galaxy_id, image_dir = args
    print 'Galaxy ID:', galaxy_id

    # first get flux ratios (colors)
    total_flux = 0.0
    for band in range(3):
        image = np.load(image_dir + galaxy_id + '_' + str(band) + '.npy')
        total_flux += image.sum()

    # now do the DCT on each image
    for band in range(3):
        if not os.path.isfile(image_dir + galaxy_id + '_' + str(band) + '.npy'):
            return None
        image = np.load(image_dir + galaxy_id + '_' + str(band) + '.npy')
        image /= total_flux  # normalize to flux over all 3 bands is unity
        border = np.hstack((image[:, 0], image[:, -1], image[0, 1:-1], image[-1, 1:-1]))
        sigsqr = np.median(np.abs(border - np.median(border))) ** 2
        max_order = min(min(image.shape), max_order0)

        if verbose:
            print 'Band:', band
            print 'Estimated noise level:', np.sqrt(sigsqr)
            print 'Noise relative to center:', np.sqrt(sigsqr) / image[image.shape[0]/2, image.shape[1]/2]
            print 'Image size:', image.shape

        # check image size, values
        if min(image.shape) < 5:
            print "Image dimensions need to be at least 5 pixels on either side."
            return None
        if not np.all(np.isfinite(image)):
            print "Non-finite values detected in image, ignoring."
            return None

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
        if doshow:
            plt.show()
        plt.close()

        smoother2d.galaxy_id = galaxy_id
        smoother2d.band = band

        cPickle.dump(smoother2d, open(data_dir + 'react/' + galaxy_id + '_' + str(band) + '_dct.pickle', 'wb'))

        coefs = smoother2d.coefs
        if len(coefs) < ncoefs:
            # add zeros so all the coefficient arrays have a standard size
            nzeros = ncoefs - len(coefs)
            coefs = np.append(coefs, np.zeros(nzeros))

        if band == 0:
            dct_coefs = coefs
        else:
            dct_coefs = np.hstack((dct_coefs, coefs))

    return dct_coefs


if __name__ == "__main__":
    start_time = datetime.datetime.now()

    njobs = 4
    pool = multiprocessing.Pool(njobs)
    # warm up the pool
    pool.map(int, range(njobs))

    # run on training set images
    files_0 = glob.glob(image_dir + '*_0.npy')
    files_1 = glob.glob(image_dir + '*_1.npy')
    files_2 = glob.glob(image_dir + '*_2.npy')

    galaxy_ids_0 = set([f.split('/')[-1].split('_')[0] for f in files_0])
    galaxy_ids_1 = set([f.split('/')[-1].split('_')[0] for f in files_1])
    galaxy_ids_2 = set([f.split('/')[-1].split('_')[0] for f in files_2])

    galaxy_ids = galaxy_ids_0 & galaxy_ids_1 & galaxy_ids_2
    galaxy_ids = list(galaxy_ids)

    # find which ones we've already done
    already_done1 = glob.glob(data_dir + 'react/' + '*_0_dct.pickle')
    already_done2 = glob.glob(data_dir + 'react/' + '*_1_dct.pickle')
    already_done3 = glob.glob(data_dir + 'react/' + '*_2_dct.pickle')

    already_done1 = set([s.split('/')[-1].split('_')[0] for s in already_done1])
    already_done2 = set([s.split('/')[-1].split('_')[0] for s in already_done2])
    already_done3 = set([s.split('/')[-1].split('_')[0] for s in already_done3])

    already_done = already_done1 & already_done2 & already_done3

    print 'Already done', len(already_done), 'galaxies.'

    left_to_do = set(galaxy_ids) - already_done

    print 'Have', len(left_to_do), 'galaxies left.'

    if not do_parallel:
        # do_dct_transform(galaxy_ids)
        # cProfile.run('do_dct_transform(galaxy_ids[0])', 'dctstats')
        # profile = pstats.Stats('dctstats')
        # profile.sort_stats('cumulative').print_stats(25)
        map(do_dct_transform, left_to_do)
    else:
        pool.map(do_dct_transform, left_to_do)

    end_time = datetime.datetime.now()

    tdiff = end_time - start_time
    tdiff = tdiff.total_seconds()
    print 'Did', len(galaxy_ids), 'galaxies in', tdiff / 60.0 / 60.0, 'hours.'