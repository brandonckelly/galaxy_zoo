__author__ = 'brandonkelly'

import os
import glob
import multiprocessing
import datetime
import numpy as np
import matplotlib.pyplot as plt
import cPickle
from react import REACT2D


do_missing_dcts = True
do_parallel = True
doshow = False
verbose = False

base_dir = os.environ['HOME'] + '/Projects/Kaggle/galaxy_zoo/'
data_dir = base_dir + 'data/react/'
plot_dir = base_dir + 'plots/'
test_dir = base_dir + 'data/images_test_rev1/'
train_dir = base_dir + 'data/images_training_rev1/'

max_order0 = 50


def do_dct_transform(fbase):

    galaxy_id = fbase.split('/')[-1]
    print 'Galaxy ID:', galaxy_id

    # first get flux ratios (colors)
    total_flux = 0.0
    for band in range(3):
        if not os.path.isfile(fbase + '_' + str(band) + '.npy'):
            print 'File', fbase + '_' + str(band) + '.npy', 'not found, returning.'
            return

        image = np.load(fbase + '_' + str(band) + '.npy')
        total_flux += image.sum()

    # now do the DCT on each image
    for band in range(3):
        image = np.load(fbase + '_' + str(band) + '.npy')
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
            continue
        if not np.all(np.isfinite(image)):
            print "Non-finite values detected in image, ignoring."
            continue

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

        cPickle.dump(smoother2d, open(data_dir + galaxy_id + '_' + str(band) + '_dct.pickle', 'wb'))


def find_directory(galaxy_ids, train_ids, test_ids):

    fbase = []

    for id in galaxy_ids:
        if id in train_ids:
            fbase.append(train_dir + id)
        elif id in test_ids:
            fbase.append(test_dir + id)

    return fbase


# get galaxy ids
test_gals = glob.glob(test_dir + '*.jpg')
test_ids = set([f.split('/')[-1].split('.')[0] for f in test_gals])
train_gals = glob.glob(train_dir + '*.jpg')
train_ids = set([f.split('/')[-1].split('.')[0] for f in train_gals])

galaxy_ids_ref = test_ids | train_ids

print 'Found', len(galaxy_ids_ref), 'galaxies with JPG images.'

# find which galaxies we have a full dct for
files_0 = glob.glob(data_dir + '*_0_dct.pickle')
files_1 = glob.glob(data_dir + '*_1_dct.pickle')
files_2 = glob.glob(data_dir + '*_2_dct.pickle')

galaxy_ids_0 = set([f.split('/')[-1].split('_')[0] for f in files_0])
galaxy_ids_1 = set([f.split('/')[-1].split('_')[0] for f in files_1])
galaxy_ids_2 = set([f.split('/')[-1].split('_')[0] for f in files_2])

galaxy_ids = galaxy_ids_0 & galaxy_ids_1 & galaxy_ids_2

# find which ones are incomplete

incomplete = galaxy_ids_ref - galaxy_ids

print 'Found', len(incomplete), 'galaxies with incomplete DCTs:'
print incomplete

if do_missing_dcts and len(incomplete) > 0:
    # get the DCT for the missing sources
    start_time = datetime.datetime.now()

    njobs = 7
    pool = multiprocessing.Pool(njobs)
    # warm up the pool
    pool.map(int, range(njobs))

    fbase = find_directory(incomplete, train_ids, test_ids)
    assert len(incomplete) == len(fbase)

    print 'Getting DCTs for missing galaxies...'
    if not do_parallel:
        # do_dct_transform(galaxy_ids)
        # cProfile.run('do_dct_transform(galaxy_ids[0])', 'dctstats')
        # profile = pstats.Stats('dctstats')
        # profile.sort_stats('cumulative').print_stats(25)
        map(do_dct_transform, fbase)
    else:
        pool.map(do_dct_transform, fbase)

    end_time = datetime.datetime.now()

    tdiff = end_time - start_time
    tdiff = tdiff.total_seconds()
    print 'Did', len(incomplete), 'galaxies in', tdiff / 60.0 / 60.0, 'hours.'