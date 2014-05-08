__author__ = 'brandonkelly'

import numpy as np
import matplotlib.pyplot as plt
from react import REACT2D
import os
import glob
import multiprocessing
import cPickle
import datetime
from galaxies_to_dct import do_dct_transform
from extract_postage_stamp import extract_gal_image
import pandas as pd

base_dir = os.environ['HOME'] + '/Projects/Kaggle/galaxy_zoo/'
data_dir = base_dir + 'data/'
training_dir = data_dir + 'images_training_rev1/'
test_dir = data_dir + 'images_test_rev1/'
plot_dir = base_dir + 'plots/'

image_dir = training_dir
verbose = True
do_parallel = True


def dct_pipeline(file):
    # first extact the pixels for the galaxy
    galaxy_id = extract_gal_image(file)

    # check that galaxy image is extracted for all three bands
    for c in [0, 1, 2]:
        if not os.path.isfile(plot_dir + galaxy_id + '_' + str(c) + '.png'):
            if verbose:
                print 'Could not find all 3 images for', galaxy_id
            return galaxy_id, None

    if verbose:
        print 'Doing DCT transform...'
    dct_coefs = do_dct_transform((galaxy_id, image_dir))

    return galaxy_id, dct_coefs


if __name__ == "__main__":

    if do_parallel:
        pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
        # warm up the pool
        pool.map(int, range(multiprocessing.cpu_count() - 1))


    files = glob.glob(image_dir + '*.jpg')
    # files = files[:1000]
    # id_list = ['100380']
    # files = [file_dir + id + '.jpg' for id in id_list]

    # find which ones we've already done
    already_done1 = glob.glob(plot_dir + '*_0.png')
    already_done2 = glob.glob(plot_dir + '*_1.png')
    already_done3 = glob.glob(plot_dir + '*_2.png')

    already_done1 = set([s.split('/')[-1].split('_')[0] for s in already_done1])
    already_done2 = set([s.split('/')[-1].split('_')[0] for s in already_done2])
    already_done3 = set([s.split('/')[-1].split('_')[0] for s in already_done3])

    already_done = already_done1 & already_done2 & already_done3
    print 'Already done', len(already_done), 'galaxies.'
    all_sources = set([tfile.split('/')[-1].split('.')[0] for tfile in files])

    left_to_do = all_sources - already_done

    print 'Have', len(left_to_do), 'galaxies left.'
    # training_files = [training_dir + sID + '.jpg' for sID in left_to_do]
    files = [image_dir + sID + '.jpg' for sID in left_to_do]

    print len(files), 'galaxies...'
    assert len(files) == len(left_to_do)
    print 'Source ID...'

    start_time = datetime.datetime.now()

    if do_parallel:
        output = pool.map(dct_pipeline, files)
    else:
        output = map(dct_pipeline, files)

    end_time = datetime.datetime.now()
    tdiff = end_time - start_time
    tdiff = tdiff.total_seconds()
    print 'Did', len(files), 'galaxies in', tdiff / 60.0 / 60.0, 'hours.'

    print 'Storing results in DataFrame...'
    galid = []
    dct_coefs = []
    for out in output:
        if out[1] is not None:
            galid.append(int(out[0]))
            dct_coefs.append(out[1])

    ncoefs = len(dct_coefs[0]) / 3
    cols = ['0-' + str(coef_idx) for coef_idx in range(ncoefs)]
    cols.extend(['1-' + str(coef_idx) for coef_idx in range(ncoefs)])
    cols.extend(['2-' + str(coef_idx) for coef_idx in range(ncoefs)])
    df = pd.DataFrame(data=np.asarray(dct_coefs), index=galid, columns=cols)
    df.index.name = 'GalaxyID'
    df.to_hdf(data_dir + 'DCT_coefs.h5', 'df')