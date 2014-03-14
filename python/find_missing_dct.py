__author__ = 'brandonkelly'

import os
import glob
import multiprocessing
from galaxies_to_dct import do_dct_transform
import datetime

do_missing_dcts = True
do_parallel = True

base_dir = os.environ['HOME'] + '/Projects/Kaggle/galaxy_zoo/'
data_dir = base_dir + 'data/react/'
plot_dir = base_dir + 'plots/'
test_dir = base_dir + 'data/images_test_rev1/'
train_dir = base_dir + 'data/images_training_rev1/'

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

    print 'Getting DCTs for missing galaxies...'
    if not do_parallel:
        # do_dct_transform(galaxy_ids)
        # cProfile.run('do_dct_transform(galaxy_ids[0])', 'dctstats')
        # profile = pstats.Stats('dctstats')
        # profile.sort_stats('cumulative').print_stats(25)
        map(do_dct_transform, incomplete)
    else:
        pool.map(do_dct_transform, incomplete)

    end_time = datetime.datetime.now()

    tdiff = end_time - start_time
    tdiff = tdiff.total_seconds()
    print 'Did', len(incomplete), 'galaxies in', tdiff / 60.0 / 60.0, 'hours.'