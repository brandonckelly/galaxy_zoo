__author__ = 'brandonkelly'

import numpy as np
import matplotlib.pyplot as plt
import glob
from react import REACT2D
import cPickle
import os

base_dir = os.environ['HOME'] + '/Projects/Kaggle/galaxy_zoo/'
plot_dir = base_dir + 'plots/'
dct_dir = base_dir + 'data/react/'
test_dir = base_dir + 'data/images_test_rev1/'
train_dir = base_dir + 'data/images_training_rev1/'

doshow = False
verbose = True
do_build = True

# find which galaxies we have a full dct for
files_0 = glob.glob(dct_dir + '*_0_dct.pickle')
files_1 = glob.glob(dct_dir + '*_1_dct.pickle')
files_2 = glob.glob(dct_dir + '*_2_dct.pickle')

gfiles = glob.glob(base_dir + 'data/gauss_fit/transfer/*.csv')

galaxy_ids_0 = set([f.split('/')[-1].split('_')[0] for f in files_0])
galaxy_ids_1 = set([f.split('/')[-1].split('_')[0] for f in files_1])
galaxy_ids_2 = set([f.split('/')[-1].split('_')[0] for f in files_2])
galaxy_ids_3 = set([f.split('/')[-1].split('_')[0] for f in gfiles])

galaxy_ids = galaxy_ids_0 & galaxy_ids_1 & galaxy_ids_2 & galaxy_ids_3
del galaxy_ids_0, galaxy_ids_1, galaxy_ids_2, galaxy_ids_3

if verbose:
    print "Found", len(galaxy_ids), "galaxies."

# get galaxy ids
test_gals = glob.glob(test_dir + '*.jpg')
test_ids = set([f.split('/')[-1].split('.')[0] for f in test_gals])
train_gals = glob.glob(train_dir + '*.jpg')
train_ids = set([f.split('/')[-1].split('.')[0] for f in train_gals])

galaxy_ids_ref = test_ids | train_ids

if len(galaxy_ids_ref - galaxy_ids) != 0:
    print 'Missing data for the following galaxies:'
    print galaxy_ids_ref - galaxy_ids
    exit()

del galaxy_ids, galaxy_ids_ref

test_ids = list(test_ids)
train_ids = list(train_ids)


# load the DCT coefficients
def build_dct_array(galaxy_ids, ncoefs):

    X = np.zeros((len(galaxy_ids), ncoefs * 3), dtype=np.float32)
    print 'Loading data for source'
    for i, gal_id in enumerate(galaxy_ids):
        print i + 1
        dct_coefs = []
        for band in range(3):
            image_file = open(dct_dir + gal_id + '_' + str(band) + '_dct.pickle', 'rb')
            dct = cPickle.load(image_file)
            image_file.close()
            if len(dct.coefs) < ncoefs:
                nzeros = ncoefs - len(dct.coefs)
                dct.coefs = np.append(dct.coefs.astype(np.float32), np.zeros(nzeros))
            dct_coefs.append(dct.coefs[:ncoefs])

        X[i, :] = np.hstack(dct_coefs)

    return X

if do_build:
    print 'Getting training set...'
    dct_train = build_dct_array(train_ids, 2500)
    print 'Pickling training set...'
    cPickle.dump((train_ids, dct_train), open(base_dir + 'data/DCT_coefs_train.pickle', 'wb'))
    print 'Getting test set...'
    dct_test = build_dct_array(test_ids, 2500)
    cPickle.dump((test_ids, dct_test), open(base_dir + 'data/DCT_coefs_test.pickle', 'wb'))
else:
    print "Loading training set DCT..."
    train_ids, dct_train = cPickle.load(open(base_dir + 'data/DCT_coefs_train.pickle', 'rb'))
    print 'Loading test set DCT...'
    test_ids, dct_test = cPickle.load(open(base_dir + 'data/DCT_coefs_test.pickle', 'rb'))

print 'Reconstructing images...'

shape = (40, 40)
U = REACT2D.build_dct(shape[0], shape[1], 50).astype(np.float32)

train_images = np.zeros((len(train_ids), 3, shape[0], shape[1]), dtype=np.float32)

print '  Doing training set...'
ncoefs = 2500
for i in xrange(len(train_ids)):
    print '...', i
    train_images[i, 0, :, :] = np.dot(U[:, :ncoefs], dct_train[i, :ncoefs]).reshape(shape)
    train_images[i, 1, :, :] = np.dot(U[:, :ncoefs], dct_train[i, ncoefs:2*ncoefs]).reshape(shape)
    train_images[i, 2, :, :] = np.dot(U[:, :ncoefs], dct_train[i, 2*ncoefs:]).reshape(shape)

train_images = train_images.reshape((len(train_ids), 3 * shape[0] * shape[1]))

if doshow:
    train_images = train_images.reshape((len(train_ids), 3, shape[0], shape[1]))
    plt.subplot(321)
    plt.imshow(train_images[0, 0, :, :], cmap='hot')
    plt.title('Recon 1')
    image = np.load(train_dir + train_ids[0] + '_0.npy')
    plt.subplot(322)
    plt.imshow(image, cmap='hot')
    plt.title('Orig 1')
    plt.subplot(323)
    plt.imshow(train_images[0, 1, :, :], cmap='hot')
    plt.title('Recon 2')
    image = np.load(train_dir + train_ids[0] + '_1.npy')
    plt.subplot(324)
    plt.imshow(image, cmap='hot')
    plt.title('Orig 2')
    plt.subplot(325)
    plt.imshow(train_images[0, 2, :, :], cmap='hot')
    plt.title('Recon 3')
    image = np.load(train_dir + train_ids[0] + '_0.npy')
    plt.subplot(326)
    plt.imshow(image, cmap='hot')
    plt.title('Orig 3')
    plt.tight_layout()
    plt.show()
    train_images = train_images.reshape((len(train_ids), 3 * shape[0] * shape[1]))

del dct_train

print 'Pickling the training images...'
cPickle.dump((train_ids, train_images), open(base_dir + 'data/DCT_Images_train.pickle', 'wb'))

del train_images

print '  Doing test set...'
ncoefs = 2500
test_images = np.zeros((len(test_ids), 3, shape[0], shape[1]), dtype=np.float32)
for i in xrange(len(test_ids)):
    print '...', i
    test_images[i, 0, :, :] = np.dot(U[:, :ncoefs], dct_test[i, :ncoefs]).reshape(shape)
    test_images[i, 1, :, :] = np.dot(U[:, :ncoefs], dct_test[i, ncoefs:2*ncoefs]).reshape(shape)
    test_images[i, 2, :, :] = np.dot(U[:, :ncoefs], dct_test[i, 2*ncoefs:]).reshape(shape)

test_images = test_images.reshape((len(test_ids), 3 * shape[0] * shape[1]))

if doshow:
    test_images = test_images.reshape((len(test_ids), 3, shape[0], shape[1]))
    plt.subplot(321)
    plt.imshow(test_images[0, 0, :, :], cmap='hot')
    plt.title('Recon 1')
    image = np.load(test_dir + test_ids[0] + '_0.npy')
    plt.subplot(322)
    plt.imshow(image, cmap='hot')
    plt.title('Orig 1')
    plt.subplot(323)
    plt.imshow(test_images[0, 1, :, :], cmap='hot')
    plt.title('Recon 2')
    image = np.load(test_dir + test_ids[0] + '_1.npy')
    plt.subplot(324)
    plt.imshow(image, cmap='hot')
    plt.title('Orig 2')
    plt.subplot(325)
    plt.imshow(test_images[0, 2, :, :], cmap='hot')
    plt.title('Recon 3')
    image = np.load(test_dir + test_ids[0] + '_0.npy')
    plt.subplot(326)
    plt.imshow(image, cmap='hot')
    plt.title('Orig 3')
    plt.show()
    test_images = test_images.reshape((len(test_ids), 3 * shape[0] * shape[1]))

print 'Pickling the test images...'
cPickle.dump((test_ids, test_images), open(base_dir + 'data/DCT_Images_test.pickle', 'wb'))
