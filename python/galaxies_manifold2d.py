__author__ = 'brandonkelly'

import numpy as np
from scipy.misc import bytescale
import matplotlib.pyplot as plt
import cPickle
from sklearn import manifold
import os
import pandas as pd
from matplotlib.pylab import rcParams
from scipy.misc import bytescale

base_dir = os.environ['HOME'] + 'Projects/Kaggle/galaxy_zoo/'

# make big figures
rcParams['figure.figsize'] = 12.0, 9.0


def remove_outliers(df):
    zscore = (df - df.mean()) / df.std()
    znorm = np.norm(zscore, axis=1)
    return df[znorm < 300]


def get_galaxy_images(galaxy_id):
    images = []
    for c in range(3):
        fname = base_dir + 'data/images_training_rev1/' + str(galaxy_id) + '_' + str(c) + '.npy'
        images.append(np.load(fname))
    images = np.dstack(images)

    return bytescale(images)  # scale the arrays for nice color representation of the image


def plot_galaxies(X, galaxy_ids):
    # first plot the distribution in the 2-d manifold
    plt.plot(X[:, 0], X[:, 1], 'o', ms=2, rasterized=True, alpha=0.5)
    plt.subplots_adjust(bottom=0.05, top=0.95, left=0.05, right=0.95)
    ax = plt.gca()
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xlim((-0.1, 0.1))
    ax.set_ylim((-0.2, 0.2))
    plt.title('Laplacian Eigenmap Embedding')

    # now show galaxy images within this manifold
    anchor_pts = [(-0.08, -0.03), (-0.05, -0.02), (-0.025, -0.0), (0.0, 0.0), (0.1, 0.01), (0.25, 0.04),
                  (0.05, 0.075), (0.08, 0.1), (0.08, 0.0), (0.08, -0.1), (0.03, -0.05)]
    for anchor in anchor_pts:
        # find the galaxy that is closest to this anchor point
        dist = (X[:, 0] - anchor[0]) ** 2 + (X[:, 1] - anchor[1]) ** 2
        idx = dist.argmin()
        # plot the galaxy image
        gimage = get_galaxy_images(galaxy_ids[idx])
        ax_i = plt.axes([0.1, 0.1, 0.1, 0.1])
        ax_i.imshow(gimage)


if __name__ == "__main__":

    # load the data
    df = pd.read_hdf(base_dir + 'data/DCT_coefs.h5', 'df')

    # remove outliers
    df = remove_outliers(df)

    # do spectral embedding
    train = False
    if train:
        embedding = manifold.SpectralEmbedding(n_neighbors=15)
        X = embedding.fit_transform(df.values[:2500, :50])
        cPickle.dump(embedding, open(base_dir + 'data/spectral_embedding.pickle', 'wb'))
    else:
        embedding = cPickle.load(open(base_dir + 'data/spectral_embedding.pickle', 'rb'))
        X = embedding.embedding_

    # now plot the results
    plot_galaxies(X, df.index)