__author__ = 'brandonkelly'

import numpy as np
import matplotlib.pyplot as plt
import cPickle
import sys
import os
from react import REACT2D
from scipy.misc import bytescale

base_dir = os.environ['HOME'] + '/Projects/Kaggle/galaxy_zoo/'
dct_dir = base_dir + 'data/react/'

ncoefs = 2500


def pc_image(pca, shape, pc_idx):

    U = REACT2D.build_dct(shape[0], shape[1], 50)

    pca_images = np.empty((shape[0], shape[1], 3))

    pca_images[:, :, 0] = pca.components_[pc_idx, :ncoefs].dot(U.T[:ncoefs, :]).reshape((shape[0], shape[1]))
    pca_images[:, :, 1] = pca.components_[pc_idx, ncoefs:2*ncoefs].dot(U.T[:ncoefs, :]).reshape((shape[0], shape[1]))
    pca_images[:, :, 2] = pca.components_[pc_idx, 2*ncoefs:].dot(U.T[:ncoefs, :]).reshape((shape[0], shape[1]))

    plt.imshow(bytescale(pca_images))
    plt.title('PC ' + str(pc_idx + 1))
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    plt.tick_params(axis='y', which='both', bottom='off', top='off', labelbottom='off')
    plt.show()


if __name__ == "__main__":

    if len(sys.argv) < 2:
        sys.exit('Usage: %s (index of PC)' % sys.argv[0])

    npc = sys.argv[1]

    print 'Loading PCA...'
    pca = cPickle.load(open(base_dir + 'data/DCT_PCA.pickle', 'rb'))

    shape = (100, 100)
    pc_image(pca, shape, int(npc)-1)