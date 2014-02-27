__author__ = 'brandonkelly'

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


class PCR(object):

    def __init__(self, include_constant=True, whiten=True):
        self.pca = PCA(whiten=whiten)
        self.include_constant = include_constant
        self.coefs = np.zeros(1)
        self.sure = np.zeros(1)
        super(PCR).__init__()

    def fit(self, X, y, sigma=None):
        # first get PCA of X
        self.pca.fit(X)
