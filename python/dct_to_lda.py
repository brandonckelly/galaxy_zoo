__author__ = 'brandonkelly'

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import cPickle
import os
import glob
from react import REACT2D
import triangle
from scipy.misc import bytescale
import pandas as pd
from sklearn.lda import LDA


class ProbabilisticLDA(LDA):

    def fit(self, X, y, tol=1.0e-4):
        nsamples, nfeatures = X.shape
        nclasses = y.shape[1]
        self.classes_ = np.arange(nclasses)

        if self.priors is None:
            self.priors_ = y.sum(axis=0)
        else:
            self.priors_ = self.priors

        # compute the class means
        means = []
        self.covariance_ = np.zeros((nfeatures, nfeatures))
        for k in range(nclasses):
            meank = np.sum(X * y[:, k][:, np.newaxis], axis=0)
            means.append(meank)
            self.covariance_ += np.dot((X - meank).T, (X - meank) * y[:, k][:, np.newaxis])
        self.means_ = np.asarray(means)

        return super(ProbabilisticLDA, self).fit(X, y, store_covariance, tol)

    def fit_transform(self, X, y=None, **fit_params):
        return super(ProbabilisticLDA, self).fit_transform(X, y, **fit_params)