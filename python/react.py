__author__ = 'brandonkelly'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression


class REACT(object):

    def __init__(self, basis='DCT', n_components=None, method='monotone'):
        try:
            basis.lower() in ['dct', 'manual']
        except ValueError:
            print 'Input basis must be either DCT or manual.'

        try:
            method.lower() in ['monotone', 'nss']
        except ValueError:
            print 'method must be either monotone or nss.'

        self.basis = basis
        self.n_components = n_components
        self.method = method
        self.Isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=False)
        self.coefs = np.zeros(1)
        self.shrinkage_factors = np.zeros(1)
        self.X = np.zeros((1, 1))
        super(REACT).__init__()

    def fit(self, y, X=None, sigsqr=None):

        if X is None:
            # build the discrete cosine basis
            if self.n_components is None:
                n_components = len(y)
            X = self.build_dct(len(y), n_components)
        else:
            if self.n_components is None:
                n_components = X.shape[1]

        self.X = X
        self.coefs = np.dot(X.T, y)

        if sigsqr is None:
            # estimate noise variance using first difference estimator
            sigsqr = np.sum((y[1:] - y[:-1]) ** 2) / (2.0 * (len(y) - 1))

        if self.method == 'monotone':
            # use monotone shrinkage on the basis coefficients
            self._set_shrinkage_factors(sigsqr)
        else:
            # use nested subset selection to choose the order of the basis expansion
            self._set_nss_order(sigsqr)

        self.coefs *= self.shrinkage_factors

        ysmooth = X.dot(self.coefs)
        return ysmooth

    @staticmethod
    def build_dct(n, p):
        U = np.empty((n, p))
        return U

    def predict(self, x):
        pass

    def _set_shrinkage_factors(self, sigsqr):
        coefs_snr = (self.coefs ** 2 - sigsqr) / self.coefs ** 2  # signal-to-noise ratio of the coefficients
        coefs_snr[coefs_snr < 0] = 0.0
        self.shrinkage_factors = self.Isotonic.fit_transform(np.arange(len(coefs_snr)), coefs_snr, self.coefs ** 2)

    def _set_nss_order(self, sigsqr):
        coefs_snr = (self.coefs ** 2 - sigsqr) / self.coefs ** 2  # signal-to-noise ratio of the coefficients
        coefs_snr[coefs_snr < 0] = 0.0
        risk = np.cumsum((np.ones(len(coefs_snr)) - coefs_snr) ** 2 * self.coefs ** 2)
        best_order = risk.argmin()
        self.shrinkage_factors = np.ones(len(coefs_snr))
        self.shrinkage_factors[best_order:] = 0.0  # only keep first best_order basis coefficients


class REACT2D(REACT):

    def predict(self, x):
        super(REACT2D, self).predict(x)

    @staticmethod
    def build_dct(n, p):
        pass

    def fit(self, y, sigsqr, X=None):
        ysmooth = super(REACT2D, self).fit(y.ravel(), X, sigsqr)
        return np.reshape(ysmooth, y.shape)