__author__ = 'brandonkelly'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
import os


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

    def fit(self, y, X=None, sigsqr=None):

        # check inputs
        if X is None:
            # build the discrete cosine basis
            if self.n_components is None:
                n_components = len(y)
            X = self.build_dct(len(y), n_components)
        else:
            if self.n_components is None:
                n_components = X.shape[1]

        try:
            n_components <= len(y)
        except ValueError:
            print 'Number of components must be less than the length of y.'

        self.X = X
        self.coefs = np.dot(X.T, y)

        if sigsqr is None:
            # estimate noise variance using first difference estimator
            sigsqr = np.sum((y[1:] - y[:-1]) ** 2) / (2.0 * (len(y) - 1))

        if self.method == 'monotone':
            # use monotone shrinkage on the basis coefficients
            print 'Getting shrinkage factors...'
            self._set_shrinkage_factors(sigsqr)
        else:
            # use nested subset selection to choose the order of the basis expansion
            self._set_nss_order(sigsqr)

        self.coefs *= self.shrinkage_factors

        ysmooth = X.dot(self.coefs)
        return ysmooth

    @staticmethod
    def build_dct(n, p):
        rows, columns = np.mgrid[:n, :p]
        U = np.cos(np.pi * rows * columns / (n - 1.0))
        row_norm = 2 * np.ones(n)
        row_norm[0] = 1.0
        row_norm[-1] = 1.0
        col_norm = 2 * np.ones(p)
        col_norm[0] = 1.0
        if p == n:
            col_norm[-1] = 1.0
        U *= 0.5 * np.sqrt(2.0 * np.outer(row_norm, col_norm) / (n - 1))

        return U

    def interpolate(self, x_idx):
        try:
            self.method.lower() == 'dct'
        except AttributeError:
            print 'Interpolation only available for DCT basis.'

        n = self.X.shape[0]
        p = self.X.shape[1]
        cols = np.arange(p)
        row_norm = 2 * np.ones(n)
        row_norm[0] = 1.0
        row_norm[-1] = 1.0
        col_norm = 2 * np.ones(p)
        col_norm[0] = 1.0
        U = np.cos(np.pi * np.outer(x_idx / n, cols))
        U *= 0.5 * np.sqrt(2.0 * np.outer(row_norm, col_norm) / (n - 1))
        y_interp = U.dot(self.coefs)
        return y_interp

    def _set_shrinkage_factors(self, sigsqr):
        coefs_snr = (self.coefs ** 2 - sigsqr) / self.coefs ** 2  # signal-to-noise ratio of the coefficients
        coefs_snr[coefs_snr < 0] = 0.0
        self.shrinkage_factors = self.Isotonic.fit_transform(np.arange(len(coefs_snr)), coefs_snr, self.coefs ** 2)

    def _set_nss_order(self, sigsqr):
        coefs_snr = (self.coefs ** 2 - sigsqr) / self.coefs ** 2  # signal-to-noise ratio of the coefficients
        coefs_snr[coefs_snr < 0] = 0.0
        risk = np.empty(len(coefs_snr))
        shrinkage_factor = np.zeros(len(coefs_snr))
        for j in xrange(len(risk)):
            shrinkage_factor[:j+1] = 1.0
            risk[j] = np.mean((shrinkage_factor - coefs_snr) ** 2 * self.coefs ** 2)
        best_order = risk.argmin()
        self.shrinkage_factors = np.ones(len(coefs_snr))
        self.shrinkage_factors[best_order:] = 0.0  # only keep first best_order basis coefficients


class REACT2D(REACT):
    # TODO: set n_components using 2-D analogue

    def interpolate(self, x_idx):
        super(REACT2D, self).interpolate(x_idx)

    @staticmethod
    def build_dct(nrows, ncols, p):
        # first build 1-D basis vectors
        Urows = super(REACT2D, REACT2D).build_dct(nrows, p)
        Ucols = super(REACT2D, REACT2D).build_dct(ncols, p)
        # now build 2-d basis as outer products of 1-d basis vectors
        row_order, col_order = np.mgrid[:p, :p]
        row_order = row_order.ravel()
        col_order = col_order.ravel()
        U = np.empty((nrows * ncols, len(row_order)))
        for j in xrange(len(row_order)):
            U[:, j] = np.outer(Urows[:, row_order[j]], Ucols[:, col_order[j]]).ravel()

        return U

    def fit(self, y, sigsqr, X=None):
        ysmooth = super(REACT2D, self).fit(y.ravel(), X, sigsqr)
        return np.reshape(ysmooth, y.shape)


if __name__ == "__main__":
    image = np.load(os.environ['HOME'] + '/Projects/Kaggle/galaxy_zoo/data/images_training_rev1/767521_0.npy')
    y = image[image.shape[0] / 2, :]

    smoother = REACT(method='monotone', n_components=200)
    ysmooth = smoother.fit(y)

    plt.plot(y, 'b.')
    plt.plot(ysmooth, 'r')
    plt.show()

    plt.plot(smoother.shrinkage_factors)
    plt.ylabel('Shrinkage Factor')
    plt.show()

    coefs = smoother.coefs
    plt.plot(np.sign(coefs) * np.sqrt(np.abs(coefs)), '.')
    plt.ylabel('Signed Root of DCT Coefficients')
    plt.show()