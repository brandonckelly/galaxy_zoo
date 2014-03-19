__author__ = 'brandonkelly'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy import linalg
import triangle
import time
from memory_profiler import profile


def outlier_pursuit(X, outfrac, verbose=False):
    # algorithm hyper parameters
    eta = 0.9
    delta = 1e-5
    lamb = 3.0 / (7.0 * np.sqrt(outfrac * X.shape[1]))
    mu = 0.99 * linalg.norm(X, ord=2)
    mubar = delta * mu
    tol = 1e-6 * linalg.norm(X)

    # initial values
    niter = 0
    data_mat = np.zeros(X.shape)
    data_mat_old = np.zeros(X.shape)
    outliers_mat = np.zeros(X.shape)
    outliers_mat_old = np.zeros(X.shape)
    t = 1.0
    t_old = 1.0

    X_cent = np.median(X, axis=0)  # get data centroids

    # run the algorithm
    t1 = time.clock()
    converged = False
    if verbose:
        print 'Doing iteration ...'

    while not converged:
        if verbose:
            print niter
        Y_data = data_mat + (t_old - 1.0) / t * (data_mat - data_mat_old)
        Y_out = outliers_mat + (t_old - 1.0) / t * (outliers_mat - outliers_mat_old)

        # first update the uncontaminated data matrix
        data_mat_old = data_mat
        data_mat = 0.5 * Y_data - 0.5 * Y_out - (X - X_cent)  # temporary variable, save memory
        U, S, V = linalg.svd(data_mat, full_matrices=False, overwrite_a=True)
        S_shrunk = soft_threshold(S, mu / 2.0)  # shrink the singular values
        data_mat = U.dot(np.dot(np.diag(S_shrunk), V))

        # now update the outlier matrix
        outliers_mat_old = outliers_mat
        outliers_mat = 0.5 * Y_out - 0.5 * Y_data - (X - X_cent)  # temporary variable, save memory
        outliers_mat = col_soft_threshold(outliers_mat, lamb * mu / 2.0)

        # update the algorithm parameters
        t_old = t
        t = (1.0 + np.sqrt(4.0 * t_old ** 2 + 1.0)) / 2.0
        mu = max(eta * mu, mubar)

        niter += 1

        # check for convergence
        stop_data = 2 * (Y_data - data_mat) + (data_mat + outliers_mat - Y_data - Y_out)
        stop_out = 2 * (Y_out - outliers_mat) + (data_mat + outliers_mat - Y_data - Y_out)

        epsilon = linalg.norm(stop_data) ** 2 + linalg.norm(stop_out) ** 2
        if epsilon < tol ** 2:
            converged = True

    t2 = time.clock()
    if verbose:
        print 'Outlier pursuit convergence reached in', niter, 'iterations.'
        print 'Total time (sec):', t2 - t1

    return data_mat, outliers_mat


def soft_threshold(x, shrinkage):
    x_shrunk = x - np.sign(x) * shrinkage
    x_shrunk[np.abs(x) < shrinkage] = 0.0
    return x_shrunk


def col_soft_threshold(A, shrinkage):
    colnorm = np.sqrt(np.sum(A ** 2, axis=0))
    zero_idx = np.where(colnorm < shrinkage)[0]
    A_shrunk = A * (1.0 - shrinkage / colnorm)
    A_shrunk[:, zero_idx] = 0.0
    return A_shrunk


def row_soft_threshold(A, shrinkage):
    row_norm = np.sqrt(np.sum(A ** 2, axis=1))
    zero_idx = np.where(row_norm < shrinkage)[0]
    A_shrunk = A * (1.0 - shrinkage / row_norm[:, np.newaxis])
    A_shrunk[zero_idx, :] = 0.0
    return A_shrunk


class RobustPCA(PCA):

    def __init__(self, n_components=None, copy=True, whiten=False, noutliers=1, verbose=False):
        super(RobustPCA, self).__init__(n_components, copy, whiten)
        self.verbose = verbose
        self.noutliers = float(noutliers)
        self.outliers = np.zeros(1)

    def _fit(self, X):
        # first find outliers and remove them
        out_idx = self.find_outliers(X)
        self.outliers = out_idx
        if len(out_idx) == X.shape[0]:
            # everything is an outlier, just do normal PCA
            print 'Warning: Outlier pursuit estimates all data are outliers, just doing normal PCA.'
            X_cleaned = X
        else:
            X_cleaned = np.delete(X, out_idx, axis=0)
        return super(RobustPCA, self)._fit(X_cleaned)

    def fit_transform(self, X, y=None):
        self._fit(X)
        return self.transform(X)

    def find_outliers(self, X):
        # do outliers pursuit algorithm to find outliers
        data_matrix, out_matrix = outlier_pursuit(X.T, self.noutliers / X.shape[0], self.verbose)
        outliers = np.where(np.all(abs(out_matrix) > 0, axis=0))[0]
        return outliers


if __name__ == "__main__":
    # run an example
    outfrac = 0.10
    ndata = 10000

    ndim = 500
    means = np.random.standard_normal(ndim)
    cov = 0.5 - np.random.rand(ndim ** 2).reshape((ndim, ndim))
    cov = np.triu(cov)
    cov += cov.T - np.diag(cov.diagonal())
    cov = np.dot(cov, cov)

    X_clean = np.random.multivariate_normal(means, cov, ndata)

    ncontam = int(outfrac * ndata)

    outcov = cov.diagonal().copy()
    outcov[:30] *= 64.0
    outcov = np.diag(outcov)

    contaminants = np.random.multivariate_normal(means, outcov, ncontam)

    c_idx = np.random.permutation(ndata)[:ncontam]
    X = X_clean.copy()
    X[c_idx, :] = contaminants

    if ndim < 8:
        fig = triangle.corner(X)
        plt.show()

    X_cent = X.copy()
    X_cent -= np.median(X, axis=0)

    row_norm = np.sqrt(np.sum(X_cent ** 2, axis=1))
    mad = np.median(np.abs(row_norm - np.median(row_norm)))
    robsig = 1.48 * mad
    zscore = np.abs(row_norm - np.median(row_norm)) / robsig
    plt.hist(zscore, bins=100)
    gamma = len(np.where(zscore > 8)[0]) / float(ndata)
    gamma = max(gamma, 1.0 / ndata)
    print "Initial guess at outlier fraction is", gamma
    plt.show()

    # do_op = False
    # if do_op:
    #     L, C, term, niter = opursuit(X_cent.T, gamma=gamma)
    #     np.save('rpca_ref', C)
    #     out = []
    #     for i in xrange(C.shape[1]):
    #         is_out = np.any(C[:, i] != 0.0)
    #         if is_out:
    #             out.append(i)
    # 
    #     print 'Found', len(out), 'outliers:'
    #     print out
    #     print 'True', len(c_idx), 'outliers:'
    #     print np.sort(c_idx)
    #
    #     plt.plot(np.sqrt(np.sum(C ** 2, axis=0)), 'b.', label='C')
    #     plt.plot(np.sqrt(np.sum(X_cent ** 2, axis=1)), 'r.', label='X')
    #     plt.legend(loc='best')
    #     plt.show()
    #     # exit()

    pca = PCA(n_components=5)
    rpca = RobustPCA(verbose=True, n_components=5, noutliers=18.0)
    X_pca = pca.fit_transform(X)
    print 'Fitting Robust PCA...'
    X_rpca = rpca.fit_transform(X)

    print 'True', len(c_idx), 'indices of contaminants:'
    print np.sort(c_idx)

    print 'Derived', len(rpca.outliers), 'indices of contaminants:'
    print rpca.outliers

    plt.plot(pca.explained_variance_ratio_.cumsum(), label='PCA')
    plt.plot(rpca.explained_variance_ratio_.cumsum(), label='RPCA')
    plt.ylabel('Cumulative Fractional Explained Variance')
    plt.xlabel('Number of Components')
    plt.legend(loc='best')
    plt.show()

    labels = []
    for i in range(ndim):
        labels.append('PC ' + str(i+1))

    fig = triangle.corner(X_pca, labels=labels)
    plt.show()

    labels = []
    for i in range(ndim):
        labels.append('RPC ' + str(i+1))

    fig = triangle.corner(X_rpca, labels=labels)
    plt.show()