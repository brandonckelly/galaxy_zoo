__author__ = 'brandonkelly'

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


class ProbabilisticLDA(object):

    def __init__(self, n_components=None, priors=None):
        self.n_components = n_components
        self.priors = priors
        self.priors_ = np.zeros(1)
        self.means_ = np.zeros(1)
        self.covariance_ = np.zeros(1)
        self.classes_ = np.zeros(1)
        self.components_ = np.zeros(1)

    def fit(self, X, y):

        nsamples, nfeatures = X.shape
        nclasses = y.shape[1]
        self.classes_ = np.arange(nclasses)

        if self.priors is None:
            self.priors_ = y.sum(axis=0)
            self.priors_ /= self.priors_.sum()
        else:
            self.priors_ = self.priors

        # compute the class means and common covariance matrix
        means = []
        self.covariance_ = np.zeros((nfeatures, nfeatures))
        for k in range(nclasses):
            meank = np.sum(X * y[:, k][:, np.newaxis], axis=0) / np.sum(y[:, k])
            means.append(meank)
            self.covariance_ += np.dot((X - meank).T, (X - meank) * y[:, k][:, np.newaxis])
        self.means_ = np.asarray(means)
        self.covariance_ /= nsamples

        # do eigendecomposition of common covariance matrix
        evals, evects = linalg.eigh(self.covariance_)

        # compute between-class-covariance and transform
        mean = np.sum(self.priors_[:, np.newaxis] * self.means_, axis=0)
        between_covar = np.dot((self.means_ - mean).T, (self.means_ - mean) * self.priors_[:, np.newaxis])

        inv_cov_sqrt = evects * np.sqrt(1.0 / evals)  # V D^{-1/2}
        bcovar_transformed = inv_cov_sqrt.T.dot(between_covar.dot(inv_cov_sqrt))  # (W^{-1/2})^T B W^{-1/2}

        # do eigendecomposition of transformed between-class covariance matrix
        bevals, bevects = linalg.eigh(bcovar_transformed, eigvals=(self.means_.shape[1] - (nclasses - 1),
                                                                   self.means_.shape[1] - 1))

        # sort eigenvalues and vectors in descending order
        bevects = bevects[:, ::-1]

        # transform discriminant directions back to original space
        if self.n_components is None:
            self.n_components = nclasses - 1
        self.components_ = inv_cov_sqrt.dot(bevects[:, :self.n_components])

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def transform(self, X):
        return X.dot(self.components_)


if __name__ == "__main__":
    ndata = 6000
    ntest = 10000
    ndim = 750
    nclasses = 4

    priors = np.random.uniform(0.0, 1.0, nclasses)
    priors /= priors.sum()

    cov = 0.5 - np.random.rand(ndim ** 2).reshape((ndim, ndim))
    cov = np.triu(cov)
    cov += cov.T - np.diag(cov.diagonal())
    cov = np.dot(cov, cov)

    # cov = np.identity(ndim)

    means = np.random.multivariate_normal(np.zeros(ndim), cov / 10.0, nclasses)
    nx = np.random.multinomial(ndata, priors)
    nx_test = np.random.multinomial(ndata, priors)

    for k in range(nclasses):
        X_k = np.random.multivariate_normal(means[k, :], cov, nx[k])
        y_k = np.zeros((nx[k], nclasses))
        y_k[:, k] = 1.0
        X_k_test = np.random.multivariate_normal(means[k, :], cov, nx_test[k])
        y_k_test = np.zeros((nx_test[k], nclasses))
        y_k_test[:, k] = 1.0
        if k == 0:
            X = X_k.copy()
            y = y_k.copy()
            X_test = X_k_test.copy()
            y_test = y_k_test.copy()
        else:
            X = np.vstack((X, X_k))
            X_test = np.vstack((X_test, X_k_test))
            y = np.vstack((y, y_k))
            y_test = np.vstack(((y_test, y_k_test)))

        plt.plot(X_k[:, 0], X_k[:, 1], '.', label='Class ' + str(k+1))
    plt.legend(loc='best')
    plt.show()
    plt.clf()

    # compute probabilities when we don't know the labels
    yprob = np.zeros_like(y)
    cov_inv = linalg.inv(cov)
    for i in xrange(ndata):
        for k in range(nclasses):
            xcent = X[i, :] - means[k, :]
            chisqr = np.sum(xcent * np.dot(cov_inv, xcent))
            yprob[i, k] = np.exp(-0.5 * chisqr)
        yprob[i, :] /= np.sum(yprob[i, :])

    max_eps = np.max(np.abs(1.0 - yprob.sum(axis=1)))
    assert max_eps < 1e-6

    plda = ProbabilisticLDA()
    plda.fit(X, yprob)
    X_lda = plda.transform(X_test)
    n_components = X_lda.shape[1]

    for i in range(n_components):
        for j in range(i, n_components):
            if i == j:
                for k in range(nclasses):
                    class_idx = (y_test[:, k] > 0.99)
                    if sum(class_idx) < 100:
                        nbins = 10
                    else:
                        nbins = 25
                    plt.hist(X_lda[class_idx, i], bins=nbins, alpha=0.5)
                plt.xlabel('LDA ' + str(i+1))
                plt.show()
                plt.clf()
            else:
                for k in range(nclasses):
                    class_idx = (y_test[:, k] > 0.99)
                    plt.plot(X_lda[class_idx, i], X_lda[class_idx, j], '.')
                plt.xlabel('LDA ' + str(i+1))
                plt.ylabel('LDA ' + str(j+1))
                plt.show()
                plt.clf()
