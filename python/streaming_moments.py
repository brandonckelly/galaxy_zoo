__author__ = 'brandonkelly'

import numpy as np
import matplotlib.pyplot as plt

class StreamingMoments(object):

    def __init__(self, dim):
        self.dim = dim
        if dim == 1:
            self.mean = 0.0
            self.var = 0.0
            self.ssqr = 0.0
        else:
            self.mean = np.zeros(dim)
            self.var = np.zeros((dim, dim))
            self.ssqr = np.zeros((dim, dim))
        self.nx = 0

    def __call__(self, x):
        try:
            len(x) == self.dim
        except ValueError:
            print "Length of input must be", self.dim
            return
        self.nx += 1

        self.mean = (self.mean * (self.nx - 1) + x) / self.nx

        if self.dim == 1:
            self.ssqr = (self.ssqr * (self.nx - 1) + x ** 2) / self.nx
            self.var = self.ssqr - self.mean ** 2
        else:
            self.ssqr = (self.ssqr * (self.nx - 1) + np.outer(x, x)) / self.nx
            self.var = self.ssqr - np.outer(self.mean, self.mean)

        return self.mean, self.var


if __name__ == "__main__":
    # simple test
    ndata = 100000
    mfeat = 10
    means = np.random.rand(mfeat)

    cov = 0.5 - np.random.rand(mfeat ** 2).reshape((mfeat, mfeat))
    cov = np.triu(cov)
    cov += cov.T - np.diag(cov.diagonal())
    cov = np.dot(cov, cov)

    stream = np.random.multivariate_normal(np.zeros(mfeat), cov, ndata)

    batch_mean = stream.mean(axis=0)
    batch_covar = np.cov(stream.T)

    smom = StreamingMoments(mfeat)

    for i in xrange(ndata):
        stream_mean, stream_covar = smom(stream[i, :])

    print "Batch mean:"
    print batch_mean
    print ''
    print 'Streaming mean:'
    print stream_mean
    print ''
    print 'Batch Covar:'
    print batch_covar
    print ''
    print 'Streaming Covar:'
    print stream_covar
    print ''
    print 'Fractional difference in Covariances:'
    print np.abs(batch_covar - stream_covar) / np.abs(batch_covar)

    plt.plot(batch_covar.ravel(), '.-', label='Batch')
    plt.plot(stream_covar.ravel(), '.-', label='Stream')
    plt.plot(batch_covar.ravel() - stream_covar.ravel(), '.-', label='Difference')
    plt.legend(loc='best')
    plt.show()