
__docformat__ = 'restructedtext en'
__author__ = 'brandonkelly'

import cPickle
import os
import sys
import time

import numpy

import theano
import theano.tensor as T

from sklearn.cross_validation import train_test_split


class LinearRegression(object):
    """Linear Regression Class

    The linear regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the linear regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX), name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=numpy.zeros((n_out,), dtype=theano.config.floatX), name='b', borrow=True)

        # compute vector of predicted response in symbolic form
        self.y_pred = T.dot(input, self.W) + self.b

        # parameters of the model
        self.params = [self.W, self.b]

    def squared_error(self, y):
        """Return the squared error.

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct response.

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        err = y - self.y_pred
        return T.mean((err * err).sum(axis=1))

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('float'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return self.squared_error(y)
        else:
            raise NotImplementedError()


def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, shared_y


def sgd_optimization_mnist(X_train, X_valid, y_train, y_valid, learning_rate=0.13, n_epochs=1000,
                           batch_size=600):
    """
    Demonstrate stochastic gradient descent optimization of a linear regression model.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    """

    # compute number of minibatches for training, validation and testing
    n_train_batches = X_train.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = X_valid.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.matrix('y')  # the multi-output response is a matrix

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    regressor = LinearRegression(input=x, n_in=X_train.get_value(borrow=True).shape[1],
                                 n_out=y_train.get_value(borrow=True).shape[1])

    # the cost we minimize during training is the squared error of
    # the model in symbolic format
    cost = regressor.squared_error(y)

    validate_model = theano.function(inputs=[index], outputs=regressor.errors(y),
                                     givens={x: X_valid[index * batch_size:(index + 1) * batch_size],
                                             y: y_valid[index * batch_size:(index + 1) * batch_size]})

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=regressor.W)
    g_b = T.grad(cost=cost, wrt=regressor.b)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(regressor.W, regressor.W - learning_rate * g_W),
               (regressor.b, regressor.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(inputs=[index], outputs=cost, updates=updates,
                                  givens={x: X_train[index * batch_size:(index + 1) * batch_size],
                                          y: y_train[index * batch_size:(index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                    (epoch, minibatch_index + 1, n_train_batches,
                    this_validation_loss))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete with best validation score of %f' % best_validation_loss)
    print 'The code run for %d epochs, with %f epochs/sec' % (epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))

    return regressor

if __name__ == '__main__':

    ndata = 10000
    nouts = 5
    nfeatures = 10

    sigma = 1.0

    alpha = numpy.random.uniform(-2.0, 2.0, nouts)
    betas = numpy.random.standard_normal((nfeatures, nouts))

    means = numpy.random.standard_normal(nfeatures)
    cov = 0.5 - numpy.random.rand(nfeatures ** 2).reshape((nfeatures, nfeatures))
    cov = numpy.triu(cov)
    cov += cov.T - numpy.diag(cov.diagonal())
    cov = numpy.dot(cov, cov)

    X = numpy.random.multivariate_normal(means, cov, ndata)

    y = X.dot(betas) + alpha + sigma * numpy.random.standard_normal((ndata, nouts))

    train_set_x, valid_set_x, train_set_y, valid_set_y = train_test_split(X, y)

    valid_set_x, valid_set_y = shared_dataset((valid_set_x, valid_set_y))
    train_set_x, train_set_y = shared_dataset((train_set_x, train_set_y))

    print train_set_y.get_value().shape
    print valid_set_y.get_value().shape
    print valid_set_x.get_value().shape
    print train_set_x.get_value().shape

    linreg = sgd_optimization_mnist(train_set_x, valid_set_x, train_set_y, valid_set_y, learning_rate=0.01)

    ahat = linreg.b.get_value()
    Bhat = linreg.W.get_value()

    print 'True value of the constant:'
    print alpha
    print 'Estimated value of the constant:'
    print ahat
    print ''

    print 'True value of the slopes:'
    print betas
    print 'Estimated value of the slopes:'
    print Bhat

    print 'Difference:'
    print numpy.abs(Bhat - betas).astype(numpy.float32)