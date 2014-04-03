"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import cPickle
import gzip
import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from sklearn.cross_validation import train_test_split
from theano_linreg import LinearRegression
from ANN import HiddenLayer
import pandas as pd
from make_prediction_file import write_predictions

base_dir = os.environ['HOME'] + '/Projects/Kaggle/galaxy_zoo/'
data_dir = base_dir + 'data/'
dct_dir = data_dir + 'react/'
ann_dir = data_dir + 'nnets/'
plot_dir = base_dir + 'plots/'

do_standardize = False


def clean_features(df):

    for color in ['blue', 'green', 'red']:
        df[color].ix[df[color] == -9999] = df[color].median()

    df['GalaxyCentDist'].ix[df['GalaxyCentDist'] == -9999] = np.log(0.5)

    # standardize inputs for non-PC variables
    mad = (df - df.median()).abs().median()
    df -= df.median()
    df /= 1.5 * mad

    return df


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(poolsize))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(np.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W, filter_shape=filter_shape, image_shape=image_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


def evaluate_lenet5(learning_rate=0.01, n_epochs=100, nkerns=[20, 50], batch_size=500):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    ann_id = 'CNN_lrate' + str(learning_rate)
    if do_standardize:
        ann_id += '_standardzed'

    rng = np.random.RandomState()

    print 'Loading data...'

    # load the training data for the features
    df = pd.read_hdf(base_dir + 'data/galaxy_features.h5', 'df')

    train_id, images = cPickle.load(open(data_dir + 'DCT_Images_train_short.pickle', 'rb'))
    train_id = np.asarray(train_id, dtype=np.int)
    features = ['blue', 'red', 'green', 'GalaxyCentDist', 'GalaxyMajor', 'GalaxyAratio', 'GalaxyFlux']
    df = clean_features(df[features])
    ishape = (40, 40)  # this is the size of galaxy images
    images = images.reshape((len(train_id), 3, ishape[0], ishape[1]))

    # normalize inputs
    if do_standardize:
        images -= images.mean(axis=0)
        images /= images.std(axis=0)
    else:
        images -= images.min(axis=0)
        images /= images.max(axis=0)  # in range [0,1]
        images -= 0.5  # in range [-0.5, 0.5]
        images *= 2.0  # in range [-1.0, 1.0]
        print 'Image Range: ', images.min(), images.max()

    images = images.reshape((len(train_id), 3 * ishape[0] * ishape[1]))

    idx = np.arange(images.shape[0])

    train_idx, valid_idx = train_test_split(idx)

    y_df = pd.read_csv(base_dir + 'data/training_solutions_rev1.csv').set_index('GalaxyID')
    y = y_df.ix[train_id].values

    train_set_x = theano.shared(np.asarray(images[train_idx], dtype=theano.config.floatX), borrow=True)
    ftrain_set_x = theano.shared(np.asarray(df.ix[train_id].values[train_idx], dtype=theano.config.floatX), borrow=True)
    train_set_y = theano.shared(np.asarray(y[train_idx], dtype=theano.config.floatX), borrow=True)
    valid_set_x = theano.shared(np.asarray(images[valid_idx], dtype=theano.config.floatX), borrow=True)
    fvalid_set_x = theano.shared(np.asarray(df.ix[train_id].values[valid_idx], dtype=theano.config.floatX), borrow=True)
    valid_set_y = theano.shared(np.asarray(y[valid_idx], dtype=theano.config.floatX), borrow=True)

    assert np.all(np.isfinite(train_set_x.get_value()))
    assert np.all(np.isfinite(ftrain_set_x.get_value()))
    assert np.all(np.isfinite(valid_set_x.get_value()))
    assert np.all(np.isfinite(train_set_y.get_value()))
    assert np.all(np.isfinite(fvalid_set_x.get_value()))
    assert np.all(np.isfinite(valid_set_y.get_value()))

    nout = y.shape[1]

    # test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    # n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    # n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')
    x_extra = T.matrix('x_extra')
    y = T.matrix('y')  # the outputs are a matrix

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size,28*28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # layer0_input = x.reshape((batch_size, 3, ishape[0], ishape[1]))

    layer0_input = x.reshape((batch_size, 3, 40, 40))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (40-5+1,40-5+1)=(36,36)
    # maxpooling reduces this further to (36/2,36/2) = (18,18)
    # 4D output tensor is thus of shape (batch_size,nkerns[0],18,18)
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input, image_shape=(batch_size, 3, 40, 40),
                                filter_shape=(nkerns[0], 3, 5, 5), poolsize=(2, 2))

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (18-5+1,18-5+1)=(14,14)
    # maxpooling reduces this further to (14/2,14/2) = (7,7)
    # 4D output tensor is thus of shape (nkerns[0],nkerns[1],7,7)
    layer1 = LeNetConvPoolLayer(rng, input=layer0.output, image_shape=(batch_size, nkerns[0], 18, 18),
                                filter_shape=(nkerns[1], nkerns[0], 5, 5), poolsize=(2, 2))

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (20,32*4*4) = (20,512)
    layer2_input = T.concatenate([layer1.output.flatten(2), x_extra], axis=1)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[1] * 7 * 7 + len(df.columns),
                         n_out=500, activation=T.tanh)

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LinearRegression(input=layer2.output, n_in=500, n_out=nout)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.squared_error(y)

    # create a function to compute the mistakes that are made by the model
    # test_model = theano.function([index], layer3.errors(y),
    #                              givens={x: test_set_x[index * batch_size: (index + 1) * batch_size],
    #                                      y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function([index], layer3.errors(y),
                                     givens={x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                                             x_extra: fvalid_set_x[index * batch_size: (index + 1) * batch_size],
                                             y: valid_set_y[index * batch_size: (index + 1) * batch_size]})

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i],grads[i]) pairs.
    updates = []
    for param_i, grad_i in zip(params, grads):
        updates.append((param_i, param_i - learning_rate * grad_i))

    train_model = theano.function([index], cost, updates=updates,
                                  givens={x: train_set_x[index * batch_size: (index + 1) * batch_size],
                                          x_extra: ftrain_set_x[index * batch_size: (index + 1) * batch_size],
                                          y: train_set_y[index * batch_size: (index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
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
    best_validation_loss = np.inf
    best_iter = 0
    # test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f' % (epoch, minibatch_index + 1, n_train_batches,
                                                                          this_validation_loss))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    best_params = [p.get_value(borrow=False) for p in params]

                    if epoch > 100:
                        print 'Storing values...'
                        cPickle.dump(best_params, open(ann_dir + ann_id + '_best_params.pickle', 'wb'))

                    # test it on the test set
                    # test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    # test_score = np.mean(test_losses)
                    # print(('     epoch %i, minibatch %i/%i, test error of best '
                    #        'model %f %%') %
                    #       (epoch, minibatch_index + 1, n_train_batches,
                    #        test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f obtained at iteration %i,' % (best_validation_loss, best_iter + 1))
    print ('The code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time) / 60.))

    print 'Restoring best values...'
    current_params = [p.get_value(borrow=True) for p in params]
    for i in xrange(len(params)):
        params[i].set_value(best_params[i], borrow=True)

    print 'Storing values...'
    cPickle.dump(best_params, open(ann_dir + ann_id + '_best_params.pickle', 'wb'))
    cPickle.dump(current_params, open(ann_dir + ann_id + '_last_params.pickle', 'wb'))

    print 'Writing predictions...'
    # predict future data
    n_train = len(train_id)
    if n_train % batch_size > 0:
        # need to pad the array to be a multiple of batchsize
        n_extra = batch_size - n_train % batch_size
        padded_data = np.zeros((n_extra, 3 * ishape[0] * ishape[1]))
        images_predict = np.append(images, padded_data, axis=0)
        padded_data = np.zeros((n_extra, len(df.columns)))
        features_predict = np.append(df.ix[train_id].values, padded_data, axis=0)
    else:
        images_predict = images

    n_train_batches = images_predict.shape[0] / batch_size
    new_data_x = theano.shared(np.asarray(images_predict, dtype=theano.config.floatX), borrow=True)
    fnew_data_x = theano.shared(np.asarray(features_predict, dtype=theano.config.floatX), borrow=True)
    predict = theano.function([index], layer3.y_pred,
                              givens={x: new_data_x[index * batch_size: (index + 1) * batch_size],
                                      x_extra: fnew_data_x[index * batch_size: (index + 1) * batch_size]})

    y_predict = np.vstack([predict(i) for i in xrange(n_train_batches)])
    print y_predict.shape

    y_predict = pd.DataFrame(data=y_predict[:n_train], index=train_id, columns=y_df.columns)
    y_predict.index.name = 'GalaxyID'
    y_predict[y_predict < 0] = 0.0
    y_predict[y_predict > 1] = 1.0
    y_predict.to_csv(data_dir + ann_id + '_predictions_train.csv')

    write_predictions(y_predict, ann_id + '_train')

    test_id, images = cPickle.load(open(data_dir + 'DCT_Images_test_short.pickle', 'rb'))
    test_id = np.asarray(test_id, dtype=np.int)
    images = images.reshape((len(test_id), 3, ishape[0], ishape[1]))
    # normalize inputs
    if do_standardize:
        images -= images.mean(axis=0)
        images /= images.std(axis=0)
    else:
        images -= images.min(axis=0)
        images /= images.max(axis=0)  # in range [0,1]
        images -= 0.5  # in range [-0.5, 0.5]
        images *= 2.0  # in range [-1.0, 1.0]
    images = images.reshape((len(test_id), 3 * ishape[0] * ishape[1]))

    # predict future data
    n_test = len(test_id)
    if n_test % batch_size > 0:
        # need to pad the array to be a multiple of batchsize
        n_extra = batch_size - n_test % batch_size
        padded_data = np.zeros((n_extra, 3 * ishape[0] * ishape[1]))
        images_predict = np.append(images, padded_data, axis=0)
        padded_data = np.zeros((n_extra, len(df.columns)))
        features_predict = np.append(df.ix[test_id].values, padded_data, axis=0)
    else:
        images_predict = images

    n_test_batches = images_predict.shape[0] / batch_size
    test_data_x = theano.shared(np.asarray(images_predict, dtype=theano.config.floatX), borrow=True)
    ftest_data_x = theano.shared(np.asarray(features_predict, dtype=theano.config.floatX), borrow=True)
    test_predict = theano.function([index], layer3.y_pred,
                              givens={x: test_data_x[index * batch_size: (index + 1) * batch_size],
                                      x_extra: ftest_data_x[index * batch_size: (index + 1) * batch_size]})

    y_predict = np.vstack([test_predict(i) for i in xrange(n_test_batches)])
    print y_predict.shape

    y_predict = pd.DataFrame(data=y_predict[:len(test_id)], index=test_id, columns=y_df.columns)
    y_predict.index.name = 'GalaxyID'
    y_predict[y_predict < 0] = 0.0
    y_predict[y_predict > 1] = 1.0
    y_predict.to_csv(data_dir + ann_id + '_predictions_test.csv')

    write_predictions(y_predict, ann_id + '_test')


if __name__ == '__main__':
    evaluate_lenet5()