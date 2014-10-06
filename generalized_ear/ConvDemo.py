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

import numpy

import theano
import theano.tensor as T
# from theano.tensor.signal import downsample
# from theano.tensor.nnet import conv

from logistic_sgd import load_data
from collections import OrderedDict

from theano.sandbox.cuda.basic_ops import gpu_contiguous
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from pylearn2.sandbox.cuda_convnet.pool import MaxPool
from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams

from NetLayers import ConvPoolLayer, HiddenLayer, relu_actfun, \
                      Reshape2D4DLayer, Reshape4D2DLayer
from output_losses import LogisticRegression
from utils import visualize_samples

def evaluate_lenet5(learning_rate=0.05, n_epochs=500,
                    dataset='./data/mnist.pkl.gz',
                    nkerns=[48, 64], batch_size=256):
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

    rng = numpy.random.RandomState(23455)

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ishape = (28, 28)  # this is the size of MNIST images

    start_rate = numpy.asarray([0.05]).astype(theano.config.floatX)
    learning_rate = theano.shared(value=start_rate, name='learning_rate')

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    tanh = lambda vals: T.tanh(vals)
    relu = lambda vals: relu_actfun(vals)

    # Reshape matrix of rasterized images of shape (batch_size,28*28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_prep = Reshape2D4DLayer(input=x, out_shape=(1, 28, 28))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-7+1,28-7+1)=(22,22)
    # maxpooling reduces this further to (22/2,22/2) = (11,11)
    # 4D output tensor is thus of shape (batch_size,nkerns[0],11,11)
    layer0 = ConvPoolLayer(rng, input=layer0_prep.output, \
            filt_def=(nkerns[0], 1, 7, 7), pool_def=(2, 2), \
            activation=relu, drop_rate=0.0, input_noise=0.1, bias_noise=0.05, \
            W=None, b=None, name="layer0", W_scale=2.0)

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (11-4+1,11-4+1)=(8,8)
    # maxpooling reduces this further to (8/2,8/2) = (4,4)
    # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
    layer1 = ConvPoolLayer(rng, input=layer0.output, \
            filt_def=(nkerns[1], nkerns[0], 4, 4), pool_def=(2, 2), \
            activation=relu, drop_rate=0.0, input_noise=0.0, bias_noise=0.05, \
            W=None, b=None, name="layer1", W_scale=2.0)

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (20,32*4*4) = (20,512)
    layer2_prep = Reshape4D2DLayer(layer1.output)

    # construct a fully-connected relu layer
    layer2 = HiddenLayer(rng, layer2_prep.output, nkerns[1]*4*4, 512, \
                 activation=relu, pool_size=0, \
                 drop_rate=0.0, input_noise=0.0, bias_noise=0.05, \
                 W=None, b=None, name="layer2", W_scale=2.0)

    # construct an output layer to predict classes
    layer3 = HiddenLayer(rng, layer2.output, 512, 10, \
             activation=relu, pool_size=0, \
             drop_rate=0.5, input_noise=0.0, bias_noise=0.0, \
             W=None, b=None, name="layer2", W_scale=2.0)

    # get a loss function to apply to the output layer
    loss_func = LogisticRegression(layer3)

    # the cost we minimize during training is the NLL of the model
    cost = loss_func.loss_func(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function([index], loss_func.errors(y),
             givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function([index], loss_func.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]})

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params
    moms = OrderedDict()
    for p in params:
        moms[p] = theano.shared(value=numpy.zeros( \
            p.get_value(borrow=True).shape).astype(theano.config.floatX))

    # create a list of gradients for all model parameters
    grads = OrderedDict()
    for p in params:
        grads[p] = T.grad(cost, p)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i],grads[i]) pairs.
    updates = []
    for p in params:
        mom_update = (moms[p], (0.8 * moms[p]) + (0.2 * grads[p]))
        param_update = (p, p - learning_rate[0] * moms[p])
        updates.append(mom_update)
        updates.append(param_update)

    train_model = theano.function([index], cost, updates=updates,
          givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
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
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
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
                this_validation_loss = numpy.mean(validation_losses) / batch_size
                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                      (epoch, minibatch_index + 1, n_train_batches, \
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses) / batch_size
                    print(('     epoch %i, minibatch %i/%i, test error of best '
                           'model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

        if ((epoch+1 % 30) == 0):
            new_rate = 0.5 * learning_rate.get_value(borrow=True)
            learning_rate.set_value(new_rate)

        if ((epoch % 10) == 0):
            W_l0 = layer0.W.get_value(borrow=False)
            W_l0 = W_l0.reshape((W_l0.shape[0], numpy.prod(W_l0.shape[1:])))
            visualize_samples(W_l0, 'A1_CONV_FILTS.png')

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i,'\
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    evaluate_lenet5()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
