import numpy as np
import cPickle
import gzip
import os
import sys
import time
from collections import OrderedDict

import theano
import theano.tensor as T
from theano.ifelse import ifelse
import theano.printing
import theano.tensor.shared_randomstreams

import utils as utils
from output_losses import LogisticRegression, MCL2Hinge
from load_data import load_umontreal_data, load_mnist


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out,
                 activation, W=None, b=None,
                 use_bias=False):

        self.input = input
        self.activation = activation

        if W is None:
            W_values = np.asarray(0.01 * rng.standard_normal(
                size=(n_in, n_out)), dtype=theano.config.floatX)
            W = theano.shared(value=W_values, name='W')

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX) + 0.001
            b = theano.shared(value=b_values, name='b')

        self.W = W
        self.b = b

        # Compute linear "pre-activation" for this layer
        if use_bias:
            self.linear_output = T.dot(input, self.W) + self.b
        else:
            self.linear_output = T.dot(input, self.W)

        # Apply desired transform to compute "activation" for this layer
        if activation is None:
            self.output = self.linear_output
        else:
            self.output = activation(self.linear_output)

        # Compute expected sum of squared activations, to regularize
        self.act_sq_sum = T.sum(self.output**2.0) / self.output.size

        # Conveniently package layer parameters
        if use_bias:
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]


def _dropout_from_layer(rng, layer, p):
    """p is the probablity of dropping a unit
    """
    srng = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output

class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out,
                 activation, use_bias, W=None, b=None):
        super(DropoutHiddenLayer, self).__init__( \
                rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b, \
                activation=activation, use_bias=use_bias)
        undropped_output = self.output
        self.output = _dropout_from_layer(rng, undropped_output, p=0.5)


class DEV_MLP(object):
    """A multipurpose layer-based feedforward net.

    This class is capable of standard backprop training, training with
    dropout, and training with Dropout Ensemble Variance regularization.
    """
    def __init__(self,
            rng,
            input,
            params):

        # Setup simple activation function for this net
        rectified_linear_activation = lambda x: T.maximum(0.0, x)
        # Grab some of the parameters for this net
        layer_sizes = params['layer_sizes']
        lam_l2a = params['lam_l2a']
        dev_clones = params['dev_clones']
        use_bias = params['use_bias']
        dev_types = params['dev_types']
        dev_lams_as_numbers = [a for a in params['dev_lams']]
        dev_lams = np.asarray(params['dev_lams'], dtype=theano.config.floatX)
        dev_lams = theano.shared(value=dev_lams, name='dev_lams')
        # Make a dict to tell which parameters are norm-boundable
        self.clip_params = {}
        # Set up all the hidden layers
        weight_matrix_sizes = zip(layer_sizes, layer_sizes[1:])
        self.layers = []
        self.drop_nets = [[] for i in range(dev_clones)]
        # Initialize "next inputs", to be piped into new layers
        next_raw_input = input
        next_drop_inputs = [_dropout_from_layer(rng, input, p=0.2) \
                for i in range(dev_clones)]
        # Iteratively append layers to the RAW net and each of some number
        # of droppy DEV clones.
        first_layer = True
        for n_in, n_out in weight_matrix_sizes:
            # Add a new layer to the RAW (i.e. undropped) net
            self.layers.append(HiddenLayer(rng=rng,
                    input=next_raw_input,
                    activation=rectified_linear_activation,
                    n_in=n_in, n_out=n_out,
                    use_bias=use_bias))
            next_raw_input = self.layers[-1].output
            self.clip_params[self.layers[-1].W] = 1
            self.clip_params[self.layers[-1].b] = 0
            # Add a new dropout layer to each DEV clone, using the previous
            # layer in the corresponding DEV clone as input. The new DEV clone
            # layers all share parameters with the new RAW layer.
            W_drop = ((1.0/0.8) if first_layer else (1.0/0.5)) * self.layers[-1].W
            b_drop = self.layers[-1].b
            for i in range(dev_clones):
                self.drop_nets[i].append(DropoutHiddenLayer(rng=rng, \
                        input=next_drop_inputs[i], \
                        activation=rectified_linear_activation, \
                        W=W_drop, \
                        b=b_drop, \
                        n_in=n_in, n_out=n_out, use_bias=use_bias))
                next_drop_inputs[i] = self.drop_nets[i][-1].output
            first_layer = False
        # Grab all the parameters together.
        self.params = [ param for layer in self.layers for param in layer.params ]
        self.layer_count = len(self.layers)

        # Use the negative log likelihood of the logistic regression layer of
        # the first DEV clone as dropout optimization objective.
        self.sde_out_func = MCL2Hinge(self.drop_nets[0][-1])
        self.sde_class_loss = self.sde_out_func.loss_func
        self.sde_reg_loss = lam_l2a * T.sum([lay.act_sq_sum for lay in self.drop_nets[0]])
        self.sde_errors = self.sde_out_func.errors

        # Use the negative log likelihood of the logistic regression layer of
        # the RAW net as the standard optimization objective.
        self.raw_out_func = MCL2Hinge(self.layers[-1])
        self.raw_class_loss = self.raw_out_func.loss_func
        self.raw_reg_loss = lam_l2a * T.sum([lay.act_sq_sum for lay in self.layers])
        self.raw_errors = self.raw_out_func.errors

        # Compute DEV loss based on the classification performance of the RAW
        # net and the "Dropout Ensemble Variance"
        self.dev_out_func = self.raw_out_func
        self.dev_class_loss = self.raw_out_func.loss_func #lambda y: \
                #(self.raw_class_loss(y) + self.sde_class_loss(y)) / 2.0
        self.dev_reg_loss = (self.raw_reg_loss + self.sde_reg_loss) / 2.0
        self.dev_dev_loss = self.dev_loss(dev_types, dev_lams)
        self.dev_errors = self.raw_out_func.errors
        if (sum(dev_lams_as_numbers) < 1e-5):
            # When all dev_lams are 0 (ish), just switch to standard SGD
            self.dev_reg_loss = self.raw_reg_loss
            self.dev_dev_loss = T.sum(dev_lams)

    def dev_loss(self, dev_types=[], dev_lams=[]):
        var_fun = lambda x1, x2: T.sum((x1 - x2)**2.0) / x1.shape[0]
        tanh_fun = lambda x1, x2: var_fun(T.tanh(x1), T.tanh(x2))
        norm_fun = lambda x1, x2: var_fun( \
                (x1 / T.sqrt(T.sum(x1**2.0,axis=1,keepdims=1) + 1e-6)), \
                (x2 / T.sqrt(T.sum(x2**2.0,axis=1,keepdims=1) + 1e-6)))
        sigm_fun = lambda x1, x2: var_fun(T.nnet.sigmoid(x1), T.nnet.sigmoid(x2))
        cent_fun = lambda xt, xo: T.sum(T.nnet.binary_crossentropy( \
                T.nnet.sigmoid(xo), T.nnet.sigmoid(xt))) / xt.shape[0]
        L = 0.0
        for i in xrange(self.layer_count):
            if (i < (self.layer_count - 1)):
                x1 = self.layers[i].output
                x2 = self.drop_nets[0][i].output
            else:
                x1 = self.layers[i].linear_output
                x2 = self.drop_nets[0][i].linear_output
            if (dev_types[i] == 1):
                L = L + (dev_lams[i] * norm_fun(x1, x2))
            elif (dev_types[i] == 2):
                L = L + (dev_lams[i] * tanh_fun(x1, x2))
            elif (dev_types[i] == 3):
                L = L + (dev_lams[i] * sigm_fun(x1, x2))
            elif (dev_types[i] == 4):
                L = L + (dev_lams[i] * cent_fun(x1, x2))
            else:
                L = L + (dev_lams[i] * var_fun(x1, x2))
        return L


def test_mlp(
        mlp_params,
        sgd_params,
        datasets):
    """
    Datasets should be a three-tuple in which each item is a (matrix, vector)
    pair of (inputs, labels) for (training, validation, testing).
    """
    initial_learning_rate = sgd_params['start_rate']
    learning_rate_decay = sgd_params['decay_rate']
    n_epochs = sgd_params['epochs']
    batch_size = sgd_params['batch_size']
    mlp_type = sgd_params['mlp_type']
    wt_norm_bound = sgd_params['wt_norm_bound']
    result_tag = sgd_params['result_tag']
    txt_file_name = "results_{0}.txt".format(result_tag)
    img_file_name = "weights_{0}.png".format(result_tag)

    Xtr, Ytr = datasets[0]
    Xva, Yva = datasets[1]
    Xte, Yte = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = Xtr.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = Xva.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = Xte.get_value(borrow=True).shape[0] / batch_size

    print "train batches: {0:d}, valid batches: {1:d}, test_batches: {2:d}".format( \
            n_train_batches, n_valid_batches, n_test_batches)

    ######################
    # BUILD ACTUAL MODEL #
    ######################

    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar() # index to a [mini]batch
    epoch = T.scalar()  # epoch counter
    x = T.matrix('x')   # data is presented as rasterized images
    y = T.ivector('y')  # labels are presented as integer categories
    learning_rate = theano.shared(np.asarray(initial_learning_rate,
        dtype=theano.config.floatX))

    rng = np.random.RandomState(12345)

    # construct the MLP class
    NET = DEV_MLP(rng=rng, input=x, params=mlp_params)

    # Build the expressions for the cost functions. If training without SDE or
    # DEV regularization, the DEV loss/cost will be used, but the weights for
    # DEV regularization on each layer will be set to 0 before training.
    sde_cost = NET.sde_class_loss(y) + NET.sde_reg_loss
    dev_cost = 0.5 * (NET.dev_class_loss(y) + NET.dev_dev_loss) + NET.dev_reg_loss
    NET_metrics = [NET.raw_errors(y), NET.raw_class_loss(y), NET.dev_dev_loss, NET.raw_reg_loss]

    ############################################################################
    # Compile testing and validation models. These models are evaluated on     #
    # batches of the same size as used in training. Trying to jam a large      #
    # validation or test set through the net may take too much memory.         #
    ############################################################################
    test_model = theano.function(inputs=[index],
            outputs=NET_metrics,
            givens={
                x: Xte[index * batch_size:(index + 1) * batch_size],
                y: T.cast(Yte[index * batch_size:(index + 1) * batch_size], 'int32')})
    #theano.printing.pydotprint(test_model, outfile="test_file.png",
    #        var_with_name_simple=True)
    validate_model = theano.function(inputs=[index],
            outputs=NET_metrics,
            givens={
                x: Xva[index * batch_size:(index + 1) * batch_size],
                y: T.cast(Yva[index * batch_size:(index + 1) * batch_size], 'int32')})
    #theano.printing.pydotprint(validate_model, outfile="validate_file.png",
    #        var_with_name_simple=True)

    ############################################################################
    # Prepare momentum and gradient variables, and construct the updates that  #
    # Theano will perform on the network parameters.                           #
    ############################################################################
    sde_grads = []
    dev_grads = []
    for param in NET.params:
        sde_grads.append(T.grad(sde_cost, param))
        dev_grads.append(T.grad(dev_cost, param))

    sde_moms = []
    dev_moms = []
    for param in NET.params:
        sde_moms.append(theano.shared(np.zeros( \
                param.get_value(borrow=True).shape, dtype=theano.config.floatX)))
        dev_moms.append(theano.shared(np.zeros( \
                param.get_value(borrow=True).shape, dtype=theano.config.floatX)))

    # Compute momentum for the current epoch
    mom = ifelse(epoch < 500,
            0.5*(1. - epoch/500.) + 0.99*(epoch/500.),
            0.99)

    # Use a "smoothed" learning rate, to ease into optimization
    gentle_rate = ifelse(epoch < 5,
            (epoch / 5.0) * learning_rate,
            learning_rate)

    # Update the step direction using a momentus update
    sde_updates = OrderedDict()
    dev_updates = OrderedDict()
    for i in range(len(NET.params)):
        sde_updates[sde_moms[i]] = mom * sde_moms[i] + (1. - mom) * sde_grads[i]
        dev_updates[dev_moms[i]] = mom * dev_moms[i] + (1. - mom) * dev_grads[i]

    # ... and take a step along that direction
    for i in range(len(NET.params)):
        param = NET.params[i]
        sde_param = param - (gentle_rate * sde_updates[sde_moms[i]])
        dev_param = param - (gentle_rate * dev_updates[dev_moms[i]])
        # Clip the updated param to bound its norm (where applicable)
        if (NET.clip_params.has_key(param) and \
                (NET.clip_params[param] == 1)):
            sde_norms = T.sum(sde_param**2, axis=1).reshape((sde_param.shape[0],1))
            sde_scale = T.clip(T.sqrt(wt_norm_bound / sde_norms), 0., 1.)
            sde_updates[param] = sde_param * sde_scale
            dev_norms = T.sum(dev_param**2, axis=1).reshape((dev_param.shape[0],1))
            dev_scale = T.clip(T.sqrt(wt_norm_bound / dev_norms), 0., 1.)
            dev_updates[param] = dev_param * dev_scale
        else:
            sde_updates[param] = sde_param
            dev_updates[param] = dev_param

    # Compile theano functions for training.  These return the training cost
    # update the model parameters.
    train_sde = theano.function(inputs=[epoch, index], outputs=NET_metrics,
            updates=sde_updates,
            givens={
                x: Xtr[index * batch_size:(index + 1) * batch_size],
                y: T.cast(Ytr[index * batch_size:(index + 1) * batch_size], 'int32')})

    train_dev = theano.function(inputs=[epoch, index], outputs=NET_metrics,
            updates=dev_updates,
            givens={
                x: Xtr[index * batch_size:(index + 1) * batch_size],
                y: T.cast(Ytr[index * batch_size:(index + 1) * batch_size], 'int32')})

    # Theano function to decay the learning rate, this is separate from the
    # training function because we only want to do this once each epoch instead
    # of after each minibatch.
    set_learning_rate = theano.function(inputs=[], outputs=learning_rate,
            updates={learning_rate: learning_rate * learning_rate_decay})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    best_params = None
    best_validation_errors = np.inf
    best_iter = 0
    test_score = 0.
    epoch_counter = 0
    start_time = time.clock()

    results_file = open(txt_file_name, 'wb')
    results_file.write("mlp_type: {0}\n".format(mlp_type))
    results_file.write("lam_l2a: {0:.4f}\n".format(mlp_params['lam_l2a']))
    results_file.write("dev_types: {0}\n".format(str(mlp_params['dev_types'])))
    results_file.write("dev_lams: {0}\n".format(str(mlp_params['dev_lams'])))
    results_file.flush()

    e_time = time.clock()
    epoch_metrics = train_sde(1, 1)
    while epoch_counter < n_epochs:
        # Train this epoch
        epoch_counter = epoch_counter + 1
        for minibatch_index in xrange(n_train_batches):
            if ((epoch_counter <= 0) or (mlp_type == 'sde')):
                batch_metrics = train_sde(epoch_counter, minibatch_index)
            else:
                batch_metrics = train_dev(epoch_counter, minibatch_index)
            epoch_metrics = [(em + bm) for (em, bm) in zip(epoch_metrics, batch_metrics)]

        # Compute classification errors on validation set
        validation_metrics = [validate_model(i) for i in xrange(n_valid_batches)]
        this_validation_errors = np.sum([m[0] for m in validation_metrics])

        # Report and save progress.
        epoch_metrics = [(float(v) / float(n_train_batches)) for v in epoch_metrics]
        tag = (" **" if (this_validation_errors < best_validation_errors) else " ")
        print "epoch {0}: t_err={1:.2f}, t_loss={2:.4f}, t_dev={3:.4f}, t_reg={4:.4f}, v_err={5:d}{6}".format( \
                epoch_counter, epoch_metrics[0], epoch_metrics[1], epoch_metrics[2], epoch_metrics[3], \
                int(this_validation_errors), tag)
        print "--time: {0:.4f}".format((time.clock() - e_time))
        epoch_metrics = [0.0 for v in epoch_metrics]
        e_time = time.clock()

        best_validation_errors = min(best_validation_errors, this_validation_errors)
        results_file.write("{0}\n".format(this_validation_errors))
        results_file.flush()

        new_learning_rate = set_learning_rate()

        # Save first layer weights to an image locally
        utils.visualize(NET, 0, img_file_name)

    end_time = time.clock()

    # Compute loss on test set
    test_metrics = [test_model(i) for i in xrange(n_test_batches)]
    test_score = np.sum([m[0] for m in test_metrics])
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_errors * 100., best_iter, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    import sys

    # Set SGD-related parameters (and bound on net weights)
    sgd_params = {}
    sgd_params['start_rate'] = 0.25
    sgd_params['decay_rate'] = 0.998
    sgd_params['wt_norm_bound'] = 3.75
    sgd_params['epochs'] = 1000
    sgd_params['batch_size'] = 100
    # Set parameters for the network to be trained
    mlp_params = {}
    mlp_params['layer_sizes'] = [28*28, 800, 800, 10]
    mlp_params['lam_l2a'] = 1e-3
    mlp_params['dev_clones'] = 1
    mlp_params['dev_types'] = [1, 1, 2]
    mlp_params['dev_lams'] = [0.1, 0.1, 2.0]
    mlp_params['use_bias'] = 1
    # Pick a some data to train with
    datasets = load_mnist('data/mnist_batches.npz')
    #datasets = load_umontreal_data('data/mnist.pkl')

    # Set the type of network to train, based on user input
    if (len(sys.argv) != 3):
        print "Usage: {0} [raw|sde|dev] [result_tag]".format(sys.argv[0])
        exit(1)
    elif sys.argv[1] == "raw":
        sgd_params['mlp_type'] = 'raw'
        sgd_params['result_tag'] = sys.argv[2]
        mlp_params['dev_lams'] = [0.0 for l in mlp_params['dev_lams']]
    elif sys.argv[1] == "sde":
        sgd_params['mlp_type'] = 'sde'
        sgd_params['result_tag'] = sys.argv[2]
    elif sys.argv[1] == "dev":
        sgd_params['mlp_type'] = 'dev'
        sgd_params['result_tag'] = sys.argv[2]
    else:
        print "I don't know how to '{0}'".format(sys.argv[1])
        exit(1)

    test_mlp(mlp_params=mlp_params, \
            sgd_params=sgd_params, \
            datasets=datasets)









##############
# EYE BUFFER #
##############

