##########################################################
# Semi-supervised DEV-regularized multilayer perceptron. #
##########################################################

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
from output_losses import MCL2HingeSS
from load_data import load_udm_ss


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
            b_values = np.zeros((n_out,), dtype=theano.config.floatX) + 0.1
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
        self.act_l2_sum = T.sum(self.output**2.0) / self.output.size

        # Conveniently package layer parameters
        if use_bias:
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]


def _dropout_from_layer(rng, layer, p):
    """p is the probablity of dropping a unit
    """
    srng = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(100000))
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


class SS_DEV_MLP(object):
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
        ss_ratio = params['ss_ratio']
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
        self.sde_out_func = MCL2HingeSS(self.drop_nets[0][-1])
        self.sde_class_loss = self.sde_out_func.loss_func
        self.sde_reg_loss = lam_l2a * T.sum([lay.act_l2_sum for lay in self.drop_nets[0]])
        self.sde_errors = self.sde_out_func.errors

        # Use the negative log likelihood of the logistic regression layer of
        # the RAW net as the standard optimization objective.
        self.raw_out_func = MCL2HingeSS(self.layers[-1])
        self.raw_class_loss = self.raw_out_func.loss_func
        self.raw_reg_loss = lam_l2a * T.sum([lay.act_l2_sum for lay in self.layers])
        self.raw_errors = self.raw_out_func.errors

        # Compute DEV loss based on the classification performance of the RAW
        # net and the "Dropout Ensemble Variance"
        self.dev_out_func = self.raw_out_func
        self.dev_class_loss = lambda y: \
                (self.raw_class_loss(y) + self.sde_class_loss(y)) / 2.0
        self.dev_reg_loss = (self.raw_reg_loss + self.sde_reg_loss) / 2.0
        self.dev_dev_loss = lambda y: \
                self.dev_loss(dev_types, dev_lams, ss_ratio, y)
        self.dev_errors = self.raw_out_func.errors
        if (sum(dev_lams_as_numbers) < 1e-5):
            # When all dev_lams are 0 (ish), just switch to standard SGD
            self.dev_class_loss = self.raw_out_func.loss_func
            self.dev_reg_loss = self.raw_reg_loss
            self.dev_dev_loss = lambda y: T.sum(dev_lams)

    def dev_loss(self, dev_types, dev_lams, ss_ratio, y):
        su_mask = ss_ratio * T.neq(y, 0).reshape((y.shape[0], 1))
        un_mask = T.eq(y, 0).reshape((y.shape[0], 1))
        ss_mask = su_mask + un_mask
        var_fun = lambda x1, x2: T.sum(((x1 - x2) * ss_mask)**2.0) / T.sum(ss_mask)
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
        rng,
        mlp_params,
        sgd_params,
        datasets):
    """
    Datasets should be a four-tuple, in which the first item is a matrix/vector
    pair of inputs/labels for training, the second item is a matrix of
    unlabeled inputs for training, the third item is a matrix/vector pair of
    inputs/labels for validation, and the fourth is a matrix/vector pair of
    inputs/labels for testing.
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

    # Get supervised and unsupervised portions of training data, and create
    # arrays of start/end indices for easy minibatch slicing.
    (Xtr_su, Ytr_su) = (datasets[0][0], T.cast(datasets[0][1], 'int32'))
    (Xtr_un, Ytr_un) = (datasets[1][0], T.cast(datasets[1][1], 'int32'))
    su_samples = Xtr_su.get_value(borrow=True).shape[0]
    un_samples = Xtr_un.get_value(borrow=True).shape[0]
    tr_batches = 250
    mlp_params['ss_ratio'] = float(su_samples) / (su_samples + un_samples)
    su_bsize = batch_size / 2
    un_bsize = batch_size - su_bsize
    su_batches = int(np.ceil(float(su_samples) / su_bsize))
    un_batches = int(np.ceil(float(un_samples) / un_bsize))
    su_bidx = [[i*su_bsize, min(su_samples, (i+1)*su_bsize)] for i in range(su_batches)]
    un_bidx = [[i*un_bsize, min(un_samples, (i+1)*un_bsize)] for i in range(un_batches)]
    su_bidx = theano.shared(value=np.asarray(su_bidx, dtype=theano.config.floatX))
    un_bidx = theano.shared(value=np.asarray(un_bidx, dtype=theano.config.floatX))
    su_bidx = T.cast(su_bidx, 'int32')
    un_bidx = T.cast(un_bidx, 'int32')
    # Get the validation and testing sets and create arrays of start/end
    # indices for easy minibatch slicing
    Xva, Yva = (datasets[2][0], T.cast(datasets[2][1], 'int32'))
    Xte, Yte = (datasets[3][0], T.cast(datasets[3][1], 'int32'))
    va_samples = Xva.get_value(borrow=True).shape[0]
    te_samples = Xte.get_value(borrow=True).shape[0]
    va_batches = int(np.ceil(va_samples / 100.0))
    te_batches = int(np.ceil(te_samples / 100.0))
    va_bidx = [[i*100, min(va_samples, (i+1)*100)] for i in range(va_batches)]
    te_bidx = [[i*100, min(te_samples, (i+1)*100)] for i in range(te_batches)]
    va_bidx = theano.shared(value=np.asarray(va_bidx, dtype=theano.config.floatX))
    te_bidx = theano.shared(value=np.asarray(te_bidx, dtype=theano.config.floatX))
    va_bidx = T.cast(va_bidx, 'int32')
    te_bidx = T.cast(te_bidx, 'int32')

    print "Dataset info:"
    print "  supervised samples: {0:d}, unsupervised samples: {1:d}".format( \
            su_samples, un_samples)
    print "  samples/minibatch: {0:d}, minibatches/epoch: {1:d}".format( \
            (su_bsize + un_bsize), tr_batches)
    print "  validation samples: {0:d}, testing samples: {1:d}".format( \
            va_samples, te_samples)

    ######################
    # BUILD ACTUAL MODEL #
    ######################

    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar() # index to a [mini]batch
    epoch = T.scalar()  # epoch counter
    su_idx = T.lscalar() # symbolic batch index into supervised samples
    un_idx = T.lscalar() # symbolic batch index into unsupervised samples
    x = T.matrix('x')   # some observations have labels
    y = T.ivector('y')  # the labels are presented as integer categories
    learning_rate = theano.shared(np.asarray(initial_learning_rate,
        dtype=theano.config.floatX))

    # construct the MLP class
    NET = SS_DEV_MLP(rng=rng, input=x, params=mlp_params)

    # Build the expressions for the cost functions. If training without SDE or
    # DEV regularization, the DEV loss/cost will be used, but the weights for
    # DEV regularization on each layer will be set to 0 before training.
    sde_cost = NET.sde_class_loss(y) + NET.sde_reg_loss
    dev_cost = NET.dev_class_loss(y) + NET.dev_dev_loss(y) + NET.dev_reg_loss
    NET_metrics = [NET.raw_errors(y), NET.raw_class_loss(y), \
                   NET.dev_dev_loss(y), NET.raw_reg_loss]

    ############################################################################
    # Compile testing and validation models. These models are evaluated on     #
    # batches of the same size as used in training. Trying to jam a large      #
    # validation or test set through the net may take too much memory.         #
    ############################################################################
    test_model = theano.function(inputs=[index],
            outputs=NET_metrics,
            givens={
                x: Xte[te_bidx[index,0]:te_bidx[index,1],:],
                y: Yte[te_bidx[index,0]:te_bidx[index,1]]})

    validate_model = theano.function(inputs=[index],
            outputs=NET_metrics,
            givens={
                x: Xva[va_bidx[index,0]:va_bidx[index,1],:],
                y: Yva[va_bidx[index,0]:va_bidx[index,1]]})

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
    # and update the model parameters.
    train_sde = theano.function(inputs=[epoch, su_idx, un_idx], outputs=NET_metrics,
            updates=sde_updates,
            givens={
                x: T.concatenate([Xtr_su[su_bidx[su_idx,0]:su_bidx[su_idx,1],:], \
                        Xtr_un[un_bidx[un_idx,0]:un_bidx[un_idx,1],:]]),
                y: T.concatenate([Ytr_su[su_bidx[su_idx,0]:su_bidx[su_idx,1]], \
                        Ytr_un[un_bidx[un_idx,0]:un_bidx[un_idx,1]]])})

    train_dev = theano.function(inputs=[epoch, su_idx, un_idx], outputs=NET_metrics,
            updates=dev_updates,
            givens={
                x: T.concatenate([Xtr_su[su_bidx[su_idx,0]:su_bidx[su_idx,1],:], \
                        Xtr_un[un_bidx[un_idx,0]:un_bidx[un_idx,1],:]]),
                y: T.concatenate([Ytr_su[su_bidx[su_idx,0]:su_bidx[su_idx,1]], \
                        Ytr_un[un_bidx[un_idx,0]:un_bidx[un_idx,1]]])})

    # Theano function to decay the learning rate, this is separate from the
    # training function because we only want to do this once each epoch instead
    # of after each minibatch.
    set_learning_rate = theano.function(inputs=[], outputs=learning_rate,
            updates={learning_rate: learning_rate * learning_rate_decay})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    validation_errors = 1e6
    test_errors = 1e6
    min_validation_errors = 1e6
    min_test_errors = 1e6
    epoch_counter = 0
    start_time = time.clock()

    results_file = open(txt_file_name, 'wb')
    results_file.write("mlp_type: {0}\n".format(mlp_type))
    results_file.write("lam_l2a: {0:.4f}\n".format(mlp_params['lam_l2a']))
    results_file.write("dev_types: {0}\n".format(str(mlp_params['dev_types'])))
    results_file.write("dev_lams: {0}\n".format(str(mlp_params['dev_lams'])))
    results_file.flush()

    e_time = time.clock()
    su_index = 0
    un_index = 0
    # Get array of epoch metrics (on a single minibatch)
    epoch_metrics = train_sde(1, su_index, un_index)
    # Compute metrics on validation set
    validation_metrics = [validate_model(i) for i in xrange(va_batches)]
    validation_errors = np.sum([m[0] for m in validation_metrics])
    # Compute metrics on testing set
    test_metrics = [test_model(i) for i in xrange(te_batches)]
    test_errors = np.sum([m[0] for m in test_metrics])
    while epoch_counter < n_epochs:
        ######################################################
        # Process some number of minibatches for this epoch. #
        ######################################################
        epoch_counter = epoch_counter + 1
        epoch_metrics = [0.0 for v in epoch_metrics]
        for minibatch_index in xrange(tr_batches):
            # Compute update for some joint supervised/unsupervised minibatch
            if ((epoch_counter <= 0) or (mlp_type == 'sde')):
                batch_metrics = train_sde(epoch_counter, su_index, un_index)
            else:
                batch_metrics = train_dev(epoch_counter, su_index, un_index)
            epoch_metrics = [(em + bm) for (em, bm) in zip(epoch_metrics, batch_metrics)]
            su_index = (su_index + 1) if ((su_index + 1) < su_batches) else 0
            un_index = (un_index + 1) if ((un_index + 1) < un_batches) else 0
        # Update the learning rate
        new_learning_rate = set_learning_rate()

        ######################################################
        # Validation, testing, and general diagnostic stuff. #
        ######################################################
        # Compute metrics on validation set
        validation_metrics = [validate_model(i) for i in xrange(va_batches)]
        validation_errors = np.sum([m[0] for m in validation_metrics])

        # Compute test error if new best validation error was found
        tag = " "
        #if ((validation_errors < min_validation_errors) or ((epoch_counter % 10) == 0)):
        if ('I' == 'I'):
            # Compute metrics on testing set
            test_metrics = [test_model(i) for i in xrange(te_batches)]
            test_errors = np.sum([m[0] for m in test_metrics])
            tag = ", test={0:d}".format(test_errors)
            if (validation_errors < min_validation_errors):
                min_validation_errors = validation_errors
                min_test_errors = test_errors
        results_file.write("{0:d} {1:d}\n".format(validation_errors, test_errors))
        results_file.flush()

        # Report and save progress.
        epoch_metrics[0] = float(epoch_metrics[0]) / (tr_batches * su_bsize)
        epoch_metrics[1:] = [(float(v) / tr_batches) for v in epoch_metrics[1:]]
        print "epoch {0:d}: t_err={1:.2f}, t_loss={2:.4f}, t_dev={3:.4f}, t_reg={4:.4f}, valid={5:d}{6}".format( \
                epoch_counter, epoch_metrics[0], epoch_metrics[1], epoch_metrics[2], epoch_metrics[3], \
                int(validation_errors), tag)
        print "--time: {0:.4f}".format((time.clock() - e_time))
        e_time = time.clock()
        # Save first layer weights to an image locally
        utils.visualize(NET, 0, img_file_name)

    print("Optimization complete. Best validation score of {0:.4f}, with test score {1:.4f}".format( \
          (min_validation_errors / 100.0), (min_test_errors / 100.0)))
    results_file.write("{0:d} {1:d}\n".format(min_validation_errors, min_test_errors))
    results_file.flush()

if __name__ == '__main__':
    import sys

    # Initialize a random number generator for this test
    rng = np.random.RandomState(13579)

    # Set SGD-related parameters (and bound on net weights)
    sgd_params = {}
    sgd_params['start_rate'] = 0.1
    sgd_params['decay_rate'] = 0.998
    sgd_params['wt_norm_bound'] = 3.75
    sgd_params['epochs'] = 1000
    sgd_params['batch_size'] = 128
    # Set parameters for the network to be trained
    mlp_params = {}
    mlp_params['layer_sizes'] = [28*28, 500, 500, 11]
    mlp_params['lam_l2a'] = 1e-3
    mlp_params['dev_clones'] = 1
    mlp_params['dev_types'] = [1, 1, 2]
    mlp_params['dev_lams'] = [0.1, 0.1, 2.0]
    mlp_params['use_bias'] = 1

    # Load some data to train/validate/test with
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm_ss(dataset, 1000, rng)

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

    # Run the test
    test_mlp(rng=rng, \
            mlp_params=mlp_params, \
            sgd_params=sgd_params, \
            datasets=datasets)









##############
# EYE BUFFER #
##############

