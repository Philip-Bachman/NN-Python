import numpy as np
import os
import sys
import time
from collections import OrderedDict

import theano
import theano.tensor as T
from theano.ifelse import ifelse
import theano.printing

import utils as utils

def train_mlp(
        NET,
        mlp_params,
        sgd_params,
        datasets):
    """
    Train NET using purely supervised data.

    This trains a standard supervised MLP, using any of: no droppish stuff,
    standard dropout, or dropout ensemble variance regularization. Datasets
    should be a three-tuple, where each item is an (X, Y) pair of observation
    matrix X and class vector Y. The (X, Y) pairs will be used for training,
    validation, and testing respectively.
    """
    initial_learning_rate = sgd_params['start_rate']
    learning_rate_decay = sgd_params['decay_rate']
    n_epochs = sgd_params['epochs']
    batch_size = sgd_params['batch_size']
    mlp_type = sgd_params['mlp_type']
    wt_norm_bound = sgd_params['wt_norm_bound']
    result_tag = sgd_params['result_tag']
    txt_file_name = "results_mlp_{0}.txt".format(result_tag)
    img_file_name = "weights_mlp_{0}.png".format(result_tag)

    ###########################################################################
    # We will use minibatches for training, as well as for computing stats    #
    # over the validation and testing sets. For Theano reasons, it will be    #
    # easiest if we set up arrays storing the start/end index of each batch   #
    # w.r.t. the relevant observation/class matrices/vectors.                 #
    ###########################################################################
    # Get the training observations and classes
    Xtr, Ytr = (datasets[0][0], T.cast(datasets[0][1], 'int32'))
    tr_samples = Xtr.get_value(borrow=True).shape[0]
    tr_batches = int(np.ceil(tr_samples / float(batch_size)))
    tr_bidx = [[i*batch_size, min(tr_samples, (i+1)*batch_size)] \
            for i in range(tr_batches)]
    tr_bidx = theano.shared(value=np.asarray(tr_bidx, dtype=theano.config.floatX))
    tr_bidx = T.cast(tr_bidx, 'int32')
    # Get the validation and testing observations and classes
    Xva, Yva = (datasets[1][0], T.cast(datasets[1][1], 'int32'))
    Xte, Yte = (datasets[2][0], T.cast(datasets[2][1], 'int32'))
    va_samples = Xva.get_value(borrow=True).shape[0]
    te_samples = Xte.get_value(borrow=True).shape[0]
    va_batches = int(np.ceil(va_samples / 100.))
    te_batches = int(np.ceil(te_samples / 100.))
    va_bidx = [[i*100, min(va_samples, (i+1)*100)] for i in range(va_batches)]
    te_bidx = [[i*100, min(te_samples, (i+1)*100)] for i in range(te_batches)]
    va_bidx = theano.shared(value=np.asarray(va_bidx, dtype=theano.config.floatX))
    te_bidx = theano.shared(value=np.asarray(te_bidx, dtype=theano.config.floatX))
    va_bidx = T.cast(va_bidx, 'int32')
    te_bidx = T.cast(te_bidx, 'int32')

    # Print some useful information about the dataset
    print "dataset info:"
    print "  training samples: {0:d}".format(tr_samples)
    print "  samples/minibatch: {0:d}, minibatches/epoch: {1:d}".format( \
            batch_size, tr_batches)
    print "  validation samples: {0:d}, testing samples: {1:d}".format( \
            va_samples, te_samples)

    ######################
    # build actual model #
    ######################

    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    epoch = T.scalar()   # epoch counter
    su_idx = T.lscalar() # symbolic batch index into supervised samples
    un_idx = T.lscalar() # symbolic batch index into unsupervised samples
    x = NET.input        # some observations have labels
    y = T.ivector('y')   # the labels are presented as integer categories
    learning_rate = theano.shared(np.asarray(initial_learning_rate, \
        dtype=theano.config.floatX))

    # Build the expressions for the cost functions. if training without sde or
    # dev regularization, the dev loss/cost will be used, but the weights for
    # dev regularization on each layer will be set to 0 before training.
    sde_cost = NET.sde_cost(y)
    dev_cost = NET.dev_cost(y)
    NET_metrics = [NET.class_errors(y), NET.raw_class_loss(y), \
                   NET.dev_reg_loss(y), NET.raw_reg_loss]

    ############################################################################
    # Compile testing and validation models. these models are evaluated on     #
    # batches of the same size as used in training. trying to jam a large      #
    # validation or test set through the net may take too much memory.         #
    ############################################################################
    test_model = theano.function(inputs=[index], \
            outputs=NET_metrics, \
            givens={ \
                x: Xte[te_bidx[index,0]:te_bidx[index,1],:], \
                y: Yte[te_bidx[index,0]:te_bidx[index,1]]})

    validate_model = theano.function(inputs=[index], \
            outputs=NET_metrics, \
            givens={ \
                x: Xva[va_bidx[index,0]:va_bidx[index,1],:], \
                y: Yva[va_bidx[index,0]:va_bidx[index,1]]})

    ############################################################################
    # prepare momentum and gradient variables, and construct the updates that  #
    # theano will perform on the network parameters.                           #
    ############################################################################
    opt_params = NET.mlp_params
    if sgd_params.has_key('top_only'):
        if sgd_params['top_only']:
            opt_params = NET.class_params
    sde_grads = []
    dev_grads = []
    for param in opt_params:
        sde_grads.append(T.grad(sde_cost, param))
        dev_grads.append(T.grad(dev_cost, param))

    sde_moms = []
    dev_moms = []
    for param in opt_params:
        sde_moms.append(theano.shared(np.zeros( \
                param.get_value(borrow=True).shape, dtype=theano.config.floatX)))
        dev_moms.append(theano.shared(np.zeros( \
                param.get_value(borrow=True).shape, dtype=theano.config.floatX)))

    # compute momentum for the current epoch
    mom = ifelse(epoch < 500,
            0.5*(1. - epoch/500.) + 0.99*(epoch/500.),
            0.99)

    # use a "smoothed" learning rate, to ease into optimization
    gentle_rate = ifelse(epoch < 5,
            (epoch / 5.) * learning_rate,
            learning_rate)

    # update the step direction using a momentus update
    sde_updates = OrderedDict()
    dev_updates = OrderedDict()
    for i in range(len(opt_params)):
        sde_updates[sde_moms[i]] = mom * sde_moms[i] + (1. - mom) * sde_grads[i]
        dev_updates[dev_moms[i]] = mom * dev_moms[i] + (1. - mom) * dev_grads[i]

    # ... and take a step along that direction
    for i in range(len(opt_params)):
        param = opt_params[i]
        sde_param = param - (gentle_rate * sde_updates[sde_moms[i]])
        dev_param = param - (gentle_rate * dev_updates[dev_moms[i]])
        # clip the updated param to bound its norm (where applicable)
        if (NET.clip_params.has_key(param) and \
                (NET.clip_params[param] == 1)):
            sde_norms = T.sum(sde_param**2, axis=1, keepdims=1)
            sde_scale = T.clip(T.sqrt(wt_norm_bound / sde_norms), 0., 1.)
            sde_updates[param] = sde_param * sde_scale
            dev_norms = T.sum(dev_param**2, axis=1, keepdims=1)
            dev_scale = T.clip(T.sqrt(wt_norm_bound / dev_norms), 0., 1.)
            dev_updates[param] = dev_param * dev_scale
        else:
            sde_updates[param] = sde_param
            dev_updates[param] = dev_param

    # compile theano functions for training.  these return the training cost
    # and update the model parameters.
    train_sde = theano.function(inputs=[epoch, index], outputs=NET_metrics, \
            updates=sde_updates, \
            givens={ \
                x: Xtr[tr_bidx[index,0]:tr_bidx[index,1],:], \
                y: Ytr[tr_bidx[index,0]:tr_bidx[index,1]]})

    train_dev = theano.function(inputs=[epoch, index], outputs=NET_metrics, \
            updates=dev_updates, \
            givens={ \
                x: Xtr[tr_bidx[index,0]:tr_bidx[index,1],:], \
                y: Ytr[tr_bidx[index,0]:tr_bidx[index,1]]})

    # theano function to decay the learning rate, this is separate from the
    # training function because we only want to do this once each epoch instead
    # of after each minibatch.
    set_learning_rate = theano.function(inputs=[], outputs=learning_rate, \
            updates={learning_rate: learning_rate * learning_rate_decay})

    ###############
    # train model #
    ###############
    print '... training'

    validation_error = 100.
    test_error = 100.
    min_validation_error = 100.
    min_test_error = 100.
    epoch_counter = 0
    start_time = time.clock()

    results_file = open(txt_file_name, 'wb')
    results_file.write("mlp_type: {0}\n".format(mlp_type))
    results_file.write("lam_l2a: {0:.4f}\n".format(mlp_params['lam_l2a']))
    results_file.write("dev_types: {0}\n".format(str(mlp_params['dev_types'])))
    results_file.write("dev_lams: {0}\n".format(str(mlp_params['dev_lams'])))
    results_file.flush()

    e_time = time.clock()
    # get array of epoch metrics (on a single minibatch)
    epoch_metrics = train_sde(1, 0)
    validation_metrics = [0. for v in epoch_metrics]
    test_metrics = [0. for v in epoch_metrics]
    # compute metrics on testing set
    while epoch_counter < n_epochs:
        ######################################################
        # process some number of minibatches for this epoch. #
        ######################################################
        NET.set_bias_noise(0.1)
        epoch_counter = epoch_counter + 1
        epoch_metrics = [0. for v in epoch_metrics]
        for b_idx in xrange(tr_batches):
            # compute update for some this minibatch
            if ((epoch_counter <= 0) or (mlp_type == 'sde')):
                batch_metrics = train_sde(epoch_counter, b_idx)
            else:
                batch_metrics = train_dev(epoch_counter, b_idx)
            epoch_metrics = [(em + bm) for (em, bm) in zip(epoch_metrics, batch_metrics)]
        # Compute 'averaged' values over the minibatches
        epoch_metrics[0] = 100 * (float(epoch_metrics[0]) / tr_samples)
        epoch_metrics[1:] = [(float(v) / tr_batches) for v in epoch_metrics[1:]]
        train_error = epoch_metrics[0]
        train_loss = epoch_metrics[1]
        # update the learning rate
        new_learning_rate = set_learning_rate()

        ######################################################
        # validation, testing, and general diagnostic stuff. #
        ######################################################
        NET.set_bias_noise(0.0)
        # compute metrics on validation set
        validation_metrics = [0. for v in epoch_metrics]
        for b_idx in xrange(va_batches):
            batch_metrics = validate_model(b_idx)
            validation_metrics = [(em + bm) for (em, bm) in zip(validation_metrics, batch_metrics)]
        # Compute 'averaged' values over the minibatches
        validation_error = 100 * (float(validation_metrics[0]) / va_samples)
        validation_metrics[1:] = [(float(v) / va_batches) for v in validation_metrics[1:]]
        validation_loss = validation_metrics[1]

        # compute test error if new best validation error was found
        tag = " "
        # compute metrics on testing set
        test_metrics = [0. for v in epoch_metrics]
        for b_idx in xrange(te_batches):
            batch_metrics = test_model(b_idx)
            test_metrics = [(em + bm) for (em, bm) in zip(test_metrics, batch_metrics)]
        # Compute 'averaged' values over the minibatches
        test_error = 100 * (float(test_metrics[0]) / te_samples)
        test_metrics[1:] = [(float(v) / te_batches) for v in test_metrics[1:]]
        test_loss = test_metrics[1]
        if (validation_error < min_validation_error):
            min_validation_error = validation_error
            min_test_error = test_error
            tag = ", test={0:.2f}".format(test_error)
        results_file.write("{0:.2f} {1:.2f} {2:.2f} {3:.4f} {4:.4f} {5:.4f}\n".format( \
                train_error, validation_error, test_error, train_loss, validation_loss, test_loss))
        results_file.flush()

        # report and save progress.
        print "epoch {0:d}: t_err={1:.2f}, t_loss={2:.4f}, t_dev={3:.4f}, t_reg={4:.4f}, valid={5:.2f}{6}".format( \
                epoch_counter, epoch_metrics[0], epoch_metrics[1], epoch_metrics[2], epoch_metrics[3], \
                validation_error, tag)
        print "--time: {0:.4f}".format((time.clock() - e_time))
        e_time = time.clock()
        # save first layer weights to an image locally
        utils.visualize(NET, 0, img_file_name)

    print("optimization complete. best validation error {0:.4f}, with test error {1:.4f}".format( \
          (min_validation_error), (min_test_error)))

def train_ss_mlp(
        NET,
        mlp_params,
        sgd_params,
        datasets):
    """
    Train NET using a mix of labeled an unlabeled data.

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
    txt_file_name = "results_mlp_{0}.txt".format(result_tag)
    img_file_name = "weights_mlp_{0}.png".format(result_tag)

    # Get supervised and unsupervised portions of training data, and create
    # arrays of start/end indices for easy minibatch slicing.
    (Xtr_su, Ytr_su) = (datasets[0][0], T.cast(datasets[0][1], 'int32'))
    (Xtr_un, Ytr_un) = (datasets[1][0], T.cast(datasets[1][1], 'int32'))
    su_samples = Xtr_su.get_value(borrow=True).shape[0]
    un_samples = Xtr_un.get_value(borrow=True).shape[0]
    tr_batches = 250
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
    # get the validation and testing sets and create arrays of start/end
    # indices for easy minibatch slicing
    Xva, Yva = (datasets[2][0], T.cast(datasets[2][1], 'int32'))
    Xte, Yte = (datasets[3][0], T.cast(datasets[3][1], 'int32'))
    va_samples = Xva.get_value(borrow=True).shape[0]
    te_samples = Xte.get_value(borrow=True).shape[0]
    va_batches = int(np.ceil(va_samples / 100.))
    te_batches = int(np.ceil(te_samples / 100.))
    va_bidx = [[i*100, min(va_samples, (i+1)*100)] for i in range(va_batches)]
    te_bidx = [[i*100, min(te_samples, (i+1)*100)] for i in range(te_batches)]
    va_bidx = theano.shared(value=np.asarray(va_bidx, dtype=theano.config.floatX))
    te_bidx = theano.shared(value=np.asarray(te_bidx, dtype=theano.config.floatX))
    va_bidx = T.cast(va_bidx, 'int32')
    te_bidx = T.cast(te_bidx, 'int32')

    # Print some useful information about the dataset
    print "dataset info:"
    print "  supervised samples: {0:d}, unsupervised samples: {1:d}".format( \
            su_samples, un_samples)
    print "  samples/minibatch: {0:d}, minibatches/epoch: {1:d}".format( \
            (su_bsize + un_bsize), tr_batches)
    print "  validation samples: {0:d}, testing samples: {1:d}".format( \
            va_samples, te_samples)

    ######################
    # build actual model #
    ######################

    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    epoch = T.scalar()   # epoch counter
    su_idx = T.lscalar() # symbolic batch index into supervised samples
    un_idx = T.lscalar() # symbolic batch index into unsupervised samples
    x = NET.input        # some observations have labels
    y = T.ivector('y')   # the labels are presented as integer categories
    learning_rate = theano.shared(np.asarray(initial_learning_rate, \
        dtype=theano.config.floatX))

    # build the expressions for the cost functions. if training without sde or
    # dev regularization, the dev loss/cost will be used, but the weights for
    # dev regularization on each layer will be set to 0 before training.
    #ent_loss = 0.005 * NET._ent_loss(NET.mlp_layers[-1].linear_output, y, ent_type=1)
    #g1_loss = 2.0 * NET.grad_losses[-1][0]
    sde_cost = NET.sde_cost(y) #+ ent_loss #+ g1_loss
    dev_cost = NET.dev_cost(y) #+ ent_loss #+ g1_loss
    NET_metrics = [NET.class_errors(y), NET.raw_class_loss(y), \
                   NET.dev_reg_loss(y), NET.raw_reg_loss]

    ############################################################################
    # compile testing and validation models. these models are evaluated on     #
    # batches of the same size as used in training. trying to jam a large      #
    # validation or test set through the net may take too much memory.         #
    ############################################################################
    test_model = theano.function(inputs=[index], \
            outputs=NET_metrics, \
            givens={ \
                x: Xte[te_bidx[index,0]:te_bidx[index,1],:], \
                y: Yte[te_bidx[index,0]:te_bidx[index,1]]})

    validate_model = theano.function(inputs=[index], \
            outputs=NET_metrics, \
            givens={ \
                x: Xva[va_bidx[index,0]:va_bidx[index,1],:], \
                y: Yva[va_bidx[index,0]:va_bidx[index,1]]})

    ############################################################################
    # prepare momentum and gradient variables, and construct the updates that  #
    # theano will perform on the network parameters.                           #
    ############################################################################
    opt_params = NET.mlp_params
    if sgd_params.has_key('top_only'):
        if sgd_params['top_only']:
            opt_params = NET.class_params
    sde_grads = []
    dev_grads = []
    for param in opt_params:
        sde_grads.append(T.grad(sde_cost, param))
        dev_grads.append(T.grad(dev_cost, param))

    sde_moms = []
    dev_moms = []
    for param in opt_params:
        sde_moms.append(theano.shared(np.zeros( \
                param.get_value(borrow=True).shape, dtype=theano.config.floatX)))
        dev_moms.append(theano.shared(np.zeros( \
                param.get_value(borrow=True).shape, dtype=theano.config.floatX)))

    # compute momentum for the current epoch
    mom = ifelse(epoch < 500,
            0.5*(1. - epoch/500.) + 0.99*(epoch/500.),
            0.99)

    # use a "smoothed" learning rate, to ease into optimization
    gentle_rate = ifelse(epoch < 5,
            ((epoch / 5.)**2.) * learning_rate,
            learning_rate)

    # update the step direction using a momentus update
    sde_updates = OrderedDict()
    dev_updates = OrderedDict()
    for i in range(len(opt_params)):
        sde_updates[sde_moms[i]] = mom * sde_moms[i] + (1. - mom) * sde_grads[i]
        dev_updates[dev_moms[i]] = mom * dev_moms[i] + (1. - mom) * dev_grads[i]

    # ... and take a step along that direction
    for i in range(len(opt_params)):
        param = opt_params[i]
        sde_param = param - (gentle_rate * sde_updates[sde_moms[i]])
        dev_param = param - (gentle_rate * dev_updates[dev_moms[i]])
        # clip the updated param to bound its norm (where applicable)
        if (NET.clip_params.has_key(param) and \
                (NET.clip_params[param] == 1)):
            sde_norms = T.sum(sde_param**2, axis=1, keepdims=1)
            sde_scale = T.clip(T.sqrt(wt_norm_bound / sde_norms), 0., 1.)
            sde_updates[param] = sde_param * sde_scale
            dev_norms = T.sum(dev_param**2, axis=1, keepdims=1)
            dev_scale = T.clip(T.sqrt(wt_norm_bound / dev_norms), 0., 1.)
            dev_updates[param] = dev_param * dev_scale
        else:
            sde_updates[param] = sde_param
            dev_updates[param] = dev_param

    # compile theano functions for training.  these return the training cost
    # and update the model parameters.
    train_sde = theano.function(inputs=[epoch, su_idx, un_idx], outputs=NET_metrics, \
            updates=sde_updates, \
            givens={ \
                x: T.concatenate([Xtr_su[su_bidx[su_idx,0]:su_bidx[su_idx,1],:], \
                        Xtr_un[un_bidx[un_idx,0]:un_bidx[un_idx,1],:]]), \
                y: T.concatenate([Ytr_su[su_bidx[su_idx,0]:su_bidx[su_idx,1]], \
                        Ytr_un[un_bidx[un_idx,0]:un_bidx[un_idx,1]]])})

    train_dev = theano.function(inputs=[epoch, su_idx, un_idx], outputs=NET_metrics, \
            updates=dev_updates, \
            givens={ \
                x: T.concatenate([Xtr_su[su_bidx[su_idx,0]:su_bidx[su_idx,1],:], \
                        Xtr_un[un_bidx[un_idx,0]:un_bidx[un_idx,1],:]]),
                y: T.concatenate([Ytr_su[su_bidx[su_idx,0]:su_bidx[su_idx,1]], \
                        Ytr_un[un_bidx[un_idx,0]:un_bidx[un_idx,1]]])})

    # theano function to decay the learning rate, this is separate from the
    # training function because we only want to do this once each epoch instead
    # of after each minibatch.
    set_learning_rate = theano.function(inputs=[], outputs=learning_rate, \
            updates={learning_rate: learning_rate * learning_rate_decay})

    ###############
    # train model #
    ###############
    print '... training'

    validation_error = 100.
    test_error = 100.
    min_validation_error = 100.
    min_test_error = 100.
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
    # get array of epoch metrics (on a single minibatch)
    epoch_metrics = train_sde(1, 0, 0)
    validation_metrics = [0. for v in epoch_metrics]
    test_metrics = [0. for v in epoch_metrics]
    # compute metrics on testing set
    while epoch_counter < n_epochs:
        ######################################################
        # process some number of minibatches for this epoch. #
        ######################################################
        # Set to training mode
        NET.set_bias_noise(0.1)
        epoch_counter = epoch_counter + 1
        epoch_metrics = [0. for v in epoch_metrics]
        for b_idx in xrange(tr_batches):
            # compute update for some this minibatch
            if ((epoch_counter <= 0) or (mlp_type == 'sde')):
                batch_metrics = train_sde(epoch_counter, su_index, un_index)
            else:
                batch_metrics = train_dev(epoch_counter, su_index, un_index)
            epoch_metrics = [(em + bm) for (em, bm) in zip(epoch_metrics, batch_metrics)]
            su_index = (su_index + 1) if ((su_index + 1) < su_batches) else 0
            un_index = (un_index + 1) if ((un_index + 1) < un_batches) else 0
        # Compute 'averaged' values over the minibatches
        epoch_metrics[0] = 100 * (float(epoch_metrics[0]) / (tr_batches * su_bsize))
        epoch_metrics[1:] = [(float(v) / tr_batches) for v in epoch_metrics[1:]]
        train_error = epoch_metrics[0]
        train_loss = epoch_metrics[1]
        # update the learning rate
        new_learning_rate = set_learning_rate()

        ######################################################
        # validation, testing, and general diagnostic stuff. #
        ######################################################
        # Set to testing mode
        NET.set_bias_noise(0.0)
        # compute metrics on validation set
        validation_metrics = [0. for v in epoch_metrics]
        for b_idx in xrange(va_batches):
            batch_metrics = validate_model(b_idx)
            validation_metrics = [(em + bm) for (em, bm) in zip(validation_metrics, batch_metrics)]
        # Compute 'averaged' values over the minibatches
        validation_error = 100 * (float(validation_metrics[0]) / va_samples)
        validation_metrics[1:] = [(float(v) / va_batches) for v in validation_metrics[1:]]
        validation_loss = validation_metrics[1]

        # compute test error if new best validation error was found
        tag = " "
        # compute metrics on testing set
        test_metrics = [0. for v in epoch_metrics]
        for b_idx in xrange(te_batches):
            batch_metrics = test_model(b_idx)
            test_metrics = [(em + bm) for (em, bm) in zip(test_metrics, batch_metrics)]
        # Compute 'averaged' values over the minibatches
        test_error = 100 * (float(test_metrics[0]) / te_samples)
        test_metrics[1:] = [(float(v) / te_batches) for v in test_metrics[1:]]
        test_loss = test_metrics[1]
        if (validation_error < min_validation_error):
            min_validation_error = validation_error
            min_test_error = test_error
            tag = ", test={0:.2f}".format(test_error)
        results_file.write("{0:.2f} {1:.2f} {2:.2f} {3:.4f} {4:.4f} {5:.4f}\n".format( \
                train_error, validation_error, test_error, train_loss, validation_loss, test_loss))
        results_file.flush()

        # report and save progress.
        print "epoch {0:d}: t_err={1:.2f}, t_loss={2:.4f}, t_dev={3:.4f}, t_reg={4:.4f}, valid={5:.2f}{6}".format( \
                epoch_counter, epoch_metrics[0], epoch_metrics[1], epoch_metrics[2], epoch_metrics[3], \
                validation_error, tag)
        print "--time: {0:.4f}".format((time.clock() - e_time))
        e_time = time.clock()
        # save first layer weights to an image locally
        utils.visualize(NET, 0, img_file_name)

    print("optimization complete. best validation error {0:.4f}, with test error {1:.4f}".format( \
          (min_validation_error), (min_test_error)))

def train_dae(
    NET,
    dae_layer,
    mlp_params,
    sgd_params,
    datasets):
    """
    Train some layer of NET as an autoencoder of its input source.

    Datasets should be a three-tuple, in which each item is a matrix of
    observations. the first, second, and third matrices will be used for
    training, validation, and testing, respectively.
    """
    initial_learning_rate = sgd_params['start_rate']
    learning_rate_decay = sgd_params['decay_rate']
    n_epochs = sgd_params['epochs']
    batch_size = sgd_params['batch_size']
    mlp_type = sgd_params['mlp_type']
    wt_norm_bound = sgd_params['wt_norm_bound']
    result_tag = sgd_params['result_tag']
    txt_file_name = "results_dae_{0}.txt".format(result_tag)
    img_file_name = "weights_dae_{0}.png".format(result_tag)

    # Get the training data and create arrays of start/end indices for
    # easy minibatch slicing
    Xtr = datasets[0][0]
    tr_samples = Xtr.get_value(borrow=True).shape[0]
    tr_batches = int(np.ceil(float(tr_samples) / batch_size))
    tr_bidx = [[i*batch_size, min(tr_samples, (i+1)*batch_size)] for i in range(tr_batches)]
    tr_bidx = T.cast(tr_bidx, 'int32')
    # Get the validation and testing sets and create arrays of start/end
    # indices for easy minibatch slicing
    Xva = datasets[1][0]
    Xte = datasets[2][0]
    va_samples = Xva.get_value(borrow=True).shape[0]
    te_samples = Xte.get_value(borrow=True).shape[0]
    va_batches = int(np.ceil(va_samples / 100.))
    te_batches = int(np.ceil(te_samples / 100.))
    va_bidx = [[i*100, min(va_samples, (i+1)*100)] for i in range(va_batches)]
    te_bidx = [[i*100, min(te_samples, (i+1)*100)] for i in range(te_batches)]
    va_bidx = theano.shared(value=np.asarray(va_bidx, dtype=theano.config.floatX))
    te_bidx = theano.shared(value=np.asarray(te_bidx, dtype=theano.config.floatX))
    va_bidx = T.cast(va_bidx, 'int32')
    te_bidx = T.cast(te_bidx, 'int32')

    print "Dataset info:"
    print "  training samples: {0:d}".format(tr_samples)
    print "  samples/minibatch: {0:d}, minibatches/epoch: {1:d}".format( \
        batch_size, tr_batches)
    print "  validation samples: {0:d}, testing samples: {1:d}".format( \
        va_samples, te_samples)

    ######################
    # BUILD ACTUAL MODEL #
    ######################

    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    epoch = T.scalar()   # epoch counter
    x = NET.input        # symbolic matrix for inputs to NET
    learning_rate = theano.shared(np.asarray(initial_learning_rate,
        dtype=theano.config.floatX))

    # Build the expressions for the cost functions.
    #grd_loss = 0.5 * (NET.grad_losses[dae_layer][0] + 10.0*NET.grad_losses[dae_layer][1])
    #sde_cost = NET.sde_dae_losses[dae_layer][0] + NET.sde_dae_losses[dae_layer][1] #+ grd_loss
    #raw_cost = NET.raw_dae_losses[dae_layer][0] + NET.raw_dae_losses[dae_layer][1] #+ grd_loss
    #NET_metrics = [raw_cost, NET.raw_dae_losses[dae_layer][0], \
    #        NET.raw_dae_losses[dae_layer][1]]
    sde_cost = NET.dae_costs[dae_layer][0] + NET.dae_costs[dae_layer][1]
    raw_cost = NET.dae_costs[dae_layer][0] + NET.dae_costs[dae_layer][1]
    NET_metrics = [raw_cost, NET.dae_costs[dae_layer][0], NET.dae_costs[dae_layer][1]]
    opt_params = NET.dae_params[dae_layer]

    ############################################################################
    # Compile testing and validation models. These models are evaluated on     #
    # batches of the same size as used in training. Trying to jam a large      #
    # validation or test set through the net may be stupid.                    #
    ############################################################################
    test_model = theano.function(inputs=[index],
        outputs=NET_metrics,
        givens={ x: Xte[te_bidx[index,0]:te_bidx[index,1],:] })

    validate_model = theano.function(inputs=[index],
        outputs=NET_metrics,
        givens={ x: Xva[va_bidx[index,0]:va_bidx[index,1],:] })

    ############################################################################
    # Prepare momentum and gradient variables, and construct the updates that  #
    # Theano will perform on the network parameters.                           #
    ############################################################################
    sde_grads = []
    raw_grads = []
    for param in opt_params:
        sde_grads.append(T.grad(sde_cost, param))
        raw_grads.append(T.grad(raw_cost, param))

    sde_moms = []
    raw_moms = []
    for param in opt_params:
        sde_moms.append(theano.shared(np.zeros( \
                param.get_value(borrow=True).shape, dtype=theano.config.floatX)))
        raw_moms.append(theano.shared(np.zeros( \
                param.get_value(borrow=True).shape, dtype=theano.config.floatX)))

    # Compute momentum for the current epoch
    mom = ifelse(epoch < 500,
        0.5*(1. - epoch/500.) + 0.99*(epoch/500.),
        0.99)

    # Use a "smoothed" learning rate, to ease into optimization
    gentle_rate = ifelse(epoch < 20,
        ((epoch / 20.)**1.) * learning_rate,
        learning_rate)

    # Update the step direction using a momentus update
    sde_updates = OrderedDict()
    raw_updates = OrderedDict()
    for i in range(len(opt_params)):
        sde_updates[sde_moms[i]] = mom * sde_moms[i] + (1. - mom) * sde_grads[i]
        raw_updates[raw_moms[i]] = mom * raw_moms[i] + (1. - mom) * raw_grads[i]

    # ... and take a step along that direction
    for i in range(len(opt_params)):
        param = opt_params[i]
        sde_param = param - (gentle_rate * sde_updates[sde_moms[i]])
        raw_param = param - (gentle_rate * raw_updates[raw_moms[i]])
        # Clip the updated param to bound its norm (where applicable)
        if (NET.clip_params.has_key(param) and \
                (NET.clip_params[param] == 1)):
            sde_norms = T.sum(sde_param**2, axis=1, keepdims=1)
            sde_scale = T.clip(T.sqrt(wt_norm_bound / sde_norms), 0., 1.)
            sde_updates[param] = sde_param * sde_scale
            raw_norms = T.sum(raw_param**2, axis=1, keepdims=1)
            raw_scale = T.clip(T.sqrt(wt_norm_bound / raw_norms), 0., 1.)
            raw_updates[param] = raw_param * raw_scale
        else:
            sde_updates[param] = sde_param
            raw_updates[param] = raw_param

    # Compile theano functions for training.  These return the training cost
    # and update the model parameters.
    train_sde = theano.function(inputs=[epoch, index], \
        outputs=NET_metrics, \
        updates=sde_updates, \
        givens={ x: Xtr[tr_bidx[index,0]:tr_bidx[index,1],:] })

    train_raw = theano.function(inputs=[epoch, index], \
        outputs=NET_metrics, \
        updates=raw_updates, \
        givens={ x: Xtr[tr_bidx[index,0]:tr_bidx[index,1],:] })

    # Theano function to decay the learning rate, this is separate from the
    # training function because we only want to do this once each epoch instead
    # of after each minibatch.
    set_learning_rate = theano.function(inputs=[], outputs=learning_rate,
        updates={learning_rate: learning_rate * learning_rate_decay})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    validation_loss = 1e6
    test_loss = 1e6
    min_validation_loss = 1e6
    min_test_loss = 1e6
    epoch_counter = 0
    start_time = time.clock()

    results_file = open(txt_file_name, 'wb')
    results_file.write("mlp_type: {0}\n".format(mlp_type))
    results_file.write("lam_l2a: {0:.4f}\n".format(mlp_params['lam_l2a']))
    results_file.write("dev_types: {0}\n".format(str(mlp_params['dev_types'])))
    results_file.write("dev_lams: {0}\n".format(str(mlp_params['dev_lams'])))
    results_file.flush()

    epoch_metrics = train_sde(1, 0)
    while epoch_counter < n_epochs:
        ######################################################
        # Process some number of minibatches for this epoch. #
        ######################################################
        e_time = time.clock()
        epoch_counter = epoch_counter + 1
        epoch_metrics = [0. for val in epoch_metrics]
        for minibatch_index in xrange(tr_batches):
            # Compute update for some joint supervised/unsupervised minibatch
            if ((mlp_type == 'sde') or (mlp_type == 'dev')):
                batch_metrics = train_sde(epoch_counter, minibatch_index)
            else:
                batch_metrics = train_raw(epoch_counter, minibatch_index)
            epoch_metrics = [(em + bm) for (em, bm) in zip(epoch_metrics, batch_metrics)]
        epoch_metrics = [(val / tr_batches) for val in epoch_metrics]

        # Update the learning rate
        new_learning_rate = set_learning_rate()

        ######################################################
        # Validation, testing, and general diagnostic stuff. #
        ######################################################
        # Compute metrics on validation set
        validation_metrics = [validate_model(i) for i in xrange(va_batches)]
        validation_loss = np.mean([vm[0] for vm in validation_metrics])

        # Compute test error if new best validation error was found
        tag = " "
        if ((validation_loss < min_validation_loss) or ((epoch_counter % 10) == 0)):
            # Compute metrics on testing set
            test_metrics = [test_model(i) for i in xrange(te_batches)]
            test_loss = np.mean([tm[0] for tm in test_metrics])
            if (validation_loss < min_validation_loss):
                min_validation_loss = validation_loss
                min_test_loss = test_loss
                tag = ", te_loss={0:.4f}".format(test_loss)
        results_file.write("{0:.4f} {1:.4f}\n".format(validation_loss, test_loss))
        results_file.flush()

        # Report and save progress.
        print "epoch {0:d}: tr_loss={1:.4f}, tr_recon={2:.4f}, tr_sparse={3:.4f}, va_loss={4:.4f}{5}".format( \
                epoch_counter, epoch_metrics[0], epoch_metrics[1], epoch_metrics[2], validation_loss, tag)
        print "--time: {0:.4f}".format((time.clock() - e_time))
        # Save first layer weights to an image locally
        utils.visualize(NET, 0, img_file_name)















##############
# EYE BUFFER #
##############
