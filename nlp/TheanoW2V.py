import numpy as np
import os
import sys
import time
from collections import OrderedDict

import theano
import theano.tensor as T
from theano.ifelse import ifelse
import theano.printing
import numpy.random as npr
import numpy as np

import utils as utils

def shuffle_rows(shitty_shared_var):
    """Shuffle a matrix row-wise, which Theano seems to hate."""
    np_var = shitty_shared_var.get_value(borrow=False)
    # Shuffle the np_var inplace, which is actually a very simple and somewhat
    # fundamental operation that Theano nonetheless discourages adamantly.
    npr.shuffle(np_var)
    shitty_shared_var.set_value(np_var)
    return

def train_ad(
    NET,
    ad_layers,
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
    txt_file_name = "results_ad_{0}.txt".format(result_tag)
    img_file_name = "weights_ad_{0}.png".format(result_tag)

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
    sde_cost = sum([NET.ad_costs[adl][0] for adl in ad_layers]) + \
               sum([NET.ad_costs[adl][1] for adl in range(ad_layers[-1]+1)])
    raw_cost = sum([NET.ad_costs[adl][0] for adl in ad_layers]) + \
               sum([NET.ad_costs[adl][1] for adl in range(ad_layers[-1]+1)])
    NET_metrics = [raw_cost, NET.ad_costs[ad_layers[-1]][0], NET.ad_costs[ad_layers[-1]][1]]
    opt_params = []
    for adl in range(ad_layers[-1]+1):
        if (adl in ad_layers):
            opt_params.extend(NET.ad_params[adl])
        else:
            opt_params.extend(NET.mlp_layers[adl].params)

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

    rng = np.random.RandomState(123)
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(100000))
    epoch_metrics = train_sde(1, 0)
    while epoch_counter < n_epochs:
        ######################################################
        # Process some number of minibatches for this epoch. #
        ######################################################
        e_time = time.clock()
        shuffle_rows(Xtr)
        shuffle_rows(Xva)
        shuffle_rows(Xte)
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
