#!/usr/bin/env python

from __future__ import division, print_function

import logging
import numpy as np

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)

import theano
import theano.tensor as T
import fuel
import ipdb

from argparse import ArgumentParser
from collections import OrderedDict
from theano import tensor

from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.datasets.binarized_mnist import BinarizedMNIST

from blocks.algorithms import GradientDescent, CompositeRule, StepClipping, RMSProp, Adam
from blocks.initialization import Constant, IsotropicGaussian, Orthogonal 
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.roles import PARAMETER
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.plot import Plot
from blocks.extensions.saveload import Checkpoint, Dump
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.bricks import Tanh, Identity, Rectifier
from blocks.bricks.cost import BinaryCrossEntropy
from blocks.bricks.recurrent import SimpleRecurrent, LSTM
import cPickle as pickle

from models import *

fuel.config.floatX = theano.config.floatX

#----------------------------------------------------------------------------
def main(name, epochs, batch_size, learning_rate,
         z_dim, mix_dim, enc_dim, dec_dim, oldmodel):

    datasource = 'mnist'
    x_dim = 28*28
    im_shape = (28, 28)
    im_rows = 28
    im_cols = 28
    
    rnninits = {
        #'weights_init': Orthogonal(),
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }
    inits = {
        #'weights_init': Orthogonal(),
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }

    #----------------------------------------------------------------------

    # Learning rate
    def lr_tag(value):
        """ Convert a float into a short tag-usable string representation. E.g.:
            0.1   -> 11
            0.01  -> 12
            0.001 -> 13
            0.005 -> 53
        """
        exp = np.floor(np.log10(value))
        leading = ("%e"%value)[0]
        return "%s%d" % (leading, -exp)

    lr_str = lr_tag(learning_rate)
    name = "%s-enc%d-dec%d-zd%d-md%d-lr%s" % (name, enc_dim, dec_dim, z_dim, mix_dim, lr_str)

    print("\nRunning experiment %s" % name)
    print("         learning rate: %5.3f" % learning_rate)
    print("     encoder dimension: %d" % enc_dim)
    print("     decoder dimension: %d" % dec_dim)
    print("           z dimension: %d" % z_dim)
    print("         mix dimension: %d" % mix_dim)
    
    print()

    #----------------------------------------------------------------------
    
    # setup the mixture weight sampler
    enc_x_to_z = CondNet(activations=[Rectifier()], 
            dims=[x_dim, enc_dim, z_dim], **inits)
    enc_z_to_mix = MLP(activations=[Rectifier(), Tanh()],
            dims=[z_dim, enc_dim, (2*dec_dim + mix_dim)], **inits)
    dec_rnn = LSTM(dim=dec_dim, name="RNN_dec", **rnninits)
    dec_mlp_in = MLP(activations=[None],
            dims=[(im_rows + dec_dim + mix_dim), 4*dec_dim], **inits)
    dec_mlp_out = MLP(activations=[None],
            dims=[dec_dim, im_rows], **inits)

    dm_model = DotMatrix(
                enc_x_to_z=enc_x_to_z,
                enc_z_to_mix=enc_z_to_mix,
                dec_rnn=dec_rnn,
                dec_mlp_in=dec_mlp_in,
                dec_mlp_out=dec_mlp_out,
                im_shape=im_shape,
                mix_dim=mix_dim)
    dm_model.initialize()

    #------------------------------------------------------------------------
    x = tensor.matrix('features')
    
    x_recons, kl_terms = dm_model.reconstruct(x, x)
    x_recons.name = "OO_x_recons_OO"

    nll_term = BinaryCrossEntropy().apply(x, x_recons)
    nll_term.name = "nll_term"

    kld_term = kl_terms.mean()
    kld_term.name = "kld_term"

    nll_bound = nll_term + kld_term
    nll_bound.name = "nll_bound"

    # grab the computation graph for the VFE bound on NLL
    cg = ComputationGraph([nll_bound])
    params = VariableFilter(roles=[PARAMETER])(cg.variables)

    # apply some l2 regularization to the model parameters
    reg_term = (1e-5 * sum([tensor.sum(p**2.0) for p in params]))
    reg_term.name = "reg_term"

    # compute the final cost of VFE + regularization
    total_cost = nll_bound + reg_term

    algorithm = GradientDescent(
        cost=total_cost, 
        params=params,
        step_rule=CompositeRule([
            StepClipping(10.), 
            Adam(learning_rate),
        ])
    )

    #------------------------------------------------------------------------
    # Setup monitors
    monitors = [nll_bound, nll_term, kld_term, reg_term]

    train_monitors = monitors[:]
    train_monitors += [aggregation.mean(algorithm.total_gradient_norm)]
    train_monitors += [aggregation.mean(algorithm.total_step_norm)]
    # Live plotting...
    plot_channels = [
        ["train_nll_bound", "valid_nll_bound"],
        ["train_total_gradient_norm", "train_total_step_norm"]
    ]

    #------------------------------------------------------------

    if datasource == 'mnist':
        mnist_train = BinarizedMNIST("train", sources=['features'], flatten=['features'])
        mnist_valid = BinarizedMNIST("valid", sources=['features'], flatten=['features'])
        #mnist_test = BinarizedMNIST("test", sources=['features'], flatten=['features'])
        train_stream = DataStream(mnist_train, iteration_scheme=SequentialScheme(mnist_train.num_examples, batch_size))
        valid_stream = DataStream(mnist_valid, iteration_scheme=SequentialScheme(mnist_valid.num_examples, batch_size))
        #test_stream  = DataStream(mnist_test,  iteration_scheme=SequentialScheme(mnist_test.num_examples, batch_size))
    else:
        raise Exception('Unknown name %s'%datasource)


    main_loop = MainLoop(
        model=Model(total_cost),
        data_stream=train_stream,
        algorithm=algorithm,
        extensions=[
            Timing(),
            FinishAfter(after_n_epochs=epochs),
            TrainingDataMonitoring(
                train_monitors,
                prefix="train",
                after_epoch=True),
            DataStreamMonitoring(
                monitors,
                valid_stream,
                prefix="valid"),
            # DataStreamMonitoring(
            #     monitors,
            #     test_stream,
            #     prefix="test"),
            Checkpoint(name+".pkl", after_epoch=True, save_separately=['log', 'model']),
            Plot(name, channels=plot_channels),
            ProgressBar(),
            Printing()])
    if oldmodel is not None:
        print("Initializing parameters with old model %s"%oldmodel)
        with open(oldmodel, "rb") as f:
            oldmodel = pickle.load(f)
            main_loop.model.set_param_values(oldmodel.get_param_values())
        del oldmodel
    main_loop.run()

#-----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, dest="name",
                default="mnist", help="Name for this experiment")
    parser.add_argument("--epochs", type=int, dest="epochs",
                default=100, help="Number of training epochs to do")
    parser.add_argument("--bs", "--batch-size", type=int, dest="batch_size",
                default=200, help="Size of each mini-batch")
    parser.add_argument("--lr", "--learning-rate", type=float, dest="learning_rate",
                default=1e-3, help="Learning rate")
    parser.add_argument("--z-dim", type=int, dest="z_dim",
                default=25, help="Z-vector dimension")
    parser.add_argument("--mix-dim", type=int, dest="mix_dim",
                default=10, help="Continuous mixture dimension")
    parser.add_argument("--enc-dim", type=int, dest="enc_dim",
                default=256, help="Encoder MLP dimension")
    parser.add_argument("--dec-dim", type=int, dest="dec_dim",
                default=256, help="Decoder RNN dimension")
    parser.add_argument("--oldmodel", type=str,
                help="Use a model pkl file created by a previous run as a starting point for all parameters")
    args = parser.parse_args()

    main(**vars(args))

