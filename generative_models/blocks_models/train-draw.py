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
from blocks.bricks import Tanh, Identity
from blocks.bricks.cost import BinaryCrossEntropy
from blocks.bricks.recurrent import SimpleRecurrent, LSTM
import cPickle as pickle

from models import *

import sys
sys.path.append("../datasets")

fuel.config.floatX = theano.config.floatX

#----------------------------------------------------------------------------
def main(name, epochs, batch_size, learning_rate, 
         attention, n_iter, enc_dim, dec_dim, z_dim, oldmodel):

    datasource = name
    if datasource == 'mnist':
        x_dim = 28*28
        img_height, img_width = (28, 28)
    elif datasource == 'sketch':
        x_dim = 56*56
        img_height, img_width = (56, 56)
    else:
        raise Exception('Unknown name %s'%datasource)
    
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
    
    if attention != "":
        read_N, write_N = attention.split(',')
    
        read_N = int(read_N)
        write_N = int(write_N)
        read_dim = 2*read_N**2

        reader = AttentionReader(x_dim=x_dim, dec_dim=dec_dim,
                                 width=img_width, height=img_height,
                                 N=read_N, **inits)
        writer = AttentionWriter(input_dim=dec_dim, output_dim=x_dim,
                                 width=img_width, height=img_height,
                                 N=write_N, **inits)
        attention_tag = "r%d-w%d" % (read_N, write_N)
    else:
        read_dim = 2*x_dim

        reader = Reader(x_dim=x_dim, dec_dim=dec_dim, **inits)
        writer = Writer(input_dim=dec_dim, output_dim=x_dim, **inits)

        attention_tag = "full"

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
    name = "DRAW-%s-%s-t%d-enc%d-dec%d-z%d-lr%s" % (name, attention_tag, n_iter, enc_dim, dec_dim, z_dim, lr_str)

    print("\nRunning experiment %s" % name)
    print("         learning rate: %5.5f" % learning_rate)
    print("             attention: %s" % attention)
    print("          n_iterations: %d" % n_iter)
    print("     encoder dimension: %d" % enc_dim)
    print("           z dimension: %d" % z_dim)
    print("     decoder dimension: %d" % dec_dim)
    print()

    #----------------------------------------------------------------------

    encoder_rnn = LSTM(dim=enc_dim, name="RNN_enc", **rnninits)
    decoder_rnn = LSTM(dim=dec_dim, name="RNN_dec", **rnninits)
    encoder_mlp = MLP([Identity()], [(read_dim+dec_dim), 4*enc_dim], name="MLP_enc", **inits)
    decoder_mlp = MLP([Identity()], [             z_dim, 4*dec_dim], name="MLP_dec", **inits)
    q_sampler = Qsampler(input_dim=enc_dim, output_dim=z_dim, **inits)

    draw = DrawModel(
                n_iter, 
                reader=reader,
                encoder_mlp=encoder_mlp,
                encoder_rnn=encoder_rnn,
                sampler=q_sampler,
                decoder_mlp=decoder_mlp,
                decoder_rnn=decoder_rnn,
                writer=writer)
    draw.initialize()

    #------------------------------------------------------------------------
    x = tensor.matrix('features')
    
    #x_recons = 1. + x
    x_recons, kl_terms = draw.reconstruct(x)
    #x_recons, _, _, _, _ = draw.silly(x, n_steps=10, batch_size=100)
    #x_recons = x_recons[-1,:,:]

    #samples = draw.sample(100) 
    #x_recons = samples[-1, :, :]
    #x_recons = samples[-1, :, :]

    nll_term = BinaryCrossEntropy().apply(x, x_recons)
    nll_term.name = "nll_term"

    kld_term = kl_terms.sum(axis=0).mean()
    kld_term.name = "kld_term"

    nll_bound = nll_term + kld_term
    nll_bound.name = "nll_bound"

    # grab the computation graph for the VFE bound on NLL
    cg = ComputationGraph([nll_bound])
    params = VariableFilter(roles=[PARAMETER])(cg.variables)

    # apply some l2 regularization to the model parameters
    reg_term = 1e-5 * sum([tensor.sum(p**2.0) for p in params])
    reg_term.name = "reg_term"

    # compute the final cost of VFE + regularization
    cost = nll_bound + reg_term
    cost.name = "full_cost"

    algorithm = GradientDescent(
        cost=cost, 
        params=params,
        step_rule=CompositeRule([
            StepClipping(10.), 
            Adam(learning_rate),
        ])
        #step_rule=RMSProp(learning_rate),
        #step_rule=Momentum(learning_rate=learning_rate, momentum=0.95)
    )
    #algorithm.add_updates(scan_updates)


    #------------------------------------------------------------------------
    # Setup monitors
    monitors = [cost, nll_bound]
    for t in range(n_iter):
        kl_term_t = kl_terms[t,:].mean()
        kl_term_t.name = "kl_term_%d" % t

        #x_recons_t = T.nnet.sigmoid(c[t,:,:])
        #recons_term_t = BinaryCrossEntropy().apply(x, x_recons_t)
        #recons_term_t = recons_term_t.mean()
        #recons_term_t.name = "recons_term_%d" % t

        monitors +=[kl_term_t]

    train_monitors = monitors[:]
    train_monitors += [aggregation.mean(algorithm.total_gradient_norm)]
    train_monitors += [aggregation.mean(algorithm.total_step_norm)]
    # Live plotting...
    plot_channels = [
        ["train_nll_bound", "valid_nll_bound"],
        ["train_kl_term_%d" % t for t in range(n_iter)],
        #["train_recons_term_%d" % t for t in range(n_iter)],
        ["train_total_gradient_norm", "train_total_step_norm"]
    ]

    #------------------------------------------------------------

    if datasource == 'mnist':
        mnist_train = BinarizedMNIST("train", sources=['features'], flatten=['features'])
        mnist_valid = BinarizedMNIST("test", sources=['features'], flatten=['features'])
        # mnist_test = BinarizedMNIST("test", sources=['features'], flatten=['features'])
        train_stream = DataStream(mnist_train, iteration_scheme=SequentialScheme(mnist_train.num_examples, batch_size))
        valid_stream = DataStream(mnist_valid, iteration_scheme=SequentialScheme(mnist_valid.num_examples, batch_size))
        # test_stream  = DataStream(mnist_test,  iteration_scheme=SequentialScheme(mnist_test.num_examples, batch_size))
    else:
        raise Exception('Unknown name %s'%datasource)


    main_loop = MainLoop(
        model=Model(cost),
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
            # Dump(name),
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
                default=1000, help="Number of training epochs to do")
    parser.add_argument("--bs", "--batch-size", type=int, dest="batch_size",
                default=400, help="Size of each mini-batch")
    parser.add_argument("--lr", "--learning-rate", type=float, dest="learning_rate",
                default=1e-3, help="Learning rate")
    parser.add_argument("--attention", "-a", type=str, default="",
                help="Use attention mechanism (read_window,write_window)")
    parser.add_argument("--niter", type=int, dest="n_iter",
                default=10, help="No. of iterations")
    parser.add_argument("--enc-dim", type=int, dest="enc_dim",
                default=256, help="Encoder RNN state dimension")
    parser.add_argument("--dec-dim", type=int, dest="dec_dim",
                default=256, help="Decoder  RNN state dimension")
    parser.add_argument("--z-dim", type=int, dest="z_dim",
                default=100, help="Z-vector dimension")
    parser.add_argument("--oldmodel", type=str,
                help="Use a model pkl file created by a previous run as a starting point for all parameters")
    args = parser.parse_args()

    main(**vars(args))
