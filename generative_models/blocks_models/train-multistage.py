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

fuel.config.floatX = theano.config.floatX

#----------------------------------------------------------------------------
def main(name, epochs, batch_size, learning_rate,
         n_iter, z_mix_dim, z_gen_dim, oldmodel):

    datasource = name
    if datasource == 'mnist':
        x_dim = 28*28
        img_height, img_width = (28, 28)
    else:
        raise Exception('Unknown name %s'%datasource)
    
    inits = {
        #'weights_init': Orthogonal(),
        'weights_init': IsotropicGaussian(0.02),
        'biases_init': Constant(0.2),
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
    name = "%s-t%d-zm%d-zg%d-lr%s" % (name, n_iter, z_mix_dim, z_gen_dim, lr_str)

    print("\nRunning experiment %s" % name)
    print("         learning rate: %5.3f" % learning_rate)
    print("          n_iterations: %d" % n_iter)
    print("       z_mix dimension: %d" % z_mix_dim)
    print("       z_gen dimension: %d" % z_gen_dim)
    print()

    #----------------------------------------------------------------------

    s_mix_dim = z_mix_dim
    # setup the guide functions
    q_z_given_x = CondNet(activations=[Softplus(), Softplus()], 
            dims=[x_dim, 500, 500, z_mix_dim], **inits)
    q_hi_given_x_si = CondNet(activations=[Softplus(), Softplus()],
            dims=[(x_dim+x_dim+s_mix_dim), 500, 500, z_gen_dim], **inits)
    # setup the generator functions
    p_hi_given_si = CondNet(activations=[Softplus(), Softplus()],
            dims=[(x_dim+s_mix_dim), 500, 500, z_gen_dim], **inits)
    p_sip1_given_si_hi = MLP(activations=[Softplus(), Softplus(), None],
            dims=[(z_gen_dim + s_mix_dim), 500, 500, x_dim], **inits)

    MSM = MultiStageModel(
                 p_hi_given_si=p_hi_given_si,
                 p_sip1_given_si_hi=p_sip1_given_si_hi,
                 q_z_given_x=q_z_given_x,
                 q_hi_given_x_si=q_hi_given_x_si,
                 n_iter=n_iter)
    MSM.initialize()

    #------------------------------------------------------------------------
    x = tensor.matrix('features')
   
    x_recons, kl_terms = MSM.reconstruct(x, x)

    recons_term = BinaryCrossEntropy().apply(x, x_recons)
    recons_term.name = "recons_term"

    cost = recons_term + kl_terms.sum(axis=0).mean()
    cost.name = "nll_bound"

    #------------------------------------------------------------
    cg = ComputationGraph([cost])
    params = VariableFilter(roles=[PARAMETER])(cg.variables)

    algorithm = GradientDescent(
        cost=cost, 
        params=params,
        step_rule=CompositeRule([
            StepClipping(10.), 
            Adam(learning_rate=learning_rate,\
                 beta1=0.25, beta2=0.02),
        ])
    )

    #------------------------------------------------------------------------
    # Setup monitors
    monitors = [cost]
    for t in range(n_iter+1):
        kl_term_t = kl_terms[t,:].mean()
        kl_term_t.name = "kl_term_%d" % t

        #x_recons_t = T.nnet.sigmoid(c[t,:,:])
        #recons_term_t = BinaryCrossEntropy().apply(x, x_recons_t)
        #recons_term_t = recons_term_t.mean()
        #recons_term_t.name = "recons_term_%d" % t

        monitors += [kl_term_t]

    train_monitors = monitors[:]
    train_monitors += [aggregation.mean(algorithm.total_gradient_norm)]
    train_monitors += [aggregation.mean(algorithm.total_step_norm)]
    # Live plotting...
    plot_channels = [
        ["train_nll_bound", "test_nll_bound"],
        ["train_kl_term_%d" % t for t in range(n_iter+1)],
        #["train_recons_term_%d" % t for t in range(n_iter)],
        ["train_total_gradient_norm", "train_total_step_norm"]
    ]

    #------------------------------------------------------------

    if datasource == 'mnist':
        mnist_train = BinarizedMNIST("train", sources=['features'], flatten=['features'])
        # mnist_valid = BinarizedMNIST("valid", sources=['features'])
        mnist_test = BinarizedMNIST("test", sources=['features'], flatten=['features'])
        train_stream = DataStream(mnist_train, iteration_scheme=SequentialScheme(mnist_train.num_examples, batch_size))
        # valid_stream = DataStream(mnist_valid, iteration_scheme=SequentialScheme(mnist_valid.num_examples, batch_size))
        test_stream  = DataStream(mnist_test,  iteration_scheme=SequentialScheme(mnist_test.num_examples, batch_size))
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
            # DataStreamMonitoring(
            #     monitors,
            #     valid_stream,
            #     prefix="valid"),
            DataStreamMonitoring(
                monitors,
                test_stream,
                prefix="test"),
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
                default=100, help="Number of training epochs to do")
    parser.add_argument("--bs", "--batch-size", type=int, dest="batch_size",
                default=200, help="Size of each mini-batch")
    parser.add_argument("--lr", "--learning-rate", type=float, dest="learning_rate",
                default=1e-3, help="Learning rate")
    parser.add_argument("--niter", type=int, dest="n_iter",
                default=10, help="No. of iterations")
    parser.add_argument("--z_mix_dim", type=int, dest="z_mix_dim",
                default=25, help="Continuous mixture dimension")
    parser.add_argument("--z_gen_dim", type=int, dest="z_gen_dim",
                default=100, help="Latent step embedding dimension")
    parser.add_argument("--oldmodel", type=str,
                help="Use a model pkl file created by a previous run as a starting point for all parameters")
    args = parser.parse_args()

    main(**vars(args))

