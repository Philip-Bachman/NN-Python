#########################################
# Testing scripts for MNIST experiments #
#########################################

import numpy as np
import theano
import theano.tensor as T
import theano.tensor.shared_randomstreams

from EarNet import EAR_NET
from load_data import load_udm, load_udm_ss, load_mnist
import NetTrainers as NT

def init_biases(NET, b_init=0.0):
    # Initialize biases in each hidden layer of each proto-network.
    for proto_net in NET.proto_nets:
        for (num, proto_layer) in enumerate(proto_net):
            b_init = proto_layer.b.get_value(borrow=False)
            b_const = np.zeros(b_init.shape, dtype=theano.config.floatX)
            if (num < (len(proto_net)-1)):
                b_const = b_const + b_init
            proto_layer.b.set_value(b_const)
    return

def train_ss_mlp(NET, sgd_params, rng, su_count=1000):
    """Run semi-supervised EA-regularized test."""

    # Load some data to train/validate/test with
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm_ss(dataset, su_count, rng)

    # Tell the net that it's semi-supervised, which will force it to use only
    # unlabeled examples when computing the EA regularizer.
    NET.reg_all_obs = 1

    # Run training on the given NET
    NT.train_ss_mlp(NET=NET, \
        sgd_params=sgd_params, \
        datasets=datasets)
    return

def train_mlp(NET, sgd_params):
    """Run mlp training test."""

    # Load some data to train/validate/test with
    #dataset = 'data/mnist.pkl.gz'
    #datasets = load_udm(dataset)
    dataset = 'data/mnist_batches.npz'
    datasets = load_mnist(dataset)

    # Tell the net that it's not semi-supervised, which will force it to use
    # _all_ examples when computing the EAR regularizer.
    NET.reg_all_obs = 1

    # Train the net
    NT.train_mlp(NET=NET, \
        sgd_params=sgd_params, \
        datasets=datasets)
    return

def train_dae(NET, dae_layer, sgd_params):
    """Run DAE training test."""

    # Load some data to train/validate/test with
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm(dataset)

    # Run denoising autoencoder training on the given layer of NET
    NT.train_dae(NET=NET, \
        dae_layer=dae_layer, \
        sgd_params=sgd_params, \
        datasets=datasets)
    return

if __name__=="__main__":
    # Set some reasonable sgd parameters
    sgd_params = {}
    sgd_params['start_rate'] = 0.1
    sgd_params['decay_rate'] = 0.998
    sgd_params['wt_norm_bound'] = 3.5
    sgd_params['epochs'] = 600
    sgd_params['batch_size'] = 128
    sgd_params['result_tag'] = "TestScript_OUT"
    # Set some reasonable mlp parameters
    mlp_params = {}
    # Set up some proto-networks
    pc0 = [28*28, 500, 500, 11]
    pc1 = [28*28, 500, 500, 11]
    pc2 = [28*28, 500, 500, 11]
    mlp_params['proto_configs'] = [pc0, pc1, pc2]
    # Set up some spawn networks
    sc0 = {'proto_key': 0, 'input_noise': 0.1, 'bias_noise': 0.05, 'do_dropout': True}
    sc1 = {'proto_key': 0, 'input_noise': 0.1, 'bias_noise': 0.05, 'do_dropout': True}
    sc2 = {'proto_key': 1, 'input_noise': 0.1, 'bias_noise': 0.05, 'do_dropout': True}
    sc3 = {'proto_key': 1, 'input_noise': 0.1, 'bias_noise': 0.05, 'do_dropout': True}
    sc4 = {'proto_key': 2, 'input_noise': 0.1, 'bias_noise': 0.05, 'do_dropout': True}
    sc5 = {'proto_key': 2, 'input_noise': 0.1, 'bias_noise': 0.05, 'do_dropout': True}
    mlp_params['spawn_configs'] = [sc0, sc1, sc2, sc3, sc4, sc5]
    mlp_params['spawn_weights'] = [(1.0 / len(mlp_params['spawn_configs'])) \
            for sc in mlp_params['spawn_configs']]
    # Set remaining params
    mlp_params['ear_type'] = 2
    mlp_params['ear_lam'] = 3.0
    mlp_params['lam_l2a'] = 1e-3
    mlp_params['use_bias'] = 1

    # Goofy symbolic sacrament to Theano
    x_in = T.matrix('x_in')

    test_num = 1234
    rng = np.random.RandomState(test_num)
    # Construct the EAR_NET object that we will be training
    NET = EAR_NET(rng=rng, input=x_in, params=mlp_params)
    init_biases(NET, b_init=0.1)
    rng = np.random.RandomState(test_num)
    train_ss_mlp(NET, sgd_params, rng, 1000)










##############
# EYE BUFFER #
##############
