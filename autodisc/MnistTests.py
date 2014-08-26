#########################################
# Testing scripts for MNIST experiments #
#########################################

import numpy as np
import theano
import theano.tensor as T
import theano.tensor.shared_randomstreams

from DexNet import DEX_NET
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
    NET.is_semisupervised = 1

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
    NET.is_semisupervised = 0

    # Train the net
    NT.train_mlp(NET=NET, \
        sgd_params=sgd_params, \
        datasets=datasets)
    return

def train_dex(NET, dex_layer, sgd_params):
    """Run DEX training test."""

    # Load some data to train/validate/test with
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm(dataset)

    # Run deep exemplar SVM training
    NT.train_dex(NET=NET, \
        dex_layer=dex_layer, \
        sgd_params=sgd_params, \
        datasets=datasets)
    return

def batch_test_ss_mlp_pt(test_count=10, su_count=1000):
    """Setup basic test for semisupervised EAR-regularized MLP."""

    # Set some reasonable sgd parameters
    sgd_params = {}
    sgd_params['start_rate'] = 0.01
    sgd_params['decay_rate'] = 0.998
    sgd_params['wt_norm_bound'] = 2.0
    sgd_params['epochs'] = 1000
    sgd_params['batch_size'] = 100
    sgd_params['result_tag'] = 'xxx'
    # Set some reasonable mlp parameters
    mlp_params = {}
    # Set up some proto-networks
    pc0 = [28*28, 500, 500, 200, 11]
    mlp_params['proto_configs'] = [pc0]
    # Set up some spawn networks
    sc0 = {'proto_key': 0, 'input_noise': 0.1, 'bias_noise': 0.05, 'do_dropout': True}
    sc1 = {'proto_key': 0, 'input_noise': 0.1, 'bias_noise': 0.05, 'do_dropout': True}
    mlp_params['spawn_configs'] = [sc0, sc1]
    mlp_params['spawn_weights'] = [0.5, 0.5]
    # Set remaining params
    mlp_params['ear_type'] = 5
    mlp_params['ear_lam'] = 0.2
    mlp_params['lam_l2a'] = 1e-3
    mlp_params['use_bias'] = 1

    for test_num in range(test_count):
        rng_seed = test_num
        sgd_params['result_tag'] = "test_{0:d}".format(test_num)

        # Initialize a random number generator for this test
        rng = np.random.RandomState(rng_seed)

        # Construct the DEX_NET object that we will be training
        x_in = T.matrix('x_in')
        NET = DEX_NET(rng=rng, input=x_in, params=mlp_params)
        init_biases(NET, b_init=0.0)

        ##########################################
        # First, pretrain each layer in the mlp. #
        ##########################################
        sgd_params['result_tag'] = "ss_ear_pt_s{0:d}_t{1:d}".format(su_count,test_num)
        sgd_params['batch_size'] = 25
        sgd_params['start_rate'] = 0.01
        sgd_params['epochs'] = 40
        for i in range(len(NET.dex_costs)):
            print("==================================================")
            print("Pretraining hidden layer(s) at depth {0:d}".format(i+1))
            print("==================================================")
            train_dex(NET, i, sgd_params)

        # Run semisupervised training on the given MLP
        sgd_params['batch_size'] = 100
        sgd_params['start_rate'] = 0.02
        # Train with weak EAR regularization
        sgd_params['top_only'] = True
        sgd_params['epochs'] = 5
        NET.set_ear_lam(0.01)
        rng = np.random.RandomState(rng_seed)
        train_ss_mlp(NET, sgd_params, rng, su_count)
        # Train with more EAR regularization
        sgd_params['top_only'] = False
        sgd_params['epochs'] = 5
        NET.set_ear_lam(0.02)
        rng = np.random.RandomState(rng_seed)
        train_ss_mlp(NET, sgd_params, rng, su_count)
        # Train with more EAR regularization
        sgd_params['epochs'] = 10
        NET.set_ear_lam(0.05)
        rng = np.random.RandomState(rng_seed)
        train_ss_mlp(NET, sgd_params, rng, su_count)
        # Train with more EAR regularization
        sgd_params['epochs'] = 10
        NET.set_ear_lam(0.10)
        rng = np.random.RandomState(rng_seed)
        train_ss_mlp(NET, sgd_params, rng, su_count)
        # Train with most EAR regularization
        sgd_params['epochs'] = 500
        NET.set_ear_lam(0.20)
        rng = np.random.RandomState(rng_seed)
        train_ss_mlp(NET, sgd_params, rng, su_count)
    return

if __name__ == '__main__':

    # Run multiple tests of semisupervised learning with DEX pretraining
    batch_test_ss_mlp_pt(test_count=30, su_count=100)
    #batch_test_ss_mlp_pt(test_count=10, su_count=600)
    #batch_test_ss_mlp_pt(test_count=10, su_count=1000)
    #batch_test_ss_mlp_pt(test_count=10, su_count=3000)









##############
# EYE BUFFER #
##############
