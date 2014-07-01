#########################################
# Testing scripts for MNIST experiments #
#########################################

import numpy as np
import theano
import theano.tensor as T
import theano.tensor.shared_randomstreams

from ADNet import AD_NET
from load_data import load_udm, load_udm_ss, load_mnist
import NetTrainers as NT

def train_ss_mlp(NET, mlp_params, sgd_params, rng, su_count=1000):
    """Run semisupervised DEV-regularized test."""

    # Load some data to train/validate/test with
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm_ss(dataset, su_count, rng)

    # Tell the net that it's semisupervised, which will force it to use only
    # unlabeled examples for computing the DEV regularizer, and it will compute
    # classification loss only for one of the droppy child networks.
    NET.is_semisupervised = 1

    # Run training on the given NET
    NT.train_ss_mlp(NET=NET, \
        mlp_params=mlp_params, \
        sgd_params=sgd_params, \
        datasets=datasets)
    return 1

def train_mlp(NET, mlp_params, sgd_params):
    """Run mlp training test."""

    # Load some data to train/validate/test with
    #dataset = 'data/mnist.pkl.gz'
    #datasets = load_udm(dataset)
    dataset = 'data/mnist_batches.npz'
    datasets = load_mnist(dataset)

    # Tell the net that it's not semisupervised, which will force it to use
    # _all_ examples for computing the DEV regularizer, and it will compute
    # classification loss only for the drop-free network.
    NET.is_semisupervised = 0

    # Train the net
    NT.train_mlp(NET=NET, \
        mlp_params=mlp_params, \
        sgd_params=sgd_params, \
        datasets=datasets)
    return 1

def train_ad(NET, ad_layers, mlp_params, sgd_params):
    """Run DAE training test."""

    # Load some data to train/validate/test with
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm(dataset)

    # Run denoising autoencoder training on the given layer of NET
    NT.train_ad(NET=NET, \
        ad_layers=ad_layers, \
        mlp_params=mlp_params, \
        sgd_params=sgd_params, \
        datasets=datasets)
    return 1

def test_ad(ad_layers=[0], mlp_params=False, sgd_params=False):
    """Setup basic test for semisupervised DEV-regularized MLP."""

    if not sgd_params:
        # Set some reasonable sgd parameters
        sgd_params = {}
        sgd_params['start_rate'] = 0.03
        sgd_params['decay_rate'] = 0.998
        sgd_params['wt_norm_bound'] = 3.0
        sgd_params['epochs'] = 100
        sgd_params['batch_size'] = 32
        sgd_params['mlp_type'] = 'sde'
        sgd_params['result_tag'] = 'xxx'
    if not mlp_params:
        # Set some reasonable mlp parameters
        mlp_params = {}
        mlp_params['layer_sizes'] = [28*28, 256, 256, 11]
        mlp_params['dev_types'] = [1, 1, 2]
        mlp_params['dev_lams'] = [0.1, 0.1, 2.0]
        mlp_params['lam_l2a'] = 1e-3
        mlp_params['use_bias'] = 1

    # Initialize a random number generator for this test
    rng = np.random.RandomState(12345)

    # Construct the SS_DEV_NET object that we will be training
    x_in = T.matrix('x_in')
    NET = AD_NET(rng=rng, input=x_in, params=mlp_params)

    # Initialize biases in each net layer (except final layer) to zero
    for layer in NET.mlp_layers:
        b_init = layer.b.get_value(borrow=False)
        b_const = np.zeros(b_init.shape, dtype=theano.config.floatX) + 0.05
        layer.b.set_value(b_const)

    NET.set_bias_noise(0.1)

    # Run semisupervised training on the given MLP
    train_ad(NET, ad_layers, mlp_params, sgd_params)
    return 1

if __name__ == '__main__':

    # Run a test of denoising autoencoder training
    test_ad(ad_layers=[0], mlp_params=False, sgd_params=False)













##############
# EYE BUFFER #
##############
