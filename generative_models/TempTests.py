import time
import numpy as np
import numpy.random as npr
import theano
import theano.tensor as T
import utils as utils
from load_data import load_udm, load_udm_ss, load_mnist, load_svhn, load_tfd
from PeaNet import PeaNet, load_peanet_from_file
from InfNet import InfNet, load_infnet_from_file
from GenNet import GenNet, load_gennet_from_file
import PeaNet as PNet
import InfNet as INet
import GenNet as GNet
from GIPair import GIPair
from GIPair2 import GIPair2
from GIStack import GIStack
from NetLayers import relu_actfun, softplus_actfun, \
                      safe_softmax, safe_log, tanh_actfun
from DKCode import PCA_theano

####################
# HELPER FUNCTIONS #
####################

def nan_debug_print(x, str='damn the nan'):
    """
    Work around theano's debugging deficiencies.
    """
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        print(str)
    return x

def binarize_data(X):
    """
    Make a sample of bernoulli variables with probabilities given by X.
    """
    X_shape = X.shape
    probs = npr.rand(*X_shape)
    X_binary = 1.0 * (probs < X)
    return X_binary.astype(theano.config.floatX)

def mnist_prob_embed(X, Y):
    """
    Embed the predicted class probabilities in Y in the digits images in X.
    """
    obs_count = X.shape[0]
    class_count = Y.shape[1]
    Xy = np.zeros(X.shape)
    for i in range(obs_count):
        x_sq = X[i,:].reshape((28,28))
        for j in range(class_count):
            x_sq[((2*j)+1):((2*j+2)+1),0:3] = Y[i,j]
            x_sq[((2*j)+1):((2*j+2)+1),3] = 0.2
            x_sq[((2*j)+1):((2*j+2)+1),0] = 0.2
        x_sq[0,0:4] = 0.2
        x_sq[((2*class_count)+1),0:4] = 0.2
        Xy[i,:] = x_sq.flatten()
    return Xy

def rand_sample(param_list):
    """
    Sample a value uniformly at random from the given (python) list.
    """
    new_list = [val for val in param_list]
    npr.shuffle(new_list)
    rand_val = new_list[0]
    return rand_val

def one_hot_np(Yc, cat_dim=None):
    """
    Given a numpy integer column vector Yc, generate a matrix Yoh in which
    Yoh[i,:] is a one-hot vector -- Yoh[i,Yc[i]] = 1.0 and other Yoh[i,j] = 0
    """
    if cat_dim is None:
        cat_dim = np.max(Yc) + 1
    Yoh = np.zeros((Yc.size, cat_dim))
    Yoh[np.arange(Yc.size),Yc.flatten()] = 1.0
    return Yoh

def zmuv(X, axis=1):
    X = X - np.mean(X, axis=axis, keepdims=True)
    X = X / np.std(X, axis=axis, keepdims=True)
    return X

########################
########################
## TESTING FOR GIPair ##
########################
########################

def test_gip2_mnist_60k():
    # Initialize a source of randomness
    rng = np.random.RandomState(1234)

    # Load some data to train/validate/test with
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm(dataset, zero_mean=False)
    Xtr_shared = datasets[0][0]
    Xva_shared = datasets[1][0]
    Xtr = Xtr_shared.get_value(borrow=False).astype(theano.config.floatX)
    Xva = Xva_shared.get_value(borrow=False).astype(theano.config.floatX)
    tr_samples = Xtr.shape[0]
    batch_size = 1000
    batch_reps = 5

    # Setup basic symbolic variables and model parameters
    Xp = T.matrix('Xp_base')
    Xd = T.matrix('Xd_base')
    Xc = T.matrix('Xc_base')
    Xm = T.matrix('Xm_base')
    data_dim = Xtr.shape[1]
    prior_dim = 32
    prior_dim_2 = 32
    prior_sigma = 1.0

    # Load the bottom inferencer and generator from disk
    gn_fname = "MMS_RESULTS_32D/pt60k_walk_params_b300000_GN.pkl"
    in_fname = "MMS_RESULTS_32D/pt60k_walk_params_b300000_IN.pkl"
    IN = INet.load_infnet_from_file(f_name=in_fname, rng=rng, Xd=Xd, Xc=Xc, Xm=Xm)
    GN = GNet.load_gennet_from_file(f_name=gn_fname, rng=rng, Xp=Xp)

    # choose some parameters for the top generator network
    gn2_params = {}
    gn2_config = [prior_dim_2, 500, 500, prior_dim]
    gn2_params['mlp_config'] = gn2_config
    gn2_params['activation'] = tanh_actfun
    gn2_params['out_type'] = 'gaussian'
    gn2_params['mean_transform'] = 'sigmoid'
    gn2_params['logvar_type'] = 'multi_shared'
    gn2_params['init_scale'] = 2.0
    gn2_params['lam_l2a'] = 1e-3
    gn2_params['vis_drop'] = 0.0
    gn2_params['hid_drop'] = 0.0
    gn2_params['bias_noise'] = 0.1
    # choose some parameters for the top inference network
    in2_params = {}
    shared_config = [prior_dim, 500, 500]
    top_config = [shared_config[-1], prior_dim_2]
    in2_params['shared_config'] = shared_config
    in2_params['mu_config'] = top_config
    in2_params['sigma_config'] = top_config
    in2_params['activation'] = tanh_actfun
    in2_params['init_scale'] = 1.0
    in2_params['lam_l2a'] = 1e-3
    in2_params['vis_drop'] = 0.0
    in2_params['hid_drop'] = 0.0
    in2_params['bias_noise'] = 0.1
    in2_params['input_noise'] = 0.0
    # Initialize the base networks for this GIPair
    IN2 = InfNet(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, prior_sigma=prior_sigma, \
            params=in2_params, shared_param_dicts=None)
    GN2 = GenNet(rng=rng, Xp=Xp, prior_sigma=prior_sigma, \
            params=gn2_params, shared_param_dicts=None)
    # Initialize biases in IN and GN
    IN2.init_biases(0.0)
    GN2.init_biases(0.0)
    # Initialize the GIPair
    GIP2 = GIPair2(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, \
            g_net=GN, i_net=IN, \
            data_dim=data_dim, prior_dim=prior_dim, \
            g_net_2=GN2, i_net_2=IN2, \
            prior_dim_2=prior_dim_2, \
            params=None, shared_param_dicts=None)
    GIP2.set_lam_l2w(1e-5)

    # ####################
    # # RICA PRETRAINING #
    # ####################
    # IN2.W_rica.set_value(0.05 * IN2.W_rica.get_value(borrow=False))
    # GN2.W_rica.set_value(0.05 * GN2.W_rica.get_value(borrow=False))
    # for i in range(15000):
    #     scale = min(1.0, (float(i+1) / 5000.0))
    #     l_rate = 0.0002 * scale
    #     lam_l1 = 0.075
    #     tr_idx = npr.randint(low=0,high=tr_samples,size=(1000,))
    #     Xs = Xtr.take(tr_idx, axis=0)
    #     Xd_batch = IN.sample_posterior(Xs, 0.0*Xs, 0.0*Xs)
    #     inr_out = IN2.train_rica(Xd_batch, l_rate, lam_l1)
    #     gnr_out = GN2.train_rica(Xd_batch, l_rate, lam_l1)
    #     #inr_out = [v for v in gnr_out]
    #     if ((i % 1000) == 0):
    #         print("rica batch {0:d}: in_recon={1:.4f}, in_spars={2:.4f}, gn_recon={3:.4f}, gn_spars={4:.4f}".format( \
    #                 i, 1.*inr_out[1], 1.*inr_out[2], 1.*gnr_out[1], 1.*gnr_out[2]))

    ####################################
    # NORMAL TRAINING OF THE GIP STACK #
    ####################################
    out_file = open("pt_gip2_results.txt", 'wb')
    # Set initial learning rate and basic SGD hyper parameters
    cost_1 = [0. for i in range(10)]
    learn_rate = 0.004
    for i in range(150000):
        scale = min(1.0, float(i) / 20000.0)
        # do a minibatch update of the model, and compute some costs
        tr_idx = npr.randint(low=0,high=tr_samples,size=(batch_size,))
        Xb = Xtr.take(tr_idx, axis=0)
        Xd_batch = IN.sample_posterior(Xb, 0.0*Xb, 0.0*Xb)
        Xd_batch = np.repeat(Xd_batch, batch_reps, axis=0)
        Xc_batch = 0.0 * Xd_batch
        Xm_batch = 0.0 * Xd_batch
        # do a minibatch update of the model, and compute some costs
        GIP2.set_all_sgd_params(lr_gn=(scale*learn_rate), \
                lr_in=(scale*learn_rate), mom_1=0.9, mom_2=0.999)
        GIP2.set_lam_nll(1.0)
        GIP2.set_lam_kld(0.2)
        outputs = GIP2.train_top(Xd_batch, Xc_batch, Xm_batch)
        cost_1 = [(cost_1[k] + 1.*outputs[k]) for k in range(len(outputs))]
        # derp?
        if ((i % 1000) == 0):
            cost_1 = [(v / 1000.) for v in cost_1]
            o_str = "batch: {0:d}, joint_cost: {1:.4f}, data_nll_cost: {2:.4f}, post_kld_cost: {3:.4f}, other_reg_cost: {4:.4f}".format( \
                    i, cost_1[0], cost_1[1], cost_1[2], cost_1[3])
            print(o_str)
            out_file.write(o_str+"\n")
            out_file.flush()
            cost_1 = [0. for v in cost_1]
        if ((i % 5000) == 0):
            cost_2 = GIP2.compute_costs(Xva, 0.*Xva, 0.*Xva)
            o_str = "--val: {0:d}, joint_cost: {1:.4f}, data_nll_cost: {2:.4f}, post_kld_cost: {3:.4f}, other_reg_cost: {4:.4f}".format( \
                    i, 1.*cost_2[0], 1.*cost_2[1], 1.*cost_2[2], 1.*cost_2[3])
            print(o_str)
            out_file.write(o_str+"\n")
            out_file.flush()
        if ((i % 5000) == 0):
            tr_idx = npr.randint(low=0,high=tr_samples,size=(100,))
            Xd_batch = Xtr.take(tr_idx, axis=0)
            # sample from top/bottom chains
            file_name = "pt_gip2_chain_samples_b{0:d}_top.png".format(i)
            Xd_samps = np.repeat(Xd_batch[0:10,:], 3, axis=0)
            sample_lists = GIP2.sample_from_chain(Xd_samps, loop_iters=20, which_gip='top')
            Xs = np.vstack(sample_lists["data samples"])
            utils.visualize_samples(Xs, file_name, num_rows=20)
            file_name = "pt_gip2_chain_samples_b{0:d}_bot.png".format(i)
            Xd_samps = np.repeat(Xd_batch[0:10,:], 3, axis=0)
            sample_lists = GIP2.sample_from_chain(Xd_samps, loop_iters=20, which_gip='bot')
            Xs = np.vstack(sample_lists["data samples"])
            utils.visualize_samples(Xs, file_name, num_rows=20)
            # sample from top/bottom priors
            file_name = "pt_gip2_prior_samples_b{0:d}_top.png".format(i)
            Xs = GIP2.sample_from_prior(20*20, which_gip='top')
            utils.visualize_samples(Xs, file_name, num_rows=20)
            file_name = "pt_gip2_prior_samples_b{0:d}_bot.png".format(i)
            Xs = GIP2.sample_from_prior(20*20, which_gip='bot')
            utils.visualize_samples(Xs, file_name, num_rows=20)
    return

#############################################
# TESTING FOR PARZEN LOG_DENSITY ESTIMATION #
#############################################

def test_gip_sigma_scale_mnist():
    from LogPDFs import cross_validate_sigma
    # Simple test code, to check that everything is basically functional.
    print("TESTING...")

    # Initialize a source of randomness
    rng = np.random.RandomState(12345)

    # Load some data to train/validate/test with
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm(dataset, zero_mean=False)
    Xtr = datasets[0][0]
    Xtr = Xtr.get_value(borrow=False)
    Xva = datasets[1][0]
    Xva = Xva.get_value(borrow=False)
    print("Xtr.shape: {0:s}, Xva.shape: {1:s}".format(str(Xtr.shape),str(Xva.shape)))

    # get and set some basic dataset information
    tr_samples = Xtr.shape[0]
    data_dim = Xtr.shape[1]
    batch_size = 100
    prior_dim = 50
    prior_sigma = 1.0
    Xtr_mean = np.mean(Xtr, axis=0, keepdims=True)
    Xtr_mean = (0.0 * Xtr_mean) + np.mean(Xtr)
    Xc_mean = np.repeat(Xtr_mean, batch_size, axis=0).astype(theano.config.floatX)

    # Symbolic inputs
    Xd = T.matrix(name='Xd')
    Xc = T.matrix(name='Xc')
    Xm = T.matrix(name='Xm')
    Xt = T.matrix(name='Xt')
    Xp = T.matrix(name='Xp')

    # Load inferencer and generator from saved parameters
    gn_fname = "MMS_RESULTS_50D_DROPLESS/pt60k_walk_params_b30000_GN.pkl"
    in_fname = "MMS_RESULTS_50D_DROPLESS/pt60k_walk_params_b30000_IN.pkl"
    IN = INet.load_infnet_from_file(f_name=in_fname, rng=rng, Xd=Xd, Xc=Xc, Xm=Xm)
    GN = GNet.load_gennet_from_file(f_name=gn_fname, rng=rng, Xp=Xp)
    # construct a GIPair with the loaded InfNet and GenNet
    GIP = GIPair(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, g_net=GN, i_net=IN, \
            data_dim=data_dim, prior_dim=prior_dim, params=None)
    # draw many samples from the GIP
    for i in range(10):
        tr_idx = npr.randint(low=0,high=tr_samples,size=(100,))
        Xd_batch = Xtr.take(tr_idx, axis=0)
        sample_lists = GIP.sample_from_chain(Xd_batch[0,:].reshape((1,data_dim)), loop_iters=1000, \
                sigma_scale=1.5)
        Xs = np.vstack(sample_lists["data samples"])
        file_name = "AAA_TEST_{0:d}.png".format(i)
        utils.visualize_samples(Xs, file_name, num_rows=30)
    file_name = "AAA_TEST_PRIOR.png"
    Xs = GIP.sample_from_prior(32*32, sigma=1.0)
    utils.visualize_samples(Xs, file_name, num_rows=32)
    # test Parzen density estimator built from prior samples
    Xs = GIP.sample_from_prior(10000, sigma=1.0)
    cross_validate_sigma(Xs, Xva, [0.1, 0.13, 0.15, 0.18, 0.2], 50)
    return

def test_gip_sigma_scale_tfd():
    from LogPDFs import cross_validate_sigma
    # Simple test code, to check that everything is basically functional.
    print("TESTING...")

    # Initialize a source of randomness
    rng = np.random.RandomState(12345)

    # Load some data to train/validate/test with
    data_file = 'data/tfd_data_48x48.pkl'
    dataset = load_tfd(tfd_pkl_name=data_file, which_set='unlabeled', fold='all')
    Xtr_unlabeled = dataset[0]
    dataset = load_tfd(tfd_pkl_name=data_file, which_set='train', fold='all')
    Xtr_train = dataset[0]
    Xtr = np.vstack([Xtr_unlabeled, Xtr_train])
    dataset = load_tfd(tfd_pkl_name=data_file, which_set='test', fold='all')
    Xva = dataset[0]
    tr_samples = Xtr.shape[0]
    va_samples = Xva.shape[0]
    print("Xtr.shape: {0:s}, Xva.shape: {1:s}".format(str(Xtr.shape),str(Xva.shape)))

    # get and set some basic dataset information
    tr_samples = Xtr.shape[0]
    data_dim = Xtr.shape[1]
    batch_size = 100

    # Symbolic inputs
    Xd = T.matrix(name='Xd')
    Xc = T.matrix(name='Xc')
    Xm = T.matrix(name='Xm')
    Xt = T.matrix(name='Xt')
    Xp = T.matrix(name='Xp')

    # Load inferencer and generator from saved parameters
    gn_fname = "TMS_RESULTS_DROPLESS/pt_params_b120000_GN.pkl"
    in_fname = "TMS_RESULTS_DROPLESS/pt_params_b120000_IN.pkl"
    IN = INet.load_infnet_from_file(f_name=in_fname, rng=rng, Xd=Xd, Xc=Xc, Xm=Xm)
    GN = GNet.load_gennet_from_file(f_name=gn_fname, rng=rng, Xp=Xp)
    prior_dim = GN.latent_dim
    prior_sigma = GN.prior_sigma
    # construct a GIPair with the loaded InfNet and GenNet
    GIP = GIPair(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, g_net=GN, i_net=IN, \
            data_dim=data_dim, prior_dim=prior_dim, params=None)
    # draw many samples from the GIP
    for i in range(10):
        tr_idx = npr.randint(low=0,high=tr_samples,size=(100,))
        Xd_batch = Xtr.take(tr_idx, axis=0)
        sample_lists = GIP.sample_from_chain(Xd_batch[0,:].reshape((1,data_dim)), loop_iters=300, \
                sigma_scale=1.0)
        Xs = np.vstack(sample_lists["data samples"])
        file_name = "AAA_TEST_{0:d}.png".format(i)
        utils.visualize_samples(Xs, file_name, num_rows=15)
    file_name = "AAA_TEST_PRIOR.png"
    Xs = GIP.sample_from_prior(32*32, sigma=1.0)
    utils.visualize_samples(Xs, file_name, num_rows=32)
    # test Parzen density estimator built from prior samples
    Xs = GIP.sample_from_prior(10000, sigma=1.0)
    cross_validate_sigma(Xs, Xva, [0.08, 0.09, 0.1, 0.11, 0.12, 0.15], 10)
    return

###################
# TEST DISPATCHER #
###################

if __name__=="__main__":
    #test_gip2_mnist_60k()
    #test_gip_sigma_scale_mnist()
    test_gip_sigma_scale_tfd()
    
