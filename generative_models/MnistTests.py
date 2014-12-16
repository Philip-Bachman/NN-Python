import time
import numpy as np
import numpy.random as npr
import theano
import theano.tensor as T
import utils as utils
from load_data import load_udm, load_udm_ss, load_mnist
from PeaNet import PeaNet
from InfNet import InfNet
from GenNet import GenNet, projected_moments
from GIPair import GIPair
from GCPair import GCPair
from GIStack import GIStack
from GITrip import GITrip
from GITonGIP import GITonGIP
from NetLayers import relu_actfun, softplus_actfun, \
                      safe_softmax, safe_log

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

##########################
##########################
## TESTING FOR GITonGIP ##
##########################
##########################

def test_git_on_gip(hyper_params=None, rng_seed=1234):
    assert(not (hyper_params is None))
    # Initialize a source of randomness
    rng = np.random.RandomState(rng_seed)

    # Load some data to train/validate/test with
    sup_count = 100
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm_ss(dataset, sup_count, rng, zero_mean=False)
    Xtr_su = datasets[0][0].get_value(borrow=False)
    Ytr_su = datasets[0][1].get_value(borrow=False)
    Xtr_un = datasets[1][0].get_value(borrow=False)
    Ytr_un = datasets[1][1].get_value(borrow=False)
    # get the unlabeled data
    Xtr_un = np.vstack([Xtr_su, Xtr_un]).astype(theano.config.floatX)
    Ytr_un = np.vstack([Ytr_su[:,np.newaxis], Ytr_un[:,np.newaxis]]).astype(np.int32)
    Ytr_un = 0 * Ytr_un
    # get the labeled data
    Xtr_su = Xtr_su.astype(theano.config.floatX)
    Ytr_su = Ytr_su[:,np.newaxis].astype(np.int32)
    # get observations and labels for the validation set
    Xva = datasets[2][0].get_value(borrow=False).astype(theano.config.floatX)
    Yva = datasets[2][1].get_value(borrow=False).astype(np.int32)
    Yva = Yva[:,np.newaxis] # numpy is dumb
    # get size information for the data
    un_samples = Xtr_un.shape[0]
    su_samples = Xtr_su.shape[0]
    va_samples = Xva.shape[0]

    # set up some symbolic variables for input/output
    Xp = T.matrix('Xp_base')
    Xd = T.matrix('Xd_base')
    Xc = T.matrix('Xc_base')
    Xm = T.matrix('Xm_base')
    Yd = T.icol('Yd_base')

    # set some "shape" parameters for the networks
    data_dim = Xtr_un.shape[1]
    label_dim = 10
    prior_1_dim = 50
    prior_2_dim = 50
    prior_sigma = 1.0
    batch_size = 100

    ##################
    # SETUP A GIPAIR #
    ##################
    gn1_params = {}
    gn1_config = [prior_1_dim, 600, 600, data_dim]
    gn1_params['mlp_config'] = gn1_config
    gn1_params['activation'] = softplus_actfun
    gn1_params['out_type'] = 'bernoulli'
    gn1_params['lam_l2a'] = 1e-3
    gn1_params['vis_drop'] = 0.0
    gn1_params['hid_drop'] = 0.0
    gn1_params['bias_noise'] = 0.1
    gn1_params['out_noise'] = 0.0
    # choose some parameters for the continuous inferencer
    in1_params = {}
    shared_config = [data_dim, 600, 600]
    top_config = [shared_config[-1], prior_1_dim]
    in1_params['shared_config'] = shared_config
    in1_params['mu_config'] = top_config
    in1_params['sigma_config'] = top_config
    in1_params['activation'] = softplus_actfun
    in1_params['lam_l2a'] = 1e-3
    in1_params['vis_drop'] = 0.0
    in1_params['hid_drop'] = 0.0
    in1_params['bias_noise'] = 0.1
    in1_params['input_noise'] = 0.0
    # Initialize the base networks for this GIPair
    IN1 = InfNet(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, prior_sigma=prior_sigma, \
            params=in1_params, shared_param_dicts=None)
    GN1 = GenNet(rng=rng, Xp=Xp, prior_sigma=prior_sigma, \
            params=gn1_params, shared_param_dicts=None)
    # Initialize biases in IN and GN
    IN1.init_biases(0.0)
    GN1.init_biases(0.0)
    # Initialize the GIPair
    GIP = GIPair(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, g_net=GN1, i_net=IN1, \
            data_dim=data_dim, prior_dim=prior_1_dim, \
            params=None, shared_param_dicts=None)
    # Set cost weighting parameters
    GIP.set_lam_nll(1.0)
    GIP.set_lam_kld(1.0)
    GIP.set_lam_l2w(1e-4)

    ##################
    # SETUP A GITRIP #
    ##################
    # set parameters for the generator network
    gn2_params = {}
    gn2_config = [(prior_2_dim + label_dim), 300, prior_1_dim]
    gn2_params['mlp_config'] = gn2_config
    gn2_params['activation'] = softplus_actfun
    gn2_params['out_type'] = 'gaussian'
    gn2_params['lam_l2a'] = 1e-3
    gn2_params['vis_drop'] = 0.0
    gn2_params['hid_drop'] = 0.0
    gn2_params['bias_noise'] = 0.1
    gn2_params['out_noise'] = 0.0
    # choose some parameters for the continuous inferencer
    in2_params = {}
    shared_config = [prior_1_dim, 300]
    top_config = [shared_config[-1], prior_2_dim]
    in2_params['shared_config'] = shared_config
    in2_params['mu_config'] = top_config
    in2_params['sigma_config'] = top_config
    in2_params['activation'] = softplus_actfun
    in2_params['lam_l2a'] = 1e-3
    in2_params['vis_drop'] = 0.0
    in2_params['hid_drop'] = 0.0
    in2_params['bias_noise'] = 0.1
    in2_params['input_noise'] = 0.0
    # choose some parameters for the categorical inferencer
    pn2_params = {}
    pc0 = [prior_1_dim, 300, label_dim]
    pn2_params['proto_configs'] = [pc0]
    # Set up some spawn networks
    sc0 = {'proto_key': 0, 'input_noise': 0.0, 'bias_noise': 0.1, 'do_dropout': False}
    #sc1 = {'proto_key': 0, 'input_noise': 0.1, 'bias_noise': 0.1, 'do_dropout': True}
    pn2_params['spawn_configs'] = [sc0] #[sc0, sc1]
    pn2_params['spawn_weights'] = [1.0] #[0.5, 0.5]
    # Set remaining params
    pn2_params['activation'] = softplus_actfun
    pn2_params['ear_type'] = 6
    pn2_params['lam_l2a'] = 1e-3
    pn2_params['vis_drop'] = 0.0
    pn2_params['hid_drop'] = 0.0

    # Initialize the base networks for this GITrip
    GN2 = GenNet(rng=rng, Xp=Xp, prior_sigma=prior_sigma, \
            params=gn2_params, shared_param_dicts=None)
    IN2 = InfNet(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, prior_sigma=prior_sigma, \
            params=in2_params, shared_param_dicts=None)
    PN2 = PeaNet(rng=rng, Xd=Xd, params=pn2_params)
    # Initialize biases in GN, IN, and PN
    GN2.init_biases(0.0)
    IN2.init_biases(0.0)
    PN2.init_biases(0.0)

    # Initialize the GITrip
    GIT = GITrip(rng=rng, \
            Xd=Xd, Yd=Yd, Xc=Xc, Xm=Xm, \
            g_net=GN2, i_net=IN2, p_net=PN2, \
            data_dim=prior_1_dim, prior_dim=prior_2_dim, \
            label_dim=label_dim, batch_size=batch_size, \
            params=None, shared_param_dicts=None)
    # Set cost weighting parameters
    GIT.set_lam_nll(1.0)
    GIT.set_lam_kld(1.0)
    GIT.set_lam_cat(0.0)
    GIT.set_lam_pea(0.0)
    GIT.set_lam_ent(0.0)
    GIT.set_lam_l2w(1e-4)

    #####################################################
    # CONSTRUCT A GITonGIP STACKED, SEMI-SUPERVISED VAE #
    #####################################################
    GOG = GITonGIP(rng=rng, \
            Xd=Xd, Yd=Yd, Xc=Xc, Xm=Xm, \
            gip_vae=GIP, git_vae=GIT, \
            data_dim=data_dim, prior_1_dim=prior_1_dim, \
            prior_2_dim=prior_2_dim, label_dim=label_dim, \
            batch_size=batch_size, \
            params=None, shared_param_dicts=None)

    #################################
    # WRITE SOME INFO TO "LOG" FILE #
    #################################
    learn_rate_git = hyper_params['learn_rate_git']
    lam_pea_git = hyper_params['lam_pea_git']
    lam_cat_git = hyper_params['lam_cat_git']
    lam_ent_git = hyper_params['lam_ent_git']
    lam_l2w_git = hyper_params['lam_l2w_git']
    out_name = hyper_params['out_name']

    out_file = open(out_name, 'wb')
    out_file.write("**TODO: More informative output, and maybe a real log**\n")
    out_file.write("learn_rate_git: {0:.4f}\n".format(learn_rate_git))
    out_file.write("lam_pea_git: {0:.4f}\n".format(lam_pea_git))
    out_file.write("lam_cat_git: {0:.4f}\n".format(lam_cat_git))
    out_file.write("lam_ent_git: {0:.4f}\n".format(lam_ent_git))
    out_file.write("lam_l2w_git: {0:.4f}\n".format(lam_l2w_git))
    out_file.flush()

    ##################################################
    # TRAIN THE GIPair FOR SOME NUMBER OF ITERATIONS #
    ##################################################
    learn_rate = 0.002
    for i in range(250000):
        if ((i+1 % 100000) == 0):
            learn_rate = learn_rate * 0.8
        scale = min(1.0, (float(i+1) / 50000.0))
        GIP.set_all_sgd_params(learn_rate=(scale*learn_rate), momentum=0.98)
        GIP.set_lam_nll(lam_nll=1.0)
        GIP.set_lam_kld(lam_kld=scale)
        # sample some unlabeled data to train with
        tr_idx = npr.randint(low=0,high=un_samples,size=(batch_size,))
        Xd_batch = binarize_data(Xtr_un.take(tr_idx, axis=0))
        Xc_batch = 0.0 * Xd_batch
        Xm_batch = 0.0 * Xd_batch
        # do a minibatch update of the model, and compute some costs
        outputs = GOG.train_gip(Xd_batch, Xc_batch, Xm_batch)
        joint_cost = 1.0 * outputs[0]
        data_nll_cost = 1.0 * outputs[1]
        post_kld_cost = 1.0 * outputs[2]
        other_reg_cost = 1.0 * outputs[3]
        if ((i % 1000) == 0):
            o_str = "batch: {0:d}, joint_cost: {1:.4f}, data_nll_cost: {2:.4f}, post_kld_cost: {3:.4f}, other_reg_cost: {4:.4f}".format( \
                    i, joint_cost, data_nll_cost, post_kld_cost, other_reg_cost)
            print(o_str)
            out_file.write("{}\n".format(o_str))
            out_file.flush()
        if ((i % 5000) == 0):
            file_name = "GOG_GIP_SAMPLES_b{0:d}.png".format(i)
            Xd_samps = np.repeat(Xd_batch[0:10,:], 3, axis=0)
            sample_lists = GIP.sample_gil_from_data(Xd_samps, loop_iters=10)
            Xs = np.vstack(sample_lists["data samples"])
            utils.visualize_samples(Xs, file_name)

    ########################################################
    # REMOVE (SORT OF) UNUSED DIMENSIONS FROM LATENT SPACE #
    ########################################################
    #tr_idx = npr.randint(low=0,high=un_samples,size=(10000,))
    #Xd_batch = binarize_data(Xtr_un.take(tr_idx, axis=0))
    #Xp_batch = GIP.IN.mean_posterior(Xd_batch, 0.0*Xd_batch, 0.0*Xd_batch)
    #Xp_std = np.std(Xp_batch, axis=0, keepdims=True)
    #dim_mask = 1.0 * (Xp_std > 0.1)
    #GIT.set_input_mask(dim_mask)
    #print("MASK NNZ: {0:.4f}".format(np.sum(dim_mask)))

    ##################################################
    # TRAIN THE GITrip FOR SOME NUMBER OF ITERATIONS #
    ##################################################
    GIT.set_lam_l2w(lam_l2w=lam_l2w_git)
    learn_rate = learn_rate_git
    GIT.set_all_sgd_params(learn_rate=learn_rate, momentum=0.98)
    for i in range(250000):
        scale = 1.0
        if (i < 25000):
            scale = float(i+1) / 25000.0
        if ((i+1 % 50000) == 0):
            learn_rate = learn_rate * 0.8
        # do a minibatch update using unlabeled data
        if True:
            # get some data to train with
            un_idx = npr.randint(low=0,high=un_samples,size=(batch_size,))
            Xd_un = binarize_data(Xtr_un.take(un_idx, axis=0))
            Yd_un = Ytr_un.take(un_idx, axis=0)
            Xc_un = 0.0 * Xd_un
            Xm_un = 0.0 * Xd_un
            # do a minibatch update of the model, and compute some costs
            GIT.set_all_sgd_params(learn_rate=(scale*learn_rate), momentum=0.98)
            GIT.set_lam_nll(1.0)
            GIT.set_lam_kld((scale**2.0) * 1.0)
            GIT.set_lam_cat(0.0)
            GIT.set_lam_pea(scale * lam_pea_git)
            GIT.set_lam_ent((scale**2.0) * lam_ent_git)
            outputs = GOG.train_git(Xd_un, Xc_un, Xm_un, Yd_un)
            joint_cost = 1.0 * outputs[0]
            data_nll_cost = 1.0 * outputs[1]
            post_kld_cost = 1.0 * outputs[2]
            post_cat_cost = 1.0 * outputs[3]
            post_pea_cost = 1.0 * outputs[4]
            post_ent_cost = 1.0 * outputs[5]
            other_reg_cost = 1.0 * outputs[6]
        if True:
            # get some data to train with
            su_idx = npr.randint(low=0,high=su_samples,size=(batch_size,))
            Xd_su = binarize_data(Xtr_su.take(su_idx, axis=0))
            Yd_su = Ytr_su.take(su_idx, axis=0)
            Xc_su = 0.0 * Xd_su
            Xm_su = 0.0 * Xd_su
            # update only based on the label-based classification cost
            GIT.set_all_sgd_params(learn_rate=(scale*learn_rate), momentum=0.98)
            GIT.set_lam_nll(0.0)
            GIT.set_lam_kld(0.0)
            GIT.set_lam_cat(scale * lam_cat_git)
            GIT.set_lam_pea(scale * lam_pea_git)
            GIT.set_lam_ent((scale**2.0) * lam_ent_git)
            outputs = GOG.train_git(Xd_su, Xc_su, Xm_su, Yd_su)
            joint_2 = 1.0 * outputs[0]
            data_nll_2 = 1.0 * outputs[1]
            post_kld_2 = 1.0 * outputs[2]
            post_cat_cost = 1.0 * outputs[3]
            post_pea_2 = 1.0 * outputs[4]
            post_ent_2 = 1.0 * outputs[5]
            other_reg_cost = 1.0 * outputs[6]
        if ((i % 500) == 0):
            o_str = "batch: {0:d}, joint_cost: {1:.4f}, nll: {2:.4f}, kld: {3:.4f}, cat: {4:.4f}, pea: {5:.4f}, ent: {6:.4f}, other_reg: {7:.4f}".format( \
                    i, joint_cost, data_nll_cost, post_kld_cost, post_cat_cost, post_pea_cost, post_ent_cost, other_reg_cost)
            print(o_str)
            out_file.write("{}\n".format(o_str))
            out_file.flush()
            if ((i % 2500) == 0):
                # check classification error on training and validation set
                train_err = GOG.classification_error(Xtr_su, Ytr_su)
                va_err = GOG.classification_error(Xva, Yva)
                o_str = "    tr_err: {0:.4f}, va_err: {1:.4f}".format(train_err, va_err)
                print(o_str)
                out_file.write("{}\n".format(o_str))
                out_file.flush()
        if ((i % 5000) == 0):
            file_name = "GoG_GIT_SAMPLES_b{0:d}.png".format(i)
            va_idx = npr.randint(low=0,high=va_samples,size=(5,))
            Xd_samps = np.vstack([Xd_un[0:5,:], binarize_data(Xva[va_idx,:])])
            Xd_samps = np.repeat(Xd_samps, 3, axis=0)
            sample_lists = GOG.sample_git_from_data(Xd_samps, loop_iters=10)
            Xs = np.vstack(sample_lists["data samples"])
            Ys = GOG.class_probs(Xs)
            Xs = mnist_prob_embed(Xs, Ys)
            utils.visualize_samples(Xs, file_name)

########################
########################
## TESTING FOR GITrip ##
########################
########################

def test_gi_trip(hyper_params=None, sup_count=600, rng_seed=1234):
    assert(not (hyper_params is None))
    # Initialize a source of randomness
    rng = np.random.RandomState(rng_seed)

    # Load some data to train/validate/test with
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm_ss(dataset, sup_count, rng, zero_mean=False)
    Xtr_su = datasets[0][0].get_value(borrow=False)
    Ytr_su = datasets[0][1].get_value(borrow=False)
    Xtr_un = datasets[1][0].get_value(borrow=False)
    Ytr_un = datasets[1][1].get_value(borrow=False)
    # get the unlabeled data
    Xtr_un = np.vstack([Xtr_su, Xtr_un]).astype(theano.config.floatX)
    Ytr_un = np.vstack([Ytr_su[:,np.newaxis], Ytr_un[:,np.newaxis]]).astype(np.int32)
    Ytr_un = 0 * Ytr_un
    # get the labeled data
    Xtr_su = Xtr_su.astype(theano.config.floatX)
    Ytr_su = Ytr_su[:,np.newaxis].astype(np.int32)
    # get observations and labels for the validation set
    Xva = datasets[2][0].get_value(borrow=False).astype(theano.config.floatX)
    Yva = datasets[2][1].get_value(borrow=False).astype(np.int32)
    Yva = Yva[:,np.newaxis] # numpy is dumb
    # get size information for the data
    un_samples = Xtr_un.shape[0]
    su_samples = Xtr_su.shape[0]
    va_samples = Xva.shape[0]

    # set up some symbolic variables for input to the GITrip
    Xp = T.matrix('Xp_base')
    Xd = T.matrix('Xd_base')
    Xc = T.matrix('Xc_base')
    Xm = T.matrix('Xm_base')
    Yd = T.icol('Yd_base')
    # set some "shape" parameters for the networks
    data_dim = Xtr_un.shape[1]
    label_dim = 10
    prior_dim = 50
    prior_sigma = 1.0
    batch_size = 150
    # set parameters for the generator network
    gn_params = {}
    gn_config = [(prior_dim + label_dim), 600, 600, data_dim]
    gn_params['mlp_config'] = gn_config
    gn_params['activation'] = softplus_actfun
    gn_params['lam_l2a'] = 1e-3
    gn_params['vis_drop'] = 0.0
    gn_params['hid_drop'] = 0.0
    gn_params['bias_noise'] = 0.1
    gn_params['out_noise'] = 0.1
    # choose some parameters for the continuous inferencer
    in_params = {}
    shared_config = [data_dim, 600, 600]
    top_config = [shared_config[-1], prior_dim]
    in_params['shared_config'] = shared_config
    in_params['mu_config'] = top_config
    in_params['sigma_config'] = top_config
    in_params['activation'] = softplus_actfun
    in_params['lam_l2a'] = 1e-3
    in_params['vis_drop'] = 0.2
    in_params['hid_drop'] = 0.0
    in_params['bias_noise'] = 0.1
    in_params['input_noise'] = 0.1
    in_params['out_noise'] = 0.1
    # choose some parameters for the categorical inferencer
    pn_params = {}
    pc0 = [data_dim, 800, 800, label_dim]
    pn_params['proto_configs'] = [pc0]
    # Set up some spawn networks
    sc0 = {'proto_key': 0, 'input_noise': 0.1, 'bias_noise': 0.1, 'do_dropout': True}
    sc1 = {'proto_key': 0, 'input_noise': 0.1, 'bias_noise': 0.1, 'do_dropout': True}
    pn_params['spawn_configs'] = [sc0, sc1]
    pn_params['spawn_weights'] = [0.5, 0.5]
    # Set remaining params
    pn_params['activation'] = relu_actfun
    pn_params['ear_type'] = 6
    pn_params['lam_l2a'] = 1e-3
    pn_params['vis_drop'] = 0.2
    pn_params['hid_drop'] = 0.5

    # Initialize the base networks for this GITrip
    GN = GenNet(rng=rng, Xp=Xp, prior_sigma=prior_sigma, \
            params=gn_params, shared_param_dicts=None)
    IN = InfNet(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, prior_sigma=prior_sigma, \
            params=in_params, shared_param_dicts=None)
    PN = PeaNet(rng=rng, Xd=Xd, params=pn_params)
    # Initialize biases in GN, IN, and PN
    GN.init_biases(0.0)
    IN.init_biases(0.0)
    PN.init_biases(0.1)

    # Initialize the GITrip
    GIT = GITrip(rng=rng, \
            Xd=Xd, Yd=Yd, Xc=Xc, Xm=Xm, \
            g_net=GN, i_net=IN, p_net=PN, \
            data_dim=data_dim, prior_dim=prior_dim, \
            label_dim=label_dim, batch_size=batch_size, \
            params={}, shared_param_dicts=None)
    # set weighting parameters for the various costs...
    GIT.set_lam_nll(1.0)
    GIT.set_lam_kld(1.0)
    GIT.set_lam_cat(0.0)
    GIT.set_lam_pea(0.0)
    GIT.set_lam_ent(0.0)
    
    # Set initial learning rate and basic SGD hyper parameters
    num_updates = hyper_params['num_updates']
    learn_rate = hyper_params['learn_rate']
    lam_pea = hyper_params['lam_pea']
    lam_cat = hyper_params['lam_cat']
    lam_ent = hyper_params['lam_ent']
    lam_l2w = hyper_params['lam_l2w']
    out_name = hyper_params['out_name']

    out_file = open(out_name, 'wb')
    out_file.write("**TODO: More informative output, and maybe a real log**\n")
    out_file.write("sup_count: {0:d}\n".format(sup_count))
    out_file.write("learn_rate: {0:.4f}\n".format(learn_rate))
    out_file.write("lam_pea: {0:.4f}\n".format(lam_pea))
    out_file.write("lam_cat: {0:.4f}\n".format(lam_cat))
    out_file.write("lam_ent: {0:.4f}\n".format(lam_ent))
    out_file.write("lam_l2w: {0:.4f}\n".format(lam_l2w))
    out_file.flush()

    GIT.set_lam_l2w(lam_l2w)
    GIT.set_all_sgd_params(learn_rate=learn_rate, momentum=0.98)
    for i in range(num_updates):
        if (i < 75000):
            scale = float(i+1) / 75000.0
            lam_ent = -1.0
        else:
            scale = 1.0
            lam_ent = hyper_params['lam_ent']
        if ((i+1 % 100000) == 0):
            learn_rate = learn_rate * 0.8
        # do a minibatch update using unlabeled data
        if True:
            # get some data to train with
            un_idx = npr.randint(low=0,high=un_samples,size=(batch_size,))
            Xd_un = binarize_data(Xtr_un.take(un_idx, axis=0))
            Yd_un = Ytr_un.take(un_idx, axis=0)
            Xc_un = 0.0 * Xd_un
            Xm_un = 0.0 * Xd_un
            # do a minibatch update of the model, and compute some costs
            GIT.set_all_sgd_params(learn_rate=(scale*learn_rate), momentum=0.98)
            GIT.set_lam_nll(1.0)
            GIT.set_lam_kld(scale * 1.0)
            GIT.set_lam_cat(0.0)
            GIT.set_lam_pea(scale * lam_pea)
            GIT.set_lam_ent(scale * lam_ent)
            outputs = GIT.train_joint(Xd_un, Xc_un, Xm_un, Yd_un)
            joint_cost = nan_debug_print(1.0 * outputs[0], 'NaN in joint 1')
            data_nll_cost = nan_debug_print(1.0 * outputs[1], 'NaN in nll 1')
            post_kld_cost = nan_debug_print(1.0 * outputs[2], 'NaN in kld 1')
            post_cat_cost = nan_debug_print(1.0 * outputs[3], 'NaN in cat 1')
            post_pea_cost = nan_debug_print(1.0 * outputs[4], 'NaN in pea 1')
            post_ent_cost = nan_debug_print(1.0 * outputs[5], 'NaN in ent 1')
            other_reg_cost = nan_debug_print(1.0 * outputs[6], 'NaN in other reg 1')
            #grad_sq_sum = nan_debug_print(1.0 * outputs[7], 'NaN in grad 1')
            #gp_cost = nan_debug_print(1.0 * outputs[8], 'NaN in gp_cost 1')
            #ip_cost = nan_debug_print(1.0 * outputs[9], 'NaN in ip_cost 1')
            #pp_cost = nan_debug_print(1.0 * outputs[9], 'NaN in pp_cost 1')
            #act_cost = nan_debug_print(1.0 * outputs[10], 'NaN in act_cost 1')
            #xxx = nan_debug_print(1.0 * outputs[11], 'NaN in GIT.IN.output_mu 1')
            #xxx = nan_debug_print(1.0 * outputs[12], 'NaN in GIT.IN.output_sigma 1')
        # do another minibatch update incorporating label information
        if True:
            # get some data to train with
            su_idx = npr.randint(low=0,high=su_samples,size=(batch_size,))
            Xd_su = binarize_data(Xtr_su.take(su_idx, axis=0))
            Yd_su = Ytr_su.take(su_idx, axis=0)
            Xc_su = 0.0 * Xd_su
            Xm_su = 0.0 * Xd_su
            # update only based on the label-based classification cost
            GIT.set_all_sgd_params(learn_rate=(scale*learn_rate), momentum=0.98)
            GIT.set_lam_nll(0.0)
            GIT.set_lam_kld(0.0)
            GIT.set_lam_cat(scale * lam_cat)
            GIT.set_lam_pea(scale * lam_pea)
            GIT.set_lam_ent(0.0)
            outputs = GIT.train_joint(Xd_su, Xc_su, Xm_su, Yd_su)
            joint_2 = nan_debug_print(1.0 * outputs[0], 'NaN in joint 2')
            data_nll_2 = nan_debug_print(1.0 * outputs[1], 'NaN in nll 2')
            post_kld_2 = nan_debug_print(1.0 * outputs[2], 'NaN in kld 2')
            post_cat_cost = nan_debug_print(1.0 * outputs[3], 'NaN in cat 2')
            post_pea_2 = nan_debug_print(1.0 * outputs[4], 'NaN in pea 2')
            post_ent_2 = nan_debug_print(1.0 * outputs[5], 'NaN in ent 2')
            other_reg_cost = nan_debug_print(1.0 * outputs[6], 'NaN in other reg 2')
            #grad_sq_sum = nan_debug_print(1.0 * outputs[7], 'NaN in grad 2')
            #gp_cost = nan_debug_print(1.0 * outputs[8], 'NaN in gp_cost 2')
            #ip_cost = nan_debug_print(1.0 * outputs[9], 'NaN in ip_cost 2')
            #pp_cost = nan_debug_print(1.0 * outputs[9], 'NaN in pp_cost 2')
            #act_cost = nan_debug_print(1.0 * outputs[10], 'NaN in act_cost 2')
            #xxx = nan_debug_print(1.0 * outputs[11], 'NaN in GIT.IN.output_mu 2')
            #xxx = nan_debug_print(1.0 * outputs[12], 'NaN in GIT.IN.output_sigma 2')
        assert(not (np.isnan(joint_cost)))
        if ((i % 500) == 0):
            o_str = "batch: {0:d}, joint_cost: {1:.4f}, nll: {2:.4f}, kld: {3:.4f}, cat: {4:.4f}, pea: {5:.4f}, ent: {6:.4f}, other_reg: {7:.4f}".format( \
                    i, joint_cost, data_nll_cost, post_kld_cost, post_cat_cost, post_pea_cost, post_ent_cost, other_reg_cost)
            print(o_str)
            out_file.write("{}\n".format(o_str))
            if ((i % 1000) == 0):
                # check classification error on training and validation set
                train_err = GIT.classification_error(Xtr_su, Ytr_su)
                va_err = GIT.classification_error(Xva, Yva)
                o_str = "    tr_err: {0:.4f}, va_err: {1:.4f}".format(train_err, va_err)
                print(o_str)
                out_file.write("{}\n".format(o_str))
            out_file.flush()
        if ((i % 5000) == 0):
            file_name = "GIT_SAMPLES_b{0:d}.png".format(i)
            va_idx = npr.randint(low=0,high=va_samples,size=(5,))
            Xd_samps = np.vstack([Xd_un[0:5,:], binarize_data(Xva[va_idx,:])])
            Xd_samps = np.repeat(Xd_samps, 3, axis=0)
            sample_lists = GIT.sample_git_from_data(Xd_samps, loop_iters=10)
            Xs = np.vstack(sample_lists["data samples"])
            Ys = GIT.class_probs(Xs)
            Xs = mnist_prob_embed(Xs, Ys)
            utils.visualize_samples(Xs, file_name)
    print("TESTING COMPLETE!")
    out_file.close()
    return

##########################
##########################
## TESTING FOR GIStack ##
##########################
##########################

def test_gi_stack(hyper_params=None, sup_count=600, rng_seed=1234):
    assert(not (hyper_params is None))
    # Initialize a source of randomness
    rng = np.random.RandomState(rng_seed)

    # Load some data to train/validate/test with
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm_ss(dataset, sup_count, rng, zero_mean=False)
    Xtr_su = datasets[0][0].get_value(borrow=False)
    Ytr_su = datasets[0][1].get_value(borrow=False)
    Xtr_un = datasets[1][0].get_value(borrow=False)
    Ytr_un = datasets[1][1].get_value(borrow=False)
    # get the unlabeled data
    Xtr_un = np.vstack([Xtr_su, Xtr_un]).astype(theano.config.floatX)
    Ytr_un = np.vstack([Ytr_su[:,np.newaxis], Ytr_un[:,np.newaxis]]).astype(np.int32)
    Ytr_un = 0 * Ytr_un
    # get the labeled data
    Xtr_su = Xtr_su.astype(theano.config.floatX)
    Ytr_su = Ytr_su[:,np.newaxis].astype(np.int32)
    # get observations and labels for the validation set
    Xva = datasets[2][0].get_value(borrow=False).astype(theano.config.floatX)
    Yva = datasets[2][1].get_value(borrow=False).astype(np.int32)
    Yva = Yva[:,np.newaxis] # numpy is dumb
    # get size information for the data
    un_samples = Xtr_un.shape[0]
    su_samples = Xtr_su.shape[0]
    va_samples = Xva.shape[0]

    # Construct a GenNet and an InfNet, then test constructor for GIPair.
    # Do basic testing, to make sure classes aren't completely broken.
    Xp = T.matrix('Xp_base')
    Xd = T.matrix('Xd_base')
    Xc = T.matrix('Xc_base')
    Xm = T.matrix('Xm_base')
    Yd = T.icol('Yd_base')
    data_dim = Xtr_un.shape[1]
    label_dim = 10
    prior_dim = 50
    prior_sigma = 1.0
    batch_size = 150
    # Choose some parameters for the generator network
    gn_params = {}
    gn_config = [prior_dim, 600, 600, data_dim]
    gn_params['mlp_config'] = gn_config
    gn_params['activation'] = softplus_actfun
    gn_params['lam_l2a'] = 1e-3
    gn_params['vis_drop'] = 0.0
    gn_params['hid_drop'] = 0.0
    gn_params['bias_noise'] = 0.1
    gn_params['out_noise'] = 0.1
    # choose some parameters for the continuous inferencer
    in_params = {}
    shared_config = [data_dim, 600, 600]
    top_config = [shared_config[-1], prior_dim]
    in_params['shared_config'] = shared_config
    in_params['mu_config'] = top_config
    in_params['sigma_config'] = top_config
    in_params['activation'] = softplus_actfun
    in_params['lam_l2a'] = 1e-3
    in_params['vis_drop'] = 0.0
    in_params['hid_drop'] = 0.0
    in_params['bias_noise'] = 0.1
    in_params['input_noise'] = 0.1
    in_params['out_noise'] = 0.1
    # choose some parameters for the categorical inferencer
    pn_params = {}
    pc0 = [prior_dim, 800, 800, label_dim]
    pn_params['proto_configs'] = [pc0]
    # Set up some spawn networks
    sc0 = {'proto_key': 0, 'input_noise': 0.1, 'bias_noise': 0.1, 'do_dropout': True}
    sc1 = {'proto_key': 0, 'input_noise': 0.1, 'bias_noise': 0.1, 'do_dropout': True}
    pn_params['spawn_configs'] = [sc0, sc1]
    pn_params['spawn_weights'] = [0.5, 0.5]
    # Set remaining params
    pn_params['activation'] = relu_actfun
    pn_params['init_scale'] = 2.0
    pn_params['ear_type'] = 6
    pn_params['lam_l2a'] = 1e-3
    pn_params['vis_drop'] = 0.0
    pn_params['hid_drop'] = 0.5

    # Initialize the base networks for this GIPair
    GN = GenNet(rng=rng, Xp=Xp, prior_sigma=prior_sigma, \
            params=gn_params, shared_param_dicts=None)
    IN = InfNet(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, prior_sigma=prior_sigma, \
            params=in_params, shared_param_dicts=None)
    PN = PeaNet(rng=rng, Xd=Xd, params=pn_params)
    # Initialize biases in GN, IN, and PN
    GN.init_biases(0.0)
    IN.init_biases(0.0)
    PN.init_biases(0.1)
    # Initialize the GIStack
    GIS = GIStack(rng=rng, \
            Xd=Xd, Yd=Yd, Xc=Xc, Xm=Xm, \
            g_net=GN, i_net=IN, p_net=PN, \
            data_dim=data_dim, prior_dim=prior_dim, \
            label_dim=label_dim, batch_size=batch_size, \
            params={}, shared_param_dicts=None)
    # set weighting parameters for the various costs...
    GIS.set_lam_nll(1.0)
    GIS.set_lam_kld(1.0)
    GIS.set_lam_cat(0.0)
    GIS.set_lam_pea(0.0)
    GIS.set_lam_ent(0.0)

    # Set initial learning rate and basic SGD hyper parameters
    num_updates = hyper_params['num_updates']
    learn_rate = hyper_params['learn_rate']
    lam_pea = hyper_params['lam_pea']
    lam_cat = hyper_params['lam_cat']
    lam_ent = hyper_params['lam_ent']
    lam_l2w = hyper_params['lam_l2w']
    out_name = hyper_params['out_name']

    out_file = open(out_name, 'wb')
    out_file.write("**TODO: More informative output, and maybe a real log**\n")
    out_file.write("sup_count: {0:d}\n".format(sup_count))
    out_file.write("learn_rate: {0:.4f}\n".format(learn_rate))
    out_file.write("lam_pea: {0:.4f}\n".format(lam_pea))
    out_file.write("lam_cat: {0:.4f}\n".format(lam_cat))
    out_file.write("lam_ent: {0:.4f}\n".format(lam_ent))
    out_file.write("lam_l2w: {0:.4f}\n".format(lam_l2w))
    out_file.flush()

    GIS.set_lam_l2w(lam_l2w)
    GIS.set_all_sgd_params(learn_rate=learn_rate, momentum=0.98)
    for i in range(num_updates):
        if (i < 100000):
            # start with some updates only for the VAE (InfNet and GenNet)
            scale = float(i+1) / 100000.0
            lam_cat = 0.0
            lam_pea = 0.0
            lam_ent = 0.0
            learn_rate_pn = 0.0
        else:
            # move on to updates that include loss from the PeaNet
            scale = 1.0
            lam_cat = hyper_params['lam_cat']
            lam_pea = hyper_params['lam_pea']
            lam_ent = hyper_params['lam_ent']
            learn_rate_pn = learn_rate
        if ((i+1 % 100000) == 0):
            learn_rate = learn_rate * 0.8
        # do a minibatch update using unlabeled data
        if True:
            # get some data to train with
            un_idx = npr.randint(low=0,high=un_samples,size=(batch_size,))
            Xd_un = binarize_data(Xtr_un.take(un_idx, axis=0))
            Yd_un = Ytr_un.take(un_idx, axis=0)
            Xc_un = 0.0 * Xd_un
            Xm_un = 0.0 * Xd_un
            # do a minibatch update of the model, and compute some costs
            GIS.set_all_sgd_params(learn_rate=(scale*learn_rate), momentum=0.98)
            GIS.set_pn_sgd_params(learn_rate=(scale*learn_rate_pn), momentum=0.98)
            GIS.set_lam_nll(1.0)
            GIS.set_lam_kld(scale * 1.0)
            GIS.set_lam_cat(0.0)
            GIS.set_lam_pea(scale * lam_pea)
            GIS.set_lam_ent(0.0)
            outputs = GIS.train_joint(Xd_un, Xc_un, Xm_un, Yd_un)
            joint_cost = 1.0 * outputs[0]
            data_nll_cost = 1.0 * outputs[1]
            post_kld_cost = 1.0 * outputs[2]
            post_cat_cost = 1.0 * outputs[3]
            post_pea_cost = 1.0 * outputs[4]
            post_ent_cost = 1.0 * outputs[5]
            other_reg_cost = 1.0 * outputs[6]
        # do another minibatch update incorporating label information
        if (i >= 100000):
            # get some data to train with
            su_idx = npr.randint(low=0,high=su_samples,size=(batch_size,))
            Xd_su = binarize_data(Xtr_su.take(su_idx, axis=0))
            Yd_su = Ytr_su.take(su_idx, axis=0)
            Xc_su = 0.0 * Xd_su
            Xm_su = 0.0 * Xd_su
            # update only based on the label-based classification cost
            GIS.set_all_sgd_params(learn_rate=(scale*learn_rate), momentum=0.98)
            GIS.set_pn_sgd_params(learn_rate=(scale*learn_rate_pn), momentum=0.98)
            GIS.set_lam_nll(0.0)
            GIS.set_lam_kld(0.0)
            GIS.set_lam_cat(scale * lam_cat)
            GIS.set_lam_pea(scale * lam_pea)
            GIS.set_lam_ent(scale * lam_ent)
            outputs = GIS.train_joint(Xd_su, Xc_su, Xm_su, Yd_su)
            post_cat_cost = 1.0 * outputs[3]
        assert(not (np.isnan(joint_cost)))
        if ((i % 500) == 0):
            o_str = "batch: {0:d}, joint_cost: {1:.4f}, nll: {2:.4f}, kld: {3:.4f}, cat: {4:.4f}, pea: {5:.4f}, ent: {6:.4f}, other_reg: {7:.4f}".format( \
                    i, joint_cost, data_nll_cost, post_kld_cost, post_cat_cost, post_pea_cost, post_ent_cost, other_reg_cost)
            print(o_str)
            out_file.write("{}\n".format(o_str))
            if ((i % 1000) == 0):
                # check classification error on training and validation set
                train_err = GIS.classification_error(Xtr_su, Ytr_su)
                va_err = GIS.classification_error(Xva, Yva)
                o_str = "    tr_err: {0:.4f}, va_err: {1:.4f}".format(train_err, va_err)
                print(o_str)
                out_file.write("{}\n".format(o_str))
            out_file.flush()
        if ((i % 5000) == 0):
            file_name = "GIS_SAMPLES_b{0:d}.png".format(i)
            va_idx = npr.randint(low=0,high=va_samples,size=(5,))
            Xd_samps = np.vstack([Xd_un[0:5,:], binarize_data(Xva[va_idx,:])])
            Xd_samps = np.repeat(Xd_samps, 3, axis=0)
            sample_lists = GIS.sample_gis_from_data(Xd_samps, loop_iters=10)
            Xs = np.vstack(sample_lists["data samples"])
            Ys = GIS.class_probs(Xs)
            Xs = mnist_prob_embed(Xs, Ys)
            utils.visualize_samples(Xs, file_name)
    print("TESTING COMPLETE!")
    out_file.close()
    return

########################
########################
## TESTING FOR GCPair ##
########################
########################

def test_gc_pair():
    # Simple test code, to check that everything is basically functional.
    print("TESTING...")

    # Initialize a source of randomness
    rng = np.random.RandomState(1234)

    # Load some data to train/validate/test with
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm(dataset, zero_mean=False)
    Xtr = datasets[0][0]
    tr_samples = Xtr.get_value(borrow=True).shape[0]
    data_dim = Xtr.get_value(borrow=True).shape[1]
    mm_proj_dim = 250

    # Do moment matching in some transformed space
    #P = np.identity(data_dim)
    P = npr.randn(data_dim, mm_proj_dim) / np.sqrt(float(mm_proj_dim))
    P = theano.shared(value=P.astype(theano.config.floatX), name='P_proj')

    target_mean, target_cov = projected_moments(Xtr, P, ary_type='theano')
    P = P.get_value(borrow=False).astype(theano.config.floatX)

    ###########################
    # Setup generator network #
    ###########################

    # Choose some parameters for the generative network
    gn_params = {}
    gn_config = [200, 800, 800, 28*28]
    gn_params['mlp_config'] = gn_config
    gn_params['lam_l2a'] = 1e-3
    gn_params['vis_drop'] = 0.0
    gn_params['hid_drop'] = 0.0
    gn_params['bias_noise'] = 0.1
    gn_params['out_noise'] = 0.1
    gn_params['activation'] = softplus_actfun

    # Symbolic input matrix to generator network
    Xp_sym = T.matrix(name='Xp_sym')
    Xd_sym = T.matrix(name='Xd_sym')

    # Initialize a generator network object
    GN = GenNet(rng=rng, Xp=Xp_sym, prior_sigma=5.0, params=gn_params)

    ###############################
    # Setup discriminator network #
    ###############################

    # Set some reasonable mlp parameters
    dn_params = {}
    # Set up some proto-networks
    pc0 = [28*28, (200, 4), (200, 4), 11]
    dn_params['proto_configs'] = [pc0]
    # Set up some spawn networks
    sc0 = {'proto_key': 0, 'input_noise': 0.1, 'bias_noise': 0.1, 'do_dropout': True}
    #sc1 = {'proto_key': 0, 'input_noise': 0.1, 'bias_noise': 0.1, 'do_dropout': True}
    dn_params['spawn_configs'] = [sc0]
    dn_params['spawn_weights'] = [1.0]
    # Set remaining params
    dn_params['ear_type'] = 2
    dn_params['ear_lam'] = 0.0
    dn_params['lam_l2a'] = 1e-3
    dn_params['vis_drop'] = 0.2
    dn_params['hid_drop'] = 0.5

    # Initialize a network object to use as the discriminator
    DN = PeaNet(rng=rng, Xd=Xd_sym, params=dn_params)

    ########################################################################
    # Initialize the joint controller for the generator/discriminator pair #
    ########################################################################

    gcp_params = {}
    gcp_params['d_net'] = DN
    gcp_params['g_net'] = GN
    gcp_params['lam_l2d'] = 1e-2
    gcp_params['mom_mix_rate'] = 0.05
    gcp_params['mom_match_weight'] = 0.05
    gcp_params['mom_match_proj'] = P
    gcp_params['target_mean'] = target_mean
    gcp_params['target_cov'] = target_cov

    # Initialize a GCPair instance using the previously constructed generator and
    # discriminator networks.
    GCP = GCPair(rng=rng, Xd=Xd_sym, Xp=Xp_sym, d_net=DN, g_net=GN, \
            data_dim=28*28, params=gcp_params)

    gn_learn_rate = 0.04
    dn_learn_rate = 0.02
    GCP.set_gn_sgd_params(learn_rate=gn_learn_rate, momentum=0.98)
    GCP.set_dn_sgd_params(learn_rate=dn_learn_rate, momentum=0.98)
    # Init generator's mean and covariance estimates with many samples
    GCP.init_moments(10000)

    batch_idx = T.lvector('batch_idx')
    batch_sample = theano.function(inputs=[ batch_idx ], \
            outputs=Xtr.take(batch_idx, axis=0))

    for i in range(750000):
        tr_idx = npr.randint(low=0,high=tr_samples,size=(100,)).astype(np.int32)
        Xn_np = GN.sample_from_prior(100)
        Xd_batch = batch_sample(tr_idx)
        Xd_batch = Xd_batch.astype(theano.config.floatX)
        Xn_batch = Xn_np.astype(theano.config.floatX)
        all_idx = np.arange(200)
        data_idx = all_idx[:100]
        noise_idx = all_idx[100:]
        scale = min(1.0, float(i+1)/10000.0)
        GCP.set_disc_weights(dweight_gn=scale, dweight_dn=scale)
        outputs = GCP.train_joint(Xd_batch, Xn_batch, data_idx, noise_idx)
        mom_match_cost = 1.0 * outputs[0]
        disc_cost_gn = 1.0 * outputs[1]
        disc_cost_dn = 1.0 * outputs[2]
        if ((i+1 % 100000) == 0):
            gn_learn_rate = gn_learn_rate * 0.7
            dn_learn_rate = dn_learn_rate * 0.7
            GCP.set_gn_sgd_params(learn_rate=gn_learn_rate, momentum=0.98)
            GCP.set_dn_sgd_params(learn_rate=dn_learn_rate, momentum=0.98)
        if ((i % 500) == 0):
            print("batch: {0:d}, mom_match_cost: {1:.4f}, disc_cost_gn: {2:.4f}, disc_cost_dn: {3:.4f}".format( \
                    i, mom_match_cost, disc_cost_gn, disc_cost_dn))
        if ((i % 500) == 0):
            file_name = "GCP_SAMPLES_b{0:d}.png".format(i)
            Xs = GCP.sample_from_gn(200)
            utils.visualize_samples(Xs, file_name)
            file_name = "GCP_WEIGHTS_b{0:d}.png".format(i)
            utils.visualize(GCP.DN, 0, 0, file_name)
    print("TESTING COMPLETE!")
    return

########################
########################
## TESTING FOR GIPair ##
########################
########################

def test_gi_pair():
    # Initialize a source of randomness
    rng = np.random.RandomState(1234)

    # Load some data to train/validate/test with
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm(dataset, zero_mean=False)
    Xtr = datasets[0][0].get_value(borrow=False).astype(theano.config.floatX)
    tr_samples = Xtr.shape[0]

    # Construct a GenNet and an InfNet, then test constructor for GIPair.
    # Do basic testing, to make sure classes aren't completely broken.
    Xp = T.matrix('Xp_base')
    Xd = T.matrix('Xd_base')
    Xc = T.matrix('Xc_base')
    Xm = T.matrix('Xm_base')
    data_dim = Xtr.shape[1]
    prior_dim = 128
    prior_sigma = 2.0
    # Choose some parameters for the generator network
    gn_params = {}
    gn_config = [prior_dim, 800, 800, data_dim]
    gn_params['mlp_config'] = gn_config
    gn_params['activation'] = softplus_actfun
    gn_params['lam_l2a'] = 1e-3
    gn_params['vis_drop'] = 0.0
    gn_params['hid_drop'] = 0.0
    gn_params['bias_noise'] = 0.1
    gn_params['out_noise'] = 0.0
    # choose some parameters for the continuous inferencer
    in_params = {}
    shared_config = [data_dim, (200, 4)]
    top_config = [shared_config[-1], (200, 4), prior_dim]
    in_params['shared_config'] = shared_config
    in_params['mu_config'] = top_config
    in_params['sigma_config'] = top_config
    in_params['activation'] = relu_actfun
    in_params['lam_l2a'] = 1e-3
    in_params['vis_drop'] = 0.0
    in_params['hid_drop'] = 0.0
    in_params['bias_noise'] = 0.1
    in_params['input_noise'] = 0.0
    # Initialize the base networks for this GIPair
    IN = InfNet(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, prior_sigma=prior_sigma, \
            params=in_params, shared_param_dicts=None)
    GN = GenNet(rng=rng, Xp=Xp, prior_sigma=prior_sigma, \
            params=gn_params, shared_param_dicts=None)
    # Initialize biases in IN and GN
    IN.init_biases(0.1)
    GN.init_biases(0.1)
    # Initialize the GIPair
    GIP = GIPair(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, g_net=GN, i_net=IN, \
            data_dim=data_dim, prior_dim=prior_dim, params=None)
    GIP.set_lam_l2w(1e-3)
    # Set initial learning rate and basic SGD hyper parameters
    learn_rate = 0.0025
    GIP.set_all_sgd_params(learn_rate=learn_rate, momentum=0.8)

    for i in range(750000):
        if (i < 100000):
            scale = float(i) / 50000.0
            if (i < 50000):
                GIP.set_all_sgd_params(learn_rate=(scale*learn_rate), momentum=0.8)
            GIP.set_lam_kld(lam_kld=scale)
        if ((i+1 % 100000) == 0):
            learn_rate = learn_rate * 0.75
            GIP.set_all_sgd_params(learn_rate=learn_rate, momentum=0.9)
        # get some data to train with
        tr_idx = npr.randint(low=0,high=tr_samples,size=(100,))
        Xd_batch = binarize_data(Xtr.take(tr_idx, axis=0))
        Xc_batch = 0.0 * Xd_batch
        Xm_batch = 0.0 * Xd_batch
        # do a minibatch update of the model, and compute some costs
        outputs = GIP.train_joint(Xd_batch, Xc_batch, Xm_batch)
        joint_cost = 1.0 * outputs[0]
        data_nll_cost = 1.0 * outputs[1]
        post_kld_cost = 1.0 * outputs[2]
        other_reg_cost = 1.0 * outputs[3]
        if ((i % 500) == 0):
            print("batch: {0:d}, joint_cost: {1:.4f}, data_nll_cost: {2:.4f}, post_kld_cost: {3:.4f}, other_reg_cost: {4:.4f}".format( \
                    i, joint_cost, data_nll_cost, post_kld_cost, other_reg_cost))
        if ((i % 2500) == 0):
            file_name = "GIP_SAMPLES_b{0:d}.png".format(i)
            Xd_samps = np.repeat(Xd_batch[0:10,:], 3, axis=0)
            sample_lists = GIP.sample_gil_from_data(Xd_samps, loop_iters=10)
            Xs = np.vstack(sample_lists["data samples"])
            utils.visualize_samples(Xs, file_name)
    print("TESTING COMPLETE!")
    return

def multitest_git_on_gip():
    """
    Do random hyperparameter optimization.
    """
    learn_rate_git = [0.002, 0.004]
    lam_cat_git = [2.0, 4.0]
    lam_pea_git = [2.0, 4.0, 8.0]
    lam_ent_git = [-0.2, 0.0, 0.2]
    lam_l2w_git = [1e-4]
    for t_num in range(100):
        # select the hyperparameters for this test uniformly at random
        hyper_params = {}
        hyper_params['out_name'] = "GOG_TEST_{0:d}.txt".format(t_num)
        hyper_params['learn_rate_git'] = rand_sample(learn_rate_git)
        hyper_params['lam_cat_git'] = rand_sample(lam_cat_git)
        hyper_params['lam_pea_git'] = rand_sample(lam_pea_git)
        hyper_params['lam_ent_git'] = rand_sample(lam_ent_git)
        hyper_params['lam_l2w_git'] = rand_sample(lam_l2w_git)
        # run the test and record results
        test_git_on_gip(hyper_params=hyper_params, rng_seed=t_num)
    return

def multitest_gi_trip():
    """
    Do random hyperparameter optimization.
    """
    num_updates = 600000
    learn_rate = [ 0.004 ]
    lam_cat = [ 4.0 ]
    lam_pea = [ 4.0 ]
    lam_ent = [ -1.0, 0.0, 1.0 ]
    sup_count = [600, 1000, 3000, 100]
    lam_l2w = [1e-4]
    t_num = 0
    for sc in sup_count:
        for le in lam_ent:
            # select the hyperparameters for this test uniformly at random
            hyper_params = {}
            hyper_params['out_name'] = "GIT_TEST_{0:d}.txt".format(t_num)
            hyper_params['num_updates'] = num_updates
            hyper_params['learn_rate'] = rand_sample(learn_rate)
            hyper_params['lam_cat'] = rand_sample(lam_cat)
            hyper_params['lam_pea'] = rand_sample(lam_pea)
            hyper_params['lam_ent'] = le
            hyper_params['lam_l2w'] = rand_sample(lam_l2w)
            # run the test and record results
            test_gi_trip(hyper_params=hyper_params, sup_count=sc, rng_seed=t_num)
            t_num += 1
    return

def multitest_gi_stack():
    """
    Do random hyperparameter optimization.
    """
    num_updates = 600000
    learn_rate = [ 0.004 ]
    lam_cat = [ 4.0 ]
    lam_pea = [ 4.0 ]
    lam_ent = [ -1.0, 0.0, 1.0 ]
    sup_count = [600, 1000, 3000, 100]
    lam_l2w = [1e-4]
    t_num = 0
    for sc in sup_count:
        for le in lam_ent:
            # select the hyperparameters for this test uniformly at random
            hyper_params = {}
            hyper_params['out_name'] = "GIS_TEST_{0:d}.txt".format(t_num)
            hyper_params['num_updates'] = num_updates
            hyper_params['learn_rate'] = rand_sample(learn_rate)
            hyper_params['lam_cat'] = rand_sample(lam_cat)
            hyper_params['lam_pea'] = rand_sample(lam_pea)
            hyper_params['lam_ent'] = le
            hyper_params['lam_l2w'] = rand_sample(lam_l2w)
            # run the test and record results
            test_gi_stack(hyper_params=hyper_params, sup_count=sc, rng_seed=t_num)
            t_num += 1
    return


###################
# TEST DISPATCHER #
###################

if __name__=="__main__":
    #test_gc_pair()
    #test_gi_pair()
    #multitest_git_on_gip()
    multitest_gi_trip()
    #multitest_gi_stack()
