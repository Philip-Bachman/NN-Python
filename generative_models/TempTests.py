import time
import numpy as np
import numpy.random as npr
import theano
import theano.tensor as T
import utils as utils
from load_data import load_udm, load_udm_ss, load_mnist
from PeaNet import PeaNet, load_peanet_from_file
from InfNet import InfNet, load_infnet_from_file
from GenNet import GenNet, load_gennet_from_file
from GIPair import GIPair
from GIStack import GIStack
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
    # choose some parameters for the continuous inferencer
    in_params = {}
    shared_config = [data_dim, 600, 600]
    top_config = [shared_config[-1], prior_dim]
    in_params['shared_config'] = shared_config
    in_params['mu_config'] = top_config
    in_params['sigma_config'] = top_config
    in_params['activation'] = softplus_actfun
    in_params['init_scale'] = 2.0
    in_params['lam_l2a'] = 1e-3
    in_params['vis_drop'] = 0.0
    in_params['hid_drop'] = 0.0
    in_params['bias_noise'] = 0.1
    in_params['input_noise'] = 0.1
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
            params=None, shared_param_dicts=None)
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
        if (i == 26000):
            print("***SAVING AND RELOADING MODEL FROM DISK***")
            # Make names for the pickle files
            gn_fname = 'TEST_GN_PKL.pkl'
            in_fname = 'TEST_IN_PKL.pkl'
            pn_fname = 'TEST_PN_PKL.pkl'
            # Dump parameters to pickle files
            GN.save_to_file(f_name=gn_fname)
            IN.save_to_file(f_name=in_fname)
            PN.save_to_file(f_name=pn_fname)
            # Totally wipe out the models
            GN = None
            IN = None
            PN = None
            GIS = None
            # Load parameters back in from pickle files
            GN = load_gennet_from_file(f_name=gn_fname, rng=rng, Xp=Xp)
            IN = load_infnet_from_file(f_name=in_fname, rng=rng, Xd=Xd, Xc=Xc, Xm=Xm)
            PN = load_peanet_from_file(f_name=pn_fname, rng=rng, Xd=Xd)
            # Initialize the GIStack
            GIS = GIStack(rng=rng, \
                    Xd=Xd, Yd=Yd, Xc=Xc, Xm=Xm, \
                    g_net=GN, i_net=IN, p_net=PN, \
                    data_dim=data_dim, prior_dim=prior_dim, \
                    label_dim=label_dim, batch_size=batch_size, \
                    params=None, shared_param_dicts=None)
        if (i < 10000):
            # start with some updates only for the VAE (InfNet and GenNet)
            scale = float(min(i+1, 10000)) / 10000.0
            lam_cat = 0.0
            lam_pea = 0.0
            lam_ent = 0.0
            learn_rate_pn = 0.0
        else:
            # move on to updates that include loss from the PeaNet
            scale = 1.0
            lam_cat = hyper_params['lam_cat']
            lam_pea = hyper_params['lam_pea']
            lam_ent = 0.0
            learn_rate_pn = learn_rate
        if ((i+1 % 100000) == 0):
            learn_rate = learn_rate * 0.7
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
            GIS.set_lam_kld(0.01 + (0.99*scale))
            GIS.set_lam_cat(0.0)
            GIS.set_lam_pea(lam_pea)
            GIS.set_lam_ent(lam_ent)
            outputs = GIS.train_joint(Xd_un, Xc_un, Xm_un, Yd_un)
            joint_cost = 1.0 * outputs[0]
            data_nll_cost = 1.0 * outputs[1]
            post_kld_cost = 1.0 * outputs[2]
            post_cat_cost = 1.0 * outputs[3]
            post_pea_cost = 1.0 * outputs[4]
            post_ent_cost = 1.0 * outputs[5]
            other_reg_cost = 1.0 * outputs[6]
        # do another minibatch update incorporating label information
        if (i >= 10000):
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
            GIS.set_lam_cat(lam_cat)
            GIS.set_lam_pea(lam_pea)
            GIS.set_lam_ent(0.0)
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
            # draw inference net first layer weights
            file_name = "GIS_IN_WEIGHTS_b{0:d}.png".format(i)
            utils.visualize_net_layer(IN.shared_layers[0], file_name)
            # draw generator net final layer weights
            file_name = "GIS_GN_WEIGHTS_b{0:d}.png".format(i)
            utils.visualize_net_layer(GN.mlp_layers[-1], file_name, use_transpose=True)
    print("TESTING COMPLETE!")
    out_file.close()
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
    prior_dim = 50
    prior_sigma = 1.0
    # Choose some parameters for the generator network
    gn_params = {}
    gn_config = [prior_dim, 800, 800, data_dim]
    gn_params['mlp_config'] = gn_config
    gn_params['activation'] = relu_actfun
    gn_params['out_type'] = 'gaussian'
    gn_params['init_scale'] = 2.0
    gn_params['lam_l2a'] = 1e-2
    gn_params['vis_drop'] = 0.0
    gn_params['hid_drop'] = 0.0
    gn_params['bias_noise'] = 0.1
    # choose some parameters for the continuous inferencer
    in_params = {}
    shared_config = [data_dim, (200, 4), (200, 4)]
    top_config = [shared_config[-1], (100, 4), prior_dim]
    in_params['shared_config'] = shared_config
    in_params['mu_config'] = top_config
    in_params['sigma_config'] = top_config
    in_params['activation'] = relu_actfun
    in_params['init_scale'] = 2.0
    in_params['lam_l2a'] = 1e-2
    in_params['vis_drop'] = 0.0
    in_params['hid_drop'] = 0.0
    in_params['bias_noise'] = 0.1
    in_params['input_noise'] = 0.1
    # Initialize the base networks for this GIPair
    IN = InfNet(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, prior_sigma=prior_sigma, \
            params=in_params, shared_param_dicts=None)
    GN = GenNet(rng=rng, Xp=Xp, prior_sigma=prior_sigma, \
            params=gn_params, shared_param_dicts=None)
    # Initialize biases in IN and GN
    IN.init_biases(0.0)
    GN.init_biases(0.2)
    # Initialize the GIPair
    GIP = GIPair(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, g_net=GN, i_net=IN, \
            data_dim=data_dim, prior_dim=prior_dim, params=None)
    GIP.set_lam_l2w(1e-4)

    # Set initial learning rate and basic SGD hyper parameters
    learn_rate = 0.001
    for i in range(750000):
        scale = min(1.0, float(i) / 50000.0)
        if ((i+1 % 100000) == 0):
            learn_rate = learn_rate * 0.75
        GIP.set_all_sgd_params(learn_rate=(scale*learn_rate), momentum=0.98)
        GIP.set_lam_nll(1.0)
        GIP.set_lam_kld(0.1 + (0.9 * scale))
        # get some data to train with
        tr_idx = npr.randint(low=0,high=tr_samples,size=(100,))
        Xd_batch = Xtr.take(tr_idx, axis=0) #binarize_data(Xtr.take(tr_idx, axis=0))
        Xc_batch = 0.0 * Xd_batch
        Xm_batch = 0.0 * Xd_batch
        # do a minibatch update of the model, and compute some costs
        outputs = GIP.train_joint(Xd_batch, Xc_batch, Xm_batch)
        joint_cost = 1.0 * outputs[0]
        data_nll_cost = 1.0 * outputs[1]
        post_kld_cost = 1.0 * outputs[2]
        other_reg_cost = 1.0 * outputs[3]
        if ((i % 1000) == 0):
            print("batch: {0:d}, joint_cost: {1:.4f}, data_nll_cost: {2:.4f}, post_kld_cost: {3:.4f}, other_reg_cost: {4:.4f}".format( \
                    i, joint_cost, data_nll_cost, post_kld_cost, other_reg_cost))
        if ((i % 5000) == 0):
            file_name = "GIP_CHAIN_SAMPLES_b{0:d}.png".format(i)
            Xd_samps = np.repeat(Xd_batch[0:10,:], 3, axis=0)
            sample_lists = GIP.sample_gil_from_data(Xd_samps, loop_iters=20)
            Xs = np.vstack(sample_lists["data samples"])
            utils.visualize_samples(Xs, file_name, num_rows=20)
            # draw samples freely from the generative model's prior
            file_name = "GIP_PRIOR_SAMPLES_b{0:d}.png".format(i)
            Xs = GIP.sample_from_gn(20*20)
            utils.visualize_samples(Xs, file_name, num_rows=20)
            # draw inference net first layer weights
            file_name = "GIP_INF_WEIGHTS_b{0:d}.png".format(i)
            utils.visualize_net_layer(GIP.IN.shared_layers[0], file_name)
            # draw generator net final layer weights
            file_name = "GIP_GEN_WEIGHTS_b{0:d}.png".format(i)
            utils.visualize_net_layer(GIP.GN.mlp_layers[-1], file_name, use_transpose=True)
    print("TESTING COMPLETE!")
    return

def multitest_gi_stack():
    """
    Do random hyperparameter optimization.
    """
    sup_count = [600, 1000, 3000, 100]
    num_updates = 400000
    learn_rate = [ 0.004 ]
    lam_cat = [ 4.0 ]
    lam_pea = [ 4.0 ]
    lam_ent = [ 0.0, 1.0 ]
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
    test_gi_pair()
    #multitest_gi_stack()
