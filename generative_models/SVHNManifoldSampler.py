import time
import utils as utils
import numpy as np
import numpy.random as npr
import theano
import theano.tensor as T

from load_data import load_svhn, load_svhn_gray, load_svhn_all_gray_zca
from PeaNet import PeaNet, load_peanet_from_file
from InfNet import InfNet, load_infnet_from_file
from GenNet import GenNet, load_gennet_from_file
from PeaNetSeq import PeaNetSeq
from VCGLoop import VCGLoop
from GIPair import GIPair
from NetLayers import relu_actfun, softplus_actfun, \
                      safe_softmax, safe_log
import GenNet as GNet
import InfNet as INet
import PeaNet as PNet
from DKCode import PCA_theano

import sys, resource
resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
sys.setrecursionlimit(10**6)

# DERP
RESULT_PATH = "SMS_RESULTS/"


#####################################
# HELPER FUNCTIONS FOR DATA MASKING #
#####################################

def sample_masks(X, drop_prob=0.3):
    """
    Sample a binary mask to apply to the matrix X, with rate mask_prob.
    """
    probs = npr.rand(*X.shape)
    mask = 1.0 * (probs > drop_prob)
    return mask.astype(theano.config.floatX)

def sample_patch_masks(X, im_shape, patch_shape):
    """
    Sample a random patch mask for each image in X.
    """
    obs_count = X.shape[0]
    rs = patch_shape[0]
    cs = patch_shape[1]
    off_row = npr.randint(1,high=(im_shape[0]-rs-1), size=(obs_count,))
    off_col = npr.randint(1,high=(im_shape[1]-cs-1), size=(obs_count,))
    dummy = np.zeros(im_shape)
    mask = np.zeros(X.shape)
    for i in range(obs_count):
        dummy = (0.0 * dummy) + 1.0
        dummy[off_row[i]:(off_row[i]+rs), off_col[i]:(off_col[i]+cs)] = 0.0
        mask[i,:] = dummy.ravel()
    return mask.astype(theano.config.floatX)

def shift_and_scale_into_01(X):
    X = X - np.min(X, axis=1, keepdims=True)
    X = X / np.max(X, axis=1, keepdims=True)
    return X

def train_valid_split(X, valid_count=1000):
    """
    Split the observations in the rows of X into train/validate sets.
    """
    obs_count = X.shape[0]
    idx = np.arange(obs_count)
    npr.shuffle(idx)
    va_idx = idx[:valid_count]
    tr_idx = idx[valid_count:]
    Xtr = X.take(tr_idx, axis=0)
    Xva = X.take(va_idx, axis=0)
    return Xtr, Xva


##########################################
##########################################
## TEST SEMISUPERVISED LEARNING ON SVHN ##
##########################################
##########################################

def test_semisupervised():
    import utils as utils
    from load_data import load_udm, load_udm_ss, load_mnist
    from NetLayers import relu_actfun

    # Initialize a source of randomness
    rng = np.random.RandomState(123)

    sup_count = 1000
    va_count = 10000
    # Load some data to train/validate/test with
    tr_file = 'data/svhn_train_gray.pkl'
    te_file = 'data/svhn_test_gray.pkl'
    ex_file = 'data/svhn_extra_gray.pkl'
    data = load_svhn_gray(tr_file, te_file, ex_file=ex_file, ex_count=200000)
    X_mean = np.mean(data['Xtr'], axis=0, keepdims=True)
    X_std = np.std(data['Xtr'], axis=0, keepdims=True)
    data['Xtr'] = (data['Xtr'] - X_mean) / X_std
    data['Xte'] = (data['Xte'] - X_mean) / X_std
    data['Xex'] = (data['Xex'] - X_mean) / X_std
    idx = np.arange(data['Xtr'].shape[0])
    npr.shuffle(idx)
    Xva = data['Xte'][:,:] #[idx[0:va_count],:]
    Yva = data['Yte'][:,:].astype(np.int32) # [idx[0:va_count],:].astype(np.int32)
    Xtr_su = data['Xtr'][idx[va_count:(va_count+sup_count)], :]
    Ytr_su = data['Ytr'][idx[va_count:(va_count+sup_count)], :].astype(np.int32)
    Xtr_un = np.vstack([data['Xtr'][idx[va_count:], :], data['Xex']])
    Ytr_un = np.zeros((Xtr_un.shape[0],1)).astype(np.int32)
    print("unique(Ytr_su): {0:s}".format(str(np.unique(Ytr_su))))
    print("unique(Ytr_un): {0:s}".format(str(np.unique(Ytr_un))))
    print("Xtr_su.shape: {0:s}, Ytr_su.shape: {1:s}".format(str(Xtr_su.shape), str(Ytr_su.shape)))
    print("Xva.shape: {0:s}, Yva.shape: {1:s}".format(str(Xva.shape), str(Yva.shape)))

    un_samples = Xtr_un.shape[0]
    su_samples = Xtr_su.shape[0]
    va_samples = Xva.shape[0]

    # set up some symbolic variables for input to the GITrip
    Xd = T.matrix('Xd_base')
    Yd = T.icol('Yd_base')
    # set some "shape" parameters for the networks
    data_dim = Xtr_un.shape[1]
    label_dim = 10
    batch_size = 200 # we'll take 2x this per batch, for sup and unsup

    # choose some parameters for the categorical inferencer
    pn_params = {}
    pc0 = [data_dim, 800, 800, label_dim]
    pn_params['proto_configs'] = [pc0]
    # Set up some spawn networks
    sc0 = {'proto_key': 0, 'input_noise': 0.1, 'bias_noise': 0.2, 'do_dropout': True}
    pn_params['spawn_configs'] = [ sc0 ]
    pn_params['spawn_weights'] = [ 1.0 ]
    # Set remaining params
    pn_params['activation'] = relu_actfun
    pn_params['init_scale'] = 0.5
    pn_params['lam_l2a'] = 1e-3
    pn_params['vis_drop'] = 0.2
    pn_params['hid_drop'] = 0.5

    # Initialize the base network for this PNSeq
    PN = PeaNet(rng=rng, Xd=Xd, params=pn_params)
    PN.init_biases(0.1)

    # Initialize the PeaNetSeq
    PNS = PeaNetSeq(rng=rng, pea_net=PN, seq_len=2, seq_Xd=None, params=None)

    # set weighting parameters for the various costs...
    PNS.set_lam_class(1.0)
    PNS.set_lam_pea_su(0.0)
    PNS.set_lam_pea_un(1.0)
    PNS.set_lam_ent(0.0)
    PNS.set_lam_l2w(1e-5)

    out_file = open("SVHN_SS_TEST.txt", 'wb')
    cost_1 = [0. for i in range(10)]
    learn_rate = 0.02
    PNS.set_pn_sgd_params(lr_pn=learn_rate, mom_1=0.9, mom_2=0.999)
    for i in range(300000):
        # get some data to train with
        su_idx = npr.randint(low=0,high=su_samples,size=(batch_size,))
        Xd_su = Xtr_su.take(su_idx, axis=0)
        Yd_su = Ytr_su.take(su_idx, axis=0)
        un_idx = npr.randint(low=0,high=un_samples,size=(batch_size,))
        Xd_un = Xtr_un.take(un_idx, axis=0)
        Yd_un = Ytr_un.take(un_idx, axis=0)
        Xd_batch = np.vstack((Xd_su, Xd_un))
        Yd_batch = np.vstack((Yd_su, Yd_un))
        # set learning parameters for this update
        PNS.set_pn_sgd_params(lr_pn=learn_rate, mom_1=0.9, mom_2=0.999)
        # do a minibatch update of all PeaNet parameters
        outputs = PNS.train_joint(Xd_batch, Xd_batch, Yd_batch, Yd_batch)
        cost_1 = [(cost_1[k] + 1.*outputs[k]) for k in range(len(outputs))]
        if ((i % 1000) == 0):
            cost_1 = [(v / 1000.) for v in cost_1]
            o_str = "batch: {0:d}, joint: {1:.4f}, class: {2:.4f}, pea: {3:.4f}, ent: {4:.4f}, other_reg: {5:.4f}".format( \
                    i, cost_1[0], cost_1[1], cost_1[2], cost_1[3], cost_1[4])
            print(o_str)
            out_file.write(o_str+"\n")
            out_file.flush()
            cost_1 = [0. for v in cost_1]
            # check classification error on training and validation set
            train_err = PNS.classification_error(Xtr_su, Ytr_su)
            va_err = PNS.classification_error(Xva, Yva)
            o_str = "    tr_err: {0:.4f}, va_err: {1:.4f}".format(train_err, va_err)
            print(o_str)
            out_file.write(o_str+"\n")
            out_file.flush()
        if ((i % 1000) == 0):
            # draw the main PeaNet's first-layer filters/weights
            file_name = "SVHN_SS_PN_WEIGHTS.png".format(i)
            utils.visualize_net_layer(PNS.PN.proto_nets[0][0], file_name)
    print("TESTING COMPLETE!")

######################
######################
## PRETRAIN THE GIP ##
######################
######################

def pretrain_gip_gray():
    # Initialize a source of randomness
    rng = np.random.RandomState(1234)

    # Load some data to train/validate/test with
    tr_file = 'data/svhn_train_gray.pkl'
    te_file = 'data/svhn_test_gray.pkl'
    ex_file = 'data/svhn_extra_gray.pkl'
    data = load_svhn_gray(tr_file, te_file, ex_file=ex_file, ex_count=200000)
    #all_file = 'data/svhn_all_gray_zca.pkl'
    #data = load_svhn_all_gray_zca(all_file)
    Xtr = np.vstack([data['Xtr'], data['Xex']])
    Xtr = Xtr - np.mean(Xtr, axis=1, keepdims=True)
    Xtr = Xtr / np.std(Xtr, axis=1, keepdims=True)
    Xtr = shift_and_scale_into_01(Xtr)
    Xtr, Xva = train_valid_split(Xtr, valid_count=5000)
    tr_samples = Xtr.shape[0]
    batch_size = 100
    batch_reps = 5

    # Construct a GenNet and an InfNet, then test constructor for GIPair.
    # Do basic testing, to make sure classes aren't completely broken.
    Xp = T.matrix('Xp_base')
    Xd = T.matrix('Xd_base')
    Xc = T.matrix('Xc_base')
    Xm = T.matrix('Xm_base')
    data_dim = Xtr.shape[1]
    prior_dim = 100
    prior_sigma = 1.0
    # Choose some parameters for the generator network
    gn_params = {}
    gn_config = [prior_dim, 1200, 1200, data_dim]
    gn_params['mlp_config'] = gn_config
    gn_params['activation'] = relu_actfun
    gn_params['out_type'] = 'gaussian'
    gn_params['mean_transform'] = 'sigmoid'
    gn_params['init_scale'] = 1.0
    gn_params['lam_l2a'] = 1e-2
    gn_params['vis_drop'] = 0.0
    gn_params['hid_drop'] = 0.0
    gn_params['bias_noise'] = 0.1
    # choose some parameters for the continuous inferencer
    in_params = {}
    shared_config = [data_dim, 1200, 1200]
    top_config = [shared_config[-1], 400, prior_dim]
    in_params['shared_config'] = shared_config
    in_params['mu_config'] = top_config
    in_params['sigma_config'] = top_config
    in_params['activation'] = relu_actfun
    in_params['init_scale'] = 1.0
    in_params['lam_l2a'] = 1e-2
    in_params['vis_drop'] = 0.2
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
    GIP.set_lam_l2w(1e-4)

    ####################
    # RICA PRETRAINING #
    ####################
    IN.W_rica.set_value(0.05 * IN.W_rica.get_value(borrow=False))
    GN.W_rica.set_value(0.05 * GN.W_rica.get_value(borrow=False))
    for i in range(5000):
        scale = min(1.0, (float(i+1) / 5000.0))
        l_rate = 0.0001 * scale
        lam_l1 = 0.05
        tr_idx = npr.randint(low=0,high=tr_samples,size=(1000,))
        Xd_batch = Xtr.take(tr_idx, axis=0)
        inr_out = IN.train_rica(Xd_batch, l_rate, lam_l1)
        gnr_out = GN.train_rica(Xd_batch, l_rate, lam_l1)
        inr_out = [v for v in gnr_out]
        if ((i % 1000) == 0):
            print("rica batch {0:d}: in_recon={1:.4f}, in_spars={2:.4f}, gn_recon={3:.4f}, gn_spars={4:.4f}".format( \
                    i, 1.*inr_out[1], 1.*inr_out[2], 1.*gnr_out[1], 1.*gnr_out[2]))
                        # draw inference net first layer weights
    file_name = RESULT_PATH+"pt_gray_rica_inf_weights.png".format(i)
    utils.visualize_samples(IN.W_rica.get_value(borrow=False).T, file_name, num_rows=20)
    # draw generator net final layer weights
    file_name = RESULT_PATH+"pt_gray_rica_gen_weights.png".format(i)
    if ('gaussian' in gn_params['out_type']):
        lay_num = -2
    else:
        lay_num = -1
    utils.visualize_samples(GN.W_rica.get_value(borrow=False), file_name, num_rows=20)
    ####################
    ####################

    out_file = open(RESULT_PATH+"pt_gray_gip_results.txt", 'wb')
    # Set initial learning rate and basic SGD hyper parameters
    cost_1 = [0. for i in range(10)]
    learn_rate = 0.0002
    for i in range(1000000):
        scale = min(1.0, float(i) / 10000.0)
        # do a minibatch update of the model, and compute some costs
        tr_idx = npr.randint(low=0,high=tr_samples,size=(batch_size,))
        Xd_batch = Xtr.take(tr_idx, axis=0)
        Xd_batch = np.repeat(Xd_batch, batch_reps, axis=0)
        Xc_batch = 0.0 * Xd_batch
        Xm_batch = 0.0 * Xd_batch
        # do a minibatch update of the model, and compute some costs
        GIP.set_all_sgd_params(lr_gn=(scale*learn_rate), \
                lr_in=(scale*learn_rate), mom_1=0.9, mom_2=0.999)
        GIP.set_lam_nll(1.0)
        GIP.set_lam_kld(1.0)
        outputs = GIP.train_joint(Xd_batch, Xc_batch, Xm_batch)
        cost_1 = [(cost_1[k] + 1.*outputs[k]) for k in range(len(outputs))]
        if ((i % 1000) == 0):
            cost_1 = [(v / 1000.) for v in cost_1]
            o_str = "batch: {0:d}, joint_cost: {1:.4f}, data_nll_cost: {2:.4f}, post_kld_cost: {3:.4f}, other_reg_cost: {4:.4f}".format( \
                    i, cost_1[0], cost_1[1], cost_1[2], cost_1[3])
            print(o_str)
            out_file.write(o_str+"\n")
            out_file.flush()
            cost_1 = [0. for v in cost_1]
        if ((i % 5000) == 0):
            cost_2 = GIP.compute_costs(Xva, 0.*Xva, 0.*Xva)
            o_str = "--val: {0:d}, joint_cost: {1:.4f}, data_nll_cost: {2:.4f}, post_kld_cost: {3:.4f}, other_reg_cost: {4:.4f}".format( \
                    i, 1.*cost_2[0], 1.*cost_2[1], 1.*cost_2[2], 1.*cost_2[3])
            print(o_str)
            out_file.write(o_str+"\n")
            out_file.flush()
        if ((i % 10000) == 0):
            tr_idx = npr.randint(low=0,high=tr_samples,size=(100,))
            Xd_batch = Xtr.take(tr_idx, axis=0)
            file_name = RESULT_PATH+"pt_gray_gip_chain_samples_b{0:d}.png".format(i)
            Xd_samps = np.repeat(Xd_batch[0:10,:], 3, axis=0)
            sample_lists = GIP.sample_gil_from_data(Xd_samps, loop_iters=20)
            Xs = np.vstack(sample_lists["data samples"])
            utils.visualize_samples(Xs, file_name, num_rows=20)
            # draw samples freely from the generative model's prior
            file_name = RESULT_PATH+"pt_gray_gip_prior_samples_b{0:d}.png".format(i)
            Xs = GIP.sample_from_gn(20*20)
            utils.visualize_samples(Xs, file_name, num_rows=20)
            # draw inference net first layer weights
            file_name = RESULT_PATH+"pt_gray_gip_inf_weights_b{0:d}.png".format(i)
            utils.visualize_net_layer(GIP.IN.shared_layers[0], file_name)
            # draw generator net final layer weights
            file_name = RESULT_PATH+"pt_gray_gip_gen_weights_b{0:d}.png".format(i)
            if (gn_params['out_type'] == 'gaussian'):
                lay_num = -2
            else:
                lay_num = -1
            utils.visualize_net_layer(GIP.GN.mlp_layers[lay_num], file_name, \
                    colorImg=False, use_transpose=True)
            IN.save_to_file(f_name=RESULT_PATH+"pt_gray_params_IN.pkl")
            GN.save_to_file(f_name=RESULT_PATH+"pt_gray_params_GN.pkl")
    IN.save_to_file(f_name=RESULT_PATH+"pt_gray_params_IN.pkl")
    GN.save_to_file(f_name=RESULT_PATH+"pt_gray_params_GN.pkl")
    return

#####################################################
# Train a VCGLoop starting from a pretrained GIPair #
#####################################################

def train_vcgl_from_pretrained_gip():
    # Simple test code, to check that everything is basically functional.
    print("TESTING...")

    # Initialize a source of randomness
    rng = np.random.RandomState(1234)

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
    prior_dim = 64
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

    ###############################
    # Setup discriminator network #
    ###############################
    # Set some reasonable mlp parameters
    dn_params = {}
    # Set up some proto-networks
    pc0 = [data_dim, (300, 4), (300, 4), 10]
    dn_params['proto_configs'] = [pc0]
    # Set up some spawn networks
    sc0 = {'proto_key': 0, 'input_noise': 0.1, 'bias_noise': 0.1, 'do_dropout': True}
    #sc1 = {'proto_key': 0, 'input_noise': 0.1, 'bias_noise': 0.1, 'do_dropout': True}
    dn_params['spawn_configs'] = [sc0]
    dn_params['spawn_weights'] = [1.0]
    # Set remaining params
    dn_params['init_scale'] = 0.5
    dn_params['lam_l2a'] = 1e-2
    dn_params['vis_drop'] = 0.2
    dn_params['hid_drop'] = 0.5
    # Initialize a network object to use as the discriminator
    DN = PeaNet(rng=rng, Xd=Xd, params=dn_params)
    DN.init_biases(0.0)

    #######################################################
    # Load inferencer and generator from saved parameters #
    #######################################################
    gn_fname = RESULT_PATH+"pt60k_params_GN.pkl"
    in_fname = RESULT_PATH+"pt60k_params_IN.pkl"
    IN = INet.load_infnet_from_file(f_name=in_fname, rng=rng, Xd=Xd, Xc=Xc, Xm=Xm)
    GN = GNet.load_gennet_from_file(f_name=gn_fname, rng=rng, Xp=Xp)

    ###############################
    # Initialize the main VCGLoop #
    ###############################
    vcgl_params = {}
    vcgl_params['lam_l2d'] = 1e-2
    VCGL = VCGLoop(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, Xt=Xt, i_net=IN, \
                 g_net=GN, d_net=DN, chain_len=6, data_dim=data_dim, \
                 prior_dim=prior_dim, params=vcgl_params)
    VCGL.set_lam_l2w(1e-4)

    out_file = open(RESULT_PATH+"pt60k_vcgl_results.txt", 'wb')
    ####################################################
    # Train the VCGLoop by unrolling and applying BPTT #
    ####################################################
    learn_rate = 0.0005
    cost_1 = [0. for i in range(10)]
    cost_2 = [0. for i in range(10)]
    for i in range(1000000):
        scale = float(min((i+1), 25000)) / 25000.0
        if ((i+1 % 100000) == 0):
            learn_rate = learn_rate * 0.75
        if True:
            ########################################
            # TRAIN THE CHAIN IN FREE-RUNNING MODE #
            ########################################
            VCGL.set_all_sgd_params(learn_rate=(scale*learn_rate), \
                    mom_1=0.9, mom_2=0.999)
            VCGL.set_dn_sgd_params(learn_rate=0.5*scale*learn_rate)
            VCGL.set_disc_weights(dweight_gn=5.0, dweight_dn=5.0)
            VCGL.set_lam_chain_nll(1.0)
            VCGL.set_lam_chain_kld(3.0)
            VCGL.set_lam_chain_vel(0.0)
            VCGL.set_lam_mask_nll(0.0)
            VCGL.set_lam_mask_kld(0.0)
            # get some data to train with
            tr_idx = npr.randint(low=0,high=tr_samples,size=(batch_size,))
            Xd_batch = Xtr.take(tr_idx, axis=0)
            Xc_batch = 0.0 * Xd_batch
            Xm_batch = 0.0 * Xd_batch
            tr_idx = npr.randint(low=0,high=tr_samples,size=(batch_size,))
            Xt_batch = Xtr.take(tr_idx, axis=0)
            # do 4 repetitions of the batch
            Xd_batch = np.repeat(Xd_batch, 5, axis=0)
            Xc_batch = np.repeat(Xc_batch, 5, axis=0)
            Xm_batch = np.repeat(Xm_batch, 5, axis=0)
            Xt_batch = np.repeat(Xt_batch, 5, axis=0)
            # do a minibatch update of the model, and compute some costs
            outputs = VCGL.train_joint(Xd_batch, Xc_batch, Xm_batch, Xt_batch)
            cost_1 = [(cost_1[k] + 1.*outputs[k]) for k in range(len(outputs))]
            #cost_2 = [0. for v in cost_1]
        if ((i % 2) == 0):
            #########################################
            # TRAIN THE CHAIN UNDER PARTIAL CONTROL #
            #########################################
            VCGL.set_all_sgd_params(learn_rate=(scale*learn_rate), \
                    mom_1=0.9, mom_2=0.999)
            VCGL.set_dn_sgd_params(learn_rate=0.5*scale*learn_rate)
            VCGL.set_disc_weights(dweight_gn=5.0, dweight_dn=5.0)
            VCGL.set_lam_chain_nll(0.0)
            VCGL.set_lam_chain_kld(0.0)
            VCGL.set_lam_chain_vel(0.0)
            VCGL.set_lam_mask_nll(1.0)
            VCGL.set_lam_mask_kld(0.2)
            # get some data to train with
            tr_idx = npr.randint(low=0,high=tr_samples,size=(batch_size,))
            Xd_batch = Xc_mean
            Xc_batch = Xtr.take(tr_idx, axis=0)
            Xm_rand = sample_masks(Xc_batch, drop_prob=0.3)
            Xm_patch = sample_patch_masks(Xc_batch, (32,32), (14,14))
            Xm_batch = Xm_rand * Xm_patch
            tr_idx = npr.randint(low=0,high=tr_samples,size=(batch_size,))
            Xt_batch = Xtr.take(tr_idx, axis=0)
            # do 4 repetitions of the batch
            Xd_batch = np.repeat(Xd_batch, 5, axis=0)
            Xc_batch = np.repeat(Xc_batch, 5, axis=0)
            Xm_batch = np.repeat(Xm_batch, 5, axis=0)
            Xt_batch = np.repeat(Xt_batch, 5, axis=0)
            # do a minibatch update of the model, and compute some costs
            outputs = VCGL.train_joint(Xd_batch, Xc_batch, Xm_batch, Xt_batch)
            cost_2 = [(cost_2[k] + 1.*outputs[k]) for k in range(len(outputs))]
            # cost_1 = [0. for k in range(10)]
        if ((i % 1000) == 0):
            cost_1 = [(v / 1000.0) for v in cost_1]
            cost_2 = [(v / 500.0) for v in cost_2]
            o_str_1 = "batch: {0:d}, joint_cost: {1:.4f}, chain_nll_cost: {2:.4f}, chain_kld_cost: {3:.4f}, disc_cost_gn: {4:.4f}, disc_cost_dn: {5:.4f}".format( \
                    i, cost_1[0], cost_1[1], cost_1[2], cost_1[6], cost_1[7])
            o_str_2 = "------ {0:d}, joint_cost: {1:.4f}, mask_nll_cost: {2:.4f}, mask_kld_cost: {3:.4f}, disc_cost_gn: {4:.4f}, disc_cost_dn: {5:.4f}".format( \
                    i, cost_2[0], cost_2[4], cost_2[5], cost_2[6], cost_2[7])
            print(o_str_1)
            print(o_str_2)
            out_file.write(o_str_1+"\n")
            out_file.write(o_str_2+"\n")
            out_file.flush()
            cost_1 = [0. for v in cost_1]
            cost_2 = [0. for v in cost_2]
        if ((i % 5000) == 0):
            tr_idx = npr.randint(low=0,high=Xtr.shape[0],size=(5,))
            va_idx = npr.randint(low=0,high=Xva.shape[0],size=(5,))
            Xd_batch = np.vstack([Xtr.take(tr_idx, axis=0), Xva.take(va_idx, axis=0)])
            # draw some chains of samples from the VAE loop
            file_name = RESULT_PATH+"pt60k_vcgl_chain_samples_b{0:d}.png".format(i)
            Xd_samps = np.repeat(Xd_batch, 3, axis=0)
            sample_lists = VCGL.GIP.sample_gil_from_data(Xd_samps, loop_iters=20)
            Xs = np.vstack(sample_lists["data samples"])
            utils.visualize_samples(Xs, file_name, num_rows=20)
            # draw some masked chains of samples from the VAE loop
            file_name = RESULT_PATH+"pt60k_vcgl_mask_samples_b{0:d}.png".format(i)
            Xd_samps = np.repeat(Xc_mean[0:Xd_batch.shape[0],:], 3, axis=0)
            Xc_samps = np.repeat(Xd_batch, 3, axis=0)
            Xm_rand = sample_masks(Xc_samps, drop_prob=0.3)
            Xm_patch = sample_patch_masks(Xc_samps, (32,32), (14,14))
            Xm_samps = Xm_rand * Xm_patch
            sample_lists = VCGL.GIP.sample_gil_from_data(Xd_samps, \
                    X_c=Xc_samps, X_m=Xm_samps, loop_iters=20)
            Xs = np.vstack(sample_lists["data samples"])
            utils.visualize_samples(Xs, file_name, num_rows=20)
            # draw some samples independently from the GenNet's prior
            file_name = RESULT_PATH+"pt60k_vcgl_prior_samples_b{0:d}.png".format(i)
            Xs = VCGL.sample_from_prior(20*20)
            utils.visualize_samples(Xs, file_name, num_rows=20)
            # draw discriminator network's weights
            file_name = RESULT_PATH+"pt60k_vcgl_dis_weights_b{0:d}.png".format(i)
            utils.visualize_net_layer(VCGL.DN.proto_nets[0][0], file_name)
            # draw inference net first layer weights
            file_name = RESULT_PATH+"pt60k_vcgl_inf_weights_b{0:d}.png".format(i)
            utils.visualize_net_layer(VCGL.IN.shared_layers[0], file_name)
            # draw generator net final layer weights
            file_name = RESULT_PATH+"pt60k_vcgl_gen_weights_b{0:d}.png".format(i)
            if GN.out_type == 'sigmoid':
                utils.visualize_net_layer(VCGL.GN.mlp_layers[-1], file_name, use_transpose=True)
            else:
                utils.visualize_net_layer(VCGL.GN.mlp_layers[-2], file_name, use_transpose=True)
        # DUMP PARAMETERS FROM TIME-TO-TIME
        if (i % 10000 == 0):
            DN.save_to_file(f_name=RESULT_PATH+"pt60k_vcgl_params_DN.pkl")
            IN.save_to_file(f_name=RESULT_PATH+"pt60k_vcgl_params_IN.pkl")
            GN.save_to_file(f_name=RESULT_PATH+"pt60k_vcgl_params_GN.pkl")
    return

if __name__=="__main__":
	pretrain_gip_gray()
	#train_vcgl_from_pretrained_gip_gray()
    #test_semisupervised()