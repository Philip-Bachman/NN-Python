import time
import utils as utils
import numpy as np
import numpy.random as npr
import theano
import theano.tensor as T

from load_data import load_tfd
from PeaNet import PeaNet, load_peanet_from_file
from InfNet import InfNet, load_infnet_from_file
from VCGLoop import VCGLoop
from OneStageModel import OneStageModel
from NetLayers import relu_actfun, softplus_actfun, \
                      safe_softmax, row_shuffle

import sys, resource
resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
sys.setrecursionlimit(10**6)

# DERP
#RESULT_PATH = "TFD_WALKOUT_TEST_KLD/"
RESULT_PATH = "TFD_WALKOUT_TEST_VAE/"
#RESULT_PATH = "TFD_WALKOUT_TEST_MAX_KLD/"
PRIOR_DIM = 50
LOGVAR_BOUND = 6.0

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

def posterior_klds(IN, Xtr, batch_size, batch_count):
    """
    Get posterior KLd cost for some inputs from Xtr.
    """
    post_klds = []
    for i in range(batch_count):
        batch_idx = npr.randint(low=0, high=Xtr.shape[0], size=(batch_size,))
        X = Xtr.take(batch_idx, axis=0)
        post_klds.extend([k for k in IN.kld_func(X)])
    return post_klds

def collect_obs_costs(batch_costs, batch_reps):
    """
    Collect per-observation costs from a cost vector containing the cost for
    multiple repetitions of each observation.
    """
    obs_count = int(batch_costs.shape[0] / batch_reps)
    obs_costs = np.zeros((obs_count,))
    obs_idx = -1
    for i in range(batch_costs.shape[0]):
        if ((i % batch_reps) == 0):
            obs_idx = obs_idx + 1
        obs_costs[obs_idx] = obs_costs[obs_idx] + batch_costs[i]
    obs_costs = obs_costs / batch_reps
    return obs_costs

###########################################
###########################################
## VAE PRETRAINING FOR THE OneStageModel ##
###########################################
###########################################

def pretrain_osm(lam_kld=0.0):
    # Initialize a source of randomness
    rng = np.random.RandomState(1234)

    # Load some data to train/validate/test with
    data_file = 'data/tfd_data_48x48.pkl'
    dataset = load_tfd(tfd_pkl_name=data_file, which_set='unlabeled', fold='all')
    Xtr_unlabeled = dataset[0]
    dataset = load_tfd(tfd_pkl_name=data_file, which_set='train', fold='all')
    Xtr_train = dataset[0]
    Xtr = np.vstack([Xtr_unlabeled, Xtr_train])
    dataset = load_tfd(tfd_pkl_name=data_file, which_set='valid', fold='all')
    Xva = dataset[0]
    tr_samples = Xtr.shape[0]
    va_samples = Xva.shape[0]
    batch_size = 400
    batch_reps = 6
    carry_frac = 0.25
    carry_size = int(batch_size * carry_frac)
    reset_prob = 0.04

    # setup some symbolic variables and stuff
    Xd = T.matrix('Xd_base')
    Xc = T.matrix('Xc_base')
    Xm = T.matrix('Xm_base')
    data_dim = Xtr.shape[1]
    Xtr_mean = np.mean(Xtr, axis=0)

    ##########################
    # NETWORK CONFIGURATIONS #
    ##########################
    gn_params = {}
    shared_config = [PRIOR_DIM, 1500, 1500]
    top_config = [shared_config[-1], data_dim]
    gn_params['shared_config'] = shared_config
    gn_params['mu_config'] = top_config
    gn_params['sigma_config'] = top_config
    gn_params['activation'] = relu_actfun
    gn_params['init_scale'] = 1.4
    gn_params['lam_l2a'] = 0.0
    gn_params['vis_drop'] = 0.0
    gn_params['hid_drop'] = 0.0
    gn_params['bias_noise'] = 0.0
    gn_params['input_noise'] = 0.0
    # choose some parameters for the continuous inferencer
    in_params = {}
    shared_config = [data_dim, 1500, 1500]
    top_config = [shared_config[-1], PRIOR_DIM]
    in_params['shared_config'] = shared_config
    in_params['mu_config'] = top_config
    in_params['sigma_config'] = top_config
    in_params['activation'] = relu_actfun
    in_params['init_scale'] = 1.4
    in_params['lam_l2a'] = 0.0
    in_params['vis_drop'] = 0.0
    in_params['hid_drop'] = 0.0
    in_params['bias_noise'] = 0.0
    in_params['input_noise'] = 0.0
    # Initialize the base networks for this OneStageModel
    IN = InfNet(rng=rng, Xd=Xd, \
            params=in_params, shared_param_dicts=None)
    GN = InfNet(rng=rng, Xd=Xd, \
            params=gn_params, shared_param_dicts=None)
    # Initialize biases in IN and GN
    IN.init_biases(0.2)
    GN.init_biases(0.2)

    ######################################
    # LOAD AND RESTART FROM SAVED PARAMS #
    ######################################
    # gn_fname = RESULT_PATH+"pt_osm_params_b110000_GN.pkl"
    # in_fname = RESULT_PATH+"pt_osm_params_b110000_IN.pkl"
    # IN = load_infnet_from_file(f_name=in_fname, rng=rng, Xd=Xd, \
    #         new_params=None)
    # GN = load_infnet_from_file(f_name=gn_fname, rng=rng, Xd=Xd, \
    #         new_params=None)
    # in_params = IN.params
    # gn_params = GN.params

    #########################
    # INITIALIZE THE GIPAIR #
    #########################
    osm_params = {}
    osm_params['x_type'] = 'bernoulli'
    osm_params['xt_transform'] = 'sigmoid'
    osm_params['logvar_bound'] = LOGVAR_BOUND
    OSM = OneStageModel(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, \
            p_x_given_z=GN, q_z_given_x=IN, \
            x_dim=data_dim, z_dim=PRIOR_DIM, params=osm_params)
    OSM.set_lam_l2w(1e-5)
    safe_mean = (0.9 * Xtr_mean) + 0.05
    safe_mean_logit = np.log(safe_mean / (1.0 - safe_mean))
    OSM.set_output_bias(safe_mean_logit)
    OSM.set_input_bias(-Xtr_mean)

    ######################
    # BASIC VAE TRAINING #
    ######################
    out_file = open(RESULT_PATH+"pt_osm_results.txt", 'wb')
    # Set initial learning rate and basic SGD hyper parameters
    obs_costs = np.zeros((batch_size,))
    costs = [0. for i in range(10)]
    learn_rate = 0.002
    for i in range(200000):
        scale = min(1.0, float(i) / 5000.0)
        if ((i > 1) and ((i % 20000) == 0)):
            learn_rate = learn_rate * 0.8
        if (i < 50000):
            momentum = 0.5
        elif (i < 10000):
            momentum = 0.7
        else:
            momentum = 0.9
        if ((i == 0) or (npr.rand() < reset_prob)):
            # sample a fully random batch
            batch_idx = npr.randint(low=0,high=tr_samples,size=(batch_size,))
        else:
            # sample a partially random batch, which retains some portion of
            # the worst scoring examples from the previous batch
            fresh_idx = npr.randint(low=0,high=tr_samples,size=(batch_size-carry_size,))
            batch_idx = np.concatenate((fresh_idx.ravel(), carry_idx.ravel()))
        # do a minibatch update of the model, and compute some costs
        tr_idx = npr.randint(low=0,high=tr_samples,size=(batch_size,))
        Xd_batch = Xtr.take(tr_idx, axis=0)
        Xc_batch = 0.0 * Xd_batch
        Xm_batch = 0.0 * Xd_batch
        # do a minibatch update of the model, and compute some costs
        OSM.set_sgd_params(lr_1=(scale*learn_rate), \
                mom_1=(scale*momentum), mom_2=0.98)
        OSM.set_lam_nll(1.0)
        OSM.set_lam_kld(lam_kld_1=scale*lam_kld, lam_kld_2=0.0, lam_kld_c=50.0)
        result = OSM.train_joint(Xd_batch, Xc_batch, Xm_batch, batch_reps)
        batch_costs = result[4] + result[5]
        obs_costs = collect_obs_costs(batch_costs, batch_reps)
        carry_idx = batch_idx[np.argsort(-obs_costs)[0:carry_size]]
        costs = [(costs[j] + result[j]) for j in range(len(result))]
        if ((i % 1000) == 0):
            # record and then reset the cost trackers
            costs = [(v / 1000.0) for v in costs]
            str_1 = "-- batch {0:d} --".format(i)
            str_2 = "    joint_cost: {0:.4f}".format(costs[0])
            str_3 = "    nll_cost  : {0:.4f}".format(costs[1])
            str_4 = "    kld_cost  : {0:.4f}".format(costs[2])
            str_5 = "    reg_cost  : {0:.4f}".format(costs[3])
            costs = [0.0 for v in costs]
            # print out some diagnostic information
            joint_str = "\n".join([str_1, str_2, str_3, str_4, str_5])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
        if ((i % 2000) == 0):
            Xva = row_shuffle(Xva)
            model_samps = OSM.sample_from_prior(500)
            file_name = RESULT_PATH+"pt_osm_samples_b{0:d}_XG.png".format(i)
            utils.visualize_samples(model_samps, file_name, num_rows=20)
            file_name = RESULT_PATH+"pt_osm_inf_weights_b{0:d}.png".format(i)
            utils.visualize_samples(OSM.inf_weights.get_value(borrow=False).T, \
                    file_name, num_rows=30)
            file_name = RESULT_PATH+"pt_osm_gen_weights_b{0:d}.png".format(i)
            utils.visualize_samples(OSM.gen_weights.get_value(borrow=False), \
                    file_name, num_rows=30)
            # compute information about free-energy on validation set
            file_name = RESULT_PATH+"pt_osm_free_energy_b{0:d}.png".format(i)
            fe_terms = OSM.compute_fe_terms(Xva[0:2500], 20)
            fe_mean = np.mean(fe_terms[0]) + np.mean(fe_terms[1])
            fe_str = "    nll_bound : {0:.4f}".format(fe_mean)
            print(fe_str)
            out_file.write(fe_str+"\n")
            utils.plot_scatter(fe_terms[1], fe_terms[0], file_name, \
                    x_label='Posterior KLd', y_label='Negative Log-likelihood')
            # compute information about posterior KLds on validation set
            file_name = RESULT_PATH+"pt_osm_post_klds_b{0:d}.png".format(i)
            post_klds = OSM.compute_post_klds(Xva[0:2500])
            post_dim_klds = np.mean(post_klds, axis=0)
            utils.plot_stem(np.arange(post_dim_klds.shape[0]), post_dim_klds, \
                    file_name)
        if ((i % 5000) == 0):
            IN.save_to_file(f_name=RESULT_PATH+"pt_osm_params_b{0:d}_IN.pkl".format(i))
            GN.save_to_file(f_name=RESULT_PATH+"pt_osm_params_b{0:d}_GN.pkl".format(i))
    IN.save_to_file(f_name=RESULT_PATH+"pt_osm_params_IN.pkl")
    GN.save_to_file(f_name=RESULT_PATH+"pt_osm_params_GN.pkl")
    return

############################################################
# Train a VCGLoop starting from a pretrained OneStageModel #
############################################################

def train_walk_from_pretrained_osm(lam_kld=0.0):
    # Simple test code, to check that everything is basically functional.
    print("TESTING...")

    # Initialize a source of randomness
    rng = np.random.RandomState(1234)

    # Load some data to train/validate/test with
    data_file = 'data/tfd_data_48x48.pkl'
    dataset = load_tfd(tfd_pkl_name=data_file, which_set='unlabeled', fold='all')
    Xtr_unlabeled = dataset[0]
    dataset = load_tfd(tfd_pkl_name=data_file, which_set='train', fold='all')
    Xtr_train = dataset[0]
    Xtr = np.vstack([Xtr_unlabeled, Xtr_train])
    dataset = load_tfd(tfd_pkl_name=data_file, which_set='valid', fold='all')
    Xva = dataset[0]
    print("Xtr.shape: {0:s}, Xva.shape: {1:s}".format(str(Xtr.shape),str(Xva.shape)))

    # get and set some basic dataset information
    tr_samples = Xtr.shape[0]
    va_samples = Xva.shape[0]
    data_dim = Xtr.shape[1]
    batch_size = 400
    batch_reps = 5
    Xtr_mean = np.mean(Xtr, axis=0, keepdims=True)
    Xtr_mean = (0.0 * Xtr_mean) + np.mean(np.mean(Xtr,axis=1))
    Xc_mean = np.repeat(Xtr_mean, batch_size, axis=0)

    # Symbolic inputs
    Xd = T.matrix(name='Xd')
    Xc = T.matrix(name='Xc')
    Xm = T.matrix(name='Xm')
    Xt = T.matrix(name='Xt')

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
    dn_params['init_scale'] = 1.0
    dn_params['lam_l2a'] = 1e-2
    dn_params['vis_drop'] = 0.2
    dn_params['hid_drop'] = 0.5
    # Initialize a network object to use as the discriminator
    DN = PeaNet(rng=rng, Xd=Xd, params=dn_params)
    DN.init_biases(0.0)

    #######################################################
    # Load inferencer and generator from saved parameters #
    #######################################################
    gn_fname = RESULT_PATH+"pt_osm_params_b100000_GN.pkl"
    in_fname = RESULT_PATH+"pt_osm_params_b100000_IN.pkl"
    IN = load_infnet_from_file(f_name=in_fname, rng=rng, Xd=Xd)
    GN = load_infnet_from_file(f_name=gn_fname, rng=rng, Xd=Xd)

    ########################################################
    # Define parameters for the VCGLoop, and initialize it #
    ########################################################
    print("Building the VCGLoop...")
    vcgl_params = {}
    vcgl_params['x_type'] = 'gaussian'
    vcgl_params['xt_transform'] = 'sigmoid'
    vcgl_params['logvar_bound'] = LOGVAR_BOUND
    vcgl_params['cost_decay'] = 0.1
    vcgl_params['chain_type'] = 'walkout'
    vcgl_params['lam_l2d'] = 5e-2
    VCGL = VCGLoop(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, Xt=Xt, \
                 i_net=IN, g_net=GN, d_net=DN, chain_len=5, \
                 data_dim=data_dim, prior_dim=PRIOR_DIM, params=vcgl_params)

    out_file = open(RESULT_PATH+"pt_walk_results.txt", 'wb')
    ####################################################
    # Train the VCGLoop by unrolling and applying BPTT #
    ####################################################
    learn_rate = 0.0005
    cost_1 = [0. for i in range(10)]
    for i in range(100000):
        scale = float(min((i+1), 5000)) / 5000.0
        if ((i+1 % 25000) == 0):
            learn_rate = learn_rate * 0.8
        ########################################
        # TRAIN THE CHAIN IN FREE-RUNNING MODE #
        ########################################
        VCGL.set_all_sgd_params(learn_rate=(scale*learn_rate), \
                mom_1=0.9, mom_2=0.99)
        VCGL.set_disc_weights(dweight_gn=25.0, dweight_dn=25.0)
        VCGL.set_lam_chain_nll(1.0)
        VCGL.set_lam_chain_kld(lam_kld)
        VCGL.set_lam_mask_nll(0.0)
        VCGL.set_lam_mask_kld(0.0)
        # get some data to train with
        tr_idx = npr.randint(low=0,high=tr_samples,size=(batch_size,))
        Xd_batch = Xtr.take(tr_idx, axis=0)
        Xc_batch = 0.0 * Xd_batch
        Xm_batch = 0.0 * Xd_batch
        # examples from the target distribution, to train discriminator
        tr_idx = npr.randint(low=0,high=tr_samples,size=(2*batch_size,))
        Xt_batch = Xtr.take(tr_idx, axis=0)
        # do a minibatch update of the model, and compute some costs
        outputs = VCGL.train_joint(Xd_batch, Xc_batch, Xm_batch, Xt_batch, batch_reps)
        cost_1 = [(cost_1[k] + 1.*outputs[k]) for k in range(len(outputs))]
        if ((i % 500) == 0):
            cost_1 = [(v / 500.0) for v in cost_1]
            o_str_1 = "batch: {0:d}, joint_cost: {1:.4f}, chain_nll_cost: {2:.4f}, chain_kld_cost: {3:.4f}, disc_cost_gn: {4:.4f}, disc_cost_dn: {5:.4f}".format( \
                    i, cost_1[0], cost_1[1], cost_1[2], cost_1[5], cost_1[6])
            print(o_str_1)
            cost_1 = [0. for v in cost_1]
        if ((i % 1000) == 0):
            tr_idx = npr.randint(low=0,high=Xtr.shape[0],size=(5,))
            va_idx = npr.randint(low=0,high=Xva.shape[0],size=(5,))
            Xd_batch = np.vstack([Xtr.take(tr_idx, axis=0), Xva.take(va_idx, axis=0)])
            # draw some chains of samples from the VAE loop
            file_name = RESULT_PATH+"pt_walk_chain_samples_b{0:d}.png".format(i)
            Xd_samps = np.repeat(Xd_batch, 3, axis=0)
            sample_lists = VCGL.OSM.sample_from_chain(Xd_samps, loop_iters=20)
            Xs = np.vstack(sample_lists["data samples"])
            utils.visualize_samples(Xs, file_name, num_rows=20)
            # draw some masked chains of samples from the VAE loop
            file_name = RESULT_PATH+"pt_walk_mask_samples_b{0:d}.png".format(i)
            Xd_samps = np.repeat(Xc_mean[0:Xd_batch.shape[0],:], 3, axis=0)
            Xc_samps = np.repeat(Xd_batch, 3, axis=0)
            Xm_rand = sample_masks(Xc_samps, drop_prob=0.0)
            Xm_patch = sample_patch_masks(Xc_samps, (48,48), (25,25))
            Xm_samps = Xm_rand * Xm_patch
            sample_lists = VCGL.OSM.sample_from_chain(Xd_samps, \
                    X_c=Xc_samps, X_m=Xm_samps, loop_iters=20)
            Xs = np.vstack(sample_lists["data samples"])
            utils.visualize_samples(Xs, file_name, num_rows=20)
            # draw some samples independently from the GenNet's prior
            file_name = RESULT_PATH+"pt_walk_prior_samples_b{0:d}.png".format(i)
            Xs = VCGL.sample_from_prior(20*20)
            utils.visualize_samples(Xs, file_name, num_rows=20)
        # DUMP PARAMETERS FROM TIME-TO-TIME
        if (i % 5000 == 0):
            DN.save_to_file(f_name=RESULT_PATH+"pt_walk_params_b{0:d}_DN.pkl".format(i))
            IN.save_to_file(f_name=RESULT_PATH+"pt_walk_params_b{0:d}_IN.pkl".format(i))
            GN.save_to_file(f_name=RESULT_PATH+"pt_walk_params_b{0:d}_GN.pkl".format(i))
    return

if __name__=="__main__":
    # FOR EXTREME KLD REGULARIZATION
	#pretrain_osm(lam_kld=60.0)
	#train_walk_from_pretrained_osm(lam_kld=60.0)

    # FOR KLD MODEL
    # pretrain_osm(lam_kld=15.0)
    # train_walk_from_pretrained_osm(lam_kld=15.0)

    # FOR VAE MODEL
    pretrain_osm(lam_kld=1.0)
    #train_walk_from_pretrained_osm(lam_kld=1.0)