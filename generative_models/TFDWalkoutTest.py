import time
import utils as utils
import numpy as np
import numpy.random as npr
import theano
import theano.tensor as T

from load_data import load_tfd
from PeaNet import PeaNet, load_peanet_from_file
from InfNet import InfNet, load_infnet_from_file
from HydraNet import HydraNet, load_hydranet_from_file
from VCGLoop import VCGLoop
from OneStageModel import OneStageModel
from NetLayers import relu_actfun, softplus_actfun, \
                      safe_softmax, row_shuffle
from HelperFuncs import sample_masks, sample_patch_masks, posterior_klds, \
                        collect_obs_costs

import sys, resource
resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
sys.setrecursionlimit(10**6)

# DERP
#RESULT_PATH = "TFD_WALKOUT_TEST_KLD/"
RESULT_PATH = "TFD_WALKOUT_TEST_VAE/"
#RESULT_PATH = "TFD_WALKOUT_TEST_MAX_KLD/"
PRIOR_DIM = 100
LOGVAR_BOUND = 6.0

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
    batch_size = 100
    batch_reps = 5

    # setup some symbolic variables and stuff
    Xd = T.matrix('Xd_base')
    data_dim = Xtr.shape[1]
    Xtr_mean = np.mean(Xtr, axis=0)

    ##########################
    # NETWORK CONFIGURATIONS #
    ##########################
    gn_params = {}
    shared_config = [PRIOR_DIM, 1500, 1500]
    output_config = [data_dim, data_dim]
    gn_params['shared_config'] = shared_config
    gn_params['output_config'] = output_config
    gn_params['activation'] = relu_actfun
    gn_params['init_scale'] = 1.2
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
    in_params['init_scale'] = 1.2
    in_params['lam_l2a'] = 0.0
    in_params['vis_drop'] = 0.0
    in_params['hid_drop'] = 0.0
    in_params['bias_noise'] = 0.0
    in_params['input_noise'] = 0.0
    # Initialize the base networks for this OneStageModel
    IN = InfNet(rng=rng, Xd=Xd, \
            params=in_params, shared_param_dicts=None)
    GN = HydraNet(rng=rng, Xd=Xd, \
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
    # GN = load_hydranet_from_file(f_name=gn_fname, rng=rng, Xd=Xd, \
    #         new_params=None)
    # in_params = IN.params
    # gn_params = GN.params

    #########################
    # INITIALIZE THE GIPAIR #
    #########################
    osm_params = {}
    osm_params['x_type'] = 'gaussian'
    osm_params['xt_transform'] = 'sigmoid'
    osm_params['logvar_bound'] = LOGVAR_BOUND
    OSM = OneStageModel(rng=rng, x_in=Xd, \
            p_x_given_z=GN, q_z_given_x=IN, \
            x_dim=data_dim, z_dim=PRIOR_DIM, params=osm_params)
    OSM.set_lam_l2w(1e-4)

    ######################
    # BASIC VAE TRAINING #
    ######################
    out_file = open(RESULT_PATH+"pt_osm_results.txt", 'wb')
    # Set initial learning rate and basic SGD hyper parameters
    obs_costs = np.zeros((batch_size,))
    costs = [0. for i in range(10)]
    learn_rate = 0.0002
    momentum = 0.8
    for i in range(200000):
        kld_scale = min(1.0, float(i) / 20000.0)
        if ((i > 1) and ((i % 10000) == 0)):
            learn_rate = learn_rate * 0.9
        # do a minibatch update of the model, and compute some costs
        tr_idx = npr.randint(low=0,high=tr_samples,size=(batch_size,))
        Xb = Xtr.take(tr_idx, axis=0)
        # do a minibatch update of the model, and compute some costs
        OSM.set_sgd_params(lr=learn_rate, mom_1=momentum, mom_2=0.98)
        OSM.set_lam_nll(1.0)
        OSM.set_lam_kld(lam_kld_1=(1.0 + (kld_scale * (lam_kld - 1.0))), \
                        lam_kld_2=0.0)
        result = OSM.train_joint(Xb, batch_reps)
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

if __name__=="__main__":
    # FOR EXTREME KLD REGULARIZATION
	#pretrain_osm(lam_kld=50.0)
	#train_walk_from_pretrained_osm(lam_kld=60.0)

    # FOR KLD MODEL
    # pretrain_osm(lam_kld=15.0)
    # train_walk_from_pretrained_osm(lam_kld=15.0)

    # FOR VAE MODEL
    pretrain_osm(lam_kld=1.0)
    #train_walk_from_pretrained_osm(lam_kld=1.0)