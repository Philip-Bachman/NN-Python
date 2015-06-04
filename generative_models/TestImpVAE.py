##################################################################
# Code for testing the variational Multi-Stage Generative Model. #
##################################################################

# basic python
import numpy as np
import numpy.random as npr
import cPickle

# theano business
import theano
import theano.tensor as T

# phil's sweetness
import utils
from NetLayers import relu_actfun, softplus_actfun, tanh_actfun
from InfNet import InfNet
from HydraNet import HydraNet
from GPSImputer import GPSImputer
from OneStageModel import OneStageModel
from load_data import load_udm, load_udm_ss, load_mnist, load_binarized_mnist, \
                      load_tfd, load_svhn_gray
from HelperFuncs import construct_masked_data, shift_and_scale_into_01, \
                        row_shuffle, to_fX

RESULT_PATH = "IMP_MNIST_VAE_500/"

###############################
###############################
## TEST GPS IMPUTER ON MNIST ##
###############################
###############################

def test_mnist(occ_dim=15, drop_prob=0.0):
    #########################################
    # Format the result tag more thoroughly #
    #########################################
    dp_int = int(100.0 * drop_prob)
    result_tag = "{}VAE_OD{}_DP{}".format(RESULT_PATH, occ_dim, dp_int)

    ##########################
    # Get some training data #
    ##########################
    rng = np.random.RandomState(1234)
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm(dataset, as_shared=False, zero_mean=False)
    Xtr = datasets[0][0]
    Xva = datasets[1][0]
    Xtr = to_fX(shift_and_scale_into_01(Xtr))
    Xva = to_fX(shift_and_scale_into_01(Xva))
    tr_samples = Xtr.shape[0]
    va_samples = Xva.shape[0]
    batch_size = 200
    batch_reps = 1
    all_pix_mean = np.mean(np.mean(Xtr, axis=1))
    data_mean = to_fX(all_pix_mean * np.ones((Xtr.shape[1],)))

    ############################################################
    # Setup some parameters for the Iterative Refinement Model #
    ############################################################
    obs_dim = Xtr.shape[1]
    z_dim = 100
    imp_steps = 15 # we'll check for the best step count (found oracularly)
    init_scale = 1.0

    x_in_sym = T.matrix('x_in_sym')
    x_out_sym = T.matrix('x_out_sym')
    x_mask_sym = T.matrix('x_mask_sym')

    #################
    # p_zi_given_xi #
    #################
    params = {}
    shared_config = [obs_dim, 500, 500]
    top_config = [shared_config[-1], z_dim]
    params['shared_config'] = shared_config
    params['mu_config'] = top_config
    params['sigma_config'] = top_config
    params['activation'] = relu_actfun
    params['init_scale'] = init_scale
    params['lam_l2a'] = 0.0
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    p_zi_given_xi = InfNet(rng=rng, Xd=x_in_sym, \
            params=params, shared_param_dicts=None)
    p_zi_given_xi.init_biases(0.2)
    ###################
    # p_xip1_given_zi #
    ###################
    params = {}
    shared_config = [z_dim, 500, 500]
    output_config = [obs_dim, obs_dim]
    params['shared_config'] = shared_config
    params['output_config'] = output_config
    params['activation'] = relu_actfun
    params['init_scale'] = init_scale
    params['lam_l2a'] = 0.0
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    p_xip1_given_zi = HydraNet(rng=rng, Xd=x_in_sym, \
            params=params, shared_param_dicts=None)
    p_xip1_given_zi.init_biases(0.2)
    ###################
    # q_zi_given_x_xi #
    ###################
    params = {}
    shared_config = [(obs_dim + obs_dim), 500, 500]
    top_config = [shared_config[-1], z_dim]
    params['shared_config'] = shared_config
    params['mu_config'] = top_config
    params['sigma_config'] = top_config
    params['activation'] = relu_actfun
    params['init_scale'] = init_scale
    params['lam_l2a'] = 0.0
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    q_zi_given_x_xi = InfNet(rng=rng, Xd=x_in_sym, \
            params=params, shared_param_dicts=None)
    q_zi_given_x_xi.init_biases(0.2)


    ###########################################################
    # Define parameters for the GPSImputer, and initialize it #
    ###########################################################
    print("Building the GPSImputer...")
    gpsi_params = {}
    gpsi_params['obs_dim'] = obs_dim
    gpsi_params['z_dim'] = z_dim
    gpsi_params['imp_steps'] = imp_steps
    gpsi_params['step_type'] = 'jump'
    gpsi_params['x_type'] = 'bernoulli'
    gpsi_params['obs_transform'] = 'sigmoid'
    gpsi_params['use_osm_mode'] = True
    GPSI = GPSImputer(rng=rng, 
            x_in=x_in_sym, x_out=x_out_sym, x_mask=x_mask_sym, \
            p_zi_given_xi=p_zi_given_xi, \
            p_xip1_given_zi=p_xip1_given_zi, \
            q_zi_given_x_xi=q_zi_given_x_xi, \
            params=gpsi_params, \
            shared_param_dicts=None)
    #########################################################################
    # Define parameters for the underlying OneStageModel, and initialize it #
    #########################################################################
    print("Building the OneStageModel...")
    osm_params = {}
    osm_params['x_type'] = 'bernoulli'
    osm_params['xt_transform'] = 'sigmoid'
    OSM = OneStageModel(rng=rng, \
            x_in=x_in_sym, \
            p_x_given_z=p_xip1_given_zi, \
            q_z_given_x=p_zi_given_xi, \
            x_dim=obs_dim, z_dim=z_dim, \
            params=osm_params)

    ################################################################
    # Apply some updates, to check that they aren't totally broken #
    ################################################################
    log_name = "{}_RESULTS.txt".format(result_tag)
    out_file = open(log_name, 'wb')
    costs = [0. for i in range(10)]
    learn_rate = 0.0002
    momentum = 0.5
    batch_idx = np.arange(batch_size) + tr_samples
    for i in range(200000):
        scale = min(1.0, ((i+1) / 5000.0))
        if (((i + 1) % 15000) == 0):
            learn_rate = learn_rate * 0.92
        if (i > 10000):
            momentum = 0.90
        else:
            momentum = 0.50
        # get the indices of training samples for this batch update
        batch_idx += batch_size
        if (np.max(batch_idx) >= tr_samples):
            # we finished an "epoch", so we rejumble the training set
            Xtr = row_shuffle(Xtr)
            batch_idx = np.arange(batch_size)
        # set sgd and objective function hyperparams for this update
        OSM.set_sgd_params(lr=scale*learn_rate, \
                           mom_1=scale*momentum, mom_2=0.99)
        OSM.set_lam_nll(lam_nll=1.0)
        OSM.set_lam_kld(lam_kld_1=1.0, lam_kld_2=0.0)
        OSM.set_lam_l2w(1e-4)
        # perform a minibatch update and record the cost for this batch
        xb = to_fX( Xtr.take(batch_idx, axis=0) )
        result = OSM.train_joint(xb, batch_reps)
        costs = [(costs[j] + result[j]) for j in range(len(result)-1)]
        if ((i % 250) == 0):
            costs = [(v / 250.0) for v in costs]
            str1 = "-- batch {0:d} --".format(i)
            str2 = "    joint_cost: {0:.4f}".format(costs[0])
            str3 = "    nll_cost  : {0:.4f}".format(costs[1])
            str4 = "    kld_cost  : {0:.4f}".format(costs[2])
            str5 = "    reg_cost  : {0:.4f}".format(costs[3])
            joint_str = "\n".join([str1, str2, str3, str4, str5])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            costs = [0.0 for v in costs]
        if ((i % 1000) == 0):
            Xva = row_shuffle(Xva)
            # record an estimate of performance on the test set
            xi, xo, xm = construct_masked_data(Xva[0:5000], drop_prob=drop_prob, \
                                               occ_dim=occ_dim, data_mean=data_mean)
            step_nll, step_kld = GPSI.compute_per_step_cost(xi, xo, xm, sample_count=10)
            min_nll = np.min(step_nll)
            str1 = "    va_nll_bound : {}".format(min_nll)
            str2 = "    va_nll_min  : {}".format(min_nll)
            str3 = "    va_nll_final : {}".format(step_nll[-1])
            joint_str = "\n".join([str1, str2, str3])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
        if ((i % 5000) == 0):
            # Get some validation samples for evaluating model performance
            xb = to_fX( Xva[0:100] )
            xi, xo, xm = construct_masked_data(xb, drop_prob=drop_prob, \
                                    occ_dim=occ_dim, data_mean=data_mean)
            xi = np.repeat(xi, 2, axis=0)
            xo = np.repeat(xo, 2, axis=0)
            xm = np.repeat(xm, 2, axis=0)
            # draw some sample imputations from the model
            samp_count = xi.shape[0]
            _, model_samps = GPSI.sample_imputer(xi, xo, xm, use_guide_policy=False)
            seq_len = len(model_samps)
            seq_samps = np.zeros((seq_len*samp_count, model_samps[0].shape[1]))
            idx = 0
            for s1 in range(samp_count):
                for s2 in range(seq_len):
                    seq_samps[idx] = model_samps[s2][s1]
                    idx += 1
            file_name = "{}_samples_ng_b{}.png".format(result_tag, i)
            utils.visualize_samples(seq_samps, file_name, num_rows=20)
            # get visualizations of policy parameters
            file_name = "{}_gen_gen_weights_b{}.png".format(result_tag, i)
            W = GPSI.gen_gen_weights.get_value(borrow=False)
            utils.visualize_samples(W[:,:obs_dim], file_name, num_rows=20)
            file_name = "{}_gen_inf_weights_b{}.png".format(result_tag, i)
            W = GPSI.gen_inf_weights.get_value(borrow=False).T
            utils.visualize_samples(W[:,:obs_dim], file_name, num_rows=20)

#############################
#############################
## TEST GPS IMPUTER ON TFD ##
#############################
#############################

def test_tfd(occ_dim=15, drop_prob=0.0):
    RESULT_PATH = "IMP_TFD_VAE/"
    #########################################
    # Format the result tag more thoroughly #
    #########################################
    dp_int = int(100.0 * drop_prob)
    result_tag = "{}VAE_OD{}_DP{}".format(RESULT_PATH, occ_dim, dp_int)

    ##########################
    # Get some training data #
    ##########################
    data_file = 'data/tfd_data_48x48.pkl'
    dataset = load_tfd(tfd_pkl_name=data_file, which_set='unlabeled', fold='all')
    Xtr_unlabeled = dataset[0]
    dataset = load_tfd(tfd_pkl_name=data_file, which_set='train', fold='all')
    Xtr_train = dataset[0]
    Xtr = np.vstack([Xtr_unlabeled, Xtr_train])
    dataset = load_tfd(tfd_pkl_name=data_file, which_set='valid', fold='all')
    Xva = dataset[0]
    Xtr = to_fX(shift_and_scale_into_01(Xtr))
    Xva = to_fX(shift_and_scale_into_01(Xva))
    tr_samples = Xtr.shape[0]
    va_samples = Xva.shape[0]
    batch_size = 250
    all_pix_mean = np.mean(np.mean(Xtr, axis=1))
    data_mean = to_fX( all_pix_mean * np.ones((Xtr.shape[1],)) )

    ############################################################
    # Setup some parameters for the Iterative Refinement Model #
    ############################################################
    obs_dim = Xtr.shape[1]
    z_dim = 100
    imp_steps = 15 # we'll check for the best step count (found oracularly)
    init_scale = 1.0

    x_in_sym = T.matrix('x_in_sym')
    x_out_sym = T.matrix('x_out_sym')
    x_mask_sym = T.matrix('x_mask_sym')

    #################
    # p_zi_given_xi #
    #################
    params = {}
    shared_config = [obs_dim, 1000, 1000]
    top_config = [shared_config[-1], z_dim]
    params['shared_config'] = shared_config
    params['mu_config'] = top_config
    params['sigma_config'] = top_config
    params['activation'] = relu_actfun
    params['init_scale'] = init_scale
    params['lam_l2a'] = 0.0
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    p_zi_given_xi = InfNet(rng=rng, Xd=x_in_sym, \
            params=params, shared_param_dicts=None)
    p_zi_given_xi.init_biases(0.2)
    ###################
    # p_xip1_given_zi #
    ###################
    params = {}
    shared_config = [z_dim, 1000, 1000]
    output_config = [obs_dim, obs_dim]
    params['shared_config'] = shared_config
    params['output_config'] = output_config
    params['activation'] = relu_actfun
    params['init_scale'] = init_scale
    params['lam_l2a'] = 0.0
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    p_xip1_given_zi = HydraNet(rng=rng, Xd=x_in_sym, \
            params=params, shared_param_dicts=None)
    p_xip1_given_zi.init_biases(0.2)
    ###################
    # q_zi_given_x_xi #
    ###################
    params = {}
    shared_config = [(obs_dim + obs_dim), 1000, 1000]
    top_config = [shared_config[-1], z_dim]
    params['shared_config'] = shared_config
    params['mu_config'] = top_config
    params['sigma_config'] = top_config
    params['activation'] = relu_actfun
    params['init_scale'] = init_scale
    params['lam_l2a'] = 0.0
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    q_zi_given_x_xi = InfNet(rng=rng, Xd=x_in_sym, \
            params=params, shared_param_dicts=None)
    q_zi_given_x_xi.init_biases(0.2)


    ###########################################################
    # Define parameters for the GPSImputer, and initialize it #
    ###########################################################
    print("Building the GPSImputer...")
    gpsi_params = {}
    gpsi_params['obs_dim'] = obs_dim
    gpsi_params['z_dim'] = z_dim
    gpsi_params['imp_steps'] = imp_steps
    gpsi_params['step_type'] = 'jump'
    gpsi_params['x_type'] = 'bernoulli'
    gpsi_params['obs_transform'] = 'sigmoid'
    gpsi_params['use_osm_mode'] = True
    GPSI = GPSImputer(rng=rng, 
            x_in=x_in_sym, x_out=x_out_sym, x_mask=x_mask_sym, \
            p_zi_given_xi=p_zi_given_xi, \
            p_xip1_given_zi=p_xip1_given_zi, \
            q_zi_given_x_xi=q_zi_given_x_xi, \
            params=gpsi_params, \
            shared_param_dicts=None)
    #########################################################################
    # Define parameters for the underlying OneStageModel, and initialize it #
    #########################################################################
    print("Building the OneStageModel...")
    osm_params = {}
    osm_params['x_type'] = 'bernoulli'
    osm_params['xt_transform'] = 'sigmoid'
    OSM = OneStageModel(rng=rng, \
            x_in=x_in_sym, \
            p_x_given_z=p_xip1_given_zi, \
            q_z_given_x=p_zi_given_xi, \
            x_dim=obs_dim, z_dim=z_dim, \
            params=osm_params)

    ################################################################
    # Apply some updates, to check that they aren't totally broken #
    ################################################################
    log_name = "{}_RESULTS.txt".format(result_tag)
    out_file = open(log_name, 'wb')
    costs = [0. for i in range(10)]
    learn_rate = 0.0002
    momentum = 0.5
    batch_idx = np.arange(batch_size) + tr_samples
    for i in range(200005):
        scale = min(1.0, ((i+1) / 5000.0))
        if (((i + 1) % 15000) == 0):
            learn_rate = learn_rate * 0.92
        if (i > 10000):
            momentum = 0.90
        else:
            momentum = 0.50
        # get the indices of training samples for this batch update
        batch_idx += batch_size
        if (np.max(batch_idx) >= tr_samples):
            # we finished an "epoch", so we rejumble the training set
            Xtr = row_shuffle(Xtr)
            batch_idx = np.arange(batch_size)
        # set sgd and objective function hyperparams for this update
        OSM.set_sgd_params(lr=scale*learn_rate, \
                           mom_1=scale*momentum, mom_2=0.99)
        OSM.set_lam_nll(lam_nll=1.0)
        OSM.set_lam_kld(lam_kld_1=1.0, lam_kld_2=0.0)
        OSM.set_lam_l2w(1e-4)
        # perform a minibatch update and record the cost for this batch
        xb = to_fX( Xtr.take(batch_idx, axis=0) )
        result = OSM.train_joint(xb, batch_reps)
        costs = [(costs[j] + result[j]) for j in range(len(result)-1)]
        if ((i % 250) == 0):
            costs = [(v / 250.0) for v in costs]
            str1 = "-- batch {0:d} --".format(i)
            str2 = "    joint_cost: {0:.4f}".format(costs[0])
            str3 = "    nll_cost  : {0:.4f}".format(costs[1])
            str4 = "    kld_cost  : {0:.4f}".format(costs[2])
            str5 = "    reg_cost  : {0:.4f}".format(costs[3])
            joint_str = "\n".join([str1, str2, str3, str4, str5])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            costs = [0.0 for v in costs]
        if ((i % 1000) == 0):
            Xva = row_shuffle(Xva)
            # record an estimate of performance on the test set
            xi, xo, xm = construct_masked_data(Xva[0:5000], drop_prob=drop_prob, \
                                               occ_dim=occ_dim, data_mean=data_mean)
            step_nll, step_kld = GPSI.compute_per_step_cost(xi, xo, xm, sample_count=10)
            min_nll = np.min(step_nll)
            str1 = "    va_nll_bound : {}".format(min_nll)
            str2 = "    va_nll_min  : {}".format(min_nll)
            str3 = "    va_nll_final : {}".format(step_nll[-1])
            joint_str = "\n".join([str1, str2, str3])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
        if ((i % 10000) == 0):
            # Get some validation samples for evaluating model performance
            xb = to_fX( Xva[0:100] )
            xi, xo, xm = construct_masked_data(xb, drop_prob=drop_prob, \
                                    occ_dim=occ_dim, data_mean=data_mean)
            xi = np.repeat(xi, 2, axis=0)
            xo = np.repeat(xo, 2, axis=0)
            xm = np.repeat(xm, 2, axis=0)
            # draw some sample imputations from the model
            samp_count = xi.shape[0]
            _, model_samps = GPSI.sample_imputer(xi, xo, xm, use_guide_policy=False)
            seq_len = len(model_samps)
            seq_samps = np.zeros((seq_len*samp_count, model_samps[0].shape[1]))
            idx = 0
            for s1 in range(samp_count):
                for s2 in range(seq_len):
                    seq_samps[idx] = model_samps[s2][s1]
                    idx += 1
            file_name = "{}_samples_ng_b{}.png".format(result_tag, i)
            utils.visualize_samples(seq_samps, file_name, num_rows=20)
            # get visualizations of policy parameters
            file_name = "{}_gen_gen_weights_b{}.png".format(result_tag, i)
            W = GPSI.gen_gen_weights.get_value(borrow=False)
            utils.visualize_samples(W[:,:obs_dim], file_name, num_rows=20)
            file_name = "{}_gen_inf_weights_b{}.png".format(result_tag, i)
            W = GPSI.gen_inf_weights.get_value(borrow=False).T
            utils.visualize_samples(W[:,:obs_dim], file_name, num_rows=20)


##############################
##############################
## TEST GPS IMPUTER ON SVHN ##
##############################
##############################

def test_svhn(occ_dim=15, drop_prob=0.0):
    RESULT_PATH = "IMP_SVHN_VAE/"
    #########################################
    # Format the result tag more thoroughly #
    #########################################
    dp_int = int(100.0 * drop_prob)
    result_tag = "{}VAE_OD{}_DP{}".format(RESULT_PATH, occ_dim, dp_int)

    ##########################
    # Get some training data #
    ##########################
    tr_file = 'data/svhn_train_gray.pkl'
    te_file = 'data/svhn_test_gray.pkl'
    ex_file = 'data/svhn_extra_gray.pkl'
    data = load_svhn_gray(tr_file, te_file, ex_file=ex_file, ex_count=200000)
    Xtr = to_fX( shift_and_scale_into_01(np.vstack([data['Xtr'], data['Xex']])) )
    Xva = to_fX( shift_and_scale_into_01(data['Xte']) )
    tr_samples = Xtr.shape[0]
    va_samples = Xva.shape[0]
    batch_size = 250
    all_pix_mean = np.mean(np.mean(Xtr, axis=1))
    data_mean = to_fX( all_pix_mean * np.ones((Xtr.shape[1],)) )

    ############################################################
    # Setup some parameters for the Iterative Refinement Model #
    ############################################################
    obs_dim = Xtr.shape[1]
    z_dim = 100
    imp_steps = 15 # we'll check for the best step count (found oracularly)
    init_scale = 1.0

    x_in_sym = T.matrix('x_in_sym')
    x_out_sym = T.matrix('x_out_sym')
    x_mask_sym = T.matrix('x_mask_sym')

    #################
    # p_zi_given_xi #
    #################
    params = {}
    shared_config = [obs_dim, 1000, 1000]
    top_config = [shared_config[-1], z_dim]
    params['shared_config'] = shared_config
    params['mu_config'] = top_config
    params['sigma_config'] = top_config
    params['activation'] = relu_actfun
    params['init_scale'] = init_scale
    params['lam_l2a'] = 0.0
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    p_zi_given_xi = InfNet(rng=rng, Xd=x_in_sym, \
            params=params, shared_param_dicts=None)
    p_zi_given_xi.init_biases(0.2)
    ###################
    # p_xip1_given_zi #
    ###################
    params = {}
    shared_config = [z_dim, 1000, 1000]
    output_config = [obs_dim, obs_dim]
    params['shared_config'] = shared_config
    params['output_config'] = output_config
    params['activation'] = relu_actfun
    params['init_scale'] = init_scale
    params['lam_l2a'] = 0.0
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    p_xip1_given_zi = HydraNet(rng=rng, Xd=x_in_sym, \
            params=params, shared_param_dicts=None)
    p_xip1_given_zi.init_biases(0.2)
    ###################
    # q_zi_given_x_xi #
    ###################
    params = {}
    shared_config = [(obs_dim + obs_dim), 1000, 1000]
    top_config = [shared_config[-1], z_dim]
    params['shared_config'] = shared_config
    params['mu_config'] = top_config
    params['sigma_config'] = top_config
    params['activation'] = relu_actfun
    params['init_scale'] = init_scale
    params['lam_l2a'] = 0.0
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    q_zi_given_x_xi = InfNet(rng=rng, Xd=x_in_sym, \
            params=params, shared_param_dicts=None)
    q_zi_given_x_xi.init_biases(0.2)


    ###########################################################
    # Define parameters for the GPSImputer, and initialize it #
    ###########################################################
    print("Building the GPSImputer...")
    gpsi_params = {}
    gpsi_params['obs_dim'] = obs_dim
    gpsi_params['z_dim'] = z_dim
    gpsi_params['imp_steps'] = imp_steps
    gpsi_params['step_type'] = 'jump'
    gpsi_params['x_type'] = 'bernoulli'
    gpsi_params['obs_transform'] = 'sigmoid'
    gpsi_params['use_osm_mode'] = True
    GPSI = GPSImputer(rng=rng, 
            x_in=x_in_sym, x_out=x_out_sym, x_mask=x_mask_sym, \
            p_zi_given_xi=p_zi_given_xi, \
            p_xip1_given_zi=p_xip1_given_zi, \
            q_zi_given_x_xi=q_zi_given_x_xi, \
            params=gpsi_params, \
            shared_param_dicts=None)
    #########################################################################
    # Define parameters for the underlying OneStageModel, and initialize it #
    #########################################################################
    print("Building the OneStageModel...")
    osm_params = {}
    osm_params['x_type'] = 'bernoulli'
    osm_params['xt_transform'] = 'sigmoid'
    OSM = OneStageModel(rng=rng, \
            x_in=x_in_sym, \
            p_x_given_z=p_xip1_given_zi, \
            q_z_given_x=p_zi_given_xi, \
            x_dim=obs_dim, z_dim=z_dim, \
            params=osm_params)

    ################################################################
    # Apply some updates, to check that they aren't totally broken #
    ################################################################
    log_name = "{}_RESULTS.txt".format(result_tag)
    out_file = open(log_name, 'wb')
    costs = [0. for i in range(10)]
    learn_rate = 0.0002
    momentum = 0.5
    batch_idx = np.arange(batch_size) + tr_samples
    for i in range(200005):
        scale = min(1.0, ((i+1) / 5000.0))
        if (((i + 1) % 15000) == 0):
            learn_rate = learn_rate * 0.92
        if (i > 10000):
            momentum = 0.90
        else:
            momentum = 0.50
        # get the indices of training samples for this batch update
        batch_idx += batch_size
        if (np.max(batch_idx) >= tr_samples):
            # we finished an "epoch", so we rejumble the training set
            Xtr = row_shuffle(Xtr)
            batch_idx = np.arange(batch_size)
        # set sgd and objective function hyperparams for this update
        OSM.set_sgd_params(lr=scale*learn_rate, \
                           mom_1=scale*momentum, mom_2=0.99)
        OSM.set_lam_nll(lam_nll=1.0)
        OSM.set_lam_kld(lam_kld_1=1.0, lam_kld_2=0.0)
        OSM.set_lam_l2w(1e-4)
        # perform a minibatch update and record the cost for this batch
        xb = to_fX( Xtr.take(batch_idx, axis=0) )
        result = OSM.train_joint(xb, batch_reps)
        costs = [(costs[j] + result[j]) for j in range(len(result)-1)]
        if ((i % 250) == 0):
            costs = [(v / 250.0) for v in costs]
            str1 = "-- batch {0:d} --".format(i)
            str2 = "    joint_cost: {0:.4f}".format(costs[0])
            str3 = "    nll_cost  : {0:.4f}".format(costs[1])
            str4 = "    kld_cost  : {0:.4f}".format(costs[2])
            str5 = "    reg_cost  : {0:.4f}".format(costs[3])
            joint_str = "\n".join([str1, str2, str3, str4, str5])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            costs = [0.0 for v in costs]
        if ((i % 1000) == 0):
            Xva = row_shuffle(Xva)
            # record an estimate of performance on the test set
            xi, xo, xm = construct_masked_data(Xva[0:5000], drop_prob=drop_prob, \
                                               occ_dim=occ_dim, data_mean=data_mean)
            step_nll, step_kld = GPSI.compute_per_step_cost(xi, xo, xm, sample_count=10)
            min_nll = np.min(step_nll)
            str1 = "    va_nll_bound : {}".format(min_nll)
            str2 = "    va_nll_min  : {}".format(min_nll)
            str3 = "    va_nll_final : {}".format(step_nll[-1])
            joint_str = "\n".join([str1, str2, str3])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
        if ((i % 10000) == 0):
            # Get some validation samples for evaluating model performance
            xb = to_fX( Xva[0:100] )
            xi, xo, xm = construct_masked_data(xb, drop_prob=drop_prob, \
                                    occ_dim=occ_dim, data_mean=data_mean)
            xi = np.repeat(xi, 2, axis=0)
            xo = np.repeat(xo, 2, axis=0)
            xm = np.repeat(xm, 2, axis=0)
            # draw some sample imputations from the model
            samp_count = xi.shape[0]
            _, model_samps = GPSI.sample_imputer(xi, xo, xm, use_guide_policy=False)
            seq_len = len(model_samps)
            seq_samps = np.zeros((seq_len*samp_count, model_samps[0].shape[1]))
            idx = 0
            for s1 in range(samp_count):
                for s2 in range(seq_len):
                    seq_samps[idx] = model_samps[s2][s1]
                    idx += 1
            file_name = "{}_samples_ng_b{}.png".format(result_tag, i)
            utils.visualize_samples(seq_samps, file_name, num_rows=20)
            # get visualizations of policy parameters
            file_name = "{}_gen_gen_weights_b{}.png".format(result_tag, i)
            W = GPSI.gen_gen_weights.get_value(borrow=False)
            utils.visualize_samples(W[:,:obs_dim], file_name, num_rows=20)
            file_name = "{}_gen_inf_weights_b{}.png".format(result_tag, i)
            W = GPSI.gen_inf_weights.get_value(borrow=False).T
            utils.visualize_samples(W[:,:obs_dim], file_name, num_rows=20)


if __name__=="__main__":
    #########
    # MNIST #
    #########
    #test_mnist(occ_dim=0, drop_prob=0.6)
    #test_mnist(occ_dim=0, drop_prob=0.7)
    #test_mnist(occ_dim=0, drop_prob=0.8)
    #test_mnist(occ_dim=0, drop_prob=0.9)
    #test_mnist(occ_dim=14, drop_prob=0.0)
    #test_mnist(occ_dim=16, drop_prob=0.0)

    #######
    # TFD #
    #######
    #test_tfd(occ_dim=25, drop_prob=0.0)
    #test_tfd(occ_dim=25, drop_prob=0.8)

    ########
    # SVHN #
    ########
    #test_svhn(occ_dim=17, drop_prob=0.0)
    #test_svhn(occ_dim=17, drop_prob=0.8)
