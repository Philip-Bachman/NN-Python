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
from LogPDFs import log_prob_bernoulli, log_prob_gaussian2, gaussian_kld
from NetLayers import relu_actfun, softplus_actfun, tanh_actfun, \
                      apply_mask, binarize_data, row_shuffle, to_fX
from InfNet import InfNet
from GPSImputer import GPSImputer
from load_data import load_udm, load_udm_ss, load_mnist, load_binarized_mnist, \
                      load_tfd, load_svhn_gray
from HelperFuncs import sample_masks, sample_patch_masks, posterior_klds, \
                        collect_obs_costs



def construct_masked_data(xi, \
                          drop_prob=0.0, \
                          occ_dim=None, \
                          data_mean=None):
    """
    Construct randomly masked data from xi.
    """
    if data_mean is None:
        data_mean = np.zeros((xi.shape[1],))
    im_dim = int(xi.shape[1]**0.5) # images should be square
    xo = xi.copy()
    if drop_prob > 0.0:
        # apply fully-random occlusion
        xm_rand = sample_masks(xi, drop_prob=drop_prob)
    else:
        # don't apply fully-random occlusion
        xm_rand = np.ones(xi.shape)
    if occ_dim is None:
        # don't apply rectangular occlusion
        xm_patch = np.ones(xi.shape)
    else:
        # apply rectangular occlusion
        xm_patch = sample_patch_masks(xi, (im_dim,im_dim), (occ_dim,occ_dim))
    xm = xm_rand * xm_patch
    xi = (xm * xi) + ((1.0 - xm) * data_mean)
    xi = to_fX(xi)
    xo = to_fX(xo)
    xm = to_fX(xm)
    return xi, xo, xm

def shift_and_scale_into_01(X):
    X = X - np.min(X, axis=1, keepdims=True)
    X = X / np.max(X, axis=1, keepdims=True)
    return X

###############################
###############################
## TEST GPS IMPUTER ON MNIST ##
###############################
###############################

def test_mnist(lam_q2p=0.5, 
               lam_p2q=0.5, \
               prob_type='bernoulli',
               result_tag='gpsi_mnist'):
    #########################################
    # Format the result tag more thoroughly #
    #########################################
    result_tag = "{0:s}_osm_q2p{1:02d}_p2q{2:02d}_{3:s}".format(result_tag, \
            int(10 * lam_q2p), int(10 * lam_p2q), prob_type[0:4])

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
    batch_size = 500
    batch_reps = 1
    all_pix_mean = np.mean(np.mean(Xtr, axis=1))
    data_mean = to_fX( all_pix_mean * np.ones((Xtr.shape[1],)) )

    ############################################################
    # Setup some parameters for the Iterative Refinement Model #
    ############################################################
    obs_dim = Xtr.shape[1]
    z_dim = 100
    imp_steps = 5
    init_scale = 1.0
    x_type = prob_type

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
    #############
    # p_xip1_zi #
    #############
    params = {}
    shared_config = [z_dim, 1000, 1000]
    top_config = [shared_config[-1], obs_dim]
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
    p_xip1_given_zi = InfNet(rng=rng, Xd=x_in_sym, \
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
    gpsi_params['x_type'] = x_type
    gpsi_params['obs_transform'] = 'sigmoid'
    gpsi_params['use_osm_mode'] = True
    GPSI = GPSImputer(rng=rng, 
            x_in=x_in_sym, x_out=x_out_sym, x_mask=x_mask_sym, \
            p_zi_given_xi=p_zi_given_xi, \
            p_xip1_given_zi=p_xip1_given_zi, \
            q_zi_given_x_xi=q_zi_given_x_xi, \
            obs_dim=obs_dim, \
            z_dim=z_dim, \
            imp_steps=imp_steps, \
            step_type='swap', \
            params=gpsi_params, \
            shared_param_dicts=None)
    #########################################################################
    # Define parameters for the underlying OneStageModel, and initialize it #
    #########################################################################
    print("Building the OneStageModel...")
    osm_params = {}
    osm_params['x_type'] = x_type
    OSM = OneStageModel(rng=rng, \
            Xd=x_in_sym, Xc=x_out_sym, Xm=x_mask_sym, \
            p_x_given_z=p_xip1_given_zi, q_z_given_x=p_zi_given_xi, \
            x_dim=obs_dim, z_dim=z_dim, \
            params=osm_params)

    ################################################################
    # Apply some updates, to check that they aren't totally broken #
    ################################################################
    log_name = "{}_train_log.txt".format(result_tag)
    out_file = open(log_name, 'wb')
    costs = [0. for i in range(10)]
    learn_rate = 0.0003
    momentum = 0.5
    batch_idx = np.arange(batch_size) + tr_samples
    train_result_dict = {'step_nll': [], 'step_kld': [], \
                         'step_kld_q2p': [], 'step_kld_p2q': []}
    valid_result_dict = {'step_nll': [], 'step_kld': [], \
                         'step_kld_q2p': [], 'step_kld_p2q': []}
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
        OSM.set_train_switch(1.0)
        OSM.set_lam_nll(lam_nll=1.0)
        OSM.set_lam_kld(lam_kld_1=1.0, lam_kld_2=0.0)
        OSM.set_lam_l2w(1e-4)
        # perform a minibatch update and record the cost for this batch
        xb = to_fX( Xtr.take(batch_idx, axis=0) )
        result = OSM.train_joint(xb, 0.0*xb, 0.0*xb, batch_reps)
        costs = [(costs[j] + result[j]) for j in range(len(result)-1)]
        if ((i % 250) == 0):
            costs = [(v / 250.0) for v in costs]
            str1 = "-- batch {0:d} --".format(i)
            str2 = "    joint_cost: {0:.4f}".format(costs[0])
            str3 = "    nll_cost  : {0:.4f}".format(costs[1])
            str4 = "    kld_cost  : {0:.4f}".format(costs[2])
            str5 = "    reg_cost  : {0:.4f}".format(costs[3])
            joint_str = "\n".join([str1, str2, str3, str4, str5, str6])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            costs = [0.0 for v in costs]
            # record some scores for the test set
            xi, xo, xm = construct_masked_data(Xtr[0:2000], drop_prob=0.0, \
                                               occ_dim=15, data_mean=data_mean)
            raw_costs = GPSI.compute_raw_costs(xi, xo, xm)
            step_nll, step_kld, step_kld_q2p, step_kld_p2q = raw_costs
            train_result_dict['step_nll'].append((i, step_nll))
            train_result_dict['step_kld'].append((i, step_kld))
            train_result_dict['step_kld_q2p'].append((i, step_kld_q2p))
            train_result_dict['step_kld_p2q'].append((i, step_kld_p2q))
            # record some scores for the validation set
            xi, xo, xm = construct_masked_data(Xva[0:2000], drop_prob=0.0, \
                                               occ_dim=15, data_mean=data_mean)
            raw_costs = GPSI.compute_raw_costs(xi, xo, xm)
            step_nll, step_kld, step_kld_q2p, step_kld_p2q = raw_costs
            valid_result_dict['step_nll'].append((i, step_nll))
            valid_result_dict['step_kld'].append((i, step_kld))
            valid_result_dict['step_kld_q2p'].append((i, step_kld_q2p))
            valid_result_dict['step_kld_p2q'].append((i, step_kld_p2q))
        # save results to a pickle file
        result_dicts = {'train_results': train_result_dict, \
                'valid_results': valid_result_dict}
        f_handle = file("{0:s}_result_dicts.pkl".format(result_tag), 'wb')
        cPickle.dump(result_dicts, f_handle, protocol=-1)
        f_handle.close()
        if ((i % 5000) == 0):
            # Get some validation samples for evaluating model performance
            Xva = row_shuffle(Xva)
            xb = to_fX( Xva[0:100] )
            xi, xo, xm = construct_masked_data(xb, drop_prob=0.0, occ_dim=15, \
                                               data_mean=data_mean)
            xi = np.repeat(xi, 2, axis=0)
            xo = np.repeat(xo, 2, axis=0)
            xm = np.repeat(xm, 2, axis=0)
            # draw some independent random samples from the model
            samp_count = xi.shape[0]
            _, model_samps = GPSI.sample_imputer(xi, xo, xm, use_guide_policy=False)
            seq_len = len(model_samps)
            seq_samps = np.zeros((seq_len*samp_count, model_samps[0].shape[1]))
            idx = 0
            for s1 in range(samp_count):
                for s2 in range(seq_len):
                    seq_samps[idx] = model_samps[s2][s1]
                    idx += 1
            file_name = "{0:s}_samples_ng_b{1:d}.png".format(result_tag, i)
            utils.visualize_samples(seq_samps, file_name, num_rows=20)
            # draw some conditional random samples from the model
            samp_count = xi.shape[0]
            _, model_samps = GPSI.sample_imputer(xi, xo, xm, use_guide_policy=True)
            seq_len = len(model_samps)
            seq_samps = np.zeros((seq_len*samp_count, model_samps[0].shape[1]))
            idx = 0
            for s1 in range(samp_count):
                for s2 in range(seq_len):
                    seq_samps[idx] = model_samps[s2][s1]
                    idx += 1
            file_name = "{0:s}_samples_yg_b{1:d}.png".format(result_tag, i)
            utils.visualize_samples(seq_samps, file_name, num_rows=20)
            # get visualizations of policy parameters
            file_name = "{0:s}_gen_gen_weights_b{1:d}.png".format(result_tag, i)
            W = GPSI.gen_gen_weights.get_value(borrow=False)
            utils.visualize_samples(W[:,:obs_dim], file_name, num_rows=20)
            file_name = "{0:s}_gen_inf_weights_b{1:d}.png".format(result_tag, i)
            W = GPSI.gen_inf_weights.get_value(borrow=False).T
            utils.visualize_samples(W[:,:obs_dim], file_name, num_rows=20)
            # check some useful information about usage of model capacity
            xi, xo, xm = construct_masked_data(Xva[0:2500], drop_prob=0.0, 
                                               occ_dim=15, data_mean=data_mean)
            raw_costs = GPSI.compute_raw_costs(xi, xo, xm)
            step_nll, step_kld, step_kld_q2p, step_kld_p2q = raw_costs
            file_name = "{0:s}_klds_q2p_b{1:d}.png".format(result_tag, i)
            utils.plot_stem(np.arange(step_kld_q2p.shape[1]), \
                    np.mean(step_kld_q2p, axis=0), file_name)
            file_name = "{0:s}_klds_p2q_b{1:d}.png".format(result_tag, i)
            utils.plot_stem(np.arange(step_kld_p2q.shape[1]), \
                    np.mean(step_kld_p2q, axis=0), file_name)
            file_name = "{0:s}_step_nlls_b{1:d}.png".format(result_tag, i)
            utils.plot_stem(np.arange(step_nll.shape[0]), \
                    step_nll, file_name)
            file_name = "{0:s}_step_klds_b{1:d}.png".format(result_tag, i)
            utils.plot_stem(np.arange(step_kld.shape[0]), \
                    step_kld, file_name)
            file_name = "{0:s}_step_vfes_b{1:d}.png".format(result_tag, i)
            utils.plot_stem(np.arange(step_kld.shape[0]).ravel(), \
                    (np.cumsum(step_kld.ravel())+step_nll.ravel()), file_name)

#############################
#############################
## TEST GPS IMPUTER ON TFD ##
#############################
#############################

def test_tfd(lam_q2p=0.5, 
             lam_p2q=0.5, \
             prob_type='bernoulli',
             result_tag='gpsi_tfd'):
    #########################################
    # Format the result tag more thoroughly #
    #########################################
    result_tag = "{0:s}_osm_q2p{2:02d}_p2q{3:02d}_{4:s}".format(result_tag, \
            int(10 * lam_q2p), int(10 * lam_p2q), prob_type[0:4])
    ##########################
    # Get some training data #
    ##########################
    rng = np.random.RandomState(1234)
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
    batch_size = 500
    batch_reps = 1
    all_pix_mean = np.mean(np.mean(Xtr, axis=1))
    data_mean = to_fX( all_pix_mean * np.ones((Xtr.shape[1],)) )

    ############################################################
    # Setup some parameters for the Iterative Refinement Model #
    ############################################################
    obs_dim = Xtr.shape[1]
    z_dim = 200
    imp_steps = 5
    init_scale = 1.0
    x_type = prob_type

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
    #############
    # p_xip1_zi #
    #############
    params = {}
    shared_config = [z_dim, 1000, 1000]
    top_config = [shared_config[-1], obs_dim]
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
    p_xip1_given_zi = InfNet(rng=rng, Xd=x_in_sym, \
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
    gpsi_params['x_type'] = x_type
    gpsi_params['obs_transform'] = 'sigmoid'
    GPSI = GPSImputer(rng=rng, 
            x_in=x_in_sym, x_out=x_out_sym, x_mask=x_mask_sym, \
            p_zi_given_xi=p_zi_given_xi, \
            p_xip1_given_zi=p_xip1_given_zi, \
            q_zi_given_x_xi=q_zi_given_x_xi, \
            obs_dim=obs_dim, \
            z_dim=z_dim, \
            imp_steps=imp_steps, \
            step_type='swap', \
            params=gpsi_params, \
            shared_param_dicts=None)

    ################################################################
    # Apply some updates, to check that they aren't totally broken #
    ################################################################
    log_name = "{}_train_log.txt".format(result_tag)
    out_file = open(log_name, 'wb')
    costs = [0. for i in range(10)]
    learn_rate = 0.0003
    momentum = 0.5
    batch_idx = np.arange(batch_size) + tr_samples
    train_result_dict = {'step_nll': [], 'step_kld': [], \
                         'step_kld_q2p': [], 'step_kld_p2q': []}
    valid_result_dict = {'step_nll': [], 'step_kld': [], \
                         'step_kld_q2p': [], 'step_kld_p2q': []}
    for i in range(200000):
        scale = min(1.0, ((i+1) / 5000.0))
        kld_scale = min(1.0, ((i+1) / 40000.0))
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
        GPSI.set_sgd_params(lr=scale*learn_rate, \
                            mom_1=scale*momentum, mom_2=0.99)
        GPSI.set_train_switch(1.0)
        GPSI.set_lam_nll(lam_nll=1.0)
        GPSI.set_lam_kld(lam_kld_p=(kld_scale*lam_p2q), \
                         lam_kld_q=(kld_scale*lam_q2p))
        GPSI.set_lam_ent(lam_ent_p=0.00, lam_ent_q=0.01)
        GPSI.set_lam_l2w(1e-5)
        # perform a minibatch update and record the cost for this batch
        xb = to_fX( Xtr.take(batch_idx, axis=0) )
        xi, xo, xm = construct_masked_data(xb, drop_prob=0.0, occ_dim=26, \
                                           data_mean=data_mean)
        result = GPSI.train_joint(xi, xo, xm, batch_reps)
        batch_costs = result[-1] # get the per-input costs
        obs_costs = collect_obs_costs(batch_costs, batch_reps)
        costs = [(costs[j] + result[j]) for j in range(len(result)-1)]
        if ((i % 250) == 0):
            costs = [(v / 250.0) for v in costs]
            str1 = "-- batch {0:d} --".format(i)
            str2 = "    joint_cost: {0:.4f}".format(costs[0])
            str3 = "    nll_cost  : {0:.4f}".format(costs[1])
            str4 = "    kld_cost  : {0:.4f}".format(costs[2])
            str5 = "    ent_cost  : {0:.4f}".format(costs[3])
            str6 = "    reg_cost  : {0:.4f}".format(costs[4])
            joint_str = "\n".join([str1, str2, str3, str4, str5, str6])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            costs = [0.0 for v in costs]
            # record some scores for the test set
            xi, xo, xm = construct_masked_data(Xtr[0:2000], drop_prob=0.0,
                                               occ_dim=26, data_mean=data_mean)
            raw_costs = GPSI.compute_raw_costs(xi, xo, xm)
            step_nll, step_kld, step_kld_q2p, step_kld_p2q = raw_costs
            train_result_dict['step_nll'].append((i, step_nll))
            train_result_dict['step_kld'].append((i, step_kld))
            train_result_dict['step_kld_q2p'].append((i, step_kld_q2p))
            train_result_dict['step_kld_p2q'].append((i, step_kld_p2q))
            # record some scores for the validation set
            xi, xo, xm = construct_masked_data(Xva[0:2000], drop_prob=0.0,
                                               occ_dim=26, data_mean=data_mean)
            raw_costs = GPSI.compute_raw_costs(xi, xo, xm)
            step_nll, step_kld, step_kld_q2p, step_kld_p2q = raw_costs
            valid_result_dict['step_nll'].append((i, step_nll))
            valid_result_dict['step_kld'].append((i, step_kld))
            valid_result_dict['step_kld_q2p'].append((i, step_kld_q2p))
            valid_result_dict['step_kld_p2q'].append((i, step_kld_p2q))
        # save results to a pickle file
        result_dicts = {'train_results': train_result_dict, \
                'valid_results': valid_result_dict}
        f_handle = file("{0:s}_result_dicts.pkl".format(result_tag), 'wb')
        cPickle.dump(result_dicts, f_handle, protocol=-1)
        f_handle.close()
        if ((i % 5000) == 0):
            # Get some validation samples for evaluating model performance
            Xva = row_shuffle(Xva)
            xb = to_fX( Xva[0:100] )
            xi, xo, xm = construct_masked_data(xb, drop_prob=0.0, occ_dim=26, \
                                               data_mean=data_mean)
            xi = np.repeat(xi, 2, axis=0)
            xo = np.repeat(xo, 2, axis=0)
            xm = np.repeat(xm, 2, axis=0)
            # draw some independent random samples from the model
            samp_count = xi.shape[0]
            _, model_samps = GPSI.sample_imputer(xi, xo, xm, use_guide_policy=False)
            seq_len = len(model_samps)
            seq_samps = np.zeros((seq_len*samp_count, model_samps[0].shape[1]))
            idx = 0
            for s1 in range(samp_count):
                for s2 in range(seq_len):
                    seq_samps[idx] = model_samps[s2][s1]
                    idx += 1
            file_name = "{0:s}_samples_ng_b{1:d}.png".format(result_tag, i)
            utils.visualize_samples(seq_samps, file_name, num_rows=20)
            # draw some conditional random samples from the model
            samp_count = xi.shape[0]
            _, model_samps = GPSI.sample_imputer(xi, xo, xm, use_guide_policy=True)
            seq_len = len(model_samps)
            seq_samps = np.zeros((seq_len*samp_count, model_samps[0].shape[1]))
            idx = 0
            for s1 in range(samp_count):
                for s2 in range(seq_len):
                    seq_samps[idx] = model_samps[s2][s1]
                    idx += 1
            file_name = "{0:s}_samples_yg_b{1:d}.png".format(result_tag, i)
            utils.visualize_samples(seq_samps, file_name, num_rows=20)
            # get visualizations of policy parameters
            file_name = "{0:s}_gen_gen_weights_b{1:d}.png".format(result_tag, i)
            W = GPSI.gen_gen_weights.get_value(borrow=False)
            utils.visualize_samples(W[:,:obs_dim], file_name, num_rows=20)
            file_name = "{0:s}_gen_inf_weights_b{1:d}.png".format(result_tag, i)
            W = GPSI.gen_inf_weights.get_value(borrow=False).T
            utils.visualize_samples(W[:,:obs_dim], file_name, num_rows=20)
            # check some useful information about usage of model capacity
            xi, xo, xm = construct_masked_data(Xva[0:2500], drop_prob=0.0, \
                                               occ_dim=26, data_mean=data_mean)
            raw_costs = GPSI.compute_raw_costs(xi, xo, xm)
            step_nll, step_kld, step_kld_q2p, step_kld_p2q = raw_costs
            file_name = "{0:s}_klds_q2p_b{1:d}.png".format(result_tag, i)
            utils.plot_stem(np.arange(step_kld_q2p.shape[1]), \
                    np.mean(step_kld_q2p, axis=0), file_name)
            file_name = "{0:s}_klds_p2q_b{1:d}.png".format(result_tag, i)
            utils.plot_stem(np.arange(step_kld_p2q.shape[1]), \
                    np.mean(step_kld_p2q, axis=0), file_name)
            file_name = "{0:s}_step_nlls_b{1:d}.png".format(result_tag, i)
            utils.plot_stem(np.arange(step_nll.shape[0]), \
                    step_nll, file_name)
            file_name = "{0:s}_step_klds_b{1:d}.png".format(result_tag, i)
            utils.plot_stem(np.arange(step_kld.shape[0]), \
                    step_kld, file_name)
            file_name = "{0:s}_step_vfes_b{1:d}.png".format(result_tag, i)
            utils.plot_stem(np.arange(step_kld.shape[0]).ravel(), \
                    (np.cumsum(step_kld.ravel())+step_nll.ravel()), file_name)

##############################
##############################
## TEST GPS IMPUTER ON SVHN ##
##############################
##############################

def test_svhn(lam_q2p=0.5, 
              lam_p2q=0.5, \
              prob_type='bernoulli',
              result_tag='gpsi_svhn'):
    #########################################
    # Format the result tag more thoroughly #
    #########################################
    result_tag = "{0:s}_osm_q2p{2:02d}_p2q{3:02d}_{4:s}".format(result_tag, \
            int(10 * lam_q2p), int(10 * lam_p2q), prob_type[0:4])
    ##########################
    # Get some training data #
    ##########################
    rng = np.random.RandomState(1234)
     # Load some data to train/validate/test with
    tr_file = 'data/svhn_train_gray.pkl'
    te_file = 'data/svhn_test_gray.pkl'
    ex_file = 'data/svhn_extra_gray.pkl'
    data = load_svhn_gray(tr_file, te_file, ex_file=ex_file, ex_count=200000)
    #all_file = 'data/svhn_all_gray_zca.pkl'
    #data = load_svhn_all_gray_zca(all_file)
    Xtr = to_fX( shift_and_scale_into_01(np.vstack([data['Xtr'], data['Xex']])) )
    Xva = to_fX( shift_and_scale_into_01(data['Xte']) )
    tr_samples = Xtr.shape[0]
    va_samples = Xva.shape[0]
    batch_size = 500
    batch_reps = 1
    all_pix_mean = np.mean(np.mean(Xtr, axis=1))
    data_mean = to_fX( all_pix_mean * np.ones((Xtr.shape[1],)) )

    ############################################################
    # Setup some parameters for the Iterative Refinement Model #
    ############################################################
    obs_dim = Xtr.shape[1]
    z_dim = 200
    imp_steps = 5
    init_scale = 1.0
    x_type = prob_type

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
    #############
    # p_xip1_zi #
    #############
    params = {}
    shared_config = [z_dim, 1000, 1000]
    top_config = [shared_config[-1], obs_dim]
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
    p_xip1_given_zi = InfNet(rng=rng, Xd=x_in_sym, \
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
    gpsi_params['x_type'] = x_type
    gpsi_params['obs_transform'] = 'sigmoid'
    GPSI = GPSImputer(rng=rng, 
            x_in=x_in_sym, x_out=x_out_sym, x_mask=x_mask_sym, \
            p_zi_given_xi=p_zi_given_xi, \
            p_xip1_given_zi=p_xip1_given_zi, \
            q_zi_given_x_xi=q_zi_given_x_xi, \
            obs_dim=obs_dim, \
            z_dim=z_dim, \
            imp_steps=imp_steps, \
            step_type='swap', \
            params=gpsi_params, \
            shared_param_dicts=None)

    ################################################################
    # Apply some updates, to check that they aren't totally broken #
    ################################################################
    log_name = "{}_train_log.txt".format(result_tag)
    out_file = open(log_name, 'wb')
    costs = [0. for i in range(10)]
    learn_rate = 0.0003
    momentum = 0.5
    batch_idx = np.arange(batch_size) + tr_samples
    train_result_dict = {'step_nll': [], 'step_kld': [], \
                         'step_kld_q2p': [], 'step_kld_p2q': []}
    valid_result_dict = {'step_nll': [], 'step_kld': [], \
                         'step_kld_q2p': [], 'step_kld_p2q': []}
    for i in range(200000):
        scale = min(1.0, ((i+1) / 5000.0))
        kld_scale = min(1.0, ((i+1) / 40000.0))
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
        GPSI.set_sgd_params(lr=scale*learn_rate, \
                            mom_1=scale*momentum, mom_2=0.99)
        GPSI.set_train_switch(1.0)
        GPSI.set_lam_nll(lam_nll=1.0)
        GPSI.set_lam_kld(lam_kld_p=(kld_scale*lam_p2q), \
                         lam_kld_q=(kld_scale*lam_q2p))
        GPSI.set_lam_ent(lam_ent_p=0.00, lam_ent_q=0.01)
        GPSI.set_lam_l2w(1e-5)
        # perform a minibatch update and record the cost for this batch
        xb = to_fX( Xtr.take(batch_idx, axis=0) )
        xi, xo, xm = construct_masked_data(xb, drop_prob=0.0, occ_dim=18, \
                                           data_mean=data_mean)
        result = GPSI.train_joint(xi, xo, xm, batch_reps)
        batch_costs = result[-1] # get the per-input costs
        obs_costs = collect_obs_costs(batch_costs, batch_reps)
        costs = [(costs[j] + result[j]) for j in range(len(result)-1)]
        if ((i % 250) == 0):
            costs = [(v / 250.0) for v in costs]
            str1 = "-- batch {0:d} --".format(i)
            str2 = "    joint_cost: {0:.4f}".format(costs[0])
            str3 = "    nll_cost  : {0:.4f}".format(costs[1])
            str4 = "    kld_cost  : {0:.4f}".format(costs[2])
            str5 = "    ent_cost  : {0:.4f}".format(costs[3])
            str6 = "    reg_cost  : {0:.4f}".format(costs[4])
            joint_str = "\n".join([str1, str2, str3, str4, str5, str6])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            costs = [0.0 for v in costs]
            # record some scores for the test set
            xi, xo, xm = construct_masked_data(Xtr[0:2000], drop_prob=0.0, \
                                               occ_dim=18, data_mean=data_mean)
            raw_costs = GPSI.compute_raw_costs(xi, xo, xm)
            step_nll, step_kld, step_kld_q2p, step_kld_p2q = raw_costs
            train_result_dict['step_nll'].append((i, step_nll))
            train_result_dict['step_kld'].append((i, step_kld))
            train_result_dict['step_kld_q2p'].append((i, step_kld_q2p))
            train_result_dict['step_kld_p2q'].append((i, step_kld_p2q))
            # record some scores for the validation set
            xi, xo, xm = construct_masked_data(Xva[0:2000], drop_prob=0.0, \
                                               occ_dim=18, data_mean=data_mean)
            raw_costs = GPSI.compute_raw_costs(xi, xo, xm)
            step_nll, step_kld, step_kld_q2p, step_kld_p2q = raw_costs
            valid_result_dict['step_nll'].append((i, step_nll))
            valid_result_dict['step_kld'].append((i, step_kld))
            valid_result_dict['step_kld_q2p'].append((i, step_kld_q2p))
            valid_result_dict['step_kld_p2q'].append((i, step_kld_p2q))
        # save results to a pickle file
        result_dicts = {'train_results': train_result_dict, \
                'valid_results': valid_result_dict}
        f_handle = file("{0:s}_result_dicts.pkl".format(result_tag), 'wb')
        cPickle.dump(result_dicts, f_handle, protocol=-1)
        f_handle.close()
        if ((i % 5000) == 0):
            # Get some validation samples for evaluating model performance
            Xva = row_shuffle(Xva)
            xb = to_fX( Xva[0:100] )
            xi, xo, xm = construct_masked_data(xb, drop_prob=0.0, occ_dim=18, \
                                               data_mean=data_mean)
            xi = np.repeat(xi, 2, axis=0)
            xo = np.repeat(xo, 2, axis=0)
            xm = np.repeat(xm, 2, axis=0)
            # draw some independent random samples from the model
            samp_count = xi.shape[0]
            _, model_samps = GPSI.sample_imputer(xi, xo, xm, use_guide_policy=False)
            seq_len = len(model_samps)
            seq_samps = np.zeros((seq_len*samp_count, model_samps[0].shape[1]))
            idx = 0
            for s1 in range(samp_count):
                for s2 in range(seq_len):
                    seq_samps[idx] = model_samps[s2][s1]
                    idx += 1
            file_name = "{0:s}_samples_ng_b{1:d}.png".format(result_tag, i)
            utils.visualize_samples(seq_samps, file_name, num_rows=20)
            # draw some conditional random samples from the model
            samp_count = xi.shape[0]
            _, model_samps = GPSI.sample_imputer(xi, xo, xm, use_guide_policy=True)
            seq_len = len(model_samps)
            seq_samps = np.zeros((seq_len*samp_count, model_samps[0].shape[1]))
            idx = 0
            for s1 in range(samp_count):
                for s2 in range(seq_len):
                    seq_samps[idx] = model_samps[s2][s1]
                    idx += 1
            file_name = "{0:s}_samples_yg_b{1:d}.png".format(result_tag, i)
            utils.visualize_samples(seq_samps, file_name, num_rows=20)
            # get visualizations of policy parameters
            file_name = "{0:s}_gen_gen_weights_b{1:d}.png".format(result_tag, i)
            W = GPSI.gen_gen_weights.get_value(borrow=False)
            utils.visualize_samples(W[:,:obs_dim], file_name, num_rows=20)
            file_name = "{0:s}_gen_inf_weights_b{1:d}.png".format(result_tag, i)
            W = GPSI.gen_inf_weights.get_value(borrow=False).T
            utils.visualize_samples(W[:,:obs_dim], file_name, num_rows=20)
            # check some useful information about usage of model capacity
            xi, xo, xm = construct_masked_data(Xva[0:2500], drop_prob=0.0, \
                                               occ_dim=18, data_mean=data_mean)
            raw_costs = GPSI.compute_raw_costs(xi, xo, xm)
            step_nll, step_kld, step_kld_q2p, step_kld_p2q = raw_costs
            file_name = "{0:s}_klds_q2p_b{1:d}.png".format(result_tag, i)
            utils.plot_stem(np.arange(step_kld_q2p.shape[1]), \
                    np.mean(step_kld_q2p, axis=0), file_name)
            file_name = "{0:s}_klds_p2q_b{1:d}.png".format(result_tag, i)
            utils.plot_stem(np.arange(step_kld_p2q.shape[1]), \
                    np.mean(step_kld_p2q, axis=0), file_name)
            file_name = "{0:s}_step_nlls_b{1:d}.png".format(result_tag, i)
            utils.plot_stem(np.arange(step_nll.shape[0]), \
                    step_nll, file_name)
            file_name = "{0:s}_step_klds_b{1:d}.png".format(result_tag, i)
            utils.plot_stem(np.arange(step_kld.shape[0]), \
                    step_kld, file_name)
            file_name = "{0:s}_step_vfes_b{1:d}.png".format(result_tag, i)
            utils.plot_stem(np.arange(step_kld.shape[0]).ravel(), \
                    (np.cumsum(step_kld.ravel())+step_nll.ravel()), file_name)

if __name__=="__main__":
    #########
    # MNIST #
    #########
    # test_mnist(lam_q2p=0.5, lam_p2q=0.5, prob_type='bernoulli', result_tag='gpsi_mnist')
    # test_mnist(lam_q2p=1.0, lam_p2q=0.0, prob_type='bernoulli', result_tag='gpsi_mnist')

    #######
    # TFD #
    #######
    # test_tfd(lam_q2p=0.4, lam_p2q=0.4, prob_type='bernoulli', result_tag='gpsi_tfd')
    # test_tfd(lam_q2p=0.8, lam_p2q=0.0, prob_type='bernoulli', result_tag='gpsi_tfd')

    ########
    # SVHN #
    ########
    # test_svhn(lam_q2p=0.4, lam_p2q=0.4, prob_type='bernoulli', result_tag='gpsi_svhn')
    # test_svhn(lam_q2p=0.8, lam_p2q=0.0, prob_type='bernoulli', result_tag='gpsi_svhn')