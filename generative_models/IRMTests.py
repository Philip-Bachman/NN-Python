################################################################
# Code for testing the variational Iterative Refinement Model. #
################################################################

# basic python
import numpy as np
import numpy.random as npr

# theano business
import theano
import theano.tensor as T

# phil's sweetness
from LogPDFs import log_prob_bernoulli, log_prob_gaussian2, gaussian_kld
from NetLayers import relu_actfun, softplus_actfun, safe_log, \
                      apply_mask, binarize_data, row_shuffle
from GenNet import GenNet
from InfNet import InfNet
from IRModel import IRModel
from load_data import load_udm, load_udm_ss, load_mnist
import utils

#####################################
#####################################
## TEST WITH CONSTANT INITIAL STEP ##
#####################################
#####################################

def test_with_constant_init():
    ##########################
    # Get some training data #
    ##########################
    rng = np.random.RandomState(1234)
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm(dataset, zero_mean=False)
    Xtr_shared = datasets[0][0]
    Xva_shared = datasets[1][0]
    Xtr = Xtr_shared.get_value(borrow=False).astype(theano.config.floatX)
    Xva = Xva_shared.get_value(borrow=False).astype(theano.config.floatX)
    tr_samples = Xtr.shape[0]
    batch_size = 500
    batch_reps = 10

    #########################################
    # Setup some parameters for the IRModel #
    #########################################
    prior_sigma = 1.0
    x_dim = Xtr.shape[1]
    z_dim = 50
    xt_dim = x_dim
    zt_dim = 200
    x_type = 'bernoulli'
    xt_type = 'observed'

    # some InfNet instances to build the IRModel from
    X_sym = T.matrix('X_sym')

    ###################
    # p_zti_given_xti #
    ###################
    params = {}
    shared_config = [xt_dim, 500, 500]
    top_config = [shared_config[-1], zt_dim]
    params['shared_config'] = shared_config
    params['mu_config'] = top_config
    params['sigma_config'] = top_config
    params['activation'] = relu_actfun
    params['init_scale'] = 1.0
    params['lam_l2a'] = 0.0
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    p_zti_given_xti = InfNet(rng=rng, Xd=X_sym, prior_sigma=prior_sigma, \
            params=params, shared_param_dicts=None)
    p_zti_given_xti.init_biases(0.1)
    #######################
    # p_xti_given_xti_zti #
    #######################
    params = {}
    shared_config = [zt_dim, 500, 500]
    top_config = [shared_config[-1], x_dim]
    params['shared_config'] = shared_config
    params['mu_config'] = top_config
    params['sigma_config'] = top_config
    params['activation'] = relu_actfun
    params['init_scale'] = 1.0
    params['lam_l2a'] = 0.0
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    p_xti_given_xti_zti = InfNet(rng=rng, Xd=X_sym, prior_sigma=prior_sigma, \
            params=params, shared_param_dicts=None)
    p_xti_given_xti_zti.init_biases(0.1)
    #####################
    # q_zti_given_x_xti #
    #####################
    params = {}
    shared_config = [x_dim, 500, 500]
    top_config = [shared_config[-1], zt_dim]
    params['shared_config'] = shared_config
    params['mu_config'] = top_config
    params['sigma_config'] = top_config
    params['activation'] = relu_actfun
    params['init_scale'] = 1.0
    params['lam_l2a'] = 0.0
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    q_zti_given_x_xti = InfNet(rng=rng, Xd=X_sym, prior_sigma=prior_sigma, \
            params=params, shared_param_dicts=None)
    q_zti_given_x_xti.init_biases(0.1)

    ########################################################
    # Define parameters for the IRModel, and initialize it #
    ########################################################
    print("Building the IRModel...")
    irm_params = {}
    irm_params['x_type'] = x_type
    irm_params['xt_type'] = xt_type
    irm_params['xt_transform'] = 'sigmoid'
    IRM = IRModel(rng=rng, x_in=X_sym, \
            p_xt0_given_z=None, p_zti_given_xti=p_zti_given_xti, \
            p_xti_given_xti_zti=p_xti_given_xti_zti, \
            q_z_given_x=None, q_zti_given_x_xti=q_zti_given_x_xti, \
            x_dim=x_dim, z_dim=z_dim, xt_dim=xt_dim, zt_dim=zt_dim, \
            ir_steps=3, params=irm_params)
    obs_mean = (0.9 * np.mean(Xtr, axis=0)) + 0.05
    obs_mean_logit = np.log(obs_mean / (1.0 - obs_mean))
    IRM.set_output_bias(0.0*obs_mean)
    IRM.set_init_bias(0.25*obs_mean_logit)

    ################################################################
    # Apply some updates, to check that they aren't totally broken #
    ################################################################
    costs = [0. for i in range(10)]
    learn_rate = 0.005
    for i in range(150000):
        scale_1 = min(1.0, ((i+1) / 10000.0))
        scale_2 = min(1.0, ((i+1) / 10000.0))
        if (((i + 1) % 10000) == 0):
            learn_rate = learn_rate * 0.8
        # randomly sample a minibatch
        tr_idx = npr.randint(low=0,high=tr_samples,size=(batch_size,))
        Xb = binarize_data(Xtr.take(tr_idx, axis=0))
        Xb = Xb.astype(theano.config.floatX)
        # train the coarse approximation and corrector model jointly
        IRM.set_sgd_params(lr_1=scale_1*learn_rate, lr_2=scale_1*learn_rate, \
                mom_1=0.8, mom_2=0.99)
        IRM.set_train_switch(1.0)
        IRM.set_lam_nll(lam_nll=1.0)
        IRM.set_lam_kld(lam_kld_1=1.0, lam_kld_2=1.0)
        IRM.set_lam_l2w(1e-5)
        IRM.set_kzg_weight(0.01)
        # perform a minibatch update and record the cost for this batch
        result = IRM.train_joint(Xb, batch_reps)
        costs = [(costs[j] + result[j]) for j in range(len(result))]
        if ((i % 500) == 0):
            costs = [(v / 500.0) for v in costs]
            print("-- batch {0:d} --".format(i))
            print("    joint_cost: {0:.4f}".format(costs[0]))
            print("    nll_cost  : {0:.4f}".format(costs[1]))
            print("    kld_cost  : {0:.4f}".format(costs[2]))
            print("    reg_cost  : {0:.4f}".format(costs[3]))
            costs = [0.0 for v in costs]
        if ((i % 2000) == 0):
            Xva = row_shuffle(Xva)
            # draw some independent random samples from the model
            samp_count = 200
            model_samps = IRM.sample_from_prior(samp_count)
            seq_len = len(model_samps)
            seq_samps = np.zeros((seq_len*samp_count, model_samps[0].shape[1]))
            idx = 0
            for s1 in range(samp_count): 
                for s2 in range(seq_len):
                    seq_samps[idx] = model_samps[s2][s1]
                    idx += 1
            file_name = "CI_SAMPLES_b{0:d}.png".format(i)
            utils.visualize_samples(seq_samps, file_name, num_rows=20)
            # visualize some important weights in the model
            file_name = "CI_INF_WEIGHTS_X_b{0:d}.png".format(i)
            utils.visualize_samples(IRM.inf_2_weights.get_value(borrow=False).T, \
                    file_name, num_rows=20)
            file_name = "CI_GEN_WEIGHTS_b{0:d}.png".format(i)
            utils.visualize_samples(IRM.gen_2_weights.get_value(borrow=False), \
                    file_name, num_rows=20)
            # compute information about posterior KLds on validation set
            post_klds = IRM.compute_post_klds(Xva[0:5000])
            # file_name = "CI_Z_KLDS_b{0:d}.png".format(i)
            # utils.plot_stem(np.arange(post_klds[0].shape[1]), \
            #         np.mean(post_klds[0], axis=0), file_name)
            file_name = "CI_ZTI_COND_KLDS_b{0:d}.png".format(i)
            utils.plot_stem(np.arange(post_klds[1].shape[1]), \
                    np.mean(post_klds[1], axis=0), file_name)
            file_name = "CI_ZTI_GLOB_KLDS_b{0:d}.png".format(i)
            utils.plot_stem(np.arange(post_klds[2].shape[1]), \
                    np.mean(post_klds[2], axis=0), file_name)
            # compute information about free-energy on validation set
            fe_terms = IRM.compute_fe_terms(binarize_data(Xva[0:5000]), 20)
            fe_mean = np.mean(fe_terms[0]) + np.mean(fe_terms[1])
            print("    nll_bound : {0:.4f}".format(fe_mean))
            file_name = "CI_FREE_ENERGY_b{0:d}.png".format(i)
            utils.plot_scatter(fe_terms[1], fe_terms[0], file_name, \
                    x_label='Posterior KLd', y_label='Negative Log-likelihood')
    return


########################################
########################################
## TEST WITH MODEL-BASED INITIAL STEP ##
########################################
########################################

def test_with_model_init():
    ##########################
    # Get some training data #
    ##########################
    rng = np.random.RandomState(1234)
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm(dataset, zero_mean=False)
    Xtr_shared = datasets[0][0]
    Xva_shared = datasets[1][0]
    Xtr = Xtr_shared.get_value(borrow=False).astype(theano.config.floatX)
    Xva = Xva_shared.get_value(borrow=False).astype(theano.config.floatX)
    tr_samples = Xtr.shape[0]
    batch_size = 300
    batch_reps = 10

    ############################################################
    # Setup some parameters for the Iterative Refinement Model #
    ############################################################
    prior_sigma = 1.0
    x_dim = Xtr.shape[1]
    z_dim = 100
    xt_dim = x_dim
    zt_dim = 200
    x_type = 'bernoulli'
    xt_type = 'observed'

    # some InfNet instances to build the TwoStageModel from
    X_sym = T.matrix('X_sym')

    #################
    # p_xt0_given_z #
    #################
    params = {}
    shared_config = [z_dim, 500, 500]
    top_config = [shared_config[-1], xt_dim]
    params['shared_config'] = shared_config
    params['mu_config'] = top_config
    params['sigma_config'] = top_config
    params['activation'] = relu_actfun
    params['init_scale'] = 1.0
    params['lam_l2a'] = 1e-3
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    p_xt0_given_z = InfNet(rng=rng, Xd=X_sym, prior_sigma=prior_sigma, \
            params=params, shared_param_dicts=None)
    p_xt0_given_z.init_biases(0.1)
    ###################
    # p_zti_given_xti #
    ###################
    params = {}
    shared_config = [xt_dim, 500, 500]
    top_config = [shared_config[-1], zt_dim]
    params['shared_config'] = shared_config
    params['mu_config'] = top_config
    params['sigma_config'] = top_config
    params['activation'] = relu_actfun
    params['init_scale'] = 1.0
    params['lam_l2a'] = 0.0
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    p_zti_given_xti = InfNet(rng=rng, Xd=X_sym, prior_sigma=prior_sigma, \
            params=params, shared_param_dicts=None)
    p_zti_given_xti.init_biases(0.1)
    #######################
    # p_xti_given_xti_zti #
    #######################
    params = {}
    shared_config = [zt_dim, 500, 500]
    top_config = [shared_config[-1], x_dim]
    params['shared_config'] = shared_config
    params['mu_config'] = top_config
    params['sigma_config'] = top_config
    params['activation'] = relu_actfun
    params['init_scale'] = 1.0
    params['lam_l2a'] = 0.0
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    p_xti_given_xti_zti = InfNet(rng=rng, Xd=X_sym, prior_sigma=prior_sigma, \
            params=params, shared_param_dicts=None)
    p_xti_given_xti_zti.init_biases(0.1)
    #####################
    # q_zti_given_x_xti #
    #####################
    params = {}
    shared_config = [x_dim, 500, 500]
    top_config = [shared_config[-1], zt_dim]
    params['shared_config'] = shared_config
    params['mu_config'] = top_config
    params['sigma_config'] = top_config
    params['activation'] = relu_actfun
    params['init_scale'] = 1.0
    params['lam_l2a'] = 0.0
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    q_zti_given_x_xti = InfNet(rng=rng, Xd=X_sym, prior_sigma=prior_sigma, \
            params=params, shared_param_dicts=None)
    q_zti_given_x_xti.init_biases(0.1)
    ###############
    # q_z_given_x #
    ###############
    params = {}
    shared_config = [x_dim, 500, 500]
    top_config = [shared_config[-1], z_dim]
    params['shared_config'] = shared_config
    params['mu_config'] = top_config
    params['sigma_config'] = top_config
    params['activation'] = relu_actfun
    params['init_scale'] = 1.0
    params['lam_l2a'] = 0.0
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    q_z_given_x = InfNet(rng=rng, Xd=X_sym, prior_sigma=prior_sigma, \
            params=params, shared_param_dicts=None)
    q_z_given_x.init_biases(0.1)
 
    ########################################################
    # Define parameters for the IRModel, and initialize it #
    ########################################################
    print("Building the IRModel...")
    irm_params = {}
    irm_params['x_type'] = x_type
    irm_params['xt_type'] = xt_type
    irm_params['xt_transform'] = 'sigmoid'
    IRM = IRModel(rng=rng, x_in=X_sym, \
            p_xt0_given_z=p_xt0_given_z, \
            p_zti_given_xti=p_zti_given_xti, \
            p_xti_given_xti_zti=p_xti_given_xti_zti, \
            q_z_given_x=q_z_given_x, \
            q_zti_given_x_xti=q_zti_given_x_xti, \
            x_dim=x_dim, z_dim=z_dim, xt_dim=xt_dim, zt_dim=zt_dim, \
            ir_steps=2, params=irm_params)
    obs_mean = (0.9 * np.mean(Xtr, axis=0)) + 0.05
    obs_mean_logit = np.log(obs_mean / (1.0 - obs_mean))
    IRM.set_output_bias(0.0*obs_mean)
    IRM.set_init_bias(0.25*obs_mean_logit)

    ################################################################
    # Apply some updates, to check that they aren't totally broken #
    ################################################################
    costs = [0. for i in range(10)]
    learn_rate = 0.005
    for i in range(250000):
        scale = min(1.0, ((i+1) / 8000.0))
        if (((i + 1) % 60000) == 0):
            learn_rate = learn_rate * 0.8
        # randomly sample a minibatch
        tr_idx = npr.randint(low=0,high=tr_samples,size=(batch_size,))
        Xb = binarize_data(Xtr.take(tr_idx, axis=0))
        Xb = Xb.astype(theano.config.floatX)
        # train the coarse approximation and corrector model jointly
        if False: #((i < 10000) and ((i % 100) > 75)):
            IRM.set_sgd_params(lr_1=0.0*learn_rate, lr_2=scale*learn_rate, \
                    mom_1=0.8, mom_2=0.99)
        else:
            IRM.set_sgd_params(lr_1=scale*learn_rate, lr_2=scale*learn_rate, \
                    mom_1=scale*0.9, mom_2=0.99)
        IRM.set_train_switch(1.0)
        IRM.set_l1l2_weight(scale)
        IRM.set_lam_nll(lam_nll=1.0)
        IRM.set_lam_kld(lam_kld_1=1.0, lam_kld_2=1.0)
        IRM.set_lam_l2w(1e-5)
        IRM.set_kzg_weight(0.01)
        # perform a minibatch update and record the cost for this batch
        result = IRM.train_joint(Xb, batch_reps)
        costs = [(costs[j] + result[j]) for j in range(len(result))]
        if ((i % 500) == 0):
            costs = [(v / 500.0) for v in costs]
            print("-- batch {0:d} --".format(i))
            print("    joint_cost: {0:.4f}".format(costs[0]))
            print("    nll_cost  : {0:.4f}".format(costs[1]))
            print("    kld_cost  : {0:.4f}".format(costs[2]))
            print("    reg_cost  : {0:.4f}".format(costs[3]))
            costs = [0.0 for v in costs]
        if ((i % 2000) == 0):
            Xva = row_shuffle(Xva)
            # draw some independent random samples from the model
            samp_count = 200
            model_samps = IRM.sample_from_prior(samp_count)
            seq_len = len(model_samps)
            seq_samps = np.zeros((seq_len*samp_count, model_samps[0].shape[1]))
            idx = 0
            for s1 in range(samp_count): 
                for s2 in range(seq_len):
                    seq_samps[idx] = model_samps[s2][s1]
                    idx += 1
            file_name = "MI_SAMPLES_b{0:d}.png".format(i)
            utils.visualize_samples(seq_samps, file_name, num_rows=20)
            # visualize some important weights in the model
            file_name = "MI_INF_1_WEIGHTS_b{0:d}.png".format(i)
            utils.visualize_samples(IRM.inf_1_weights.get_value(borrow=False).T, \
                    file_name, num_rows=20)
            file_name = "MI_GEN_1_WEIGHTS_b{0:d}.png".format(i)
            utils.visualize_samples(IRM.gen_1_weights.get_value(borrow=False), \
                    file_name, num_rows=20)
            file_name = "MI_INF_2_WEIGHTS_b{0:d}.png".format(i)
            utils.visualize_samples(IRM.inf_2_weights.get_value(borrow=False).T, \
                    file_name, num_rows=20)
            file_name = "MI_GEN_2_WEIGHTS_b{0:d}.png".format(i)
            utils.visualize_samples(IRM.gen_2_weights.get_value(borrow=False), \
                    file_name, num_rows=20)
            file_name = "MI_GEN_INF_WEIGHTS_b{0:d}.png".format(i)
            utils.visualize_samples(IRM.gen_inf_weights.get_value(borrow=False).T, \
                    file_name, num_rows=20)
            # compute information about posterior KLds on validation set
            post_klds = IRM.compute_post_klds(Xva[0:5000])
            file_name = "MI_Z_KLDS_b{0:d}.png".format(i)
            utils.plot_stem(np.arange(post_klds[0].shape[1]), \
                    np.mean(post_klds[0], axis=0), file_name)
            file_name = "MI_ZTI_COND_KLDS_b{0:d}.png".format(i)
            utils.plot_stem(np.arange(post_klds[1].shape[1]), \
                    np.mean(post_klds[1], axis=0), file_name)
            file_name = "MI_ZTI_GLOB_KLDS_b{0:d}.png".format(i)
            utils.plot_stem(np.arange(post_klds[2].shape[1]), \
                    np.mean(post_klds[2], axis=0), file_name)
            # compute information about free-energy on validation set
            file_name = "MI_FREE_ENERGY_b{0:d}.png".format(i)
            fe_terms = IRM.compute_fe_terms(binarize_data(Xva[0:5000]), 20)
            fe_mean = np.mean(fe_terms[0]) + np.mean(fe_terms[1])
            print("    nll_bound : {0:.4f}".format(fe_mean))
            utils.plot_scatter(fe_terms[1], fe_terms[0], file_name, \
                    x_label='Posterior KLd', y_label='Negative Log-likelihood')
    return

if __name__=="__main__":
    #test_with_constant_init()
    test_with_model_init()