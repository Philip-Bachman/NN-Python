##################################################################
# Code for testing the variational Multi-Stage Generative Model. #
##################################################################

# basic python
import numpy as np
import numpy.random as npr

# theano business
import theano
import theano.tensor as T

# phil's sweetness
from LogPDFs import log_prob_bernoulli, log_prob_gaussian2, gaussian_kld
from NetLayers import relu_actfun, softplus_actfun, \
                      apply_mask, binarize_data, row_shuffle
from GenNet import GenNet
from InfNet import InfNet
from MultiStageModel import MultiStageModel
from load_data import load_udm, load_udm_ss, load_mnist
import utils

#####################################
#####################################
## TEST WITH CONSTANT INITIAL STEP ##
#####################################
#####################################



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
    batch_reps = 6

    ############################################################
    # Setup some parameters for the Iterative Refinement Model #
    ############################################################
    obs_dim = Xtr.shape[1]
    rnn_dim = 25
    jnt_dim = obs_dim + rnn_dim
    z_dim = 25
    h_dim = 100
    x_type = 'bernoulli'
    prior_sigma = 1.0

    # some InfNet instances to build the TwoStageModel from
    X_sym = T.matrix('X_sym')

    ################
    # p_s0_given_z #
    ################
    params = {}
    shared_config = [z_dim, 250, 250]
    top_config = [shared_config[-1], jnt_dim]
    params['shared_config'] = shared_config
    params['mu_config'] = top_config
    params['sigma_config'] = top_config
    params['activation'] = relu_actfun
    params['init_scale'] = 1.2
    params['lam_l2a'] = 1e-3
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    p_s0_given_z = InfNet(rng=rng, Xd=X_sym, prior_sigma=prior_sigma, \
            params=params, shared_param_dicts=None)
    p_s0_given_z.init_biases(0.2)
    #################
    # p_hi_given_si #
    #################
    params = {}
    shared_config = [jnt_dim, 500, 500]
    top_config = [shared_config[-1], h_dim]
    params['shared_config'] = shared_config
    params['mu_config'] = top_config
    params['sigma_config'] = top_config
    params['activation'] = relu_actfun
    params['init_scale'] = 1.2
    params['lam_l2a'] = 0.0
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    p_hi_given_si = InfNet(rng=rng, Xd=X_sym, prior_sigma=prior_sigma, \
            params=params, shared_param_dicts=None)
    p_hi_given_si.init_biases(0.2)
    ######################
    # p_sip1_given_si_hi #
    ######################
    params = {}
    shared_config = [(h_dim + rnn_dim), 500, 500]
    top_config = [shared_config[-1], obs_dim]
    params['shared_config'] = shared_config
    params['mu_config'] = top_config
    params['sigma_config'] = top_config
    params['activation'] = relu_actfun
    params['init_scale'] = 1.2
    params['lam_l2a'] = 0.0
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    p_sip1_given_si_hi = InfNet(rng=rng, Xd=X_sym, prior_sigma=prior_sigma, \
            params=params, shared_param_dicts=None)
    p_sip1_given_si_hi.init_biases(0.2)
    ###############
    # q_z_given_x #
    ###############
    params = {}
    shared_config = [obs_dim, 250, 250]
    top_config = [shared_config[-1], z_dim]
    params['shared_config'] = shared_config
    params['mu_config'] = top_config
    params['sigma_config'] = top_config
    params['activation'] = relu_actfun
    params['init_scale'] = 1.2
    params['lam_l2a'] = 0.0
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    q_z_given_x = InfNet(rng=rng, Xd=X_sym, prior_sigma=prior_sigma, \
            params=params, shared_param_dicts=None)
    q_z_given_x.init_biases(0.2)
    ###################
    # q_hi_given_x_si #
    ###################
    params = {}
    shared_config = [(obs_dim + jnt_dim), 500, 500]
    top_config = [shared_config[-1], h_dim]
    params['shared_config'] = shared_config
    params['mu_config'] = top_config
    params['sigma_config'] = top_config
    params['activation'] = relu_actfun
    params['init_scale'] = 1.2
    params['lam_l2a'] = 0.0
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    q_hi_given_x_si = InfNet(rng=rng, Xd=X_sym, prior_sigma=prior_sigma, \
            params=params, shared_param_dicts=None)
    q_hi_given_x_si.init_biases(0.2)

 
    ########################################################
    # Define parameters for the MultiStageModel, and initialize it #
    ########################################################
    print("Building the MultiStageModel...")
    msm_params = {}
    msm_params['x_type'] = x_type
    msm_params['obs_transform'] = 'sigmoid'
    MSM = MultiStageModel(rng=rng, x_in=X_sym, \
            p_s0_given_z=p_s0_given_z, \
            p_hi_given_si=p_hi_given_si, \
            p_sip1_given_si_hi=p_sip1_given_si_hi, \
            q_z_given_x=q_z_given_x, \
            q_hi_given_x_si=q_hi_given_x_si, \
            obs_dim=obs_dim, rnn_dim=rnn_dim, z_dim=z_dim, h_dim=h_dim, \
            model_init=True, ir_steps=3, params=msm_params)
    obs_mean = (0.9 * np.mean(Xtr, axis=0)) + 0.05
    obs_mean_logit = np.log(obs_mean / (1.0 - obs_mean))
    MSM.set_input_bias(-obs_mean)
    MSM.set_output_bias(0.5*obs_mean_logit)

    ################################################################
    # Apply some updates, to check that they aren't totally broken #
    ################################################################
    costs = [0. for i in range(10)]
    learn_rate = 0.0015
    momentum = 0.5
    for i in range(300000):
        scale = min(1.0, ((i+1) / 5000.0))
        l1l2_weight = 1.0 #min(1.0, ((i+1) / 2500.0))
        if (((i + 1) % 10000) == 0):
            learn_rate = learn_rate * 0.95
        if (i > 100000):
            momentum = 0.9
        if (i > 50000):
            momentum = 0.7
        else:
            momentum = 0.5
        # randomly sample a minibatch
        tr_idx = npr.randint(low=0,high=tr_samples,size=(batch_size,))
        Xb = binarize_data(Xtr.take(tr_idx, axis=0))
        Xb = Xb.astype(theano.config.floatX)
        MSM.set_sgd_params(lr_1=scale*learn_rate, lr_2=scale*learn_rate, \
                mom_1=(scale*momentum), mom_2=0.99)
        MSM.set_train_switch(1.0)
        MSM.set_l1l2_weight(l1l2_weight)
        MSM.set_lam_nll(lam_nll=1.0)
        MSM.set_lam_kld(lam_kld_1=1.0, lam_kld_2=1.0)
        MSM.set_lam_l2w(1e-5)
        MSM.set_kzg_weight(0.01)
        # perform a minibatch update and record the cost for this batch
        result = MSM.train_joint(Xb, batch_reps)
        costs = [(costs[j] + result[j]) for j in range(len(result))]
        if ((i % 250) == 0):
            costs = [(v / 250.0) for v in costs]
            print("-- batch {0:d} --".format(i))
            print("    joint_cost: {0:.4f}".format(costs[0]))
            print("    nll_cost  : {0:.4f}".format(costs[1]))
            print("    kld_cost  : {0:.4f}".format(costs[2]))
            print("    reg_cost  : {0:.4f}".format(costs[3]))
            costs = [0.0 for v in costs]
        if ((i % 1000) == 0):
            Xva = row_shuffle(Xva)
            # draw some independent random samples from the model
            samp_count = 200
            model_samps = MSM.sample_from_prior(samp_count)
            seq_len = len(model_samps)
            seq_samps = np.zeros((seq_len*samp_count, model_samps[0].shape[1]))
            idx = 0
            for s1 in range(samp_count): 
                for s2 in range(seq_len):
                    seq_samps[idx] = model_samps[s2][s1]
                    idx += 1
            file_name = "MX_SAMPLES_b{0:d}.png".format(i)
            utils.visualize_samples(seq_samps, file_name, num_rows=20)
            # visualize some important weights in the model
            file_name = "MX_INF_1_WEIGHTS_b{0:d}.png".format(i)
            W = MSM.inf_1_weights.get_value(borrow=False).T
            utils.visualize_samples(W[:,:obs_dim], file_name, num_rows=20)
            file_name = "MX_INF_2_WEIGHTS_b{0:d}.png".format(i)
            W = MSM.inf_2_weights.get_value(borrow=False).T
            utils.visualize_samples(W[:,:obs_dim], file_name, num_rows=20)
            file_name = "MX_GEN_1_WEIGHTS_b{0:d}.png".format(i)
            W = MSM.gen_1_weights.get_value(borrow=False)
            utils.visualize_samples(W[:,:obs_dim], file_name, num_rows=20)
            file_name = "MX_GEN_2_WEIGHTS_b{0:d}.png".format(i)
            W = MSM.gen_2_weights.get_value(borrow=False)
            utils.visualize_samples(W[:,:obs_dim], file_name, num_rows=20)
            file_name = "MX_GEN_INF_WEIGHTS_b{0:d}.png".format(i)
            W = MSM.gen_inf_weights.get_value(borrow=False).T
            utils.visualize_samples(W[:,:obs_dim], file_name, num_rows=20)
            # compute information about posterior KLds on validation set
            post_klds = MSM.compute_post_klds(Xva[0:5000])
            file_name = "MX_H0_KLDS_b{0:d}.png".format(i)
            utils.plot_stem(np.arange(post_klds[0].shape[1]), \
                    np.mean(post_klds[0], axis=0), file_name)
            file_name = "MX_HI_COND_KLDS_b{0:d}.png".format(i)
            utils.plot_stem(np.arange(post_klds[1].shape[1]), \
                    np.mean(post_klds[1], axis=0), file_name)
            file_name = "MX_HI_GLOB_KLDS_b{0:d}.png".format(i)
            utils.plot_stem(np.arange(post_klds[2].shape[1]), \
                    np.mean(post_klds[2], axis=0), file_name)
            # compute information about free-energy on validation set
            file_name = "MX_FREE_ENERGY_b{0:d}.png".format(i)
            fe_terms = MSM.compute_fe_terms(binarize_data(Xva[0:5000]), 20)
            fe_mean = np.mean(fe_terms[0]) + np.mean(fe_terms[1])
            print("    nll_bound : {0:.4f}".format(fe_mean))
            utils.plot_scatter(fe_terms[1], fe_terms[0], file_name, \
                    x_label='Posterior KLd', y_label='Negative Log-likelihood')
    return

if __name__=="__main__":
    #test_with_constant_init()
    test_with_model_init()