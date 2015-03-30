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
from load_data import load_udm, load_udm_ss, load_mnist, load_binarized_mnist
import utils


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
    #dataset = 'data/mnist.pkl.gz'
    #datasets = load_udm(dataset, zero_mean=False)
    #Xtr_shared = datasets[0][0]
    #Xva_shared = datasets[1][0]
    #Xtr = Xtr_shared.get_value(borrow=False).astype(theano.config.floatX)
    #Xva = Xva_shared.get_value(borrow=False).astype(theano.config.floatX)
    Xtr, Xva, Xte = load_binarized_mnist(data_path='./data/')
    tr_samples = Xtr.shape[0]
    batch_size = 200
    batch_reps = 5

    ############################################################
    # Setup some parameters for the Iterative Refinement Model #
    ############################################################
    obs_dim = Xtr.shape[1]
    z_rnn_dim = 25
    z_obs_dim = 5
    jnt_dim = obs_dim + z_rnn_dim
    h_dim = 100
    x_type = 'bernoulli'
    prior_sigma = 1.0

    # some InfNet instances to build the TwoStageModel from
    x_in_sym = T.matrix('x_in_sym')
    x_out_sym = T.matrix('x_out_sym')

    ########################
    # p_s0_obs_given_z_obs #
    ########################
    params = {}
    shared_config = [z_obs_dim, 250, 250]
    top_config = [shared_config[-1], obs_dim]
    params['shared_config'] = shared_config
    params['mu_config'] = top_config
    params['sigma_config'] = top_config
    params['activation'] = softplus_actfun
    params['init_scale'] = 1.2
    params['lam_l2a'] = 1e-3
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    p_s0_obs_given_z_obs = InfNet(rng=rng, Xd=x_in_sym, prior_sigma=prior_sigma, \
            params=params, shared_param_dicts=None)
    p_s0_obs_given_z_obs.init_biases(0.2)
    #################
    # p_hi_given_si #
    #################
    params = {}
    shared_config = [jnt_dim, 500, 500]
    top_config = [shared_config[-1], h_dim]
    params['shared_config'] = shared_config
    params['mu_config'] = top_config
    params['sigma_config'] = top_config
    params['activation'] = softplus_actfun
    params['init_scale'] = 1.2
    params['lam_l2a'] = 0.0
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    p_hi_given_si = InfNet(rng=rng, Xd=x_in_sym, prior_sigma=prior_sigma, \
            params=params, shared_param_dicts=None)
    p_hi_given_si.init_biases(0.2)
    ######################
    # p_sip1_given_si_hi #
    ######################
    params = {}
    shared_config = [h_dim, 500, 500]
    top_config = [shared_config[-1], obs_dim]
    params['shared_config'] = shared_config
    params['mu_config'] = top_config
    params['sigma_config'] = top_config
    params['activation'] = softplus_actfun
    params['init_scale'] = 1.2
    params['lam_l2a'] = 0.0
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    p_sip1_given_si_hi = InfNet(rng=rng, Xd=x_in_sym, prior_sigma=prior_sigma, \
            params=params, shared_param_dicts=None)
    p_sip1_given_si_hi.init_biases(0.2)
    ###############
    # q_z_given_x #
    ###############
    params = {}
    shared_config = [obs_dim, 250, 250]
    top_config = [shared_config[-1], (z_rnn_dim + z_obs_dim)]
    params['shared_config'] = shared_config
    params['mu_config'] = top_config
    params['sigma_config'] = top_config
    params['activation'] = softplus_actfun
    params['init_scale'] = 1.2
    params['lam_l2a'] = 0.0
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    q_z_given_x = InfNet(rng=rng, Xd=x_in_sym, prior_sigma=prior_sigma, \
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
    params['activation'] = softplus_actfun
    params['init_scale'] = 1.2
    params['lam_l2a'] = 0.0
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    q_hi_given_x_si = InfNet(rng=rng, Xd=x_in_sym, prior_sigma=prior_sigma, \
            params=params, shared_param_dicts=None)
    q_hi_given_x_si.init_biases(0.2)


    ################################################################
    # Define parameters for the MultiStageModel, and initialize it #
    ################################################################
    print("Building the MultiStageModel...")
    msm_params = {}
    msm_params['x_type'] = x_type
    msm_params['obs_transform'] = 'sigmoid'
    MSM = MultiStageModel(rng=rng, x_in=x_in_sym, x_out=x_out_sym, \
            p_s0_obs_given_z_obs=p_s0_obs_given_z_obs, \
            p_hi_given_si=p_hi_given_si, \
            p_sip1_given_si_hi=p_sip1_given_si_hi, \
            q_z_given_x=q_z_given_x, \
            q_hi_given_x_si=q_hi_given_x_si, \
            obs_dim=obs_dim, z_rnn_dim=z_rnn_dim, z_obs_dim=z_obs_dim, \
            h_dim=h_dim, model_init_obs=False, model_init_rnn=True, \
            ir_steps=3, params=msm_params)
    obs_mean = (0.9 * np.mean(Xtr, axis=0)) + 0.05
    obs_mean_logit = np.log(obs_mean / (1.0 - obs_mean))
    MSM.set_input_bias(-obs_mean)
    MSM.set_obs_bias(0.1*obs_mean_logit)

    ################################################################
    # Apply some updates, to check that they aren't totally broken #
    ################################################################
    out_file = open(RESULT_PATH+"MW_RESULTS.txt", 'wb')
    costs = [0. for i in range(10)]
    learn_rate = 0.003
    momentum = 0.5
    for i in range(400000):
        scale = min(1.0, ((i+1) / 8000.0))
        l1l2_weight = min(1.0, ((i+1) / 8000.0))
        if (((i + 1) % 10000) == 0):
            learn_rate = learn_rate * 0.92
        if (i > 150000):
            momentum = 0.90
        elif (i > 100000):
            momentum = 0.70
        elif (i > 50000):
            momentum = 0.60
        else:
            momentum = 0.50
        # randomly sample a minibatch
        tr_idx = npr.randint(low=0,high=tr_samples,size=(batch_size,))
        #Xb = binarize_data(Xtr.take(tr_idx, axis=0))
        Xb = Xtr.take(tr_idx, axis=0)
        Xb = Xb.astype(theano.config.floatX)
        # set sgd and objective function hyperparams for this update
        MSM.set_sgd_params(lr_1=scale*learn_rate, lr_2=scale*learn_rate, \
                mom_1=(scale*momentum), mom_2=0.99)
        MSM.set_train_switch(1.0)
        MSM.set_l1l2_weight(l1l2_weight)
        MSM.set_lam_nll(lam_nll=1.0)
        MSM.set_lam_kld(lam_kld_1=1.0, lam_kld_2=1.0)
        MSM.set_lam_l2w(1e-5)
        MSM.set_kzg_weight(0.01)
        # perform a minibatch update and record the cost for this batch
        result = MSM.train_joint(Xb, Xb, batch_reps)
        costs = [(costs[j] + result[j]) for j in range(len(result))]
        if ((i % 500) == 0):
            costs = [(v / 500.0) for v in costs]
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
        if (((i % 2000) == 0) or ((i < 10000) and ((i % 1000) == 0))):
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
            file_name = "MW_SAMPLES_b{0:d}.png".format(i)
            utils.visualize_samples(seq_samps, file_name, num_rows=20)
            # Get some validation samples to compute some diagnostics
            #Xb = binarize_data(Xva[0:5000])
            Xb = Xva[0:5000]
            # visualize some important weights in the model
            file_name = "MW_A_INF_1_WEIGHTS_b{0:d}.png".format(i)
            W = MSM.inf_1_weights.get_value(borrow=False).T
            utils.visualize_samples(W[:,:obs_dim], file_name, num_rows=20)
            file_name = "MW_A_INF_2_WEIGHTS_b{0:d}.png".format(i)
            W = MSM.inf_2_weights.get_value(borrow=False).T
            utils.visualize_samples(W[:,:obs_dim], file_name, num_rows=20)
            file_name = "MW_A_GEN_1_WEIGHTS_b{0:d}.png".format(i)
            W = MSM.gen_1_weights.get_value(borrow=False)
            utils.visualize_samples(W[:,:obs_dim], file_name, num_rows=20)
            file_name = "MW_A_GEN_2_WEIGHTS_b{0:d}.png".format(i)
            W = MSM.gen_2_weights.get_value(borrow=False)
            utils.visualize_samples(W[:,:obs_dim], file_name, num_rows=20)
            file_name = "MW_A_GEN_INF_WEIGHTS_b{0:d}.png".format(i)
            W = MSM.gen_inf_weights.get_value(borrow=False).T
            utils.visualize_samples(W[:,:obs_dim], file_name, num_rows=20)
            # compute information about posterior KLds on validation set
            post_klds = MSM.compute_post_klds(Xb, Xb)
            file_name = "MW_B_H0_KLDS_b{0:d}.png".format(i)
            utils.plot_stem(np.arange(post_klds[0].shape[1]), \
                    np.mean(post_klds[0], axis=0), file_name)
            file_name = "MW_B_HI_COND_KLDS_b{0:d}.png".format(i)
            utils.plot_stem(np.arange(post_klds[1].shape[1]), \
                    np.mean(post_klds[1], axis=0), file_name)
            file_name = "MW_B_HI_GLOB_KLDS_b{0:d}.png".format(i)
            utils.plot_stem(np.arange(post_klds[2].shape[1]), \
                    np.mean(post_klds[2], axis=0), file_name)
            # compute information about free-energy on validation set
            file_name = "MW_B_FREE_ENERGY_b{0:d}.png".format(i)
            fe_terms = MSM.compute_fe_terms(Xb, Xb, 20)
            fe_mean = np.mean(fe_terms[0]) + np.mean(fe_terms[1])
            print("    nll_bound : {0:.4f}".format(fe_mean))
            utils.plot_scatter(fe_terms[1], fe_terms[0], file_name, \
                    x_label='Posterior KLd', y_label='Negative Log-likelihood')
    return

if __name__=="__main__":
    test_with_model_init()