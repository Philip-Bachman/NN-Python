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
from NetLayers import relu_actfun, softplus_actfun, tanh_actfun, \
                      apply_mask, binarize_data, row_shuffle, to_fX
from InfNet import InfNet
from MultiStageModelNS import MultiStageModel
from load_data import load_udm, load_udm_ss, load_mnist, load_binarized_mnist
from HelperFuncs import collect_obs_costs
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
    #datasets = load_udm(dataset, as_shared=False, zero_mean=False)
    #Xtr = datasets[0][0]
    #Xva = datasets[1][0]
    Xtr, Xva, Xte = load_binarized_mnist(data_path='./data/')
    del Xte
    tr_samples = Xtr.shape[0]
    va_samples = Xva.shape[0]
    batch_size = 500
    carry_size = 20
    batch_reps = 1
    reset_prob = 0.05


    ############################################################
    # Setup some parameters for the Iterative Refinement Model #
    ############################################################
    obs_dim = Xtr.shape[1]
    z_dim = 20
    h_dim = 100
    rnn_dim = z_dim
    jnt_dim = obs_dim + rnn_dim
    ir_steps = 4
    init_scale = 1.0
    
    x_type = 'bernoulli'

    # some InfNet instances to build the TwoStageModel from
    x_in_sym = T.matrix('x_in_sym')
    x_out_sym = T.matrix('x_out_sym')

    #################
    # p_hi_given_si #
    #################
    params = {}
    shared_config = [(jnt_dim+ir_steps), 500, 500]
    top_config = [shared_config[-1], h_dim]
    params['shared_config'] = shared_config
    params['mu_config'] = top_config
    params['sigma_config'] = top_config
    params['activation'] = softplus_actfun
    params['init_scale'] = init_scale
    params['lam_l2a'] = 0.0
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    p_hi_given_si = InfNet(rng=rng, Xd=x_in_sym, \
            params=params, shared_param_dicts=None)
    p_hi_given_si.init_biases(0.0)
    ######################
    # p_sip1_given_si_hi #
    ######################
    params = {}
    #shared_config = [h_dim, 500, 500]
    shared_config = [(h_dim + ir_steps), 500, 500]
    top_config = [shared_config[-1], obs_dim]
    params['shared_config'] = shared_config
    params['mu_config'] = top_config
    params['sigma_config'] = top_config
    params['activation'] = softplus_actfun
    params['init_scale'] = init_scale
    params['lam_l2a'] = 0.0
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    p_sip1_given_si_hi = InfNet(rng=rng, Xd=x_in_sym, \
            params=params, shared_param_dicts=None)
    p_sip1_given_si_hi.init_biases(0.0)
    ###############
    # q_z_given_x #
    ###############
    params = {}
    shared_config = [obs_dim, 500, 500]
    top_config = [shared_config[-1], z_dim]
    params['shared_config'] = shared_config
    params['mu_config'] = top_config
    params['sigma_config'] = top_config
    params['activation'] = softplus_actfun
    params['init_scale'] = init_scale
    params['lam_l2a'] = 0.0
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    q_z_given_x = InfNet(rng=rng, Xd=x_in_sym, \
            params=params, shared_param_dicts=None)
    q_z_given_x.init_biases(0.0)
    ###################
    # q_hi_given_x_si #
    ###################
    params = {}
    shared_config = [(obs_dim + jnt_dim + ir_steps), 500, 500]
    top_config = [shared_config[-1], h_dim]
    params['shared_config'] = shared_config
    params['mu_config'] = top_config
    params['sigma_config'] = top_config
    params['activation'] = softplus_actfun
    params['init_scale'] = init_scale
    params['lam_l2a'] = 0.0
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    q_hi_given_x_si = InfNet(rng=rng, Xd=x_in_sym, \
            params=params, shared_param_dicts=None)
    q_hi_given_x_si.init_biases(0.0)


    ################################################################
    # Define parameters for the MultiStageModel, and initialize it #
    ################################################################
    print("Building the MultiStageModel...")
    msm_params = {}
    msm_params['x_type'] = x_type
    msm_params['obs_transform'] = 'sigmoid'
    MSM_TR = MultiStageModel(rng=rng, x_in=x_in_sym, x_out=x_out_sym, \
            p_hi_given_si=p_hi_given_si, \
            p_sip1_given_si_hi=p_sip1_given_si_hi, \
            q_z_given_x=q_z_given_x, \
            q_hi_given_x_si=q_hi_given_x_si, \
            obs_dim=obs_dim, rnn_dim=rnn_dim, z_dim=z_dim, h_dim=h_dim, \
            model_init_rnn=True, ir_steps=ir_steps, params=msm_params)
    MSM_VA = None

    ################################################################
    # Apply some updates, to check that they aren't totally broken #
    ################################################################
    out_file = open("MWA_RESULTS.txt", 'wb')
    costs = [0. for i in range(10)]
    learn_rate = 0.0002
    momentum = 0.5
    fresh_idx = np.arange(batch_size) + tr_samples
    carry_idx = np.arange(carry_size)
    for i in range(500000):
        scale = min(1.0, ((i+1) / 2500.0))
        if (((i + 1) % 10000) == 0):
            learn_rate = learn_rate * 0.95
        if (i > 50000):
            momentum = 0.90
        else:
            momentum = 0.50
        # get the indices of training samples for this batch update
        fresh_idx += batch_size
        if (np.max(fresh_idx) >= tr_samples):
            # we finished an "epoch", so we rejumble the training set
            Xtr = row_shuffle(Xtr)
            fresh_idx = np.arange(batch_size)
        if ((i == 0) or (npr.rand() < reset_prob)):
            # sample a fully random batch
            carry_idx = npr.randint(low=0,high=tr_samples,size=(carry_size,))
        batch_idx = fresh_idx #np.concatenate([fresh_idx, carry_idx])
        # train on the training set
        MSM = MSM_TR
        MSM_VA = None
        lam_kld = 1.0
        # set sgd and objective function hyperparams for this update
        MSM.set_sgd_params(lr_1=scale*learn_rate, lr_2=scale*learn_rate, \
                mom_1=scale*momentum, mom_2=0.98)
        MSM.set_train_switch(1.0)
        MSM.set_lam_nll(lam_nll=1.0)
        MSM.set_lam_kld(lam_kld_1=lam_kld, lam_kld_2=lam_kld)
        MSM.set_lam_l2w(1e-4)
        MSM.set_kzg_weight(0.1)
        MSM.set_drop_rate(0.0)
        MSM.p_hi_given_si.set_bias_noise(0.0)
        MSM.p_sip1_given_si_hi.set_bias_noise(0.0)
        # perform a minibatch update and record the cost for this batch
        Xb_tr = to_fX( Xtr.take(batch_idx, axis=0) )
        result = MSM.train_joint(Xb_tr, Xb_tr, batch_reps)
        batch_costs = result[-1] # get the per-input costs
        obs_costs = collect_obs_costs(batch_costs, batch_reps)
        carry_idx = batch_idx[np.argsort(-obs_costs)[0:carry_size]]

        costs = [(costs[j] + result[j]) for j in range(len(result)-1)]
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
            MSM.set_drop_rate(0.0)
            MSM.p_hi_given_si.set_bias_noise(0.0)
            MSM.p_sip1_given_si_hi.set_bias_noise(0.0)
            # Get some validation samples for computing diagnostics
            Xva = row_shuffle(Xva)
            Xb_va = to_fX( Xva[0:5000] )
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
            file_name = "MWA_B_SAMPLES_IND_b{0:d}.png".format(i)
            utils.visualize_samples(seq_samps, file_name, num_rows=20)
            # draw some conditional random samples from the model
            samp_count = 200
            Xs = np.vstack((Xb_tr[0:(samp_count/4)], Xb_va[0:(samp_count/4)]))
            Xs = np.repeat(Xs, 2, axis=0)
            model_samps = MSM.sample_from_input(Xs, guided_decoding=True)
            model_samps.append(Xs)
            seq_len = len(model_samps)
            seq_samps = np.zeros((seq_len*samp_count, model_samps[0].shape[1]))
            idx = 0
            for s1 in range(samp_count): 
                for s2 in range(seq_len):
                    seq_samps[idx] = model_samps[s2][s1]
                    idx += 1
            file_name = "MWA_B_SAMPLES_CND_GD_b{0:d}.png".format(i)
            utils.visualize_samples(seq_samps, file_name, num_rows=20)
            # draw some conditional random samples from the model
            model_samps = MSM.sample_from_input(Xs, guided_decoding=False)
            model_samps.append(Xs)
            seq_len = len(model_samps)
            seq_samps = np.zeros((seq_len*samp_count, model_samps[0].shape[1]))
            idx = 0
            for s1 in range(samp_count): 
                for s2 in range(seq_len):
                    seq_samps[idx] = model_samps[s2][s1]
                    idx += 1
            file_name = "MWA_B_SAMPLES_CND_UD_b{0:d}.png".format(i)
            utils.visualize_samples(seq_samps, file_name, num_rows=20)
            # visualize some important weights in the model
            # file_name = "MWA_A_INF_1_WEIGHTS_b{0:d}.png".format(i)
            # W = MSM.inf_1_weights.get_value(borrow=False).T
            # utils.visualize_samples(W[:,:obs_dim], file_name, num_rows=20)
            # file_name = "MWA_A_INF_2_WEIGHTS_b{0:d}.png".format(i)
            # W = MSM.inf_2_weights.get_value(borrow=False).T
            # utils.visualize_samples(W[:,:obs_dim], file_name, num_rows=20)
            # file_name = "MWA_A_GEN_GEN_WEIGHTS_b{0:d}.png".format(i)
            # W = MSM.gen_gen_weights.get_value(borrow=False)
            # utils.visualize_samples(W[:,:obs_dim], file_name, num_rows=20)
            file_name = "MWA_A_GEN_INF_WEIGHTS_b{0:d}.png".format(i)
            W = MSM.gen_inf_weights.get_value(borrow=False).T
            utils.visualize_samples(W[:,:obs_dim], file_name, num_rows=20)
            # compute information about posterior KLds on validation set
            raw_costs = MSM.compute_raw_costs(Xb_va, Xb_va)
            init_nll, init_kld, cond_kld, glob_kld, step_nll, step_kld = raw_costs
            step_nll[0] = step_nll[1] # scale of first NLL is overwhemling
            file_name = "MWA_B_H0_KLDS_b{0:d}.png".format(i)
            utils.plot_stem(np.arange(init_kld.shape[1]), \
                    np.mean(init_kld, axis=0), file_name)
            file_name = "MWA_B_HI_COND_KLDS_b{0:d}.png".format(i)
            utils.plot_stem(np.arange(cond_kld.shape[1]), \
                    np.mean(cond_kld, axis=0), file_name)
            file_name = "MWA_B_HI_GLOB_KLDS_b{0:d}.png".format(i)
            utils.plot_stem(np.arange(glob_kld.shape[1]), \
                    np.mean(glob_kld, axis=0), file_name)
            file_name = "MWA_B_STEP_NLLS_b{0:d}.png".format(i)
            utils.plot_stem(np.arange(step_nll.shape[0]), \
                    step_nll, file_name)
            file_name = "MWA_B_STEP_KLDS_b{0:d}.png".format(i)
            utils.plot_stem(np.arange(step_kld.shape[0]), \
                    step_kld, file_name)
            file_name = "MWA_B_STEP_VFES_b{0:d}.png".format(i)
            utils.plot_stem(np.arange(step_kld.shape[0]).ravel(), \
                    (np.cumsum(step_kld.ravel())+step_nll.ravel()), file_name)
            Xb_tr = to_fX( Xtr[0:5000] )
            fe_terms = MSM.compute_fe_terms(Xb_tr, Xb_tr, 30)
            fe_nll = np.mean(fe_terms[0])
            fe_kld = np.mean(fe_terms[1])
            fe_joint = fe_nll + fe_kld
            joint_str = "    vfe-tr: {0:.4f}, nll: ({1:.4f}, {2:.4f}, {3:.4f}), kld: ({4:.4f}, {5:.4f}, {6:.4f})".format( \
                    fe_joint, fe_nll, np.min(fe_terms[0]), np.max(fe_terms[0]), fe_kld, np.min(fe_terms[1]), np.max(fe_terms[1]))
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            # file_name = "MWA_B_FREE_ENERGY_TR_b{0:d}.png".format(i)
            # utils.plot_scatter(fe_terms[1], fe_terms[0], file_name, \
            #         x_label='Posterior KLd', y_label='Negative Log-likelihood')
            # compute free-energy terms on validation samples
            fe_terms = MSM.compute_fe_terms(Xb_va, Xb_va, 30)
            fe_nll = np.mean(fe_terms[0])
            fe_kld = np.mean(fe_terms[1])
            fe_joint = fe_nll + fe_kld
            joint_str = "    vfe-va: {0:.4f}, nll: ({1:.4f}, {2:.4f}, {3:.4f}), kld: ({4:.4f}, {5:.4f}, {6:.4f})".format( \
                    fe_joint, fe_nll, np.min(fe_terms[0]), np.max(fe_terms[0]), fe_kld, np.min(fe_terms[1]), np.max(fe_terms[1]))
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            # file_name = "MWA_B_FREE_ENERGY_VA_b{0:d}.png".format(i)
            # utils.plot_scatter(fe_terms[1], fe_terms[0], file_name, \
            #         x_label='Posterior KLd', y_label='Negative Log-likelihood')

if __name__=="__main__":
    test_with_model_init()