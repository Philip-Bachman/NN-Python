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
from MultiStageModelSS2 import MultiStageModelSS
from load_data import load_udm, load_udm_ss, load_mnist, load_binarized_mnist
from HelperFuncs import collect_obs_costs
import utils

def make_class_groups(X, Y):
    """
    Make a dict of np matrices, where the dict keys are class labels and
    the dict items are matrices of examples from that class.
    """
    class_groups = {}
    class_keys = np.unique(Y)
    for k in class_keys:
        k_idx = (Y == k)
        class_groups[k] = X[k_idx]
    return class_groups

def sample_class_groups(Y, class_groups):
    """
    For each class key in Y, sample an in group example and an out of
    group example.
    """
    class_labels = np.unique(np.asarray(class_groups.keys()))
    sample_count = Y.shape[0]
    obs_dim = class_groups[Y[0]].shape[1]
    Xp = np.zeros((sample_count, obs_dim))
    Xn = np.zeros((sample_count, obs_dim))
    for i in range(sample_count):
        # sample in-group example
        xp = class_groups[Y[i]]
        p_idx = npr.randint(0, high=xp.shape[0])
        Xp[i] = xp[p_idx]
        # sample out-group example
        npr.shuffle(class_labels)
        if class_labels[0] != Y[i]:
            xn = class_groups[class_labels[0]]
        else:
            xn = class_groups[class_labels[1]]
        n_idx = npr.randint(0, high=xn.shape[0])
        Xn[i] = xn[n_idx]
    Xp = to_fX(Xp)
    Xn = to_fX(Xn)
    return Xp, Xn

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
    datasets = load_udm(dataset, as_shared=False, zero_mean=False)
    Xtr = to_fX(datasets[0][0])
    Xva = to_fX(datasets[1][0])
    Ytr = datasets[0][1]
    Yva = datasets[1][1]
    Xtr_class_groups = make_class_groups(Xtr, Ytr)

    tr_samples = Xtr.shape[0]
    va_samples = Xva.shape[0]
    batch_size = 300

    BD = lambda ary: binarize_data(ary)

    ############################################################
    # Setup some parameters for the Iterative Refinement Model #
    ############################################################
    obs_dim = Xtr.shape[1]
    z_dim = 30
    h_dim = 100
    rnn_dim = 2
    ir_steps = 3
    init_scale = 1.0
    
    x_type = 'bernoulli'

    # some InfNet instances to build the TwoStageModel from
    x_in = T.matrix('x_in')
    x_pos = T.matrix('x_pos')
    x_neg = T.matrix('x_neg')
    y_in = T.lvector('y_in')

    #################
    # p_hi_given_si #
    #################
    params = {}
    shared_config = [(obs_dim + rnn_dim), 500, 500]
    top_config = [shared_config[-1], h_dim]
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
    p_hi_given_si = InfNet(rng=rng, Xd=x_in, \
            params=params, shared_param_dicts=None)
    p_hi_given_si.init_biases(0.2)
    ######################
    # p_sip1_given_si_hi #
    ######################
    params = {}
    shared_config = [(h_dim + obs_dim), 500, 500]
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
    p_sip1_given_si_hi = InfNet(rng=rng, Xd=x_in, \
            params=params, shared_param_dicts=None)
    p_sip1_given_si_hi.init_biases(0.2)
    ################
    # p_s0_given_z #
    ################
    params = {}
    shared_config = [z_dim, 500, 500]
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
    p_s0_given_z = InfNet(rng=rng, Xd=x_in, \
            params=params, shared_param_dicts=None)
    p_s0_given_z.init_biases(0.2)
    ###############
    # q_z_given_x #
    ###############
    params = {}
    shared_config = [obs_dim, (400, 4), (400, 4)]
    top_config = [shared_config[-1], (z_dim + rnn_dim)]
    params['shared_config'] = shared_config
    params['mu_config'] = top_config
    params['sigma_config'] = top_config
    params['activation'] = relu_actfun
    params['init_scale'] = init_scale
    params['lam_l2a'] = 0.0
    params['vis_drop'] = 0.20
    params['hid_drop'] = 0.25
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    q_z_given_x = InfNet(rng=rng, Xd=x_in, \
            params=params, shared_param_dicts=None)
    q_z_given_x.init_biases(0.0)
    ###################
    # q_hi_given_x_si #
    ###################
    params = {}
    shared_config = [(obs_dim + obs_dim + rnn_dim), 800, 800]
    top_config = [shared_config[-1], h_dim]
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
    q_hi_given_x_si = InfNet(rng=rng, Xd=x_in, \
            params=params, shared_param_dicts=None)
    q_hi_given_x_si.init_biases(0.2)


    ################################################################
    # Define parameters for the MultiStageModel, and initialize it #
    ################################################################
    print("Building the MultiStageModel...")
    msm_params = {}
    msm_params['x_type'] = x_type
    msm_params['obs_transform'] = 'sigmoid'
    MSM = MultiStageModelSS(rng=rng, \
            x_in=x_in, x_pos=x_pos, x_neg=x_neg, y_in=y_in, \
            p_s0_given_z=p_s0_given_z, \
            p_hi_given_si=p_hi_given_si, \
            p_sip1_given_si_hi=p_sip1_given_si_hi, \
            q_z_given_x=q_z_given_x, \
            q_hi_given_x_si=q_hi_given_x_si, \
            class_count=10, use_rnn=False, \
            obs_dim=obs_dim, rnn_dim=rnn_dim, z_dim=z_dim, h_dim=h_dim, \
            ir_steps=ir_steps, params=msm_params)
    MSM.set_vfe_margin(vfe_margin=10.0)
    MSM.set_lam_class(lam_class=20.0)
    MSM.set_lam_nll(lam_nll=1.0)
    MSM.set_lam_kld(lam_kld_z=1.0, lam_kld_q2p=0.9, \
                    lam_kld_p2q=0.1)
    MSM.set_lam_l2w(1e-4)

    ################################################################
    # Apply some updates, to check that they aren't totally broken #
    ################################################################
    out_file = open("MSS_RESULTS.txt", 'wb')
    costs = [0. for i in range(10)]
    learn_rate = 0.0002
    momentum = 0.5
    batch_idx = np.arange(batch_size) + tr_samples
    for i in range(250000):
        scale = min(1.0, ((i+1) / 2000.0))
        if (((i + 1) % 10000) == 0):
            learn_rate = learn_rate * 0.95
        if (i > 20000):
            momentum = 0.90
        else:
            momentum = 0.50
        # get the indices of training samples for this batch update
        batch_idx += batch_size
        if (np.max(batch_idx) >= tr_samples):
            # we finished an "epoch", so we rejumble the training set
            Xtr, Ytr = row_shuffle(Xtr, Ytr)
            batch_idx = np.arange(batch_size)
        # set sgd and objective function hyperparams for this update
        MSM.set_sgd_params(lr_1=scale*learn_rate, lr_2=scale*learn_rate, \
                           mom_1=scale*momentum, mom_2=0.99)
        MSM.set_train_switch(1.0)
        MSM.set_drop_rate(0.0)
        MSM.q_hi_given_x_si.set_bias_noise(0.0)
        MSM.p_hi_given_si.set_bias_noise(0.0)
        MSM.p_sip1_given_si_hi.set_bias_noise(0.0)
        # perform a minibatch update and record the cost for this batch
        Xi_tr = Xtr.take(batch_idx, axis=0)
        Yi_tr = Ytr.take(batch_idx, axis=0)
        Xp_tr, Xn_tr = sample_class_groups(Yi_tr, Xtr_class_groups)
        result = MSM.train_joint(Xi_tr, Xp_tr, Xn_tr, Yi_tr)
        costs = [(costs[j] + result[j]) for j in range(len(result)-1)]
        # output useful information about training progress
        if ((i % 500) == 0):
            costs = [(v / 500.0) for v in costs]
            str1 = "-- batch {0:d} --".format(i)
            str2 = "    joint_cost  : {0:.4f}".format(costs[0])
            str3 = "    class_cost  : {0:.4f}".format(costs[1])
            str4 = "    margin_cost : {0:.4f}".format(costs[2])
            str5 = "    nll_cost    : {0:.4f}".format(costs[3])
            str6 = "    kld_cost    : {0:.4f}".format(costs[4])
            str7 = "    reg_cost    : {0:.4f}".format(costs[5])
            joint_str = "\n".join([str1, str2, str3, str4, str5, str6, str7])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            costs = [0.0 for v in costs]
        if (((i % 2000) == 0) or ((i < 10000) and ((i % 1000) == 0))):
            MSM.set_drop_rate(0.0)
            MSM.q_hi_given_x_si.set_bias_noise(0.0)
            MSM.p_hi_given_si.set_bias_noise(0.0)
            MSM.p_sip1_given_si_hi.set_bias_noise(0.0)
            # Get some validation samples for computing diagnostics
            Xva, Yva = row_shuffle(Xva, Yva)
            Xb_va = Xva[0:2500]
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
            file_name = "MSS_B_SAMPLES_IND_b{0:d}.png".format(i)
            utils.visualize_samples(seq_samps, file_name, num_rows=20)
            # draw some conditional random samples from the model
            samp_count = 200
            #Xs = np.vstack((Xi_tr[0:(samp_count/4)], Xb_va[0:(samp_count/4)]))
            Xs = Xb_va[0:(samp_count/4)] # only use validation set samples
            Xs = np.repeat(Xs, 4, axis=0)
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
            file_name = "MSS_B_SAMPLES_CND_UD_b{0:d}.png".format(i)
            utils.visualize_samples(seq_samps, file_name, num_rows=20)
            file_name = "MSS_A_GEN_GEN_WEIGHTS_b{0:d}.png".format(i)
            W = MSM.gen_gen_weights.get_value(borrow=False)
            utils.visualize_samples(W[:,:obs_dim], file_name, num_rows=20)
            file_name = "MSS_A_GEN_INF_WEIGHTS_b{0:d}.png".format(i)
            W = MSM.gen_inf_weights.get_value(borrow=False).T
            utils.visualize_samples(W[:,:obs_dim], file_name, num_rows=20)
            # compute information about posterior KLds on validation set
            raw_costs = MSM.compute_raw_costs(Xb_va, Xb_va)
            init_nll, init_kld, q2p_kld, p2q_kld, step_nll, step_kld = raw_costs
            step_nll[0] = step_nll[1] # scale of first NLL is overwhemling
            file_name = "MSS_B_H0_KLDS_b{0:d}.png".format(i)
            utils.plot_stem(np.arange(init_kld.shape[1]), \
                    np.mean(init_kld, axis=0), file_name)
            file_name = "MSS_B_HI_Q2P_KLDS_b{0:d}.png".format(i)
            utils.plot_stem(np.arange(q2p_kld.shape[1]), \
                    np.mean(q2p_kld, axis=0), file_name)
            file_name = "MSS_B_HI_P2Q_KLDS_b{0:d}.png".format(i)
            utils.plot_stem(np.arange(p2q_kld.shape[1]), \
                    np.mean(p2q_kld, axis=0), file_name)
            Xb_tr = to_fX( Xtr[0:2500] )
            fe_terms = MSM.compute_fe_terms(Xb_tr, Xb_tr, 30)
            fe_nll = np.mean(fe_terms[0])
            fe_kld = np.mean(fe_terms[1])
            fe_joint = fe_nll + fe_kld
            joint_str = "    vfe-tr: {0:.4f}, nll: ({1:.4f}, {2:.4f}, {3:.4f}), kld: ({4:.4f}, {5:.4f}, {6:.4f})".format( \
                    fe_joint, fe_nll, np.min(fe_terms[0]), np.max(fe_terms[0]), fe_kld, np.min(fe_terms[1]), np.max(fe_terms[1]))
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
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
            # compute multi-sample estimate of classification error
            va_error, va_preds = MSM.class_error(Xva[:2000], Yva[:2000], samples=30)
            joint_str = "    va-class-error: {0:.4f}".format(va_error)
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()


if __name__=="__main__":
    test_with_model_init()