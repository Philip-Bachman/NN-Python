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
from TwoStageModel import TwoStageModel
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
    batch_size = 200
    batch_reps = 1

    ############################################################
    # Setup some parameters for the Iterative Refinement Model #
    ############################################################
    x_dim = Xtr.shape[1]
    z_dim = 20
    h_dim = 50
    s_dim = 50
    init_scale = 1.0
    
    x_type = 'bernoulli'

    # some InfNet instances to build the TwoStageModel from
    x_in_sym = T.matrix('x_in_sym')
    x_out_sym = T.matrix('x_out_sym')

    ###############
    # p_h_given_s #
    ###############
    params = {}
    shared_config = [s_dim, 250, 250]
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
    p_h_given_s = InfNet(rng=rng, Xd=x_in_sym, \
            params=params, shared_param_dicts=None)
    p_h_given_s.init_biases(0.2)
    #################
    # p_x_given_s_h #
    #################
    params = {}
    shared_config = [(s_dim + h_dim), 250, 250]
    top_config = [shared_config[-1], x_dim]
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
    p_x_given_s_h = InfNet(rng=rng, Xd=x_in_sym, \
            params=params, shared_param_dicts=None)
    p_x_given_s_h.init_biases(0.2)
    ###############
    # p_s_given_z #
    ###############
    params = {}
    shared_config = [z_dim, 250]
    top_config = [shared_config[-1], s_dim]
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
    p_s_given_z = InfNet(rng=rng, Xd=x_in_sym, \
            params=params, shared_param_dicts=None)
    p_s_given_z.init_biases(0.2)
    ###############
    # q_z_given_x #
    ###############
    params = {}
    shared_config = [x_dim, 250, 250]
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
    q_z_given_x.init_biases(0.2)
    #################
    # q_h_given_x_s #
    #################
    params = {}
    shared_config = [(x_dim + s_dim), 500, 500]
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
    q_h_given_x_s = InfNet(rng=rng, Xd=x_in_sym, \
            params=params, shared_param_dicts=None)
    q_h_given_x_s.init_biases(0.2)


    ##############################################################
    # Define parameters for the TwoStageModel, and initialize it #
    ##############################################################
    print("Building the TwoStageModel...")
    msm_params = {}
    msm_params['x_type'] = x_type
    msm_params['obs_transform'] = 'sigmoid'
    TSM = TwoStageModel(rng=rng, \
            x_in=x_in_sym, x_out=x_out_sym, \
            p_s_given_z=p_s_given_z, \
            p_h_given_s=p_h_given_s, \
            p_x_given_s_h=p_x_given_s_h, \
            q_z_given_x=q_z_given_x, \
            q_h_given_x_s=q_h_given_x_s, \
            x_dim=x_dim, \
            z_dim=z_dim, s_dim=s_dim, h_dim=h_dim, \
            params=msm_params)

    ################################################################
    # Apply some updates, to check that they aren't totally broken #
    ################################################################
    out_file = open("TSM_A_RESULTS.txt", 'wb')
    costs = [0. for i in range(10)]
    learn_rate = 0.0003
    momentum = 0.5
    batch_idx = np.arange(batch_size) + tr_samples
    for i in range(250000):
        scale = min(1.0, ((i+1) / 3000.0))
        if (((i + 1) % 10000) == 0):
            learn_rate = learn_rate * 0.95
        if (i > 50000):
            momentum = 0.90
        else:
            momentum = 0.50
        # get the indices of training samples for this batch update
        batch_idx += batch_size
        if (np.max(batch_idx) >= tr_samples):
            # we finished an "epoch", so we rejumble the training set
            Xtr = row_shuffle(Xtr)
            batch_idx = np.arange(batch_size)
        # train on the training set
        lam_kld = 1.0
        # set sgd and objective function hyperparams for this update
        TSM.set_sgd_params(lr_1=scale*learn_rate, lr_2=scale*learn_rate, \
                mom_1=scale*momentum, mom_2=0.99)
        TSM.set_train_switch(1.0)
        TSM.set_lam_nll(lam_nll=1.0)
        TSM.set_lam_kld(lam_kld_z=1.0, lam_kld_q2p=0.8, lam_kld_p2q=0.2)
        TSM.set_lam_kld_l1l2(lam_kld_l1l2=scale)
        TSM.set_lam_l2w(1e-4)
        TSM.set_drop_rate(0.0)
        TSM.q_h_given_x_s.set_bias_noise(0.0)
        TSM.p_h_given_s.set_bias_noise(0.0)
        TSM.p_x_given_s_h.set_bias_noise(0.0)
        # perform a minibatch update and record the cost for this batch
        Xb_tr = to_fX( Xtr.take(batch_idx, axis=0) )
        result = TSM.train_joint(Xb_tr, Xb_tr, batch_reps)
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
            TSM.set_drop_rate(0.0)
            TSM.q_h_given_x_s.set_bias_noise(0.0)
            TSM.p_h_given_s.set_bias_noise(0.0)
            TSM.p_x_given_s_h.set_bias_noise(0.0)
            # Get some validation samples for computing diagnostics
            Xva = row_shuffle(Xva)
            Xb_va = to_fX( Xva[0:2000] )
            # draw some independent random samples from the model
            samp_count = 500
            model_samps = TSM.sample_from_prior(samp_count)
            file_name = "TSM_A_SAMPLES_IND_b{0:d}.png".format(i)
            utils.visualize_samples(model_samps, file_name, num_rows=20)
            Xb_tr = to_fX( Xtr[0:2000] )
            fe_terms = TSM.compute_fe_terms(Xb_tr, Xb_tr, 30)
            fe_nll = np.mean(fe_terms[0])
            fe_kld = np.mean(fe_terms[1])
            fe_joint = fe_nll + fe_kld
            joint_str = "    vfe-tr: {0:.4f}, nll: ({1:.4f}, {2:.4f}, {3:.4f}), kld: ({4:.4f}, {5:.4f}, {6:.4f})".format( \
                    fe_joint, fe_nll, np.min(fe_terms[0]), np.max(fe_terms[0]), fe_kld, np.min(fe_terms[1]), np.max(fe_terms[1]))
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            fe_terms = TSM.compute_fe_terms(Xb_va, Xb_va, 30)
            fe_nll = np.mean(fe_terms[0])
            fe_kld = np.mean(fe_terms[1])
            fe_joint = fe_nll + fe_kld
            joint_str = "    vfe-va: {0:.4f}, nll: ({1:.4f}, {2:.4f}, {3:.4f}), kld: ({4:.4f}, {5:.4f}, {6:.4f})".format( \
                    fe_joint, fe_nll, np.min(fe_terms[0]), np.max(fe_terms[0]), fe_kld, np.min(fe_terms[1]), np.max(fe_terms[1]))
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()

if __name__=="__main__":
    test_with_model_init()