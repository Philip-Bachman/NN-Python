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
from GPSImputer import GPSImputer
from load_data import load_udm, load_udm_ss, load_mnist, load_binarized_mnist
from HelperFuncs import sample_masks, sample_patch_masks, posterior_klds, \
                        collect_obs_costs

import utils


def construct_masked_data(xi, data_mean=None):
    """
    Construct randomly masked data from xi.
    """
    if data_mean is None:
        data_mean = np.zeros((xi.shape[1],))
    xo = xi.copy()
    xm_rand = sample_masks(xi, drop_prob=0.0)
    xm_patch = sample_patch_masks(xi, (28,28), (14,14))
    xm = xm_rand * xm_patch
    xi = (xm * xi) + ((1.0 - xm) * data_mean)
    xi = to_fX(xi)
    xo = to_fX(xo)
    xm = to_fX(xm)
    return xi, xo, xm

########################################
########################################
## TEST WITH MODEL-BASED INITIAL STEP ##
########################################
########################################

def test_mnist():
    ##########################
    # Get some training data #
    ##########################
    rng = np.random.RandomState(1234)
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm(dataset, as_shared=False, zero_mean=False)
    Xtr = datasets[0][0]
    Xva = datasets[1][0]
    tr_samples = Xtr.shape[0]
    va_samples = Xva.shape[0]
    batch_size = 500
    batch_reps = 1
    data_mean = to_fX( 0.0 * np.mean(Xtr, axis=0) )

    ############################################################
    # Setup some parameters for the Iterative Refinement Model #
    ############################################################
    obs_dim = Xtr.shape[1]
    z_dim = 100
    imp_steps = 4
    init_scale = 1.0
    x_type = 'bernoulli'

    x_in_sym = T.matrix('x_in_sym')
    x_out_sym = T.matrix('x_out_sym')
    x_mask_sym = T.matrix('x_mask_sym')

    #################
    # p_zi_given_xi #
    #################
    params = {}
    shared_config = [obs_dim, 800, 800]
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
    p_xip1_given_zi = InfNet(rng=rng, Xd=x_in_sym, \
            params=params, shared_param_dicts=None)
    p_xip1_given_zi.init_biases(0.2)
    ###################
    # q_zi_given_x_xi #
    ###################
    params = {}
    shared_config = [(obs_dim + obs_dim), 800, 800]
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
            params=gpsi_params, \
            shared_param_dicts=None)

    ################################################################
    # Apply some updates, to check that they aren't totally broken #
    ################################################################
    out_file = open("GPSIX_RESULTS.txt", 'wb')
    costs = [0. for i in range(10)]
    learn_rate = 0.0005
    momentum = 0.5
    batch_idx = np.arange(batch_size) + tr_samples
    for i in range(250000):
        scale = min(1.0, ((i+1) / 2000.0))
        if (((i + 1) % 10000) == 0):
            learn_rate = learn_rate * 0.95
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
        GPSI.set_lam_kld(lam_kld_p=0.25, lam_kld_q=0.75)
        GPSI.set_lam_ent(lam_ent_p=0.00, lam_ent_q=0.01)
        GPSI.set_lam_l2w(1e-4)
        # perform a minibatch update and record the cost for this batch
        xb = to_fX( Xtr.take(batch_idx, axis=0) )
        xi, xo, xm = construct_masked_data(xb, data_mean)
        result = GPSI.train_joint(xi, xo, xm, batch_reps)
        batch_costs = result[-1] # get the per-input costs
        obs_costs = collect_obs_costs(batch_costs, batch_reps)
        costs = [(costs[j] + result[j]) for j in range(len(result)-1)]
        if ((i % 500) == 0):
            costs = [(v / 500.0) for v in costs]
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
        if (((i % 2000) == 0) or ((i < 10000) and ((i % 1000) == 0))):
            # Get some validation samples for evaluating model performance
            Xva = row_shuffle(Xva)
            xb = to_fX( Xva[0:50] )
            xi, xo, xm = construct_masked_data(xb, data_mean)
            xi = np.repeat(xi, 4, axis=0)
            xo = np.repeat(xo, 4, axis=0)
            xm = np.repeat(xm, 4, axis=0)
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
            file_name = "GPSIX_SAMPLES_NG_b{0:d}.png".format(i)
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
            file_name = "GPSIX_SAMPLES_YG_b{0:d}.png".format(i)
            utils.visualize_samples(seq_samps, file_name, num_rows=20)

if __name__=="__main__":
    test_mnist()