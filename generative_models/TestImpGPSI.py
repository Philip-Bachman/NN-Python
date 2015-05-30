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
from GPSImputer import GPSImputer, load_gpsimputer_from_file
from load_data import load_udm, load_tfd, load_svhn_gray
from HelperFuncs import construct_masked_data, shift_and_scale_into_01, \
                        row_shuffle, to_fX

RESULT_PATH = "IMP_MNIST_GPSI/"

###############################
###############################
## TEST GPS IMPUTER ON MNIST ##
###############################
###############################

def test_mnist(step_type='add',
               occ_dim=15,
               drop_prob=0.0):
    #########################################
    # Format the result tag more thoroughly #
    #########################################
    dp_int = int(100.0 * drop_prob)
    result_tag = "{}GPSI_OD{}_DP{}_{}_NA".format(RESULT_PATH, occ_dim, dp_int, step_type)

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
    batch_size = 250
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
    gpsi_params['step_type'] = step_type
    gpsi_params['x_type'] = 'bernoulli'
    gpsi_params['obs_transform'] = 'sigmoid'
    GPSI = GPSImputer(rng=rng, 
            x_in=x_in_sym, x_out=x_out_sym, x_mask=x_mask_sym, \
            p_zi_given_xi=p_zi_given_xi, \
            p_xip1_given_zi=p_xip1_given_zi, \
            q_zi_given_x_xi=q_zi_given_x_xi, \
            params=gpsi_params, \
            shared_param_dicts=None)

    # # test model saving
    # print("Testing model save to file...")
    # GPSI.save_to_file("AAA_GPSI_SAVE_TEST.pkl")
    # # test model loading
    # print("Testing model load from file...")
    # GPSI = load_gpsimputer_from_file(f_name="AAA_GPSI_SAVE_TEST.pkl", rng=rng)

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
        GPSI.set_sgd_params(lr=scale*learn_rate, \
                            mom_1=scale*momentum, mom_2=0.98)
        GPSI.set_train_switch(1.0)
        GPSI.set_lam_nll(lam_nll=1.0)
        GPSI.set_lam_kld(lam_kld_p=0.1, lam_kld_q=0.9)
        GPSI.set_lam_l2w(1e-4)
        # perform a minibatch update and record the cost for this batch
        xb = to_fX( Xtr.take(batch_idx, axis=0) )
        xi, xo, xm = construct_masked_data(xb, drop_prob=drop_prob, \
                                        occ_dim=occ_dim, data_mean=data_mean)
        result = GPSI.train_joint(xi, xo, xm, batch_reps)
        # do diagnostics and general training tracking
        costs = [(costs[j] + result[j]) for j in range(len(result)-1)]
        if ((i % 250) == 0):
            costs = [(v / 250.0) for v in costs]
            str1 = "-- batch {0:d} --".format(i)
            str2 = "    joint_cost: {0:.4f}".format(costs[0])
            str3 = "    nll_bound : {0:.4f}".format(costs[1])
            str4 = "    nll_cost  : {0:.4f}".format(costs[2])
            str5 = "    kld_cost  : {0:.4f}".format(costs[3])
            str6 = "    reg_cost  : {0:.4f}".format(costs[4])
            joint_str = "\n".join([str1, str2, str3, str4, str5, str6])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            costs = [0.0 for v in costs]
        if ((i % 1000) == 0):
            Xva = row_shuffle(Xva)
            # record an estimate of performance on the test set
            xi, xo, xm = construct_masked_data(Xva[0:5000], drop_prob=drop_prob, \
                                               occ_dim=occ_dim, data_mean=data_mean)
            nll, kld = GPSI.compute_fe_terms(xi, xo, xm, sample_count=10)
            vfe = np.mean(nll) + np.mean(kld)
            str1 = "    va_nll_bound : {}".format(vfe)
            str2 = "    va_nll_term  : {}".format(np.mean(nll))
            str3 = "    va_kld_q2p   : {}".format(np.mean(kld))
            joint_str = "\n".join([str1, str2, str3])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
        if ((i % 5000) == 0):
            GPSI.save_to_file("{}_PARAMS.pkl".format(result_tag))
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
            file_name = "{0:s}_samples_ng_b{1:d}.png".format(result_tag, i)
            utils.visualize_samples(seq_samps, file_name, num_rows=20)
            # get visualizations of policy parameters
            file_name = "{0:s}_gen_gen_weights_b{1:d}.png".format(result_tag, i)
            W = GPSI.gen_gen_weights.get_value(borrow=False)
            utils.visualize_samples(W[:,:obs_dim], file_name, num_rows=20)
            file_name = "{0:s}_gen_inf_weights_b{1:d}.png".format(result_tag, i)
            W = GPSI.gen_inf_weights.get_value(borrow=False).T
            utils.visualize_samples(W[:,:obs_dim], file_name, num_rows=20)

#################################
#################################
## CHECK MNIST IMPUTER RESULTS ##
#################################
#################################

def test_mnist_results(step_type='add',
                       occ_dim=15,
                       drop_prob=0.0):
    #########################################
    # Format the result tag more thoroughly #
    #########################################
    dp_int = int(100.0 * drop_prob)
    result_tag = "{}GPSI_OD{}_DP{}_{}_NA".format(RESULT_PATH, occ_dim, dp_int, step_type)

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
    batch_size = 250
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

    x_in_sym = T.matrix('x_in_sym')
    x_out_sym = T.matrix('x_out_sym')
    x_mask_sym = T.matrix('x_mask_sym')

    # Load parameters from a previously trained model
    print("Testing model load from file...")
    GPSI = load_gpsimputer_from_file(f_name="{}_PARAMS.pkl".format(result_tag), \
                                     rng=rng)

    ################################################################
    # Apply some updates, to check that they aren't totally broken #
    ################################################################
    log_name = "{}_FINAL_RESULTS.txt".format(result_tag)
    out_file = open(log_name, 'wb')

    Xva = row_shuffle(Xva)
    # record an estimate of performance on the test set
    str0 = "GUIDED SAMPLE BOUND:"
    print(str0)
    xi, xo, xm = construct_masked_data(Xva[:5000], drop_prob=drop_prob, \
                                       occ_dim=occ_dim, data_mean=data_mean)
    nll_0, kld_0 = GPSI.compute_fe_terms(xi, xo, xm, sample_count=10, \
                                         use_guide_policy=True)
    xi, xo, xm = construct_masked_data(Xva[5000:], drop_prob=drop_prob, \
                                       occ_dim=occ_dim, data_mean=data_mean)
    nll_1, kld_1 = GPSI.compute_fe_terms(xi, xo, xm, sample_count=10, \
                                         use_guide_policy=True)
    nll = np.concatenate((nll_0, nll_1))
    kld = np.concatenate((kld_0, kld_1))
    vfe = np.mean(nll) + np.mean(kld)
    str1 = "    va_nll_bound : {}".format(vfe)
    str2 = "    va_nll_term  : {}".format(np.mean(nll))
    str3 = "    va_kld_q2p   : {}".format(np.mean(kld))
    joint_str = "\n".join([str0, str1, str2, str3])
    print(joint_str)
    out_file.write(joint_str+"\n")
    out_file.flush()
    # record an estimate of performance on the test set
    str0 = "UNGUIDED SAMPLE BOUND:"
    print(str0)
    xi, xo, xm = construct_masked_data(Xva[:5000], drop_prob=drop_prob, \
                                       occ_dim=occ_dim, data_mean=data_mean)
    nll_0, kld_0 = GPSI.compute_fe_terms(xi, xo, xm, sample_count=10, \
                                         use_guide_policy=False)
    xi, xo, xm = construct_masked_data(Xva[5000:], drop_prob=drop_prob, \
                                       occ_dim=occ_dim, data_mean=data_mean)
    nll_1, kld_1 = GPSI.compute_fe_terms(xi, xo, xm, sample_count=10, \
                                         use_guide_policy=False)
    nll = np.concatenate((nll_0, nll_1))
    kld = np.concatenate((kld_0, kld_1))
    str1 = "    va_nll_bound : {}".format(np.mean(nll))
    str2 = "    va_nll_term  : {}".format(np.mean(nll))
    str3 = "    va_kld_q2p   : {}".format(np.mean(kld))
    joint_str = "\n".join([str0, str1, str2, str3])
    print(joint_str)
    out_file.write(joint_str+"\n")
    out_file.flush()

if __name__=="__main__":
    #########
    # MNIST #
    #########
    # TRAINING
    #test_mnist(step_type='jump', occ_dim=14, drop_prob=0.0)
    #test_mnist(step_type='jump', occ_dim=16, drop_prob=0.0)
    #test_mnist(step_type='jump', occ_dim=0, drop_prob=0.6)
    #test_mnist(step_type='jump', occ_dim=0, drop_prob=0.8)
    #test_mnist(step_type='add', occ_dim=14, drop_prob=0.0)
    #test_mnist(step_type='add', occ_dim=16, drop_prob=0.0)
    #test_mnist(step_type='add', occ_dim=0, drop_prob=0.6)
    #test_mnist(step_type='add', occ_dim=0, drop_prob=0.8)

    # RESULTS
    test_mnist_results(step_type='jump', occ_dim=14, drop_prob=0.0)
    test_mnist_results(step_type='jump', occ_dim=16, drop_prob=0.0)
    test_mnist_results(step_type='jump', occ_dim=0, drop_prob=0.6)
    test_mnist_results(step_type='jump', occ_dim=0, drop_prob=0.7)
    test_mnist_results(step_type='jump', occ_dim=0, drop_prob=0.8)
    test_mnist_results(step_type='jump', occ_dim=0, drop_prob=0.9)
    test_mnist_results(step_type='add', occ_dim=14, drop_prob=0.0)
    test_mnist_results(step_type='add', occ_dim=16, drop_prob=0.0)
    test_mnist_results(step_type='add', occ_dim=0, drop_prob=0.6)
    test_mnist_results(step_type='add', occ_dim=0, drop_prob=0.7)
    test_mnist_results(step_type='add', occ_dim=0, drop_prob=0.8)
    test_mnist_results(step_type='add', occ_dim=0, drop_prob=0.9)