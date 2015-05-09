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
from ClassModel import ClassModel
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
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm(dataset, as_shared=False, zero_mean=False)
    Xtr = to_fX(datasets[0][0])
    Xva = to_fX(datasets[1][0])
    Ytr = datasets[0][1]
    Yva = datasets[1][1]

    tr_samples = Xtr.shape[0]
    va_samples = Xva.shape[0]
    batch_size = 200

    BD = lambda ary: binarize_data(ary)

    #######################################
    # Setup some parameters for the model #
    #######################################
    obs_dim = Xtr.shape[1]
    z_dim = 64
    init_scale = 1.0

    # some InfNet instances to build the TwoStageModel from
    x_in = T.matrix('x_in')
    y_in = T.lvector('y_in')

    ###############
    # q_z_given_x #
    ###############
    params = {}
    shared_config = [obs_dim, 1000, 1000]
    top_config = [shared_config[-1], z_dim]
    params['shared_config'] = shared_config
    params['mu_config'] = top_config
    params['sigma_config'] = top_config
    params['activation'] = relu_actfun
    params['init_scale'] = init_scale
    params['lam_l2a'] = 0.0
    params['vis_drop'] = 0.2
    params['hid_drop'] = 0.5
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['build_theano_funcs'] = False
    q_z_given_x = InfNet(rng=rng, Xd=x_in, \
            params=params, shared_param_dicts=None)
    q_z_given_x.init_biases(0.2)


    ###########################################################
    # Define parameters for the ClassModel, and initialize it #
    ###########################################################
    print("Building the ClassModel...")
    CM = ClassModel(rng=rng, \
            x_in=x_in, y_in=y_in, \
            q_z_given_x=q_z_given_x, \
            class_count=10, \
            z_dim=z_dim, \
            use_samples=False)
    CM.set_drop_rate(0.5)
    CM.set_lam_nll(lam_nll=1.0)
    CM.set_lam_kld(lam_kld_q2p=1.0, lam_kld_p2q=0.0)
    CM.set_lam_l2w(lam_l2w=1e-5)

    ################################################################
    # Apply some updates, to check that they aren't totally broken #
    ################################################################
    out_file = open("CM_RESULTS.txt", 'wb')
    costs = [0. for i in range(10)]
    learn_rate = 0.0002
    momentum = 0.9
    batch_idx = np.arange(batch_size) + tr_samples
    for i in range(250000):
        scale = min(1.0, ((i+1) / 1000.0))
        if (((i + 1) % 10000) == 0):
            learn_rate = learn_rate * 0.95
        # get the indices of training samples for this batch update
        batch_idx += batch_size
        if (np.max(batch_idx) >= tr_samples):
            # we finished an "epoch", so we rejumble the training set
            Xtr, Ytr = row_shuffle(Xtr, Ytr)
            batch_idx = np.arange(batch_size)
        # set sgd and objective function hyperparams for this update
        CM.set_sgd_params(lr_1=scale*learn_rate, lr_2=scale*learn_rate, \
                          mom_1=scale*momentum, mom_2=0.99)
        # perform a minibatch update and record the cost for this batch
        Xi_tr = Xtr.take(batch_idx, axis=0)
        Yi_tr = Ytr.take(batch_idx, axis=0)
        result = CM.train_joint(Xi_tr, Yi_tr)
        costs = [(costs[j] + result[j]) for j in range(len(result)-1)]
        # output useful information about training progress
        if ((i % 500) == 0):
            costs = [(v / 500.0) for v in costs]
            str1 = "-- batch {0:d} --".format(i)
            str2 = "    joint_cost  : {0:.4f}".format(costs[0])
            str3 = "    nll_cost    : {0:.4f}".format(costs[1])
            str4 = "    kld_cost    : {0:.4f}".format(costs[2])
            str5 = "    reg_cost    : {0:.4f}".format(costs[3])
            joint_str = "\n".join([str1, str2, str3, str4, str5])
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            costs = [0.0 for v in costs]
        if (((i % 2000) == 0) or ((i < 10000) and ((i % 1000) == 0))):
            #####################################################
            # compute multi-sample estimates of the free-energy #
            #####################################################
            # training set...
            fe_terms = CM.compute_fe_terms(Xtr[0:2500],Ytr[0:2500], 30)
            fe_nll = np.mean(fe_terms[0])
            fe_kld = np.mean(fe_terms[1])
            fe_joint = fe_nll + fe_kld
            joint_str = "    vfe-tr: {0:.4f}, nll: ({1:.4f}, {2:.4f}, {3:.4f}), kld: ({4:.4f}, {5:.4f}, {6:.4f})".format( \
                    fe_joint, fe_nll, np.min(fe_terms[0]), np.max(fe_terms[0]), fe_kld, np.min(fe_terms[1]), np.max(fe_terms[1]))
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            # validation set...
            Xva, Yva = row_shuffle(Xva, Yva)
            fe_terms = CM.compute_fe_terms(Xva[0:2500], Yva[0:2500], 30)
            fe_nll = np.mean(fe_terms[0])
            fe_kld = np.mean(fe_terms[1])
            fe_joint = fe_nll + fe_kld
            joint_str = "    vfe-va: {0:.4f}, nll: ({1:.4f}, {2:.4f}, {3:.4f}), kld: ({4:.4f}, {5:.4f}, {6:.4f})".format( \
                    fe_joint, fe_nll, np.min(fe_terms[0]), np.max(fe_terms[0]), fe_kld, np.min(fe_terms[1]), np.max(fe_terms[1]))
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            ##########################################################
            # compute multi-sample estimates of classification error #
            ##########################################################
            # training set...
            va_error, va_preds = CM.class_error(Xtr[:2500], Ytr[:2500], samples=30)
            joint_str = "    tr-class-error: {0:.4f}".format(va_error)
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            # validation set...
            va_error, va_preds = CM.class_error(Xva[:2500], Yva[:2500], samples=30)
            joint_str = "    va-class-error: {0:.4f}".format(va_error)
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()

if __name__=="__main__":
    test_with_model_init()