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
from InfNet import InfNet
from MultiStageDRAW import MultiStageDRAW
from MSDUtils import SimpleMLP, SimpleReader, SimpleWriter, \
                     SimpleInfNet, SimpleLSTM
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
    z_dim = 20                 # initialization latent variables
    h_dim = 100                # per-step latent variables
    obs_dim = Xtr.shape[1]     # input data
    rnn_dim = 256              # encoder/decoder LSTM state
    mix_dim = 50               # mixture state
    read_dim = 2 * obs_dim     # reader output
    x_type = 'bernoulli'       # output distribution type

    # some InfNet instances to build the TwoStageModel from
    x_in_sym = T.matrix('x_in_sym')
    x_out_sym = T.matrix('x_out_sym')

    ###############
    # q_z_given_x #
    ###############
    params = {}
    shared_config = [obs_dim, 500, 500]
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
    q_z_given_x = InfNet(rng=rng, Xd=x_in_sym, \
            params=params, shared_param_dicts=None)
    q_z_given_x.init_biases(0.2)

    #################
    # simple models #
    #################
    p_writer = SimpleWriter(rnn_dim, obs_dim, W_scale=0.01)
    q_reader = SimpleReader(obs_dim, rnn_dim, W_scale=0.01)
    p_hi_given_sim1_dec = SimpleInfNet(rng, rnn_dim, h_dim, W_scale=0.01)
    q_hi_given_si_enc = SimpleInfNet(rng, rnn_dim, h_dim, W_scale=0.01)

    ################################################################
    # Define parameters for the MultiStageModel, and initialize it #
    ################################################################
    print("Building the MultiStageModel...")
    msm_params = {}
    msm_params['x_type'] = x_type
    msm_params['obs_transform'] = 'sigmoid'
    MSM_TR = MultiStageDRAW(rng=rng, x_in=x_in_sym, x_out=x_out_sym, \
            q_z_given_x=q_z_given_x, init_from_z=True, \
            p_hi_given_sim1_dec=p_hi_given_sim1_dec, \
            q_hi_given_si_enc=q_hi_given_si_enc, \
            p_writer=p_writer, q_reader=q_reader, \
            z_dim=z_dim, h_dim=h_dim, \
            obs_dim=obs_dim, rnn_dim=rnn_dim, \
            mix_dim=mix_dim, read_dim=read_dim, \
            ir_steps=5, init_scale=0.01, params=msm_params)
    obs_mean = (0.9 * np.mean(Xtr, axis=0)) + 0.05
    obs_mean_logit = np.log(obs_mean / (1.0 - obs_mean))
    MSM_TR.set_input_bias(-obs_mean)
    MSM_TR.set_obs_bias(0.2*obs_mean_logit)
    MSM_VA = None

    ################################################################
    # Apply some updates, to check that they aren't totally broken #
    ################################################################
    out_file = open("MZ_RESULTS.txt", 'wb')
    costs = [0. for i in range(10)]
    learn_rate = 0.001
    momentum = 0.5
    tr_idx = np.arange(batch_size) + tr_samples
    for i in range(500000):
        scale = min(1.0, ((i+1) / 5000.0))
        l1l2_weight = min(1.0, ((i+1) / 2500.0))
        if (((i + 1) % 10000) == 0):
            learn_rate = learn_rate * 0.95
        if (i > 150000):
            momentum = 0.90
        elif (i > 100000):
            momentum = 0.70
        elif (i > 50000):
            momentum = 0.60
        else:
            momentum = 0.50
        # get the indices of training samples for this batch update
        tr_idx += batch_size
        if (np.max(tr_idx) >= tr_samples):
            # we finished an "epoch", so we rejumble the training set
            Xtr = row_shuffle(Xtr)
            tr_idx = np.arange(batch_size)
        # train on the training set
        MSM = MSM_TR
        MSM_VA = None
        lr_1 = scale*learn_rate
        lr_2 = scale*learn_rate
        lam_kld = 1.0 #+ (scale * 0.25)
        #Xb_tr = binarize_data(Xtr.take(tr_idx, axis=0))
        Xb_tr = Xtr.take(tr_idx, axis=0)
        Xb_tr = Xb_tr.astype(theano.config.floatX)
        # set sgd and objective function hyperparams for this update
        MSM.set_sgd_params(lr_1=lr_1, lr_2=lr_2, \
                mom_1=(scale*momentum), mom_2=0.99)
        MSM.set_train_switch(1.0)
        MSM.set_l1l2_weight(l1l2_weight)
        MSM.set_lam_nll(lam_nll=1.0)
        MSM.set_lam_kld(lam_kld_1=lam_kld, lam_kld_2=lam_kld)
        MSM.set_lam_l2w(1e-4)
        MSM.set_kzg_weight(0.1)
        # perform a minibatch update and record the cost for this batch
        result = MSM.train_joint(Xb_tr, Xb_tr, batch_reps)
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
            # Get some validation samples for computing diagnostics
            Xva = row_shuffle(Xva)
            Xb_va = Xva[0:5000].astype(theano.config.floatX)
            #Xb_va = binarize_data(Xva[0:5000])
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
            file_name = "MZ_B_SAMPLES_IND_b{0:d}.png".format(i)
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
            file_name = "MZ_B_SAMPLES_CND_GD_b{0:d}.png".format(i)
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
            file_name = "MZ_B_SAMPLES_CND_UD_b{0:d}.png".format(i)
            utils.visualize_samples(seq_samps, file_name, num_rows=20)
            # compute information about posterior KLds on validation set
            raw_costs = MSM.compute_raw_costs(Xb_va, Xb_va)
            init_nll, init_kld, cond_kld, glob_kld, step_nll, step_kld = raw_costs
            step_nll[0] = step_nll[1] # scale of first NLL is overwhemling
            file_name = "MZ_B_H0_KLDS_b{0:d}.png".format(i)
            utils.plot_stem(np.arange(init_kld.shape[1]), \
                    np.mean(init_kld, axis=0), file_name)
            file_name = "MZ_B_HI_COND_KLDS_b{0:d}.png".format(i)
            utils.plot_stem(np.arange(cond_kld.shape[1]), \
                    np.mean(cond_kld, axis=0), file_name)
            file_name = "MZ_B_HI_GLOB_KLDS_b{0:d}.png".format(i)
            utils.plot_stem(np.arange(glob_kld.shape[1]), \
                    np.mean(glob_kld, axis=0), file_name)
            file_name = "MZ_B_STEP_NLLS_b{0:d}.png".format(i)
            utils.plot_stem(np.arange(step_nll.shape[0]), \
                    step_nll, file_name)
            file_name = "MZ_B_STEP_KLDS_b{0:d}.png".format(i)
            utils.plot_stem(np.arange(step_kld.shape[0]), \
                    step_kld, file_name)
            file_name = "MZ_B_STEP_VFES_b{0:d}.png".format(i)
            utils.plot_stem(np.arange(step_kld.shape[0]).ravel(), \
                    (np.cumsum(step_kld.ravel())+step_nll.ravel()), file_name)
            #Xb_tr = binarize_data(Xtr.take(tr_idx, axis=0))
            tr_idx = npr.randint(low=0,high=tr_samples,size=(5000,))
            Xb_tr = Xtr.take(tr_idx, axis=0)
            Xb_tr = Xb_tr.astype(theano.config.floatX)
            fe_terms = MSM.compute_fe_terms(Xb_tr, Xb_tr, 30)
            fe_nll = np.mean(fe_terms[0])
            fe_kld = np.mean(fe_terms[1])
            fe_joint = fe_nll + fe_kld
            joint_str = "    vfe-tr: {0:.4f}, nll: ({1:.4f}, {2:.4f}, {3:.4f}), kld: ({4:.4f}, {5:.4f}, {6:.4f})".format( \
                    fe_joint, fe_nll, np.min(fe_terms[0]), np.max(fe_terms[0]), fe_kld, np.min(fe_terms[1]), np.max(fe_terms[1]))
            print(joint_str)
            out_file.write(joint_str+"\n")
            out_file.flush()
            # file_name = "MZ_B_FREE_ENERGY_TR_b{0:d}.png".format(i)
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
            # file_name = "MZ_B_FREE_ENERGY_VA_b{0:d}.png".format(i)
            # utils.plot_scatter(fe_terms[1], fe_terms[0], file_name, \
            #         x_label='Posterior KLd', y_label='Negative Log-likelihood')

if __name__=="__main__":
    test_with_model_init()