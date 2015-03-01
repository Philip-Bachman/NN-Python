import time
import numpy as np
import numpy.random as npr
import theano
import theano.tensor as T
import utils as utils
from load_data import load_udm, load_udm_ss, load_mnist, load_svhn, load_tfd
from PeaNet import PeaNet, load_peanet_from_file
from InfNet import InfNet, load_infnet_from_file
from GenNet import GenNet, load_gennet_from_file
import PeaNet as PNet
import InfNet as INet
import GenNet as GNet
from GIPair import GIPair
from GIPair2 import GIPair2
from GIStack import GIStack
from NetLayers import relu_actfun, softplus_actfun, \
                      safe_softmax, safe_log, tanh_actfun
from DKCode import PCA_theano
from VideoUtils import VideoSink

####################
# HELPER FUNCTIONS #
####################

def nan_debug_print(x, str='damn the nan'):
    """
    Work around theano's debugging deficiencies.
    """
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        print(str)
    return x

def binarize_data(X):
    """
    Make a sample of bernoulli variables with probabilities given by X.
    """
    X_shape = X.shape
    probs = npr.rand(*X_shape)
    X_binary = 1.0 * (probs < X)
    return X_binary.astype(theano.config.floatX)

def mnist_prob_embed(X, Y):
    """
    Embed the predicted class probabilities in Y in the digits images in X.
    """
    obs_count = X.shape[0]
    class_count = Y.shape[1]
    Xy = np.zeros(X.shape)
    for i in range(obs_count):
        x_sq = X[i,:].reshape((28,28))
        for j in range(class_count):
            x_sq[((2*j)+1):((2*j+2)+1),0:3] = Y[i,j]
            x_sq[((2*j)+1):((2*j+2)+1),3] = 0.2
            x_sq[((2*j)+1):((2*j+2)+1),0] = 0.2
        x_sq[0,0:4] = 0.2
        x_sq[((2*class_count)+1),0:4] = 0.2
        Xy[i,:] = x_sq.flatten()
    return Xy

def rand_sample(param_list):
    """
    Sample a value uniformly at random from the given (python) list.
    """
    new_list = [val for val in param_list]
    npr.shuffle(new_list)
    rand_val = new_list[0]
    return rand_val

def one_hot_np(Yc, cat_dim=None):
    """
    Given a numpy integer column vector Yc, generate a matrix Yoh in which
    Yoh[i,:] is a one-hot vector -- Yoh[i,Yc[i]] = 1.0 and other Yoh[i,j] = 0
    """
    if cat_dim is None:
        cat_dim = np.max(Yc) + 1
    Yoh = np.zeros((Yc.size, cat_dim))
    Yoh[np.arange(Yc.size),Yc.flatten()] = 1.0
    return Yoh

def zmuv(X, axis=1):
    X = X - np.mean(X, axis=axis, keepdims=True)
    X = X / np.std(X, axis=axis, keepdims=True)
    return X

def to_video(X, shape, v_file, frame_rate=30):
    """
    Convert grayscale image sequence to video.
    """
    # check that this is a floaty grayscale image array
    assert((np.min(X) >= 0.0) and (np.max(X) <= 1.0))
    # convert 0...1 float grayscale to 0...255 uint8 grayscale
    X = 255.0 * X
    X = X.astype(np.uint8)
    # open a video encoding stream to receive the images
    vsnk = VideoSink(v_file, size=shape, rate=frame_rate, colorspace='y8')
    for i in range(X.shape[0]):
        # reshape this frame, and push it to the video encoding stream
        frame = X[i].reshape(shape)
        vsnk(frame)
    vsnk.close()
    return

def group_chains(chain_list):
    chain_len = len(chain_list)
    chain_count = chain_list[0].shape[0]
    obs_dim = chain_list[0].shape[1]
    Xs = np.zeros((chain_len*chain_count, obs_dim))
    idx = 0
    for i in range(chain_count):
        for j in range(chain_len):
            Xs[idx] = chain_list[j][i]
            idx = idx + 1
    return Xs

########################
########################
## TESTING FOR GIPair ##
########################
########################



def posterior_klds(IN, Xtr, batch_size, batch_count):
    """
    Get posterior KLd cost for some inputs from Xtr.
    """
    post_klds = []
    for i in range(batch_count):
        batch_idx = npr.randint(low=0, high=Xtr.shape[0], size=(batch_size,))
        X = Xtr.take(batch_idx, axis=0)
        post_klds.extend([k for k in IN.kld_func(X)])
    return post_klds

######################################
######################################
## CODE FOR PRETRAINING AS A GIPAIR ##
######################################
######################################

PRIOR_DIM = 50
RESULT_PATH = './'

def pretrain_gip(extra_lam_kld=0.0, kld2_scale=0.0):
    # Initialize a source of randomness
    rng = np.random.RandomState(1234)

    # Load some data to train/validate/test with
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm(dataset, zero_mean=False)
    Xtr_shared = datasets[0][0]
    Xva_shared = datasets[1][0]
    Xtr = Xtr_shared.get_value(borrow=False).astype(theano.config.floatX)
    Xva = Xva_shared.get_value(borrow=False).astype(theano.config.floatX)
    tr_samples = Xtr.shape[0]
    batch_size = 200
    batch_reps = 10

    # Construct a GenNet and an InfNet, then test constructor for GIPair.
    # Do basic testing, to make sure classes aren't completely broken.
    Xp = T.matrix('Xp_base')
    Xd = T.matrix('Xd_base')
    Xc = T.matrix('Xc_base')
    Xm = T.matrix('Xm_base')
    data_dim = Xtr.shape[1]
    prior_sigma = 1.0
    # Choose some parameters for the generator network
    gn_params = {}
    gn_config = [PRIOR_DIM, 500, 500, data_dim]
    gn_params['mlp_config'] = gn_config
    gn_params['activation'] = relu_actfun
    gn_params['out_type'] = 'gaussian'
    gn_params['mean_transform'] = 'sigmoid'
    gn_params['logvar_type'] = 'single_shared'
    gn_params['init_scale'] = 1.0
    gn_params['lam_l2a'] = 1e-2
    gn_params['vis_drop'] = 0.0
    gn_params['hid_drop'] = 0.0
    gn_params['bias_noise'] = 0.1
    # choose some parameters for the continuous inferencer
    in_params = {}
    shared_config = [data_dim, 500, 500]
    top_config = [shared_config[-1], PRIOR_DIM]
    in_params['shared_config'] = shared_config
    in_params['mu_config'] = top_config
    in_params['sigma_config'] = top_config
    in_params['activation'] = relu_actfun
    in_params['init_scale'] = 1.0
    in_params['lam_l2a'] = 1e-2
    in_params['vis_drop'] = 0.0
    in_params['hid_drop'] = 0.0
    in_params['bias_noise'] = 0.1
    in_params['input_noise'] = 0.0
    in_params['kld2_scale'] = kld2_scale
    # Initialize the base networks for this GIPair
    IN = InfNet(rng=rng, Xd=Xd, prior_sigma=prior_sigma, \
            params=in_params, shared_param_dicts=None)
    GN = GenNet(rng=rng, Xp=Xp, prior_sigma=prior_sigma, \
            params=gn_params, shared_param_dicts=None)
    # Initialize biases in IN and GN
    IN.init_biases(0.1)
    GN.init_biases(0.1)

    # Initialize the GIPair
    GIP = GIPair(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, g_net=GN, i_net=IN, \
            data_dim=data_dim, prior_dim=PRIOR_DIM, params=None)
    GIP.set_lam_l2w(1e-4)

    ####################
    # RICA PRETRAINING #
    ####################
    #IN.W_rica.set_value(0.05 * IN.W_rica.get_value(borrow=False))
    GN.W_rica.set_value(0.05 * GN.W_rica.get_value(borrow=False))
    for i in range(5000):
        scale = min(1.0, (float(i+1) / 5000.0))
        l_rate = 0.0001 * scale
        lam_l1 = 0.05
        tr_idx = npr.randint(low=0,high=tr_samples,size=(1000,))
        Xd_batch = Xtr.take(tr_idx, axis=0)
        #inr_out = IN.train_rica(Xd_batch, l_rate, lam_l1)
        gnr_out = GN.train_rica(Xd_batch, l_rate, lam_l1)
        inr_out = [v for v in gnr_out]
        if ((i % 1000) == 0):
            print("rica batch {0:d}: in_recon={1:.4f}, in_spars={2:.4f}, gn_recon={3:.4f}, gn_spars={4:.4f}".format( \
                    i, 1.*inr_out[1], 1.*inr_out[2], 1.*gnr_out[1], 1.*gnr_out[2]))
                        # draw inference net first layer weights
    file_name = RESULT_PATH+"pt_rica_inf_weights.png".format(i)
    utils.visualize_samples(IN.W_rica.get_value(borrow=False).T, file_name, num_rows=20)
    # draw generator net final layer weights
    file_name = RESULT_PATH+"pt_rica_gen_weights.png".format(i)
    utils.visualize_samples(GN.W_rica.get_value(borrow=False), file_name, num_rows=20)

    ######################
    # BASIC VAE TRAINING #
    ######################
    out_file = open(RESULT_PATH+"pt_gip_results.txt", 'wb')
    # Set initial learning rate and basic SGD hyper parameters
    cost_1 = [0. for i in range(10)]
    learn_rate = 0.0003
    for i in range(220000):
        scale = min(1.0, float(i) / 30000.0)
        # do a minibatch update of the model, and compute some costs
        tr_idx = npr.randint(low=0,high=tr_samples,size=(batch_size,))
        Xd_batch = Xtr.take(tr_idx, axis=0)
        Xd_batch = np.repeat(Xd_batch, batch_reps, axis=0)
        Xc_batch = 0.0 * Xd_batch
        Xm_batch = 0.0 * Xd_batch
        # do a minibatch update of the model, and compute some costs
        GIP.set_all_sgd_params(lr_gn=(scale*learn_rate), \
                lr_in=(scale*learn_rate), mom_1=0.9, mom_2=0.999)
        GIP.set_lam_nll(1.0)
        GIP.set_lam_kld(1.0 + extra_lam_kld*scale)
        outputs = GIP.train_joint(Xd_batch, Xc_batch, Xm_batch)
        cost_1 = [(cost_1[k] + 1.*outputs[k]) for k in range(len(outputs))]
        if ((i % 1000) == 0):
            cost_1 = [(v / 1000.) for v in cost_1]
            o_str = "batch: {0:d}, joint_cost: {1:.4f}, data_nll_cost: {2:.4f}, post_kld_cost: {3:.4f}, other_reg_cost: {4:.4f}".format( \
                    i, cost_1[0], cost_1[1], cost_1[2], cost_1[3])
            print(o_str)
            out_file.write(o_str+"\n")
            out_file.flush()
            cost_1 = [0. for v in cost_1]
        if ((i % 5000) == 0):
            cost_2 = GIP.compute_costs(Xva, 0.*Xva, 0.*Xva)
            o_str = "--val: {0:d}, joint_cost: {1:.4f}, data_nll_cost: {2:.4f}, post_kld_cost: {3:.4f}, other_reg_cost: {4:.4f}".format( \
                    i, 1.*cost_2[0], 1.*cost_2[1], 1.*cost_2[2], 1.*cost_2[3])
            print(o_str)
            out_file.write(o_str+"\n")
            out_file.flush()
        if ((i % 5000) == 0):
            tr_idx = npr.randint(low=0,high=tr_samples,size=(100,))
            Xd_batch = Xtr.take(tr_idx, axis=0)
            file_name = RESULT_PATH+"pt_gip_chain_samples_b{0:d}.png".format(i)
            Xd_samps = np.repeat(Xd_batch[0:10,:], 3, axis=0)
            sample_lists = GIP.sample_from_chain(Xd_samps, loop_iters=20)
            Xs = np.vstack(sample_lists["data samples"])
            utils.visualize_samples(Xs, file_name, num_rows=20)
            # draw samples freely from the generative model's prior
            file_name = RESULT_PATH+"pt_gip_prior_samples_b{0:d}.png".format(i)
            Xs = GIP.sample_from_prior(20*20)
            utils.visualize_samples(Xs, file_name, num_rows=20)
            # draw inference net first layer weights
            file_name = RESULT_PATH+"pt_gip_inf_weights_b{0:d}.png".format(i)
            utils.visualize_samples(GIP.IN.W_rica.get_value(borrow=False).T, \
                    file_name, num_rows=20)
            # draw generator net final layer weights
            file_name = RESULT_PATH+"pt_gip_gen_weights_b{0:d}.png".format(i)
            utils.visualize_samples(GIP.GN.W_rica.get_value(borrow=False), \
                    file_name, num_rows=20)
            IN.save_to_file(f_name=RESULT_PATH+"pt_gip_params_IN.pkl")
            GN.save_to_file(f_name=RESULT_PATH+"pt_gip_params_GN.pkl")
            #########################
            # Check posterior KLds. #
            #########################
            post_klds = posterior_klds(IN, Xtr, 5000, 5)
            file_name = RESULT_PATH+"pt_gip_post_klds_b{0:d}.png".format(i)
            utils.plot_kde_histogram2( \
                    np.asarray(post_klds), np.asarray(post_klds), file_name, bins=30)
        if ((i % 10000) == 0):
            IN.save_to_file(f_name=RESULT_PATH+"pt_gip_params_b{0:d}_IN.pkl".format(i))
            GN.save_to_file(f_name=RESULT_PATH+"pt_gip_params_b{0:d}_GN.pkl".format(i))
    return

def test_gip2_mnist_60k():
    # Initialize a source of randomness
    rng = np.random.RandomState(1234)

    # Load some data to train/validate/test with
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm(dataset, zero_mean=False)
    Xtr_shared = datasets[0][0]
    Xva_shared = datasets[1][0]
    Xtr = Xtr_shared.get_value(borrow=False).astype(theano.config.floatX)
    Xva = Xva_shared.get_value(borrow=False).astype(theano.config.floatX)
    tr_samples = Xtr.shape[0]
    batch_size = 1000
    batch_reps = 5

    # Setup basic symbolic variables and model parameters
    Xp = T.matrix('Xp_base')
    Xd = T.matrix('Xd_base')
    Xc = T.matrix('Xc_base')
    Xm = T.matrix('Xm_base')
    data_dim = Xtr.shape[1]
    prior_dim = 32
    prior_dim_2 = 32
    prior_sigma = 1.0

    # Load the bottom inferencer and generator from disk
    gn_fname = "MMS_RESULTS_32D/pt60k_walk_params_b300000_GN.pkl"
    in_fname = "MMS_RESULTS_32D/pt60k_walk_params_b300000_IN.pkl"
    IN = INet.load_infnet_from_file(f_name=in_fname, rng=rng, Xd=Xd)
    GN = GNet.load_gennet_from_file(f_name=gn_fname, rng=rng, Xp=Xp)

    # choose some parameters for the top generator network
    gn2_params = {}
    gn2_config = [prior_dim_2, 500, 500, prior_dim]
    gn2_params['mlp_config'] = gn2_config
    gn2_params['activation'] = tanh_actfun
    gn2_params['out_type'] = 'gaussian'
    gn2_params['mean_transform'] = 'sigmoid'
    gn2_params['logvar_type'] = 'multi_shared'
    gn2_params['init_scale'] = 2.0
    gn2_params['lam_l2a'] = 1e-3
    gn2_params['vis_drop'] = 0.0
    gn2_params['hid_drop'] = 0.0
    gn2_params['bias_noise'] = 0.1
    # choose some parameters for the top inference network
    in2_params = {}
    shared_config = [prior_dim, 500, 500]
    top_config = [shared_config[-1], prior_dim_2]
    in2_params['shared_config'] = shared_config
    in2_params['mu_config'] = top_config
    in2_params['sigma_config'] = top_config
    in2_params['activation'] = tanh_actfun
    in2_params['init_scale'] = 1.0
    in2_params['lam_l2a'] = 1e-3
    in2_params['vis_drop'] = 0.0
    in2_params['hid_drop'] = 0.0
    in2_params['bias_noise'] = 0.1
    in2_params['input_noise'] = 0.0
    # Initialize the base networks for this GIPair
    IN2 = InfNet(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, prior_sigma=prior_sigma, \
            params=in2_params, shared_param_dicts=None)
    GN2 = GenNet(rng=rng, Xp=Xp, prior_sigma=prior_sigma, \
            params=gn2_params, shared_param_dicts=None)
    # Initialize biases in IN and GN
    IN2.init_biases(0.0)
    GN2.init_biases(0.0)
    # Initialize the GIPair
    GIP2 = GIPair2(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, \
            g_net=GN, i_net=IN, \
            data_dim=data_dim, prior_dim=prior_dim, \
            g_net_2=GN2, i_net_2=IN2, \
            prior_dim_2=prior_dim_2, \
            params=None, shared_param_dicts=None)
    GIP2.set_lam_l2w(1e-5)

    # ####################
    # # RICA PRETRAINING #
    # ####################
    # IN2.W_rica.set_value(0.05 * IN2.W_rica.get_value(borrow=False))
    # GN2.W_rica.set_value(0.05 * GN2.W_rica.get_value(borrow=False))
    # for i in range(15000):
    #     scale = min(1.0, (float(i+1) / 5000.0))
    #     l_rate = 0.0002 * scale
    #     lam_l1 = 0.075
    #     tr_idx = npr.randint(low=0,high=tr_samples,size=(1000,))
    #     Xs = Xtr.take(tr_idx, axis=0)
    #     Xd_batch = IN.sample_posterior(Xs, 0.0*Xs, 0.0*Xs)
    #     inr_out = IN2.train_rica(Xd_batch, l_rate, lam_l1)
    #     gnr_out = GN2.train_rica(Xd_batch, l_rate, lam_l1)
    #     #inr_out = [v for v in gnr_out]
    #     if ((i % 1000) == 0):
    #         print("rica batch {0:d}: in_recon={1:.4f}, in_spars={2:.4f}, gn_recon={3:.4f}, gn_spars={4:.4f}".format( \
    #                 i, 1.*inr_out[1], 1.*inr_out[2], 1.*gnr_out[1], 1.*gnr_out[2]))

    ####################################
    # NORMAL TRAINING OF THE GIP STACK #
    ####################################
    out_file = open("pt_gip2_results.txt", 'wb')
    # Set initial learning rate and basic SGD hyper parameters
    cost_1 = [0. for i in range(10)]
    learn_rate = 0.004
    for i in range(150000):
        scale = min(1.0, float(i) / 20000.0)
        # do a minibatch update of the model, and compute some costs
        tr_idx = npr.randint(low=0,high=tr_samples,size=(batch_size,))
        Xb = Xtr.take(tr_idx, axis=0)
        Xd_batch = IN.sample_posterior(Xb, 0.0*Xb, 0.0*Xb)
        Xd_batch = np.repeat(Xd_batch, batch_reps, axis=0)
        Xc_batch = 0.0 * Xd_batch
        Xm_batch = 0.0 * Xd_batch
        # do a minibatch update of the model, and compute some costs
        GIP2.set_all_sgd_params(lr_gn=(scale*learn_rate), \
                lr_in=(scale*learn_rate), mom_1=0.9, mom_2=0.999)
        GIP2.set_lam_nll(1.0)
        GIP2.set_lam_kld(0.2)
        outputs = GIP2.train_top(Xd_batch, Xc_batch, Xm_batch)
        cost_1 = [(cost_1[k] + 1.*outputs[k]) for k in range(len(outputs))]
        # derp?
        if ((i % 1000) == 0):
            cost_1 = [(v / 1000.) for v in cost_1]
            o_str = "batch: {0:d}, joint_cost: {1:.4f}, data_nll_cost: {2:.4f}, post_kld_cost: {3:.4f}, other_reg_cost: {4:.4f}".format( \
                    i, cost_1[0], cost_1[1], cost_1[2], cost_1[3])
            print(o_str)
            out_file.write(o_str+"\n")
            out_file.flush()
            cost_1 = [0. for v in cost_1]
        if ((i % 5000) == 0):
            cost_2 = GIP2.compute_costs(Xva, 0.*Xva, 0.*Xva)
            o_str = "--val: {0:d}, joint_cost: {1:.4f}, data_nll_cost: {2:.4f}, post_kld_cost: {3:.4f}, other_reg_cost: {4:.4f}".format( \
                    i, 1.*cost_2[0], 1.*cost_2[1], 1.*cost_2[2], 1.*cost_2[3])
            print(o_str)
            out_file.write(o_str+"\n")
            out_file.flush()
        if ((i % 5000) == 0):
            tr_idx = npr.randint(low=0,high=tr_samples,size=(100,))
            Xd_batch = Xtr.take(tr_idx, axis=0)
            # sample from top/bottom chains
            file_name = "pt_gip2_chain_samples_b{0:d}_top.png".format(i)
            Xd_samps = np.repeat(Xd_batch[0:10,:], 3, axis=0)
            sample_lists = GIP2.sample_from_chain(Xd_samps, loop_iters=20, which_gip='top')
            Xs = np.vstack(sample_lists["data samples"])
            utils.visualize_samples(Xs, file_name, num_rows=20)
            file_name = "pt_gip2_chain_samples_b{0:d}_bot.png".format(i)
            Xd_samps = np.repeat(Xd_batch[0:10,:], 3, axis=0)
            sample_lists = GIP2.sample_from_chain(Xd_samps, loop_iters=20, which_gip='bot')
            Xs = np.vstack(sample_lists["data samples"])
            utils.visualize_samples(Xs, file_name, num_rows=20)
            # sample from top/bottom priors
            file_name = "pt_gip2_prior_samples_b{0:d}_top.png".format(i)
            Xs = GIP2.sample_from_prior(20*20, which_gip='top')
            utils.visualize_samples(Xs, file_name, num_rows=20)
            file_name = "pt_gip2_prior_samples_b{0:d}_bot.png".format(i)
            Xs = GIP2.sample_from_prior(20*20, which_gip='bot')
            utils.visualize_samples(Xs, file_name, num_rows=20)
    return

#############################################
# TESTING FOR PARZEN LOG_DENSITY ESTIMATION #
#############################################

def test_gip_sigma_scale_mnist():
    from LogPDFs import cross_validate_sigma
    # Simple test code, to check that everything is basically functional.
    print("TESTING...")

    # Initialize a source of randomness
    rng = np.random.RandomState(12345)

    # Load some data to train/validate/test with
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm(dataset, zero_mean=False)
    Xtr = datasets[0][0]
    Xtr = Xtr.get_value(borrow=False)
    Xva = datasets[2][0]
    Xva = Xva.get_value(borrow=False)
    print("Xtr.shape: {0:s}, Xva.shape: {1:s}".format(str(Xtr.shape),str(Xva.shape)))

    # get and set some basic dataset information
    tr_samples = Xtr.shape[0]
    data_dim = Xtr.shape[1]
    batch_size = 100
    Xtr_mean = np.mean(Xtr, axis=0, keepdims=True)
    Xtr_mean = (0.0 * Xtr_mean) + np.mean(Xtr)
    Xc_mean = np.repeat(Xtr_mean, batch_size, axis=0).astype(theano.config.floatX)

    # Symbolic inputs
    Xd = T.matrix(name='Xd')
    Xc = T.matrix(name='Xc')
    Xm = T.matrix(name='Xm')
    Xt = T.matrix(name='Xt')
    Xp = T.matrix(name='Xp')

    # Load inferencer and generator from saved parameters
    gn_fname = "MNIST_WALKOUT_TEST_VAE/pt_walk_params_b100000_GN.pkl"
    in_fname = "MNIST_WALKOUT_TEST_VAE/pt_walk_params_b100000_IN.pkl"
    IN = INet.load_infnet_from_file(f_name=in_fname, rng=rng, Xd=Xd)
    GN = GNet.load_gennet_from_file(f_name=gn_fname, rng=rng, Xp=Xp)
    prior_dim = GN.latent_dim
    prior_sigma = GN.prior_sigma
    # construct a GIPair with the loaded InfNet and GenNet
    GIP = GIPair(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, g_net=GN, i_net=IN, \
            data_dim=data_dim, prior_dim=prior_dim, params=None)
    # compute variational likelihood bound and its sub-components
    bound_results = GIP.compute_ll_bound(Xva)
    ll_bounds = bound_results[0]
    post_klds = bound_results[1]
    log_likelihoods = bound_results[2]
    max_lls = bound_results[3]
    print("mean ll bound: {0:.4f}".format(np.mean(ll_bounds)))
    print("mean posterior KLd: {0:.4f}".format(np.mean(post_klds)))
    print("mean log-likelihood: {0:.4f}".format(np.mean(log_likelihoods)))
    print("mean max log-likelihood: {0:.4f}".format(np.mean(max_lls)))

    # draw many samples from the GIP
    for i in range(10):
        tr_idx = npr.randint(low=0,high=tr_samples,size=(100,))
        Xd_batch = Xtr.take(tr_idx, axis=0)
        sample_lists = GIP.sample_from_chain(Xd_batch[0:30,:], loop_iters=150, \
                sigma_scale=1.2)
        Xs = group_chains(sample_lists['data samples'])
        to_video(Xs, (28,28), "A_CHAIN_VIDEO_{0:d}.avi".format(i), frame_rate=20)
        #Xs = sample_lists['data samples']
        #Xs = [Xs[j] for j in range(len(Xs)) if ((j < -2) or ((j % 5) == 0))]
        #row_count = int(np.sqrt(len(Xs)))
        #Xs = np.vstack(Xs)
        #file_name = "A_CHAIN_IMAGE_{0:d}.png".format(i)
        #utils.visualize_samples(Xs, file_name, num_rows=row_count)
    file_name = "A_PRIOR_SAMPLE.png"
    Xs = GIP.sample_from_prior(32*32, sigma=1.0)
    utils.visualize_samples(Xs, file_name, num_rows=32)
    # test Parzen density estimator built from prior samples
    Xs = GIP.sample_from_prior(10000, sigma=1.0)
    cross_validate_sigma(Xs, Xva, [0.1, 0.12, 0.14, 0.15, 0.16, 0.17, 0.18, 0.2], 50)
    return

def test_gip_sigma_scale_tfd():
    from LogPDFs import cross_validate_sigma
    # Simple test code, to check that everything is basically functional.
    print("TESTING...")

    # Initialize a source of randomness
    rng = np.random.RandomState(12345)

    # Load some data to train/validate/test with
    data_file = 'data/tfd_data_48x48.pkl'
    dataset = load_tfd(tfd_pkl_name=data_file, which_set='unlabeled', fold='all')
    Xtr_unlabeled = dataset[0]
    dataset = load_tfd(tfd_pkl_name=data_file, which_set='train', fold='all')
    Xtr_train = dataset[0]
    Xtr = np.vstack([Xtr_unlabeled, Xtr_train])
    dataset = load_tfd(tfd_pkl_name=data_file, which_set='test', fold='all')
    Xva = dataset[0]
    tr_samples = Xtr.shape[0]
    va_samples = Xva.shape[0]
    print("Xtr.shape: {0:s}, Xva.shape: {1:s}".format(str(Xtr.shape),str(Xva.shape)))

    # get and set some basic dataset information
    tr_samples = Xtr.shape[0]
    data_dim = Xtr.shape[1]
    batch_size = 100

    # Symbolic inputs
    Xd = T.matrix(name='Xd')
    Xc = T.matrix(name='Xc')
    Xm = T.matrix(name='Xm')
    Xt = T.matrix(name='Xt')
    Xp = T.matrix(name='Xp')

    # Load inferencer and generator from saved parameters
    gn_fname = "TFD_WALKOUT_TEST_50D_SMALL/pt_gip_params_b100000_GN.pkl"
    in_fname = "TFD_WALKOUT_TEST_50D_SMALL/pt_gip_params_b100000_IN.pkl"
    IN = INet.load_infnet_from_file(f_name=in_fname, rng=rng, Xd=Xd)
    GN = GNet.load_gennet_from_file(f_name=gn_fname, rng=rng, Xp=Xp)
    prior_dim = GN.latent_dim
    prior_sigma = GN.prior_sigma
    # construct a GIPair with the loaded InfNet and GenNet
    GIP = GIPair(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, g_net=GN, i_net=IN, \
            data_dim=data_dim, prior_dim=prior_dim, params=None)
    # compute variational likelihood bound and its sub-components
    bound_results = GIP.compute_ll_bound(Xva)
    ll_bounds = bound_results[0]
    post_klds = bound_results[1]
    log_likelihoods = bound_results[2]
    max_lls = bound_results[3]
    print("mean ll bound: {0:.4f}".format(np.mean(ll_bounds)))
    print("mean posterior KLd: {0:.4f}".format(np.mean(post_klds)))
    print("mean log-likelihood: {0:.4f}".format(np.mean(log_likelihoods)))
    print("mean max log-likelihood: {0:.4f}".format(np.mean(max_lls)))
    print("min ll bound: {0:.4f}".format(np.min(ll_bounds)))
    print("max posterior KLd: {0:.4f}".format(np.max(post_klds)))
    print("min log-likelihood: {0:.4f}".format(np.min(log_likelihoods)))
    print("min max log-likelihood: {0:.4f}".format(np.min(max_lls)))
    # compute some information about the approximate posteriors
    post_stats = GIP.compute_post_stats(Xva, 0.0*Xva, 0.0*Xva)
    all_post_klds = np.sort(post_stats[0].ravel()) # post KLds for each obs and dim
    obs_post_klds = np.sort(post_stats[1]) # summed post KLds for each obs
    post_dim_klds = post_stats[2] # average post KLds for each post dim
    post_dim_vars = post_stats[3] # average squared mean for each post dim
    utils.plot_line(np.arange(all_post_klds.shape[0]), all_post_klds, "AAA_ALL_POST_KLDS.png")
    utils.plot_line(np.arange(obs_post_klds.shape[0]), obs_post_klds, "AAA_OBS_POST_KLDS.png")
    utils.plot_stem(np.arange(post_dim_klds.shape[0]), post_dim_klds, "AAA_POST_DIM_KLDS.png")
    utils.plot_stem(np.arange(post_dim_vars.shape[0]), post_dim_vars, "AAA_POST_DIM_VARS.png")

    # repeat on part of the training set
    print("==================================================")
    print("==================================================")
    tr_idx = np.arange(tr_samples)
    npr.shuffle(tr_idx)
    _Xtr_ = Xtr[tr_idx[0:5000]]
    bound_results = GIP.compute_ll_bound(_Xtr_)
    ll_bounds = bound_results[0]
    post_klds = bound_results[1]
    log_likelihoods = bound_results[2]
    max_lls = bound_results[3]
    print("mean ll bound: {0:.4f}".format(np.mean(ll_bounds)))
    print("mean posterior KLd: {0:.4f}".format(np.mean(post_klds)))
    print("mean log-likelihood: {0:.4f}".format(np.mean(log_likelihoods)))
    print("mean max log-likelihood: {0:.4f}".format(np.mean(max_lls)))
    print("min ll bound: {0:.4f}".format(np.min(ll_bounds)))
    print("max posterior KLd: {0:.4f}".format(np.max(post_klds)))
    print("min log-likelihood: {0:.4f}".format(np.min(log_likelihoods)))
    print("min max log-likelihood: {0:.4f}".format(np.min(max_lls)))
    # compute some information about the approximate posteriors
    post_stats = GIP.compute_post_stats(_Xtr_, 0.0*_Xtr_, 0.0*_Xtr_)
    all_post_klds = np.sort(post_stats[0].ravel()) # post KLds for each obs and dim
    obs_post_klds = np.sort(post_stats[1]) # summed post KLds for each obs
    post_dim_klds = post_stats[2] # average post KLds for each post dim
    post_dim_vars = post_stats[3] # average squared mean for each post dim
    utils.plot_line(np.arange(all_post_klds.shape[0]), all_post_klds, "AAB_ALL_POST_KLDS.png")
    utils.plot_line(np.arange(obs_post_klds.shape[0]), obs_post_klds, "AAB_OBS_POST_KLDS.png")
    utils.plot_stem(np.arange(post_dim_klds.shape[0]), post_dim_klds, "AAB_POST_DIM_KLDS.png")
    utils.plot_stem(np.arange(post_dim_vars.shape[0]), post_dim_vars, "AAB_POST_DIM_VARS.png")

    # draw many samples from the GIP
    for i in range(10):
        tr_idx = npr.randint(low=0,high=tr_samples,size=(100,))
        Xd_batch = Xtr.take(tr_idx, axis=0)
        sample_lists = GIP.sample_from_chain(Xd_batch[0:20,:], loop_iters=50, \
                sigma_scale=1.0)
        Xs = group_chains(sample_lists['data samples'])
        to_video(Xs, (48,48), "A_CHAIN_VIDEO_{0:d}.avi".format(i), frame_rate=10)
        #sample_lists = GIP.sample_from_chain(Xd_batch[0,:].reshape((1,data_dim)), loop_iters=300, \
        #        sigma_scale=1.0)
        #Xs = np.vstack(sample_lists["data samples"])
        #file_name = "TFD_TEST_{0:d}.png".format(i)
        #utils.visualize_samples(Xs, file_name, num_rows=15)
    file_name = "A_PRIOR_SAMPLE.png"
    Xs = GIP.sample_from_prior(32*32, sigma=1.0)
    utils.visualize_samples(Xs, file_name, num_rows=32)
    # test Parzen density estimator built from prior samples
    Xs = GIP.sample_from_prior(10000, sigma=1.0)
    [best_sigma, best_ll, best_lls] = \
            cross_validate_sigma(Xs, Xva, [0.09, 0.095, 0.1, 0.105, 0.11], 10)
    best_lls = np.sort(best_lls)
    sort_idx = np.argsort(best_lls)
    utils.plot_line(np.arange(best_lls.shape[0]), best_lls, "BEST_LLS_1.png")
    bad_faces = Xva[sort_idx[0:1024]]
    utils.visualize_samples(bad_faces, "BAD_FACES_1.png", num_rows=32)
    ##########
    # AGAIN! #
    ##########
    Xs = GIP.sample_from_prior(10000, sigma=1.0)
    [best_sigma, best_ll, best_lls] = \
            cross_validate_sigma(Xs, Xva, [0.09, 0.095, 0.1, 0.105, 0.11], 10)
    best_lls = np.sort(best_lls)
    sort_idx = np.argsort(best_lls)
    utils.plot_line(np.arange(best_lls.shape[0]), best_lls, "BEST_LLS_2.png")
    bad_faces = Xva[sort_idx[0:1024]]
    utils.visualize_samples(bad_faces, "BAD_FACES_2.png", num_rows=32)
    return

###################
# TEST DISPATCHER #
###################

if __name__=="__main__":
    #test_gip2_mnist_60k()
    #pretrain_gip(extra_lam_kld=9.0, kld2_scale=0.1)
    #test_gip_sigma_scale_mnist()
    test_gip_sigma_scale_tfd()
    
