import time
import numpy as np
import numpy.random as npr
import theano
import theano.tensor as T
import utils as utils
from load_data import load_udm, load_udm_ss, load_mnist, load_svhn
from PeaNet import PeaNet, load_peanet_from_file
from InfNet import InfNet, load_infnet_from_file
from GenNet import GenNet, load_gennet_from_file
import PeaNet as PNet
import InfNet as INet
import GenNet as GNet
from GIPair import GIPair
from GIStack import GIStack
from NetLayers import relu_actfun, softplus_actfun, \
                      safe_softmax, safe_log
from DKCode import PCA_theano

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


########################
########################
## TESTING FOR GIPair ##
########################
########################

def test_gip_sigma_scale():
    from LogPDFs import cross_validate_sigma
    # Simple test code, to check that everything is basically functional.
    print("TESTING...")

    # Initialize a source of randomness
    rng = np.random.RandomState(1234)

    # Load some data to train/validate/test with
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm(dataset, zero_mean=False)
    Xtr = datasets[0][0]
    Xtr = Xtr.get_value(borrow=False)
    Xva = datasets[1][0]
    Xva = Xva.get_value(borrow=False)
    print("Xtr.shape: {0:s}, Xva.shape: {1:s}".format(str(Xtr.shape),str(Xva.shape)))

    # get and set some basic dataset information
    tr_samples = Xtr.shape[0]
    data_dim = Xtr.shape[1]
    batch_size = 100
    prior_dim = 32
    prior_sigma = 1.0
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
    gn_fname = "MMS_RESULTS_32D/pt60k_vcgl_params_b100000_GN.pkl"
    in_fname = "MMS_RESULTS_32D/pt60k_vcgl_params_b100000_IN.pkl"
    IN = INet.load_infnet_from_file(f_name=in_fname, rng=rng, Xd=Xd, Xc=Xc, Xm=Xm)
    GN = GNet.load_gennet_from_file(f_name=gn_fname, rng=rng, Xp=Xp)
    # construct a GIPair with the loaded InfNet and GenNet
    GIP = GIPair(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, g_net=GN, i_net=IN, \
            data_dim=data_dim, prior_dim=prior_dim, params=None)
    # draw many samples from the GIP
    for sigma_scale in [1.0, 1.5, 2.0]:
        tr_idx = npr.randint(low=0,high=tr_samples,size=(100,))
        Xd_batch = Xtr.take(tr_idx, axis=0)
        ss_int = int(10. * sigma_scale)
        sample_lists = GIP.sample_gil_from_data(Xd_batch[0,:].reshape((1,data_dim)), loop_iters=500, \
                sigma_scale=sigma_scale)
        Xs = np.vstack(sample_lists["data samples"])
        file_name = "AAA_TEST_{0:d}.png".format(ss_int)
        utils.visualize_samples(Xs, file_name, num_rows=20)
    file_name = "AAA_TEST_PRIOR.png"
    Xs = GIP.sample_from_gn(25*25)
    utils.visualize_samples(Xs, file_name, num_rows=25)
    # test Parzen density estimator built from prior samples
    Xs = GIP.sample_from_gn(10000)
    cross_validate_sigma(Xs, Xva, [0.1, 0.13, 0.15, 0.18, 0.2], 100)
    return

def test_gip_mnist_60k():
    # Initialize a source of randomness
    rng = np.random.RandomState(1234)

    # Load some data to train/validate/test with
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm(dataset, zero_mean=False)
    Xtr_shared = datasets[0][0]
    Xva_shared = datasets[1][0]
    Xtr = Xtr_shared.get_value(borrow=False)
    Xva = Xva_shared.get_value(borrow=False)
    Xtr = Xtr + ((1. / 256.) * npr.rand(Xtr.shape[0], Xtr.shape[1]))
    Xva = Xva + ((1. / 256.) * npr.rand(Xva.shape[0], Xva.shape[1]))
    Xtr = Xtr / np.max(Xtr, axis=1, keepdims=True)
    Xva = Xva / np.max(Xva, axis=1, keepdims=True)
    Xtr = Xtr.astype(theano.config.floatX)
    Xva = Xva.astype(theano.config.floatX)
    tr_samples = Xtr.shape[0]
    va_samples = Xva.shape[0]
    batch_size = 100
    batch_reps = 5

    # Construct a GenNet and an InfNet, then test constructor for GIPair.
    # Do basic testing, to make sure classes aren't completely broken.
    Xp = T.matrix('Xp_base')
    Xd = T.matrix('Xd_base')
    Xc = T.matrix('Xc_base')
    Xm = T.matrix('Xm_base')
    data_dim = Xtr.shape[1]
    prior_dim = 50
    prior_sigma = 1.0

    # Choose some parameters for the generator network
    gn_params = {}
    gn_config = [prior_dim, 800, 800, data_dim]
    gn_params['mlp_config'] = gn_config
    gn_params['activation'] = relu_actfun
    gn_params['out_type'] = 'gaussian'
    gn_params['logvar_type'] = 'multi_shared'
    gn_params['mean_transform'] = 'sigmoid'
    gn_params['init_scale'] = 1.0
    gn_params['lam_l2a'] = 1e-2
    gn_params['vis_drop'] = 0.0
    gn_params['hid_drop'] = 0.0
    gn_params['bias_noise'] = 0.1
    # choose some parameters for the continuous inferencer
    in_params = {}
    shared_config = [data_dim, 800, 800]
    top_config = [shared_config[-1], 200, prior_dim]
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
    # Initialize the base networks for this GIPair
    IN = InfNet(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, prior_sigma=prior_sigma, \
            params=in_params, shared_param_dicts=None)
    GN = GenNet(rng=rng, Xp=Xp, prior_sigma=prior_sigma, \
            params=gn_params, shared_param_dicts=None)
    # Initialize biases in IN and GN
    IN.init_biases(0.1)
    GN.init_biases(0.1)
    # Initialize the GIPair
    GIP = GIPair(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, g_net=GN, i_net=IN, \
            data_dim=data_dim, prior_dim=prior_dim, params=None)
    GIP.set_lam_l2w(1e-4)

    ####################
    # RICA PRETRAINING #
    ####################
    IN.W_rica.set_value(0.05 * IN.W_rica.get_value(borrow=False))
    GN.W_rica.set_value(0.05 * GN.W_rica.get_value(borrow=False))
    for i in range(4000):
        scale = min(1.0, (float(i+1) / 5000.0))
        l_rate = 0.0001 * scale
        lam_l1 = 0.05
        tr_idx = npr.randint(low=0,high=tr_samples,size=(1000,))
        Xd_batch = Xtr.take(tr_idx, axis=0)
        inr_out = IN.train_rica(Xd_batch, l_rate, lam_l1)
        gnr_out = GN.train_rica(Xd_batch, l_rate, lam_l1)
        if ((i % 1000) == 0):
            print("rica batch {0:d}: in_recon={1:.4f}, in_spars={2:.4f}, gn_recon={3:.4f}, gn_spars={4:.4f}".format( \
                    i, 1.*inr_out[1], 1.*inr_out[2], 1.*gnr_out[1], 1.*gnr_out[2]))
                        # draw inference net first layer weights
    file_name = "pt60k_rica_inf_weights.png"
    utils.visualize_samples(IN.W_rica.get_value(borrow=False).T, file_name, num_rows=20)
    # draw generator net final layer weights
    file_name = "pt60k_rica_gen_weights.png"
    if ('gaussian' in gn_params['out_type']):
        lay_num = -2
    else:
        lay_num = -1
    utils.visualize_samples(GN.W_rica.get_value(borrow=False), file_name, num_rows=20)
    ####################
    ####################

    out_file = open("pt60k_gip_results.txt", 'wb')
    # Set initial learning rate and basic SGD hyper parameters
    cost_1 = [0. for i in range(10)]
    learn_rate = 0.0004
    for i in range(500000):
        scale = min(1.0, float(i) / 30000.0)
        if ((i+1) % 100000) == 0:
            learn_rate = learn_rate * 0.5
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
        GIP.set_lam_kld(1.0 + 1.0*scale)
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
            file_name = "pt60k_gip_chain_samples_b{0:d}.png".format(i)
            Xd_samps = np.repeat(Xd_batch[0:10,:], 3, axis=0)
            sample_lists = GIP.sample_gil_from_data(Xd_samps, loop_iters=20)
            Xs = np.vstack(sample_lists["data samples"])
            utils.visualize_samples(Xs, file_name, num_rows=20)
            # draw samples freely from the generative model's prior
            file_name = "pt60k_gip_prior_samples_b{0:d}.png".format(i)
            Xs = GIP.sample_from_gn(20*20)
            utils.visualize_samples(Xs, file_name, num_rows=20)
            # draw inference net first layer weights
            file_name = "pt60k_gip_inf_weights_b{0:d}.png".format(i)
            utils.visualize_net_layer(GIP.IN.shared_layers[0], file_name)
            # draw generator net final layer weights
            file_name = "pt60k_gip_gen_weights_b{0:d}.png".format(i)
            if (gn_params['out_type'] == 'gaussian'):
                lay_num = -2
            else:
                lay_num = -1
            utils.visualize_net_layer(GIP.GN.mlp_layers[lay_num], file_name, \
                    colorImg=False, use_transpose=True)
    return

def test_gi_pair_svhn_pca():
    # Initialize a source of randomness
    rng = np.random.RandomState(1234)

    # Load some data to train/validate/test with
    tr_file = 'data/svhn_train.pkl'
    te_file = 'data/svhn_test.pkl'
    data_dict = load_svhn(tr_file, te_file, ex_file=None, ex_count=None)
    Xtr = data_dict['Xtr'] / 256.0
    tr_samples = Xtr.shape[0]

    # get pca transformation to use as "preprocessing"
    f_enc, f_dec, pca_shared_params = \
            PCA_theano(Xtr, cutoff=600, global_sd=True)
    pca_dim = pca_shared_params['pca_dim']
    # make numpy versions of encoder and decoder
    sym_mat = T.matrix('sym_mat')
    fe_output = f_enc(sym_mat)
    fd_output = f_dec(sym_mat)
    f_enc_np = theano.function([sym_mat], outputs=fe_output)
    f_dec_np = theano.function([sym_mat], outputs=fd_output)

    # Construct a GenNet and an InfNet, then test constructor for GIPair.
    # Do basic testing, to make sure classes aren't completely broken.
    Xp = T.matrix('Xp_base')
    Xd = T.matrix('Xd_base')
    Xc = T.matrix('Xc_base')
    Xm = T.matrix('Xm_base')
    data_dim = Xtr.shape[1]
    prior_dim = 300
    prior_sigma = 1.0

    # Choose some parameters for the generator network
    gn_params = {}
    gn_config = [prior_dim, 800, 800, pca_dim]
    gn_params['mlp_config'] = gn_config
    gn_params['activation'] = relu_actfun
    gn_params['out_type'] = 'gaussian'
    gn_params['encoder'] = f_enc
    gn_params['decoder'] = f_dec
    gn_params['init_scale'] = 1.0
    gn_params['lam_l2a'] = 1e-3
    gn_params['vis_drop'] = 0.0
    gn_params['hid_drop'] = 0.0
    gn_params['bias_noise'] = 0.1
    # choose some parameters for the continuous inferencer
    in_params = {}
    shared_config = [pca_dim, 800, 800]
    top_config = [shared_config[-1], prior_dim]
    in_params['shared_config'] = shared_config
    in_params['mu_config'] = top_config
    in_params['sigma_config'] = top_config
    in_params['activation'] = relu_actfun
    in_params['encoder'] = f_enc
    in_params['decoder'] = f_dec
    in_params['init_scale'] = 1.0
    in_params['lam_l2a'] = 1e-3
    in_params['vis_drop'] = 0.0
    in_params['hid_drop'] = 0.0
    in_params['bias_noise'] = 0.1
    in_params['input_noise'] = 0.0
    # Initialize the base networks for this GIPair
    IN = InfNet(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, prior_sigma=prior_sigma, \
            params=in_params, shared_param_dicts=None)
    GN = GenNet(rng=rng, Xp=Xp, prior_sigma=prior_sigma, \
            params=gn_params, shared_param_dicts=None)
    # Initialize biases in IN and GN
    IN.init_biases(0.1)
    GN.init_biases(0.1)
    # Initialize the GIPair
    GIP = GIPair(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, g_net=GN, i_net=IN, \
            data_dim=pca_dim, prior_dim=prior_dim, params=None)
    GIP.set_lam_l2w(1e-5)

    COMMENT="""
    IN.W_rica.set_value(0.05 * IN.W_rica.get_value(borrow=False))
    GN.W_rica.set_value(0.05 * GN.W_rica.get_value(borrow=False))
    for i in range(50000):
        scale = min(1.0, (float(i+1) / 5000.0))
        l_rate = 0.001 * scale
        lam_l1 = 0.05 * scale
        tr_idx = npr.randint(low=0,high=tr_samples,size=(500,))
        Xd_batch = Xtr.take(tr_idx, axis=0)
        inr_out = IN.train_rica(Xd_batch, l_rate, lam_l1)
        gnr_out = GN.train_rica(Xd_batch, l_rate, lam_l1)
        if ((i % 1000) == 0):
            print("rica batch {0:d}: in_recon={1:.4f}, in_spars={2:.4f}, gn_recon={3:.4f}, gn_spars={4:.4f}".format( \
                    i, 1.*inr_out[1], 1.*inr_out[2], 1.*gnr_out[1], 1.*gnr_out[2]))
                        # draw inference net first layer weights
            if ((i % 2000) == 0):
                file_name = "GIP_SVHN_PCA_RICA_INF_WEIGHTS_b{0:d}.png".format(i)
                utils.mat_to_img(f_dec_np(IN.W_rica.get_value(borrow=False).T), file_name, \
                        (32,32), num_rows=20, scale=True, colorImg=True, tile_spacing=(1,1))
                # draw generator net final layer weights
                file_name = "GIP_SVHN_PCA_RICA_GEN_WEIGHTS_b{0:d}.png".format(i)
                if ('gaussian' in gn_params['out_type']):
                    lay_num = -2
                else:
                    lay_num = -1
                utils.mat_to_img(f_dec_np(GN.W_rica.get_value(borrow=False)), file_name, \
                        (32,32), num_rows=20, scale=True, colorImg=True, tile_spacing=(1,1))
    """

    out_file = open("GIP_SVHN_PCA_AAA.txt", 'wb')
    # Set initial learning rate and basic SGD hyper parameters
    learn_rate = 0.0003
    cost_1 = [0. for i in range(10)]
    for i in range(750000):
        scale = min(1.0, float(i) / 5000.0)
        if ((i+1 % 100000) == 0):
            learn_rate = learn_rate * 0.8
        GIP.set_all_sgd_params(lr_gn=scale*learn_rate, \
                lr_in=scale*learn_rate, mom_1=0.9, mom_2=0.999)
        GIP.set_lam_kld(0.1)
        GIP.set_lam_nll(1.0)
        # get some data to train with
        tr_idx = npr.randint(low=0,high=tr_samples,size=(100,))
        Xd_batch = Xtr.take(tr_idx, axis=0)
        Xd_batch = np.repeat(Xd_batch, 5, axis=0)
        Xc_batch = 0.0 * Xd_batch
        Xm_batch = 0.0 * Xd_batch

        # do a minibatch update of the model, and compute some costs
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
            tr_idx = npr.randint(low=0,high=tr_samples,size=(100,))
            Xd_batch = Xtr.take(tr_idx, axis=0)
            file_name = "GIP_SVHN_PCA_CHAIN_SAMPLES_b{0:d}.png".format(i)
            Xd_samps = np.repeat(Xd_batch[0:10,:], 3, axis=0)
            sample_lists = GIP.sample_gil_from_data(Xd_samps, loop_iters=15)
            Xs = np.vstack(sample_lists["data samples"])
            utils.mat_to_img(Xs, file_name, (32,32), num_rows=15, \
                    scale=True, colorImg=True, tile_spacing=(1,1))
            # draw samples freely from the generative model's prior
            file_name = "GIP_SVHN_PCA_PRIOR_SAMPLES_b{0:d}.png".format(i)
            Xs = GIP.sample_from_gn(15*15)
            utils.mat_to_img(Xs, file_name, (32,32), num_rows=15, \
                    scale=True, colorImg=True, tile_spacing=(1,1))
            # draw inference net first layer weights
            file_name = "GIP_SVHN_PCA_INF_WEIGHTS_b{0:d}.png".format(i)
            utils.visualize_net_layer(GIP.IN.shared_layers[0], file_name, \
                    colorImg=True, use_transpose=False, transform=f_dec_np)
            # draw generator net final layer weights
            file_name = "GIP_SVHN_PCA_GEN_WEIGHTS_b{0:d}.png".format(i)
            if (gn_params['out_type'] == 'gaussian'):
                lay_num = -2
            else:
                lay_num = -1
            utils.visualize_net_layer(GIP.GN.mlp_layers[lay_num], file_name, \
                    colorImg=True, use_transpose=True, transform=f_dec_np)
    print("TESTING COMPLETE!")
    return


###################
# TEST DISPATCHER #
###################

if __name__=="__main__":
    #test_gi_pair_svhn_pca()
    #test_gip_sigma_scale()
    test_gip_mnist_60k()
    
