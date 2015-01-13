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

def test_gi_pair_mnist():
    # Initialize a source of randomness
    rng = np.random.RandomState(1234)

    # Load some data to train/validate/test with
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm(dataset, zero_mean=False)
    Xtr_shared = datasets[0][0]
    Xva_shared = datasets[1][0]
    Xtr = Xtr_shared.get_value(borrow=False).astype(theano.config.floatX)
    Xva = Xva_shared.get_value(borrow=False).astype(theano.config.floatX)
    #Xtr = Xtr - np.mean(Xtr, axis=0, keepdims=True)
    #Xtr = Xtr / Xtr.std()
    #Xtr = Xtr / np.std(Xtr, axis=1, keepdims=True)
    tr_samples = Xtr.shape[0]
    batch_size = 100
    batch_reps = 5
    max_bs_idx = tr_samples - batch_size - 1

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
    gn_params['mean_transform'] = lambda x: T.nnet.sigmoid(x)
    gn_params['init_scale'] = 1.0
    gn_params['lam_l2a'] = 1e-2
    gn_params['vis_drop'] = 0.0
    gn_params['hid_drop'] = 0.0
    gn_params['bias_noise'] = 0.1
    # choose some parameters for the continuous inferencer
    in_params = {}
    shared_config = [data_dim, 800, 800]
    top_config = [shared_config[-1], prior_dim]
    in_params['shared_config'] = shared_config
    in_params['mu_config'] = top_config
    in_params['sigma_config'] = top_config
    in_params['activation'] = relu_actfun
    in_params['init_scale'] = 1.0
    in_params['lam_l2a'] = 1e-2
    in_params['vis_drop'] = 0.2
    in_params['hid_drop'] = 0.5
    in_params['bias_noise'] = 0.1
    in_params['input_noise'] = 0.0
    # Initialize the base networks for this GIPair
    IN = InfNet(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, prior_sigma=prior_sigma, \
            params=in_params, shared_param_dicts=None)
    GN = GenNet(rng=rng, Xp=Xp, prior_sigma=prior_sigma, \
            params=gn_params, shared_param_dicts=None)
    # Initialize biases in IN and GN
    IN.init_biases(0.2)
    GN.init_biases(0.2)
    # Initialize the GIPair
    GIP = GIPair(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, g_net=GN, i_net=IN, \
            data_dim=data_dim, prior_dim=prior_dim, params=None)
    GIP.set_lam_l2w(1e-4)

    #IN.W_rica.set_value(0.05 * IN.W_rica.get_value(borrow=False))
    GN.W_rica.set_value(0.05 * GN.W_rica.get_value(borrow=False))
    for i in range(2500):
        scale = min(1.0, (float(i+1) / 5000.0))
        l_rate = 0.001 * scale
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
            if ((i % 2000) == 0):
                file_name = "GIP_MNIST_RICA_INF_WEIGHTS_b{0:d}.png".format(i)
                utils.visualize_samples(IN.W_rica.get_value(borrow=False).T, file_name, num_rows=20)
                # draw generator net final layer weights
                file_name = "GIP_MNIST_RICA_GEN_WEIGHTS_b{0:d}.png".format(i)
                if ('gaussian' in gn_params['out_type']):
                    lay_num = -2
                else:
                    lay_num = -1
                utils.visualize_samples(GN.W_rica.get_value(borrow=False), file_name, num_rows=20)

    out_file = open("GIP_MNIST_AAA.txt", 'wb')
    # Set initial learning rate and basic SGD hyper parameters
    cost_1 = [0. for i in range(10)]
    learn_rate = 0.0005
    for i in range(750000):
        scale = min(1.0, float(i) / 25000.0)
        if ((i+1 % 100000) == 0):
            learn_rate = learn_rate * 0.8
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
        GIP.set_lam_kld(1.0 + 1.5*scale)
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
            file_name = "GIP_MNIST_CHAIN_SAMPLES_b{0:d}.png".format(i)
            Xd_samps = np.repeat(Xd_batch[0:10,:], 3, axis=0)
            sample_lists = GIP.sample_gil_from_data(Xd_samps, loop_iters=20)
            Xs = np.vstack(sample_lists["data samples"])
            utils.visualize_samples(Xs, file_name, num_rows=20)
            # draw samples freely from the generative model's prior
            file_name = "GIP_MNIST_PRIOR_SAMPLES_b{0:d}.png".format(i)
            Xs = GIP.sample_from_gn(20*20)
            utils.visualize_samples(Xs, file_name, num_rows=20)
            # draw inference net first layer weights
            file_name = "GIP_MNIST_INF_WEIGHTS_b{0:d}.png".format(i)
            utils.visualize_net_layer(GIP.IN.shared_layers[0], file_name)
            # draw generator net final layer weights
            file_name = "GIP_MNIST_GEN_WEIGHTS_b{0:d}.png".format(i)
            if (gn_params['out_type'] == 'gaussian'):
                lay_num = -2
            else:
                lay_num = -1
            utils.visualize_net_layer(GIP.GN.mlp_layers[lay_num], file_name, \
                    colorImg=False, use_transpose=True)
    print("TESTING COMPLETE!")
    return

def test_gi_pair_mnist_pca():
    # Initialize a source of randomness
    rng = np.random.RandomState(1234)

    # Load some data to train/validate/test with
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm(dataset, zero_mean=False)
    Xtr_shared = datasets[0][0]
    Xtr = Xtr_shared.get_value(borrow=False).astype(theano.config.floatX)
    tr_samples = Xtr.shape[0]

    # get pca transformation to use as "preprocessing"
    f_enc, f_dec, pca_shared_params = \
            PCA_theano(Xtr, cutoff=250, global_sd=True)
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
    prior_dim = 50
    prior_sigma = 1.0

    # Choose some parameters for the generator network
    gn_params = {}
    gn_config = [prior_dim, 500, 500, pca_dim]
    gn_params['mlp_config'] = gn_config
    gn_params['activation'] = softplus_actfun
    gn_params['encoder'] = f_enc
    gn_params['decoder'] = f_dec
    gn_params['out_type'] = 'gaussian'
    gn_params['init_scale'] = 1.0
    gn_params['lam_l2a'] = 1e-2
    gn_params['vis_drop'] = 0.0
    gn_params['hid_drop'] = 0.0
    gn_params['bias_noise'] = 0.0
    # choose some parameters for the continuous inferencer
    in_params = {}
    shared_config = [pca_dim, 500, 500]
    top_config = [shared_config[-1], prior_dim]
    in_params['shared_config'] = shared_config
    in_params['mu_config'] = top_config
    in_params['sigma_config'] = top_config
    in_params['activation'] = softplus_actfun
    in_params['encoder'] = f_enc
    in_params['decoder'] = f_dec
    in_params['init_scale'] = 1.0
    in_params['lam_l2a'] = 1e-2
    in_params['vis_drop'] = 0.0
    in_params['hid_drop'] = 0.0
    in_params['bias_noise'] = 0.0
    in_params['input_noise'] = 0.0
    # Initialize the base networks for this GIPair
    IN = InfNet(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, prior_sigma=prior_sigma, \
            params=in_params, shared_param_dicts=None)
    GN = GenNet(rng=rng, Xp=Xp, prior_sigma=prior_sigma, \
            params=gn_params, shared_param_dicts=None)
    # Initialize biases in IN and GN
    IN.init_biases(0.0)
    GN.init_biases(0.0)
    # Initialize the GIPair
    GIP = GIPair(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, g_net=GN, i_net=IN, \
            data_dim=pca_dim, prior_dim=prior_dim, params=None)
    GIP.set_lam_l2w(1e-4)

    #
    # Pretrain the input layer of IN and output layer of GN
    #
    for i in range(10000):
        tr_idx = npr.randint(low=0,high=tr_samples,size=(500,))
        Xd_batch = Xtr.take(tr_idx, axis=0)
        inr_out = IN.train_rica(Xd_batch)
        gnr_out = GN.train_rica(Xd_batch)
        if ((i % 1000) == 0):
            print("rica batch {0:d}: in_rica_cost={1:.4f}, gn_rica_cost={2:.4f}".format( \
                    i, 1.*inr_out[0], 1.*gnr_out[1]))

    out_file = open("GIP_MNIST_PCA_AAA.txt", 'wb')
    # Set initial learning rate and basic SGD hyper parameters
    learn_rate = 0.0002
    for i in range(750000):
        scale = min(1.0, float(i) / 5000.0)
        if ((i+1 % 100000) == 0):
            learn_rate = learn_rate * 0.8
        GIP.set_all_sgd_params(lr_gn=scale*learn_rate, \
                lr_in=scale*learn_rate, mom_1=0.9, mom_2=0.999)
        GIP.set_lam_kld(0.05)
        GIP.set_lam_nll(1.0)
        # get some data to train with
        tr_idx = npr.randint(low=0,high=tr_samples,size=(100,))
        Xd_batch = Xtr.take(tr_idx, axis=0)
        Xd_batch = np.repeat(Xd_batch, 5, axis=0)
        Xc_batch = 0.0 * Xd_batch
        Xm_batch = 0.0 * Xd_batch
        # do a minibatch update of the model, and compute some costs
        outputs = GIP.train_joint(Xd_batch, Xc_batch, Xm_batch)
        joint_cost = 1.0 * outputs[0]
        data_nll_cost = 1.0 * outputs[1]
        post_kld_cost = 1.0 * outputs[2]
        other_reg_cost = 1.0 * outputs[3]
        if ((i % 1000) == 0):
            o_str = "batch: {0:d}, joint_cost: {1:.4f}, data_nll_cost: {2:.4f}, post_kld_cost: {3:.4f}, other_reg_cost: {4:.4f}".format( \
                    i, joint_cost, data_nll_cost, post_kld_cost, other_reg_cost)
            print(o_str)
            out_file.write(o_str+"\n")
            out_file.flush()
        if ((i % 5000) == 0):
            tr_idx = npr.randint(low=0,high=tr_samples,size=(100,))
            Xd_batch = Xtr.take(tr_idx, axis=0)
            file_name = "GIP_MNIST_PCA_CHAIN_SAMPLES_b{0:d}.png".format(i)
            Xd_samps = np.repeat(Xd_batch[0:10,:], 3, axis=0)
            sample_lists = GIP.sample_gil_from_data(Xd_samps, loop_iters=15)
            Xs = np.vstack(sample_lists["data samples"])
            utils.mat_to_img(Xs, file_name, (28,28), num_rows=15, \
                    scale=True, colorImg=False, tile_spacing=(1,1))
            # draw samples freely from the generative model's prior
            file_name = "GIP_MNIST_PCA_PRIOR_SAMPLES_b{0:d}.png".format(i)
            Xs = GIP.sample_from_gn(15*15)
            utils.mat_to_img(Xs, file_name, (28,28), num_rows=15, \
                    scale=True, colorImg=False, tile_spacing=(1,1))
            # draw inference net first layer weights
            file_name = "GIP_MNIST_PCA_INF_WEIGHTS_b{0:d}.png".format(i)
            utils.visualize_net_layer(GIP.IN.shared_layers[0], file_name, \
                    colorImg=False, use_transpose=False, transform=f_dec_np)
            # draw generator net final layer weights
            file_name = "GIP_MNIST_PCA_GEN_WEIGHTS_b{0:d}.png".format(i)
            if (gn_params['out_type'] == 'gaussian'):
                lay_num = -2
            else:
                lay_num = -1
            utils.visualize_net_layer(GIP.GN.mlp_layers[lay_num], file_name, \
                    colorImg=False, use_transpose=True, transform=f_dec_np)
    print("TESTING COMPLETE!")
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
            PCA_theano(Xtr, cutoff=750, global_sd=True)
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
    gn_params['encoder'] = f_enc
    gn_params['decoder'] = f_dec
    gn_params['out_type'] = 'gaussian'
    gn_params['init_scale'] = 1.3
    gn_params['lam_l2a'] = 1e-3
    gn_params['vis_drop'] = 0.0
    gn_params['hid_drop'] = 0.0
    gn_params['bias_noise'] = 0.0
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
    in_params['init_scale'] = 1.3
    in_params['lam_l2a'] = 1e-3
    in_params['vis_drop'] = 0.0
    in_params['hid_drop'] = 0.0
    in_params['bias_noise'] = 0.0
    in_params['input_noise'] = 0.0
    # Initialize the base networks for this GIPair
    IN = InfNet(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, prior_sigma=prior_sigma, \
            params=in_params, shared_param_dicts=None)
    GN = GenNet(rng=rng, Xp=Xp, prior_sigma=prior_sigma, \
            params=gn_params, shared_param_dicts=None)
    # Initialize biases in IN and GN
    IN.init_biases(0.0)
    GN.init_biases(0.0)
    # Initialize the GIPair
    GIP = GIPair(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, g_net=GN, i_net=IN, \
            data_dim=pca_dim, prior_dim=prior_dim, params=None)
    GIP.set_lam_l2w(1e-5)

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

    out_file = open("GIP_SVHN_PCA_AAA.txt", 'wb')
    # Set initial learning rate and basic SGD hyper parameters
    learn_rate = 0.0002
    for i in range(750000):
        scale = min(1.0, float(i) / 5000.0)
        if ((i+1 % 100000) == 0):
            learn_rate = learn_rate * 0.8
        GIP.set_all_sgd_params(lr_gn=scale*learn_rate, \
                lr_in=scale*learn_rate, mom_1=0.9, mom_2=0.999)
        GIP.set_lam_kld(0.1)
        GIP.set_lam_nll(1.0)
        # get some data to train with
        tr_idx = npr.randint(low=0,high=tr_samples,size=(500,))
        Xd_batch = Xtr.take(tr_idx, axis=0)
        Xd_batch = np.repeat(Xd_batch, 5, axis=0)
        Xc_batch = 0.0 * Xd_batch
        Xm_batch = 0.0 * Xd_batch

        # do a minibatch update of the model, and compute some costs
        outputs = GIP.train_joint(Xd_batch, Xc_batch, Xm_batch)
        joint_cost = 1.0 * outputs[0]
        data_nll_cost = 1.0 * outputs[1]
        post_kld_cost = 1.0 * outputs[2]
        other_reg_cost = 1.0 * outputs[3]
        if ((i % 1000) == 0):
            o_str = "batch: {0:d}, joint_cost: {1:.4f}, data_nll_cost: {2:.4f}, post_kld_cost: {3:.4f}, other_reg_cost: {4:.4f}".format( \
                    i, joint_cost, data_nll_cost, post_kld_cost, other_reg_cost)
            print(o_str)
            out_file.write(o_str+"\n")
            out_file.flush()
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
    #test_gi_pair_mnist_pca()
    test_gi_pair_mnist()
    
