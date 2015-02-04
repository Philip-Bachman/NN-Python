import time
import utils as utils
import numpy as np
import numpy.random as npr
import theano
import theano.tensor as T

from load_data import load_udm, load_udm_ss, load_mnist
from PeaNet import PeaNet, load_peanet_from_file
from InfNet import InfNet, load_infnet_from_file
from GenNet import GenNet, load_gennet_from_file
from VCGLoop import VCGLoop
from GIPair import GIPair
from NetLayers import relu_actfun, softplus_actfun, \
                      safe_softmax, safe_log
import GenNet as GNet
import InfNet as INet
import PeaNet as PNet
from DKCode import PCA_theano
from LogPDFs import cross_validate_sigma



import sys, resource
resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
sys.setrecursionlimit(10**6)

#KLD_PATH = "TFD_WALKOUT_TEST_KLD/"
#VAE_PATH = "TFD_WALKOUT_TEST_VAE/"

####################
# HELPER FUNCTIONS #
####################

def draw_posterior_kld_hist(X_kld, X_vae, f_name, bins=25):
    """
    Plot KDE-smoothed histograms.
    """
    import matplotlib.pyplot as plt
    # make a figure and configure an axis
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Posterior KLd Density')
    ax.set_title('Posterior KLds: Regularized vs. Standard')
    ax.hold(True)
    for (X, style, label) in [(X_kld, '-', 'ORK'), (X_vae, '--', 'VAR')]:
        X_samp = X.ravel()[:,np.newaxis]
        X_min = np.min(X_samp)
        X_max = np.max(X_samp)
        X_range = X_max - X_min
        sigma = X_range / float(bins)
        plot_min = X_min - (X_range/3.0)
        plot_max = X_max + (X_range/3.0)
        plot_X = np.linspace(plot_min, plot_max, 1000)[:,np.newaxis]
        # make a kernel density estimator for the data in X
        kde = KernelDensity(kernel='gaussian', bandwidth=sigma).fit(X_samp)
        ax.plot(plot_X, np.exp(kde.score_samples(plot_X)), linestyle=style, label=label)
    ax.legend()
    fig.savefig(f_name, dpi=None, facecolor='w', edgecolor='w', \
        orientation='portrait', papertype=None, format=None, \
        transparent=False, bbox_inches=None, pad_inches=0.1, \
        frameon=None)
    plt.close(fig)
    return

def draw_parzen_vs_variational_scatter(p_vals_kld, v_vals_kld, p_vals_vae, v_vals_vae, f_name=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Variational Bound')
    ax.set_ylabel('Parzen Bound')
    ax.set_title('Comparing Parzen vs. Variational Bounds')
    ax.hold(True)
    ax.scatter(v_vals_kld, p_vals_kld, s=32, c=u'b', marker=u'o', label='ORK')
    ax.scatter(v_vals_vae, p_vals_vae, s=32, c=u'r', marker=u'+', label='VAR')
    ax.legend()
    fig.savefig(f_name, dpi=None, facecolor='w', edgecolor='w', \
        orientation='portrait', papertype=None, format=None, \
        transparent=False, bbox_inches=None, pad_inches=0.1, \
        frameon=None)
    plt.close(fig)
    return

def draw_kld_vs_likelihood_scatter(kl_vals_kld, ll_vals_kld, kl_vals_vae, ll_vals_vae, f_name=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Posterior KLd')
    ax.set_ylabel('Log-likelihood')
    ax.set_title('Trading KLd for Log-likelihood')
    ax.hold(True)
    ax.scatter(kl_vals_kld, ll_vals_kld, s=16, alpha=0.5, c=u'b', marker=u'o', label='ORK')
    ax.scatter(kl_vals_vae, ll_vals_vae, s=16, alpha=0.5, c=u'r', marker=u'+', label='VAR')
    ax.legend()
    fig.savefig(f_name, dpi=None, facecolor='w', edgecolor='w', \
        orientation='portrait', papertype=None, format=None, \
        transparent=False, bbox_inches=None, pad_inches=0.1, \
        frameon=None)
    plt.close(fig)
    return

def sample_masks(X, drop_prob=0.3):
    """
    Sample a binary mask to apply to the matrix X, with rate mask_prob.
    """
    probs = npr.rand(*X.shape)
    mask = 1.0 * (probs > drop_prob)
    return mask.astype(theano.config.floatX)

def sample_patch_masks(X, im_shape, patch_shape):
    """
    Sample a random patch mask for each image in X.
    """
    obs_count = X.shape[0]
    rs = patch_shape[0]
    cs = patch_shape[1]
    off_row = npr.randint(1,high=(im_shape[0]-rs-1), size=(obs_count,))
    off_col = npr.randint(1,high=(im_shape[1]-cs-1), size=(obs_count,))
    dummy = np.zeros(im_shape)
    mask = np.zeros(X.shape)
    for i in range(obs_count):
        dummy = (0.0 * dummy) + 1.0
        dummy[off_row[i]:(off_row[i]+rs), off_col[i]:(off_col[i]+cs)] = 0.0
        mask[i,:] = dummy.ravel()
    return mask.astype(theano.config.floatX)

def posterior_klds(IN, Xtr, batch_size, batch_count):
    """
    Get posterior KLd cost for some inputs from Xtr.
    """
    post_klds = []
    for i in range(batch_count):
        batch_idx = npr.randint(low=0, high=Xtr.shape[0], size=(batch_size,))
        X = Xtr.take(batch_idx, axis=0)
        post_klds.extend([k for k in IN.kld_func(X, 0.0*X, 0.0*X)])
    return post_klds

#########################################
#########################################
## CHECK RESULTS OF VAE AND KLD MODELS ##
#########################################
#########################################

def check_mnist_results():
    # DERPA DERPA DOO
    KLD_PATH = "MNIST_WALKOUT_TEST_KLD/"
    VAE_PATH = "MNIST_WALKOUT_TEST_VAE/"
    RESULT_PATH = "MNIST_WALKOUT_RESULTS/"

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
    prior_sigma = 1.0

    p_vals_kld, v_vals_kld, p_vals_vae, v_vals_vae = [], [], [], []
    kl_vals_kld, ll_vals_kld, kl_vals_vae, ll_vals_vae = [], [], [], []

    ########################################################
    # CHECK MODEL BEHAVIOR AT DIFFERENT STAGES OF TRAINING #
    ########################################################
    for i in range(10000,200000):
        if ((i % 10000) == 0):
            if (i <= 80000):
                net_type = 'gip'
                b = i
            else:
                net_type = 'walk'
                b = i - 80000
            #############################################################
            # Process the GIPair trained with strong KLd regularization #
            #############################################################
            gn_fname = KLD_PATH + "pt_{0:s}_params_b{1:d}_GN.pkl".format(net_type, b)
            in_fname = KLD_PATH + "pt_{0:s}_params_b{1:d}_IN.pkl".format(net_type, b)
            IN = INet.load_infnet_from_file(f_name=in_fname, rng=rng, Xd=Xd, Xc=Xc, Xm=Xm)
            GN = GNet.load_gennet_from_file(f_name=gn_fname, rng=rng, Xp=Xp)
            IN.set_sigma_scale(1.0)
            prior_dim = GN.latent_dim
            post_klds_kld = posterior_klds(IN, Xtr, 5000, 5)
            # Initialize the GIPair
            GIP_KLD = GIPair(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, g_net=GN, i_net=IN, \
                    data_dim=data_dim, prior_dim=prior_dim, params=None)
            GIP_KLD.set_lam_l2w(1e-4)
            GIP_KLD.set_lam_nll(1.0)
            GIP_KLD.set_lam_kld(1.0)
            # draw some sequential samples from the self-loop chain
            tr_idx = npr.randint(low=0,high=tr_samples,size=(100,))
            Xd_batch = Xtr.take(tr_idx, axis=0)
            Xd_samps = np.repeat(Xd_batch[0:10,:], 3, axis=0)
            sample_lists = GIP_KLD.sample_from_chain(Xd_samps, loop_iters=20)
            Xs = np.vstack(sample_lists["data samples"])
            file_name = RESULT_PATH + "chain_samples_b{0:d}_kld.png".format(i)
            utils.visualize_samples(Xs, file_name, num_rows=20)
            # draw samples freely from the generative model's prior
            Xs = GIP_KLD.sample_from_prior(20*20)
            file_name = RESULT_PATH + "prior_samples_b{0:d}_kld.png".format(i)
            utils.visualize_samples(Xs, file_name, num_rows=20)
            # test Parzen density estimator built from prior samples
            Xs = GIP_KLD.sample_from_prior(10000, sigma=1.0)
            parzen_vals_kld = cross_validate_sigma(Xs, Xva, [0.1, 0.13, 0.15, 0.18, 0.2], 50)
            # get variational bound info
            var_vals_kld = GIP_KLD.compute_ll_bound(Xva)
            # record info about variational and parzen bounds
            p_vals_kld.append(parzen_vals_kld[1])
            v_vals_kld.append(np.mean(var_vals_kld[0]))
            ################################################################
            # Process the GIPair trained with basic VAE KLd regularization #
            ################################################################
            gn_fname = VAE_PATH + "pt_{0:s}_params_b{1:d}_GN.pkl".format(net_type, b)
            in_fname = VAE_PATH + "pt_{0:s}_params_b{1:d}_IN.pkl".format(net_type, b)
            IN = INet.load_infnet_from_file(f_name=in_fname, rng=rng, Xd=Xd, Xc=Xc, Xm=Xm)
            GN = GNet.load_gennet_from_file(f_name=gn_fname, rng=rng, Xp=Xp)
            IN.set_sigma_scale(1.0)
            prior_dim = GN.latent_dim
            post_klds_vae = posterior_klds(IN, Xtr, 5000, 5)
            # Initialize the GIPair
            GIP_VAE = GIPair(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, g_net=GN, i_net=IN, \
                    data_dim=data_dim, prior_dim=prior_dim, params=None)
            GIP_VAE.set_lam_l2w(1e-4)
            GIP_VAE.set_lam_nll(1.0)
            GIP_VAE.set_lam_kld(1.0)
            # draw some sequential samples from the self-loop chain
            tr_idx = npr.randint(low=0,high=tr_samples,size=(100,))
            Xd_batch = Xtr.take(tr_idx, axis=0)
            Xd_samps = np.repeat(Xd_batch[0:10,:], 3, axis=0)
            sample_lists = GIP_VAE.sample_from_chain(Xd_samps, loop_iters=20)
            Xs = np.vstack(sample_lists["data samples"])
            file_name = RESULT_PATH + "chain_samples_b{0:d}_vae.png".format(i)
            utils.visualize_samples(Xs, file_name, num_rows=20)
            # draw samples freely from the generative model's prior
            Xs = GIP_VAE.sample_from_prior(20*20)
            file_name = RESULT_PATH + "prior_samples_b{0:d}_vae.png".format(i)
            utils.visualize_samples(Xs, file_name, num_rows=20)
            # test Parzen density estimator built from prior samples
            Xs = GIP_VAE.sample_from_prior(10000, sigma=1.0)
            parzen_vals_vae = cross_validate_sigma(Xs, Xva, [0.12, 0.15, 0.18, 0.20, 0.25], 50)
            # get variational bound info
            var_vals_vae = GIP_VAE.compute_ll_bound(Xva)
            # record info about variational and parzen bounds
            p_vals_vae.append(parzen_vals_vae[1])
            v_vals_vae.append(np.mean(var_vals_vae[0]))
            ########################
            # Plot posterior KLds. #
            ########################
            file_name = RESULT_PATH + "post_klds_b{0:d}.png".format(i)
            draw_posterior_kld_hist( \
                    np.asarray(post_klds_kld), np.asarray(post_klds_vae), file_name, bins=30)
            if i in [20000, 50000, 80000, 110000, 150000, (200000-1)]:
                # select random random indices into the validation set
                va_idx = npr.randint(0,high=va_samples,size=(200,))
                # record information about their current variational bounds
                kl_vals_kld.extend([v for v in var_vals_kld[1][va_idx]])
                ll_vals_kld.extend([v for v in var_vals_kld[2][va_idx]])
                kl_vals_vae.extend([v for v in var_vals_vae[1][va_idx]])
                ll_vals_vae.extend([v for v in var_vals_vae[2][va_idx]])
                # do some plotting
                s1_name = RESULT_PATH + "parzen_vs_variational.png"
                s2_name = RESULT_PATH + "kld_vs_likelihood.png"
                draw_parzen_vs_variational_scatter(p_vals_kld, v_vals_kld, \
                        p_vals_vae, v_vals_vae, f_name=s1_name)
                draw_kld_vs_likelihood_scatter(kl_vals_kld, ll_vals_kld, \
                        kl_vals_vae, ll_vals_vae, f_name=s2_name)
    return

#####################################################
# Train a VCGLoop starting from a pretrained GIPair #
#####################################################

def train_walk_from_pretrained_gip(extra_lam_kld=0.0):
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
    batch_reps = 5
    prior_sigma = 1.0
    Xtr_mean = np.mean(Xtr, axis=0, keepdims=True)
    Xtr_mean = (0.0 * Xtr_mean) + np.mean(Xtr)
    Xc_mean = np.repeat(Xtr_mean, batch_size, axis=0).astype(theano.config.floatX)


    out_file = open(RESULT_PATH+"pt_walk_results.txt", 'wb')
    ####################################################
    # Train the VCGLoop by unrolling and applying BPTT #
    ####################################################
    learn_rate = 0.0003
    cost_1 = [0. for i in range(10)]
    for i in range(200000):
        if ((i % 5000) == 0):
            tr_idx = npr.randint(low=0,high=Xtr.shape[0],size=(5,))
            va_idx = npr.randint(low=0,high=Xva.shape[0],size=(5,))
            Xd_batch = np.vstack([Xtr.take(tr_idx, axis=0), Xva.take(va_idx, axis=0)])
            # draw some chains of samples from the VAE loop
            file_name = RESULT_PATH+"pt_walk_chain_samples_b{0:d}.png".format(i)
            Xd_samps = np.repeat(Xd_batch, 3, axis=0)
            sample_lists = VCGL.GIP.sample_from_chain(Xd_samps, loop_iters=20)
            Xs = np.vstack(sample_lists["data samples"])
            utils.visualize_samples(Xs, file_name, num_rows=20)
            # draw some masked chains of samples from the VAE loop
            file_name = RESULT_PATH+"pt_walk_mask_samples_b{0:d}.png".format(i)
            Xd_samps = np.repeat(Xc_mean[0:Xd_batch.shape[0],:], 3, axis=0)
            Xc_samps = np.repeat(Xd_batch, 3, axis=0)
            Xm_rand = sample_masks(Xc_samps, drop_prob=0.3)
            Xm_patch = sample_patch_masks(Xc_samps, (28,28), (14,14))
            Xm_samps = Xm_rand * Xm_patch
            sample_lists = VCGL.GIP.sample_from_chain(Xd_samps, \
                    X_c=Xc_samps, X_m=Xm_samps, loop_iters=20)
            Xs = np.vstack(sample_lists["data samples"])
            utils.visualize_samples(Xs, file_name, num_rows=20)
            # draw some samples independently from the GenNet's prior
            file_name = RESULT_PATH+"pt_walk_prior_samples_b{0:d}.png".format(i)
            Xs = VCGL.sample_from_prior(20*20)
            utils.visualize_samples(Xs, file_name, num_rows=20)
            # draw discriminator network's weights
            file_name = RESULT_PATH+"pt_walk_dis_weights_b{0:d}.png".format(i)
            utils.visualize_net_layer(VCGL.DN.proto_nets[0][0], file_name)
            # draw inference net first layer weights
            file_name = RESULT_PATH+"pt_walk_inf_weights_b{0:d}.png".format(i)
            utils.visualize_net_layer(VCGL.IN.shared_layers[0], file_name)
            # draw generator net final layer weights
            file_name = RESULT_PATH+"pt_walk_gen_weights_b{0:d}.png".format(i)
            if GN.out_type == 'sigmoid':
                utils.visualize_net_layer(VCGL.GN.mlp_layers[-1], file_name, use_transpose=True)
            else:
                utils.visualize_net_layer(VCGL.GN.mlp_layers[-2], file_name, use_transpose=True)
            #########################
            # Check posterior KLds. #
            #########################
            post_klds = posterior_klds(IN, Xtr, 5000, 5)
            file_name = RESULT_PATH+"pt_walk_post_klds_b{0:d}.png".format(i)
            utils.plot_kde_histogram2( \
                    np.asarray(post_klds), np.asarray(post_klds), file_name, bins=30)
    return


if __name__=="__main__":
    # MAKE SURE TO SET RESULT_PATH FOR THE PROPER TEST
    check_mnist_results()