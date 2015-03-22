import time
import numpy as np
import numpy.random as npr
import theano
import theano.tensor as T
import utils as utils
from load_data import load_udm, load_udm_ss, load_mnist, load_svhn, load_tfd
from InfNet import InfNet, load_infnet_from_file
from OneStageModel import OneStageModel
from GIPair2 import GIPair2
from GIStack import GIStack
from NetLayers import relu_actfun, softplus_actfun, \
                      safe_softmax, tanh_actfun, row_shuffle
from DKCode import PCA_theano
from VideoUtils import VideoSink


LOGVAR_BOUND = 6.0

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

def block_video(seq_matrix, im_dim, block_dim):
    b_rows = block_dim[0]
    b_cols = block_dim[1]
    i_rows = im_dim[0]
    i_cols = im_dim[1]
    seq_len = seq_matrix[0][0].shape[0]
    gap_px = 4
    block_im_dim = (b_rows*(i_rows+gap_px), b_cols*(i_cols+gap_px))
    # make the square-form multi-block video
    full_vid_sq = np.zeros((seq_len, block_im_dim[0], block_im_dim[1]))
    for i in range(seq_len):
        for br in range(b_rows):
            for bc in range(b_cols):
                r_start = br * (i_rows + gap_px)
                r_end = r_start + i_rows
                c_start = bc * (i_cols + gap_px)
                c_end = c_start + i_cols
                full_vid_sq[i, r_start:r_end, c_start:c_end] = \
                        seq_matrix[br][bc][i,:].reshape((i_rows, i_cols))
    full_vid_flat = np.zeros((seq_len, block_im_dim[0]*block_im_dim[1]))
    for i in range(seq_len):
        full_vid_flat[i,:] = full_vid_sq[i,:,:].ravel()
    return [full_vid_flat, block_im_dim]


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
    batch_size = 100
    Xtr_mean = np.mean(Xtr, axis=0, keepdims=True)
    Xtr_mean = (0.0 * Xtr_mean) + np.mean(Xtr)
    Xc_mean = np.repeat(Xtr_mean, batch_size, axis=0).astype(theano.config.floatX)

    # Symbolic inputs
    Xd = T.matrix(name='Xd')
    Xc = T.matrix(name='Xc')
    Xm = T.matrix(name='Xm')
    Xt = T.matrix(name='Xt')

    # Load inferencer and generator from saved parameters
    gn_fname = "MNIST_WALKOUT_TEST_MAX_KLD/pt_walk_params_b70000_GN.pkl"
    in_fname = "MNIST_WALKOUT_TEST_MAX_KLD/pt_walk_params_b70000_IN.pkl"
    IN = load_infnet_from_file(f_name=in_fname, rng=rng, Xd=Xd)
    GN = load_infnet_from_file(f_name=gn_fname, rng=rng, Xd=Xd)
    x_dim = IN.shared_layers[0].in_dim
    z_dim = IN.mu_layers[-1].out_dim
    # construct a GIPair with the loaded InfNet and GenNet
    osm_params = {}
    osm_params['x_type'] = 'gaussian'
    osm_params['xt_transform'] = 'sigmoid'
    osm_params['logvar_bound'] = LOGVAR_BOUND
    OSM = OneStageModel(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, \
            p_x_given_z=GN, q_z_given_x=IN, \
            x_dim=x_dim, z_dim=z_dim, params=osm_params)
    # compute variational likelihood bound and its sub-components
    Xva = row_shuffle(Xva)
    Xb = Xva[0:5000]
    file_name = "A_MNIST_POST_KLDS.png"
    post_klds = OSM.compute_post_klds(Xb)
    post_dim_klds = np.mean(post_klds, axis=0)
    utils.plot_stem(np.arange(post_dim_klds.shape[0]), post_dim_klds, \
            file_name)
    # compute information about free-energy on validation set
    file_name = "A_MNIST_FREE_ENERGY.png"
    fe_terms = OSM.compute_fe_terms(Xb, 20)
    utils.plot_scatter(fe_terms[1], fe_terms[0], file_name, \
            x_label='Posterior KLd', y_label='Negative Log-likelihood')

    # bound_results = OSM.compute_ll_bound(Xva)
    # ll_bounds = bound_results[0]
    # post_klds = bound_results[1]
    # log_likelihoods = bound_results[2]
    # max_lls = bound_results[3]
    # print("mean ll bound: {0:.4f}".format(np.mean(ll_bounds)))
    # print("mean posterior KLd: {0:.4f}".format(np.mean(post_klds)))
    # print("mean log-likelihood: {0:.4f}".format(np.mean(log_likelihoods)))
    # print("mean max log-likelihood: {0:.4f}".format(np.mean(max_lls)))
    # print("min ll bound: {0:.4f}".format(np.min(ll_bounds)))
    # print("max posterior KLd: {0:.4f}".format(np.max(post_klds)))
    # print("min log-likelihood: {0:.4f}".format(np.min(log_likelihoods)))
    # print("min max log-likelihood: {0:.4f}".format(np.min(max_lls)))
    # # compute some information about the approximate posteriors
    # post_stats = OSM.compute_post_stats(Xva, 0.0*Xva, 0.0*Xva)
    # all_post_klds = np.sort(post_stats[0].ravel()) # post KLds for each obs and dim
    # obs_post_klds = np.sort(post_stats[1]) # summed post KLds for each obs
    # post_dim_klds = post_stats[2] # average post KLds for each post dim
    # post_dim_vars = post_stats[3] # average squared mean for each post dim
    # utils.plot_line(np.arange(all_post_klds.shape[0]), all_post_klds, "AAA_ALL_POST_KLDS.png")
    # utils.plot_line(np.arange(obs_post_klds.shape[0]), obs_post_klds, "AAA_OBS_POST_KLDS.png")
    # utils.plot_stem(np.arange(post_dim_klds.shape[0]), post_dim_klds, "AAA_POST_DIM_KLDS.png")
    # utils.plot_stem(np.arange(post_dim_vars.shape[0]), post_dim_vars, "AAA_POST_DIM_VARS.png")

    # draw many samples from the GIP
    for i in range(5):
        tr_idx = npr.randint(low=0,high=tr_samples,size=(100,))
        Xd_batch = Xtr.take(tr_idx, axis=0)
        Xs = []
        for row in range(3):
            Xs.append([])
            for col in range(3):
                sample_lists = OSM.sample_from_chain(Xd_batch[0:10,:], loop_iters=100, \
                        sigma_scale=1.0)
                Xs[row].append(group_chains(sample_lists['data samples']))
        Xs, block_im_dim = block_video(Xs, (28,28), (3,3))
        to_video(Xs, block_im_dim, "A_MNIST_KLD_CHAIN_VIDEO_{0:d}.avi".format(i), frame_rate=10)
        #sample_lists = GIP.sample_from_chain(Xd_batch[0,:].reshape((1,data_dim)), loop_iters=300, \
        #        sigma_scale=1.0)
        #Xs = np.vstack(sample_lists["data samples"])
        #file_name = "TFD_TEST_{0:d}.png".format(i)
        #utils.visualize_samples(Xs, file_name, num_rows=15)
    file_name = "A_MNIST_KLD_PRIOR_SAMPLE.png"
    Xs = OSM.sample_from_prior(20*20)
    utils.visualize_samples(Xs, file_name, num_rows=20)
    # # test Parzen density estimator built from prior samples
    # Xs = OSM.sample_from_prior(10000)
    # [best_sigma, best_ll, best_lls] = \
    #         cross_validate_sigma(Xs, Xva, [0.12, 0.14, 0.15, 0.16, 0.18], 20)
    # sort_idx = np.argsort(best_lls)
    # sort_idx = sort_idx[0:400]
    # utils.plot_line(np.arange(sort_idx.shape[0]), best_lls[sort_idx], "A_MNIST_BEST_LLS_1.png")
    # utils.visualize_samples(Xva[sort_idx], "A_MNIST_BAD_DIGITS_1.png", num_rows=20)
    # ##########
    # # AGAIN! #
    # ##########
    # Xs = OSM.sample_from_prior(10000)
    # tr_idx = npr.randint(low=0,high=tr_samples,size=(5000,))
    # Xva = Xtr.take(tr_idx, axis=0)
    # [best_sigma, best_ll, best_lls] = \
    #         cross_validate_sigma(Xs, Xva, [0.12, 0.14, 0.15, 0.16, 0.18], 20)
    # sort_idx = np.argsort(best_lls)
    # sort_idx = sort_idx[0:400]
    # utils.plot_line(np.arange(sort_idx.shape[0]), best_lls[sort_idx], "A_MNIST_BEST_LLS_2.png")
    # utils.visualize_samples(Xva[sort_idx], "A_MNIST_BAD_DIGITS_2.png", num_rows=20)
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

    # Load inferencer and generator from saved parameters
    gn_fname = "TFD_WALKOUT_TEST_KLD/pt_walk_params_b25000_GN.pkl"
    in_fname = "TFD_WALKOUT_TEST_KLD/pt_walk_params_b25000_IN.pkl"
    IN = load_infnet_from_file(f_name=in_fname, rng=rng, Xd=Xd)
    GN = load_infnet_from_file(f_name=gn_fname, rng=rng, Xd=Xd)
    x_dim = IN.shared_layers[0].in_dim
    z_dim = IN.mu_layers[-1].out_dim
    # construct a GIPair with the loaded InfNet and GenNet
    osm_params = {}
    osm_params['x_type'] = 'gaussian'
    osm_params['xt_transform'] = 'sigmoid'
    osm_params['logvar_bound'] = LOGVAR_BOUND
    OSM = OneStageModel(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, \
            p_x_given_z=GN, q_z_given_x=IN, \
            x_dim=x_dim, z_dim=z_dim, params=osm_params)

    # # compute variational likelihood bound and its sub-components
    Xva = row_shuffle(Xva)
    Xb = Xva[0:5000]
    # file_name = "A_TFD_POST_KLDS.png"
    # post_klds = OSM.compute_post_klds(Xb)
    # post_dim_klds = np.mean(post_klds, axis=0)
    # utils.plot_stem(np.arange(post_dim_klds.shape[0]), post_dim_klds, \
    #         file_name)
    # compute information about free-energy on validation set
    file_name = "A_TFD_KLD_FREE_ENERGY.png"
    fe_terms = OSM.compute_fe_terms(Xb, 20)
    utils.plot_scatter(fe_terms[1], fe_terms[0], file_name, \
            x_label='Posterior KLd', y_label='Negative Log-likelihood')

    # bound_results = OSM.compute_ll_bound(Xva)
    # ll_bounds = bound_results[0]
    # post_klds = bound_results[1]
    # log_likelihoods = bound_results[2]
    # max_lls = bound_results[3]
    # print("mean ll bound: {0:.4f}".format(np.mean(ll_bounds)))
    # print("mean posterior KLd: {0:.4f}".format(np.mean(post_klds)))
    # print("mean log-likelihood: {0:.4f}".format(np.mean(log_likelihoods)))
    # print("mean max log-likelihood: {0:.4f}".format(np.mean(max_lls)))
    # print("min ll bound: {0:.4f}".format(np.min(ll_bounds)))
    # print("max posterior KLd: {0:.4f}".format(np.max(post_klds)))
    # print("min log-likelihood: {0:.4f}".format(np.min(log_likelihoods)))
    # print("min max log-likelihood: {0:.4f}".format(np.min(max_lls)))
    # # compute some information about the approximate posteriors
    # post_stats = OSM.compute_post_stats(Xva, 0.0*Xva, 0.0*Xva)
    # all_post_klds = np.sort(post_stats[0].ravel()) # post KLds for each obs and dim
    # obs_post_klds = np.sort(post_stats[1]) # summed post KLds for each obs
    # post_dim_klds = post_stats[2] # average post KLds for each post dim
    # post_dim_vars = post_stats[3] # average squared mean for each post dim
    # utils.plot_line(np.arange(all_post_klds.shape[0]), all_post_klds, "AAA_ALL_POST_KLDS.png")
    # utils.plot_line(np.arange(obs_post_klds.shape[0]), obs_post_klds, "AAA_OBS_POST_KLDS.png")
    # utils.plot_stem(np.arange(post_dim_klds.shape[0]), post_dim_klds, "AAA_POST_DIM_KLDS.png")
    # utils.plot_stem(np.arange(post_dim_vars.shape[0]), post_dim_vars, "AAA_POST_DIM_VARS.png")

    # draw many samples from the GIP
    for i in range(5):
        tr_idx = npr.randint(low=0,high=tr_samples,size=(100,))
        Xd_batch = Xtr.take(tr_idx, axis=0)
        Xs = []
        for row in range(3):
            Xs.append([])
            for col in range(3):
                sample_lists = OSM.sample_from_chain(Xd_batch[0:10,:], loop_iters=100, \
                        sigma_scale=1.0)
                Xs[row].append(group_chains(sample_lists['data samples']))
        Xs, block_im_dim = block_video(Xs, (48,48), (3,3))
        to_video(Xs, block_im_dim, "A_TFD_KLD_CHAIN_VIDEO_{0:d}.avi".format(i), frame_rate=10)
        #sample_lists = GIP.sample_from_chain(Xd_batch[0,:].reshape((1,data_dim)), loop_iters=300, \
        #        sigma_scale=1.0)
        #Xs = np.vstack(sample_lists["data samples"])
        #file_name = "TFD_TEST_{0:d}.png".format(i)
        #utils.visualize_samples(Xs, file_name, num_rows=15)
    file_name = "A_TFD_KLD_PRIOR_SAMPLE.png"
    Xs = OSM.sample_from_prior(20*20)
    utils.visualize_samples(Xs, file_name, num_rows=20)

    # test Parzen density estimator built from prior samples
    # Xs = OSM.sample_from_prior(10000)
    # [best_sigma, best_ll, best_lls] = \
    #         cross_validate_sigma(Xs, Xva, [0.09, 0.095, 0.1, 0.105, 0.11], 10)
    # sort_idx = np.argsort(best_lls)
    # sort_idx = sort_idx[0:400]
    # utils.plot_line(np.arange(sort_idx.shape[0]), best_lls[sort_idx], "A_TFD_BEST_LLS_1.png")
    # utils.visualize_samples(Xva[sort_idx], "A_TFD_BAD_FACES_1.png", num_rows=20)
    # ##########
    # # AGAIN! #
    # ##########
    # Xs = OSM.sample_from_prior(10000)
    # tr_idx = npr.randint(low=0,high=tr_samples,size=(5000,))
    # Xva = Xtr.take(tr_idx, axis=0)
    # [best_sigma, best_ll, best_lls] = \
    #         cross_validate_sigma(Xs, Xva, [0.09, 0.095, 0.1, 0.105, 0.11], 10)
    # sort_idx = np.argsort(best_lls)
    # sort_idx = sort_idx[0:400]
    # utils.plot_line(np.arange(sort_idx.shape[0]), best_lls[sort_idx], "A_TFD_BEST_LLS_2.png")
    # utils.visualize_samples(Xva[sort_idx], "A_TFD_BAD_FACES_2.png", num_rows=20)
    return

###################
# TEST DISPATCHER #
###################

if __name__=="__main__":
    #test_gip_sigma_scale_mnist()
    test_gip_sigma_scale_tfd()
    
