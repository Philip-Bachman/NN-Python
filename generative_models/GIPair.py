################################################################
# Code for managing and training a generator/inferencer pair.  #
################################################################

# basic python
import numpy as np
import numpy.random as npr
from collections import OrderedDict

# theano business
import theano
import theano.tensor as T
#from theano.tensor.shared_randomstreams import RandomStreams as RandStream
from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams as RandStream

# phil's sweetness
from NetLayers import HiddenLayer, DiscLayer, relu_actfun, softplus_actfun, \
                      safe_log
from GenNet import GenNet
from InfNet import InfNet
from PeaNet import PeaNet
from DKCode import get_adam_updates, get_adadelta_updates

#
#
# Important symbolic variables:
#   Xd: Xd represents input at the "data variables" of the inferencer
#   Xc: Xc represents input at the "control variables" of the inferencer
#   Xm: Xm represents input at the "mask variables" of the inferencer
#
#

class GIPair(object):
    """
    Controller for training a variational autoencoder.

    The generator must be an instance of the GenNet class implemented in
    "GenNet.py". The inferencer must be an instance of the InfNet class
    implemented in "InfNet.py".

    Parameters:
        rng: numpy.random.RandomState (for reproducibility)
        Xd: symbolic "data" input to this VAE
        Xc: symbolic "control" input to this VAE
        Xm: symbolic "mask" input to this VAE
        g_net: The GenNet instance that will serve as the base generator
        i_net: The InfNet instance that will serve as the base inferer
        data_dim: dimension of the "observable data" variables
        prior_dim: dimension of the "latent prior" variables
        params: dict for passing additional parameters
        shared_param_dicts: dict for retrieving some shared parameters required
                            by a GIPair. if this parameter is passed, then this
                            GIPair will be initialized as a "shared-parameter"
                            clone of some other GIPair.
    """
    def __init__(self, rng=None, \
            Xd=None, Xc=None, Xm=None, \
            g_net=None, i_net=None, \
            data_dim=None, prior_dim=None, \
            params=None, shared_param_dicts=None):
        # setup a rng for this GIPair
        self.rng = RandStream(rng.randint(100000))
        if params is None:
            self.params = {}
        else:
            self.params = params

        # record the symbolic variables that will provide inputs to the
        # computation graph created to describe this GIPair
        self.Xd = Xd
        self.Xc = Xc
        self.Xm = Xm
        # check whether we'll be working with "encoded" inputs
        self.use_encoder = i_net.use_encoder
        print("i_net.use_encoder: {0:s}, g_net.use_decoder: {1:s}".format( \
                str(i_net.use_encoder), str(g_net.use_decoder)))
        assert(self.use_encoder == g_net.use_decoder)
        # create a "shared-parameter" clone of the inferencer, set up to
        # receive input from the appropriate symbolic variables.
        self.IN = i_net.shared_param_clone(rng=rng, \
                Xd=self.Xd, Xc=self.Xc, Xm=self.Xm)
        self.posterior_means = self.IN.output_mean
        self.posterior_sigmas = self.IN.output_sigma
        self.posterior_norms = T.sqrt(T.sum(self.posterior_means**2.0, axis=1, keepdims=1))
        self.posterior_klds = self.IN.kld_cost
        # capture a handle for samples from the variational posterior
        self.Xp = self.IN.output
        # create a "shared-parameter" clone of the generator, set up to
        # receive input from samples from the variational posterior
        self.GN = g_net.shared_param_clone(rng=rng, Xp=self.IN.output)
        # capture a handle for sampled reconstructions from the generator
        self.Xg = self.GN.output

        # record and validate the data dimensionality parameters
        self.data_dim = data_dim
        self.prior_dim = prior_dim
        # output of the generator and input to the inferencer should both be
        # equal to self.data_dim
        assert(self.data_dim == self.GN.mlp_layers[-1].out_dim)
        assert(self.data_dim == self.IN.shared_layers[0].in_dim)
        # input of the generator and mu/sigma outputs of the inferencer should
        # both be equal to self.prior_dim
        assert(self.prior_dim == self.GN.mlp_layers[0].in_dim)
        assert(self.prior_dim == self.IN.mu_layers[-1].out_dim)
        assert(self.prior_dim == self.IN.sigma_layers[-1].out_dim)

        # determine whether this GIPair is a clone or an original
        if shared_param_dicts is None:
            # This is not a clone, and we will need to make a dict for
            # referring to the parameters of each network layer
            self.shared_param_dicts = {}
            self.is_clone = False
        else:
            # This is a clone, and its layer parameters can be found by
            # referring to the given param dict (i.e. shared_param_dicts).
            self.shared_param_dicts = shared_param_dicts
            self.is_clone = True

        if not self.is_clone:
            # shared var learning rate for generator and inferencer
            zero_ary = np.zeros((1,)).astype(theano.config.floatX)
            self.lr_gn = theano.shared(value=zero_ary, name='gip_lr_gn')
            self.lr_in = theano.shared(value=zero_ary, name='gip_lr_in')
            # shared var momentum parameters for generator and inferencer
            self.mom_1 = theano.shared(value=zero_ary, name='gip_mom_1')
            self.mom_2 = theano.shared(value=zero_ary, name='gip_mom_2')
            self.it_count = theano.shared(value=zero_ary, name='gip_it_count')
            # init parameters for controlling learning dynamics
            self.set_all_sgd_params()
            # init shared var for weighting nll of data given posterior sample
            self.lam_nll = theano.shared(value=zero_ary, name='gip_lam_nll')
            self.set_lam_nll(lam_nll=1.0)
            # init shared var for weighting prior kld against reconstruction
            self.lam_kld = theano.shared(value=zero_ary, name='gip_lam_kld')
            self.set_lam_kld(lam_kld=1.0)
            # init shared var for controlling l2 regularization on params
            self.lam_l2w = theano.shared(value=zero_ary, name='gip_lam_l2w')
            self.set_lam_l2w(1e-4)
            # record shared parameters that are to be shared among clones
            self.shared_param_dicts['gip_lr_gn'] = self.lr_gn
            self.shared_param_dicts['gip_lr_in'] = self.lr_in
            self.shared_param_dicts['gip_mom_1'] = self.mom_1
            self.shared_param_dicts['gip_mom_2'] = self.mom_2
            self.shared_param_dicts['gip_it_count'] = self.it_count
            self.shared_param_dicts['gip_lam_nll'] = self.lam_nll
            self.shared_param_dicts['gip_lam_kld'] = self.lam_kld
            self.shared_param_dicts['gip_lam_l2w'] = self.lam_l2w
        else:
            # use some shared parameters that are shared among all clones of
            # some "base" GIPair
            self.lr_gn = self.shared_param_dicts['gip_lr_gn']
            self.lr_in = self.shared_param_dicts['gip_lr_in']
            self.mom_1 = self.shared_param_dicts['gip_mom_1']
            self.mom_2 = self.shared_param_dicts['gip_mom_2']
            self.it_count = self.shared_param_dicts['gip_it_count']
            self.lam_nll = self.shared_param_dicts['gip_lam_nll']
            self.lam_kld = self.shared_param_dicts['gip_lam_kld']
            self.lam_l2w = self.shared_param_dicts['gip_lam_l2w']

        # Grab the full set of "optimizable" parameters from the generator
        # and inferencer networks that we'll be working with.
        self.in_params = [p for p in self.IN.mlp_params]
        self.gn_params = [p for p in self.GN.mlp_params]
        self.joint_params = self.in_params + self.gn_params

        ###################################
        # CONSTRUCT THE COSTS TO OPTIMIZE #
        ###################################
        self.data_nll_cost = self.lam_nll[0] * self._construct_data_nll_cost()
        self.post_kld_cost = self.lam_kld[0] * self._construct_post_kld_cost(kc2_scale=0.1)
        self.other_reg_cost = self._construct_other_reg_cost()
        self.joint_cost = self.data_nll_cost + self.post_kld_cost + \
                self.other_reg_cost

        # Get the gradient of the joint cost for all optimizable parameters
        self.joint_grads = OrderedDict()
        for p in self.joint_params:
            self.joint_grads[p] = T.grad(self.joint_cost, p).clip(-0.1, 0.1)

        # Construct the updates for the generator and inferencer networks
        self.gn_updates = get_adam_updates(params=self.gn_params, \
                grads=self.joint_grads, alpha=self.lr_gn, \
                beta1=self.mom_1, beta2=self.mom_2, it_count=self.it_count, \
                mom2_init=1e-3, smoothing=1e-8)
        self.in_updates = get_adam_updates(params=self.in_params, \
                grads=self.joint_grads, alpha=self.lr_in, \
                beta1=self.mom_1, beta2=self.mom_2, it_count=self.it_count, \
                mom2_init=1e-3, smoothing=1e-8)
        self.joint_updates = OrderedDict()
        for k in self.gn_updates:
            self.joint_updates[k] = self.gn_updates[k]
        for k in self.in_updates:
            self.joint_updates[k] = self.in_updates[k]
        self.joint_updates[self.IN.kld_mean] = self.IN.kld_mean_update

        # Construct a function for jointly training the generator/inferencer
        self.train_joint = self._construct_train_joint()
        self.compute_costs = self._construct_compute_costs()
        return

    def set_all_sgd_params(self, lr_gn=0.01, lr_in=0.01, \
                mom_1=0.9, mom_2=0.999):
        """
        Set learning rate and momentum parameter for all updates.
        """
        zero_ary = np.zeros((1,))
        # set learning rates
        new_lr_gn = zero_ary + lr_gn
        self.lr_gn.set_value(new_lr_gn.astype(theano.config.floatX))
        new_lr_in = zero_ary + lr_in
        self.lr_in.set_value(new_lr_in.astype(theano.config.floatX))
        # set momentums
        new_mom_1 = zero_ary + mom_1
        self.mom_1.set_value(new_mom_1.astype(theano.config.floatX))
        new_mom_2 = zero_ary + mom_2
        self.mom_2.set_value(new_mom_2.astype(theano.config.floatX))
        return

    def set_lam_nll(self, lam_nll=1.0):
        """
        Set weight for controlling the influence of the data likelihood.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_nll
        self.lam_nll.set_value(new_lam.astype(theano.config.floatX))
        return

    def set_lam_kld(self, lam_kld=1.0):
        """
        Set the relative weight of prior KL-divergence vs. data likelihood.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_kld
        self.lam_kld.set_value(new_lam.astype(theano.config.floatX))
        return

    def set_lam_l2w(self, lam_l2w=1e-3):
        """
        Set the relative strength of l2 regularization on network params.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_l2w
        self.lam_l2w.set_value(new_lam.astype(theano.config.floatX))
        return

    def _construct_data_nll_cost(self):
        """
        Construct the negative log-likelihood part of cost to minimize.
        """
        if self.use_encoder:
            # compare the encoded input to the inferencer with the encoded
            # output of the generator
            log_prob_cost = self.GN.compute_log_prob(self.IN.Xd_encoded)
        else:
            log_prob_cost = self.GN.compute_log_prob(self.IN.Xd)
        nll_cost = -T.sum(log_prob_cost) / T.cast(self.Xd.shape[0], 'floatX')
        return nll_cost

    def _construct_post_kld_cost(self, kc2_scale=0.0):
        """
        Construct the posterior KL-d from prior part of cost to minimize.
        """
        obs_count = T.cast(self.Xd.shape[0], 'floatX')
        # basic variational term on KL divergence between post and prior
        kld_cost_1 = self.IN.kld_cost
        # extra term for the squre of KLd in excess of the mean
        kld_mean = self.IN.kld_mean[0]
        kld_too_big = theano.gradient.consider_constant( \
            (self.IN.kld_cost > kld_mean))
        kld_cost_2 = kc2_scale * (kld_too_big * (self.IN.kld_cost - kld_mean))**2.0
        # combine the two types of KLd costs
        kld_cost = T.sum(kld_cost_1 + kld_cost_2) / obs_count
        return kld_cost

    def _construct_other_reg_cost(self):
        """
        Construct the cost for low-level basic regularization. E.g. for
        applying l2 regularization to the network activations and parameters.
        """
        act_reg_cost = (self.IN.act_reg_cost + self.GN.act_reg_cost)
        gp_cost = sum([T.sum(par**2.0) for par in self.gn_params])
        ip_cost = sum([T.sum(par**2.0) for par in self.in_params])
        param_reg_cost = self.lam_l2w[0] * (gp_cost + ip_cost)
        other_reg_cost = (act_reg_cost / T.cast(self.Xd.shape[0], 'floatX')) + \
                param_reg_cost
        return other_reg_cost

    def _construct_train_joint(self):
        """
        Construct theano function to train inferencer and generator jointly.
        """
        outputs = [self.joint_cost, self.data_nll_cost, self.post_kld_cost, \
                self.other_reg_cost, self.posterior_norms, self.posterior_klds]
        func = theano.function(inputs=[ self.Xd, self.Xc, self.Xm ], \
                outputs=outputs, \
                updates=self.joint_updates)
        return func

    def _construct_compute_costs(self):
        """
        Construct theano function to compute the assorted costs without
        applying any updates (e.g. to use with a validation set).
        """
        outputs = [self.joint_cost, self.data_nll_cost, self.post_kld_cost, \
                self.other_reg_cost]
        func = theano.function(inputs=[ self.Xd, self.Xc, self.Xm ], \
                outputs=outputs)
        return func

    def shared_param_clone(self, rng=None, Xd=None, Xc=None, Xm=None):
        """
        Create a "shared-parameter" clone of this GIPair.

        This can be used for chaining VAEs for BPTT.
        """
        clone_gip = GIPair(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, \
            g_net=self.GN, i_net=self.IN, \
            data_dim=self.data_dim, prior_dim=self.prior_dim, \
            params=self.params, shared_param_dicts=self.shared_param_dicts)
        return clone_gip

    def sample_from_chain(self, X_d, X_c=None, X_m=None, loop_iters=5, \
            sigma_scale=None):
        """
        Sample for several rounds through the I<->G loop, initialized with the
        the "data variable" samples in X_d.
        """
        data_samples = []
        prior_samples = []
        if X_c is None:
            X_c = 0.0 * X_d
        if X_m is None:
            X_m = 0.0 * X_d
        if sigma_scale is None:
            sigma_scale = self.GN.prior_sigma
        # set sigma_scale on our InfNet
        old_scale = self.IN.sigma_scale.get_value(borrow=False)
        self.IN.set_sigma_scale(sigma_scale)
        for i in range(loop_iters):
            # apply mask, mixing foreground and background data
            X_d = ((1.0 - X_m) * X_d) + (X_m * X_c)
            # record the data samples for this iteration
            data_samples.append(1.0 * X_d)
            # sample from their inferred posteriors
            X_p = self.IN.sample_posterior(X_d, X_c, X_m)
            # record the sampled points (in the "prior space")
            prior_samples.append(1.0 * X_p)
            # get next data samples by transforming the prior-space points
            X_d = self.GN.transform_prior(X_p)
        # reset sigma_scale on our InfNet
        self.IN.set_sigma_scale(old_scale[0])
        result = {"data samples": data_samples, "prior samples": prior_samples}
        return result

    def sample_from_prior(self, samp_count, sigma=None):
        """
        Draw independent samples from the model's prior, using the gaussian
        continuous prior of the underlying GenNet.
        """
        if sigma is None:
            sigma = self.GN.prior_sigma
        # sample from the GenNet, with either the GenNet's prior sigma or some
        # user-defined sigma
        Xs = self.GN.scaled_sampler(samp_count, sigma)
        return Xs

def binarize_data(X):
    """
    Make a sample of bernoulli variables with probabilities given by X.
    """
    X_shape = X.shape
    probs = npr.rand(*X_shape)
    X_binary = 1.0 * (probs < X)
    return X_binary.astype(theano.config.floatX)

if __name__=="__main__":
    from load_data import load_udm, load_udm_ss, load_mnist
    import utils as utils

    # Initialize a source of randomness
    rng = np.random.RandomState(1234)

    # Load some data to train/validate/test with
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm(dataset, zero_mean=False)
    Xtr = datasets[0][0].get_value(borrow=False).astype(theano.config.floatX)
    tr_samples = Xtr.shape[0]

    # Construct a GenNet and an InfNet, then test constructor for GIPair.
    # Do basic testing, to make sure classes aren't completely broken.
    Xp = T.matrix('Xp_base')
    Xd = T.matrix('Xd_base')
    Xc = T.matrix('Xc_base')
    Xm = T.matrix('Xm_base')
    data_dim = Xtr.shape[1]
    prior_dim = 128
    prior_sigma = 2.0
    # Choose some parameters for the generator network
    gn_params = {}
    gn_config = [prior_dim, 800, 800, data_dim]
    gn_params['mlp_config'] = gn_config
    gn_params['activation'] = softplus_actfun
    gn_params['lam_l2a'] = 1e-3
    gn_params['vis_drop'] = 0.0
    gn_params['hid_drop'] = 0.0
    gn_params['bias_noise'] = 0.1
    # choose some parameters for the continuous inferencer
    in_params = {}
    shared_config = [data_dim, (200, 4)]
    top_config = [shared_config[-1], (200, 4), prior_dim]
    in_params['shared_config'] = shared_config
    in_params['mu_config'] = top_config
    in_params['sigma_config'] = top_config
    in_params['activation'] = relu_actfun
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
            data_dim=data_dim, prior_dim=prior_dim, params=None)
    GIP.set_lam_l2w(1e-3)
    # Set initial learning rate and basic SGD hyper parameters
    learn_rate = 0.001
    GIP.set_all_sgd_params(learn_rate=learn_rate, momentum=0.8)

    for i in range(750000):
        if (i < 100000):
            scale = float(i) / 50000.0
            if (i < 50000):
                GIP.set_all_sgd_params(learn_rate=(scale*learn_rate), momentum=0.8)
            GIP.set_lam_kld(lam_kld=scale)
        if ((i+1 % 100000) == 0):
            learn_rate = learn_rate * 0.75
            GIP.set_all_sgd_params(learn_rate=learn_rate, momentum=0.9)
        # get some data to train with
        tr_idx = npr.randint(low=0,high=tr_samples,size=(100,))
        Xd_batch = binarize_data(Xtr.take(tr_idx, axis=0))
        Xc_batch = 0.0 * Xd_batch
        Xm_batch = 0.0 * Xd_batch
        # do a minibatch update of the model, and compute some costs
        outputs = GIP.train_joint(Xd_batch, Xc_batch, Xm_batch)
        joint_cost = 1.0 * outputs[0]
        data_nll_cost = 1.0 * outputs[1]
        post_kld_cost = 1.0 * outputs[2]
        other_reg_cost = 1.0 * outputs[3]
        if ((i % 500) == 0):
            print("batch: {0:d}, joint_cost: {1:.4f}, data_nll_cost: {2:.4f}, post_kld_cost: {3:.4f}, other_reg_cost: {4:.4f}".format( \
                    i, joint_cost, data_nll_cost, post_kld_cost, other_reg_cost))
        if ((i % 2500) == 0):
            file_name = "GIP_SAMPLES_b{0:d}.png".format(i)
            Xd_samps = np.repeat(Xd_batch[0:10,:], 3, axis=0)
            sample_lists = GIP.sample_from_chain(Xd_samps, loop_iters=10)
            Xs = np.vstack(sample_lists["data samples"])
            utils.visualize_samples(Xs, file_name)

    print("TESTING COMPLETE!")




##############
# EYE BUFFER #
##############
