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
                      safe_log, apply_mask
from GenNet import GenNet
from InfNet import InfNet
from PeaNet import PeaNet
from DKCode import get_adam_updates, get_adadelta_updates
from LogPDFs import log_prob_bernoulli, log_prob_gaussian2, gaussian_kld


#
#
# Important symbolic variables:
#   Xd: Xd represents input at the "data variables" of the inferencer
#   Xc: Xc represents input at the "control variables" of the inferencer
#   Xm: Xm represents input at the "mask variables" of the inferencer
#
#

def batch_normalize(Y):
    """
    Set columns of Y to zero mean and unit variance.
    """
    Y_zmuv = (Y - T.mean(Y, axis=0, keepdims=True)) / \
            T.std(Y, axis=0, keepdims=True)
    return Y_zmuv

class OneStageModel(object):
    """
    Controller for training a basic one step VAE.

    Parameters:
        rng: numpy.random.RandomState (for reproducibility)
        Xd: symbolic "data" input to this 2S-VAE
        Xc: symbolic "control" input to this 2S-VAE
        Xm: symbolic "mask" input to this 2S-VAE
        p_xt_given_z: InfNet for xt given z
        q_z_given_x: InfNet for z given x
        x_dim: dimension of the "instances" variables
        z_dim: dimension of the "latent prototypes" variables
        xt_dim: dimension of the "prototypes" variables
        params: REQUIRED PARAMS SHOWN BELOW
                x_type: can be "bernoulli" or "gaussian"
                xt_type: must be "observed"
    """
    def __init__(self, rng=None, \
            Xd=None, Xc=None, Xm=None, \
            p_xt_given_z=None, q_z_given_x=None, \
            x_dim=None, z_dim=None, xt_dim=None, \
            params=None):
        # setup a rng for this GIPair
        self.rng = RandStream(rng.randint(100000))

        # grab the user-provided parameters
        if params is None:
            self.params = {}
        else:
            self.params = params
        self.x_type = self.params['x_type']
        self.xt_type = self.params['xt_type']
        #
        # x_type: this tells if we're using bernoulli or gaussian model for
        #         the observations
        # xt_type: this tells how we incorporate the protoypes in the model
        #
        assert((self.x_type == 'bernoulli') or (self.x_type == 'gaussian'))
        assert(self.xt_type == 'observed')

        # record the dimensions of various spaces relevant to this model
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.xt_dim = xt_dim
        assert((self.xt_dim == self.x_dim))

        # set parameters for the isotropic Gaussian prior over z
        self.z_prior_mean = 0.0
        self.z_prior_logvar = 0.0

        # record the symbolic variables that will provide inputs to the
        # computation graph created to describe this OneStageModel
        self.Xd = Xd
        self.Xc = Xc
        self.Xm = Xm
        self.batch_reps = T.lscalar()
        self.x = apply_mask(self.Xd, self.Xc, self.Xm)

        # self.output_bias/self.output_logvar modify the output distribution
        zero_ary = np.zeros((1,)).astype(theano.config.floatX)
        zero_row = np.zeros((self.x_dim,)).astype(theano.config.floatX)
        self.output_bias = theano.shared(value=zero_row, name='tsm_output_bias')
        self.output_logvar = theano.shared(value=zero_ary, name='tsm_output_logvar')

        #####################################################################
        # Setup the computation graph that provides values in our objective #
        #####################################################################
        # inferencer model for latent prototypes given instances
        self.q_z_given_x = q_z_given_x.shared_param_clone(rng=rng, Xd=self.x)
        self.z = self.q_z_given_x.output
        self.z_mean = self.q_z_given_x.output_mean
        self.z_logvar = self.q_z_given_x.output_logvar
        # generator model for prototypes given latent prototypes
        self.p_xt_given_z = p_xt_given_z.shared_param_clone(rng=rng, Xd=self.z)
        self.xt = self.p_xt_given_z.output_mean # use deterministic output

        if self.x_type == 'bernoulli':
            self.xg = T.nnet.sigmoid(self.xt + self.output_bias)
        else:
            self.xg = T.nnet.sigmoid(self.xt + self.output_bias)

        ######################################################################
        # ALL SYMBOLIC VARS NEEDED FOR THE OBJECTIVE SHOULD NOW BE AVAILABLE #
        ######################################################################

        # shared var learning rate for generator and inferencer
        zero_ary = np.zeros((1,)).astype(theano.config.floatX)
        self.lr_1 = theano.shared(value=zero_ary, name='osm_lr_1')
        # shared var momentum parameters for generator and inferencer
        self.mom_1 = theano.shared(value=zero_ary, name='osm_mom_1')
        self.mom_2 = theano.shared(value=zero_ary, name='osm_mom_2')
        self.it_count = theano.shared(value=zero_ary, name='osm_it_count')
        # init parameters for controlling learning dynamics
        self.set_sgd_params()
        # init shared var for weighting nll of data given posterior sample
        self.lam_nll = theano.shared(value=zero_ary, name='osm_lam_nll')
        self.set_lam_nll(lam_nll=1.0)
        # init shared var for weighting prior kld against reconstruction
        self.lam_kld_1 = theano.shared(value=zero_ary, name='osm_lam_kld_1')
        self.lam_kld_2 = theano.shared(value=zero_ary, name='osm_lam_kld_2')
        self.set_lam_kld(lam_kld_1=1.0, lam_kld_2=0.0)
        # init shared var for controlling l2 regularization on params
        self.lam_l2w = theano.shared(value=zero_ary, name='osm_lam_l2w')
        self.set_lam_l2w(1e-4)
        # init shared var for moment matching cost on z
        self.lam_zmm = theano.shared(value=zero_ary, name='osm_lam_zmm')
        self.set_lam_zmm(1.0)

        # Grab all of the "optimizable" parameters in "group 1"
        self.group_1_params = []
        self.group_1_params.extend(self.q_z_given_x.mlp_params)
        self.group_1_params.extend(self.p_xt_given_z.mlp_params)
        # deal with some additional helper parameters (add them to group 1)
        other_params = [self.output_bias, self.output_logvar]
        # Make a joint list of parameters group 1/2
        self.joint_params = self.group_1_params + other_params

        ###################################
        # CONSTRUCT THE COSTS TO OPTIMIZE #
        ###################################
        self.zmm_cost = self.lam_zmm[0] * self._construct_zmm_cost()
        self.nll_costs = self.lam_nll[0] * self._construct_nll_costs()
        self.nll_cost = T.mean(self.nll_costs)
        self.kld_costs_1, self.kld_costs_2 = self._construct_kld_costs()
        self.kld_costs = (self.lam_kld_1[0] * self.kld_costs_1) + \
                (self.lam_kld_2[0] * self.kld_costs_2)
        self.kld_cost = T.mean(self.kld_costs)
        act_reg_cost, param_reg_cost = self._construct_reg_costs()
        self.reg_cost = self.lam_l2w[0] * param_reg_cost
        self.joint_cost = self.nll_cost + self.kld_cost + self.reg_cost

        # Get the gradient of the joint cost for all optimizable parameters
        self.joint_grads = OrderedDict()
        for p in self.joint_params:
            self.joint_grads[p] = T.grad(self.joint_cost, p)

        # Construct the updates for the generator and inferencer networks
        self.joint_updates = get_adam_updates(params=self.joint_params, \
                grads=self.joint_grads, alpha=self.lr_1, \
                beta1=self.mom_1, beta2=self.mom_2, it_count=self.it_count, \
                mom2_init=1e-3, smoothing=1e-8, max_grad_norm=10.0)

        # Construct a function for jointly training the generator/inferencer
        self.train_joint = self._construct_train_joint()
        self.compute_fe_terms = self._construct_compute_fe_terms()
        self.compute_post_klds = self._construct_compute_post_klds()
        self.sample_from_prior = self._construct_sample_from_prior()
        self.inf_weights = self.q_z_given_x.shared_layers[0].W
        self.gen_weights = self.p_xt_given_z.mu_layers[-1].W
        return

    def set_sgd_params(self, lr_1=0.01, mom_1=0.9, mom_2=0.999):
        """
        Set learning rate and momentum parameter for all updates.
        """
        zero_ary = np.zeros((1,))
        # set learning rates
        new_lr_1 = zero_ary + lr_1
        self.lr_1.set_value(new_lr_1.astype(theano.config.floatX))
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

    def set_lam_kld(self, lam_kld_1=1.0, lam_kld_2=0.0):
        """
        Set the relative weight of prior KL-divergence vs. data likelihood.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_kld_1
        self.lam_kld_1.set_value(new_lam.astype(theano.config.floatX))
        new_lam = zero_ary + lam_kld_2
        self.lam_kld_2.set_value(new_lam.astype(theano.config.floatX))
        return

    def set_lam_l2w(self, lam_l2w=1e-3):
        """
        Set the relative strength of l2 regularization on network params.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_l2w
        self.lam_l2w.set_value(new_lam.astype(theano.config.floatX))
        return

    def set_lam_zmm(self, lam_zmm=1e-3):
        """
        Set the relative weight of moment matching on posteriors over z.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_zmm
        self.lam_zmm.set_value(new_lam.astype(theano.config.floatX))
        return

    def set_output_bias(self, new_bias=None):
        """
        Set the output layer bias.
        """
        new_bias = new_bias.astype(theano.config.floatX)
        self.output_bias.set_value(new_bias)
        return

    def set_input_bias(self, new_bias=None):
        """
        Set the output layer bias.
        """
        new_bias = new_bias.astype(theano.config.floatX)
        self.q_z_given_x.shared_layers[0].b_in.set_value(new_bias)
        return

    def _construct_nll_costs(self):
        """
        Construct the negative log-likelihood part of cost to minimize.
        """
        if self.x_type == 'bernoulli':
            ll_cost = log_prob_bernoulli(self.x, self.xg)
        else:
            ll_cost = log_prob_gaussian2(self.x, self.xg, \
                    log_vars=self.output_logvar[0])
        nll_cost = -ll_cost
        return nll_cost

    def _construct_kld_costs(self):
        """
        Construct the posterior KL-d from prior part of cost to minimize.
        """
        # construct a penalty that is L2-like near 0 and L1-like away from 0.
        huber_pen = lambda x, d: \
                ((1.0 / (2.0 * d)) * ((T.abs_(x) < d) * (x**2.0))) + \
                ((T.abs_(x) >= d) * (T.abs_(x) - (d / 2.0)))
        # do some basic preparation for computation
        prior_mean = self.z_prior_mean
        prior_logvar = self.z_prior_logvar
        # compute the KLds between posteriors and priors. we compute the KLd
        # independently for each input and each latent variable dimension
        kld_z = gaussian_kld(self.q_z_given_x.output_mean, \
                self.q_z_given_x.output_logvar, \
                prior_mean, prior_logvar)
        # compute the batch-wise L1 and L2 penalties on per-dim KLds
        kld_l1_costs = T.sum(huber_pen(kld_z, 0.01), axis=1, keepdims=True)
        kld_l2_costs = T.sum(kld_z**2.0, axis=1, keepdims=True)
        return [kld_l1_costs, kld_l2_costs]

    def _construct_zmm_cost(self):
        """
        Construct moment matching cost for latent posteriors.
        """
        z_mean_mean = T.mean(self.z_mean, axis=0)
        z_mean_std = T.std(self.z_mean, axis=0)
        zmm_cost = T.mean(z_mean_mean**2.0 + z_mean_std**2.0)
        return zmm_cost

    def _construct_reg_costs(self):
        """
        Construct the cost for low-level basic regularization. E.g. for
        applying l2 regularization to the network activations and parameters.
        """
        obs_count = T.cast(self.Xd.shape[0], 'floatX')
        act_reg_cost = (self.p_xt_given_z.act_reg_cost + \
                self.q_z_given_x.act_reg_cost) / obs_count
        param_reg_cost = sum([T.sum(p**2.0) for p in self.joint_params])
        other_reg_costs = [act_reg_cost, param_reg_cost]
        return other_reg_costs

    def _construct_train_joint(self):
        """
        Construct theano function to train all networks jointly.
        """
        # setup some symbolic variables for theano to deal with
        Xd = T.matrix()
        Xc = T.matrix()
        Xm = T.matrix()
        # collect the values to output with each function evaluation
        outputs = [self.joint_cost, self.nll_cost, self.kld_cost, \
                self.reg_cost, self.zmm_cost, self.nll_costs, \
                self.kld_costs]
        func = theano.function(inputs=[ Xd, Xc, Xm, self.batch_reps ], \
                outputs=outputs, \
                givens={ self.Xd: Xd.repeat(self.batch_reps, axis=0), \
                         self.Xc: Xc.repeat(self.batch_reps, axis=0), \
                         self.Xm: Xm.repeat(self.batch_reps, axis=0) }, \
                updates=self.joint_updates)
        return func

    def _construct_compute_post_klds(self):
        """
        Construct theano function to compute the info about the variational
        approximate posteriors for some inputs.
        """
        # setup some symbolic variables for theano to deal with
        Xd = T.matrix()
        Xc = T.zeros_like(Xd)
        Xm = T.zeros_like(Xd)
        prior_mean = self.z_prior_mean
        prior_logvar = self.z_prior_logvar
        all_klds = gaussian_kld(self.q_z_given_x.output_mean, \
                self.q_z_given_x.output_logvar, \
                prior_mean, prior_logvar)
        # compile theano function for a one-sample free-energy estimate
        kld_func = theano.function(inputs=[Xd], outputs=all_klds, \
                givens={self.Xd: Xd, self.Xc: Xc, self.Xm: Xm})
        return kld_func

    def _construct_compute_fe_terms(self):
        """
        Construct theano function to compute the log-likelihood and posterior
        KL-divergence terms for the variational free-energy.
        """
        # setup some symbolic variables for theano to deal with
        Xd = T.matrix()
        Xc = T.zeros_like(Xd)
        Xm = T.zeros_like(Xd)
        # construct values to output
        if self.x_type == 'bernoulli':
            ll_term = log_prob_bernoulli(self.x, self.xg)
        else:
            ll_term = log_prob_gaussian2(self.x, self.xg, \
                    log_vars=self.output_logvar[0])
        prior_mean = self.z_prior_mean
        prior_logvar = self.z_prior_logvar
        all_klds = gaussian_kld(self.q_z_given_x.output_mean, \
                self.q_z_given_x.output_logvar, \
                prior_mean, prior_logvar)
        kld_term = T.sum(all_klds, axis=1)
        # compile theano function for a one-sample free-energy estimate
        fe_term_sample = theano.function(inputs=[Xd], \
                outputs=[ll_term, kld_term], \
                givens={self.Xd: Xd, self.Xc: Xc, self.Xm: Xm})
        # construct a wrapper function for multi-sample free-energy estimate
        def fe_term_estimator(X, sample_count):
            ll_sum = np.zeros((X.shape[0],))
            kld_sum = np.zeros((X.shape[0],))
            for i in range(sample_count):
                result = fe_term_sample(X)
                ll_sum = ll_sum + result[0].ravel()
                kld_sum = kld_sum + result[1].ravel()
            mean_nll = -ll_sum / float(sample_count)
            mean_kld = kld_sum / float(sample_count)
            return [mean_nll, mean_kld]
        return fe_term_estimator

    def _construct_sample_from_prior(self):
        """
        Construct a function for drawing independent samples from the
        distribution generated by this OneStageModel.
        """
        z_sym = T.matrix()
        oputs = [self.xg]
        sample_func = theano.function(inputs=[z_sym], outputs=oputs, \
                givens={ self.z: z_sym })
        def prior_sampler(samp_count):
            z_samps = npr.randn(samp_count, self.z_dim)
            z_samps = (np.exp(0.5 * self.z_prior_logvar) * z_samps) + \
                    self.z_prior_mean
            z_samps = z_samps.astype(theano.config.floatX)
            model_samps = sample_func(z_samps)
            result_dict = {'xg': model_samps[0]}
            return result_dict
        return prior_sampler

def compute_fe_bound(OSM, X, sample_count):
    """
    Compute free-energy bound for X, in minibatches.
    """
    obs_count = X.shape[0]
    batch_size = 2000
    X_nll = np.zeros((obs_count,))
    X_kld = np.zeros((obs_count,))
    batch_start = 0
    while True:
        batch_end = min(obs_count, batch_start+batch_size)
        print("batch index: {0:d} --> {1:d}".format(batch_start, batch_end))
        Xb = X[batch_start:batch_end]
        fe_terms = OSM.compute_fe_terms(Xb, sample_count)
        X_nll[batch_start:batch_end] = fe_terms[0].ravel()
        X_kld[batch_start:batch_end] = fe_terms[1].ravel()
        # advance batch bounds
        batch_start = batch_end
        # quit when its the right time
        if batch_start == obs_count:
            break
    X_fe = X_nll + X_kld
    return [X_fe, X_nll, X_kld]

def collect_obs_costs(batch_costs, batch_reps):
    """
    Collect per-observation costs from a cost vector containing the cost for
    multiple repetitions of each observation.
    """
    obs_count = int(batch_costs.shape[0] / batch_reps)
    obs_costs = np.zeros((obs_count,))
    obs_idx = -1
    for i in range(batch_costs.shape[0]):
        if ((i % batch_reps) == 0):
            obs_idx = obs_idx + 1
        obs_costs[obs_idx] = obs_costs[obs_idx] + batch_costs[i]
    obs_costs = obs_costs / batch_reps
    return obs_costs

if __name__=="__main__":
    from load_data import load_udm, load_udm_ss, load_mnist
    from NetLayers import binarize_data, row_shuffle
    import utils
    from LogPDFs import cross_validate_sigma
    ##########################
    # Get some training data #
    ##########################
    rng = np.random.RandomState(1234)
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm(dataset, zero_mean=False)
    Xtr_shared = datasets[0][0]
    Xva_shared = datasets[1][0]
    Xtr = Xtr_shared.get_value(borrow=False).astype(theano.config.floatX)
    Xva = Xva_shared.get_value(borrow=False).astype(theano.config.floatX)
    tr_samples = Xtr.shape[0]
    batch_size = 200
    batch_reps = 10
    carry_frac = 0.2
    carry_size = int(batch_size * carry_frac)
    carry_time = 25

    ###############################################
    # Setup some parameters for the OneStageModel #
    ###############################################
    prior_sigma = 1.0
    x_dim = Xtr.shape[1]
    z_dim = 100
    xt_dim = x_dim
    zt_dim = 128
    x_type = 'bernoulli'
    xt_type = 'observed'

    # some InfNet instances to build the OneStageModel from
    Xd = T.matrix('Xd_base')
    Xc = T.matrix('Xc_base')
    Xm = T.matrix('Xm_base')
    ################
    # p_xt_given_z #
    ################
    params = {}
    shared_config = [z_dim, 500, 500]
    top_config = [shared_config[-1], xt_dim]
    params['shared_config'] = shared_config
    params['mu_config'] = top_config
    params['sigma_config'] = top_config
    params['activation'] = relu_actfun
    params['init_scale'] = 1.0
    params['lam_l2a'] = 1e-3
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['kld2_scale'] = 0.0
    p_xt_given_z = InfNet(rng=rng, Xd=Xd, prior_sigma=prior_sigma, \
            params=params, shared_param_dicts=None)
    p_xt_given_z.init_biases(0.2)
    ###############
    # q_z_given_x #
    ###############
    params = {}
    shared_config = [x_dim, 500, 500]
    top_config = [shared_config[-1], z_dim]
    params['shared_config'] = shared_config
    params['mu_config'] = top_config
    params['sigma_config'] = top_config
    params['activation'] = relu_actfun
    params['init_scale'] = 1.0
    params['lam_l2a'] = 1e-3
    params['vis_drop'] = 0.0
    params['hid_drop'] = 0.0
    params['bias_noise'] = 0.0
    params['input_noise'] = 0.0
    params['kld2_scale'] = 0.0
    q_z_given_x = InfNet(rng=rng, Xd=Xd, prior_sigma=prior_sigma, \
            params=params, shared_param_dicts=None)
    q_z_given_x.init_biases(0.2)

    ##############################################################
    # Define parameters for the OneStageModel, and initialize it #
    ##############################################################
    print("Building the OneStageModel...")
    osm_params = {}
    osm_params['x_type'] = x_type
    osm_params['xt_type'] = xt_type
    OSM = OneStageModel(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, \
            p_xt_given_z=p_xt_given_z, q_z_given_x=q_z_given_x, \
            x_dim=x_dim, z_dim=z_dim, xt_dim=xt_dim , \
            params=osm_params)
    obs_mean = np.mean(Xtr, axis=0)
    OSM.set_output_bias(obs_mean)
    OSM.set_input_bias(0.0 * obs_mean)

    ################################################################
    # Apply some updates, to check that they aren't totally broken #
    ################################################################
    costs = [0. for i in range(20)]
    obs_costs = np.zeros((batch_size,))
    learn_rate = 0.01
    for i in range(100000):
        scale = min(1.0, (float(i+1) / 5000.0))
        if ((i % carry_time) == 0):
            # sample a fully random batch
            batch_idx = npr.randint(low=0,high=tr_samples,size=(batch_size,))
        else:
            # sample a partially random batch, which retains some portion of
            # the worst scoring examples from the previous batch
            fresh_idx = npr.randint(low=0,high=tr_samples,size=(batch_size-carry_size,))
            batch_idx = np.concatenate((fresh_idx.ravel(), carry_idx.ravel()))
        # set model parameters for this update
        OSM.set_sgd_params(lr_1=scale*learn_rate, mom_1=0.8, mom_2=0.99)
        OSM.set_lam_nll(lam_nll=1.0)
        OSM.set_lam_kld(lam_kld_1=1.0, lam_kld_2=0.0)
        OSM.set_lam_zmm(lam_zmm=0.0)
        OSM.set_lam_l2w(1e-4)

        # perform a minibatch update and record the cost for this batch
        Xb = binarize_data(Xtr.take(batch_idx, axis=0))
        Xb = Xb.astype(theano.config.floatX)
        result = OSM.train_joint(Xb, 0.0*Xb, 0.0*Xb, batch_reps)
        batch_costs = result[5] + result[6]
        obs_costs = collect_obs_costs(batch_costs, batch_reps)
        carry_idx = batch_idx[np.argsort(-obs_costs)[0:carry_size]]
        costs = [(costs[j] + result[j]) for j in range(len(result))]
        if ((i % 500) == 0):
            costs = [(v / 500.0) for v in costs]
            print("-- batch {0:d} --".format(i))
            print("    joint_cost: {0:.4f}".format(costs[0]))
            print("    nll_cost  : {0:.4f}".format(costs[1]))
            print("    kld_cost  : {0:.4f}".format(costs[2]))
            print("    reg_cost  : {0:.4f}".format(costs[3]))
            print("    zmm_cost  : {0:.4f}".format(costs[4]))
            costs = [0.0 for v in costs]
        if ((i % 1000) == 0):
            Xva = row_shuffle(Xva)
            model_samps = OSM.sample_from_prior(500)
            file_name = "OSM_SAMPLES_b{0:d}_XG.png".format(i)
            utils.visualize_samples(model_samps['xg'], file_name, num_rows=20)
            file_name = "OSM_INF_WEIGHTS_b{0:d}.png".format(i)
            utils.visualize_samples(OSM.inf_weights.get_value(borrow=False).T, \
                    file_name, num_rows=20)
            file_name = "OSM_GEN_WEIGHTS_b{0:d}.png".format(i)
            utils.visualize_samples(OSM.gen_weights.get_value(borrow=False), \
                    file_name, num_rows=20)
            # compute information about free-energy on validation set
            file_name = "OSM_FREE_ENERGY_b{0:d}.png".format(i)
            fe_terms = OSM.compute_fe_terms(binarize_data(Xva[0:5000]), 20)
            utils.plot_scatter(fe_terms[1], fe_terms[0], file_name, \
                    x_label='Posterior KLd', y_label='Negative Log-likelihood')
            # compute information about posterior KLds on validation set
            file_name = "OSM_POST_KLDS_b{0:d}.png".format(i)
            post_klds = OSM.compute_post_klds(binarize_data(Xva[0:5000]))
            post_dim_klds = np.mean(post_klds, axis=0)
            utils.plot_stem(np.arange(post_dim_klds.shape[0]), post_dim_klds, \
                    file_name)
        if ((i > 5000) and ((i % 2000) == 0)):
            # test Parzen density estimator built from prior samples
            model_samps = OSM.sample_from_prior(10000)
            cross_validate_sigma(model_samps['xg'], Xva, [0.14, 0.15, 0.16, 0.17], 50)
    ########
    # DONE #
    ########
    print("TESTING COMPLETE!")




##############
# EYE BUFFER #
##############
