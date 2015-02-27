#########################################################################
# Code for managing and training a two stage variational auto-encoder.  #
#########################################################################

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

class TwoStageModel(object):
    """
    Controller for training a two-stepping, self-correcting VAE.

    Parameters:
        rng: numpy.random.RandomState (for reproducibility)
        Xd: symbolic "data" input to this 2S-VAE
        Xc: symbolic "control" input to this 2S-VAE
        Xm: symbolic "mask" input to this 2S-VAE
        p_xt_given_z: InfNet for xt given z
        p_zt_given_xt: InfNet for zt given xt
        p_x_given_xt_zt: InfNet for x given xt and zt
        q_z_given_x: InfNet for z given x
        q_zt_given_x_xt: InfNet for zt given x and xt
        x_dim: dimension of the "instances" variables
        z_dim: dimension of the "latent prototypes" variables
        xt_dim: dimension of the "prototypes" variables
        zt_dim: dimension of the "variations" variables
        params: REQUIRED PARAMS SHOWN BELOW
                x_type: can be "bernoulli" or "gaussian"
                xt_type: can be "latent" or "observed"
    """
    def __init__(self, rng=None, \
            Xd=None, Xc=None, Xm=None, \
            p_xt_given_z=None, p_zt_given_xt=None, p_x_given_xt_zt=None, \
            q_z_given_x=None, q_zt_given_x_xt=None, \
            x_dim=None, z_dim=None, xt_dim=None, zt_dim=None, \
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
        assert((self.xt_type == 'latent') or (self.xt_type == 'observed'))

        # record the dimensions of various spaces relevant to this model
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.xt_dim = xt_dim
        self.zt_dim = zt_dim

        # set parameters for the isotropic Gaussian prior over z
        self.z_prior_mean = 0.0
        self.z_prior_logvar = 0.0

        # record the symbolic variables that will provide inputs to the
        # computation graph created to describe this TwoStageModel
        self.Xd = Xd
        self.Xc = Xc
        self.Xm = Xm
        self.x = apply_mask(self.Xd, self.Xc, self.Xm)

        # setup switching variable for changing between sampling/training
        zero_ary = np.zeros((1,)).astype(theano.config.floatX)
        self.train_switch = theano.shared(value=zero_ary, name='tsm_train_switch')
        self.set_train_switch(0.0)
        self.step_switch = theano.shared(value=zero_ary, name='tsm_step_switch')
        self.set_step_switch(0.0)
        # setup a weight for controlling the ratio between global and
        # conditional priors over zt. i.e. the generative model's distribution
        # over zt given xt (i.e. p(zt | xt)) will be given by a mixture:
        #   p(zt | xt) = (a * p_zt_given_xt) + ((1-a) * p_zt)
        # where p_zt_given_xt is conditioned on xt and p_zt is more like a
        # typical prior distribution.
        self.zt_mix_weight = theano.shared(value=zero_ary, name='tsm_zt_mix_weight')
        self.set_zt_mix_weight(1.0)
        # make some parameters for biases on the reconstructions. these
        # parameters get updated during training, as they affect the objective
        zero_row = np.zeros((self.x_dim,)).astype(theano.config.floatX)
        # self.output_bias/self.output_logvar modify the output distribution
        self.output_bias = theano.shared(value=zero_row, name='tsm_output_bias')
        self.output_logvar = theano.shared(value=zero_ary, name='tsm_output_logvar')

        #####################################################################
        # Setup the computation graph that provides values in our objective #
        #####################################################################
        # inferencer model for latent prototypes given instances
        self.q_z_given_x = q_z_given_x.shared_param_clone(rng=rng, Xd=self.x)
        self.z = self.q_z_given_x.output
        self.kld2_scale = self.q_z_given_x.kld2_scale
        # generator model for prototypes given latent prototypes
        self.p_xt_given_z = p_xt_given_z.shared_param_clone(rng=rng, Xd=self.z)
        self.xt = self.p_xt_given_z.output_mean # use deterministic output
        # generator model for variations given prototypes
        if self.xt_type == 'latent':
            self.p_zt_given_xt = p_zt_given_xt.shared_param_clone(rng=rng, \
                    Xd=self.xt)
        else:
            if self.x_type == 'bernoulli':
                self.p_zt_given_xt = p_zt_given_xt.shared_param_clone(rng=rng, \
                     Xd=T.nnet.sigmoid(self.xt + self.output_bias))
            else:
                self.p_zt_given_xt = p_zt_given_xt.shared_param_clone(rng=rng, \
                     Xd=T.nnet.sigmoid(self.xt + self.output_bias))
        self.zt_p = self.p_zt_given_xt.output
        # inferencer model for variations given instances and latent prototypes
        if self.xt_type == 'latent':
            # xt is treated as totally "latent"
            self.q_zt_given_x_xt = q_zt_given_x_xt.shared_param_clone(rng=rng, \
                Xd=T.horizontal_stack(self.x, self.xt))
        else:
            if self.x_type == 'bernoulli':
                # we'll model x using independent bernoullis
                xc = T.nnet.sigmoid(self.xt + self.output_bias)
                grad_ll = self.x - xc
                
            else:
                # we'll model x using independent gaussians
                xc = T.nnet.sigmoid(self.xt + self.output_bias)
                grad_ll = self.x - xc
                
            self.q_zt_given_x_xt = q_zt_given_x_xt.shared_param_clone(rng=rng, \
                    Xd=T.horizontal_stack(self.x, grad_ll))
            #self.q_zt_given_x_xt = q_zt_given_x_xt.shared_param_clone(rng=rng, \
            #        Xd=T.horizontal_stack(self.x, xc))
        self.zt_q = self.q_zt_given_x_xt.output
        # make a zt that switches between self.zt_p and self.zt_q
        self.zt = (self.train_switch[0] * self.zt_q) + \
                ((1.0 - self.train_switch[0]) * self.zt_p)
        # generator model for instances given prototypes and variations
        if self.xt_type == 'latent':
            # xt and zt both go into p_x_given_xt_zt, and p_x_given_xt_zt
            # directly produces the output conditional's mean
            self.p_x_given_xt_zt = p_x_given_xt_zt.shared_param_clone(rng=rng, \
                    Xd=T.horizontal_stack(self.xt, self.zt))
            _xg = self.p_x_given_xt_zt.output_mean
            if self.x_type == 'bernoulli':
                self.xg = T.nnet.sigmoid(_xg + self.output_bias)
                self.xg_xt = self.xg
            else:
                self.xg = _xg + self.output_bias
                self.xg_xt = self.xg
        else:
            # only zt goes into p_x_given_xt_zt. then xt is combined with the
            # output of p_x_given_xt_zt to get the output conditional's mean
            self.p_x_given_xt_zt = p_x_given_xt_zt.shared_param_clone(rng=rng, \
                    Xd=self.zt)
            _xg = self.step_switch[0] * self.p_x_given_xt_zt.output_mean
            if self.x_type == 'bernoulli':
                self.xg = T.nnet.sigmoid(_xg + self.xt + self.output_bias)
                self.xg_xt = T.nnet.sigmoid(self.xt + self.output_bias)
            else:
                self.xg = T.nnet.sigmoid(_xg + self.xt + self.output_bias)
                self.xg_xt = T.nnet.sigmoid(self.xt + self.output_bias)

        ######################################################################
        # ALL SYMBOLIC VARS NEEDED FOR THE OBJECTIVE SHOULD NOW BE AVAILABLE #
        ######################################################################

        # shared var learning rate for generator and inferencer
        zero_ary = np.zeros((1,)).astype(theano.config.floatX)
        self.lr_1 = theano.shared(value=zero_ary, name='tsm_lr_1')
        self.lr_2 = theano.shared(value=zero_ary, name='tsm_lr_2')
        # shared var momentum parameters for generator and inferencer
        self.mom_1 = theano.shared(value=zero_ary, name='tsm_mom_1')
        self.mom_2 = theano.shared(value=zero_ary, name='tsm_mom_2')
        self.it_count = theano.shared(value=zero_ary, name='tsm_it_count')
        # init parameters for controlling learning dynamics
        self.set_sgd_params()
        # init shared var for weighting nll of data given posterior sample
        self.lam_nll = theano.shared(value=zero_ary, name='tsm_lam_nll')
        self.set_lam_nll(lam_nll=1.0)
        # init shared var for weighting prior kld against reconstruction
        self.lam_kld_1 = theano.shared(value=zero_ary, name='tsm_lam_kld_1')
        self.lam_kld_2 = theano.shared(value=zero_ary, name='tsm_lam_kld_2')
        self.set_lam_kld(lam_kld_1=1.0, lam_kld_2=1.0)
        # init shared var for controlling l2 regularization on params
        self.lam_l2w = theano.shared(value=zero_ary, name='tsm_lam_l2w')
        self.set_lam_l2w(1e-4)

        # Grab all of the "optimizable" parameters in "group 1"
        self.group_1_params = []
        self.group_1_params.extend(self.q_z_given_x.mlp_params)
        self.group_1_params.extend(self.p_xt_given_z.mlp_params)
        #self.group_1_params.extend(self.p_zt_given_xt.mlp_params)
        # Grab all of the "optimizable" parameters in "group 2"
        self.group_2_params = []
        self.group_2_params.extend(self.q_zt_given_x_xt.mlp_params)
        self.group_2_params.extend(self.p_zt_given_xt.mlp_params)
        self.group_2_params.extend(self.p_x_given_xt_zt.mlp_params)
        # deal with some additional helper parameters (add them to group 1)
        other_params = [self.output_bias, self.output_logvar]
        self.group_1_params.extend(other_params)
        # Make a joint list of parameters group 1/2
        self.joint_params = self.group_1_params + self.group_2_params
        #for p in self.joint_params:
        #    print("param: {0:s}".format(str(p)))

        ###################################
        # CONSTRUCT THE COSTS TO OPTIMIZE #
        ###################################
        self.kld_cost_1, self.kld_cost_2 = \
                self._construct_kld_costs(kld2_scale=self.kld2_scale)
        act_reg_cost, param_reg_cost = self._construct_reg_costs()
        self.nll_cost = self.lam_nll[0] * self._construct_nll_cost()
        self.kld_cost = self.lam_kld_1[0] * self.kld_cost_1 + \
                self.lam_kld_2[0] * self.kld_cost_2
        self.reg_cost = self.lam_l2w[0] * param_reg_cost
        self.joint_cost = self.nll_cost + self.kld_cost + self.reg_cost

        # Get the gradient of the joint cost for all optimizable parameters
        self.joint_grads = OrderedDict()
        for p in self.joint_params:
            self.joint_grads[p] = T.grad(self.joint_cost, p).clip(-0.1, 0.1)

        # Construct the updates for the generator and inferencer networks
        self.group_1_updates = get_adam_updates(params=self.group_1_params, \
                grads=self.joint_grads, alpha=self.lr_1, \
                beta1=self.mom_1, beta2=self.mom_2, it_count=self.it_count, \
                mom2_init=1e-3, smoothing=1e-8)
        self.group_2_updates = get_adam_updates(params=self.group_2_params, \
                grads=self.joint_grads, alpha=self.lr_2, \
                beta1=self.mom_1, beta2=self.mom_2, it_count=self.it_count, \
                mom2_init=1e-3, smoothing=1e-8)
        self.joint_updates = OrderedDict()
        for k in self.group_1_updates:
            self.joint_updates[k] = self.group_1_updates[k]
        for k in self.group_2_updates:
            self.joint_updates[k] = self.group_2_updates[k]

        # Construct a function for jointly training the generator/inferencer
        self.train_joint = self._construct_train_joint()
        self.compute_costs = self._construct_compute_costs()
        #self.compute_ll_bound = self._construct_compute_ll_bound()
        self.sample_from_prior = self._construct_sample_from_prior()
        # make easy access points for some interesting parameters
        self.inf_1_weights = self.q_z_given_x.shared_layers[0].W
        self.gen_1_weights = self.p_xt_given_z.mu_layers[-1].W
        self.inf_2_weights = self.q_zt_given_x_xt.shared_layers[0].W
        self.gen_2_weights = self.p_x_given_xt_zt.mu_layers[-1].W
        return

    def set_sgd_params(self, lr_1=0.01, lr_2=0.01, \
                mom_1=0.9, mom_2=0.999):
        """
        Set learning rate and momentum parameter for all updates.
        """
        zero_ary = np.zeros((1,))
        # set learning rates
        new_lr_1 = zero_ary + lr_1
        self.lr_1.set_value(new_lr_1.astype(theano.config.floatX))
        new_lr_2 = zero_ary + lr_2
        self.lr_2.set_value(new_lr_2.astype(theano.config.floatX))
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

    def set_lam_kld(self, lam_kld_1=1.0, lam_kld_2=1.0):
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

    def set_train_switch(self, switch_val=0.0):
        """
        Set the switch for changing between training and sampling behavior.
        """
        if (switch_val < 0.5):
            switch_val = 0.0
        else:
            switch_val = 1.0
        zero_ary = np.zeros((1,))
        new_val = zero_ary + switch_val
        new_val = new_val.astype(theano.config.floatX)
        self.train_switch.set_value(new_val)
        return

    def set_step_switch(self, switch_val=0.0):
        """
        Set the switch for controlling the contribution of the second stage.
        """
        assert(switch_val >= 0.0)
        zero_ary = np.zeros((1,))
        new_val = zero_ary + switch_val
        new_val = new_val.astype(theano.config.floatX)
        self.step_switch.set_value(new_val)
        return

    def set_zt_mix_weight(self, zt_mix_weight=0.5):
        """
        Set the weight for blending conditional and global priors over zt.
        """
        assert((zt_mix_weight >= 0.0) and (zt_mix_weight <= 1.0))
        zero_ary = np.zeros((1,))
        new_val = zero_ary + zt_mix_weight
        new_val = new_val.astype(theano.config.floatX)
        self.zt_mix_weight.set_value(new_val)
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

    def _construct_compute_ll_bound(self):
        """
        Construct a function for computing the variational likelihood bound.
        """
        # setup some symbolic variables for theano to deal with
        Xd = T.matrix()
        Xc = T.zeros_like(Xd)
        Xm = T.zeros_like(Xd)
        # get symbolic var for posterior KLds
        post_kld = self.IN.kld_cost
        # get symbolic var for log likelihoods
        log_likelihood = self.GN.compute_log_prob(self.IN.Xd)
        # construct a theano function for actually computing stuff
        outputs = [post_kld, log_likelihood]
        out_func = theano.function([Xd], outputs=outputs, \
                givens={ self.Xd: Xd, self.Xc: Xc, self.Xm: Xm })
        # construct a function for computing multi-sample averages
        def multi_sample_bound(X, sample_count=10):
            post_klds = np.zeros((X.shape[0], 1))
            log_likelihoods = np.zeros((X.shape[0], 1))
            max_lls = np.zeros((X.shape[0], 1)) - 1e8
            for i in range(sample_count):
                result = out_func(X)
                post_klds = post_klds + (1.0 * result[0])
                log_likelihoods = log_likelihoods + (1.0 * result[1])
                max_lls = np.maximum(max_lls, (1.0 * result[1]))
            post_klds = post_klds / sample_count
            log_likelihoods = log_likelihoods / sample_count
            ll_bounds = log_likelihoods - post_klds
            return [ll_bounds, post_klds, log_likelihoods, max_lls]
        return multi_sample_bound

    def _construct_nll_cost(self):
        """
        Construct the negative log-likelihood part of cost to minimize.
        """
        obs_count = T.cast(self.Xd.shape[0], 'floatX')
        if self.x_type == 'bernoulli':
            ll_cost = log_prob_bernoulli(self.x, self.xg)
        else:
            ll_cost = log_prob_gaussian2(self.x, self.xg, \
                    log_vars=self.output_logvar[0])
        nll_cost = -T.sum(ll_cost) / obs_count
        return nll_cost

    def _construct_kld_costs(self, kld2_scale=0.0):
        """
        Construct the posterior KL-d from prior part of cost to minimize.
        """
        obs_count = T.cast(self.Xd.shape[0], 'floatX')
        prior_mean = self.z_prior_mean
        prior_logvar = self.z_prior_logvar
        # construct KLd cost for the distributions over z
        kld_z = gaussian_kld(self.q_z_given_x.output_mean, \
                self.q_z_given_x.output_logvar, \
                prior_mean, prior_logvar)
        # construct KLd cost for the distributions over zt. the prior over
        # zt is given by a mixture between a distribution conditioned on xt,
        # which is estimated by self.p_zt_given_xt, and a "global" prior which
        # is not conditioned on anything.
        kld_zt_cond = gaussian_kld(self.q_zt_given_x_xt.output_mean, \
                self.q_zt_given_x_xt.output_logvar, \
                self.p_zt_given_xt.output_mean, \
                self.p_zt_given_xt.output_logvar)
        kld_zt_glob = gaussian_kld(self.q_zt_given_x_xt.output_mean, \
                self.q_zt_given_x_xt.output_logvar, \
                prior_mean, prior_logvar)
        kld_zt = (self.zt_mix_weight[0] * kld_zt_cond) + \
                ((1.0 - self.zt_mix_weight[0]) * kld_zt_glob)
        # compute the batch-wise costs
        kld_cost_1 = T.sum(kld_z) / obs_count
        kld_cost_2 = T.sum(kld_zt) / obs_count
        return [kld_cost_1, kld_cost_2]

    def _construct_reg_costs(self):
        """
        Construct the cost for low-level basic regularization. E.g. for
        applying l2 regularization to the network activations and parameters.
        """
        obs_count = T.cast(self.Xd.shape[0], 'floatX')
        act_reg_cost = (self.p_xt_given_z.act_reg_cost + \
                self.p_zt_given_xt.act_reg_cost + \
                self.p_x_given_xt_zt.act_reg_cost + \
                self.q_z_given_x.act_reg_cost + \
                self.q_zt_given_x_xt.act_reg_cost) / obs_count
        param_reg_cost = sum([T.sum(p**2.0) for p in self.joint_params])
        other_reg_costs = [act_reg_cost, param_reg_cost]
        return other_reg_costs

    def _construct_train_joint(self):
        """
        Construct theano function to train all networks jointly.
        """
        outputs = [self.joint_cost, self.nll_cost, self.kld_cost_1, \
                self.kld_cost_2, self.kld_cost, self.reg_cost]
        func = theano.function(inputs=[ self.Xd, self.Xc, self.Xm ], \
                outputs=outputs, \
                updates=self.joint_updates)
        #theano.printing.pydotprint(func,
        #                           outfile="tsm_train_graph.png",
        #                           var_with_name_simple=True)
        return func

    def _construct_compute_costs(self):
        """
        Construct theano function to compute the assorted costs without
        applying any updates (e.g. to use with a validation set).
        """
        outputs = [self.joint_cost, self.nll_cost, self.kld_cost_1, \
                self.kld_cost_2, self.reg_cost]
        func = theano.function(inputs=[ self.Xd, self.Xc, self.Xm ], \
                outputs=outputs)
        return func

    def _construct_sample_from_prior(self):
        """
        Construct a function for drawing independent samples from the
        distribution generated by this TwoStageModel.
        """
        z_sym = T.matrix()
        x_sym = T.matrix()
        oputs = [self.xg, self.xg_xt]
        sample_func = theano.function(inputs=[z_sym, x_sym], outputs=oputs, \
                givens={ self.z: z_sym, \
                        self.Xd: T.zeros_like(x_sym), \
                        self.Xc: T.zeros_like(x_sym), \
                        self.Xm: T.zeros_like(x_sym) })
        def prior_sampler(samp_count):
            z_samps = npr.randn(samp_count, self.z_dim)
            z_samps = (np.exp(0.5 * self.z_prior_logvar) * z_samps) + \
                    self.z_prior_mean
            z_samps = z_samps.astype(theano.config.floatX)
            x_samps = np.zeros((samp_count, self.x_dim))
            x_samps = x_samps.astype(theano.config.floatX)
            old_switch = self.train_switch.get_value(borrow=False)
            self.set_train_switch(switch_val=0.0)
            model_samps = sample_func(z_samps, x_samps)
            self.set_train_switch(switch_val=old_switch)
            result_dict = {'xg': model_samps[0], 'xt': model_samps[1]}
            return result_dict
        return prior_sampler


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
            X_d = apply_mask(Xd=X_d, Xc=X_c, Xm=X_m)
            # record the data samples for this iteration
            data_samples.append(1.0 * X_d)
            # sample from their inferred posteriors
            X_p = self.IN.sample_posterior(X_d)
            # record the sampled points (in the "prior space")
            prior_samples.append(1.0 * X_p)
            # get next data samples by transforming the prior-space points
            X_d = self.GN.transform_prior(X_p)
        # reset sigma_scale on our InfNet
        self.IN.set_sigma_scale(old_scale[0])
        result = {"data samples": data_samples, "prior samples": prior_samples}
        return result

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
    import utils
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
    batch_size = 500
    batch_reps = 10

    ###############################################
    # Setup some parameters for the TwoStageModel #
    ###############################################
    prior_sigma = 1.0
    x_dim = Xtr.shape[1]
    z_dim = 50
    xt_dim = x_dim
    zt_dim = 100
    x_type = 'gaussian'
    xt_type = 'observed'

    # some InfNet instances to build the TwoStageModel from
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
    p_xt_given_z.init_biases(0.1)
    #################
    # p_zt_given_xt #
    #################
    params = {}
    shared_config = [xt_dim, 1000, 1000]
    top_config = [shared_config[-1], zt_dim]
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
    p_zt_given_xt = InfNet(rng=rng, Xd=Xd, prior_sigma=prior_sigma, \
            params=params, shared_param_dicts=None)
    p_zt_given_xt.init_biases(0.1)
    ###################
    # p_x_given_xt_zt #
    ###################
    params = {}
    shared_config = [zt_dim, 1000, 1000]
    top_config = [shared_config[-1], x_dim]
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
    p_x_given_xt_zt = InfNet(rng=rng, Xd=Xd, prior_sigma=prior_sigma, \
            params=params, shared_param_dicts=None)
    p_x_given_xt_zt.init_biases(0.1)
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
    q_z_given_x.init_biases(0.1)
    ###################
    # q_zt_given_x_xt #
    ###################
    params = {}
    shared_config = [2*x_dim, 1000, 1000]
    top_config = [shared_config[-1], zt_dim]
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
    q_zt_given_x_xt = InfNet(rng=rng, Xd=Xd, prior_sigma=prior_sigma, \
            params=params, shared_param_dicts=None)
    q_zt_given_x_xt.init_biases(0.1)

    ##############################################################
    # Define parameters for the TwoStageModel, and initialize it #
    ##############################################################
    print("Building the TwoStageModel...")
    tsm_params = {}
    tsm_params['x_type'] = x_type
    tsm_params['xt_type'] = xt_type
    TSM = TwoStageModel(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, \
            p_xt_given_z=p_xt_given_z, p_zt_given_xt=p_zt_given_xt, \
            p_x_given_xt_zt=p_x_given_xt_zt, \
            q_z_given_x=q_z_given_x, q_zt_given_x_xt=q_zt_given_x_xt, \
            x_dim=x_dim, z_dim=z_dim, xt_dim=xt_dim, zt_dim=zt_dim, \
            params=tsm_params)
    obs_mean = np.mean(Xtr, axis=0)
    TSM.set_output_bias(obs_mean)
    TSM.set_input_bias(-obs_mean)

    ################################################################
    # Apply some updates, to check that they aren't totally broken #
    ################################################################
    costs = [0. for i in range(10)]
    learn_rate = 0.002
    for i in range(500000):
        if (((i + 1) % 60000) == 0):
            learn_rate = learn_rate * 0.75
        # randomly sample a minibatch
        tr_idx = npr.randint(low=0,high=tr_samples,size=(batch_size,))
        Xb = Xtr.take(tr_idx, axis=0) #binarize_data(Xtr.take(tr_idx, axis=0))
        Xb = np.repeat(Xb, batch_reps, axis=0).astype(theano.config.floatX)
        if (i < 30000):
            scale = min(1.0, (float(i+1)/10000.0))
            # train the initial coarse approximation model
            TSM.set_sgd_params(lr_1=scale*learn_rate, lr_2=0.0, \
                    mom_1=0.9, mom_2=0.99)
            TSM.set_step_switch(0.0) # scale the corrector's contribution
            TSM.set_lam_nll(lam_nll=1.0)
            TSM.set_lam_kld(lam_kld_1=25.0, lam_kld_2=0.0)
            TSM.set_lam_l2w(1e-4)
            TSM.set_zt_mix_weight(1.0)
        elif (i < 60000):
            scale = min(1.0, (float(i+1-30000)/10000.0))
            # train the secondary corrector model
            TSM.set_sgd_params(lr_1=0.0, lr_2=scale*learn_rate, \
                    mom_1=0.9, mom_2=0.99)
            TSM.set_step_switch(1.0) # scale the corrector's contribution
            TSM.set_lam_nll(lam_nll=1.0)
            TSM.set_lam_kld(lam_kld_1=25.0, lam_kld_2=scale)
            TSM.set_lam_l2w(1e-4)
            TSM.set_zt_mix_weight(1.0)
        else:
            # train the coarse approximation and corrector model jointly
            TSM.set_sgd_params(lr_1=learn_rate, lr_2=learn_rate, \
                    mom_1=0.9, mom_2=0.99)
            TSM.set_step_switch(1.0) # scale the corrector's contribution
            TSM.set_lam_nll(lam_nll=1.0)
            TSM.set_lam_kld(lam_kld_1=25.0, lam_kld_2=1.0)
            TSM.set_lam_l2w(1e-4)
            TSM.set_zt_mix_weight(1.0)
        # perform a minibatch update and record the cost for this batch
        result = TSM.train_joint(Xb, 0.0*Xb, 0.0*Xb)
        costs = [(costs[j] + result[j]) for j in range(len(result))]
        if ((i % 500) == 0):
            costs = [(v / 500.0) for v in costs]
            print("-- batch {0:d} --".format(i))
            print("    joint_cost: {0:.4f}".format(costs[0]))
            print("    nll_cost  : {0:.4f}".format(costs[1]))
            print("    kld_cost_1: {0:.4f}".format(costs[2]))
            print("    kld_cost_2: {0:.4f}".format(costs[3]))
            print("    kld_cost  : {0:.4f}".format(costs[4]))
            print("    reg_cost  : {0:.4f}".format(costs[5]))
            costs = [0.0 for v in costs]
        if ((i % 1000) == 0):
            model_samps = TSM.sample_from_prior(500)
            file_name = "TSM_SAMPLES_b{0:d}_XT.png".format(i)
            utils.visualize_samples(model_samps['xt'], file_name, num_rows=20)
            file_name = "TSM_SAMPLES_b{0:d}_XG.png".format(i)
            utils.visualize_samples(model_samps['xg'], file_name, num_rows=20)
            file_name = "TSM_INF_1_WEIGHTS_b{0:d}.png".format(i)
            utils.visualize_samples(TSM.inf_1_weights.get_value(borrow=False).T, \
                    file_name, num_rows=20)
            #file_name = "TSM_INF_2_WEIGHTS_b{0:d}.png".format(i)
            #utils.visualize_samples(TSM.inf_2_weights.get_value(borrow=False).T, \
            #        file_name, num_rows=20)
            file_name = "TSM_GEN_1_WEIGHTS_b{0:d}.png".format(i)
            utils.visualize_samples(TSM.gen_1_weights.get_value(borrow=False), \
                    file_name, num_rows=20)
            file_name = "TSM_GEN_2_WEIGHTS_b{0:d}.png".format(i)
            utils.visualize_samples(TSM.gen_2_weights.get_value(borrow=False), \
                    file_name, num_rows=20)
    ########
    # DONE #
    ########
    print("TESTING COMPLETE!")




##############
# EYE BUFFER #
##############
