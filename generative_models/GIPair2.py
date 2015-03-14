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
                      apply_mask
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

class GIPair2(object):
    """
    Controller for training a double-stacked variational autoencoder.

    The top and bottom generators must both be instances of the GenNet class
    implemented in "GenNet.py". The top and bottom inferencers must both be
    instances of the InfNet class implemented in "InfNet.py".

    Parameters:
        rng: numpy.random.RandomState (for reproducibility)
        Xd: symbolic "data" input to this VAE
        Xc: symbolic "control" input to this VAE
        Xm: symbolic "mask" input to this VAE
        g_net: The GenNet instance that will serve as the base generator
        i_net: The InfNet instance that will serve as the base inferer
        data_dim: dimension of the "observable data" variables
        prior_dim: dimension of the "latent prior" variables
        g_net_2: The GenNet for the top GIPair
        i_net_2: The InfNet for the top GIPair
        prior_dim_2: dimension of the top "latent prior" variables
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
            g_net_2=None, i_net_2=None, \
            prior_dim_2=None, \
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
                Xd=apply_mask(Xd=self.Xd, Xc=self.Xc, Xm=self.Xm))
        self.posterior_means = self.IN.output_mean
        self.posterior_sigmas = self.IN.output_sigma
        self.posterior_norms = T.sqrt(T.sum(self.posterior_means**2.0, axis=1, keepdims=1))
        self.posterior_klds = self.IN.kld_cost
        self.kld2_scale = self.IN.kld2_scale
        # capture a handle for samples from the variational posterior
        self.Xp = self.IN.output
        # create a "shared-parameter" clone of the generator, set up to
        # receive input from samples from the variational posterior
        self.GN = g_net.shared_param_clone(rng=rng, Xp=self.IN.output)
        # capture a handle for sampled reconstructions from the generator
        self.Xg = self.GN.output

        # construct a second GIPair stacked on top of the first GIPair, which
        # learns to model the posterior samples emitted by the inferencer in
        # the first GIPair
        self.IN2 = i_net_2.shared_param_clone(rng=rng, Xd=apply_mask(Xd=self.Xp, \
                Xc=T.zeros_like(self.Xp), Xm=T.zeros_like(self.Xp)))
        # capture a handle for samples from the top's variational posterior
        self.Xp2 = self.IN2.output
        # feed these variational posterior samples into the top's generator
        self.GN2 = g_net_2.shared_param_clone(rng=rng, Xp=self.Xp2)
        # capture a handle for sampled (latent) reconstructions from GN2
        self.Xg2 = self.GN2.output

        # record and validate the data dimensionality parameters
        self.data_dim = data_dim
        self.prior_dim = prior_dim
        self.prior_dim_2 = prior_dim_2
        # output of the generator and input to the inferencer should both be
        # equal to self.data_dim
        assert(self.data_dim == self.GN.mlp_layers[-1].out_dim)
        assert(self.data_dim == self.IN.shared_layers[0].in_dim)
        # input of the generator and mu/sigma outputs of the inferencer should
        # both be equal to self.prior_dim
        assert(self.prior_dim == self.GN.mlp_layers[0].in_dim)
        assert(self.prior_dim == self.IN.mu_layers[-1].out_dim)
        assert(self.prior_dim == self.IN.sigma_layers[-1].out_dim)
        # input of the generator and mu/sigma outputs of the inferencer should
        # both be equal to self.prior_dim
        assert(self.prior_dim_2 == self.GN2.mlp_layers[0].in_dim)
        assert(self.prior_dim_2 == self.IN2.mu_layers[-1].out_dim)
        assert(self.prior_dim_2 == self.IN2.sigma_layers[-1].out_dim)

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
            self.it_count_bot = theano.shared(value=zero_ary, name='gip_it_count_bot')
            self.it_count_top = theano.shared(value=zero_ary, name='gip_it_count_top')
            self.it_count_joint = theano.shared(value=zero_ary, name='gip_it_count_joint')
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
            self.shared_param_dicts['gip_it_count_bot'] = self.it_count_bot
            self.shared_param_dicts['gip_it_count_top'] = self.it_count_top
            self.shared_param_dicts['gip_it_count_joint'] = self.it_count_joint
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
            self.it_count_bot = self.shared_param_dicts['gip_it_count_bot']
            self.it_count_top = self.shared_param_dicts['gip_it_count_top']
            self.it_count_joint = self.shared_param_dicts['gip_it_count_joint']
            self.lam_nll = self.shared_param_dicts['gip_lam_nll']
            self.lam_kld = self.shared_param_dicts['gip_lam_kld']
            self.lam_l2w = self.shared_param_dicts['gip_lam_l2w']

        # grab the optimizable parameters in the bottom GIPair
        self.in_params = [p for p in self.IN.mlp_params]
        self.gn_params = [p for p in self.GN.mlp_params]
        self.bot_params = self.in_params + self.gn_params
        # grab the optimizable parameters in the top GIPair
        self.in2_params = [p for p in self.IN2.mlp_params]
        self.gn2_params = [p for p in self.GN2.mlp_params]
        self.top_params = self.in2_params + self.gn2_params
        # get the optimizable parameters of bottom + top GIPair
        self.joint_params = self.top_params + self.bot_params

        ###################################
        # CONSTRUCT THE COSTS TO OPTIMIZE #
        ###################################
        self.data_nll_cost_bot = self.lam_nll[0] * \
                self._construct_data_nll_cost(which_gip='bot')
        self.data_nll_cost_top = self.lam_nll[0] * \
                self._construct_data_nll_cost(which_gip='top')
        self.post_kld_cost_bot = self.lam_kld[0] * \
                self._construct_post_kld_cost(which_gip='bot', kld2_scale=self.kld2_scale)
        self.post_kld_cost_top = self.lam_kld[0] * \
                self._construct_post_kld_cost(which_gip='top', kld2_scale=self.kld2_scale)
        self.other_reg_cost_bot = \
                self._construct_other_reg_cost(which_gip='bot')
        self.other_reg_cost_top = \
                self._construct_other_reg_cost(which_gip='top')
        # summed costs for bottom, top, and joint objectives
        self.bot_cost = self.data_nll_cost_bot + self.post_kld_cost_bot + \
                self.other_reg_cost_bot
        self.top_cost = self.data_nll_cost_top + self.post_kld_cost_top + \
                self.other_reg_cost_top
        self.joint_cost = self.bot_cost + self.top_cost

        #########################################
        # CONSTRUCT THE GRADIENTS FOR THE COSTS #
        #########################################
        self.bot_grads = OrderedDict()
        for p in self.bot_params:
            self.bot_grads[p] = T.grad(self.bot_cost, p).clip(-0.1, 0.1)
        # Get the gradient of the top cost for all relevant parameters
        self.top_grads = OrderedDict()
        for p in self.top_params:
            self.top_grads[p] = T.grad(self.top_cost, p).clip(-0.1, 0.1)
        # Get the gradient of the joint cost for all optimizable parameters
        self.joint_grads = OrderedDict()
        for p in self.joint_params:
            self.joint_grads[p] = T.grad(self.joint_cost, p).clip(-0.1, 0.1)

        #######################################
        # CONSTRUCT THE UPDATES FOR THE COSTS #
        #######################################
        # construct updates for the bottom GIPair, for the bottom cost
        self.gn_updates_bot = get_adam_updates(params=self.gn_params, \
                grads=self.bot_grads, alpha=self.lr_gn, \
                beta1=self.mom_1, beta2=self.mom_2, \
                it_count=self.it_count_bot, \
                mom2_init=1e-3, smoothing=1e-8)
        self.in_updates_bot = get_adam_updates(params=self.in_params, \
                grads=self.bot_grads, alpha=self.lr_in, \
                beta1=self.mom_1, beta2=self.mom_2, \
                it_count=self.it_count_bot, \
                mom2_init=1e-3, smoothing=1e-8)
        # construct updates for the top GIPair, for the top cost
        self.gn2_updates_top = get_adam_updates(params=self.gn2_params, \
                grads=self.top_grads, alpha=self.lr_gn, \
                beta1=self.mom_1, beta2=self.mom_2, \
                it_count=self.it_count_top, \
                mom2_init=1e-3, smoothing=1e-8)
        self.in2_updates_top = get_adam_updates(params=self.in2_params, \
                grads=self.top_grads, alpha=self.lr_in, \
                beta1=self.mom_1, beta2=self.mom_2, \
                it_count=self.it_count_top, \
                mom2_init=1e-3, smoothing=1e-8)
        # construct updates for the bottom GIPair, for the joint cost
        self.gn_updates_joint = get_adam_updates(params=self.gn_params, \
                grads=self.joint_grads, alpha=self.lr_gn, \
                beta1=self.mom_1, beta2=self.mom_2, \
                it_count=self.it_count_joint, \
                mom2_init=1e-3, smoothing=1e-8)
        self.in_updates_joint = get_adam_updates(params=self.in_params, \
                grads=self.joint_grads, alpha=self.lr_in, \
                beta1=self.mom_1, beta2=self.mom_2, \
                it_count=self.it_count_joint, \
                mom2_init=1e-3, smoothing=1e-8)
        # construct updates for the top GIPair, for the joint cost
        self.gn2_updates_joint = get_adam_updates(params=self.gn2_params, \
                grads=self.joint_grads, alpha=self.lr_gn, \
                beta1=self.mom_1, beta2=self.mom_2, \
                it_count=self.it_count_joint, \
                mom2_init=1e-3, smoothing=1e-8)
        self.in2_updates_joint = get_adam_updates(params=self.in2_params, \
                grads=self.joint_grads, alpha=self.lr_in, \
                beta1=self.mom_1, beta2=self.mom_2, \
                it_count=self.it_count_joint, \
                mom2_init=1e-3, smoothing=1e-8)


        # Merge the bottom updates for easier application
        self.bot_updates = OrderedDict()
        for k in self.gn_updates_bot:
            self.bot_updates[k] = self.gn_updates_bot[k]
        for k in self.in_updates_bot:
            self.bot_updates[k] = self.in_updates_bot[k]
        self.bot_updates[self.IN.kld_mean] = self.IN.kld_mean_update
        # Merge the top updates for easier application
        self.top_updates = OrderedDict()
        for k in self.gn2_updates_top:
            self.top_updates[k] = self.gn2_updates_top[k]
        for k in self.in2_updates_top:
            self.top_updates[k] = self.in2_updates_top[k]
        self.top_updates[self.IN2.kld_mean] = self.IN2.kld_mean_update
        # Merge the joint updates for easier application
        self.joint_updates = OrderedDict()
        for k in self.gn_updates_joint:
            self.joint_updates[k] = self.gn_updates_joint[k]
        for k in self.in_updates_joint:
            self.joint_updates[k] = self.in_updates_joint[k]
        for k in self.gn2_updates_joint:
            self.joint_updates[k] = self.gn2_updates_joint[k]
        for k in self.in2_updates_joint:
            self.joint_updates[k] = self.in2_updates_joint[k]
        self.joint_updates[self.IN.kld_mean] = self.IN.kld_mean_update
        self.joint_updates[self.IN2.kld_mean] = self.IN2.kld_mean_update
        # Construct a function for jointly training the generator/inferencer
        self.train_bot = self._construct_train_bot()
        self.train_top = self._construct_train_top()
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

    def _construct_data_nll_cost(self, which_gip=None):
        """
        Construct the negative log-likelihood part of cost to minimize.
        """
        assert((which_gip == 'bot') or (which_gip == 'top'))
        if which_gip == 'bot':
            obs_count = T.cast(self.GN.Xp.shape[0], 'floatX')
            # get log-probability reconstruction cost in bottom GIPair
            if self.use_encoder:
                log_prob_cost = self.GN.compute_log_prob(self.IN.Xd_encoded)
            else:
                log_prob_cost = self.GN.compute_log_prob(self.IN.Xd)
        else:
            obs_count = T.cast(self.GN2.Xp.shape[0], 'floatX')
            # get log-probability reconstruction cost in top GIPair
            log_prob_cost = self.GN2.compute_log_prob(self.IN2.Xd)
        nll_cost = -T.sum(log_prob_cost) / obs_count
        return nll_cost

    def _construct_post_kld_cost(self, kld2_scale=0.0, which_gip=None):
        """
        Construct the posterior KL-d from prior part of cost to minimize.
        """
        assert((which_gip == 'bot') or (which_gip == 'top'))
        if which_gip == 'bot':
            obs_count = T.cast(self.IN.Xd.shape[0], 'floatX')
            # basic variational term on KL divergence between post and prior
            kld_cost_1 = self.IN.kld_cost
            # extra term for the squre of KLd in excess of the mean
            kld_too_big = theano.gradient.consider_constant( \
                (self.IN.kld_cost > self.IN.kld_mean[0]))
            kld_cost_2 = kld2_scale * (kld_too_big * self.IN.kld_cost)**2.0
            # combine the two types of KLd costs
            kld_cost = T.sum(kld_cost_1 + kld_cost_2) / obs_count
        else:
            obs_count = T.cast(self.IN2.Xd.shape[0], 'floatX')
            # compute cost as above, but for the top inferencer
            kld_cost_1 = self.IN2.kld_cost
            kld_too_big = theano.gradient.consider_constant( \
                (self.IN2.kld_cost > self.IN2.kld_mean[0]))
            kld_cost_2 = kld2_scale * (kld_too_big * self.IN2.kld_cost)**2.0
            kld_cost = T.sum(kld_cost_1 + kld_cost_2) / obs_count
        return kld_cost

    def _construct_other_reg_cost(self, which_gip=None):
        """
        Construct the cost for low-level basic regularization. E.g. for
        applying l2 regularization to the network activations and parameters.
        """
        assert((which_gip == 'bot') or (which_gip == 'top'))
        
        if which_gip == 'bot':
            obs_count = T.cast(self.IN.Xd.shape[0], 'floatX')
            # construct regularization cost for bottom gip
            act_reg_cost = (self.IN.act_reg_cost + self.GN.act_reg_cost)
            gp_cost = sum([T.sum(par**2.0) for par in self.gn_params])
            ip_cost = sum([T.sum(par**2.0) for par in self.in_params])
            param_reg_cost = self.lam_l2w[0] * (gp_cost + ip_cost)
            other_reg_cost = (act_reg_cost / obs_count) + param_reg_cost
        else:
            obs_count = T.cast(self.IN2.Xd.shape[0], 'floatX')
            # construct regularization cost for top gip
            act_reg_cost = (self.IN2.act_reg_cost + self.GN2.act_reg_cost)
            gp_cost = sum([T.sum(par**2.0) for par in self.gn2_params])
            ip_cost = sum([T.sum(par**2.0) for par in self.in2_params])
            param_reg_cost = self.lam_l2w[0] * (gp_cost + ip_cost)
            other_reg_cost = (act_reg_cost / obs_count) + param_reg_cost
        return other_reg_cost

    def _construct_train_joint(self):
        """
        Construct theano function to train bottom/top GIPairs jointly.
        """
        outputs = [self.joint_cost, self.data_nll_cost_bot, self.post_kld_cost_bot, \
                self.other_reg_cost_bot] #, self.posterior_norms]
        func = theano.function(inputs=[ self.Xd, self.Xc, self.Xm ], \
                outputs=outputs, \
                updates=self.joint_updates)
        return func

    def _construct_train_bot(self):
        """
        Construct theano function to train bottom inferencer and generator.
        """
        outputs = [self.bot_cost, self.data_nll_cost_bot, self.post_kld_cost_bot, \
                self.other_reg_cost_bot] #, self.posterior_norms]
        func = theano.function(inputs=[ self.Xd, self.Xc, self.Xm ], \
                outputs=outputs, \
                updates=self.bot_updates)
        return func

    def _construct_train_top(self):
        """
        Construct theano function to train top inferencer and generator.
        """
        Xd = T.matrix()
        Xc = T.matrix()
        Xm = T.matrix()
        outputs = [self.top_cost, self.data_nll_cost_top, self.post_kld_cost_top, \
                self.other_reg_cost_top] #, self.posterior_norms]
        #func = theano.function(inputs=[ self.Xd, self.Xc, self.Xm ], \
        #        outputs=outputs, \
        #        updates=self.top_updates)
        func = theano.function(inputs=[ Xd ], \
                outputs=outputs, \
                updates=self.top_updates, \
                givens={ self.IN2.Xd: Xd })
        return func

    def _construct_compute_costs(self):
        """
        Construct theano function to compute the assorted costs without
        applying any updates (e.g. to use with a validation set).
        """
        data_nll_cost = self.data_nll_cost_bot + self.data_nll_cost_top
        post_kld_cost = self.post_kld_cost_bot + self.post_kld_cost_top
        other_reg_cost = self.other_reg_cost_bot + self.other_reg_cost_top
        outputs = [self.joint_cost, self.data_nll_cost_bot, self.post_kld_cost_bot, \
                other_reg_cost]
        func = theano.function(inputs=[ self.Xd, self.Xc, self.Xm ], \
                outputs=outputs)
        return func

    def shared_param_clone(self, rng=None, Xd=None, Xc=None, Xm=None):
        """
        Create a "shared-parameter" clone of this GIPair2.

        This can be used for chaining VAEs for BPTT.
        """
        clone_gip = GIPair2(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, \
            g_net=self.GN, i_net=self.IN, \
            data_dim=self.data_dim, prior_dim=self.prior_dim, \
            g_net_2=self.GN2, i_net_2=self.IN2, \
            prior_dim_2=self.prior_dim_2, \
            params=self.params, shared_param_dicts=self.shared_param_dicts)
        return clone_gip

    def sample_from_chain(self, X_d, X_c=None, X_m=None, loop_iters=5, \
            sigma_scale=1.0, which_gip=None):
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
        if which_gip is None:
            which_gip = 'top'
        else:
            assert((which_gip == 'bot') or (which_gip == 'top'))
        if which_gip == 'bot':
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
        else:
            # set sigma_scale on our InfNet
            old_scale = self.IN.sigma_scale.get_value(borrow=False)
            self.IN.set_sigma_scale(sigma_scale)
            old_scale_2 = self.IN2.sigma_scale.get_value(borrow=False)
            self.IN2.set_sigma_scale(sigma_scale)
            for i in range(loop_iters):
                # apply mask, mixing foreground and background data
                X_d = ((1.0 - X_m) * X_d) + (X_m * X_c)
                # record the data samples for this iteration
                data_samples.append(1.0 * X_d)
                # sample from the bottom's inferred posteriors
                X_p = self.IN.sample_posterior(X_d, X_c, X_m)
                # sample from the top's inferred posteriors
                X_p_2 = self.IN2.sample_posterior(X_p, 0.0*X_p, 0.0*X_p)
                # record the sampled points (in the "prior space")
                prior_samples.append(1.0 * X_p_2)
                # pass down from top's latent space to bottom's latent space
                X_g_2 = self.GN2.transform_prior(X_p_2)
                # pass down from bottom's latent space to the data space
                X_d = self.GN.transform_prior(X_g_2)
            # reset sigma_scale on our InfNets
            self.IN.set_sigma_scale(old_scale[0])
            self.IN2.set_sigma_scale(old_scale_2[0])
        result = {"data samples": data_samples, "prior samples": prior_samples}
        return result

    def sample_from_prior(self, samp_count, sigma=None, which_gip=None):
        """
        Sample from the generative model using independent samples from either
        the top GIPair's latent space or the bottom GIPair's latent space.
        """
        if which_gip is None:
            which_gip = 'top'
        else:
            assert((which_gip == 'bot') or (which_gip == 'top'))
        if sigma is None:
            # use the default sigma for the appropriate generator if
            # none was given by the user
            if which_gip == 'bot':
                sigma = self.GN.prior_sigma
            else:
                sigma = self.GN2.prior_sigma
        if which_gip == 'bot':
            # sample directly from the bottom generator's prior
            Xd_samples = self.GN.scaled_sampler(samp_count, sigma)
        else:
            # sample directly from the top generator's prior
            Xp_samples = self.GN2.scaled_sampler(samp_count, sigma)
            # transform through the bottom generator
            Xd_samples = self.GN.transform_prior(Xp_samples)
        return Xd_samples



if __name__=="__main__":
    # TEST CODE IS ELSEWHERE
    print("NO TEST CODE HERE!")




##############
# EYE BUFFER #
##############
