#############################################################################
# Code for managing and training a variational Iterative Refinement Model.  #
#############################################################################

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
from InfNet import InfNet
from PeaNet import PeaNet
from DKCode import get_adam_updates, get_adadelta_updates
from LogPDFs import log_prob_bernoulli, log_prob_gaussian2, gaussian_kld

#
# Important symbolic variables:
#   Xd: Xd represents input at the "data variables" of the inferencer
#

class MultiStageModel(object):
    """
    Controller for training a multi-step iterative refinement model.

    Parameters:
        rng: numpy.random.RandomState (for reproducibility)
        x_in: the input data to encode
        x_out: the target output to decode
        p_hi_given_si: InfNet for hi given si
        p_sip1_given_si_hi: InfNet for sip1 given si and hi
        q_z_given_x: InfNet for z given x
        q_hi_given_x_si: InfNet for hi given x and si
        model_init_rnn: whether to use a model-based initial rnn state
        obs_dim: dimension of the observations to generate
        rnn_dim: dimension of the latent "RNN state"
        z_dim: dimension of the "initial" latent space
        h_dim: dimension of the "primary" latent space
        ir_steps: number of "iterative refinement" steps to perform
        params: REQUIRED PARAMS SHOWN BELOW
                x_type: can be "bernoulli" or "gaussian"
                obs_transform: can be 'none' or 'sigmoid'
    """
    def __init__(self, rng=None, x_in=None, x_out=None, \
            p_hi_given_si=None, \
            p_sip1_given_si_hi=None, \
            q_z_given_x=None, \
            q_hi_given_x_si=None, \
            model_init_rnn=True, 
            obs_dim=None, rnn_dim=None,\
            z_dim=None, h_dim=None, \
            ir_steps=4, params=None, \
            shared_param_dicts=None):
        # setup a rng for this GIPair
        self.rng = RandStream(rng.randint(100000))

        # decide whether to initialize from a model or from a "constant"
        self.model_init_rnn = model_init_rnn

        # grab the user-provided parameters
        self.params = params
        self.x_type = self.params['x_type']
        assert((self.x_type == 'bernoulli') or (self.x_type == 'gaussian'))
        if 'obs_transform' in self.params:
            assert((self.params['obs_transform'] == 'sigmoid') or \
                    (self.params['obs_transform'] == 'none'))
            if self.params['obs_transform'] == 'sigmoid':
                self.obs_transform = lambda x: T.nnet.sigmoid(x)
            else:
                self.obs_transform = lambda x: x
        else:
            self.obs_transform = lambda x: T.nnet.sigmoid(x)
        if self.x_type == 'bernoulli':
            self.obs_transform = lambda x: T.nnet.sigmoid(x)
        self.shared_param_dicts = shared_param_dicts

        # record the dimensions of various spaces relevant to this model
        self.obs_dim = obs_dim
        self.rnn_dim = rnn_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.ir_steps = ir_steps

        # record the symbolic variables that will provide inputs to the
        # computation graph created to describe this MultiStageModel
        self.x_in = x_in
        self.x_out = x_out
        self.batch_reps = T.lscalar()

        # setup switching variable for changing between sampling/training
        zero_ary = np.zeros((1,)).astype(theano.config.floatX)
        self.train_switch = theano.shared(value=zero_ary, name='msm_train_switch')
        self.set_train_switch(1.0)
        # setup a weight for pulling priors over hi given si towards a
        # shared global prior -- e.g. zero mean and unit variance.
        self.kzg_weight = theano.shared(value=zero_ary, name='msm_kzg_weight')
        self.set_kzg_weight(0.1)

        if self.shared_param_dicts is None:
            # initialize weights and biases for the z -> s0_rnn transform
            b_obs = np.zeros((self.obs_dim,)).astype(theano.config.floatX)
            self.b_obs = theano.shared(value=b_obs, name='msm_b_obs')
            self.obs_logvar = theano.shared(value=zero_ary, name='msm_obs_logvar')
            self.bounded_logvar = 8.0 * T.tanh((1.0/8.0) * self.obs_logvar)
            self.shared_param_dicts = {}
            self.shared_param_dicts['b_obs'] = self.b_obs
            self.shared_param_dicts['obs_logvar'] = self.obs_logvar
        else:
            self.b_obs = self.shared_param_dicts['b_obs']
            self.obs_logvar = self.shared_param_dicts['obs_logvar']
            self.bounded_logvar = 8.0 * T.tanh((1.0/8.0) * self.obs_logvar)

        #############################
        # Setup self.z and self.s0. #
        #############################
        print("Building MSM step 0...")
        rnn_scale = 0.0
        if self.model_init_rnn: # initialize rnn state from generative model
            rnn_scale = 1.0
        self.q_z_given_x = q_z_given_x.shared_param_clone(rng=rng, Xd=self.x_in)
        self.z = self.q_z_given_x.output
        self.z_rnn = self.z
        self.s0_obs = T.zeros_like(self.x_in) + self.b_obs
        _s0_rnn_model = self.z_rnn
        _s0_rnn_const = self.q_z_given_x.mu_layers[-1].b
        self.s0_rnn = (rnn_scale * _s0_rnn_model) + \
                ((1.0 - rnn_scale) * _s0_rnn_const)

        ###############################################################
        # Setup the iterative refinement loop, starting from self.s0. #
        ###############################################################
        self.p_hi_given_si = []       # holds p_hi_given_si for each i
        self.p_sip1_given_si_hi = []  # holds p_sip1_given_si_hi for each i
        self.q_hi_given_x_si = []     # holds q_hi_given_x_si for each i
        self.si = [self.s0_obs]       # holds si for each i
        self.hi = []                  # holds hi for each i
        for i in range(self.ir_steps):
            print("Building MSM step {0:d}...".format(i+1))
            si_rnn = self.s0_rnn
            si_obs = self.si[i]
            si_obs_trans = self.obs_transform(si_obs)

            # get samples of next hi, conditioned on current si
            self.p_hi_given_si.append( \
                    p_hi_given_si.shared_param_clone(rng=rng, \
                    Xd=T.horizontal_stack( \
                    si_obs_trans, si_rnn)))
            hi_p = self.p_hi_given_si[i].output
            # now we build the model for variational hi given si
            grad_ll = self.x_out - si_obs_trans
            self.q_hi_given_x_si.append( \
                    q_hi_given_x_si.shared_param_clone(rng=rng, \
                    Xd=T.horizontal_stack( \
                    grad_ll, si_obs_trans, si_rnn)))
            hi_q = self.q_hi_given_x_si[i].output
            # make hi samples that can be switched between hi_p and hi_q
            self.hi.append( ((self.train_switch[0] * hi_q) + \
                    ((1.0 - self.train_switch[0]) * hi_p)) )

            # MOD TAG 1
            # p_sip1_given_si_hi is conditioned on hi and the "rnn" part of si.
            self.p_sip1_given_si_hi.append( \
                    p_sip1_given_si_hi.shared_param_clone(rng=rng, \
                    Xd=self.hi[i]))
                    

            # construct the update from si_obs/si_rnn to sip1_obs/sip1_rnn
            sip1_obs = si_obs + self.p_sip1_given_si_hi[i].output_mean
            # record the updated state of the generative process
            self.si.append(sip1_obs)
        self._check_model_shapes()
        ######################################################################
        # ALL SYMBOLIC VARS NEEDED FOR THE OBJECTIVE SHOULD NOW BE AVAILABLE #
        ######################################################################

        # shared var learning rate for generator and inferencer
        zero_ary = np.zeros((1,)).astype(theano.config.floatX)
        self.lr_1 = theano.shared(value=zero_ary, name='msm_lr_1')
        self.lr_2 = theano.shared(value=zero_ary, name='msm_lr_2')
        # shared var momentum parameters for generator and inferencer
        self.mom_1 = theano.shared(value=zero_ary, name='msm_mom_1')
        self.mom_2 = theano.shared(value=zero_ary, name='msm_mom_2')
        # init parameters for controlling learning dynamics
        self.set_sgd_params()
        # init shared var for weighting nll of data given posterior sample
        self.lam_nll = theano.shared(value=zero_ary, name='msm_lam_nll')
        self.set_lam_nll(lam_nll=1.0)
        # init shared var for weighting prior kld against reconstruction
        self.lam_kld_1 = theano.shared(value=zero_ary, name='msm_lam_kld_1')
        self.lam_kld_2 = theano.shared(value=zero_ary, name='msm_lam_kld_2')
        self.set_lam_kld(lam_kld_1=1.0, lam_kld_2=1.0)
        # init shared var for controlling l2 regularization on params
        self.lam_l2w = theano.shared(value=zero_ary, name='msm_lam_l2w')
        self.set_lam_l2w(1e-5)


        # Grab all of the "optimizable" parameters in "group 1"
        self.group_1_params = []
        self.group_1_params.extend(self.q_z_given_x.mlp_params)
        self.group_1_params.extend(self.q_hi_given_x_si[i].mlp_params)
        # Grab all of the "optimizable" parameters in "group 2"
        self.group_2_params = [self.b_obs]
        for i in range(self.ir_steps):
            self.group_2_params.extend(self.p_hi_given_si[i].mlp_params)
            self.group_2_params.extend(self.p_sip1_given_si_hi[i].mlp_params)
        # Make a joint list of parameters group 1/2
        self.joint_params = self.group_1_params + self.group_2_params

        #################################
        # CONSTRUCT THE KLD-BASED COSTS #
        #################################
        self.kld_z, self.kld_hi_cond, self.kld_hi_glob = \
                self._construct_kld_costs()
        self.kld_cost = (self.lam_kld_1[0] * T.mean(self.kld_z)) + \
                (self.lam_kld_2[0] * (T.mean(self.kld_hi_cond) + \
                (self.kzg_weight[0] * T.mean(self.kld_hi_glob))))
        #################################
        # CONSTRUCT THE NLL-BASED COSTS #
        #################################
        self.nll_costs = self._construct_nll_costs()
        self.nll_cost = self.lam_nll[0] * T.mean(self.nll_costs)
        ########################################
        # CONSTRUCT THE REST OF THE JOINT COST #
        ########################################
        param_reg_cost = self._construct_reg_costs()
        self.reg_cost = self.lam_l2w[0] * param_reg_cost
        self.joint_cost = self.nll_cost + self.kld_cost + self.reg_cost

        # Get the gradient of the joint cost for all optimizable parameters
        print("Computing gradients of self.joint_cost...")
        self.joint_grads = OrderedDict()
        grad_list = T.grad(self.joint_cost, self.joint_params)
        for i, p in enumerate(self.joint_params):
            self.joint_grads[p] = grad_list[i]

        # Construct the updates for the generator and inferencer networks
        self.group_1_updates = get_adam_updates(params=self.group_1_params, \
                grads=self.joint_grads, alpha=self.lr_1, \
                beta1=self.mom_1, beta2=self.mom_2, \
                mom2_init=1e-3, smoothing=1e-8, max_grad_norm=10.0)
        self.group_2_updates = get_adam_updates(params=self.group_2_params, \
                grads=self.joint_grads, alpha=self.lr_2, \
                beta1=self.mom_1, beta2=self.mom_2, \
                mom2_init=1e-3, smoothing=1e-8, max_grad_norm=10.0)
        self.joint_updates = OrderedDict()
        for k in self.group_1_updates:
            self.joint_updates[k] = self.group_1_updates[k]
        for k in self.group_2_updates:
            self.joint_updates[k] = self.group_2_updates[k]

        # Construct a function for jointly training the generator/inferencer
        print("Compiling cost computer...")
        self.compute_raw_costs = self._construct_raw_costs()
        print("Compiling training function...")
        self.train_joint = self._construct_train_joint()
        print("Compiling free-energy sampler...")
        self.compute_fe_terms = self._construct_compute_fe_terms()
        print("Compiling open-loop model sampler...")
        self.sample_from_prior = self._construct_sample_from_prior()
        print("Compiling data-guided model sampler...")
        self.sample_from_input = self._construct_sample_from_input()
        # make easy access points for some interesting parameters
        self.gen_inf_weights = self.p_hi_given_si[0].shared_layers[0].W
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

    def set_kzg_weight(self, kzg_weight=0.2):
        """
        Set the weight for shaping penalty on conditional priors over zt.
        """
        assert(kzg_weight >= 0.0)
        zero_ary = np.zeros((1,))
        new_val = zero_ary + kzg_weight
        new_val = new_val.astype(theano.config.floatX)
        self.kzg_weight.set_value(new_val)
        return

    def _check_model_shapes(self):
        """
        Check that inputs/outputs of the various models will pipe together.
        """
        obs_dim = self.obs_dim
        rnn_dim = self.rnn_dim
        jnt_dim = obs_dim + rnn_dim
        z_dim = self.z_dim
        h_dim = self.h_dim
        # check shape of initialization model
        assert(self.q_z_given_x.mu_layers[-1].out_dim == z_dim)
        assert(self.q_z_given_x.shared_layers[0].in_dim == obs_dim)
        # check shape of the forward conditionals over h_i
        assert(self.p_hi_given_si[0].mu_layers[-1].out_dim == h_dim)
        assert(self.p_hi_given_si[0].shared_layers[0].in_dim == jnt_dim)
        assert(self.q_hi_given_x_si[0].mu_layers[-1].out_dim == h_dim)
        assert(self.q_hi_given_x_si[0].shared_layers[0].in_dim == (obs_dim + jnt_dim))
        # check shape of the forward conditionals over s_{i+1}
        assert(self.p_sip1_given_si_hi[0].mu_layers[-1].out_dim == obs_dim)
        assert(self.p_sip1_given_si_hi[0].shared_layers[0].in_dim == h_dim)
        return

    def _construct_nll_costs(self):
        """
        Construct the negative log-likelihood part of free energy.
        """
        # average log-likelihood over the refinement sequence
        sn = self.si[-1]
        xh = self.obs_transform(sn)
        if self.x_type == 'bernoulli':
            ll_costs = log_prob_bernoulli(self.x_out, xh)
        else:
            ll_costs = log_prob_gaussian2(self.x_out, xh, \
                    log_vars=self.bounded_logvar)
        nll_costs = -ll_costs
        return nll_costs

    def _construct_kld_costs(self):
        """
        Construct the posterior KL-divergence part of cost to minimize.
        """
        # construct KLd cost for the distributions over hi. the prior over
        # hi is given by a distribution conditioned on si, which we estimate
        # using self.p_hi_given_si[i]. the conditionals produced by each
        # self.p_hi_given_si[i] will also be regularized towards a shared
        # prior, e.g. a Gaussian with zero mean and unit variance.
        kld_hi_conds = []
        kld_hi_globs = []
        for i in range(self.ir_steps):
            kld_hi_cond = gaussian_kld( \
                    self.q_hi_given_x_si[i].output_mean, \
                    self.q_hi_given_x_si[i].output_logvar, \
                    self.p_hi_given_si[i].output_mean, \
                    self.p_hi_given_si[i].output_logvar)
            kld_hi_glob = gaussian_kld( \
                    self.p_hi_given_si[i].output_mean, \
                    self.p_hi_given_si[i].output_logvar, \
                    0.0, 0.0)
            kld_hi_conds.append(T.sum(kld_hi_cond, \
                    axis=1, keepdims=True))
            kld_hi_globs.append(T.sum(kld_hi_glob, \
                    axis=1, keepdims=True))
        # compute the batch-wise costs
        kld_hi_cond = sum(kld_hi_conds)
        kld_hi_glob = sum(kld_hi_globs)
        # construct KLd cost for the distributions over z
        kld_z_all = gaussian_kld(self.q_z_given_x.output_mean, \
                self.q_z_given_x.output_logvar, \
                0.0, 0.0)
        kld_z = T.sum(kld_z_all, \
                axis=1, keepdims=True)
        return [kld_z, kld_hi_cond, kld_hi_glob]

    def _construct_reg_costs(self):
        """
        Construct the cost for low-level basic regularization. E.g. for
        applying l2 regularization to the network activations and parameters.
        """
        param_reg_cost = sum([T.sum(p**2.0) for p in self.joint_params])
        return param_reg_cost

    def _construct_train_joint(self):
        """
        Construct theano function to train all networks jointly.
        """
        # setup some symbolic variables for theano to deal with
        xi = T.matrix()
        xo = T.matrix()
        # collect the outputs to return from this function
        outputs = [self.joint_cost, self.nll_cost, self.kld_cost, \
                self.reg_cost]
        # compile the theano function
        func = theano.function(inputs=[ xi, xo, self.batch_reps ], \
                outputs=outputs, \
                givens={ self.x_in: xi.repeat(self.batch_reps, axis=0), \
                         self.x_out: xo.repeat(self.batch_reps, axis=0) }, \
                updates=self.joint_updates)
        return func

    def _construct_raw_costs(self):
        """
        Construct all the raw, i.e. not weighted by any lambdas, costs.
        """
        # get NLL for all steps
        init_nlls = log_prob_bernoulli(self.x_out, self.obs_transform(self.si[0]))
        step_nlls = []
        for i in range(self.ir_steps):
            step_nlls.append(-1.0 * log_prob_bernoulli(self.x_out, self.obs_transform(self.si[i+1])))
        nlli = T.stack(*step_nlls)
        # get KLd for all steps
        init_klds = gaussian_kld(self.q_z_given_x.output_mean, \
                self.q_z_given_x.output_logvar, \
                0.0, 0.0)
        kld_hi_cond_list = []
        kld_hi_glob_list = []
        for i in range(self.ir_steps):
            kld_hi_cond = gaussian_kld( \
                    self.q_hi_given_x_si[i].output_mean, \
                    self.q_hi_given_x_si[i].output_logvar, \
                    self.p_hi_given_si[i].output_mean, \
                    self.p_hi_given_si[i].output_logvar)
            kld_hi_glob = gaussian_kld( \
                    self.p_hi_given_si[i].output_mean, \
                    self.p_hi_given_si[i].output_logvar, \
                    0.0, 0.0)
            kld_hi_cond_list.append(kld_hi_cond)
            kld_hi_glob_list.append(kld_hi_glob)
        kldi_cond = T.stack(*kld_hi_cond_list)
        kldi_glob = T.stack(*kld_hi_glob_list)
        # gather step-wise costs into a single list (init costs at the end)
        all_step_costs = [nlli, kldi_cond, kldi_glob, init_nlls, init_klds]
        # compile theano function for computing all relevant costs
        inputs = [self.x_in, self.x_out]
        cost_func = theano.function(inputs=inputs, outputs=all_step_costs)
        def raw_cost_computer(XI, XO):
            s0_rnn = np.zeros((XI.shape[0],self.z_dim)).astype(theano.config.floatX)
            _all_costs = cost_func(XI, XO)
            # for i, ac in enumerate(_all_costs):
            #     print("_all_costs[{0:d}].shape: {1:s}".format(i, str(ac.shape)))
            _init_nlls = _all_costs[-2]
            _init_klds = _all_costs[-1]
            _kld_cond = np.sum(np.mean(_all_costs[1], axis=1, keepdims=True), axis=0)
            _kld_glob = np.sum(np.mean(_all_costs[2], axis=1, keepdims=True), axis=0)
            _step_klds = np.mean(np.sum(_all_costs[1], axis=2, keepdims=True), axis=1)
            sk = [np.sum(np.mean(_init_klds, axis=0))]
            sk.extend([k for k in _step_klds])
            _step_klds = np.asarray(sk).astype(theano.config.floatX)
            _step_nlls = np.mean(_all_costs[0], axis=1)
            sn = [np.mean(_init_nlls, axis=0)]
            sn.extend([k for k in _step_nlls])
            _step_nlls = np.asarray(sn).astype(theano.config.floatX)
            results = [_init_nlls, _init_klds, _kld_cond, _kld_glob, \
                       _step_nlls, _step_klds]
            return results
        return raw_cost_computer

    def _construct_compute_fe_terms(self):
        """
        Construct a function for computing terms in variational free energy.
        """
        # setup some symbolic variables for theano to deal with
        xi = T.matrix()
        xo = T.matrix()
        # construct values to output
        nll = self._construct_nll_costs()
        kld = self.kld_z + self.kld_hi_cond
        # compile theano function for a one-sample free-energy estimate
        fe_term_sample = theano.function(inputs=[ xi, xo ], \
                outputs=[nll, kld], \
                givens={self.x_in: xi, \
                        self.x_out: xo})
        # construct a wrapper function for multi-sample free-energy estimate
        def fe_term_estimator(XI, XO, sample_count):
            # set values of some regularization parameters to the values that
            # produce the variational free energy bound.
            old_lam_nll = self.lam_nll.get_value(borrow=False)
            old_lam_kld_1 = self.lam_kld_1.get_value(borrow=False)
            old_lam_kld_2 = self.lam_kld_2.get_value(borrow=False)
            vfe_lam_nll = (0.0 * old_lam_nll) + 1.0
            vfe_lam_kld_1 = (0.0 * old_lam_kld_1) + 1.0
            vfe_lam_kld_2 = (0.0 * old_lam_kld_2) + 1.0
            self.lam_nll.set_value(vfe_lam_nll)
            self.lam_kld_1.set_value(vfe_lam_kld_1)
            self.lam_kld_2.set_value(vfe_lam_kld_2)
            # compute a multi-sample estimate of variational free-energy
            nll_sum = np.zeros((XI.shape[0],))
            kld_sum = np.zeros((XI.shape[0],))
            for i in range(sample_count):
                result = fe_term_sample(XI, XO)
                nll_sum += result[0].ravel()
                kld_sum += result[1].ravel()
            mean_nll = nll_sum / float(sample_count)
            mean_kld = kld_sum / float(sample_count)
            # reset regularization parameters to their previous values
            self.lam_nll.set_value(old_lam_nll)
            self.lam_kld_1.set_value(old_lam_kld_1)
            self.lam_kld_2.set_value(old_lam_kld_2)
            return [mean_nll, mean_kld]
        return fe_term_estimator

    def _construct_sample_from_prior(self):
        """
        Construct a function for drawing independent samples from the
        distribution generated by this MultiStageModel. This function returns
        the full sequence of "partially completed" examples.
        """
        z_sym = T.matrix()
        x_sym = T.matrix()
        oputs = [self.obs_transform(s) for s in self.si]
        sample_func = theano.function(inputs=[z_sym, x_sym], outputs=oputs, \
                givens={self.z: z_sym, \
                        self.x_in: T.zeros_like(x_sym), \
                        self.x_out: T.zeros_like(x_sym)})
        def prior_sampler(samp_count):
            x_samps = np.zeros((samp_count, self.obs_dim)).astype(theano.config.floatX)
            old_switch = self.train_switch.get_value(borrow=False)
            # set model to generation mode
            self.set_train_switch(switch_val=0.0)
            z_samps = npr.randn(samp_count, self.z_dim).astype(theano.config.floatX)
            model_samps = sample_func(z_samps, x_samps)
            # set model back to either training or generation mode
            self.set_train_switch(switch_val=old_switch)
            return model_samps
        return prior_sampler

    def _construct_sample_from_input(self):
        """
        Construct a function for drawing samples from the distribution
        generated by this MultiStageModel, conditioned on some inputs to the
        initial encoder stage (i.e. self.q_z_given_x). This returns the full 
        sequence of "partially completed" examples.

        The 
        """
        xi = T.matrix()
        xo = T.matrix()
        oputs = [self.obs_transform(s) for s in self.si]
        sample_func = theano.function(inputs=[xi, xo], outputs=oputs, \
                givens={self.x_in: xi, \
                        self.x_out: xo})
        def conditional_sampler(XI, XO=None, guided_decoding=False):
            XI = XI.astype(theano.config.floatX)
            if XO is None:
                XO = XI
            XO = XO.astype(theano.config.floatX)
            # set model to desired generation mode                
            old_switch = self.train_switch.get_value(borrow=False)
            if guided_decoding:
                # take samples from guide policies (i.e. variational q)
                self.set_train_switch(switch_val=1.0)
            else:
                # take samples from model's generative policy
                self.set_train_switch(switch_val=0.0)
            # draw guided/unguided conditional samples
            model_samps = sample_func(XI, XO)
            # set model back to either training or generation mode
            self.set_train_switch(switch_val=old_switch)
            return model_samps
        return conditional_sampler

if __name__=="__main__":
    print("Hello world!")







##############
# EYE BUFFER #
##############