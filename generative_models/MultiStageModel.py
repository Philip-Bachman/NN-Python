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
from DKCode import get_adam_updates, get_adadelta_updates
from LogPDFs import log_prob_bernoulli, log_prob_gaussian2, gaussian_kld
from HelperFuncs import to_fX

####################
# IMPLEMENTATATION #
####################

class MultiStageModel(object):
    """
    Controller for training a multi-step iterative refinement model.

    Parameters:
        rng: numpy.random.RandomState (for reproducibility)
        x_in: the input data to encode
        x_out: the target output to decode
        p_s0_given_z: InfNet for initializing "canvas" state
        p_hi_given_si: InfNet for hi given si
        p_sip1_given_si_hi: HydraNet for sip1 given si and hi
        q_z_given_x: InfNet for z given x
        q_hi_given_x_si: InfNet for hi given x and si
        obs_dim: dimension of the observations to generate
        z_dim: dimension of the "initial" latent space
        h_dim: dimension of the "primary" latent space
        ir_steps: number of "iterative refinement" steps to perform
        params: REQUIRED PARAMS SHOWN BELOW
                x_type: can be "bernoulli" or "gaussian"
                obs_transform: can be 'none' or 'sigmoid'
    """
    def __init__(self, rng=None, \
            x_in=None, x_out=None, \
            p_s0_given_z=None, \
            p_hi_given_si=None, \
            p_sip1_given_si_hi=None, \
            q_z_given_x=None, \
            q_hi_given_x_si=None, \
            obs_dim=None, \
            z_dim=None, h_dim=None, \
            ir_steps=4, params=None, \
            shared_param_dicts=None):
        # setup a rng for this GIPair
        self.rng = RandStream(rng.randint(100000))

        # grab the user-provided parameters
        self.params = params
        self.x_type = self.params['x_type']
        assert((self.x_type == 'bernoulli') or (self.x_type == 'gaussian'))
        if 'obs_transform' in self.params:
            assert((self.params['obs_transform'] == 'sigmoid') or \
                    (self.params['obs_transform'] == 'none'))
            if self.params['obs_transform'] == 'sigmoid':
                self.obs_transform = lambda x: T.nnet.sigmoid(20.0 * T.tanh(0.05 * x))
            else:
                self.obs_transform = lambda x: x
        else:
            self.obs_transform = lambda x: T.nnet.sigmoid(20.0 * T.tanh(0.05 * x))
        if self.x_type == 'bernoulli':
            self.obs_transform = lambda x: T.nnet.sigmoid(20.0 * T.tanh(0.05 * x))
        self.shared_param_dicts = shared_param_dicts

        # record the dimensions of various spaces relevant to this model
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.ir_steps = ir_steps

        # grab handles to the relevant InfNets
        self.q_z_given_x = q_z_given_x
        self.q_hi_given_x_si = q_hi_given_x_si
        self.p_s0_given_z = p_s0_given_z
        self.p_hi_given_si = p_hi_given_si
        self.p_sip1_given_si_hi = p_sip1_given_si_hi

        # record the symbolic variables that will provide inputs to the
        # computation graph created to describe this MultiStageModel
        self.x_in = x_in
        self.x_out = x_out
        self.hi_zmuv = T.tensor3() # for ZMUV Gaussian samples to use in scan

        # setup switching variable for changing between sampling/training
        zero_ary = to_fX( np.zeros((1,)) )
        self.train_switch = theano.shared(value=zero_ary, name='msm_train_switch')
        self.set_train_switch(1.0)
        # setup a variable for controlling dropout noise
        self.drop_rate = theano.shared(value=zero_ary, name='msm_drop_rate')
        self.set_drop_rate(0.0)
        # this weight balances l1 vs. l2 penalty on posterior KLds
        self.lam_kld_l1l2 = theano.shared(value=zero_ary, name='msm_lam_kld_l1l2')
        self.set_lam_kld_l1l2(1.0)

        if self.shared_param_dicts is None:
            # initialize "optimizable" parameters specific to this MSM
            init_vec = to_fX( np.zeros((self.z_dim,)) )
            self.p_z_mean = theano.shared(value=init_vec, name='msm_p_z_mean')
            self.p_z_logvar = theano.shared(value=init_vec, name='msm_p_z_logvar')
            init_vec = to_fX( np.zeros((self.obs_dim,)) )
            self.obs_logvar = theano.shared(value=zero_ary, name='msm_obs_logvar')
            self.bounded_logvar = 8.0 * T.tanh((1.0/8.0) * self.obs_logvar)
            self.shared_param_dicts = {}
            self.shared_param_dicts['p_z_mean'] = self.p_z_mean
            self.shared_param_dicts['p_z_logvar'] = self.p_z_logvar
            self.shared_param_dicts['obs_logvar'] = self.obs_logvar
        else:
            self.p_z_mean = self.shared_param_dicts['p_z_mean']
            self.p_z_logvar = self.shared_param_dicts['p_z_logvar']
            self.obs_logvar = self.shared_param_dicts['obs_logvar']
            self.bounded_logvar = 8.0 * T.tanh((1.0/8.0) * self.obs_logvar)

        # setup a function for computing reconstruction log likelihood
        if self.x_type == 'bernoulli':
            self.log_prob_func = lambda xo, xh: \
                    (-1.0 * log_prob_bernoulli(xo, xh))
        else:
            self.log_prob_func = lambda xo, xh: \
                    (-1.0 * log_prob_gaussian2(xo, xh, \
                     log_vars=self.bounded_logvar))

        # get a drop mask that drops things with probability p
        drop_scale = 1. / (1. - self.drop_rate[0])
        drop_rnd = self.rng.uniform(size=self.x_out.shape, \
                low=0.0, high=1.0, dtype=theano.config.floatX)
        drop_mask = drop_scale * (drop_rnd > self.drop_rate[0])

        #############################
        # Setup self.z and self.s0. #
        #############################
        print("Building MSM step 0...")
        drop_x = drop_mask * self.x_in
        self.q_z_mean, self.q_z_logvar, self.z = \
                self.q_z_given_x.apply(drop_x, do_samples=True)
        # get initial observation state
        self.s0, _ = self.p_s0_given_z.apply(self.z, do_samples=False)

        # gather KLd and NLL for the initialization step
        self.init_klds = gaussian_kld(self.q_z_mean, self.q_z_logvar, \
                                      self.p_z_mean, self.p_z_logvar)
        self.init_nlls =  -1.0 * \
                self.log_prob_func(self.x_out, self.obs_transform(self.s0))

        ##################################################
        # Setup the iterative generation loop using scan #
        ##################################################
        def ir_step_func(hi_zmuv, sim1):
            # get variables used throughout this refinement step
            sim1_obs = self.obs_transform(sim1) # transform state -> obs
            grad_ll = self.x_out - sim1_obs

            # get samples of next hi, conditioned on current si
            hi_p_mean, hi_p_logvar = self.p_hi_given_si.apply( \
                    sim1_obs, do_samples=False)
            # now we build the model for variational hi given si
            hi_q_mean, hi_q_logvar = self.q_hi_given_x_si.apply( \
                    T.horizontal_stack(grad_ll, sim1_obs), \
                    do_samples=False)
            hi_q = (T.exp(0.5 * hi_q_logvar) * hi_zmuv) + hi_q_mean
            hi_p = (T.exp(0.5 * hi_p_logvar) * hi_zmuv) + hi_p_mean

            # make hi samples that can be switched between hi_p and hi_q
            hi = ( ((self.train_switch[0] * hi_q) + \
                    ((1.0 - self.train_switch[0]) * hi_p)) )

            # p_sip1_given_si_hi is conditioned on si and  hi.
            ig_vals, fg_vals, in_vals = self.p_sip1_given_si_hi.apply(hi)
                    
            # get the transformed values (for an LSTM style update)
            i_gate = 1.0 * T.nnet.sigmoid(ig_vals + 2.0)
            f_gate = 1.0 * T.nnet.sigmoid(fg_vals + 2.0)
            # perform an LSTM-like update of the state sim1 -> si
            si = (in_vals * i_gate) + (sim1 * f_gate)

            # compute generator NLL for this step
            nlli = self.log_prob_func(self.x_out, self.obs_transform(si))
            # compute relevant KLds for this step
            kldi_q2p = gaussian_kld(hi_q_mean, hi_q_logvar, \
                                    hi_p_mean, hi_p_logvar)
            kldi_p2q = gaussian_kld(hi_p_mean, hi_p_logvar, \
                                    hi_q_mean, hi_q_logvar)
            return si, nlli, kldi_q2p, kldi_p2q

        init_values = [self.s0, None, None, None]

        self.scan_results, self.scan_updates = theano.scan(ir_step_func, \
                outputs_info=init_values, sequences=self.hi_zmuv)

        self.si = self.scan_results[0]
        self.nlli = self.scan_results[1]
        self.kldi_q2p = self.scan_results[2]
        self.kldi_p2q = self.scan_results[3]

        ######################################################################
        # ALL SYMBOLIC VARS NEEDED FOR THE OBJECTIVE SHOULD NOW BE AVAILABLE #
        ######################################################################

        # shared var learning rate for generator and inferencer
        zero_ary = to_fX( np.zeros((1,)) )
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
        self.lam_kld_z = theano.shared(value=zero_ary, name='msm_lam_kld_z')
        self.lam_kld_q2p = theano.shared(value=zero_ary, name='msm_lam_kld_q2p')
        self.lam_kld_p2q = theano.shared(value=zero_ary, name='msm_lam_kld_p2q')
        self.set_lam_kld(lam_kld_z=1.0, lam_kld_q2p=0.7, lam_kld_p2q=0.3)
        # init shared var for controlling l2 regularization on params
        self.lam_l2w = theano.shared(value=zero_ary, name='msm_lam_l2w')
        self.set_lam_l2w(1e-5)

        # Grab all of the "optimizable" parameters in "group 1"
        self.q_params = []
        self.q_params.extend(self.q_z_given_x.mlp_params)
        self.q_params.extend(self.q_hi_given_x_si.mlp_params)
        # Grab all of the "optimizable" parameters in "group 2"
        self.p_params = [self.p_z_mean, self.p_z_logvar]
        self.p_params.extend(self.p_hi_given_si.mlp_params)
        self.p_params.extend(self.p_sip1_given_si_hi.mlp_params)
        self.p_params.extend(self.p_s0_given_z.mlp_params)

        # Make a joint list of parameters group 1/2
        self.joint_params = self.q_params + self.p_params

        #################################
        # CONSTRUCT THE KLD-BASED COSTS #
        #################################
        self.kld_z_q2p, self.kld_z_p2q, self.kld_hi_q2p, self.kld_hi_p2q = \
                self._construct_kld_costs(p=1.0)
        self.kld_z = (self.lam_kld_q2p[0] * self.kld_z_q2p) + \
                     (self.lam_kld_p2q[0] * self.kld_z_p2q)
        self.kld_hi = (self.lam_kld_q2p[0] * self.kld_hi_q2p) + \
                      (self.lam_kld_p2q[0] * self.kld_hi_p2q)
        self.kld_costs = (self.lam_kld_z[0] * self.kld_z) + self.kld_hi
        # now do l2 KLd costs
        self.kl2_z_q2p, self.kl2_z_p2q, self.kl2_hi_q2p, self.kl2_hi_p2q = \
                self._construct_kld_costs(p=2.0)
        self.kl2_z = (self.lam_kld_q2p[0] * self.kl2_z_q2p) + \
                     (self.lam_kld_p2q[0] * self.kl2_z_p2q)
        self.kl2_hi = (self.lam_kld_q2p[0] * self.kl2_hi_q2p) + \
                      (self.lam_kld_p2q[0] * self.kl2_hi_p2q)
        self.kl2_costs = (self.lam_kld_z[0] * self.kl2_z) + self.kl2_hi
        # compute joint l1/l2 KLd cost
        self.kld_l1l2_costs = (self.lam_kld_l1l2[0] * self.kld_costs) + \
                ((1.0 - self.lam_kld_l1l2[0]) * self.kl2_costs)
        # compute "mean" (rather than per-input) costs
        self.kld_cost = T.mean(self.kld_costs)
        self.kl2_cost = T.mean(self.kl2_costs)
        self.kld_l1l2_cost = T.mean(self.kld_l1l2_costs)
        #################################
        # CONSTRUCT THE NLL-BASED COSTS #
        #################################
        self.nll_costs = self.nlli[-1]
        self.nll_cost = self.lam_nll[0] * T.mean(self.nll_costs)
        ########################################
        # CONSTRUCT THE REST OF THE JOINT COST #
        ########################################
        param_reg_cost = self._construct_reg_costs()
        self.reg_cost = self.lam_l2w[0] * param_reg_cost
        self.joint_cost = self.nll_cost + self.kld_l1l2_cost + \
                          self.reg_cost
        ##############################
        # CONSTRUCT A PER-INPUT COST #
        ##############################
        self.obs_costs = self.nll_costs + self.kld_l1l2_costs

        # Get the gradient of the joint cost for all optimizable parameters
        print("Computing gradients of self.joint_cost...")
        self.joint_grads = OrderedDict()
        grad_list = T.grad(self.joint_cost, self.joint_params)
        for i, p in enumerate(self.joint_params):
            self.joint_grads[p] = grad_list[i]

        # Construct the updates for the generator and inferencer networks
        self.q_updates = get_adam_updates(params=self.q_params, \
                grads=self.joint_grads, alpha=self.lr_1, \
                beta1=self.mom_1, beta2=self.mom_2, \
                mom2_init=1e-3, smoothing=1e-5, max_grad_norm=10.0)
        self.p_updates = get_adam_updates(params=self.p_params, \
                grads=self.joint_grads, alpha=self.lr_2, \
                beta1=self.mom_1, beta2=self.mom_2, \
                mom2_init=1e-3, smoothing=1e-5, max_grad_norm=10.0)
        self.joint_updates = OrderedDict()
        for k in self.q_updates:
            self.joint_updates[k] = self.q_updates[k]
        for k in self.p_updates:
            self.joint_updates[k] = self.p_updates[k]
        # add scan updates, which seem to be required
        for k in self.scan_updates:
            self.joint_updates[k] = self.scan_updates[k]

        # Construct a function for jointly training the generator/inferencer
        print("Compiling cost computer...")
        self.compute_raw_klds = self._construct_raw_klds()
        print("Compiling training function...")
        self.train_joint = self._construct_train_joint()
        print("Compiling free-energy sampler...")
        self.compute_fe_terms = self._construct_compute_fe_terms()
        print("Compiling open-loop model sampler...")
        self.sample_from_prior = self._construct_sample_from_prior()
        print("Compiling data-guided model sampler...")
        self.sample_from_input = self._construct_sample_from_input()
        return

    def set_sgd_params(self, lr_1=0.01, lr_2=0.01, \
                mom_1=0.9, mom_2=0.999):
        """
        Set learning rate and momentum parameter for all updates.
        """
        zero_ary = np.zeros((1,))
        # set learning rates
        new_lr_1 = zero_ary + lr_1
        self.lr_1.set_value(to_fX(new_lr_1))
        new_lr_2 = zero_ary + lr_2
        self.lr_2.set_value(to_fX(new_lr_2))
        # set momentums
        new_mom_1 = zero_ary + mom_1
        self.mom_1.set_value(to_fX(new_mom_1))
        new_mom_2 = zero_ary + mom_2
        self.mom_2.set_value(to_fX(new_mom_2))
        return

    def set_lam_nll(self, lam_nll=1.0):
        """
        Set weight for controlling the influence of the data likelihood.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_nll
        self.lam_nll.set_value(to_fX(new_lam))
        return

    def set_lam_kld(self, lam_kld_z=1.0, lam_kld_q2p=1.0, lam_kld_p2q=1.0):
        """
        Set the relative weight of various KL-divergences.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_kld_z
        self.lam_kld_z.set_value(to_fX(new_lam))
        new_lam = zero_ary + lam_kld_q2p
        self.lam_kld_q2p.set_value(to_fX(new_lam))
        new_lam = zero_ary + lam_kld_p2q
        self.lam_kld_p2q.set_value(to_fX(new_lam))
        return

    def set_lam_l2w(self, lam_l2w=1e-3):
        """
        Set the relative strength of l2 regularization on network params.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_l2w
        self.lam_l2w.set_value(to_fX(new_lam))
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
        self.train_switch.set_value(to_fX(new_val))
        return

    def set_lam_kld_l1l2(self, lam_kld_l1l2=1.0):
        """
        Set the weight for shaping penalty on conditional priors over zt.
        """
        zero_ary = np.zeros((1,))
        new_val = zero_ary + lam_kld_l1l2
        self.lam_kld_l1l2.set_value(to_fX(new_val))
        return

    def set_drop_rate(self, drop_rate=0.0):
        """
        Set the weight for shaping penalty on conditional priors over zt.
        """
        zero_ary = np.zeros((1,))
        new_val = zero_ary + drop_rate
        self.drop_rate.set_value(to_fX(new_val))
        return

    def _construct_zmuv_samples(self, xi, br):
        """
        Construct the necessary (symbolic) samples for computing through this
        MultiStageModel for input (sybolic) matrix X.
        """
        z_zmuv = self.rng.normal( \
                size=(xi.shape[0]*br, self.z_dim), \
                avg=0.0, std=1.0, dtype=theano.config.floatX)
        hi_zmuv = self.rng.normal( \
                size=(self.ir_steps, xi.shape[0]*br, self.h_dim), \
                avg=0.0, std=1.0, dtype=theano.config.floatX)
        return z_zmuv, hi_zmuv

    def _construct_nll_costs(self, si, xo):
        """
        Construct the negative log-likelihood part of free energy.
        """
        # average log-likelihood over the refinement sequence
        xh = self.obs_transform(si)
        if self.x_type == 'bernoulli':
            ll_costs = log_prob_bernoulli(xo, xh)
        else:
            ll_costs = log_prob_gaussian2(xo, xh, \
                    log_vars=self.bounded_logvar)
        nll_costs = -ll_costs
        return nll_costs

    def _construct_kld_costs(self, p=1.0):
        """
        Construct the posterior KL-divergence part of cost to minimize.
        """
        kld_hi_q2ps = []
        kld_hi_p2qs = []
        for i in range(self.ir_steps):
            kld_hi_q2p = self.kldi_q2p[i]
            kld_hi_p2q = self.kldi_p2q[i]
            kld_hi_q2ps.append(T.sum(kld_hi_q2p**p, \
                    axis=1, keepdims=True))
            kld_hi_p2qs.append(T.sum(kld_hi_p2q**p, \
                    axis=1, keepdims=True))
        # compute the batch-wise costs
        kld_hi_q2p = sum(kld_hi_q2ps)
        kld_hi_p2q = sum(kld_hi_p2qs)
        # construct KLd cost for the distributions over z
        kld_z_q2ps = gaussian_kld(self.q_z_mean, self.q_z_logvar, \
                                  self.p_z_mean, self.p_z_logvar)
        kld_z_p2qs = gaussian_kld(self.p_z_mean, self.p_z_logvar, \
                                  self.q_z_mean, self.q_z_logvar)
        kld_z_q2p = T.sum(kld_z_q2ps**p, axis=1, keepdims=True)
        kld_z_p2q = T.sum(kld_z_p2qs**p, axis=1, keepdims=True)
        return [kld_z_q2p, kld_z_p2q, kld_hi_q2p, kld_hi_p2q]

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
        br = T.lscalar()
        # collect the outputs to return from this function
        outputs = [self.joint_cost, self.nll_cost, self.kld_cost, \
                   self.reg_cost, self.obs_costs]
        # compile the theano function
        _, hi_zmuv = self._construct_zmuv_samples(xi, br)
        func = theano.function(inputs=[ xi, xo, br ], \
                outputs=outputs, \
                givens={ self.x_in: xi.repeat(br, axis=0), \
                         self.x_out: xo.repeat(br, axis=0), \
                         self.hi_zmuv: hi_zmuv }, \
                updates=self.joint_updates)
        return func

    def _construct_raw_klds(self):
        """
        Construct function for computing KLd per latent dimension.
        """
        # gather step-wise costs into a single list (init costs at the end)
        all_step_costs = [self.init_klds, self.kldi_q2p, self.kldi_p2q]
        # compile theano function for computing all relevant costs
        inputs = [self.x_in, self.x_out, self.hi_zmuv]
        cost_func = theano.function(inputs=inputs, outputs=all_step_costs, \
                                    updates=self.scan_updates)
        def raw_kld_computer(XI, XO):
            hi_zmuv = to_fX( npr.randn(self.ir_steps, XI.shape[0], self.h_dim) )
            _all_costs = cost_func(XI, XO, hi_zmuv)
            _init_klds = _all_costs[0]
            _kld_q2p = np.sum(np.mean(_all_costs[1], axis=1, keepdims=True), axis=0)
            _kld_p2q = np.sum(np.mean(_all_costs[2], axis=1, keepdims=True), axis=0)
            results = [_init_klds, _kld_q2p, _kld_p2q]
            return results
        return raw_kld_computer

    def _construct_compute_fe_terms(self):
        """
        Construct a function for computing terms in variational free energy.
        """
        # setup some symbolic variables for theano to deal with
        xi = T.matrix()
        xo = T.matrix()
        _, hi_zmuv = self._construct_zmuv_samples(xi, 1)
        # construct values to output
        nll = self.nlli[-1]
        kld = self.kld_z.flatten() + self.kld_hi_q2p.flatten()
        # compile theano function for a one-sample free-energy estimate
        fe_term_sample = theano.function(inputs=[ xi, xo ], \
                outputs=[nll, kld], \
                givens={self.x_in: xi, \
                        self.x_out: xo, \
                        self.hi_zmuv: hi_zmuv}, \
                updates=self.scan_updates)
        # construct a wrapper function for multi-sample free-energy estimate
        def fe_term_estimator(XI, XO, sample_count):
            # compute a multi-sample estimate of variational free-energy
            nll_sum = np.zeros((XI.shape[0],))
            kld_sum = np.zeros((XI.shape[0],))
            for i in range(sample_count):
                result = fe_term_sample(XI, XO)
                nll_sum += result[0].ravel()
                kld_sum += result[1].ravel()
            mean_nll = nll_sum / float(sample_count)
            mean_kld = kld_sum / float(sample_count)
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
        irs = self.ir_steps
        oputs = [self.obs_transform(self.s0)]
        oputs.extend([self.obs_transform(self.si[i]) for i in range(irs)])
        _, hi_zmuv = self._construct_zmuv_samples(x_sym, 1)
        sample_func = theano.function(inputs=[z_sym, x_sym], outputs=oputs, \
                givens={ self.z: z_sym, \
                         self.x_in: T.zeros_like(x_sym), \
                         self.x_out: T.zeros_like(x_sym), \
                         self.hi_zmuv: hi_zmuv }, \
                updates=self.scan_updates)
        def prior_sampler(samp_count):
            x_samps = to_fX( np.zeros((samp_count, self.obs_dim)) )
            old_switch = self.train_switch.get_value(borrow=False)
            # set model to generation mode
            self.set_train_switch(switch_val=0.0)
            z_samps = to_fX( npr.randn(samp_count, self.z_dim) )
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
        """
        xi = T.matrix()
        xo = T.matrix()
        irs = self.ir_steps
        oputs = [self.obs_transform(self.s0)]
        oputs.extend([self.obs_transform(self.si[i]) for i in range(irs)])
        _, hi_zmuv = self._construct_zmuv_samples(xi, 1)
        sample_func = theano.function(inputs=[xi, xo], outputs=oputs, \
                givens={ self.x_in: xi, \
                         self.x_out: xo, \
                         self.hi_zmuv: hi_zmuv }, \
                updates=self.scan_updates)
        def conditional_sampler(XI, XO=None, guided_decoding=False):
            XI = to_fX( XI )
            if XO is None:
                XO = XI
            XO = to_fX( XO )
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
