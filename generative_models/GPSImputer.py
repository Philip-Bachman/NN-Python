#############################################################################
# Code for managing and training a variational Iterative Refinement Model.  #
#############################################################################

# basic python
import cPickle
import numpy as np
import numpy.random as npr
from collections import OrderedDict
import numexpr as ne

# theano business
import theano
import theano.tensor as T
#from theano.tensor.shared_randomstreams import RandomStreams as RandStream
from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams as RandStream

# phil's sweetness
from DKCode import get_adam_updates, get_adadelta_updates
from LogPDFs import log_prob_bernoulli, log_prob_gaussian2, gaussian_kld
from HelperFuncs import to_fX

#######################################
# IMPLEMENT THE THING THAT DOES STUFF #
#######################################

class GPSImputer(object):
    """
    Controller for training a multi-step imputater via guided policy search.

    Parameters:
        rng: numpy.random.RandomState (for reproducibility)
        x_in: the initial state for imputation
        x_out: the goal state for imputation
        x_mask: mask for state dims to keep fixed during imputation
        p_zi_given_xi: InfNet for stochastic part of step
        p_xip1_given_zi: HydraNet for deterministic part of step
        q_zi_given_x_xi: InfNet for the guide policy
        params: REQUIRED PARAMS SHOWN BELOW
                obs_dim: dimension of inputs to reconstruct
                z_dim: dimension of latent space for policy wobble
                imp_steps: number of reconstruction steps to perform
                step_type: either "add" or "jump"
                x_type: can be "bernoulli" or "gaussian"
                obs_transform: can be 'none' or 'sigmoid'
                use_osm_mode: switch for testing imputation using a
                              pre-trained VAE
    """
    def __init__(self, rng=None, 
            x_in=None, x_mask=None, x_out=None, \
            p_zi_given_xi=None, \
            p_xip1_given_zi=None, \
            q_zi_given_x_xi=None, \
            params=None, \
            shared_param_dicts=None):
        # setup a rng for this GIPair
        self.rng = RandStream(rng.randint(100000))

        # grab the user-provided parameters
        self.params = params
        self.obs_dim = self.params['obs_dim']
        self.z_dim = self.params['z_dim']
        self.imp_steps = self.params['imp_steps']
        self.step_type = self.params['step_type']
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
        if 'use_osm_mode' in self.params:
            self.use_osm_mode = self.params['use_osm_mode']
        else:
            self.use_osm_mode = False
            self.params['use_osm_mode'] = False
        self.shared_param_dicts = shared_param_dicts
        
        assert((self.step_type == 'add') or (self.step_type == 'jump'))
        if self.use_osm_mode:
            self.step_type = 'jump'

        # grab handles to the relevant InfNets
        self.p_zi_given_xi = p_zi_given_xi
        self.p_xip1_given_zi = p_xip1_given_zi
        self.q_zi_given_x_xi = q_zi_given_x_xi

        # record the symbolic variables that will provide inputs to the
        # computation graph created to describe this MultiStageModel
        self.x_in = x_in
        self.x_out = x_out
        self.x_mask = x_mask
        self.zi_zmuv = T.tensor3()
        

        # setup switching variable for changing between sampling/training
        zero_ary = to_fX( np.zeros((1,)) )
        self.train_switch = theano.shared(value=zero_ary, name='msm_train_switch')
        self.set_train_switch(1.0)

        if self.shared_param_dicts is None:
            # initialize parameters "owned" by this model
            s0_init = to_fX( np.zeros((self.obs_dim,)) )
            self.s0 = theano.shared(value=s0_init, name='msm_s0')
            self.obs_logvar = theano.shared(value=zero_ary, name='msm_obs_logvar')
            self.bounded_logvar = 8.0 * T.tanh((1.0/8.0) * self.obs_logvar[0])
            self.shared_param_dicts = {}
            self.shared_param_dicts['s0'] = self.s0
            self.shared_param_dicts['obs_logvar'] = self.obs_logvar
        else:
            # grab the parameters required by this model from a given dict
            self.s0 = self.shared_param_dicts['s0']
            self.obs_logvar = self.shared_param_dicts['obs_logvar']
            self.bounded_logvar = 8.0 * T.tanh((1.0/8.0) * self.obs_logvar[0])

        ###################################################
        # Setup the iterative immputation loop using scan #
        ###################################################
        def imp_step_func(zi_zmuv, si):
            si_as_x = self.obs_transform(si)
            xi_masked = (self.x_mask * self.x_out) + \
                        ((1.0 - self.x_mask) * si_as_x)
            #grad_ll = self.x_out - xi_masked
            # get samples of next zi, according to the global policy
            zi_p_mean, zi_p_logvar = self.p_zi_given_xi.apply( \
                    xi_masked, do_samples=False)
            zi_p = zi_p_mean + (T.exp(0.5 * zi_p_logvar) * zi_zmuv)
            # get samples of next zi, according to the guide policy
            zi_q_mean, zi_q_logvar = self.q_zi_given_x_xi.apply( \
                    T.horizontal_stack(xi_masked, self.x_out), \
                    do_samples=False)
            zi_q = zi_q_mean + (T.exp(0.5 * zi_q_logvar) * zi_zmuv)

            if self.use_osm_mode:
                zi = zi_p
                # compute relevant KLds for this step
                kldi_q2p = gaussian_kld(zi_p_mean, zi_p_logvar, 0.0, 0.0)
                kldi_p2q = gaussian_kld(zi_p_mean, zi_p_logvar, 0.0, 0.0)
            else:
                # make zi samples that can be switched between zi_p and zi_q
                zi = ((self.train_switch[0] * zi_q) + \
                     ((1.0 - self.train_switch[0]) * zi_p))
                # compute relevant KLds for this step
                kldi_q2p = gaussian_kld(zi_q_mean, zi_q_logvar, \
                                        zi_p_mean, zi_p_logvar)
                kldi_p2q = gaussian_kld(zi_p_mean, zi_p_logvar, \
                                        zi_q_mean, zi_q_logvar)

            # compute the next si, given the sampled zi
            hydra_out = self.p_xip1_given_zi.apply(zi)
            si_step = hydra_out[-1]
            if (self.step_type == 'jump'):
                # jump steps always do a full swap (like standard VAE)
                sip1 = si_step
            else:
                # subsequent additive steps just add
                sip1 = si + si_step
            # compute NLL for the current imputation
            nlli = self._construct_nll_costs(sip1, self.x_out, self.x_mask)
            return sip1, nlli, kldi_q2p, kldi_p2q

        # apply scan op for the sequential imputation loop
        self.s0_full = T.zeros_like(self.x_in) + self.s0
        init_vals = [self.s0_full, None, None, None]
        self.scan_results, self.scan_updates = theano.scan(imp_step_func, \
                    outputs_info=init_vals, sequences=self.zi_zmuv)

        self.si = self.scan_results[0]
        self.nlli = self.scan_results[1]
        self.kldi_q2p = self.scan_results[2]
        self.kldi_p2q = self.scan_results[3]

        # get the initial imputation state
        self.x0 = (self.x_mask * self.x_in) + \
                  ((1.0 - self.x_mask) * self.obs_transform(self.s0_full))

        ######################################################################
        # ALL SYMBOLIC VARS NEEDED FOR THE OBJECTIVE SHOULD NOW BE AVAILABLE #
        ######################################################################

        # shared var learning rate for generator and inferencer
        zero_ary = to_fX( np.zeros((1,)) )
        self.lr = theano.shared(value=zero_ary, name='gpsi_lr')
        # shared var momentum parameters for generator and inferencer
        self.mom_1 = theano.shared(value=zero_ary, name='gpsi_mom_1')
        self.mom_2 = theano.shared(value=zero_ary, name='gpsi_mom_2')
        # init parameters for controlling learning dynamics
        self.set_sgd_params()
        # init shared var for weighting nll of data given posterior sample
        self.lam_nll = theano.shared(value=zero_ary, name='gpsi_lam_nll')
        self.set_lam_nll(lam_nll=1.0)
        # init shared var for weighting prior kld against reconstruction
        self.lam_kld_p = theano.shared(value=zero_ary, name='gpsi_lam_kld_p')
        self.lam_kld_q = theano.shared(value=zero_ary, name='gpsi_lam_kld_q')
        self.set_lam_kld(lam_kld_p=0.5, lam_kld_q=0.5)
        # init shared var for controlling l2 regularization on params
        self.lam_l2w = theano.shared(value=zero_ary, name='msm_lam_l2w')
        self.set_lam_l2w(1e-5)

        # Grab all of the "optimizable" parameters in "group 1"
        self.joint_params = [self.s0, self.obs_logvar]
        self.joint_params.extend(self.p_zi_given_xi.mlp_params)
        self.joint_params.extend(self.p_xip1_given_zi.mlp_params)
        self.joint_params.extend(self.q_zi_given_x_xi.mlp_params)

        #################################
        # CONSTRUCT THE KLD-BASED COSTS #
        #################################
        self.kld_p, self.kld_q = self._construct_kld_costs(p=1.0)
        self.kld_costs = (self.lam_kld_p[0] * self.kld_p) + \
                         (self.lam_kld_q[0] * self.kld_q)
        self.kld_cost = T.mean(self.kld_costs)
        #################################
        # CONSTRUCT THE NLL-BASED COSTS #
        #################################
        self.nll_costs = self.nlli[-1]
        self.nll_cost = self.lam_nll[0] * T.mean(self.nll_costs)
        self.nll_bounds = self.nll_costs.ravel() + self.kld_q.ravel()
        self.nll_bound = T.mean(self.nll_bounds)
        ########################################
        # CONSTRUCT THE REST OF THE JOINT COST #
        ########################################
        param_reg_cost = self._construct_reg_costs()
        self.reg_cost = self.lam_l2w[0] * param_reg_cost
        self.joint_cost = self.nll_cost + self.kld_cost + self.reg_cost
        ##############################
        # CONSTRUCT A PER-TRIAL COST #
        ##############################
        self.obs_costs = self.nll_costs + self.kld_costs

        # Get the gradient of the joint cost for all optimizable parameters
        print("Computing gradients of self.joint_cost...")
        self.joint_grads = OrderedDict()
        grad_list = T.grad(self.joint_cost, self.joint_params)
        for i, p in enumerate(self.joint_params):
            self.joint_grads[p] = grad_list[i]

        # Construct the updates for the generator and inferencer networks
        self.joint_updates = get_adam_updates(params=self.joint_params, \
                grads=self.joint_grads, alpha=self.lr, \
                beta1=self.mom_1, beta2=self.mom_2, \
                mom2_init=1e-3, smoothing=1e-5, max_grad_norm=10.0)
        for k, v in self.scan_updates.items():
            self.joint_updates[k] = v

        # Construct a function for jointly training the generator/inferencer
        print("Compiling cost computer...")
        self.compute_raw_costs = self._construct_raw_costs()
        print("Compiling training function...")
        self.train_joint = self._construct_train_joint()
        print("Compiling free-energy sampler...")
        self.compute_fe_terms = self._construct_compute_fe_terms()
        print("Compiling best step cost computer...")
        self.compute_per_step_cost = self._construct_compute_per_step_cost()
        print("Compiling data-guided imputer sampler...")
        self.sample_imputer = self._construct_sample_imputer()
        # make easy access points for some interesting parameters
        self.gen_inf_weights = self.p_zi_given_xi.shared_layers[0].W
        self.gen_gen_weights = self.p_xip1_given_zi.output_layers[-1].W
        return

    def set_sgd_params(self, lr=0.01, mom_1=0.9, mom_2=0.999):
        """
        Set learning rate and momentum parameter for all updates.
        """
        zero_ary = np.zeros((1,))
        # set learning rate
        new_lr = zero_ary + lr
        self.lr.set_value(to_fX(new_lr))
        # set momentums (use first and second order "momentum")
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

    def set_lam_kld(self, lam_kld_p=1.0, lam_kld_q=1.0):
        """
        Set the relative weight of prior KL-divergence vs. data likelihood.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_kld_p
        self.lam_kld_p.set_value(to_fX(new_lam))
        new_lam = zero_ary + lam_kld_q
        self.lam_kld_q.set_value(to_fX(new_lam))
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

    def _construct_zi_zmuv(self, xi, br):
        """
        Construct the necessary (symbolic) samples for computing through this
        GPSImputer for input (sybolic) matrix xi.
        """
        zi_zmuv = self.rng.normal( \
                size=(self.imp_steps, xi.shape[0]*br, self.z_dim), \
                avg=0.0, std=1.0, dtype=theano.config.floatX)
        return zi_zmuv

    def _construct_nll_costs(self, si, xo, xm):
        """
        Construct the negative log-likelihood part of free energy.
        """
        # average log-likelihood over the refinement sequence
        xh = self.obs_transform( si )
        xm_inv = 1.0 - xm # we will measure nll only where xm_inv is 1
        if self.x_type == 'bernoulli':
            ll_costs = log_prob_bernoulli(xo, xh, mask=xm_inv)
        else:
            ll_costs = log_prob_gaussian2(xo, xh, \
                    log_vars=self.bounded_logvar, mask=xm_inv)
        nll_costs = -ll_costs.flatten()
        return nll_costs

    def _construct_kld_costs(self, p=1.0):
        """
        Construct the policy KL-divergence part of cost to minimize.
        """
        kld_pis = []
        kld_qis = []
        for i in range(self.imp_steps):
            kld_pis.append(T.sum(self.kldi_p2q[i]**p, axis=1))
            kld_qis.append(T.sum(self.kldi_q2p[i]**p, axis=1))
        # compute the batch-wise costs
        kld_pi = sum(kld_pis)
        kld_qi = sum(kld_qis)
        return [kld_pi, kld_qi]

    def _construct_reg_costs(self):
        """
        Construct the cost for low-level basic regularization. E.g. for
        applying l2 regularization to the network activations and parameters.
        """
        param_reg_cost = sum([T.sum(p**2.0) for p in self.joint_params])
        return param_reg_cost

    def _construct_compute_fe_terms(self):
        """
        Construct a function for computing terms in variational free energy.
        """
        # setup some symbolic variables for theano to deal with
        xi = T.matrix()
        xo = T.matrix()
        xm = T.matrix()
        zizmuv = self._construct_zi_zmuv(xi, 1)
        # construct values to output
        nll = self.nll_costs.flatten()
        kld = self.kld_q.flatten()
        # compile theano function for a one-sample free-energy estimate
        fe_term_sample = theano.function(inputs=[ xi, xo, xm ], \
                outputs=[nll, kld], \
                givens={self.x_in: xi, \
                        self.x_out: xo, \
                        self.x_mask: xm, \
                        self.zi_zmuv: zizmuv}, \
                updates=self.scan_updates, \
                on_unused_input='ignore')
        # construct a wrapper function for multi-sample free-energy estimate
        def fe_term_estimator(XI, XO, XM, sample_count=20, use_guide_policy=True):
            # set model to desired generation mode
            old_switch = self.train_switch.get_value(borrow=False)
            if use_guide_policy:
                # take samples from guide policies (i.e. variational q)
                self.set_train_switch(switch_val=1.0)
            else:
                # take samples from model's imputation policy
                self.set_train_switch(switch_val=0.0)
            # compute a multi-sample estimate of variational free-energy                
            nll_sum = np.zeros((XI.shape[0],))
            kld_sum = np.zeros((XI.shape[0],))
            for i in range(sample_count):
                result = fe_term_sample(XI, XO, XM)
                nll_sum += result[0].ravel()
                kld_sum += result[1].ravel()
            mean_nll = nll_sum / float(sample_count)
            mean_kld = kld_sum / float(sample_count)
            # set model back to either training or generation mode
            self.set_train_switch(switch_val=old_switch)
            if not use_guide_policy:
                # no KLd if samples are from the primary policy...
                mean_kld = 0.0 * mean_kld
            return [mean_nll, mean_kld]
        return fe_term_estimator

    def _construct_raw_costs(self):
        """
        Construct all the raw, i.e. not weighted by any lambdas, costs.
        """
        # setup some symbolic variables for theano to deal with
        xi = T.matrix()
        xo = T.matrix()
        xm = T.matrix()
        zizmuv = self._construct_zi_zmuv(xi, 1)
        # compile theano function for computing the costs
        all_step_costs = [self.nlli, self.kldi_q2p, self.kldi_p2q]
        cost_func = theano.function(inputs=[xi, xo, xm], \
                    outputs=all_step_costs, \
                    givens={ self.x_in: xi, \
                             self.x_out: xo, \
                             self.x_mask: xm, \
                             self.zi_zmuv: zizmuv }, \
                    updates=self.scan_updates, \
                    on_unused_input='ignore')
        # make a function for computing multi-sample estimates of cost
        def raw_cost_computer(XI, XO, XM):
            _all_costs = cost_func(to_fX(XI), to_fX(XO), to_fX(XM))
            _kld_q2p = np.sum(np.mean(_all_costs[1], axis=1, keepdims=True), axis=0)
            _kld_p2q = np.sum(np.mean(_all_costs[2], axis=1, keepdims=True), axis=0)
            _step_klds = np.mean(np.sum(_all_costs[1], axis=2, keepdims=True), axis=1)
            _step_klds = to_fX( np.asarray([k for k in _step_klds]) )
            _step_nlls = np.mean(_all_costs[0], axis=1)
            _step_nlls = to_fX( np.asarray([k for k in _step_nlls]) )
            results = [_step_nlls, _step_klds, _kld_q2p, _kld_p2q]
            return results
        return raw_cost_computer

    def _construct_compute_per_step_cost(self):
        """
        Construct a theano function for computing the best possible cost
        achieved by sequential imputation.
        """
        # setup some symbolic variables for theano to deal with
        xi = T.matrix()
        xo = T.matrix()
        xm = T.matrix()
        zizmuv = self._construct_zi_zmuv(xi, 1)
        # construct symbolic variables for the step-wise cost
        step_mean_nll = T.mean(self.nlli, axis=1).flatten()
        step_lone_kld = T.sum(self.kldi_q2p, axis=2)
        step_cumu_kld = T.extra_ops.cumsum(step_lone_kld, axis=0)
        step_mean_kld = T.mean(step_cumu_kld, axis=1).flatten()
        # compile theano function for computing the step-wise cost
        step_cost_func = theano.function(inputs=[xi, xo, xm], \
                    outputs=[step_mean_nll, step_mean_kld], \
                    givens={ self.x_in: xi, \
                             self.x_out: xo, \
                             self.x_mask: xm, \
                             self.zi_zmuv: zizmuv }, \
                    updates=self.scan_updates, \
                    on_unused_input='ignore')
        def best_cost_computer(XI, XO, XM, sample_count=20):
            # compute a multi-sample estimate of variational free-energy
            step_nll_sum = np.zeros((self.imp_steps,))
            step_kld_sum = np.zeros((self.imp_steps,))
            for i in range(sample_count):
                result = step_cost_func(XI, XO, XM)
                step_nll_sum += result[0].ravel()
                step_kld_sum += result[1].ravel()
            mean_step_nll = step_nll_sum / float(sample_count)
            mean_step_kld = step_kld_sum / float(sample_count)
            return [mean_step_nll, mean_step_kld]
        return best_cost_computer



    def _construct_train_joint(self):
        """
        Construct theano function to train all networks jointly.
        """
        # setup some symbolic variables for theano to deal with
        xi = T.matrix()
        xo = T.matrix()
        xm = T.matrix()
        br = T.lscalar()
        zizmuv = self._construct_zi_zmuv(xi, br)
        # collect the outputs to return from this function
        outputs = [self.joint_cost, self.nll_bound, self.nll_cost, \
                   self.kld_cost, self.reg_cost, self.obs_costs]
        # compile the theano function
        func = theano.function(inputs=[ xi, xo, xm, br ], \
                outputs=outputs, \
                givens={ self.x_in: xi.repeat(br, axis=0), \
                         self.x_out: xo.repeat(br, axis=0), \
                         self.x_mask: xm.repeat(br, axis=0), \
                         self.zi_zmuv: zizmuv }, \
                updates=self.joint_updates, \
                on_unused_input='ignore')
        return func

    def _construct_sample_imputer(self):
        """
        Construct a function for drawing samples from the distribution
        generated by running this imputer.
        """
        xi = T.matrix()
        xo = T.matrix()
        xm = T.matrix()
        zizmuv = self._construct_zi_zmuv(xi, 1)
        oputs = [self.x0] + [self.obs_transform(self.si[i]) for i in range(self.imp_steps)]
        sample_func = theano.function(inputs=[xi, xo, xm], outputs=oputs, \
                givens={self.x_in: xi, \
                        self.x_out: xo, \
                        self.x_mask: xm, \
                        self.zi_zmuv: zizmuv}, \
                updates=self.scan_updates, \
                on_unused_input='ignore')
        def imputer_sampler(XI, XO, XM, use_guide_policy=False):
            XI = to_fX( XI )
            XO = to_fX( XO )
            XM = to_fX( XM )
            # set model to desired generation mode
            old_switch = self.train_switch.get_value(borrow=False)
            if use_guide_policy:
                # take samples from guide policies (i.e. variational q)
                self.set_train_switch(switch_val=1.0)
            else:
                # take samples from model's imputation policy
                self.set_train_switch(switch_val=0.0)
            # draw guided/unguided conditional samples
            model_samps = sample_func(XI, XO, XM)
            # set model back to either training or generation mode
            self.set_train_switch(switch_val=old_switch)
            # reverse engineer the "masked" samples...
            masked_samps = []
            for xs in model_samps:
                xsm = (XM * XI) + ((1.0 - XM) * xs)
                masked_samps.append(xsm)
            return model_samps, masked_samps
        return imputer_sampler

    def save_to_file(self, f_name=None):
        """
        Dump important stuff to a Python pickle, so that we can reload this
        model later.
        """
        assert(not (f_name is None))
        f_handle = file(f_name, 'wb')
        # dump the dict self.params, which just holds "simple" python values
        cPickle.dump(self.params, f_handle, protocol=-1)
        # make a copy of self.shared_param_dicts, with numpy arrays in place
        # of the theano shared variables
        numpy_param_dicts = {}
        for key in self.shared_param_dicts:
            numpy_ary = self.shared_param_dicts[key].get_value(borrow=False)
            numpy_param_dicts[key] = numpy_ary
        # dump the numpy version of self.shared_param_dicts to pickle file
        cPickle.dump(numpy_param_dicts, f_handle, protocol=-1)
        # get numpy dicts for each of the "child" models that we must save
        child_model_dicts = {}
        child_model_dicts['p_zi_given_xi'] = self.p_zi_given_xi.save_to_dict()
        child_model_dicts['p_xip1_given_zi'] = self.p_xip1_given_zi.save_to_dict()
        child_model_dicts['q_zi_given_x_xi'] = self.q_zi_given_x_xi.save_to_dict()
        # dump the numpy child model dicts to the pickle file
        cPickle.dump(child_model_dicts, f_handle, protocol=-1)
        f_handle.close()
        return

def load_gpsimputer_from_file(f_name=None, rng=None):
    """
    Load a clone of some previously trained model.
    """
    from InfNet import load_infnet_from_dict
    from HydraNet import load_hydranet_from_dict
    assert(not (f_name is None))
    pickle_file = open(f_name)
    # reload the basic python parameters
    self_dot_params = cPickle.load(pickle_file)
    # reload the theano shared parameters
    self_dot_numpy_param_dicts = cPickle.load(pickle_file)
    self_dot_shared_param_dicts = {}
    for key in self_dot_numpy_param_dicts:
        val = to_fX(self_dot_numpy_param_dicts[key])
        self_dot_shared_param_dicts[key] = theano.shared(val)
    # reload the child models
    child_model_dicts = cPickle.load(pickle_file)
    xd = T.matrix()
    p_zi_given_xi = load_infnet_from_dict( \
            child_model_dicts['p_zi_given_xi'], rng=rng, Xd=xd)
    p_xip1_given_zi = load_hydranet_from_dict( \
            child_model_dicts['p_xip1_given_zi'], rng=rng, Xd=xd)
    q_zi_given_x_xi = load_infnet_from_dict( \
            child_model_dicts['q_zi_given_x_xi'], rng=rng, Xd=xd)
    # now, create a new GPSImputer based on the loaded data
    xi = T.matrix()
    xm = T.matrix()
    xo = T.matrix()
    clone_net = GPSImputer(rng=rng, \
                           x_in=xi, x_mask=xm, x_out=xo, \
                           p_zi_given_xi=p_zi_given_xi, \
                           p_xip1_given_zi=p_xip1_given_zi, \
                           q_zi_given_x_xi=q_zi_given_x_xi, \
                           params=self_dot_params, \
                           shared_param_dicts=self_dot_shared_param_dicts)
    # helpful output
    print("==================================================")
    print("LOADED GPSImputer WITH PARAMS:")
    for k in self_dot_params:
        print("    {0:s}: {1:s}".format(str(k), str(self_dot_params[k])))
    print("==================================================")
    return clone_net

class TemplateMatchImputer(object):
    """
    Simple class for performing imputation via template matching.

    I.e. -- we fill in missing values in a partial observation by taking
            the corresponding values from the "training" observation which
            best matches the known values. we'll compute scores for matching
            on either the known values or the unknown values.

    Parameters:
            x_train: the available examples to match against
            x_type: whether to use 'gaussian' or 'bernoulli' log prob
    """
    def __init__(self, x_train=None, x_type=None):
        self.x_train = theano.shared(value=to_fX(x_train), name='x_train')
        self.x_type = x_type
        self.logvar = 0.0
        self.best_match_nll, self.best_match_img = self._construct_funcs()
        return

    def _log_bernoulli(self, p_true, p_approx, mask=None):
        """
        Compute log probability of some binary variables with probabilities
        given by p_true, for probability estimates given by p_approx. We'll
        compute joint log probabilities over row-wise groups.
        """
        if mask is None:
            mask = T.ones((1, p_approx.shape[1]))
        log_prob_1 = p_true * T.log(p_approx+1e-6)
        log_prob_0 = (1.0 - p_true) * T.log((1.0 - p_approx)+1e-6)
        log_prob_01 = log_prob_1 + log_prob_0
        row_log_probs_m_is_1 = T.sum((log_prob_01 * mask), axis=1)
        row_log_probs_m_is_0 = T.sum((log_prob_01 * (1.0-mask)), axis=1)
        return row_log_probs_m_is_1, row_log_probs_m_is_0

    def _log_gaussian(self, mu_true, mu_approx, log_vars=1.0, mask=None):
        """
        Compute log probability of some continuous variables with values given
        by mu_true, w.r.t. gaussian distributions with means given by mu_approx
        and log variances given by les_logvars.
        """
        if mask is None:
            mask = T.ones((1, mu_approx.shape[1]))
        ind_log_probs = C - (0.5 * log_vars)  - \
                ((mu_true - mu_approx)**2.0 / (2.0 * T.exp(log_vars)))
        row_log_probs = T.sum((ind_log_probs * mask), axis=1)
        row_log_probs = T.cast(row_log_probs, 'floatX')
        return row_log_probs

    def _compute_log_prob(self, x_true, x_approx, mask=None):
        """
        helper function for switching between bernoulli/gaussian.
        """
        if self.x_type == 'bernoulli':
            ll = self._log_bernoulli(x_true, x_approx, mask=mask)
        else:
            ll = self._log_gaussian(x_true, x_approx, \
                         log_vars=self.logvar, mask=mask)
        return ll

    def _construct_funcs(self):
        """
        compute log-likelihood of the imputations for values in x_test
        for which m_test is 0. imputation is performed by template matching
        against a fixed set of "training" examples.
        """
        # we'll just brute force the search.
        x_t = T.vector()
        m_t = T.vector()
        ll_m_is_1, ll_m_is_0 = self._compute_log_prob(x_t, self.x_train, \
                                                      mask=m_t)
        outputs = [ll_m_is_1, ll_m_is_0]
        theano_func = theano.function(inputs=[x_t, m_t], outputs=outputs)
        def nll_func(x_test, m_test):
            test_count = x_test.shape[0]
            nll_match_on_known = np.zeros((x_test.shape[0],))
            nll_match_on_unknown = np.zeros((x_test.shape[0],))
            print("Template matching for {} test examples:".format(test_count))
            for i in range(test_count):
                ll_m_is_1, ll_m_is_0 = theano_func(x_test[i], m_test[i])
                match_idx = np.argmax(ll_m_is_1)
                nll_match_on_known[i] = -1.0 * ll_m_is_0[match_idx]
                match_idx = np.argmax(ll_m_is_0)
                nll_match_on_unknown[i] = -1.0 * ll_m_is_0[match_idx]
                if ((i % (test_count/50)) == 0):
                    print("-- processed {} examples, nll_mok: {}, nll_mou: {}".format(i, \
                            np.mean(nll_match_on_known[:(i+1)]), \
                            np.mean(nll_match_on_unknown[:(i+1)])))
            return [nll_match_on_known, nll_match_on_unknown]
        def img_func(x_test, m_test):
            x_tr = self.x_train.get_value(borrow=False)
            test_count = x_test.shape[0]
            img_match_on_known = np.zeros(x_test.shape)
            img_match_on_unknown = np.zeros(x_test.shape)
            print("Template matching for {} test examples:".format(test_count))
            for i in range(test_count):
                xt_i = x_test[i]
                mt_i = m_test[i]
                ll_m_is_1, ll_m_is_0 = theano_func(xt_i, mt_i)
                match_idx = np.argmax(ll_m_is_1)
                img_match_on_known[i] = (mt_i * xt_i) + \
                                        ((1.0 - mt_i) * x_tr[match_idx])
                match_idx = np.argmax(ll_m_is_0)
                img_match_on_unknown[i] = (mt_i * xt_i) + \
                                          ((1.0 - mt_i) * x_tr[match_idx])
            return [img_match_on_known, img_match_on_unknown]
        return nll_func, img_func



if __name__=="__main__":
    print("Hello world!")







##############
# EYE BUFFER #
##############
