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
                      apply_mask, to_fX
from InfNet import InfNet
from PeaNet import PeaNet
from DKCode import get_adam_updates, get_adadelta_updates
from LogPDFs import log_prob_bernoulli, log_prob_gaussian2, gaussian_kld

#
# Important symbolic variables:
#   Xd: Xd represents input at the "data variables" of the inferencer
#

class GPSImputer(object):
    """
    Controller for training a multi-step imputater via guided policy search.

    Parameters:
        rng: numpy.random.RandomState (for reproducibility)
        x_in: the initial state for imputation
        x_out: the goal state for imputation
        x_mask: mask for state dims to keep fixed during imputation
        p_zi_given_xi: InfNet for stochastic part of step
        p_xip1_given_zi: InfNet for deterministic part of step
        q_zi_given_x_xi: InfNet for the guide policy
        obs_dim: dimension of the observations to generate
        z_dim: dimension of the stochastic component of p policies
        imp_steps: number of steps to run imputation procedure
        step_type: whether to use "add" steps or "swap" steps
        params: REQUIRED PARAMS SHOWN BELOW
                x_type: can be "bernoulli" or "gaussian"
                obs_transform: can be 'none' or 'sigmoid'
    """
    def __init__(self, rng=None, 
            x_in=None, x_mask=None, x_out=None, \
            p_zi_given_xi=None, \
            p_xip1_given_zi=None, \
            q_zi_given_x_xi=None, \
            obs_dim=None, \
            z_dim=None, \
            imp_steps=4, \
            step_type='add', \
            params=None, \
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
        self.shared_param_dicts = shared_param_dicts

        # record the dimensions of various spaces relevant to this model
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.imp_steps = imp_steps
        assert((step_type == 'add') or (step_type == 'swap'))
        self.step_type = step_type
        if self.use_osm_mode:
            self.step_type = 'swap'

        # grab handles to the relevant InfNets
        self.p_zi_given_xi = p_zi_given_xi
        self.p_xip1_given_zi = p_xip1_given_zi
        self.q_zi_given_x_xi = q_zi_given_x_xi

        # record the symbolic variables that will provide inputs to the
        # computation graph created to describe this MultiStageModel
        self.x_in = x_in
        self.x_out = x_out
        self.x_mask = x_mask
        self.batch_reps = T.lscalar()

        # setup switching variable for changing between sampling/training
        zero_ary = to_fX( np.zeros((1,)) )
        self.train_switch = theano.shared(value=zero_ary, name='msm_train_switch')
        self.set_train_switch(1.0)

        if self.shared_param_dicts is None:
            # initialize misc. parameters
            b_init = to_fX( np.zeros((self.obs_dim,)) )
            self.b_obs = theano.shared(value=b_init, name='msm_b_obs')
            self.obs_logvar = theano.shared(value=zero_ary, name='msm_obs_logvar')
            self.bounded_logvar = 8.0 * T.tanh((1.0/8.0) * self.obs_logvar[0])
            self.shared_param_dicts = {}
            self.shared_param_dicts['obs_logvar'] = self.obs_logvar
        else:
            self.obs_logvar = self.shared_param_dicts['obs_logvar']
            self.bounded_logvar = 8.0 * T.tanh((1.0/8.0) * self.obs_logvar[0])

        ##########################################
        # Setup the multi-stage imputation loop. #
        ##########################################
        self.xi = []            # holds xi for each i
        self.zi = []            # holds z samples for each i
        self.kldi_p2q = []      # KL(p || q) for each i
        self.kldi_q2p = []      # KL(q || p) for each i
        self.enti_q = []        # "entropy" for q for each step
        self.enti_p = []        # "entropy" for p for each step

        for i in range(self.imp_steps):
            print("Building imputation step {0:d}...".format(i+1))
            # get variables used throughout this refinement step
            if (i == 0):
                xi_recon = self.x_in
            else:
                xi_recon = self.obs_transform(self.xi[i-1])
            xi_masked = (self.x_mask * self.x_in) + \
                        ((1.0 - self.x_mask) * xi_recon)
            grad_ll = xi_masked - self.x_out

            # get samples of next zi, according to the global policy
            zi_p_mean, zi_p_logvar, zi_p = self.p_zi_given_xi.apply( \
                    xi_masked, do_samples=True)
            # get samples of next zi, according to the guide policy
            zi_q_mean, zi_q_logvar, zi_q = self.q_zi_given_x_xi.apply( \
                    T.horizontal_stack(self.x_out, xi_masked), \
                    do_samples=True)

            if self.use_osm_mode:
                self.zi.append(zi_p)
                # compute relevant KLds for this step
                self.kldi_q2p.append(gaussian_kld( \
                    zi_p_mean, zi_p_logvar, 0.0, 0.0))
                self.kldi_p2q.append(gaussian_kld( \
                    zi_p_mean, zi_p_logvar, 0.0, 0.0))
                # compute relevant "entropies" for this step
                self.enti_q.append(( \
                        (0.10 * zi_p_mean**2.0) - \
                        (0.50 * zi_p_logvar) + \
                        (0.05 * zi_p_logvar**2.0)) )
                self.enti_p.append(( \
                        (0.10 * zi_p_mean**2.0) - \
                        (0.50 * zi_p_logvar) + \
                        (0.05 * zi_p_logvar**2.0)) )
            else:
                # make zi samples that can be switched between zi_p and zi_q
                self.zi.append( ((self.train_switch[0] * zi_q) + \
                        ((1.0 - self.train_switch[0]) * zi_p)) )
                # compute relevant KLds for this step
                self.kldi_q2p.append(gaussian_kld( \
                    zi_q_mean, zi_q_logvar, zi_p_mean, zi_p_logvar))
                self.kldi_p2q.append(gaussian_kld( \
                    zi_p_mean, zi_p_logvar, zi_q_mean, zi_q_logvar))
                # compute relevant "entropies" for this step
                self.enti_q.append(( \
                        (0.10 * zi_q_mean**2.0) - \
                        (0.50 * zi_q_logvar) + \
                        (0.05 * zi_q_logvar**2.0)) )
                self.enti_p.append(( \
                        (0.10 * zi_p_mean**2.0) - \
                        (0.50 * zi_p_logvar) + \
                        (0.05 * zi_p_logvar**2.0)) )

            # compute the next xi, given the sampled zi
            xi_step, _ = self.p_xip1_given_zi.apply(self.zi[i], do_samples=False)
            if (self.step_type == 'swap'):
                # swap steps always do a full swap (like standard VAE)
                xip1 = xi_step
            else:
                # we're here because we're using additive steps...
                if (i == 0):
                    # first additive step will also receive a bias
                    xip1 = xi_step + self.b_obs
                else:
                    # subsequent additive steps just add
                    xip1 = self.xi[i-1] + xi_step
            self.xi.append( xip1 )

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
        # setup stuff for controlling entropy in the primary and guide policies
        self.lam_ent_q = theano.shared(value=zero_ary, name='msm_lam_ent_q')
        self.lam_ent_p = theano.shared(value=zero_ary, name='msm_lam_ent_p')
        self.set_lam_ent(lam_ent_p=0.0, lam_ent_q=0.01)
        # init shared var for controlling l2 regularization on params
        self.lam_l2w = theano.shared(value=zero_ary, name='msm_lam_l2w')
        self.set_lam_l2w(1e-5)

        # Grab all of the "optimizable" parameters in "group 1"
        self.joint_params = [self.b_obs, self.obs_logvar]
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
        #####################################
        # CONSTRUCT THE ENTROPY-BASED COSTS #
        #####################################
        self.ent_p, self.ent_q = self._construct_ent_costs()
        self.ent_costs = (self.lam_ent_p[0] * self.ent_p) + \
                         (self.lam_ent_q[0] * self.ent_q)
        self.ent_cost = T.mean(self.ent_costs)
        #################################
        # CONSTRUCT THE NLL-BASED COSTS #
        #################################
        self.seq_nll_costs = []
        for i in range(len(self.xi)):
            nll_i = self._construct_nll_costs(self.x_out, self.x_mask, \
                                              step_num=i)
            self.seq_nll_costs.append(nll_i)
        self.nll_costs = self.seq_nll_costs[-1]
        self.nll_cost = self.lam_nll[0] * T.mean(self.nll_costs)
        ########################################
        # CONSTRUCT THE REST OF THE JOINT COST #
        ########################################
        param_reg_cost = self._construct_reg_costs()
        self.reg_cost = self.lam_l2w[0] * param_reg_cost
        self.joint_cost = self.nll_cost + self.kld_cost + self.ent_cost + \
                          self.reg_cost
        ##############################
        # CONSTRUCT A PER-TRIAL COST #
        ##############################
        self.obs_costs = self.nll_costs + self.kld_costs + self.ent_costs

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
                mom2_init=1e-3, smoothing=1e-4, max_grad_norm=10.0)

        # Construct a function for jointly training the generator/inferencer
        print("Compiling cost computer...")
        self.compute_raw_costs = self._construct_raw_costs()
        print("Compiling training function...")
        self.train_joint = self._construct_train_joint()
        print("Compiling data-guided imputer sampler...")
        self.sample_imputer = self._construct_sample_imputer()
        # make easy access points for some interesting parameters
        self.gen_inf_weights = self.p_zi_given_xi.shared_layers[0].W
        self.gen_gen_weights = self.p_xip1_given_zi.mu_layers[-1].W
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

    def set_lam_ent(self, lam_ent_p=0.1, lam_ent_q=0.0):
        """
        Set the relative weight of various entropy penalties/rewards.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_ent_p
        self.lam_ent_p.set_value(to_fX(new_lam))
        new_lam = zero_ary + lam_ent_q
        self.lam_ent_q.set_value(to_fX(new_lam))
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

    def _construct_nll_costs(self, xo, xm, step_num=-1):
        """
        Construct the negative log-likelihood part of free energy.
        """
        # average log-likelihood over the refinement sequence
        xh = self.obs_transform( self.xi[step_num] )
        xm_inv = 1.0 - xm # we will measure nll only where xm_inv is 1
        if self.x_type == 'bernoulli':
            ll_costs = log_prob_bernoulli(xo, xh, mask=xm_inv)
        else:
            ll_costs = log_prob_gaussian2(xo, xh, \
                    log_vars=self.bounded_logvar, mask=xm_inv)
        nll_costs = -ll_costs
        return nll_costs

    def _construct_kld_costs(self, p=1.0):
        """
        Construct the policy KL-divergence part of cost to minimize.
        """
        kld_pis = []
        kld_qis = []
        for i in range(self.imp_steps):
            kldpi = self.kldi_p2q[i]
            kldqi = self.kldi_q2p[i]
            kld_pis.append(T.sum(kldpi**p, axis=1, keepdims=True))
            kld_qis.append(T.sum(kldqi**p, axis=1, keepdims=True))
        # compute the batch-wise costs
        kld_pi = sum(kld_pis)
        kld_qi = sum(kld_qis)
        return [kld_pi, kld_qi]

    def _construct_ent_costs(self):
        """
        Construct the policy entropy part of cost to minimize.
        """
        ent_pis = []
        ent_qis = []
        for i in range(self.imp_steps):
            entpi = self.enti_p[i]
            entqi = self.enti_q[i]
            ent_pis.append(T.sum(entpi, axis=1, keepdims=True))
            ent_qis.append(T.sum(entqi, axis=1, keepdims=True))
        # compute the batch-wise costs
        ent_pi = sum(ent_pis)
        ent_qi = sum(ent_qis)
        return [ent_pi, ent_qi]

    def _construct_reg_costs(self):
        """
        Construct the cost for low-level basic regularization. E.g. for
        applying l2 regularization to the network activations and parameters.
        """
        param_reg_cost = sum([T.sum(p**2.0) for p in self.joint_params])
        return param_reg_cost

    def _construct_raw_costs(self):
        """
        Construct all the raw, i.e. not weighted by any lambdas, costs.
        """
        # get NLL for all steps (per-obs, per-step, but pre-summed over dims)
        step_nlls = []
        for i in range(self.imp_steps):
            step_nlls.append(self._construct_nll_costs(self.x_out, \
                             self.x_mask, step_num=i))
        nlli = T.stack(*step_nlls)
        # get KLd for all steps (per-obs, per-step, and per-dim)
        kldi_q2p = T.stack(*self.kldi_q2p)
        kldi_p2q = T.stack(*self.kldi_p2q)
        # gather step-wise costs into a single list
        all_step_costs = [nlli, kldi_q2p, kldi_p2q]
        # compile theano function for computing the costs
        inputs = [self.x_in, self.x_out, self.x_mask]
        cost_func = theano.function(inputs=inputs, outputs=all_step_costs, \
                                    on_unused_input='ignore')
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

    def _construct_train_joint(self):
        """
        Construct theano function to train all networks jointly.
        """
        # setup some symbolic variables for theano to deal with
        xi = T.matrix()
        xo = T.matrix()
        xm = T.matrix()
        # collect the outputs to return from this function
        outputs = [self.joint_cost, self.nll_cost, self.kld_cost, \
                   self.ent_cost, self.reg_cost, self.obs_costs]
        # compile the theano function
        func = theano.function(inputs=[ xi, xo, xm, self.batch_reps ], \
                outputs=outputs, \
                givens={ self.x_in: xi.repeat(self.batch_reps, axis=0), \
                         self.x_out: xo.repeat(self.batch_reps, axis=0), \
                         self.x_mask: xm.repeat(self.batch_reps, axis=0) }, \
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
        oputs = [self.x_in] + [self.obs_transform(_xi_) for _xi_ in self.xi]
        sample_func = theano.function(inputs=[xi, xo, xm], outputs=oputs, \
                givens={self.x_in: xi, \
                        self.x_out: xo, \
                        self.x_mask: xm}, \
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

if __name__=="__main__":
    print("Hello world!")







##############
# EYE BUFFER #
##############
