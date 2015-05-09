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
        p_sip1_given_si_hi: InfNet for sip1 given si and hi
        q_z_given_x: InfNet for z given x
        q_hi_given_x_si: InfNet for hi given x and si
        model_init_mix: whether to use a model-based initial mix state
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
        self.batch_reps = T.lscalar()
        self.iter_clock = T.eye(self.ir_steps)

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
        self.s0_obs, _ = self.p_s0_given_z.apply(self.z, do_samples=False)

        ###############################################################
        # Setup the iterative refinement loop, starting from self.s0. #
        ###############################################################
        self.si_obs = [self.s0_obs]   # holds si_obs for each i
        self.hi = []                  # holds hi for each i
        self.kldi_q2p = []            # holds KL(q || p) for each i
        self.kldi_p2q = []            # holds KL(p || q) for each i
        self.enti_q = []              # holds ENT(q) for each i
        self.enti_p = []              # holds ENT(p) for each i

        for i in range(self.ir_steps):
            print("Building MSM step {0:d}...".format(i+1))
            # get variables used throughout this refinement step
            si_obs = self.si_obs[i]
            si_obs_trans = self.obs_transform(si_obs)
            grad_ll = self.x_out - si_obs_trans
            clock_vals = T.alloc(0.0, si_obs.shape[0], self.ir_steps) + \
                    self.iter_clock[i]

            # get a drop mask that drops things with probability p
            drop_scale = 1. / (1. - self.drop_rate[0])
            drop_rnd = self.rng.uniform(size=self.x_out.shape, \
                    low=0.0, high=1.0, dtype=theano.config.floatX)
            drop_mask = drop_scale * (drop_rnd > self.drop_rate[0])

            # get droppy versions of
            drop_obs = drop_mask * si_obs_trans
            drop_grad = drop_mask * grad_ll

            # get samples of next hi, conditioned on current si
            hi_p_mean, hi_p_logvar, hi_p = self.p_hi_given_si.apply( \
                    drop_obs, do_samples=True)
            # now we build the model for variational hi given si
            hi_q_mean, hi_q_logvar, hi_q = self.q_hi_given_x_si.apply( \
                    T.horizontal_stack(drop_grad, drop_obs), \
                    do_samples=True)

            # make hi samples that can be switched between hi_p and hi_q
            self.hi.append( ((self.train_switch[0] * hi_q) + \
                    ((1.0 - self.train_switch[0]) * hi_p)) )

            # compute relevant KLds for this step
            self.kldi_q2p.append(gaussian_kld( \
                hi_q_mean, hi_q_logvar, hi_p_mean, hi_p_logvar))
            self.kldi_p2q.append(gaussian_kld( \
                hi_p_mean, hi_p_logvar, hi_q_mean, hi_q_logvar))
            # compute relevant "entropy" costs for this step
            self.enti_q.append(( \
                    (0.10 * hi_q_mean**2.0) - \
                    (0.50 * hi_q_logvar) + \
                    (0.05 * hi_q_logvar**2.0)) )
            self.enti_p.append(( \
                    (0.10 * hi_p_mean**2.0) - \
                    (0.50 * hi_p_logvar) + \
                    (0.05 * hi_p_logvar**2.0)) )

            # p_sip1_given_si_hi is conditioned on si and  hi.
            si_obs_step, _ = self.p_sip1_given_si_hi.apply( \
                    T.horizontal_stack(self.hi[i], si_obs), \
                    do_samples=False)
                    
            # construct the update from si_obs to sip1_obs
            sip1_obs = si_obs + si_obs_step
            # record the updated state of the generative process
            self.si_obs.append(sip1_obs)

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
        # setup stuff for controlling entropy in the primary and guide policies
        self.lam_ent_q = theano.shared(value=zero_ary, name='msm_lam_ent_q')
        self.lam_ent_p = theano.shared(value=zero_ary, name='msm_lam_ent_p')
        self.set_lam_ent(lam_ent_q=0.1, lam_ent_p=0.0)
        # init shared var for controlling l2 regularization on params
        self.lam_l2w = theano.shared(value=zero_ary, name='msm_lam_l2w')
        self.set_lam_l2w(1e-5)

        # Grab all of the "optimizable" parameters in "group 1"
        self.group_1_params = []
        self.group_1_params.extend(self.q_z_given_x.mlp_params)
        self.group_1_params.extend(self.q_hi_given_x_si.mlp_params)
        # Grab all of the "optimizable" parameters in "group 2"
        self.group_2_params = [self.p_z_mean, self.p_z_logvar]
        self.group_2_params.extend(self.p_hi_given_si.mlp_params)
        self.group_2_params.extend(self.p_sip1_given_si_hi.mlp_params)
        self.group_2_params.extend(self.p_s0_given_z.mlp_params)

        # Make a joint list of parameters group 1/2
        self.joint_params = self.group_1_params + self.group_2_params

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
        for i in range(len(self.si_obs)):
            nll_i = self._construct_nll_costs(self.x_out, step_num=i)
            self.seq_nll_costs.append(nll_i)
        self.nll_costs = self.seq_nll_costs[-1]
        self.nll_cost = self.lam_nll[0] * T.mean(self.nll_costs)
        ########################################
        # CONSTRUCT THE REST OF THE JOINT COST #
        ########################################
        param_reg_cost = self._construct_reg_costs()
        self.reg_cost = self.lam_l2w[0] * param_reg_cost
        self.joint_cost = self.nll_cost + self.kld_l1l2_cost + \
                          self.ent_cost + self.reg_cost
        ##############################
        # CONSTRUCT A PER-INPUT COST #
        ##############################
        self.obs_costs = self.nll_costs + self.kld_l1l2_costs + self.ent_costs

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
                mom2_init=1e-3, smoothing=1e-4, max_grad_norm=10.0)
        self.group_2_updates = get_adam_updates(params=self.group_2_params, \
                grads=self.joint_grads, alpha=self.lr_2, \
                beta1=self.mom_1, beta2=self.mom_2, \
                mom2_init=1e-3, smoothing=1e-4, max_grad_norm=10.0)
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
        self.gen_inf_weights = self.p_hi_given_si.shared_layers[0].W
        self.gen_gen_weights = self.p_sip1_given_si_hi.mu_layers[-1].W
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

    def set_lam_ent(self, lam_ent_q=0.1, lam_ent_p=0.0):
        """
        Set the relative weight of various entropy penalties/rewards.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_ent_q
        self.lam_ent_q.set_value(to_fX(new_lam))
        new_lam = zero_ary + lam_ent_p
        self.lam_ent_p.set_value(to_fX(new_lam))
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

    def _construct_nll_costs(self, xo, step_num=-1):
        """
        Construct the negative log-likelihood part of free energy.
        """
        # average log-likelihood over the refinement sequence
        sn = 20.0 * T.tanh(0.05 * self.si_obs[step_num])
        xh = self.obs_transform(sn)
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

    def _construct_ent_costs(self):
        """
        Construct the posterior entropy part of cost to minimize.
        """
        ent_hi_qs = []
        ent_hi_ps = []
        for i in range(self.ir_steps):
            ent_hi_q = self.enti_q[i]
            ent_hi_p = self.enti_p[i]
            ent_hi_qs.append(T.sum(ent_hi_q, axis=1, keepdims=True))
            ent_hi_ps.append(T.sum(ent_hi_p, axis=1, keepdims=True))
        # compute the batch-wise costs
        ent_hi_q = sum(ent_hi_qs)
        ent_hi_p = sum(ent_hi_ps)
        # construct entropy cost for z
        ent_z_qs = (0.1 * self.q_z_mean**2.0) - (0.5 * self.q_z_logvar)
        ent_z_ps = T.zeros_like(ent_z_qs) + \
                   (0.1 * self.p_z_mean**2.0) - (0.5 * self.p_z_logvar)
        ent_z_q = T.sum(ent_z_qs, axis=1, keepdims=True)
        ent_z_p = T.sum(ent_z_ps, axis=1, keepdims=True)
        # sum costs for z and hi
        ent_p = ent_z_p + ent_hi_p
        ent_q = ent_z_q + ent_hi_q
        return [ent_p, ent_q]

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
                   self.ent_cost, self.reg_cost, self.obs_costs]
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
        init_nlls = self._construct_nll_costs(self.x_out, step_num=0)
        step_nlls = []
        for i in range(self.ir_steps):
            step_nlls.append(self._construct_nll_costs(self.x_out, \
                             step_num=(i+1)))
        nlli = T.stack(*step_nlls)
        # get KLd for all steps
        init_klds = gaussian_kld(self.q_z_mean, self.q_z_logvar, \
                self.p_z_mean, self.p_z_logvar)
        kld_hi_q2p_list = []
        kld_hi_p2q_list = []
        for i in range(self.ir_steps):
            kld_hi_q2p = self.kldi_q2p[i]
            kld_hi_p2q = self.kldi_p2q[i]
            kld_hi_q2p_list.append(kld_hi_q2p)
            kld_hi_p2q_list.append(kld_hi_p2q)
        kldi_q2p = T.stack(*kld_hi_q2p_list)
        kldi_p2q = T.stack(*kld_hi_p2q_list)
        # gather step-wise costs into a single list (init costs at the end)
        all_step_costs = [nlli, kldi_q2p, kldi_p2q, init_nlls, init_klds]
        # compile theano function for computing all relevant costs
        inputs = [self.x_in, self.x_out]
        cost_func = theano.function(inputs=inputs, outputs=all_step_costs)
        def raw_cost_computer(XI, XO):
            _all_costs = cost_func(XI, XO)
            _init_nlls = _all_costs[-2]
            _init_klds = _all_costs[-1]
            _kld_q2p = np.sum(np.mean(_all_costs[1], axis=1, keepdims=True), axis=0)
            _kld_p2q = np.sum(np.mean(_all_costs[2], axis=1, keepdims=True), axis=0)
            _step_klds = np.mean(np.sum(_all_costs[1], axis=2, keepdims=True), axis=1)
            sk = [np.sum(np.mean(_init_klds, axis=0))]
            sk.extend([k for k in _step_klds])
            _step_klds = to_fX( np.asarray(sk) )
            _step_nlls = np.mean(_all_costs[0], axis=1)
            sn = [np.mean(_init_nlls, axis=0)]
            sn.extend([k for k in _step_nlls])
            _step_nlls = to_fX( np.asarray(sn) )
            results = [_init_nlls, _init_klds, _kld_q2p, _kld_p2q, \
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
        nll = self._construct_nll_costs(xo, step_num=-1)
        kld = self.kld_z_q2p + self.kld_hi_q2p
        # compile theano function for a one-sample free-energy estimate
        fe_term_sample = theano.function(inputs=[ xi, xo ], \
                outputs=[nll, kld], \
                givens={self.x_in: xi, \
                        self.x_out: xo})
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
        oputs = [self.obs_transform(s) for s in self.si_obs]
        sample_func = theano.function(inputs=[z_sym, x_sym], outputs=oputs, \
                givens={self.z: z_sym, \
                        self.x_in: T.zeros_like(x_sym), \
                        self.x_out: T.zeros_like(x_sym)})
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

        The 
        """
        xi = T.matrix()
        xo = T.matrix()
        oputs = [self.obs_transform(s) for s in self.si_obs]
        sample_func = theano.function(inputs=[xi, xo], outputs=oputs, \
                givens={self.x_in: xi, \
                        self.x_out: xo})
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
