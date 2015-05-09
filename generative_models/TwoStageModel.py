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

class TwoStageModel(object):
    """
    Controller for training a multi-step iterative refinement model.

    Parameters:
        rng: numpy.random.RandomState (for reproducibility)
        x_in: the input data to encode
        x_out: the target output to decode
        p_s_given_z: InfNet for initializing "canvas" state
        p_h_given_s: InfNet for h given s
        p_x_given_s_h: InfNet for x given s and h
        q_z_given_x: InfNet for z given x
        q_h_given_x_s: InfNet for h given x and s
        x_dim: dimension of the observations to generate
        z_dim: dimension of the "first" latent space
        s_dim: dimension of the "second" latent space
        h_dim: dimension of the "third" latent space
        params: REQUIRED PARAMS SHOWN BELOW
                x_type: can be "bernoulli" or "gaussian"
                obs_transform: can be 'none' or 'sigmoid'
    """
    def __init__(self, rng=None, \
            x_in=None, x_out=None, \
            p_s_given_z=None, \
            p_h_given_s=None, \
            p_x_given_s_h=None, \
            q_z_given_x=None, \
            q_h_given_x_s=None, \
            x_dim=None, \
            z_dim=None, \
            s_dim=None, \
            h_dim=None, \
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
        self.shared_param_dicts = shared_param_dicts

        # record the dimensions of various spaces relevant to this model
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.s_dim = s_dim
        self.h_dim = h_dim

        # grab handles to the relevant InfNets
        self.q_z_given_x = q_z_given_x
        self.q_h_given_x_s = q_h_given_x_s
        self.p_s_given_z = p_s_given_z
        self.p_h_given_s = p_h_given_s
        self.p_x_given_s_h = p_x_given_s_h

        # record the symbolic variables that will provide inputs to the
        # computation graph created to describe this MultiStageModel
        self.x_in = x_in
        self.x_out = x_out
        self.batch_reps = T.lscalar()

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
            init_vec = to_fX( np.zeros((self.x_dim,)) )
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

        ##############################################
        # Setup the TwoStageModels main computation. #
        ##############################################
        print("Building TSM...")
        # samples of "first" latent state
        drop_x = drop_mask * self.x_in
        z_q_mean, z_q_logvar, self.z = \
                self.q_z_given_x.apply(drop_x, do_samples=True)
        # compute relevant KLds for this step
        self.kld_z_q2ps = gaussian_kld(z_q_mean, z_q_logvar, \
                                       self.p_z_mean, self.p_z_logvar)
        self.kld_z_p2qs = gaussian_kld(self.p_z_mean, self.p_z_logvar, \
                                       z_q_mean, z_q_logvar)
        # transform "first" latent state into "second" latent state
        self.s, _ = self.p_s_given_z.apply(self.z, do_samples=False)

        # get samples of h, conditioned on current s
        h_p_mean, h_p_logvar, h_p = self.p_h_given_s.apply( \
                self.s, do_samples=True)
        # get variational samples of h, given s and x_out
        h_q_mean, h_q_logvar, h_q = self.q_h_given_x_s.apply( \
                T.horizontal_stack(self.x_out, self.s), \
                do_samples=True)

        # make h samples that can be switched between h_p and h_q
        self.h = (self.train_switch[0] * h_q) + \
                 ((1.0 - self.train_switch[0]) * h_p)

        # compute relevant KLds for this step
        self.kld_h_q2ps = gaussian_kld(h_q_mean, h_q_logvar, \
                                       h_p_mean, h_p_logvar)
        self.kld_h_p2qs = gaussian_kld(h_p_mean, h_p_logvar, \
                                       h_q_mean, h_q_logvar)

        # p_x_given_s_h is conditioned on s and  h.
        self.x_gen, _ = self.p_x_given_s_h.apply( \
                T.horizontal_stack(self.s, self.h), \
                do_samples=False)

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
        self.group_1_params = []
        self.group_1_params.extend(self.q_z_given_x.mlp_params)
        self.group_1_params.extend(self.q_h_given_x_s.mlp_params)
        # Grab all of the "optimizable" parameters in "group 2"
        self.group_2_params = [self.p_z_mean, self.p_z_logvar]
        self.group_2_params.extend(self.p_s_given_z.mlp_params)
        self.group_2_params.extend(self.p_h_given_s.mlp_params)
        self.group_2_params.extend(self.p_x_given_s_h.mlp_params)

        # Make a joint list of parameters group 1/2
        self.joint_params = self.group_1_params + self.group_2_params

        #################################
        # CONSTRUCT THE KLD-BASED COSTS #
        #################################
        self.kld_z_q2p, self.kld_z_p2q, self.kld_h_q2p, self.kld_h_p2q = \
                self._construct_kld_costs(p=1.0)
        self.kld_z = (self.lam_kld_q2p[0] * self.kld_z_q2p) + \
                     (self.lam_kld_p2q[0] * self.kld_z_p2q)
        self.kld_h = (self.lam_kld_q2p[0] * self.kld_h_q2p) + \
                     (self.lam_kld_p2q[0] * self.kld_h_p2q)
        self.kld_costs = (self.lam_kld_z[0] * self.kld_z) + self.kld_h
        # now do l2 KLd costs
        self.kl2_z_q2p, self.kl2_z_p2q, self.kl2_h_q2p, self.kl2_h_p2q = \
                self._construct_kld_costs(p=2.0)
        self.kl2_z = (self.lam_kld_q2p[0] * self.kl2_z_q2p) + \
                     (self.lam_kld_p2q[0] * self.kl2_z_p2q)
        self.kl2_h = (self.lam_kld_q2p[0] * self.kl2_h_q2p) + \
                     (self.lam_kld_p2q[0] * self.kl2_h_p2q)
        self.kl2_costs = (self.lam_kld_z[0] * self.kl2_z) + self.kl2_h
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
        self.nll_costs = self._construct_nll_costs(self.x_out)
        self.nll_cost = self.lam_nll[0] * T.mean(self.nll_costs)
        ########################################
        # CONSTRUCT THE REST OF THE JOINT COST #
        ########################################
        param_reg_cost = self._construct_reg_costs()
        self.reg_cost = self.lam_l2w[0] * param_reg_cost
        self.joint_cost = self.nll_cost + self.kld_l1l2_cost + self.reg_cost
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
        self.group_1_updates = get_adam_updates(params=self.group_1_params, \
                grads=self.joint_grads, alpha=self.lr_1, \
                beta1=self.mom_1, beta2=self.mom_2, \
                mom2_init=1e-3, smoothing=1e-5, max_grad_norm=10.0)
        self.group_2_updates = get_adam_updates(params=self.group_2_params, \
                grads=self.joint_grads, alpha=self.lr_2, \
                beta1=self.mom_1, beta2=self.mom_2, \
                mom2_init=1e-3, smoothing=1e-5, max_grad_norm=10.0)
        self.joint_updates = OrderedDict()
        for k in self.group_1_updates:
            self.joint_updates[k] = self.group_1_updates[k]
        for k in self.group_2_updates:
            self.joint_updates[k] = self.group_2_updates[k]

        # Construct a function for jointly training the generator/inferencer
        print("Compiling training function...")
        self.train_joint = self._construct_train_joint()
        print("Compiling free-energy sampler...")
        self.compute_fe_terms = self._construct_compute_fe_terms()
        print("Compiling open-loop model sampler...")
        self.sample_from_prior = self._construct_sample_from_prior()
        print("Compiling data-guided model sampler...")
        self.sample_from_input = self._construct_sample_from_input()
        # make easy access points for some interesting parameters
        self.gen_gen_weights = self.p_x_given_s_h.mu_layers[-1].W
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

    def _construct_nll_costs(self, xo):
        """
        Construct the negative log-likelihood part of free energy.
        """
        # average log-likelihood over the refinement sequence
        xh = self.obs_transform(self.x_gen)
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
        kld_z_q2p = T.sum(self.kld_z_q2ps**p, axis=1, keepdims=True)
        kld_z_p2q = T.sum(self.kld_z_p2qs**p, axis=1, keepdims=True)
        kld_h_q2p = T.sum(self.kld_h_q2ps**p, axis=1, keepdims=True)
        kld_h_p2q = T.sum(self.kld_h_p2qs**p, axis=1, keepdims=True)
        return [kld_z_q2p, kld_z_p2q, kld_h_q2p, kld_h_p2q]

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
                   self.reg_cost, self.obs_costs]
        # compile the theano function
        func = theano.function(inputs=[ xi, xo, self.batch_reps ], \
                outputs=outputs, \
                givens={ self.x_in: xi.repeat(self.batch_reps, axis=0), \
                         self.x_out: xo.repeat(self.batch_reps, axis=0) }, \
                updates=self.joint_updates)
        return func

    def _construct_compute_fe_terms(self):
        """
        Construct a function for computing terms in variational free energy.
        """
        # construct values to output
        nll = self._construct_nll_costs(self.x_out)
        kld = self.kld_z_q2p + self.kld_h_q2p
        # compile theano function for a one-sample free-energy estimate
        fe_term_sample = theano.function(inputs=[self.x_in, self.x_out], \
                                         outputs=[nll, kld])
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
        sample_func = theano.function(inputs=[z_sym, x_sym], \
                outputs=self.obs_transform(self.x_gen), \
                givens={self.z: z_sym, \
                        self.x_in: T.zeros_like(x_sym), \
                        self.x_out: T.zeros_like(x_sym)})
        def prior_sampler(samp_count):
            x_samps = to_fX( np.zeros((samp_count, self.x_dim)) )
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
        sample_func = theano.function(inputs=[self.x_in, self.x_out], \
                outputs=self.obs_transform(self.x_gen))
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
