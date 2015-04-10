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
                      apply_mask, constFX, to_fX, DCG
from GenNet import GenNet
from InfNet import InfNet
from PeaNet import PeaNet
from DKCode import get_adam_updates, get_adadelta_updates
from LogPDFs import log_prob_bernoulli, log_prob_gaussian2, gaussian_kld
from MSDUtils import SimpleLSTM, SimpleMLP, SimpleReader, SimpleWriter

#
# Important symbolic variables:
#   Xd: Xd represents input at the "data variables" of the inferencer
#

def sample_transform(new_means, new_logvars, randn_zmuv):
    new_samples = new_means + (T.exp(0.5*new_logvars) * randn_zmuv)
    return new_samples

###############################################
# MultiStageModel with DRAW-like architecture #
###############################################

class MultiStageDRAW(object):
    """
    Controller for training a multi-step iterative refinement model.

    Parameters:
        rng: numpy.random.RandomState (for reproducibility)
        x_in: the input data to encode
        x_out: the target output to decode
        q_z_given_x: InfNet for z given x
        init_from_z: whether to use z to initialize mix/enc/dec state
        p_hi_given_sim1_dec: InfNet for hi given current decoder state
        q_hi_given_si_enc: InfNet for hi given current encoder state
        p_writer: writer object for modifying canvas
        q_reader: reader object for reading canvas, etc.
        obs_dim: dimension of the "observation" component of state
        rnn_dim: dimension of the "RNN" component of state
        mix_dim: dimension of the "mixture" component of state
        read_dim: dimension of the reader output
        h_dim: dimension of the "primary" latent space
        z_dim: dimension of the "initial" latent space
        ir_steps: number of "iterative refinement" steps to perform
        params: REQUIRED PARAMS SHOWN BELOW
                x_type: can be "bernoulli" or "gaussian"
                obs_transform: can be "none" or "sigmoid"
    """
    def __init__(self, rng=None, x_in=None, x_out=None, \
            q_z_given_x=None, init_from_z=True, \
            p_hi_given_sim1_dec=None, q_hi_given_si_enc=None, \
            p_writer=None, q_reader=None, \
            z_dim=None, h_dim=None, \
            obs_dim=None, rnn_dim=None, mix_dim=None, read_dim=None, \
            ir_steps=2, params=None):
        # setup a rng for this GIPair
        self.rng = RandStream(rng.randint(100000))

        # record the symbolic variables that will provide inputs to the
        # computation graph created to describe this MultiStageDRAW
        self.x_in = x_in
        self.x_out = x_out
        self.z_zmuv = T.matrix('z_zmuv')
        self.hi_zmuv = T.tensor3('hi_zmuv')

        self.p_hi_given_sim1_dec = p_hi_given_sim1_dec
        self.q_hi_given_si_enc = q_hi_given_si_enc
        self.p_writer = p_writer
        self.q_reader = q_reader

        # decide whether to initialize state from "z" or from a "constant"
        self.init_from_z = init_from_z

        # grab the user-provided parameters
        self.params = params
        self.x_type = self.params['x_type']
        assert((self.x_type == 'bernoulli') or (self.x_type == 'gaussian'))
        if 'obs_transform' in self.params:
            # this describes a transform to apply to the "canvas" prior to
            # comparing it with the target input
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

        # record the dimensions of various spaces relevant to this model
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.obs_dim = obs_dim
        self.rnn_dim = rnn_dim
        self.mix_dim = mix_dim
        self.read_dim = read_dim
        self.ir_steps = ir_steps

        # setup switching variable for changing between sampling/training
        zero_ary = to_fX( np.zeros((1,)) )
        self.train_switch = theano.shared(value=zero_ary, name='msm_train_switch')
        self.set_train_switch(1.0)
        # setup a weight for pulling priors over hi given si towards a
        # shared global prior -- e.g. zero mean and unit variance.
        self.kzg_weight = theano.shared(value=zero_ary, name='msm_kzg_weight')
        self.set_kzg_weight(0.1)
        # this weight balances l1 vs. l2 penalty on posterior KLds
        self.l1l2_weight = theano.shared(value=zero_ary, name='msm_l1l2_weight')
        self.set_l1l2_weight(1.0)

        # setup trainable input/output bias parameters
        b_input = to_fX( np.zeros((self.obs_dim,)) )
        b_obs = to_fX( np.zeros((self.obs_dim,)) )
        self.b_input = theano.shared(value=b_input, name='msm_b_input')
        self.b_obs = theano.shared(value=b_obs, name='msm_b_obs')

        # initialize MLP for: z -> s_mix, s0_enc, c0_enc, s0_dec, c0_dec
        d = [self.z_dim, (self.mix_dim + 4*self.rnn_dim)]
        self.mlp_mix = SimpleMLP(d[0], d[1], name='msm_mlp_mix')
        # initialize MLP for: s_mix, read, sim1_dec -> xi_enc
        d = [(self.mix_dim + self.read_dim + self.rnn_dim), 4*self.rnn_dim]
        self.q_mlp_xi = SimpleMLP(d[0], d[1], name='msm_q_mlp_xi')
        # initialize MLP for: s_mix, hi -> xi_dec
        d = [(self.mix_dim + self.h_dim), 4*self.rnn_dim]
        self.p_mlp_xi = SimpleMLP(d[0], d[1], name='msm_p_mlp_xi')
        # initialize LSTM for encoder and decoder (assume same size)
        d = [4*self.rnn_dim, self.rnn_dim]
        self.q_lstm = SimpleLSTM(d[0], d[1], name='msm_q_lstm')
        self.p_lstm = SimpleLSTM(d[0], d[1], name='msm_p_lstm')

        # setup a function for computing reconstruction log likelihood
        if self.x_type == 'bernoulli':
            self.log_prob_func = lambda xo, xh: \
                    (constFX(-1.0) * log_prob_bernoulli(xo, xh))
        else:
            self.log_prob_func = lambda xo, xh: \
                    (constFX(-1.0) * log_prob_gaussian2(xo, xh, \
                     log_vars=self.bounded_logvar))

        ###################################################
        # Setup z and initialize s_mix, s0/c0 for enc/dec #
        ###################################################
        print("Building computation graph...")
        i_scale = 0.0
        if self.init_from_z: # initialize state from generative model
            i_scale = 1.0
        # sample the initial latent variables
        self.q_z_given_x = q_z_given_x
        x_input = self.x_in + self.b_input
        self.z_mean, self.z_logvar = self.q_z_given_x.apply(x_input, \
                do_samples=False)
        self.z = sample_transform(self.z_mean, self.z_logvar, self.z_zmuv)
        # initialize the encoder/decoder state
        _init_model = self.mlp_mix.apply(self.z)
        _init_const = (constFX(0.0) * _init_model) + self.mlp_mix.get_bias()
        _init_state = T.tanh( ((constFX(i_scale) * _init_model) + \
                (constFX(1.0 - i_scale) * _init_const)) )
        # get initial mixture, encoder, and decoder states
        md, rd = self.mix_dim, self.rnn_dim
        self.s_mix = _init_state[:,0:md]
        self.s0_enc = _init_state[:,md:(md + (1*rd))]
        self.c0_enc = _init_state[:,(md + (1*rd)):(md + (2*rd))]
        self.s0_dec = _init_state[:,(md + (2*rd)):(md + (3*rd))]
        self.c0_dec = _init_state[:,(md + (3*rd)):(md + (4*rd))]

        # initialize the "constructed observation" state
        self.s0_obs = T.zeros_like(self.x_in) + self.b_obs

        # gather KLd and NLL for the initialization step
        self.init_klds = gaussian_kld(self.z_mean, \
                self.z_logvar, 0.0, 0.0)
        self.init_nlls =  constFX(-1.0) * self.log_prob_func( \
                self.x_out, self.obs_transform(self.s0_obs))

        ##################################################
        # Setup the iterative generation loop using scan #
        ##################################################
        def ir_step_func(hi_zmuv, sim1_obs, sim1_dec, cim1_dec, sim1_enc, cim1_enc):
            ###############
            # basic stuff #
            ###############
            sim1_obs_trans = self.obs_transform(sim1_obs)
            x_hat = self.x_out - sim1_obs_trans
            ########################
            # encoder computations #
            ########################
            read_i = self.q_reader.apply(self.x_out, x_hat, sim1_dec)
            q_mlp_xi_input = T.concatenate([self.s_mix, read_i, sim1_dec], axis=1)
            xi_enc = self.q_mlp_xi.apply(q_mlp_xi_input)
            # update the encoder LSTM
            si_enc, ci_enc = self.q_lstm.apply(xi_enc, sim1_enc, cim1_enc)

            ##########################################
            # latent variable sampling and kld stuff #
            ##########################################
            # get conditional latent step distribution from guide policy
            hi_q_mean, hi_q_logvar = self.q_hi_given_si_enc.apply( \
                    si_enc, do_samples=False)
            hi_q = sample_transform(hi_q_mean, hi_q_logvar, hi_zmuv)
            # get conditional latent step distribution from generator policy
            hi_p_mean, hi_p_logvar = self.p_hi_given_sim1_dec.apply( \
                    sim1_dec, do_samples=False)
            hi_p = sample_transform(hi_p_mean, hi_p_logvar, hi_zmuv)
            # get latent step switchable between generator/guide
            hi = ((self.train_switch[0] * hi_q) + \
                    ((constFX(1.0) - self.train_switch[0]) * hi_p))

            ########################
            # decoder computations #
            ########################
            p_mlp_xi_input = T.concatenate([hi, self.s_mix], axis=1)
            xi_dec = self.p_mlp_xi.apply(p_mlp_xi_input)
            # update the decoder LSTM
            si_dec, ci_dec = self.p_lstm.apply(xi_dec, sim1_dec, cim1_dec)
            # compute writer output for canvas update, and then apply it
            si_step = self.p_writer.apply(si_dec)
            si_obs = sim1_obs + si_step

            #####################
            # cost computations #
            #####################
            nlli = self.log_prob_func(self.x_out, self.obs_transform(si_obs))
            kldi_cond = gaussian_kld(hi_q_mean, hi_q_logvar, \
                    hi_p_mean, hi_p_logvar)
            kldi_glob = gaussian_kld(hi_p_mean, hi_p_logvar, 0.0, 0.0)

            ###################
            # collect results #
            ###################
            results = [si_obs, si_dec, ci_dec, si_enc, ci_dec, \
                       nlli, kldi_cond, kldi_glob]
            return results

        init_values = [self.s0_obs, self.s0_dec, self.c0_dec, \
                       self.s0_enc, self.c0_dec, None, None, None]

        self.scan_results, self.scan_updates = theano.scan(ir_step_func, \
                outputs_info=init_values, sequences=self.hi_zmuv)

        self.si_obs = self.scan_results[0]
        self.si_dec = self.scan_results[1]
        self.ci_dec = self.scan_results[2]
        self.si_enc = self.scan_results[3]
        self.ci_enc = self.scan_results[4]
        self.nlli = self.scan_results[5]
        self.kldi_cond = self.scan_results[6]
        self.kldi_glob = self.scan_results[7]

        ######################################################################
        # ALL SYMBOLIC VARS NEEDED FOR THE OBJECTIVE SHOULD NOW BE AVAILABLE #
        ######################################################################

        # shared var learning rate for generator and inferencer
        zero_ary = to_fX(np.zeros((1,)))
        self.lr_1 = theano.shared(value=zero_ary, name='msm_lr_1')
        self.lr_2 = theano.shared(value=zero_ary, name='msm_lr_2')
        # shared var momentum parameters for generator and inferencer
        self.mom_1 = theano.shared(value=zero_ary, name='msm_mom_1')
        self.mom_2 = theano.shared(value=zero_ary, name='msm_mom_2')
        self.it_count = theano.shared(value=zero_ary, name='msm_it_count')
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

        # Grab all of the "optimizable" parameters in the "encoder"
        self.group_1_params = []
        self.group_1_params.extend(self.q_z_given_x.mlp_params)
        self.group_1_params.extend(self.q_hi_given_si_enc.mlp_params)
        self.group_1_params.extend(self.q_mlp_xi.mlp_params)
        self.group_1_params.extend(self.q_reader.mlp_params)
        self.group_1_params.extend(self.q_lstm.mlp_params)
        self.group_1_params.append(self.b_input)
        # Grab all of the "optimizable" parameters in the "decoder"
        self.group_2_params = []
        self.group_2_params.extend(self.p_hi_given_sim1_dec.mlp_params)
        self.group_2_params.extend(self.p_mlp_xi.mlp_params)
        self.group_2_params.extend(self.p_writer.mlp_params)
        self.group_2_params.extend(self.p_lstm.mlp_params)
        self.group_2_params.append(self.b_obs)
        # Make a joint list of parameters group 1/2
        self.joint_params = self.group_1_params + self.group_2_params

        ##########################################
        # CONSTRUCT ALL SYMBOLIC STEP-WISE COSTS #
        ##########################################
        print("Compiling cost computer...")
        self.compute_raw_costs = self._construct_raw_costs()

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
                beta1=self.mom_1, beta2=self.mom_2, it_count=self.it_count, \
                mom2_init=1e-3, smoothing=1e-8, max_grad_norm=10.0)
        self.group_2_updates = get_adam_updates(params=self.group_2_params, \
                grads=self.joint_grads, alpha=self.lr_2, \
                beta1=self.mom_1, beta2=self.mom_2, it_count=self.it_count, \
                mom2_init=1e-3, smoothing=1e-8, max_grad_norm=10.0)
        self.joint_updates = OrderedDict()
        for k in self.group_1_updates:
            self.joint_updates[k] = self.group_1_updates[k]
        for k in self.group_2_updates:
            self.joint_updates[k] = self.group_2_updates[k]
        # add scan updates, which seem to be required
        for k in self.scan_updates:
            self.joint_updates[k] = self.scan_updates[k]

        # Construct a function for jointly training the generator/inferencer
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

    def set_lam_kld(self, lam_kld_1=1.0, lam_kld_2=1.0):
        """
        Set the relative weight of prior KL-divergence vs. data likelihood.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_kld_1
        self.lam_kld_1.set_value(to_fX(new_lam))
        new_lam = zero_ary + lam_kld_2
        self.lam_kld_2.set_value(to_fX(new_lam))
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

    def set_kzg_weight(self, kzg_weight=0.2):
        """
        Set the weight for shaping penalty on conditional priors over zt.
        """
        assert(kzg_weight >= 0.0)
        zero_ary = np.zeros((1,))
        new_val = zero_ary + kzg_weight
        self.kzg_weight.set_value(to_fX(new_val))
        return

    def set_l1l2_weight(self, l1l2_weight=1.0):
        """
        Set the weight for shaping penalty on posterior KLds.
        """
        assert((l1l2_weight >= 0.0) and (l1l2_weight <= 1.0))
        zero_ary = np.zeros((1,))
        new_val = zero_ary + l1l2_weight
        self.l1l2_weight.set_value(to_fX(new_val))
        return

    def set_input_bias(self, new_bias=None):
        """
        Set the output layer bias.
        """
        new_bias = to_fX(new_bias)
        return

    def set_obs_bias(self, new_obs_bias=None):
        """
        Set initial bias on the obs part of state, but not the rnn part.
        """
        assert(new_obs_bias.shape[0] == self.obs_dim)
        new_bias = np.zeros((self.obs_dim,)) + new_obs_bias
        self.b_obs.set_value(to_fX(new_bias))
        return

    def _construct_zmuv_samples(self, X, br):
        """
        Construct the necessary (symbolic) samples for computing through this
        MultiStageDRAW for input (sybolic) matrix X.
        """
        z_zmuv = self.rng.normal( \
                size=(X.shape[0]*br, self.z_dim), \
                avg=0.0, std=1.0, dtype=theano.config.floatX)
        hi_zmuv = self.rng.normal( \
                size=(self.ir_steps, X.shape[0]*br, self.h_dim), \
                avg=0.0, std=1.0, dtype=theano.config.floatX)
        return z_zmuv, hi_zmuv

    def _construct_raw_costs(self):
        """
        Construct all the raw, i.e. not weighted by any lambdas, costs.
        """
        # gather step-wise costs into a single list (init costs at the end)
        all_step_costs = [self.nlli, self.kldi_cond, self.kldi_glob, \
                self.init_nlls, self.init_klds]
        # compile theano function for computing all relevant costs
        inputs = [self.x_in, self.x_out, self.z_zmuv, self.hi_zmuv]
        cost_func = theano.function(inputs=inputs, outputs=all_step_costs, updates=self.scan_updates)
        def raw_cost_computer(XI, XO):
            z_zmuv = to_fX( npr.randn(XI.shape[0], self.z_dim) )
            hi_zmuv = to_fX( npr.randn(self.ir_steps, XI.shape[0], self.h_dim) )
            s0_rnn = to_fX( np.zeros((XI.shape[0],self.z_dim)) )
            _all_costs = cost_func(XI, XO, z_zmuv, hi_zmuv)
            _init_nlls = _all_costs[-2]
            _init_klds = _all_costs[-1]
            _kld_cond = np.sum(np.mean(_all_costs[1], axis=1, keepdims=True), axis=0)
            _kld_glob = np.sum(np.mean(_all_costs[2], axis=1, keepdims=True), axis=0)
            _step_klds = np.mean(np.sum(_all_costs[1], axis=2, keepdims=True), axis=1)
            sk = [np.sum(np.mean(_init_klds, axis=0))]
            sk.extend([k for k in _step_klds])
            _step_klds = to_fX( np.asarray(sk) )
            _step_nlls = np.mean(_all_costs[0], axis=1)
            sn = [np.mean(_init_nlls, axis=0)]
            sn.extend([k for k in _step_nlls])
            _step_nlls = to_fX( np.asarray(sn) )
            results = [_init_nlls, _init_klds, _kld_cond, _kld_glob, \
                       _step_nlls, _step_klds]
            return results
        return raw_cost_computer

    def _construct_nll_costs(self):
        """
        Construct the negative log-likelihood part of free energy.
        """
        nll_costs = self.nlli[-1]
        return nll_costs

    def _construct_kld_costs(self):
        """
        Construct the posterior KL-divergence part of cost to minimize.
        """
        kld_hi_cond_l1 = self.kldi_cond
        kld_hi_cond_l2 = self.kldi_cond**2.0
        kld_hi_glob_l1 = self.kldi_glob
        kld_hi_glob_l2 = self.kldi_glob**2.0
        kld_hi_cond_all = (self.l1l2_weight[0] * kld_hi_cond_l1) + \
                    ((constFX(1.0) - self.l1l2_weight[0]) * kld_hi_cond_l2)
        kld_hi_glob_all = (self.l1l2_weight[0] * kld_hi_glob_l1) + \
                    ((constFX(1.0) - self.l1l2_weight[0]) * kld_hi_glob_l2)
        kld_hi_cond = T.sum(T.sum(kld_hi_cond_all, axis=0), axis=1)
        kld_hi_glob = T.sum(T.sum(kld_hi_glob_all, axis=0), axis=1)
        # construct KLd cost for the distributions over z
        kld_z_all = self.init_klds
        kld_z_l1l2 = (self.l1l2_weight[0] * kld_z_all) + \
                ((constFX(1.0) - self.l1l2_weight[0]) * kld_z_all**2.0)
        kld_z = T.sum(kld_z_l1l2, axis=1)
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
        br = T.lscalar()
        zzmuv, hizmuv = self._construct_zmuv_samples(xi, br)
        # collect the outputs to return from this function
        outputs = [self.joint_cost, self.nll_cost, self.kld_cost, \
                self.reg_cost]
        # compile the theano function
        func = theano.function(inputs=[ xi, xo, br ], \
                outputs=outputs, \
                givens={self.x_in: xi.repeat(br, axis=0), \
                        self.x_out: xo.repeat(br, axis=0), \
                        self.z_zmuv: zzmuv, \
                        self.hi_zmuv: hizmuv }, \
                updates=self.joint_updates)
        return func

    def _construct_compute_fe_terms(self):
        """
        Construct a function for computing terms in variational free energy.
        """
        # setup some symbolic variables for theano to deal with
        xi = T.matrix()
        xo = T.matrix()
        zzmuv, hizmuv = self._construct_zmuv_samples(xi, 1)
        # construct values to output
        nll = self._construct_nll_costs()
        kld = self.kld_z + self.kld_hi_cond
        # compile theano function for a one-sample free-energy estimate
        fe_term_sample = theano.function(inputs=[ xi, xo ], \
                outputs=[nll, kld], \
                givens={self.x_in: xi, \
                        self.x_out: xo, \
                        self.z_zmuv: zzmuv, \
                        self.hi_zmuv: hizmuv}, \
                updates=self.scan_updates)
        # construct a wrapper function for multi-sample free-energy estimate
        def fe_term_estimator(XI, XO, sample_count):
            # set values of some regularization parameters to the values that
            # produce the variational free energy bound.
            old_lam_nll = self.lam_nll.get_value(borrow=False)
            old_lam_kld_1 = self.lam_kld_1.get_value(borrow=False)
            old_lam_kld_2 = self.lam_kld_2.get_value(borrow=False)
            old_l1l2_weight = self.l1l2_weight.get_value(borrow=False)
            vfe_lam_nll = (0.0 * old_lam_nll) + 1.0
            vfe_lam_kld_1 = (0.0 * old_lam_kld_1) + 1.0
            vfe_lam_kld_2 = (0.0 * old_lam_kld_2) + 1.0
            vfe_l1l2_weight = (0.0 * old_l1l2_weight) + 1.0
            self.lam_nll.set_value(vfe_lam_nll)
            self.lam_kld_1.set_value(vfe_lam_kld_1)
            self.lam_kld_2.set_value(vfe_lam_kld_2)
            self.l1l2_weight.set_value(vfe_l1l2_weight)
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
            self.l1l2_weight.set_value(old_l1l2_weight)
            return [mean_nll, mean_kld]
        return fe_term_estimator

    def _construct_sample_from_prior(self):
        """
        Construct a function for drawing independent samples from the
        distribution generated by this MultiStageDRAW. This function returns
        the full sequence of "partially completed" examples.
        """
        z_sym = T.matrix()
        x_sym = T.matrix()
        zzmuv, hizmuv = self._construct_zmuv_samples(z_sym, 1)
        oputs = [self.s0_obs]
        for i in range(self.ir_steps):
            oputs.append(self.obs_transform(self.si_obs[i]))
        sample_func = theano.function(inputs=[z_sym, x_sym], outputs=oputs, \
                givens={self.z: z_sym, \
                        self.x_in: T.zeros_like(x_sym), \
                        self.x_out: T.zeros_like(x_sym), \
                        self.z_zmuv: zzmuv, \
                        self.hi_zmuv: hizmuv}, \
                updates=self.scan_updates )
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
        generated by this MultiStageDRAW, conditioned on some inputs to the
        initial encoder stage (i.e. self.q_z_given_x). This returns the full 
        sequence of "partially completed" examples.

        The 
        """
        xi = T.matrix()
        xo = T.matrix()
        zzmuv, hizmuv = self._construct_zmuv_samples(xi, 1)
        oputs = [self.s0_obs]
        for i in range(self.ir_steps):
            oputs.append(self.obs_transform(self.si_obs[i]))
        sample_func = theano.function(inputs=[xi, xo], outputs=oputs, \
                givens={self.x_in: xi, \
                        self.x_out: xo, \
                        self.z_zmuv: zzmuv, \
                        self.hi_zmuv: hizmuv},  \
                updates=self.scan_updates)
        def conditional_sampler(XI, XO=None, guided_decoding=False):
            XI = to_fX(XI)
            if XO is None:
                XO = XI
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


if __name__=="__main__":
    print("Hello world!")







##############
# EYE BUFFER #
##############
