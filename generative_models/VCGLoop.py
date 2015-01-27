################################################################################
# Code for managing and training a Variational Collaborative Generative Loop.  #
#                                                                              #
# Note: This is ongoing research and very much in flux.                        #
################################################################################

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
from NetLayers import HiddenLayer, DiscLayer, safe_log, softplus_actfun
from DKCode import get_adam_updates, get_adadelta_updates
from GIPair import GIPair

#################
# FOR PROFILING #
#################
#from theano import ProfileMode
#profmode = theano.ProfileMode(optimizer='fast_run', linker=theano.gof.OpWiseCLinker())

#############################
# SOME HANDY LOSS FUNCTIONS #
#############################

def logreg_loss(Y, class_sign):
    """
    Simple binomial deviance (i.e. logistic regression) loss.

    This assumes that all predictions in Y have the same target class, which
    is indicated by class_sign, which should be in {-1, +1}. Note: this does
    not "normalize" for the number of predictions in Y.
    """
    loss = T.sum(softplus_actfun(-class_sign * Y))
    return loss

def ns_nce_pos(f, k=1.0):
    """
    Negative-sampling noise contrastive estimation, for target distribution.
    """
    loss = T.sum(T.log(1.0 + k*T.exp(-f)))
    return loss

def ns_nce_neg(f, k=1.0):
    """
    Negative-sampling noise contrastive estimation, for base distribution.
    """
    loss = T.sum(f + T.log(1.0 + k*T.exp(-f)))
    return loss

def lsq_loss(Yh, Yt=0.0):
    """
    Least-squares loss for predictions in Yh, given target Yt.
    """
    loss = T.sum((Yh - Yt)**2.0)
    return loss

def hinge_loss(Yh, Yt=0.0):
    """
    Unilateral hinge loss for Yh, given target Yt.
    """
    residual = Yt - Yh
    loss = T.sum((residual * (residual > 0.0)))
    return loss

def ulh_loss(Yh, Yt=0.0, delta=0.5):
    """
    Unilateral Huberized least-squares loss for Yh, given target Yt.
    """
    residual = Yt - Yh
    quad_loss = residual**2.0
    line_loss = (2.0 * delta * abs(residual)) - delta**2.0
    # Construct mask for quadratic loss region
    quad_mask = (abs(residual) < delta) * (residual > 0.0)
    # Construct mask for linear loss region
    line_mask = (abs(residual) >= delta) * (residual > 0.0)
    # Combine the quadratic and linear losses
    loss = T.sum((quad_loss * quad_mask) + (line_loss * line_mask))
    return loss

def cat_entropy(p):
    """
    Compute the entropy of (row-wise) categorical distributions in p.
    """
    row_ents = -T.sum((p * safe_log(p)), axis=1, keepdims=True)
    return row_ents

def cat_prior_dir(p, alpha=0.1):
    """
    Log probability under a dirichlet prior, with dirichlet parameter alpha.
    """
    log_prob = T.sum((1.0 - alpha) * safe_log(p))
    return log_prob

def cat_prior_ent(p, ent_weight=1.0):
    """
    Log probability under an "entropy-type" prior, with some "weight".
    """
    log_prob = -cat_entropy * ent_weight
    return log_prob

def binarize_data(X):
    """
    Make a sample of bernoulli variables with probabilities given by X.
    """
    X_shape = X.shape
    probs = npr.rand(*X_shape)
    X_binary = 1.0 * (probs < X)
    return X_binary.astype(theano.config.floatX)

def sample_masks(X, drop_prob=0.3):
    """
    Sample a binary mask to apply to the matrix X, with rate mask_prob.
    """
    probs = npr.rand(*X.shape)
    mask = 1.0 * (probs > drop_prob)
    return mask.astype(theano.config.floatX)

def sample_patch_masks(X, im_shape, patch_shape):
    """
    Sample a random patch mask for each image in X.
    """
    obs_count = X.shape[0]
    rs = patch_shape[0]
    cs = patch_shape[1]
    off_row = npr.randint(1,high=(im_shape[0]-rs-1), size=(obs_count,))
    off_col = npr.randint(1,high=(im_shape[1]-cs-1), size=(obs_count,))
    dummy = np.zeros(im_shape)
    mask = np.zeros(X.shape)
    for i in range(obs_count):
        dummy = (0.0 * dummy) + 1.0
        dummy[off_row[i]:(off_row[i]+rs), off_col[i]:(off_col[i]+cs)] = 0.0
        mask[i,:] = dummy.ravel()
    return mask.astype(theano.config.floatX)

class VCGLoop(object):
    """
    Controller for training a self-looping VAE using guidance provided by a
    classifier. The classifier tries to discriminate between samples generated
    by the looped VAE while the VAE minimizes a variational generative model
    objective and also shifts mass away from regions where the classifier can
    discern that the generated data is denser than the training data.

    This class can also train "policies" for reconstructing partially masked
    inputs. A reconstruction policy can readily be trained to share the same
    parameters as a policy for generating transitions while self-looping.

    The generator must be an instance of the GenNet class implemented in
    "GenNet.py". The discriminator must be an instance of the PeaNet class,
    as implemented in "PeaNet.py". The inferencer must be an instance of the
    InfNet class implemented in "InfNet.py".

    Parameters:
        rng: numpy.random.RandomState (for reproducibility)
        Xd: symbolic var for providing points for starting the Markov Chain
        Xt: symbolic var for providing samples from the target distribution
        i_net: The InfNet instance that will serve as the inferencer
        g_net: The GenNet instance that will serve as the generator
        d_net: The PeaNet instance that will serve as the discriminator
        chain_len: number of steps to unroll the VAE Markov Chain
        data_dim: dimension of the generated data
        prior_dim: dimension of the model prior
        params: a dict of parameters for controlling various costs
            lam_l2d: regularization on squared discriminator output
    """
    def __init__(self, rng=None, Xd=None, Xc=None, Xm=None, Xt=None, \
                 i_net=None, g_net=None, d_net=None, chain_len=None, \
                 data_dim=None, prior_dim=None, params=None):
        # Do some stuff!
        self.rng = RandStream(rng.randint(100000))
        self.data_dim = data_dim
        self.prior_dim = prior_dim

        # symbolic var for inputting samples for initializing the VAE chain
        self.Xd = Xd
        # symbolic var for masking subsets of the state variables
        self.Xm = Xm
        # symbolic var for controlling subsets of the state variables
        self.Xc = Xc
        # symbolic var for inputting samples from the target distribution
        self.Xt = Xt
        # integer number of times to cycle the VAE loop
        self.chain_len = chain_len
        # symbolic matrix of indices for data inputs
        self.It = T.arange(self.Xt.shape[0])
        # symbolic matrix of indices for noise/generated inputs
        self.Id = T.arange(self.chain_len * self.Xd.shape[0]) + self.Xt.shape[0]

        # get a clone of the desired VAE, for easy access
        self.GIP = GIPair(rng=rng, Xd=self.Xd, Xc=self.Xc, Xm=self.Xm, \
                g_net=g_net, i_net=i_net, data_dim=self.data_dim, \
                prior_dim=self.prior_dim, params=None, shared_param_dicts=None)
        self.IN = self.GIP.IN
        self.GN = self.GIP.GN
        self.use_encoder = self.IN.use_encoder
        assert(self.use_encoder == self.GN.use_decoder)
        # self-loop some clones of the main VAE into a chain.
        # ** All VAEs in the chain share the same Xc and Xm, which are the
        #    symbolic inputs for providing the observed portion of the input
        #    and a mask indicating which part of the input is "observed".
        #    These inputs are used for training "reconstruction" policies.
        self.IN_chain = []
        self.GN_chain = []
        self.Xg_chain = []
        _Xd = self.Xd
        for i in range(self.chain_len):
            if (i == 0):
                # start the chain with data provided by user
                _IN = self.IN.shared_param_clone(rng=rng, Xd=_Xd, \
                        Xc=self.Xc, Xm=self.Xm)
                _GN = self.GN.shared_param_clone(rng=rng, Xp=_IN.output)
            else:
                # continue the chain with samples from previous VAE
                _IN = self.IN.shared_param_clone(rng=rng, Xd=_Xd, \
                        Xc=self.Xc, Xm=self.Xm)
                _GN = self.GN.shared_param_clone(rng=rng, Xp=_IN.output)
            if self.use_encoder:
                # use the "decoded" output of the previous generator as input
                # to the next inferencer, which will re-encode it prior to
                # inference
                _Xd = _GN.output_decoded
            else:
                # use the "encoded" output of the previous generator as input
                # to the next inferencer, as the inferencer won't try to 
                # re-encode it prior to inference
                _Xd = _GN.output
            self.IN_chain.append(_IN)
            self.GN_chain.append(_GN)
            self.Xg_chain.append(_Xd)

        # make a clone of the desired discriminator network, which will try
        # to discriminate between samples from the training data and samples
        # generated by the self-looped VAE chain.
        self.DN = d_net.shared_param_clone(rng=rng, \
                Xd=T.vertical_stack(self.Xt, *self.Xg_chain))

        zero_ary = np.zeros((1,)).astype(theano.config.floatX)
        # init shared var for weighting nll of data given posterior sample
        self.lam_chain_nll = theano.shared(value=zero_ary, name='vcg_lam_chain_nll')
        self.set_lam_chain_nll(lam_chain_nll=1.0)
        # init shared var for weighting posterior KL-div from prior
        self.lam_chain_kld = theano.shared(value=zero_ary, name='vcg_lam_chain_kld')
        self.set_lam_chain_kld(lam_chain_kld=1.0)
        # init shared var for weighting chain diffusion rate (a.k.a. velocity)
        self.lam_chain_vel = theano.shared(value=zero_ary, name='vcg_lam_chain_vel')
        self.set_lam_chain_vel(lam_chain_vel=1.0)
        # init shared var for weighting nll of data given posterior sample
        self.lam_mask_nll = theano.shared(value=zero_ary, name='vcg_lam_mask_nll')
        self.set_lam_mask_nll(lam_mask_nll=0.0)
        # init shared var for weighting posterior KL-div from prior
        self.lam_mask_kld = theano.shared(value=zero_ary, name='vcg_lam_mask_kld')
        self.set_lam_mask_kld(lam_mask_kld=0.0)
        # init shared var for controlling l2 regularization on params
        self.lam_l2w = theano.shared(value=zero_ary, name='vcg_lam_l2w')
        self.set_lam_l2w(lam_l2w=1e-4)
        # shared var learning rates for all networks
        self.lr_dn = theano.shared(value=zero_ary, name='vcg_lr_dn')
        self.lr_gn = theano.shared(value=zero_ary, name='vcg_lr_gn')
        self.lr_in = theano.shared(value=zero_ary, name='vcg_lr_in')
        # shared var momentum parameters for all networks
        self.mom_1 = theano.shared(value=zero_ary, name='vcg_mom_1')
        self.mom_2 = theano.shared(value=zero_ary, name='vcg_mom_2')
        self.it_count = theano.shared(value=zero_ary, name='vcg_it_count')
        # shared var weights for adversarial classification objective
        self.dw_dn = theano.shared(value=zero_ary, name='vcg_dw_dn')
        self.dw_gn = theano.shared(value=zero_ary, name='vcg_dw_gn')
        # init parameters for controlling learning dynamics
        self.set_all_sgd_params()
        
        self.set_disc_weights()  # init adversarial cost weights for GN/DN
        # set a shared var for regularizing the output of the discriminator
        self.lam_l2d = theano.shared(value=(zero_ary + params['lam_l2d']), \
                name='vcg_lam_l2d')

        # setup weights for weighting the quality of the reconstruction
        # differently over multiple steps of reconstruction.
        nll_weights = np.linspace(0.0, 5.0, num=self.chain_len)
        nll_weights = nll_weights / np.sum(nll_weights)
        nll_weights = nll_weights.astype(theano.config.floatX)
        self.mask_nll_weights = theano.shared(value=nll_weights, \
                name='vcg_mask_nll_weights')

        # Grab the full set of "optimizable" parameters from the generator
        # and discriminator networks that we'll be working with. We need to
        # ignore parameters in the final layers of the proto-networks in the
        # discriminator network (a generalized pseudo-ensemble). We ignore them
        # because the VCGair requires that they be "bypassed" in favor of some
        # binary classification layers that will be managed by this VCGair.
        self.dn_params = []
        for pn in self.DN.proto_nets:
            for pnl in pn[0:-1]:
                self.dn_params.extend(pnl.params)
        self.in_params = [p for p in self.IN.mlp_params]
        self.gn_params = [p for p in self.GN.mlp_params]
        self.joint_params = self.dn_params + self.in_params + self.gn_params

        # Now construct a binary discriminator layer for each proto-net in the
        # discriminator network. And, add their params to optimization list.
        self._construct_disc_layers(rng)
        self.disc_reg_cost = self.lam_l2d[0] * \
                T.sum([dl.act_l2_sum for dl in self.disc_layers])

        # Construct costs for the generator and discriminator networks based 
        # on adversarial binary classification
        self.disc_cost_dn, self.disc_cost_gn = self._construct_disc_costs()

        # first, build the cost to be optimized by the discriminator network,
        # in general this will be treated somewhat indepedently of the
        # optimization of the generator and inferencer networks.
        self.dn_cost = self.disc_cost_dn + self.DN.act_reg_cost + \
                self.disc_reg_cost

        # construct costs relevant to the optimization of the generator and
        # discriminator networks
        self.chain_nll_cost = self.lam_chain_nll[0] * \
                self._construct_chain_nll_cost(data_weight=0.2)
        self.chain_kld_cost = self.lam_chain_kld[0] * \
                self._construct_chain_kld_cost(data_weight=0.2)
        self.chain_vel_cost = self.lam_chain_vel[0] * \
                self._construct_chain_vel_cost()
        self.mask_nll_cost = self.lam_mask_nll[0] * \
                self._construct_mask_nll_cost()
        self.mask_kld_cost = self.lam_mask_kld[0] * \
                self._construct_mask_kld_cost()
        self.other_reg_cost = self._construct_other_reg_cost()
        self.gip_cost = self.disc_cost_gn + self.chain_nll_cost + \
                self.chain_kld_cost + self.chain_vel_cost + \
                self.mask_nll_cost + self.mask_kld_cost + \
                self.other_reg_cost
        # compute total cost on the discriminator and VB generator/inferencer
        self.joint_cost = self.dn_cost + self.gip_cost

        # grab the gradients for all parameters to optimize
        self.joint_grads = OrderedDict()
        for p in self.dn_params:
            # grads for discriminator network params use a separate cost
            self.joint_grads[p] = T.grad(self.dn_cost, p).clip(-0.1,0.1)
        for p in self.in_params:
            # grads for generator network use the joint cost
            self.joint_grads[p] = T.grad(self.joint_cost, p).clip(-0.1,0.1)
        for p in self.gn_params:
            # grads for generator network use the joint cost
            self.joint_grads[p] = T.grad(self.joint_cost, p).clip(-0.1,0.1)

        # construct the updates for the discriminator, generator and 
        # inferencer networks. all networks share the same first/second
        # moment momentum and iteration count. the networks each have their
        # own learning rates, which lets you turn their learning on/off.
        self.dn_updates = get_adam_updates(params=self.dn_params, \
                grads=self.joint_grads, alpha=self.lr_dn, \
                beta1=self.mom_1, beta2=self.mom_2, it_count=self.it_count, \
                mom2_init=1e-3, smoothing=1e-8)
        self.gn_updates = get_adam_updates(params=self.gn_params, \
                grads=self.joint_grads, alpha=self.lr_gn, \
                beta1=self.mom_1, beta2=self.mom_2, it_count=self.it_count, \
                mom2_init=1e-3, smoothing=1e-8)
        self.in_updates = get_adam_updates(params=self.in_params, \
                grads=self.joint_grads, alpha=self.lr_in, \
                beta1=self.mom_1, beta2=self.mom_2, it_count=self.it_count, \
                mom2_init=1e-3, smoothing=1e-8)
        #self.dn_updates = get_adadelta_updates(params=self.dn_params, \
        #        grads=self.joint_grads, alpha=self.lr_dn, beta1=0.98)
        #self.gn_updates = get_adadelta_updates(params=self.gn_params, \
        #        grads=self.joint_grads, alpha=self.lr_gn, beta1=0.98)
        #self.in_updates = get_adadelta_updates(params=self.in_params, \
        #        grads=self.joint_grads, alpha=self.lr_in, beta1=0.98)

        # bag up all the updates required for training
        self.joint_updates = OrderedDict()
        for k in self.dn_updates:
            self.joint_updates[k] = self.dn_updates[k]
        for k in self.gn_updates:
            self.joint_updates[k] = self.gn_updates[k]
        for k in self.in_updates:
            self.joint_updates[k] = self.in_updates[k]

        # construct the function for training on training data
        self.train_joint = self._construct_train_joint()

        # construct a function for computing the ouputs of the generator
        # network for a batch of noise. Presumably, the noise will be drawn
        # from the same distribution that was used in training....
        self.sample_chain_from_data = self.GIP.sample_gil_from_data
        return

    def set_dn_sgd_params(self, learn_rate=0.01):
        """
        Set learning rate for the discriminator network.
        """
        zero_ary = np.zeros((1,))
        new_lr = zero_ary + learn_rate
        self.lr_dn.set_value(new_lr.astype(theano.config.floatX))
        return

    def set_in_sgd_params(self, learn_rate=0.01):
        """
        Set learning rate for the inferencer network.
        """
        zero_ary = np.zeros((1,))
        new_lr = zero_ary + learn_rate
        self.lr_in.set_value(new_lr.astype(theano.config.floatX))
        return

    def set_gn_sgd_params(self, learn_rate=0.01):
        """
        Set learning rate for the generator network.
        """
        zero_ary = np.zeros((1,))
        new_lr = zero_ary + learn_rate
        self.lr_gn.set_value(new_lr.astype(theano.config.floatX))
        return

    def set_all_sgd_params(self, learn_rate=0.01, mom_1=0.9, mom_2=0.999):
        """
        Set learning rate and momentum parameter for all updates.
        """
        zero_ary = np.zeros((1,))
        # set learning rates to the same value
        new_lr = zero_ary + learn_rate
        self.lr_dn.set_value(new_lr.astype(theano.config.floatX))
        self.lr_gn.set_value(new_lr.astype(theano.config.floatX))
        self.lr_in.set_value(new_lr.astype(theano.config.floatX))
        # set the first/second moment momentum parameters
        new_mom_1 = zero_ary + mom_1
        new_mom_2 = zero_ary + mom_2
        self.mom_1.set_value(new_mom_1.astype(theano.config.floatX))
        self.mom_2.set_value(new_mom_2.astype(theano.config.floatX))
        return

    def set_disc_weights(self, dweight_gn=1.0, dweight_dn=1.0):
        """
        Set weights for the adversarial classification cost.
        """
        zero_ary = np.zeros((1,)).astype(theano.config.floatX)
        new_dw_dn = zero_ary + dweight_dn
        self.dw_dn.set_value(new_dw_dn)
        new_dw_gn = zero_ary + dweight_gn
        self.dw_gn.set_value(new_dw_gn)
        return

    def set_lam_chain_nll(self, lam_chain_nll=1.0):
        """
        Set weight for controlling the influence of the data likelihood.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_chain_nll
        self.lam_chain_nll.set_value(new_lam.astype(theano.config.floatX))
        return

    def set_lam_chain_kld(self, lam_chain_kld=1.0):
        """
        Set the strength of regularization on KL-divergence for continuous
        posterior variables. When set to 1.0, this reproduces the standard
        role of KL(posterior || prior) in variational learning.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_chain_kld
        self.lam_chain_kld.set_value(new_lam.astype(theano.config.floatX))
        return

    def set_lam_chain_vel(self, lam_chain_vel=1.0):
        """
        Set the strength of regularization on Markov Chain velocity.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_chain_vel
        self.lam_chain_vel.set_value(new_lam.astype(theano.config.floatX))
        return

    def set_lam_mask_nll(self, lam_mask_nll=0.0):
        """
        Set weight for controlling the influence of the data likelihood.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_mask_nll
        self.lam_mask_nll.set_value(new_lam.astype(theano.config.floatX))
        return

    def set_lam_mask_kld(self, lam_mask_kld=1.0):
        """
        Set the strength of regularization on KL-divergence for continuous
        posterior variables. When set to 1.0, this reproduces the standard
        role of KL(posterior || prior) in variational learning.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_mask_kld
        self.lam_mask_kld.set_value(new_lam.astype(theano.config.floatX))
        return

    def set_lam_l2w(self, lam_l2w=1e-3):
        """
        Set the relative strength of l2 regularization on network params.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_l2w
        self.lam_l2w.set_value(new_lam.astype(theano.config.floatX))
        return

    def _construct_disc_layers(self, rng):
        """
        Construct binary discrimination layers for each spawn-net in the
        underlying discrimnator pseudo-ensemble. All spawn-nets spawned from
        the same proto-net will use the same disc-layer parameters.
        """
        self.disc_layers = []
        self.disc_outputs = []
        dn_init_scale = self.DN.init_scale
        for sn in self.DN.spawn_nets:
            # construct a "binary discriminator" layer to sit on top of each
            # spawn net in the discriminator pseudo-ensemble
            sn_fl = sn[-1]
            init_scale = dn_init_scale * (1. / np.sqrt(sn_fl.in_dim))
            self.disc_layers.append(DiscLayer(rng=rng, \
                    input=sn_fl.noisy_input, in_dim=sn_fl.in_dim, \
                    W_scale=dn_init_scale))
            # capture the (linear) output of the DiscLayer, for possible reuse
            self.disc_outputs.append(self.disc_layers[-1].linear_output)
            # get the params of this DiscLayer, for convenient optimization
            self.dn_params.extend(self.disc_layers[-1].params)
        return

    def _construct_disc_costs(self):
        """
        Construct the generator and discriminator adversarial costs.
        """
        gn_costs = []
        dn_costs = []
        for dl_output in self.disc_outputs:
            data_preds = dl_output.take(self.It, axis=0)
            noise_preds = dl_output.take(self.Id, axis=0)
            # compute the cost with respect to which we will be optimizing
            # the parameters of the discriminator network
            data_size = T.cast(self.It.size, 'floatX')
            noise_size = T.cast(self.Id.size, 'floatX')
            dnl_dn_cost = (logreg_loss(data_preds, 1.0) / data_size) + \
                          (logreg_loss(noise_preds, -1.0) / noise_size)
            # compute the cost with respect to which we will be optimizing
            # the parameters of the generative model
            dnl_gn_cost = ulh_loss(noise_preds, 0.0) / noise_size
            dn_costs.append(dnl_dn_cost)
            gn_costs.append(dnl_gn_cost)
        dn_cost = self.dw_dn[0] * T.sum(dn_costs)
        gn_cost = self.dw_gn[0] * T.sum(gn_costs)
        return [dn_cost, gn_cost]

    def _construct_chain_nll_cost(self, data_weight=0.2):
        """
        Construct the negative log-likelihood part of cost to minimize.

        This is for operation in "free chain" mode, where a seed point is used
        to initialize a long(ish) running markov chain.
        """
        assert((data_weight > 0.0) and (data_weight < 1.0))
        obs_count = T.cast(self.Xd.shape[0], 'floatX')
        nll_costs = []
        step_weight = 1.0
        step_weights = []
        step_decay = data_weight
        for i in range(self.chain_len):
            IN_i = self.IN_chain[i]
            GN_i = self.GN_chain[i]
            if self.use_encoder:
                # compare encoded output of the generator with the encoded
                # non-control input to the inferencer
                c = -T.sum(GN_i.compute_log_prob(Xd=IN_i.Xd_encoded)) / obs_count
            else:
                # compare encoded output of the generator with the unencoded
                # non-control input to the inferencer
                c = -T.sum(GN_i.compute_log_prob(Xd=IN_i.Xd)) / obs_count
            nll_costs.append(step_weight * c)
            step_weights.append(step_weight)
            step_weight = step_weight * step_decay
        nll_cost = sum(nll_costs) / sum(step_weights)
        return nll_cost

    def _construct_chain_kld_cost(self, data_weight=0.2):
        """
        Construct the posterior KL-d from prior part of cost to minimize.

        This is for operation in "free chain" mode, where a seed point is used
        to initialize a long(ish) running markov chain.
        """
        assert((data_weight > 0.0) and (data_weight < 1.0))
        obs_count = T.cast(self.Xd.shape[0], 'floatX')
        kld_costs = []
        step_weight = 1.0
        step_weights = []
        step_decay = data_weight
        for i in range(self.chain_len):
            IN_i = self.IN_chain[i]
            c = T.sum(IN_i.kld_cost) / obs_count
            kld_costs.append(step_weight * c)
            step_weights.append(step_weight)
            step_weight = step_weight * step_decay
        kld_cost = sum(kld_costs) / sum(step_weights)
        return kld_cost

    def _construct_chain_vel_cost(self):
        """
        Construct the Markov Chain velocity part of cost to minimize.

        This is for operation in "free chain" mode, where a seed point is used
        to initialize a long(ish) running markov chain.
        """
        obs_count = T.cast(self.Xd.shape[0], 'floatX')
        IN_start = self.IN_chain[0]
        GN_end = self.GN_chain[-1]
        vel_cost = T.sum(GN_end.compute_log_prob(Xd=IN_start.Xd)) / obs_count
        return vel_cost

    def _construct_mask_nll_cost(self):
        """
        Construct the negative log-likelihood part of cost to minimize.

        This is for "iterative reconstruction" when the seed input is subject
        to partial masking.
        """
        obs_count = T.cast(self.Xd.shape[0], 'floatX')
        nll_costs = []
        for i in range(self.chain_len):
            IN_i = self.IN_chain[i]
            GN_i = self.GN_chain[i]
            if self.use_encoder:
                # compare encoded output of the generator to the encoded
                # representation of control input to the inferencer
                c = -T.sum(GN_i.compute_log_prob(Xd=IN_i.Xc_encoded)) / obs_count
            else:
                # compare encoded output of the generator to the unencoded
                # control input to the inferencer, but only measure NLL for
                # input dimensions that are not part of the "control set"
                c = -T.sum(GN_i.masked_log_prob(Xc=self.Xc, Xm=self.Xm)) \
                        / obs_count
            nll_costs.append(self.mask_nll_weights[i] * c)
        nll_cost = sum(nll_costs)
        return nll_cost

    def _construct_mask_kld_cost(self):
        """
        Construct the posterior KL-d from prior part of cost to minimize.

        This is for "iterative reconstruction" when the seed input is subject
        to partial masking.
        """
        obs_count = T.cast(self.Xd.shape[0], 'floatX')
        kld_costs = []
        for i in range(self.chain_len):
            IN_i = self.IN_chain[i]
            c = T.sum(IN_i.kld_cost) / obs_count
            kld_costs.append(c)
        kld_cost = sum(kld_costs) / float(self.chain_len)
        return kld_cost

    def _construct_other_reg_cost(self):
        """
        Construct the cost for low-level basic regularization. E.g. for
        applying l2 regularization to the network activations and parameters.
        """
        gp_cost = sum([T.sum(par**2.0) for par in self.gn_params])
        ip_cost = sum([T.sum(par**2.0) for par in self.in_params])
        other_reg_cost = self.lam_l2w[0] * (gp_cost + ip_cost)
        return other_reg_cost

    def _construct_train_joint(self):
        """
        Construct theano function to train generator and discriminator jointly.
        """
        outputs = [self.joint_cost, self.chain_nll_cost, self.chain_kld_cost, \
                self.chain_vel_cost, self.mask_nll_cost, self.mask_kld_cost, \
                self.disc_cost_gn, self.disc_cost_dn, self.other_reg_cost]
        func = theano.function(inputs=[ self.Xd, self.Xc, self.Xm, self.Xt ], \
                outputs=outputs, updates=self.joint_updates) # , \
                #mode=profmode)
        return func

    def sample_from_prior(self, samp_count):
        """
        Draw independent samples from the model's prior, using the gaussian
        continuous prior of the underlying GenNet.
        """
        Zs = self.GN.sample_from_prior(samp_count).astype(theano.config.floatX)
        Xs = self.GN.transform_prior(Zs)
        return Xs

if __name__=="__main__":
    # TEST CODE IS ELSEWHERE
    print("NO TEST CODE HERE!")





##############
# EYE BUFFER #
##############
