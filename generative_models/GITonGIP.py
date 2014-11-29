################################################################################
# Code for managing and double stack of variational auto-encoders:             #
#                                                                              #
# 1. The lower VAE in the stack is a "basic" deep VAE.                         #
# 2. The upper VAE in the stack operates on the latent posterior distributions #
#    of the lower VAE for points in the observation space. The upper VAE has   #
#    a mixed categorical/continuous posterior, for semi-supervisedness.        #
################################################################################

# basic python
import numpy as np
import numpy.random as npr
from collections import OrderedDict

# theano business
import theano
import theano.tensor as T
import theano.printing as printing
from theano.ifelse import ifelse
#from theano.tensor.shared_randomstreams import RandomStreams as RandStream
from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams as RandStream

# phil's sweetness
from NetLayers import relu_actfun, softplus_actfun, \
                      safe_softmax, safe_log
from LogPDFs import log_prob_bernoulli, log_prob_gaussian
from GenNet import GenNet
from InfNet import InfNet
from PeaNet import PeaNet

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


#
#
# Important symbolic variables:
#   Xd: Xd represents input at the "data variables" of the inferencer
#   Yd: Yd represents input at the "label variables" of the inferencer
#   Xc: Xc represents input at the "control variables" of the inferencer
#   Xm: Xm represents input at the "mask variables" of the inferencer
#
#

class GITonGIP(object):
    """
    Controller for training a double stack of variational autoencoders.

    The lower VAE in the stack needs to be a GIPair instance, as implemented in
    GIPair.py. The upper VAE in the stack needs to be a GITrip instance, as
    implemented in GITrip.py.

    Parameters:
        rng: numpy.random.RandomState (for reproducibility)
        Xd: symbolic "data" input to this VAE
        Yd: symbolic "label" input to this VAE
        Xc: symbolic "control" input to this VAE
        Xm: symbolic "mask" input to this VAE
        gip_vae: The GIPair to use as the "lower" VAE
        git_vae: The GITrip to use as the "upper" VAE
        data_dim: dimension of the "observable data" variables
        prior_1_dim: dimension of the continuous prior of the lower VAE
        prior_2_dim: dimension of the continuous prior of the upper VAE
        label_dim: cardinality of the categorical latent variable
        batch_size: fixed size of minibatches to be used during training. you
                    have to stick to this value while training. this is to work
                    around theano problems. this only matters for computing the
                    cost function, and doesn't restrict sampling.
        params: dict for passing additional parameters
        shared_param_dicts: dict for retrieving some shared parameters required
                            by a GIPair. if this parameter is passed, then this
                            GIPair will be initialized as a "shared-parameter"
                            clone of some other GIPair.
    """
    def __init__(self, rng=None, \
            Xd=None, Yd=None, Xc=None, Xm=None, \
            gip_vae=None, git_vae=None, \
            data_dim=None, prior_dim=None, label_dim=None, \
            batch_size=None, \
            params=None, shared_param_dicts=None):
        # setup a rng for this GITrip
        self.rng = RandStream(rng.randint(100000))
        # record the symbolic variables that will provide inputs to the
        # computation graph created to describe this GITonGIP
        self.Xd = Xd
        self.Yd = Yd
        self.Xc = Xc
        self.Xm = Xm
        # record the dimensionality of the data handled by this GITonGIP
        self.data_dim = data_dim
        self.label_dim = label_dim
        self.prior_1_dim = prior_1_dim
        self.prior_2_dim = prior_2_dim
        self.batch_size = batch_size
        # construct shared parameter clone of the underlying GIPair, for use
        # as the "lower" VAE.
        self.GIP = gip_vae.shared_param_clone(rng=rng, Xd=self.Xd, \
                Xc=self.Xc, Xm=self.Xm)
        # grab a symbolic handle for samples from the approximate posteriors
        # inferred by self.GIP
        self.Xp1 = self.GIP.Xp
        # construct symbolic "dummy" control and mask inputs for the upper VAE
        self.Xc1 = T.zeros_like(self.Xp1)
        self.Xm1 = T.zeros_like(self.Xp1)
        # construct shared parameter clone if the underlying GITrip, for use
        # as the "upper" VAE.
        self.GIT = git_vae.shared_param_clone(rng=rng, Xd=self.Xp1, \
            Yd=self.Yd, Xc=self.Xc1, Xm=self.Xm1)
        # grad symbolic handles for samples from the approximate posterior of
        # the upper VAE, and its estimated distribution over the categorical
        # posterior variable.
        self.Xp2 = self.GIT.Xp
        self.Yp2 = self.GIT.Yp
        self.Yp_proto = self.GIT.Yp_proto

        # verify that the symbolic inputs/outputs are consistently-dimensioned
        assert(self.data_dim == self.GN.mlp_layers[-1].out_dim)
        assert(self.data_dim == self.IN.shared_layers[0].in_dim)
        assert(self.data_dim == self.PN.proto_nets[0][0].in_dim)
        # mu/sigma outputs of self.IN should be equal to prior_dim, output of
        # self.PN should be equal to label_dim, and input of self.GN should be
        # equal to prior_dim + label_dim
        assert(self.prior_dim == self.IN.mu_layers[-1].out_dim)
        assert(self.prior_dim == self.IN.sigma_layers[-1].out_dim)
        assert(self.label_dim == self.PN.proto_nets[0][-1].out_dim)
        assert((self.prior_dim + self.label_dim) == self.GN.mlp_layers[0].in_dim)

        # determine whether this GITrip is a clone or an original
        if shared_param_dicts is None:
            # This is not a clone, and we will need to make a dict for
            # referring to some important shared parameters.
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
            self.lr_gn = theano.shared(value=zero_ary, name='git_lr_gn')
            self.lr_in = theano.shared(value=zero_ary, name='git_lr_in')
            self.lr_pn = theano.shared(value=zero_ary, name='git_lr_pn')
            # shared var momentum parameters for generator and inferencer
            self.mo_gn = theano.shared(value=zero_ary, name='git_mo_gn')
            self.mo_in = theano.shared(value=zero_ary, name='git_mo_in')
            self.mo_pn = theano.shared(value=zero_ary, name='git_mo_pn')
            # init parameters for controlling learning dynamics
            self.set_all_sgd_params()
            # init shared var for weighting nll of data given posterior sample
            self.lam_nll = theano.shared(value=zero_ary, name='git_lam_nll')
            self.set_lam_nll(lam_nll=1.0)
            # init shared var for weighting posterior KL-div from prior
            self.lam_kld = theano.shared(value=zero_ary, name='git_lam_kld')
            self.set_lam_kld(lam_kld=1.0)
            # init shared var for weighting semi-supervised classification
            self.lam_cat = theano.shared(value=zero_ary, name='git_lam_cat')
            self.set_lam_cat(lam_cat=0.0)
            # init shared var for weighting ensemble agreement regularization
            self.lam_pea = theano.shared(value=zero_ary, name='git_lam_pea')
            self.set_lam_pea(lam_pea=0.0)
            # init shared var for weighting entropy regularization on the
            # inferred posteriors over the categorical variable of interest
            self.lam_ent = theano.shared(value=zero_ary, name='git_lam_ent')
            self.set_lam_ent(lam_ent=0.0)
            # init shared var for controlling l2 regularization on params
            self.lam_l2w = theano.shared(value=zero_ary, name='git_lam_l2w')
            self.set_lam_l2w(lam_l2w=1e-3)
            # record shared parameters that are to be shared among clones
            self.shared_param_dicts['git_lr_gn'] = self.lr_gn
            self.shared_param_dicts['git_lr_in'] = self.lr_in
            self.shared_param_dicts['git_lr_pn'] = self.lr_pn
            self.shared_param_dicts['git_mo_gn'] = self.mo_gn
            self.shared_param_dicts['git_mo_in'] = self.mo_in
            self.shared_param_dicts['git_mo_pn'] = self.mo_pn
            self.shared_param_dicts['git_lam_nll'] = self.lam_nll
            self.shared_param_dicts['git_lam_kld'] = self.lam_kld
            self.shared_param_dicts['git_lam_cat'] = self.lam_cat
            self.shared_param_dicts['git_lam_pea'] = self.lam_pea
            self.shared_param_dicts['git_lam_ent'] = self.lam_ent
            self.shared_param_dicts['git_lam_l2w'] = self.lam_l2w
        else:
            # use some shared parameters that are shared among all clones of
            # some "base" GITrip
            self.lr_gn = self.shared_param_dicts['git_lr_gn']
            self.lr_in = self.shared_param_dicts['git_lr_in']
            self.lr_pn = self.shared_param_dicts['git_lr_pn']
            self.mo_gn = self.shared_param_dicts['git_mo_gn']
            self.mo_in = self.shared_param_dicts['git_mo_in']
            self.mo_pn = self.shared_param_dicts['git_mo_pn']
            self.lam_nll = self.shared_param_dicts['git_lam_nll']
            self.lam_kld = self.shared_param_dicts['git_lam_kld']
            self.lam_cat = self.shared_param_dicts['git_lam_cat']
            self.lam_pea = self.shared_param_dicts['git_lam_pea']
            self.lam_ent = self.shared_param_dicts['git_lam_ent']
            self.lam_l2w = self.shared_param_dicts['git_lam_l2w']

        # Grab the full set of "optimizable" parameters from the generator
        # and inferencer networks that we'll be working with.
        self.gn_params = [p for p in self.GN.mlp_params]
        self.in_params = [p for p in self.IN.mlp_params]
        self.pn_params = [p for p in self.PN.proto_params]

        ###################################
        # CONSTRUCT THE COSTS TO OPTIMIZE #
        ###################################
        self.data_nll_cost = self.lam_nll[0] * self._construct_data_nll_cost()
        self.post_kld_cost = self.lam_kld[0] * self._construct_post_kld_cost()
        self.post_cat_cost = self.lam_cat[0] * self._construct_post_cat_cost()
        self.post_pea_cost = self.lam_pea[0] * self._construct_post_pea_cost()
        self.post_ent_cost = self.lam_ent[0] * self._construct_post_ent_cost()
        self.other_reg_costs = self._construct_other_reg_cost()
        self.other_reg_cost = self.other_reg_costs[0]
        self.joint_cost = self.data_nll_cost + self.post_kld_cost + self.post_cat_cost + \
                self.post_pea_cost + self.post_ent_cost + self.other_reg_cost

        # Initialize momentums for mini-batch SGD updates. All parameters need
        # to be safely nestled in their lists by now.
        self.joint_moms = OrderedDict()
        self.gn_moms = OrderedDict()
        self.in_moms = OrderedDict()
        self.pn_moms = OrderedDict()
        for p in self.gn_params:
            p_mo = np.zeros(p.get_value(borrow=True).shape) + 5.0
            self.gn_moms[p] = theano.shared(value=p_mo.astype(theano.config.floatX))
            self.joint_moms[p] = self.gn_moms[p]
        for p in self.in_params:
            p_mo = np.zeros(p.get_value(borrow=True).shape) + 5.0
            self.in_moms[p] = theano.shared(value=p_mo.astype(theano.config.floatX))
            self.joint_moms[p] = self.in_moms[p]
        for p in self.pn_params:
            p_mo = np.zeros(p.get_value(borrow=True).shape) + 5.0
            self.pn_moms[p] = theano.shared(value=p_mo.astype(theano.config.floatX))
            self.joint_moms[p] = self.pn_moms[p]

        # Now, we need to construct updates for inferencers and the generator
        self.joint_updates = OrderedDict()
        self.gn_updates = OrderedDict()
        self.in_updates = OrderedDict()
        self.pn_updates = OrderedDict()
        self.grad_sq_sums = []
        #######################################
        # Construct updates for the generator #
        #######################################
        for var in self.gn_params:
            # these updates are for trainable params in the generator net...
            # first, get gradient of cost w.r.t. var
            var_grad = T.grad(self.joint_cost, var, \
                    consider_constant=[self.GN.dist_mean, self.GN.dist_cov]).clip(-1.0,1.0)
            #var_grad = ifelse(T.any(T.isnan(nan_grad)), T.zeros_like(nan_grad), nan_grad)
            #self.grad_sq_sums.append(T.sum(var_grad**2.0))
            # get the momentum for this var
            var_mom = self.gn_moms[var]
            # update the momentum for this var using its grad
            self.gn_updates[var_mom] = (self.mo_gn[0] * var_mom) + \
                    ((1.0 - self.mo_gn[0]) * (var_grad**2.0))
            self.joint_updates[var_mom] = self.gn_updates[var_mom]
            # make basic update to the var
            var_new = var - (self.lr_gn[0] * (var_grad / T.sqrt(var_mom + 1e-1)))
            # apply "norm clipping" if desired
            if ((var in self.GN.clip_params) and \
                    (var in self.GN.clip_norms) and \
                    (self.GN.clip_params[var] == 1)):
                clip_norm = self.GN.clip_norms[var]
                var_norms = T.sum(var_new**2.0, axis=1, keepdims=True) + 1e-4
                var_scale = T.clip(T.sqrt(clip_norm / var_norms), 0., 1.)
                self.gn_updates[var] = var_new * var_scale
            else:
                self.gn_updates[var] = var_new
            # add this var's update to the joint updates too
            self.joint_updates[var] = self.gn_updates[var]
        ###################################################
        # Construct updates for the continuous inferencer #
        ###################################################
        for var in self.in_params:
            # these updates are for trainable params in the inferencer net...
            # first, get gradient of cost w.r.t. var
            var_grad = T.grad(self.joint_cost, var, \
                    consider_constant=[self.GN.dist_mean, self.GN.dist_cov]).clip(-1.0,1.0)
            #var_grad = ifelse(T.any(T.isnan(nan_grad)), T.zeros_like(nan_grad), nan_grad)
            #self.grad_sq_sums.append(T.sum(var_grad**2.0))
            # get the momentum for this var
            var_mom = self.in_moms[var]
            # update the momentum for this var using its grad
            self.in_updates[var_mom] = (self.mo_in[0] * var_mom) + \
                    ((1.0 - self.mo_in[0]) * (var_grad**2.0))
            self.joint_updates[var_mom] = self.in_updates[var_mom]
            # make basic update to the var
            var_new = var - (self.lr_in[0] * (var_grad / T.sqrt(var_mom + 1e-1)))
            # apply "norm clipping" if desired
            if ((var in self.IN.clip_params) and \
                    (var in self.IN.clip_norms) and \
                    (self.IN.clip_params[var] == 1)):
                clip_norm = self.IN.clip_norms[var]
                var_norms = T.sum(var_new**2.0, axis=1, keepdims=True) + 1e-4
                var_scale = T.clip(T.sqrt(clip_norm / var_norms), 0., 1.)
                self.in_updates[var] = var_new * var_scale
            else:
                self.in_updates[var] = var_new
            # add this var's update to the joint updates too
            self.joint_updates[var] = self.in_updates[var]
        ####################################################
        # Construct updates for the categorical inferencer #
        ####################################################
        for var in self.pn_params:
            # these updates are for trainable params in the inferencer net...
            # first, get gradient of cost w.r.t. var
            var_grad = T.grad(self.joint_cost, var, \
                    consider_constant=[self.GN.dist_mean, self.GN.dist_cov]).clip(-1.0,1.0)
            #var_grad = ifelse(T.any(T.isnan(nan_grad)), T.zeros_like(nan_grad), nan_grad)
            #self.grad_sq_sums.append(T.sum(var_grad**2.0))
            # get the momentum for this var
            var_mom = self.pn_moms[var]
            # update the momentum for this var using its grad
            self.pn_updates[var_mom] = (self.mo_pn[0] * var_mom) + \
                    ((1.0 - self.mo_pn[0]) * (var_grad**2.0))
            self.joint_updates[var_mom] = self.pn_updates[var_mom]
            # make basic update to the var
            var_new = var - (self.lr_pn[0] * (var_grad / T.sqrt(var_mom + 1e-1)))
            # apply "norm clipping" if desired
            if ((var in self.PN.clip_params) and \
                    (var in self.PN.clip_norms) and \
                    (self.PN.clip_params[var] == 1)):
                clip_norm = self.PN.clip_norms[var]
                var_norms = T.sum(var_new**2.0, axis=1, keepdims=True) + 1e-4
                var_scale = T.clip(T.sqrt(clip_norm / var_norms), 0., 1.)
                self.pn_updates[var] = var_new * var_scale
            else:
                self.pn_updates[var] = var_new
            # add this var's update to the joint updates too
            self.joint_updates[var] = self.pn_updates[var]
        # Record the sum of squared gradients (for NaN checking)
        self.grad_sq_sum = T.sum(self.grad_sq_sums)

        # Construct batch-based training functions for the generator and
        # inferer networks, as well as a joint training function.
        #self.train_gn = self._construct_train_gn()
        #self.train_in = self._construct_train_in()
        self.train_joint = self._construct_train_joint()
        return

    def set_gn_sgd_params(self, learn_rate=0.02, momentum=0.9):
        """
        Set learning rate and momentum parameter for self.GN updates.
        """
        zero_ary = np.zeros((1,))
        new_lr = zero_ary + learn_rate
        self.lr_gn.set_value(new_lr.astype(theano.config.floatX))
        new_mo = zero_ary + momentum
        self.mo_gn.set_value(new_mo.astype(theano.config.floatX))
        return

    def set_in_sgd_params(self, learn_rate=0.02, momentum=0.9):
        """
        Set learning rate and momentum parameter for self.IN updates.
        """
        zero_ary = np.zeros((1,))
        new_lr = zero_ary + learn_rate
        self.lr_in.set_value(new_lr.astype(theano.config.floatX))
        new_mo = zero_ary + momentum
        self.mo_in.set_value(new_mo.astype(theano.config.floatX))
        return

    def set_pn_sgd_params(self, learn_rate=0.02, momentum=0.9):
        """
        Set learning rate and momentum parameter for self.PN updates.
        """
        zero_ary = np.zeros((1,))
        new_lr = zero_ary + learn_rate
        self.lr_pn.set_value(new_lr.astype(theano.config.floatX))
        new_mo = zero_ary + momentum
        self.mo_pn.set_value(new_mo.astype(theano.config.floatX))
        return

    def set_all_sgd_params(self, learn_rate=0.02, momentum=0.9):
        """
        Set learning rate and momentum parameter for all updates.
        """
        zero_ary = np.zeros((1,))
        # set learning rates
        new_lr = zero_ary + learn_rate
        self.lr_gn.set_value(new_lr.astype(theano.config.floatX))
        self.lr_in.set_value(new_lr.astype(theano.config.floatX))
        self.lr_pn.set_value(new_lr.astype(theano.config.floatX))
        # set momentums
        new_mo = zero_ary + momentum
        self.mo_gn.set_value(new_mo.astype(theano.config.floatX))
        self.mo_in.set_value(new_mo.astype(theano.config.floatX))
        self.mo_pn.set_value(new_mo.astype(theano.config.floatX))
        return

    def set_lam_nll(self, lam_nll=1.0):
        """
        Set weight for controlling the influence of the data likelihood.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_nll
        self.lam_nll.set_value(new_lam.astype(theano.config.floatX))
        return

    def set_lam_cat(self, lam_cat=0.0):
        """
        Set the strength of semi-supervised classification cost.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_cat
        self.lam_cat.set_value(new_lam.astype(theano.config.floatX))
        return

    def set_lam_kld(self, lam_kld=1.0):
        """
        Set the strength of regularization on KL-divergence for continuous
        posterior variables. When set to 1.0, this reproduces the standard
        role of KL(posterior || prior) in variational learning.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_kld
        self.lam_kld.set_value(new_lam.astype(theano.config.floatX))
        return

    def set_lam_pea(self, lam_pea=0.0):
        """
        Set the strength of PEA regularization on the categorical posterior.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_pea
        self.lam_pea.set_value(new_lam.astype(theano.config.floatX))
        return

    def set_lam_ent(self, lam_ent=0.0):
        """
        Set the strength of entropy regularization on the categorical posterior.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_ent
        self.lam_ent.set_value(new_lam.astype(theano.config.floatX))
        return

    def set_lam_l2w(self, lam_l2w=1e-3):
        """
        Set the relative strength of l2 regularization on network params.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_l2w
        self.lam_l2w.set_value(new_lam.astype(theano.config.floatX))
        return

    def _construct_data_nll_cost(self, prob_type='bernoulli'):
        """
        Construct the negative log-likelihood part of cost to minimize.
        """
        assert((prob_type == 'bernoulli') or (prob_type == 'gaussian'))
        Xd_rep = T.repeat(self.Xd, self.label_dim, axis=0)
        if (prob_type == 'bernoulli'):
            log_prob_cost = log_prob_bernoulli(Xd_rep, self.GN.output)
        else:
            log_prob_cost = log_prob_gaussian(Xd_rep, self.GN.output, \
                    le_sigma=1.0)
        cat_probs = T.flatten(self.Yp).dimshuffle(0, 'x')
        nll_cost = -T.sum((cat_probs * log_prob_cost)) / self.Xd.shape[0]
        return nll_cost

    def _construct_post_kld_cost(self):
        """
        Construct the posterior KL-d from prior part of cost to minimize.
        """
        kld_cost = T.sum(self.IN.kld_cost) / self.Xd.shape[0]
        return kld_cost

    def _construct_post_cat_cost(self):
        """
        Construct the label-based semi-supervised cost.
        """
        row_idx = T.arange(self.Yd.shape[0])
        row_mask = T.neq(self.Yd, 0).reshape((self.Yd.shape[0], 1))
        wacky_mat = (self.Yp * row_mask) + (1. - row_mask)
        cat_cost = -T.sum(safe_log(wacky_mat[row_idx,(self.Yd.flatten()-1)])) \
                / (T.sum(row_mask) + 1e-4)
        return cat_cost

    def _construct_post_pea_cost(self):
        """
        Construct the pseudo-ensemble agreement cost on the approximate
        posteriors over the categorical latent variable.
        """
        pea_cost = T.sum(self.PN.pea_reg_cost) / self.Xd.shape[0]
        return pea_cost

    def _construct_post_ent_cost(self):
        """
        Construct the entropy cost on the categorical posterior.
        """
        ent_cost = T.sum(cat_entropy(self.Yp)) / self.Xd.shape[0]
        return ent_cost

    def _construct_other_reg_cost(self):
        """
        Construct the cost for low-level basic regularization. E.g. for
        applying l2 regularization to the network activations and parameters.
        """
        act_reg_cost = self.IN.act_reg_cost + self.GN.act_reg_cost + \
                self.PN.act_reg_cost
        gp_cost = sum([T.sum(par**2.0) for par in self.gn_params])
        ip_cost = sum([T.sum(par**2.0) for par in self.in_params])
        pp_cost = sum([T.sum(par**2.0) for par in self.pn_params])
        param_reg_cost = self.lam_l2w[0] * (gp_cost + ip_cost + pp_cost)
        other_reg_cost = (act_reg_cost /self.Xd.shape[0]) + param_reg_cost
        return [other_reg_cost, gp_cost, ip_cost, pp_cost, act_reg_cost]

    def _construct_train_joint(self):
        """
        Construct theano function to train inferencer and generator jointly.
        """
        #out_mu_sum = T.sum(self.IN.output_mu**2.0)
        #out_sigma_sum = T.sum(self.IN.output_sigma**2.0)
        outputs = [self.joint_cost, self.data_nll_cost, self.post_kld_cost, \
                self.post_cat_cost, self.post_pea_cost, self.post_ent_cost, \
                self.other_reg_cost]
                #, self.grad_sq_sum, self.other_reg_costs[1], \
                #self.other_reg_costs[2], self.other_reg_costs[3], self.other_reg_costs[4], \
                #out_mu_sum, out_sigma_sum]
        func = theano.function(inputs=[ self.Xd, self.Xc, self.Xm, self.Yd ], \
                outputs=outputs, updates=self.joint_updates, \
                mode=theano.Mode(linker='vm'))
        COMMENT="""
        theano.printing.pydotprint(func, \
            outfile='GITrip_train_joint.svg', compact=True, format='svg', with_ids=False, \
            high_contrast=True, cond_highlight=None, colorCodes=None, \
            max_label_size=70, scan_graphs=False, var_with_name_simple=False, \
            print_output_file=True, assert_nb_all_strings=-1)
        """
        return func

    def shared_param_clone(self, rng=None, Xd=None, Yd=None, Xc=None, Xm=None):
        """
        Create a "shared-parameter" clone of this GITrip.

        This can be used for chaining VAEs for BPTT. (and other stuff too)
        """
        clone_git = GITrip(rng=rng, Xd=Xd, Yd=Yd, Xc=Xc, Xm=Xm, \
            g_net=self.GN, i_net=self.IN, p_net=self.PN, \
            data_dim=self.data_dim, prior_dim=self.prior_dim, label_dim=self.label_dim, \
            batch_size=self.batch_size, params=self.params, \
            shared_param_dicts=self.shared_param_dicts)
        return clone_git

    def sample_git_from_data(self, X_d, loop_iters=5):
        """
        Sample for several rounds through the I<->G loop, initialized with the
        the "data variable" samples in X_d.
        """
        data_samples = []
        prior_samples = []
        X_c = 0.0 * X_d
        X_m = 0.0 * X_d
        for i in range(loop_iters):
            # record the data samples for this iteration
            data_samples.append(1.0 * X_d)
            # sample from their inferred posteriors
            X_p = self.IN.sample_posterior(X_d, X_c, X_m)
            Y_p = self.PN.sample_posterior(X_d)
            XY_p = np.hstack([Y_p, X_p]).astype(theano.config.floatX)
            # record the sampled points (in the "prior space")
            prior_samples.append(1.0 * XY_p)
            # get next data samples by transforming the prior-space points
            X_d = self.GN.transform_prior(XY_p)
        result = {"data samples": data_samples, "prior samples": prior_samples}
        return result

    def classification_error(self, X_d, Y_d, samples=20):
        """
        Compute classification error for a set of observations X_d with known
        labels Y_d, based on passing X_d through the categorical inferencer
        (i.e. self.PN). We average over multiple "binarizations" of X_d.
        """
        # first, convert labels to account for semi-supervised labeling
        Y_mask = 1.0 * (Y_d != 0)
        Y_d = Y_d - 1
        # make a function for computing self.Yp
        func = theano.function([self.Xd], outputs=self.Yp_proto)
        # compute self.Yp for the observations in X_d
        Y_p = None
        for i in range(samples):
            if Y_p == None:
                Y_p = func(binarize_data(X_d))
            else:
                Y_p += func(binarize_data(X_d))
        Y_p = Y_p / float(samples)
        # get the implied class labels
        Y_c = np.argmax(Y_p, axis=1).reshape((Y_d.shape[0],1))
        # compute the classification error for points with valid labels
        err_rate = np.sum(((Y_d != Y_c) * Y_mask)) / np.sum(Y_mask)
        return err_rate

    def class_probs(self, X_d):
        """
        Compute predicted class probabilities for the observations in X_d.
        """
        # make a function for computing self.Yp
        func = theano.function([self.Xd], outputs=self.Yp_proto)
        # compute self.Yp for the observations in X_d
        Y_p = func(X_d)
        return Y_p

if __name__=="__main__":
    # TESTING CODE MOVED TO "MnistTests.py"
    print("TESTING COMPLETE!")




##############
# EYE BUFFER #
##############
