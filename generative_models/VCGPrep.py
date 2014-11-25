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
import theano.tensor.shared_randomstreams
from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams

# phil's sweetness
from NetLayers import HiddenLayer, DiscLayer, safe_log, softplus_actfun
from GenNet import projected_moments

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

def ulh_loss(Yh, Yt=0.0, delta=1.0):
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

def log_prob_bernoulli(p_true, p_approx):
    """
    Compute log probability of some binary variables with probabilities
    given by p_true, for probability estimates given by p_approx. We'll
    compute joint log probabilities over row-wise groups.
    """
    log_prob_1 = p_true * safe_log(p_approx)
    log_prob_0 = (1.0 - p_true) * safe_log(1.0 - p_approx)
    row_log_probs = T.sum((log_prob_1 + log_prob_0), axis=1, keepdims=True)
    return row_log_probs

def log_prob_gaussian(mu_true, mu_approx, le_sigma=1.0):
    """
    Compute log probability of some continuous variables with values given
    by mu_true, w.r.t. gaussian distributions with means given by mu_approx
    and standard deviations given by le_sigma. We assume isotropy.
    """
    ind_log_probs = -( (mu_approx - mu_true)**2.0 / (2.0 * le_sigma**2.0) )
    row_log_probs = T.sum(ind_log_probs, axis=1, keepdims=True)
    return row_log_probs

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

class VCGPrep(object):
    """
    Controller for training a generator/discriminator pair.

    The generator must be an instance of the GenNet class implemented in
    "GenNet.py". The discriminator must be an instance of the PeaNet class,
    as implemented in "PeaNet.py". The inferencer must be an instance of the
    InfNet class implemented in "InfNet.py".

    Parameters:
        rng: numpy.random.RandomState (for reproducibility)
        Xd: symbolic var for providing samples from the data distribution
        Xp: symbolic var for providing samples from the generator's prior
        d_net: The PeaNet instance that will serve as the discriminator
        g_net: The GenNet instance that will serve as the generator
        i_net: The InfNet instance that will serve as the inferencer
        data_dim: dimension of the generated data
        prior_dim: dimension of the model prior
        params: a dict of parameters for controlling various costs
            lam_l2d: regularization on squared discriminator output
            mom_mix_rate: rate for updates to the running moment estimates
                          for the distribution generated by g_net
            mom_match_weight: weight for the "moment matching" cost
            mom_match_proj: projection matrix for reduced-dim mom matching
            target_mean: first-order moment to try and match with g_net
            target_cov: second-order moment to try and match with g_net
    """
    def __init__(self, rng=None, Xd=None, Xp=None, d_net=None, g_net=None, \
                 i_net=None, data_dim=None, prior_dim=None, params=None):
        # Do some stuff!
        self.rng = theano.tensor.shared_randomstreams.RandomStreams( \
                rng.randint(100000))
        self.data_dim = data_dim
        self.prior_dim = prior_dim

        # symbolic var for inputting samples from the data distribution
        self.Xd = Xd
        # symbolic var for inputting samples from the generator's prior
        self.Xp = Xp
        # symbolic matrix of indices for data inputs
        self.Id = T.lvector(name='vcg_Id')
        # symbolic matrix of indices for noise inputs
        self.In = T.lvector(name='vcg_In')

        # create clones of the given generator and discriminator, after
        # rewiring their computation graphs to take the right inputs
        self.GN = g_net.shared_param_clone(rng=rng, Xp=self.Xp)
        self.DN = d_net.shared_param_clone(rng=rng, \
                Xd=T.vertical_stack(Xd, self.GN.output))
        # create a clone of the inferencer, which takes the primary generator's
        # output as its input. we'll set self.Xg as the output of self.GN
        self.Xg = self.GN.output
        self.Xc = 0.0 * self.Xg
        self.Xm = 0.0 * self.Xg
        self.IN = i_net.shared_param_clone(rng=rng, Xd=self.Xg, Xc=self.Xc, \
                Xm=self.Xm)
        # create a second clone of the generator, which takes the output of
        # the inferencer as its input
        self.Xp2 = self.IN.output
        self.GN2 = g_net.shared_param_clone(rng=rng, Xp=self.Xp2)
        self.Xg2 = self.GN2.output

        zero_ary = np.zeros((1,)).astype(theano.config.floatX)
        # init shared var for weighting nll of data given posterior sample
        self.lam_nll = theano.shared(value=zero_ary, name='vcg_lam_nll')
        self.set_lam_nll(lam_nll=1.0)
        # init shared var for weighting posterior KL-div from prior
        self.lam_kld = theano.shared(value=zero_ary, name='vcg_lam_kld')
        self.set_lam_kld(lam_kld=1.0)
        # init shared var for controlling l2 regularization on params
        self.lam_l2w = theano.shared(value=zero_ary, name='vcg_lam_l2w')
        self.set_lam_l2w(lam_l2w=1e-4)
        # shared var learning rate for generator and discriminator
        self.lr_dn = theano.shared(value=zero_ary, name='vcg_lr_dn')
        self.lr_gn = theano.shared(value=zero_ary, name='vcg_lr_gn')
        self.lr_in = theano.shared(value=zero_ary, name='vcg_lr_in')
        # shared var momentum parameters for generator and discriminator
        self.mo_dn = theano.shared(value=zero_ary, name='vcg_mo_dn')
        self.mo_gn = theano.shared(value=zero_ary, name='vcg_mo_gn')
        self.mo_in = theano.shared(value=zero_ary, name='vcg_mo_in')
        # shared var weights for adversarial classification objective
        self.dw_dn = theano.shared(value=zero_ary, name='vcg_dw_dn')
        self.dw_gn = theano.shared(value=zero_ary, name='vcg_dw_gn')
        # init parameters for controlling learning dynamics
        self.set_dn_sgd_params() # init SGD rate/momentum for DN
        self.set_gn_sgd_params() # init SGD rate/momentum for GN
        self.set_in_sgd_params() # init SGD rate/momentum for IN
        
        self.set_disc_weights()  # init adversarial cost weights for GN/DN
        self.lam_l2d = theano.shared(value=(zero_ary + params['lam_l2d']), \
                name='vcg_lam_l2d')

        #######################################################
        # Welcome to: Moment Matching Cost Information Center #
        #######################################################
        #
        # Get parameters for managing the moment matching cost. The moment
        # matching is based on exponentially-decaying estimates of the mean
        # and covariance of the distribution induced by the generator network
        # and the (latent) noise being fed to it.
        #
        # We provide the option of performing moment matching with either the
        # raw generator output, or with linearly-transformed generator output.
        # Either way, the given target mean and covariance should have the
        # appropriate dimension for the space in which we'll be matching the
        # generator's 1st/2nd moments with the target's 1st/2nd moments. For
        # clarity, the computation we'll perform looks like:
        #
        #   Xm = X - np.mean(X, axis=0)
        #   XmP = np.dot(Xm, P)
        #   C = np.dot(XmP.T, XmP)
        #
        # where Xm is the mean-centered samples from the generator and P is
        # the matrix for the linear transform to apply prior to computing
        # the moment matching cost. For simplicity, the above code ignores the
        # use of an exponentially decaying average to track the estimated mean
        # and covariance of the generator's output distribution.
        #
        # The relative contribution of the current batch to these running
        # estimates is determined by self.mom_mix_rate. The mean estimate is
        # first updated based on the current batch, then the current batch
        # is centered with the updated mean, then the covariance estimate is
        # updated with the mean-centered samples in the current batch.
        #
        # Strength of the moment matching cost is given by self.mom_match_cost.
        # Target mean/covariance are given by self.target_mean/self.target_cov.
        # If a linear transform is to be applied prior to matching, it is given
        # by self.mom_match_proj.
        #
        zero_ary = np.zeros((1,))
        mmr = zero_ary + params['mom_mix_rate']
        self.mom_mix_rate = theano.shared(name='vcg_mom_mix_rate', \
            value=mmr.astype(theano.config.floatX))
        mmw = zero_ary + params['mom_match_weight']
        self.mom_match_weight = theano.shared(name='vcg_mom_match_weight', \
            value=mmw.astype(theano.config.floatX))
        targ_mean = params['target_mean'].astype(theano.config.floatX)
        targ_cov = params['target_cov'].astype(theano.config.floatX)
        assert(targ_mean.size == targ_cov.shape[0]) # mean and cov use same dim
        assert(targ_cov.shape[0] == targ_cov.shape[1]) # cov must be square
        self.target_mean = theano.shared(value=targ_mean, name='vcg_target_mean')
        self.target_cov = theano.shared(value=targ_cov, name='vcg_target_cov')
        mmp = np.identity(targ_cov.shape[0]) # default to identity transform
        if 'mom_match_proj' in params:
            mmp = params['mom_match_proj'] # use a user-specified transform
        assert(mmp.shape[0] == self.data_dim) # transform matches data dim
        assert(mmp.shape[1] == targ_cov.shape[0]) # and matches mean/cov dims
        mmp = mmp.astype(theano.config.floatX)
        self.mom_match_proj = theano.shared(value=mmp, name='vcg_mom_map_proj')
        # finally, we can construct the moment matching cost! and the updates
        # for the running mean/covariance estimates too!
        self.mom_match_cost, self.mom_updates = self._construct_mom_stuff()
        #########################################
        # Thank you for visiting the M.M.C.I.C. #
        #########################################

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
        self.gn_params = [p for p in self.GN.mlp_params]
        self.in_params = [p for p in self.IN.mlp_params]

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
        # optimization of the generator and inferer networks.
        self.dn_cost = self.disc_cost_dn + self.DN.act_reg_cost + self.disc_reg_cost
        # construct costs relevant to the optimization of the generator and
        # discriminator networks
        self.data_nll_cost = self.lam_nll[0] * self._construct_data_nll_cost()
        self.post_kld_cost = self.lam_kld[0] * self._construct_post_kld_cost()
        self.other_reg_cost = self._construct_other_reg_cost()
        self.gin_cost = self.disc_cost_gn + self.mom_match_cost + \
                self.data_nll_cost + self.post_kld_cost + self.other_reg_cost
        # compute total cost on the discriminator and VB generator/inferencer
        self.dgi_cost = self.dn_cost + self.gin_cost

        # Initialize momentums for mini-batch SGD updates. All parameters need
        # to be safely nestled in their lists by now.
        self.joint_moms = OrderedDict()
        self.dn_moms = OrderedDict()
        self.gn_moms = OrderedDict()
        self.in_moms = OrderedDict()
        for p in self.dn_params:
            p_mo = np.zeros(p.get_value(borrow=True).shape) + 5.0
            self.dn_moms[p] = theano.shared(value=p_mo.astype(theano.config.floatX))
            self.joint_moms[p] = self.dn_moms[p]
        for p in self.gn_params:
            p_mo = np.zeros(p.get_value(borrow=True).shape) + 5.0
            self.gn_moms[p] = theano.shared(value=p_mo.astype(theano.config.floatX))
            self.joint_moms[p] = self.gn_moms[p]
        for p in self.in_params:
            p_mo = np.zeros(p.get_value(borrow=True).shape) + 5.0
            self.in_moms[p] = theano.shared(value=p_mo.astype(theano.config.floatX))
            self.joint_moms[p] = self.in_moms[p]

        # Construct the updates for the generator and discriminator network
        self.joint_updates = OrderedDict()
        self.dn_updates = OrderedDict()
        self.gn_updates = OrderedDict()
        self.in_updates = OrderedDict()
        ########################################################
        # Construct updates for the moment tracking parameters #
        ########################################################
        for var in self.mom_updates:
            # these updates are for the generator distribution's running first
            # and second-order moment estimates
            self.gn_updates[var] = self.mom_updates[var]
            self.joint_updates[var] = self.gn_updates[var]
        ###########################################
        # Construct updates for the discriminator #
        ###########################################
        for var in self.dn_params:
            # these updates are for trainable params in the inferencer net...
            # first, get gradient of cost w.r.t. var
            var_grad = T.grad(self.dn_cost, var, \
                    consider_constant=[self.GN.dist_mean, self.GN.dist_cov])
            # get the momentum for this var
            var_mom = self.dn_moms[var]
            # update the momentum for this var using its grad
            self.dn_updates[var_mom] = (self.mo_dn[0] * var_mom) + \
                    ((1.0 - self.mo_dn[0]) * (var_grad**2.0))
            self.joint_updates[var_mom] = self.dn_updates[var_mom]
            # make basic update to the var
            var_new = var - (self.lr_dn[0] * (var_grad / T.sqrt(var_mom + 1e-2)))
            # apply "norm clipping" if desired
            if ((var in self.DN.clip_params) and \
                    (var in self.DN.clip_norms) and \
                    (self.DN.clip_params[var] == 1)):
                clip_norm = self.DN.clip_norms[var]
                var_norms = T.sum(var_new**2.0, axis=1, keepdims=True)
                var_scale = T.clip(T.sqrt(clip_norm / var_norms), 0., 1.)
                self.dn_updates[var] = var_new * var_scale
            else:
                self.dn_updates[var] = var_new
            # add this var's update to the joint updates too
            self.joint_updates[var] = self.dn_updates[var]
        #######################################
        # Construct updates for the generator #
        #######################################
        for var in self.gn_params:
            # these updates are for trainable params in the generator net...
            # first, get gradient of cost w.r.t. var
            var_grad = T.grad(self.gin_cost, var, \
                    consider_constant=[self.GN.dist_mean, self.GN.dist_cov])
            # get the momentum for this var
            var_mom = self.gn_moms[var]
            # update the momentum for this var using its grad
            self.gn_updates[var_mom] = (self.mo_gn[0] * var_mom) + \
                    ((1.0 - self.mo_gn[0]) * (var_grad**2.0))
            self.joint_updates[var_mom] = self.gn_updates[var_mom]
            # make basic update to the var
            var_new = var - (self.lr_gn[0] * (var_grad / T.sqrt(var_mom + 1e-2)))
            # apply "norm clipping" if desired
            if ((var in self.GN.clip_params) and \
                    (var in self.GN.clip_norms) and \
                    (self.GN.clip_params[var] == 1)):
                clip_norm = self.GN.clip_norms[var]
                var_norms = T.sum(var_new**2.0, axis=1, keepdims=True)
                var_scale = T.clip(T.sqrt(clip_norm / var_norms), 0., 1.)
                self.gn_updates[var] = var_new * var_scale
            else:
                self.gn_updates[var] = var_new
            # add this var's update to the joint updates too
            self.joint_updates[var] = self.gn_updates[var]
        ########################################
        # Construct updates for the inferencer #
        ########################################
        for var in self.in_params:
            # these updates are for trainable params in the generator net...
            # first, get gradient of cost w.r.t. var
            var_grad = T.grad(self.gin_cost, var, \
                    consider_constant=[self.GN.dist_mean, self.GN.dist_cov])
            # get the momentum for this var
            var_mom = self.in_moms[var]
            # update the momentum for this var using its grad
            self.in_updates[var_mom] = (self.mo_in[0] * var_mom) + \
                    ((1.0 - self.mo_in[0]) * (var_grad**2.0))
            self.joint_updates[var_mom] = self.in_updates[var_mom]
            # make basic update to the var
            var_new = var - (self.lr_in[0] * (var_grad / T.sqrt(var_mom + 1e-2)))
            # apply "norm clipping" if desired
            if ((var in self.IN.clip_params) and \
                    (var in self.IN.clip_norms) and \
                    (self.IN.clip_params[var] == 1)):
                clip_norm = self.IN.clip_norms[var]
                var_norms = T.sum(var_new**2.0, axis=1, keepdims=True)
                var_scale = T.clip(T.sqrt(clip_norm / var_norms), 0., 1.)
                self.in_updates[var] = var_new * var_scale
            else:
                self.in_updates[var] = var_new
            # add this var's update to the joint updates too
            self.joint_updates[var] = self.in_updates[var]

        # Construct the function for training on training data
        self.train_joint = self._construct_train_joint()

        # Construct a function for computing the ouputs of the generator
        # network for a batch of noise. Presumably, the noise will be drawn
        # from the same distribution that was used in training....
        self.sample_from_gn = self.GN.sample_from_model
        return

    def set_dn_sgd_params(self, learn_rate=0.02, momentum=0.9):
        """
        Set learning rate and momentum parameter for discriminator updates.
        """
        zero_ary = np.zeros((1,))
        new_lr = zero_ary + learn_rate
        self.lr_dn.set_value(new_lr.astype(theano.config.floatX))
        new_mo = zero_ary + momentum
        self.mo_dn.set_value(new_mo.astype(theano.config.floatX))
        return

    def set_gn_sgd_params(self, learn_rate=0.02, momentum=0.9):
        """
        Set learning rate and momentum parameter for generator updates.
        """
        zero_ary = np.zeros((1,))
        new_lr = zero_ary + learn_rate
        self.lr_gn.set_value(new_lr.astype(theano.config.floatX))
        new_mo = zero_ary + momentum
        self.mo_gn.set_value(new_mo.astype(theano.config.floatX))
        return

    def set_in_sgd_params(self, learn_rate=0.02, momentum=0.9):
        """
        Set learning rate and momentum parameter for self.PN updates.
        """
        zero_ary = np.zeros((1,))
        new_lr = zero_ary + learn_rate
        self.lr_in.set_value(new_lr.astype(theano.config.floatX))
        new_mo = zero_ary + momentum
        self.mo_in.set_value(new_mo.astype(theano.config.floatX))
        return

    def set_all_sgd_params(self, learn_rate=0.02, momentum=0.9):
        """
        Set learning rate and momentum parameter for all updates.
        """
        zero_ary = np.zeros((1,))
        # set learning rates
        new_lr = zero_ary + learn_rate
        self.lr_dn.set_value(new_lr.astype(theano.config.floatX))
        self.lr_gn.set_value(new_lr.astype(theano.config.floatX))
        self.lr_in.set_value(new_lr.astype(theano.config.floatX))
        # set momentums
        new_mo = zero_ary + momentum
        self.mo_dn.set_value(new_mo.astype(theano.config.floatX))
        self.mo_gn.set_value(new_mo.astype(theano.config.floatX))
        self.mo_in.set_value(new_mo.astype(theano.config.floatX))
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
        Set the strength of regularization on KL-divergence for continuous
        posterior variables. When set to 1.0, this reproduces the standard
        role of KL(posterior || prior) in variational learning.
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

    def init_moments(self, sample_count):
        """
        Initialize estimates of the generator distribution's 1st and 2nd-order
        moments based on some large sample of input noise to the generator
        network. Estimates will be performed, and subsequently tracked, in a
        transformed space based on self.mom_match_proj.
        """
        # Compute outputs for the input latent noise in X_noise
        X = self.GN.sample_from_model(sample_count)
        # Get the transform to apply prior to moment matching
        P = self.mom_match_proj.get_value(borrow=False)
        # Compute post-transform mean and covariance of the outputs
        mu, sigma = projected_moments(X, P, ary_type='numpy')
        # Initialize the generator network's running moment estimates 
        self.GN.dist_cov.set_value(sigma.astype(theano.config.floatX))
        self.GN.dist_mean.set_value(mu.astype(theano.config.floatX))
        return

    def _construct_disc_layers(self, rng):
        """
        Construct binary discrimination layers for each spawn-net in the
        underlying discrimnator pseudo-ensemble. All spawn-nets spawned from
        the same proto-net will use the same disc-layer parameters.
        """
        self.disc_layers = []
        self.disc_outputs = []
        for sn in self.DN.spawn_nets:
            # construct a "binary discriminator" layer to sit on top of each
            # spawn net in the discriminator pseudo-ensemble
            sn_fl = sn[-1]
            self.disc_layers.append(DiscLayer(rng=rng, \
                    input=sn_fl.noisy_input, in_dim=sn_fl.in_dim))
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
            data_preds = dl_output.take(self.Id, axis=0)
            noise_preds = dl_output.take(self.In, axis=0)
            # compute the cost with respect to which we will be optimizing
            # the parameters of the discriminator network
            dn_pred_count = self.Id.size + self.In.size
            dnl_dn_cost = (logreg_loss(data_preds, 1.0) + \
                    logreg_loss(noise_preds, -1.0)) / dn_pred_count
            # compute the cost with respect to which we will be optimizing
            # the parameters of the generative model
            gn_pred_count = self.In.size
            #dnl_gn_cost = hinge_loss(noise_preds, 0.0) / gn_pred_count
            dnl_gn_cost = ulh_loss(noise_preds, 0.0) / gn_pred_count
            dn_costs.append(dnl_dn_cost)
            gn_costs.append(dnl_gn_cost)
        dn_cost = self.dw_dn[0] * T.sum(dn_costs)
        gn_cost = self.dw_gn[0] * T.sum(gn_costs)
        return [dn_cost, gn_cost]

    def _construct_data_nll_cost(self, prob_type='bernoulli'):
        """
        Construct the negative log-likelihood part of cost to minimize.
        """
        assert((prob_type == 'bernoulli') or (prob_type == 'gaussian'))
        if (prob_type == 'bernoulli'):
            log_prob_cost = log_prob_bernoulli(self.Xg, self.Xg2)
        else:
            log_prob_cost = log_prob_gaussian(self.Xg, self.Xg2, le_sigma=1.0)
        nll_cost = -T.sum(log_prob_cost) / self.Xd.shape[0]
        return nll_cost

    def _construct_post_kld_cost(self):
        """
        Construct the posterior KL-d from prior part of cost to minimize.
        """
        kld_cost = T.sum(self.IN.kld_cost) / self.Xd.shape[0]
        return kld_cost

    def _construct_other_reg_cost(self):
        """
        Construct the cost for low-level basic regularization. E.g. for
        applying l2 regularization to the network activations and parameters.
        """
        act_reg_cost = self.GN.act_reg_cost + self.IN.act_reg_cost
        gp_cost = sum([T.sum(par**2.0) for par in self.gn_params])
        ip_cost = sum([T.sum(par**2.0) for par in self.in_params])
        param_reg_cost = self.lam_l2w[0] * (gp_cost + ip_cost)
        other_reg_cost = (act_reg_cost / self.Xd.shape[0]) + param_reg_cost
        return other_reg_cost

    def _construct_mom_stuff(self):
        """
        Construct the cost function for the moment-matching "regularizer".
        """
        a = self.mom_mix_rate
        dist_mean = self.GN.dist_mean
        dist_cov = self.GN.dist_cov
        # Get the generated sample observations for this batch, transformed
        # linearly into the desired space for moment matching...
        X_b = T.dot(self.GN.output, self.mom_match_proj)
        # Get their mean
        batch_mean = T.mean(X_b, axis=0)
        # Get the updated generator distribution mean
        new_mean = ((1.0 - a[0]) * self.GN.dist_mean) + (a[0] * batch_mean)
        # Use the mean to get the updated generator distribution covariance
        X_b_minus_mean = X_b - new_mean
        # Whelp, I guess this line needs the cast... for some reason...
        batch_cov = T.dot(X_b_minus_mean.T, X_b_minus_mean) / T.cast(X_b.shape[0], 'floatX')
        new_cov = ((1.0 - a[0]) * self.GN.dist_cov) + (a[0] * batch_cov)
        # Get the cost for deviation from the target distribution's moments
        mean_err = new_mean - self.target_mean
        cov_err = (new_cov - self.target_cov)
        mm_cost = self.mom_match_weight[0] * \
                (T.sum(mean_err**2.0) + T.sum(cov_err**2.0))
        # Construct the updates for the running estimates of the generator
        # distribution's first and second-order moments.
        mom_updates = OrderedDict()
        mom_updates[self.GN.dist_mean] = new_mean
        mom_updates[self.GN.dist_cov] = new_cov
        return [mm_cost, mom_updates]

    def _construct_train_joint(self):
        """
        Construct theano function to train generator and discriminator jointly.
        """
        outputs = [self.mom_match_cost, self.disc_cost_gn, self.disc_cost_dn, \
                self.data_nll_cost, self.post_kld_cost]
        func = theano.function(inputs=[ self.Xd, self.Xp, self.Id, self.In ], \
                outputs=outputs, \
                updates=self.joint_updates)
        return func

if __name__=="__main__":
    import time
    import utils as utils
    from load_data import load_udm, load_udm_ss, load_mnist
    from PeaNet import PeaNet
    from InfNet import InfNet
    from GenNet import GenNet
    from NetLayers import relu_actfun, softplus_actfun, \
                          safe_softmax, safe_log

    # Simple test code, to check that everything is basically functional.
    print("TESTING...")

    # Initialize a source of randomness
    rng = np.random.RandomState(1234)

    # Load some data to train/validate/test with
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm(dataset, zero_mean=False)
    Xtr = datasets[0][0]

    # get and set some basic dataset information
    tr_samples = Xtr.get_value(borrow=True).shape[0]
    data_dim = Xtr.get_value(borrow=True).shape[1]
    prior_dim = 100
    prior_sigma = 5.0
    
    # Do moment matching in some transformed space
    mm_proj_dim = 250
    #P = np.identity(data_dim)
    P = npr.randn(data_dim, mm_proj_dim) / np.sqrt(float(mm_proj_dim))
    P = theano.shared(value=P.astype(theano.config.floatX), name='P_proj')

    target_mean, target_cov = projected_moments(Xtr, P, ary_type='theano')
    P = P.get_value(borrow=False).astype(theano.config.floatX)

    # Symbolic inputs
    Xp_sym = T.matrix(name='Xp_sym')
    Xd_sym = T.matrix(name='Xd_sym')
    Xc_sym = T.matrix(name='Xc_sym')
    Xm_sym = T.matrix(name='Xm_sym')

    ###############################
    # Setup discriminator network #
    ###############################
    # Set some reasonable mlp parameters
    dn_params = {}
    # Set up some proto-networks
    pc0 = [data_dim, (200, 4), (200, 4), 10]
    dn_params['proto_configs'] = [pc0]
    # Set up some spawn networks
    sc0 = {'proto_key': 0, 'input_noise': 0.1, 'bias_noise': 0.1, 'do_dropout': True}
    #sc1 = {'proto_key': 0, 'input_noise': 0.1, 'bias_noise': 0.1, 'do_dropout': True}
    dn_params['spawn_configs'] = [sc0]
    dn_params['spawn_weights'] = [1.0]
    # Set remaining params
    dn_params['ear_type'] = 2
    dn_params['ear_lam'] = 0.0
    dn_params['lam_l2a'] = 1e-3
    dn_params['vis_drop'] = 0.2
    dn_params['hid_drop'] = 0.5
    # Initialize a network object to use as the discriminator
    DN = PeaNet(rng=rng, Xd=Xd_sym, params=dn_params)

    ###########################
    # Setup generator network #
    ###########################
    # Choose some parameters for the generative network
    gn_params = {}
    gn_config = [prior_dim, 800, 800, data_dim]
    gn_params['mlp_config'] = gn_config
    gn_params['lam_l2a'] = 1e-3
    gn_params['vis_drop'] = 0.0
    gn_params['hid_drop'] = 0.0
    gn_params['bias_noise'] = 0.1
    gn_params['out_noise'] = 0.1
    gn_params['activation'] = softplus_actfun
    # Initialize a generator network object
    GN = GenNet(rng=rng, Xp=Xp_sym, prior_sigma=prior_sigma, params=gn_params)

    ############################
    # Setup inferencer network #
    ############################
    # choose some parameters for the continuous inferencer
    in_params = {}
    shared_config = [data_dim, 600, 600]
    top_config = [shared_config[-1], prior_dim]
    in_params['shared_config'] = shared_config
    in_params['mu_config'] = top_config
    in_params['sigma_config'] = top_config
    in_params['activation'] = softplus_actfun
    in_params['lam_l2a'] = 1e-3
    in_params['vis_drop'] = 0.0
    in_params['hid_drop'] = 0.0
    in_params['bias_noise'] = 0.1
    in_params['input_noise'] = 0.0
    IN = InfNet(rng=rng, Xd=Xd_sym, Xc=Xc_sym, Xm=Xm_sym, \
            prior_sigma=prior_sigma, params=in_params)

    ########################################################################
    # Initialize the joint controller for the generator/discriminator pair #
    ########################################################################
    vcg_params = {}
    vcg_params['lam_l2d'] = 1e-2
    vcg_params['mom_mix_rate'] = 0.02
    vcg_params['mom_match_weight'] = 0.05
    vcg_params['mom_match_proj'] = P
    vcg_params['target_mean'] = target_mean
    vcg_params['target_cov'] = target_cov

    # Initialize a VCGair instance using the previously constructed generator and
    # discriminator networks.
    VCG = VCGPrep(rng=rng, Xd=Xd_sym, Xp=Xp_sym, d_net=DN, g_net=GN, i_net=IN, \
            data_dim=data_dim, prior_dim=prior_dim, params=vcg_params)    
    # Init generator's mean and covariance estimates with many samples
    VCG.init_moments(10000)

    # initialize sgd parameters
    learn_rate = 0.04
    VCG.set_all_sgd_params(learn_rate=learn_rate, momentum=0.98)

    batch_idx = T.lvector('batch_idx')
    batch_sample = theano.function(inputs=[ batch_idx ], \
            outputs=Xtr.take(batch_idx, axis=0))
    for i in range(500000):
        tr_idx = npr.randint(low=0,high=tr_samples,size=(100,)).astype(np.int32)
        Xn_np = GN.sample_from_prior(100)
        Xd_batch = batch_sample(tr_idx)
        Xd_batch = Xd_batch.astype(theano.config.floatX)
        Xn_batch = Xn_np.astype(theano.config.floatX)
        all_idx = np.arange(200)
        data_idx = all_idx[:100]
        noise_idx = all_idx[100:]
        # set up learning rate stuff
        scale = 1.0 * min(1.0, float(i+1)/50000.0)
        VCG.set_all_sgd_params(learn_rate=scale*learn_rate, momentum=0.98)
        if(i < 5000):
            VCG.set_lam_nll(0.0)
            VCG.set_lam_kld(0.0)
        else:
            VCG.set_lam_nll(scale / 20.0)
            VCG.set_lam_kld((scale**2.0 / 20.0))
        # do a minibatch update
        outputs = VCG.train_joint(Xd_batch, Xn_batch, data_idx, noise_idx)
        mom_match_cost = 1.0 * outputs[0]
        disc_gn = 1.0 * outputs[1]
        disc_dn = 1.0 * outputs[2]
        nll_cost = 1.0 * outputs[3]
        kld_cost = 1.0 * outputs[4]
        if ((i+1 % 100000) == 0):
            learn_rate = learn_rate * 0.75
            VCG.set_all_sgd_params(learn_rate=learn_rate, momentum=0.98)
        if ((i % 1000) == 0):
            print("batch: {0:d}, mom_match_cost: {1:.4f}, disc_dn: {2:.4f}, disc_gn: {3:.4f}, nll: {4:.4f}, kld: {5:.4f}".format( \
                    i, mom_match_cost, disc_dn, disc_gn, nll_cost, kld_cost))
        if ((i % 1000) == 0):
            file_name = "VCG_SAMPLES_b{0:d}.png".format(i)
            Xs = VCG.sample_from_gn(200)
            utils.visualize_samples(Xs, file_name)
            file_name = "VCG_WEIGHTS_b{0:d}.png".format(i)
            utils.visualize(VCG.DN, 0, 0, file_name)

    print("TESTING COMPLETE!")




##############
# EYE BUFFER #
##############
