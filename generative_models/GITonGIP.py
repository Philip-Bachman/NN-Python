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
from GenNet import GenNet
from InfNet import InfNet
from PeaNet import PeaNet
from GIPair import GIPair
from GITrip import GITrip

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
            data_dim=None, prior_1_dim=None, prior_2_dim=None, \
            label_dim=None, batch_size=None, \
            params=None, shared_param_dicts=None):
        # TODO: refactor for use with "encoded" inferencer/generator
        assert(not (gip_vae.IN.use_encoder or gip_vae.GN.use_encoder))

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
        self.Xg1 = self.GIP.Xg
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
        self.Ypp2 = self.GIT.Yp_proto
        self.Xg2 = self.GIT.Xg

        # verify that the symbolic inputs/outputs are consistently-dimensioned
        assert(self.data_dim == self.GIP.IN.shared_layers[0].in_dim)
        assert(self.data_dim == self.GIP.GN.mlp_layers[-1].out_dim)
        # inputs or outputs via self.Xp1
        assert(self.prior_1_dim == self.GIP.IN.mu_layers[-1].out_dim)
        assert(self.prior_1_dim == self.GIP.IN.sigma_layers[-1].out_dim)
        assert(self.prior_1_dim == self.GIP.GN.mlp_layers[0].in_dim)
        assert(self.prior_1_dim == self.GIT.IN.shared_layers[0].in_dim)
        assert(self.prior_1_dim == self.GIT.PN.proto_nets[0][0].in_dim)
        assert(self.prior_1_dim == self.GIT.GN.mlp_layers[-1].out_dim)
        # check input/output dimensions on the GITrip's latent space.
        assert(self.prior_2_dim == self.GIT.IN.mu_layers[-1].out_dim)
        assert(self.prior_2_dim == self.GIT.IN.sigma_layers[-1].out_dim)
        assert(self.label_dim == self.GIT.PN.proto_nets[0][-1].out_dim)
        assert((self.prior_2_dim + self.label_dim) == \
                self.GIT.GN.mlp_layers[0].in_dim)

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
            self.lr_gn = theano.shared(value=zero_ary, name='gog_lr_gn')
            self.lr_in = theano.shared(value=zero_ary, name='gog_lr_in')
            self.lr_pn = theano.shared(value=zero_ary, name='gog_lr_pn')
            # shared var momentum parameters for generator and inferencer
            self.mo_gn = theano.shared(value=zero_ary, name='gog_mo_gn')
            self.mo_in = theano.shared(value=zero_ary, name='gog_mo_in')
            self.mo_pn = theano.shared(value=zero_ary, name='gog_mo_pn')
            # init parameters for controlling learning dynamics
            self.set_all_sgd_params()
            # init shared var for weighting nll of data given posterior sample
            self.lam_nll = theano.shared(value=zero_ary, name='gog_lam_nll')
            self.set_lam_nll(lam_nll=1.0)
            # init shared var for weighting posterior KL-div from prior
            self.lam_kld = theano.shared(value=zero_ary, name='gog_lam_kld')
            self.set_lam_kld(lam_kld=1.0)
            # init shared var for weighting semi-supervised classification
            self.lam_cat = theano.shared(value=zero_ary, name='gog_lam_cat')
            self.set_lam_cat(lam_cat=0.0)
            # init shared var for weighting ensemble agreement regularization
            self.lam_pea = theano.shared(value=zero_ary, name='gog_lam_pea')
            self.set_lam_pea(lam_pea=0.0)
            # init shared var for weighting entropy regularization on the
            # inferred posteriors over the categorical variable of interest
            self.lam_ent = theano.shared(value=zero_ary, name='gog_lam_ent')
            self.set_lam_ent(lam_ent=0.0)
            # init shared var for controlling l2 regularization on params
            self.lam_l2w = theano.shared(value=zero_ary, name='gog_lam_l2w')
            self.set_lam_l2w(lam_l2w=1e-3)
            # record shared parameters that are to be shared among clones
            self.shared_param_dicts['gog_lr_gn'] = self.lr_gn
            self.shared_param_dicts['gog_lr_in'] = self.lr_in
            self.shared_param_dicts['gog_lr_pn'] = self.lr_pn
            self.shared_param_dicts['gog_mo_gn'] = self.mo_gn
            self.shared_param_dicts['gog_mo_in'] = self.mo_in
            self.shared_param_dicts['gog_mo_pn'] = self.mo_pn
            self.shared_param_dicts['gog_lam_nll'] = self.lam_nll
            self.shared_param_dicts['gog_lam_kld'] = self.lam_kld
            self.shared_param_dicts['gog_lam_cat'] = self.lam_cat
            self.shared_param_dicts['gog_lam_pea'] = self.lam_pea
            self.shared_param_dicts['gog_lam_ent'] = self.lam_ent
            self.shared_param_dicts['gog_lam_l2w'] = self.lam_l2w
        else:
            # use some shared parameters that are shared among all clones of
            # some "base" GITrip
            self.lr_gn = self.shared_param_dicts['gog_lr_gn']
            self.lr_in = self.shared_param_dicts['gog_lr_in']
            self.lr_pn = self.shared_param_dicts['gog_lr_pn']
            self.mo_gn = self.shared_param_dicts['gog_mo_gn']
            self.mo_in = self.shared_param_dicts['gog_mo_in']
            self.mo_pn = self.shared_param_dicts['gog_mo_pn']
            self.lam_nll = self.shared_param_dicts['gog_lam_nll']
            self.lam_kld = self.shared_param_dicts['gog_lam_kld']
            self.lam_cat = self.shared_param_dicts['gog_lam_cat']
            self.lam_pea = self.shared_param_dicts['gog_lam_pea']
            self.lam_ent = self.shared_param_dicts['gog_lam_ent']
            self.lam_l2w = self.shared_param_dicts['gog_lam_l2w']

        ###################################
        # CONSTRUCT THE COSTS TO OPTIMIZE #
        ###################################

        # Construct training functions to update:
        #   1: just the "lower" GIPair on its own
        #   2: just the "upper" GITrip on its own
        #   3: both the "lower" and "upper" VAEs simultaneously
        self.train_gip = self._construct_train_gip()
        self.train_git = self._construct_train_git()
        #self.train_joint = self._construct_train_joint()
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

    def _construct_train_gip(self):
        """
        Construct theano function to train inferencer and generator jointly.
        """
        outputs = [self.GIP.joint_cost, self.GIP.data_nll_cost, \
                self.GIP.post_kld_cost, self.GIP.other_reg_cost]
        func = theano.function(inputs=[ self.Xd, self.Xc, self.Xm ], \
                outputs=outputs, updates=self.GIP.joint_updates)
        return func

    def _construct_train_git(self):
        """
        Construct theano function to train the "upper" GITrip VAE.
        """
        outputs = [self.GIT.joint_cost, self.GIT.data_nll_cost, \
                self.GIT.post_kld_cost, self.GIT.post_cat_cost, \
                self.GIT.post_pea_cost, self.GIT.post_ent_cost, \
                self.GIT.other_reg_cost]
        func = theano.function(inputs=[ self.Xd, self.Xc, self.Xm, self.Yd ], \
                outputs=outputs, updates=self.GIT.joint_updates)
        return func

    def sample_git_from_data(self, X_d, loop_iters=5):
        """
        Sample for several rounds through the stacked VAE loop, initialized
        with the "data-space" samples in X_d.
        """
        data_samples = []
        prior_samples = []
        X_c = 0.0 * X_d
        X_m = 0.0 * X_d
        for i in range(loop_iters):
            # record the data samples for this iteration
            data_samples.append(1.0 * X_d)
            # sample from the inferred posterior of the GIP
            X_p_1 = self.GIP.IN.sample_posterior(X_d, X_c, X_m)
            X_c_1 = 0.0 * X_p_1
            X_m_1 = 0.0 * X_p_1
            # samples subsequently from the inferred posterior of the GIT
            X_p_2 = self.GIT.IN.sample_posterior(X_p_1, X_c_1, X_m_1)
            Y_p_2 = self.GIT.PN.sample_posterior(X_p_1)
            XY_p_2 = np.hstack([Y_p_2, X_p_2]).astype(theano.config.floatX)
            # record the sampled points (in the "prior-space" of the GIT)
            prior_samples.append(1.0 * XY_p_2)
            # pass back down from the GIT prior to the original "data-space"
            X_g_2 = self.GIT.GN.transform_prior(XY_p_2)
            X_d = self.GIP.GN.transform_prior(X_g_2)
        result = {"data samples": data_samples, "prior samples": prior_samples}
        return result

    def classification_error(self, X_d, Y_d, samples=20, binarize=True):
        """
        Compute classification error for a set of observations X_d with known
        labels Y_d, based on passing X_d through stacked VAE loop several times
        and looking at the expected categorical posterior off the GIT.
        """
        # first, convert labels to account for semi-supervised labeling
        Y_mask = 1.0 * (Y_d != 0)
        Y_d = Y_d - 1
        # get expected predicted class probabilities
        Y_p = self.class_probs(X_d, samples=samples, binarize=binarize)
        # get the implied class labels
        Y_c = np.argmax(Y_p, axis=1).reshape((Y_d.shape[0],1))
        # compute the classification error for points with valid labels
        err_rate = np.sum(((Y_d != Y_c) * Y_mask)) / np.sum(Y_mask)
        return err_rate

    def class_probs(self, X_d, samples=20, binarize=True):
        """
        Compute predicted class probabilities for the observations in X_d by
        passing X_d through the stacked VAE several times and looking at the
        expectation of the categorical posterior of the GIT.
        """
        # make dummy inputs for some symbolic variables
        X_m = 0.0 * X_d
        X_c = 0.0 * X_d
        # make a function for computing self.Ypp2
        func = theano.function([self.Xd, self.Xc, self.Xm], outputs=self.Ypp2)
        # compute several samples of self.Ypp2 for the inputs in X_d
        Y_p = None
        for i in range(samples):
            if Y_p == None:
                if binarize:
                    Y_p = func(binarize_data(X_d), X_c, X_m)
                else:
                    Y_p = func(X_d, X_c, X_m)
            else:
                if binarize:
                    Y_p += func(binarize_data(X_d), X_c, X_m)
                else:
                    Y_p += func(X_d, X_c, X_m)
        Y_p = Y_p / float(samples)
        return Y_p


if __name__=="__main__":
    #######################################
    # Testing code moved to MnistTests.py #
    #######################################
    print("TESTING COMPLETE!")




##############
# EYE BUFFER #
##############
