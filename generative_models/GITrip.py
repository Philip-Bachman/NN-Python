################################################################################
# Code for managing and training a triplet system comprising:                  #
#   1. a generator conditioned on both continuous latent variables and a       #
#      "one-hot" vector of binary latent variables                             #
#   2. an inferencer for approximating posteriors over the continuous latent   #
#      variables given some input                                              #
#   3. an inferencer for approximating posteriors over the "one-hot" binary    #
#      latent variables given some input                                       #
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
from InfNet import InfNet
from PeaNet import PeaNet

def cat_entropy(p):
    """
    Compute the entropy of (row-wise) categorical distributions in p.
    """
    row_ents = -T.sum((p * safe_log(p)), axis=1, keepdims=True)
    return row_ents

def cat_prior_dir(p, alpha=2.0):
    """
    Log probability under a dirichlet prior, with dirichlet parameter alpha.
    """
    log_prob = T.sum((alpha - 1.0) * safe_log(p))
    return log_prob

def cat_prior_ent(p, ent_weight=1.0):
    """
    Log probability under an "entropy-type" prior, with some "weight".

    The log probability here is (modulo an additive constant) equal to minus
    the categorical entropy. I.e. lower entropy is more probable...
    """
    log_prob = -cat_entropy(p) * ent_weight
    return log_prob

def binarize_data(X):
    """
    Make a sample of bernoulli variables with probabilities given by X.
    """
    X_shape = X.shape
    probs = npr.rand(*X_shape)
    X_binary = 1.0 * (probs < X)
    return X_binary.astype(theano.config.floatX)

def one_hot(Yc, cat_dim=None):
    """
    given a tensor Yc of dimension d with integer values from range(cat_dim),
    return new tensor of dimension d + 1 with values 0/1, where the last
    dimension gives a one-hot representation of the values in Yc.
    
    if cat_dim is not given, cat_dim is set to max(Yc) + 1
    """
    if cat_dim is None:
        cat_dim = T.max(Yc) + 1
    ranges = T.shape_padleft(T.arange(cat_dim), Yc.ndim)
    Yoh = T.eq(ranges, T.shape_padright(Yc, 1))
    return Yoh

def one_hot_np(Yc, cat_dim=None):
    """
    Given a numpy integer column vector Yc, generate a matrix Yoh in which
    Yoh[i,:] is a one-hot vector -- Yoh[i,Yc[i]] = 1.0 and other Yoh[i,j] = 0
    """
    if cat_dim is None:
        cat_dim = np.max(Yc) + 1
    Yoh = np.zeros((Yc.size, cat_dim))
    Yoh[np.arange(Yc.size),Yc.flatten()] = 1.0
    return Yoh

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

class GITrip(object):
    """
    Controller for training a variational autoencoder.

    The generator must be an instance of the InfNet class implemented in
    "InfNet.py". The inferencer for the continuous latent variables must be an
    instance of the InfNet class implemented in "InfNet.py". The inferencer
    for the categorical latent variables must be an instance of the PeaNet
    class implemented in "PeaNet.py".

    Parameters:
        rng: numpy.random.RandomState (for reproducibility)
        Xd: symbolic "data" input to this VAE
        Yd: symbolic "label" input to this VAE
        Xc: symbolic "control" input to this VAE
        Xm: symbolic "mask" input to this VAE
        g_net: The InfNet instance that will serve as the base generator
        i_net: The InfNet instance for inferring continuous posteriors
        p_net: The PeaNet instance for inferring categorical posteriors
        data_dim: dimension of the "observable data" variables
        prior_dim: dimension of the continuous latent variables
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
            g_net=None, i_net=None, p_net=None, \
            data_dim=None, prior_dim=None, label_dim=None, \
            batch_size=None, \
            params=None, shared_param_dicts=None):
        # TODO: refactor for use with "encoded" inferencer/generator
        assert(not (i_net.use_encoder or g_net.use_encoder))

        # setup a rng for this GITrip
        self.rng = RandStream(rng.randint(100000))
        # setup the prior distribution over the categorical variable
        if params is None:
            self.params = {}
        else:
            self.params = params

        # record the dimensionality of the data handled by this GITrip
        self.data_dim = data_dim
        self.label_dim = label_dim
        self.prior_dim = prior_dim
        self.batch_size = batch_size

        # create a mask for disabling and/or reweighting input dimensions
        row_mask = np.ones((self.data_dim,)).astype(theano.config.floatX)
        self.input_mask = theano.shared(value=row_mask, name='git_input_mask')
        
        # record the symbolic variables that will provide inputs to the
        # computation graph created to describe this GITrip
        self.Xd = self.input_mask * Xd
        self.Yd = Yd
        self.Xc = Xc
        self.Xm = Xm
        
        # construct a vertically-repeated identity matrix for marginalizing
        # over possible values of the categorical latent variable.
        Ic = np.vstack([np.identity(label_dim) for i in range(batch_size)])
        self.Ic = theano.shared(value=Ic.astype(theano.config.floatX), name='git_Ic')
        # create "shared-parameter" clones of the continuous and categorical
        # inferencers that this GITrip will be built on.
        self.IN = i_net.shared_param_clone(rng=rng, \
                Xd=self.Xd, Xc=self.Xc, Xm=self.Xm)
        self.PN = p_net.shared_param_clone(rng=rng, Xd=self.Xd)
        # create symbolic variables for the approximate posteriors over the 
        # continuous and categorical latent variables
        self.Xp = self.IN.output
        self.Yp = safe_softmax(self.PN.output_spawn[0])
        self.Yp_proto = safe_softmax(self.PN.output_proto)
        # create a symbolic variable structured to allow easy "marginalization"
        # over possible settings of the categorical latent variable. the left
        # matrix (i.e. self.Ic) comprises batch_size copies of the label_dim
        # dimensional identity matrix stacked on top of each other, and the
        # right matrix comprises a single sample from the approximate posterior
        # over the continuous latent variables for each of batch_size examples
        # with each sample repeated label_dim times.
        self.XYp = T.horizontal_stack(self.Ic, T.repeat(self.Xp, \
                self.label_dim, axis=0))
        # pipe the "convenient marginlization" matrix into a shared parameter
        # clone of the generator network
        self.GN = g_net.shared_param_clone(rng=rng, Xp=self.XYp)
        # capture a handle for sampled reconstructions from the generator
        self.Xg = self.GN.output

        # we will be assuming one proto-net in the pseudo-ensemble represented
        # by self.PN, and either one or two spawn-nets for that proto-net.
        assert(len(self.PN.proto_nets) == 1)
        assert((len(self.PN.spawn_nets) == 1) or \
                (len(self.PN.spawn_nets) == 2))
        # output of the generator and input to the inferencer should both be
        # equal to self.data_dim
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
            # init shared var for weighting dirichlet regularization on the
            # inferred posteriors over the categorical variable of interest
            self.lam_dir = theano.shared(value=zero_ary, name='git_lam_dir')
            self.set_lam_dir(lam_dir=0.0)
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
            self.shared_param_dicts['git_lam_dir'] = self.lam_dir
            self.shared_param_dicts['git_lam_l2w'] = self.lam_l2w
            self.shared_param_dicts['git_input_mask'] = self.input_mask
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
            self.lam_dir = self.shared_param_dicts['git_lam_dir']
            self.lam_l2w = self.shared_param_dicts['git_lam_l2w']
            self.input_mask = self.shared_param_dicts['git_input_mask']

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
        self.post_dir_cost = self.lam_dir[0] * self._construct_post_dir_cost()
        self.other_reg_costs = self._construct_other_reg_cost()
        self.other_reg_cost = self.other_reg_costs[0]
        self.joint_cost = self.data_nll_cost + self.post_kld_cost + self.post_cat_cost + \
                self.post_pea_cost + self.post_ent_cost + self.post_dir_cost + \
                self.other_reg_cost

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
            var_new = var - (self.lr_gn[0] * (var_grad / T.sqrt(var_mom + 1e-2)))
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
            var_new = var - (self.lr_in[0] * (var_grad / T.sqrt(var_mom + 1e-2)))
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
            var_new = var - (self.lr_pn[0] * (var_grad / T.sqrt(var_mom + 1e-2)))
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

    def set_lam_dir(self, lam_dir=0.0):
        """
        Set the strength of dirichlet regularization on the categorical posterior.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_dir
        self.lam_dir.set_value(new_lam.astype(theano.config.floatX))
        return

    def set_lam_l2w(self, lam_l2w=1e-3):
        """
        Set the relative strength of l2 regularization on network params.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_l2w
        self.lam_l2w.set_value(new_lam.astype(theano.config.floatX))
        return

    def set_input_mask(self, input_mask=None):
        """
        Set a (probably) binary mask on the input dimensions.
        """
        assert(input_mask.size == self.data_dim)
        input_mask = input_mask.reshape((self.data_dim,))
        self.input_mask.set_value(input_mask.astype(theano.config.floatX))
        self.GN.set_output_mask(input_mask)
        return

    def _construct_data_nll_cost(self, prob_type='bernoulli'):
        """
        Construct the negative log-likelihood part of cost to minimize.
        """
        # make a one-hot matrix for the labeled/unlabeled points, in which the
        # first column is 1 if and only if that input is unlabeled
        Yoh_extra_col = one_hot(T.flatten(self.Yd), cat_dim=(self.label_dim+1))
        Yoh = Yoh_extra_col[:,1:]
        # make a mask that keeps "labeled" rows and drops "unlabeled" rows
        row_mask_numpy_bs = (1.0 - Yoh_extra_col[:,0])
        row_mask = row_mask_numpy_bs.reshape((Yoh.shape[0],1))
        # make a matrix that is one-hot using known labels for labeled points
        # and is some-warm using values from self.Yp for unlabeled points
        Yoh_and_Yp = (row_mask * Yoh) + ((1.0 - row_mask) * self.Yp)
        # repeat inputs, for easy "marginalization" over categorical values
        Xd_rep = T.repeat(self.Xd, self.label_dim, axis=0)
        # compute log-probability of inputs for each categorical value 
        log_prob_cost = self.GN.compute_log_prob(Xd_rep)
        # marginalize (err, well, lower-bound via Jensen's inequality)
        cat_probs = T.flatten(Yoh_and_Yp).dimshuffle(0, 'x')
        nll_cost = -T.sum((cat_probs * log_prob_cost)) / T.cast(self.Xd.shape[0], 'floatX')
        return nll_cost

    def _construct_post_kld_cost(self):
        """
        Construct the posterior KL-d from prior part of cost to minimize.
        """
        kld_cost = T.sum(self.IN.kld_cost) / T.cast(self.Xd.shape[0], 'floatX')
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
        pea_cost = T.sum(self.PN.pea_reg_cost) / T.cast(self.Xd.shape[0], 'floatX')
        return pea_cost

    def _construct_post_ent_cost(self):
        """
        Construct the "entropy prior" cost on the categorical posterior.
        """
        obs_count = T.cast(self.Xd.shape[0], 'floatX')
        ent_cost = -T.sum(cat_prior_ent(self.Yp)) / obs_count
        return ent_cost

    def _construct_post_dir_cost(self):
        """
        Construct the "dirichlet prior" cost on the categorical posterior.
        """
        obs_count = T.cast(self.Xd.shape[0], 'floatX')
        dir_cost = -T.sum(cat_prior_dir(self.Yp)) / obs_count
        return dir_cost

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
        other_reg_cost = (act_reg_cost / T.cast(self.Xd.shape[0], 'floatX')) + \
                param_reg_cost
        return [other_reg_cost, gp_cost, ip_cost, pp_cost, act_reg_cost]

    def _construct_train_joint(self):
        """
        Construct theano function to train inferencer and generator jointly.
        """
        outputs = [self.joint_cost, self.data_nll_cost, self.post_kld_cost, \
                self.post_cat_cost, self.post_pea_cost, self.post_ent_cost, \
                self.post_dir_cost, self.other_reg_cost]
        func = theano.function(inputs=[ self.Xd, self.Xc, self.Xm, self.Yd ], \
                outputs=outputs, updates=self.joint_updates)
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

    def sample_synth_labels(self, X_d, Y_d, loop_iters=5, binarize=False):
        """
        Sample for several rounds through the I<->G loop, initialized with the
        the "data variable" samples in X_d. Include label information in the
        sampling process.

        If Y_d[i] == 0, sample freely from the categorical posterior.
        If Y_d[i] != 0, keep the categorical fixed to (Y_d[0] - 1).
        """
        # make a one-hot matrix for the labeled/unlabeled points, in which the
        # first column is 1 if and only if that input is unlabeled
        Yoh_extra_col = one_hot_np(Y_d, cat_dim=(self.label_dim+1))
        Yoh = Yoh_extra_col[:,1:]
        # make a mask that keeps "labeled" rows and drops "unlabeled" rows
        row_mask_numpy_bs = (1.0 - Yoh_extra_col[:,0])
        row_mask = row_mask_numpy_bs.reshape((Yoh.shape[0],1))
        # run multiple times through the loop, keeping some labels fixed and
        # letting others vary according to the categorical posteriors.
        data_samples = []
        prob_samples = []
        label_samples = []
        X_c = 0.0 * X_d
        X_m = 0.0 * X_d
        for i in range(loop_iters):
            # record the data samples for this iteration
            data_samples.append(1.0 * X_d)
            # sample from their inferred posteriors
            X_p = self.IN.sample_posterior(X_d, X_c, X_m)
            Y_p = self.PN.sample_posterior(X_d)
            # use given labels when available, otherwise use sampled labels
            Y_prob = (row_mask * Yoh) + ((1.0 - row_mask) * Y_p)
            # record the indices of the labels (shifted for semi-supervision)
            Y_label = np.argmax(Y_prob, axis=1) + 1
            Y_label.reshape((Y_label.size,1)) # reshape for numpy vector bs
            # construct the joint categorical/continuous posterior sample
            XY_p = np.hstack([Y_prob, X_p]).astype(theano.config.floatX)
            # record the sampled label probabilities and one-hots
            prob_samples.append(1.0 * Y_prob)
            label_samples.append(Y_label.astype(np.int32))
            # get next data samples by transforming the prior-space points
            X_d = self.GN.transform_prior(XY_p)
            if binarize:
                X_d = binarize_data(X_d)
        result = {"X_syn": data_samples, "Y_syn": label_samples, "Y_p": prob_samples}
        return result

    def sample_from_prior(self, samp_count):
        """
        Draw independent samples from the model's prior. Use the gaussian
        continuous prior of the underlying GenNet, combined with uniform
        samples for the categorical variable's prior.
        """
        Zs = self.GN.sample_from_prior(samp_count).astype(theano.config.floatX)
        Zs = Zs[:,self.label_dim:]
        Yc = npr.randint(0,high=self.label_dim,size=(samp_count,1)).astype(np.int32)
        Ys = one_hot_np(Yc, cat_dim=self.label_dim).astype(theano.config.floatX)
        Xs = self.GN.transform_prior(np.hstack([Ys, Zs]))
        return Xs

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
