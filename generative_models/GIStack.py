################################################################################
# Code for managing and training a triplet system comprising:                  #
#   1. an "observation" generator conditioned on some latent variables         #
#   2. a "label" generator conditioned on some latent variables                #
#   3. an inferencer for estimating posteriors over the latent variables given #
#      an observation.                                                         #
#                                                                              #
# -- One might call this a "variational multi-encoder"...                      #
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
from NetLayers import relu_actfun, softplus_actfun, \
                      safe_softmax, safe_log
from DKCode import get_adam_updates, get_adadelta_updates
from GenNet import GenNet
from InfNet import InfNet
from PeaNet import PeaNet

######################################################
# HELPER FUNCTIONS FOR PEAR AND CLASSIFICATION COSTS #
######################################################

def smooth_softmax(p, lam_smooth=1e-3):
    """
    Give a "smoothed" softmax distribution form for p. This is to counter
    the tendency for KL-divergence, cross-entropy, etc. to get a bit wonky
    when comparing strongly peaked categorical distributions.
    """
    dist_size = T.cast(p.shape[1], 'floatX')
    p_sm = (safe_softmax(p) + lam_smooth) / (1.0 + (dist_size * lam_smooth))
    return p_sm

def smooth_entropy(p, lam_smooth=1e-3):
    """
    Measure the entropy of distribution p after converting it from log-space
    into a normalized, sum-to-one distribution.
    """
    dist_size = T.cast(p.shape[1], 'floatX')
    p_sm = smooth_softmax(p, lam_smooth=lam_smooth)
    ent_sm = -T.sum((safe_log(p_sm) * p_sm), axis=1, keepdims=True)
    return ent_sm

def smooth_kl_divergence(p, q, lam_smooth=1e-3):
    """
    Measure the KL-divergence from "approximate" distribution q to "true"
    distribution p. Use smoothed softmax to convert p and q from log-space
    encodings into normalized, sum-to-one distributions.
    """
    dist_size = T.cast(p.shape[1], 'floatX')
    p_sm = smooth_softmax(p, lam_smooth=lam_smooth)
    q_sm = smooth_softmax(q, lam_smooth=lam_smooth)
    # This term is: cross_entropy(p, q) - entropy(p)
    kl_sm = T.sum(((safe_log(p_sm) - safe_log(q_sm)) * p_sm), axis=1, keepdims=True)
    return kl_sm

def smooth_cross_entropy(p, q):
    """
    Measure the cross-entropy between "approximate" distribution q and
    "true" distribution p. Use smoothed softmax to convert p and q from
    log-space encodings into normalized, sum-to-one distributions.
    """
    dist_size = T.cast(p.shape[1], 'floatX')
    p_sm = smooth_softmax(p, lam_smooth=lam_smooth)
    q_sm = smooth_softmax(q, lam_smooth=lam_smooth)
    # This term is: entropy(p) + kl_divergence(p, q)
    ce_sm = -T.sum((p_sm * safe_log(q_sm)), axis=1, keepdims=True)
    return ce_sm

def cat_entropy(p):
    """
    Compute the entropy of (row-wise) categorical distributions given by
    taking the softmax transform of p.
    """
    p_sm = smooth_softmax(p, lam_smooth=1e-4)
    row_ents = -T.sum((p_sm * safe_log(p_sm)), axis=1, keepdims=True)
    return row_ents

def cat_cost_xent(Yp, Yd, lam_smooth=1e-4):
    """
    Compute cross-entropy classification cost given the target labels in Yd
    and the log-space predictions in Yp. Unsupervised points are indicated by
    Yd[i] == 0, and labels for supervised points need to be shifted -1.
    """
    Yp_sm = smooth_softmax(Yp, lam_smooth=lam_smooth)
    # construct a mask that zeros-out unsupervised rows
    row_idx = T.arange(Yd.shape[0])
    row_mask = T.neq(Yd, 0).reshape((Yd.shape[0], 1))
    # make a matrix that has the predicted class probs for supervised
    # inputs' rows and is all ones for rows that are unsupervised
    wacky_mat = (Yp_sm * row_mask) + (1. - row_mask)
    # compute cross-entropy (classification) loss
    cat_cost = -T.sum(safe_log(wacky_mat[row_idx,(Yd.flatten()-1)])) \
            / (T.sum(row_mask) + 1e-4)
    return cat_cost

def one_hot(Yc, cat_dim=None):
    """
    Given a tensor Yc of dimension d with integer values from range(cat_dim),
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

#
#
# Important symbolic variables:
#   Xd: Xd represents input at the "data variables" of the inferencer
#   Yd: Yd represents label information for use in semi-supervised learning
#   Xc: Xc represents input at the "control variables" of the inferencer
#   Xm: Xm represents input at the "mask variables" of the inferencer
#
#

class GIStack(object):
    """
    Controller for training a variational encoder for joint decoding into an
    observation and a label.

    The generator for modeling conditional distributions over observations
    given the latent variables must be an instance of the GenNet class
    implemented in "GenNet.py". The inferencer for inferring conditional
    posteriors over the latent variables must be an instance of the InfNet
    class implemented in "InfNet.py". The "classifier" for modeling conditional
    distributions over a label set given the latent variables must be an
    instance of the PeaNet class implemented in "PeaNet.py".

    Parameters:
        rng: numpy.random.RandomState (for reproducibility)
        Xd: symbolic "data" input to this VME
        Yd: symbolic "label" input to this VME
        Xc: symbolic "control" input to this VME
        Xm: symbolic "mask" input to this VME
        g_net: The GenNet instance for observation given latent
        i_net: The InfNet instance for latent given observation
        p_net: The PeaNet instance for label given latent
        data_dim: dimension of the observation variables
        prior_dim: dimension of the latent variables
        label_dim: cardinality of the categorical label variable
        params: dict for passing additional parameters
    """
    def __init__(self, rng=None, \
            Xd=None, Yd=None, Xc=None, Xm=None, \
            g_net=None, i_net=None, p_net=None, \
            data_dim=None, prior_dim=None, label_dim=None, \
            params=None):
        # TODO: refactor for use with "encoded" inferencer/generator
        assert(not (i_net.use_encoder or g_net.use_decoder))

        # setup a rng for this GIStack
        self.rng = RandStream(rng.randint(100000))
        # record the symbolic variables that will provide inputs to the
        # computation graph created for this GIStack
        self.Xd = Xd
        self.Yd = Yd
        self.Xc = Xc
        self.Xm = Xm
        self.Xd2 = T.vertical_stack(self.Xd, self.Xd)
        self.Yd2 = T.vertical_stack(self.Yd, self.Yd)
        self.Xc2 = T.vertical_stack(self.Xc, self.Xc)
        self.Xm2 = T.vertical_stack(self.Xm, self.Xm)
        self.obs_count = T.cast(self.Xd2.shape[0], 'floatX')
        # record the dimensionality of the data handled by this GIStack
        self.data_dim = data_dim
        self.label_dim = label_dim
        self.prior_dim = prior_dim
        # create a "shared-parameter" clone of the latent inferencer
        self.IN2 = i_net.shared_param_clone(rng=rng, \
                Xd=self.Xd2, Xc=self.Xc2, Xm=self.Xm2)
        # capture a handle for latent samples from the inferencer
        self.Xp2 = self.IN2.output
        # feed it into a shared-parameter clone of the generator
        self.GN2 = g_net.shared_param_clone(rng=rng, Xp=self.Xp2)
        # capture a handle for outputs from the observation generator
        self.Xg2 = self.GN2.output
        # and feed it into a shared-parameter clone of the label generator
        self.PN2 = p_net.shared_param_clone(rng=rng, Xd=self.Xp2)
        # capture handles for noisy/clean outputs of the label generator
        self.Yp2 = self.PN2.output_spawn[0] # noisy predictions
        self.Yp2_proto = self.PN2.output_proto # noise-free predictions

        # we require the PeaNet to have one proto-net and one spawn net
        assert(len(self.PN2.proto_nets) == 1)
        assert(len(self.PN2.spawn_nets) == 1)
        # check that all networks agree on the latent variable dimension
        assert(self.prior_dim == self.IN2.mu_layers[-1].out_dim)
        assert(self.prior_dim == self.IN2.sigma_layers[-1].out_dim)
        assert(self.prior_dim == self.GN2.mlp_layers[0].in_dim)
        assert(self.prior_dim == self.PN2.proto_nets[0][0].in_dim)
        # check that we've been told the correct cardinality for the
        # categorical variable we will be "decoding"
        assert(self.label_dim == self.PN2.proto_nets[0][-1].out_dim)

        zero_ary = np.zeros((1,)).astype(theano.config.floatX)
        # shared var learning rates for all networks
        self.lr_gn = theano.shared(value=zero_ary, name='gis_lr_gn')
        self.lr_in = theano.shared(value=zero_ary, name='gis_lr_in')
        self.lr_pn = theano.shared(value=zero_ary, name='gis_lr_pn')
        # shared var momentum parameters for all networks
        self.mom_1 = theano.shared(value=zero_ary, name='gis_mom_1')
        self.mom_2 = theano.shared(value=zero_ary, name='gis_mom_2')
        self.it_count = theano.shared(value=zero_ary, name='gis_it_count')
        # init parameters for controlling learning dynamics
        self.set_all_sgd_params()
        # init shared var for weighting nll of data given posterior sample
        self.lam_nll = theano.shared(value=zero_ary, name='gis_lam_nll')
        self.set_lam_nll(lam_nll=1.0)
        # init shared var for weighting posterior KL-div from prior
        self.lam_kld = theano.shared(value=zero_ary, name='gis_lam_kld')
        self.set_lam_kld(lam_kld=1.0)
        # init shared var for weighting semi-supervised classification
        self.lam_cat = theano.shared(value=zero_ary, name='gis_lam_cat')
        self.set_lam_cat(lam_cat=0.0)
        # init shared var for weighting PEA cost on (un)supervised inputs
        self.lam_pea_su = theano.shared(value=zero_ary, name='gis_lam_pea_su')
        self.lam_pea_un = theano.shared(value=zero_ary, name='gis_lam_pea_un')
        self.set_lam_pea(lam_pea_su=1.0, lam_pea_un=1.0)
        # init shared var for controlling l2 regularization on params
        self.lam_l2w = theano.shared(value=zero_ary, name='gis_lam_l2w')
        self.set_lam_l2w(lam_l2w=1e-3)

        # grab the full set of "optimizable" parameters from the generator
        # and inferencer networks that we'll be working with.
        self.gn_params = [p for p in self.GN2.mlp_params]
        self.in_params = [p for p in self.IN2.mlp_params]
        self.pn_params = [p for p in self.PN2.proto_params]
        self.joint_params = self.pn_params + self.in_params + self.gn_params

        ###################################
        # CONSTRUCT THE COSTS TO OPTIMIZE #
        ###################################
        pea_cost_su, pea_cost_un = self._construct_post_pea_costs()
        self.data_nll_cost = self.lam_nll[0] * self._construct_data_nll_cost()
        self.post_kld_cost = self.lam_kld[0] * self._construct_post_kld_cost()
        self.post_cat_cost = self.lam_cat[0] * self._construct_post_cat_cost()
        self.post_pea_cost = (self.lam_pea_su[0] * pea_cost_su) + \
                (self.lam_pea_un[0] * pea_cost_un)
        self.other_reg_cost = self._construct_other_reg_cost()
        self.joint_cost = self.data_nll_cost + self.post_kld_cost + self.post_cat_cost + \
                self.post_pea_cost + self.other_reg_cost

        # grab the gradients for all parameters to optimize
        self.joint_grads = OrderedDict()
        for p in self.joint_params:
            self.joint_grads[p] = T.grad(self.joint_cost, p).clip(-0.1, 0.1)

        # construct the updates for all parameters to optimize
        self.gn_updates = get_adam_updates(params=self.gn_params, \
                grads=self.joint_grads, alpha=self.lr_gn, \
                beta1=self.mom_1, beta2=self.mom_2, it_count=self.it_count, \
                mom2_init=1e-3, smoothing=1e-8)
        self.in_updates = get_adam_updates(params=self.in_params, \
                grads=self.joint_grads, alpha=self.lr_in, \
                beta1=self.mom_1, beta2=self.mom_2, it_count=self.it_count, \
                mom2_init=1e-3, smoothing=1e-8)
        self.pn_updates = get_adam_updates(params=self.pn_params, \
                grads=self.joint_grads, alpha=self.lr_pn, \
                beta1=self.mom_1, beta2=self.mom_2, it_count=self.it_count, \
                mom2_init=1e-3, smoothing=1e-8)
        #self.gn_updates = get_adadelta_updates(params=self.gn_params, \
        #        grads=self.joint_grads, alpha=self.lr_gn, beta1=0.98)
        #self.in_updates = get_adadelta_updates(params=self.in_params, \
        #        grads=self.joint_grads, alpha=self.lr_in, beta1=0.98)
        #self.pn_updates = get_adadelta_updates(params=self.pn_params, \
        #        grads=self.joint_grads, alpha=self.lr_dn, beta1=0.98)

        # bag up all the updates required for training
        self.joint_updates = OrderedDict()
        for k in self.gn_updates:
            self.joint_updates[k] = self.gn_updates[k]
        for k in self.in_updates:
            self.joint_updates[k] = self.in_updates[k]
        for k in self.pn_updates:
            self.joint_updates[k] = self.pn_updates[k]

        # construct a training function for all parameters. training for the
        # various networks can be switched on and off via learning rates
        self.train_joint = self._construct_train_joint()
        return

    def set_pn_sgd_params(self, learn_rate=0.01):
        """
        Set learning rate for the categorical generator network.
        """
        zero_ary = np.zeros((1,))
        new_lr = zero_ary + learn_rate
        self.lr_pn.set_value(new_lr.astype(theano.config.floatX))
        return

    def set_in_sgd_params(self, learn_rate=0.01):
        """
        Set learning rate for the latent inferencer network.
        """
        zero_ary = np.zeros((1,))
        new_lr = zero_ary + learn_rate
        self.lr_in.set_value(new_lr.astype(theano.config.floatX))
        return

    def set_gn_sgd_params(self, learn_rate=0.01):
        """
        Set learning rate for the observation generator.
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
        self.lr_gn.set_value(new_lr.astype(theano.config.floatX))
        self.lr_in.set_value(new_lr.astype(theano.config.floatX))
        self.lr_pn.set_value(new_lr.astype(theano.config.floatX))
        # set the first/second moment momentum parameters
        new_mom_1 = zero_ary + mom_1
        new_mom_2 = zero_ary + mom_2
        self.mom_1.set_value(new_mom_1.astype(theano.config.floatX))
        self.mom_2.set_value(new_mom_2.astype(theano.config.floatX))
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
        Set the strength of regularization on KL-divergence between inferred
        posteriors over the latent variables and the latent prior.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_kld
        self.lam_kld.set_value(new_lam.astype(theano.config.floatX))
        return

    def set_lam_cat(self, lam_cat=0.0):
        """
        Set weight for controlling the influence of the label likelihood.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_cat
        self.lam_cat.set_value(new_lam.astype(theano.config.floatX))
        return

    def set_lam_pea(self, lam_pea_su=0.0, lam_pea_un=0.0):
        """
        Set the strength of PEA regularization on the label generator.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_pea_su
        self.lam_pea_su.set_value(new_lam.astype(theano.config.floatX))
        new_lam = zero_ary + lam_pea_un
        self.lam_pea_un.set_value(new_lam.astype(theano.config.floatX))
        return

    def set_lam_l2w(self, lam_l2w=1e-3):
        """
        Set the relative strength of l2 regularization on network params.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_l2w
        self.lam_l2w.set_value(new_lam.astype(theano.config.floatX))
        return

    def _construct_data_nll_cost(self):
        """
        Construct the negative log-likelihood part of cost to minimize.
        """
        log_prob_cost = self.GN2.compute_log_prob(self.Xd2)
        nll_cost = -T.sum(log_prob_cost) / self.obs_count
        return nll_cost

    def _construct_post_kld_cost(self):
        """
        Construct the posterior KL-d from prior part of cost to minimize.
        """
        kld_cost = T.sum(self.IN2.kld_cost) / self.obs_count
        return kld_cost

    def _construct_post_cat_cost(self):
        """
        Construct the label prediction cost for inputs with known labels.
        """
        cat_cost = cat_cost_xent(self.Yp2, self.Yd2, lam_smooth=5e-4)
        return cat_cost

    def _construct_post_pea_costs(self):
        """
        Construct the pseudo-ensemble agreement cost on the categorical
        distributions output by the label generator.
        """
        # get the two sets of predictions for the input batch, with each set
        # of predictions given by independent passes through the noisy nets
        b_size = self.Yp2.shape[0] / 2
        idx = T.arange(0, stop=b_size)
        x1 = T.take(self.Yp2, idx, axis=0)
        x2 = T.take(self.Yp2, idx+b_size, axis=0)
        # construct a mask that zeros-out unsupervised rows
        row_idx = T.arange(self.Yd.shape[0])
        row_mask = T.neq(self.Yd, 0).reshape((self.Yd.shape[0], 1))
        # compute PEA reg costs for supervised and unsupervised points
        pea_costs = (smooth_kl_divergence(x1, x2, lam_smooth=5e-3) + \
                smooth_kl_divergence(x2, x1, lam_smooth=5e-3)) / 2.0
        pea_cost_su = T.sum(row_mask * pea_costs) / (T.sum(row_mask) + 1e-4)
        pea_cost_un = T.sum((1.0 - row_mask) * pea_costs) / \
                (T.sum(1.0 - row_mask) + 1e-4)
        pea_costs = [pea_cost_su, pea_cost_un]
        return pea_costs

    def _construct_other_reg_cost(self):
        """
        Construct the cost for low-level basic regularization. E.g. for
        applying l2 regularization to the network activations and parameters.
        """
        act_reg_cost = self.IN2.act_reg_cost + self.GN2.act_reg_cost + \
                self.PN2.act_reg_cost
        gp_cost = sum([T.sum(par**2.0) for par in self.gn_params])
        ip_cost = sum([T.sum(par**2.0) for par in self.in_params])
        pp_cost = sum([T.sum(par**2.0) for par in self.pn_params])
        param_reg_cost = self.lam_l2w[0] * (gp_cost + ip_cost + pp_cost)
        other_reg_cost = (act_reg_cost / self.obs_count) + param_reg_cost
        return other_reg_cost

    def _construct_train_joint(self):
        """
        Construct theano function to train inferencer and generator jointly.
        """
        outputs = [self.joint_cost, self.data_nll_cost, self.post_kld_cost, \
                self.post_cat_cost, self.post_pea_cost, self.other_reg_cost]
        func = theano.function(inputs=[ self.Xd, self.Xc, self.Xm, self.Yd ], \
                outputs=outputs, \
                updates=self.joint_updates)
        return func

    def sample_gis_from_data(self, X_d, loop_iters=10):
        """
        Sample for several rounds through the I<->G loop, initialized with the
        the "data variable" samples in X_d.
        """
        row_count = X_d.shape[0]
        data_samples = []
        prior_samples = []
        label_samples = []
        X_c = 0.0 * X_d
        X_m = 0.0 * X_d
        for i in range(loop_iters):
            # record the data samples for this iteration
            data_samples.append(1.0 * X_d)
            # sample from their inferred posteriors
            X_p = self.IN2.sample_posterior(X_d, X_c, X_m)
            Y_p = self.PN2.sample_posterior(X_p)
            # record the sampled points (in the "prior space")
            prior_samples.append(1.0 * X_p)
            label_samples.append(1.0 * Y_p)
            # get next data samples by transforming the prior-space points
            X_d = self.GN2.transform_prior(X_p)
        result = {"data samples": data_samples, "prior samples": prior_samples, \
                "label samples": label_samples}
        return result

    def classification_error(self, X_d, Y_d, samples=20):
        """
        Compute classification error for a set of observations X_d with known
        labels Y_d, based on multiple samples from its continuous posterior
        (computed via self.IN2), passed through the label generator (self.IN2).
        """
        # first, convert labels to account for semi-supervised labeling
        Y_mask = 1.0 * (Y_d != 0)
        Y_d = Y_d - 1
        # make a function for computing the output of the label generator
        func = theano.function([self.Xd, self.Xc, self.Xm], \
            outputs=self.Yp2_proto)
        input_count = X_d.shape[0]
        X_c = 0.0 * X_d
        X_m = 0.0 * X_d
        # compute the expected output for X_d
        Y_p = None
        for i in range(samples):
            preds = func(X_d, X_c, X_m)
            preds = preds[0:input_count,:]
            if Y_p == None:
                Y_p = preds
            else:
                Y_p += preds
        Y_p = Y_p / float(samples)
        # get the implied class labels
        Y_c = np.argmax(Y_p, axis=1).reshape((input_count, 1))
        # compute the classification error for points with valid labels
        err_rate = np.sum(((Y_d != Y_c) * Y_mask)) / np.sum(Y_mask)
        return err_rate

    def class_probs(self, X_d, samples=20):
        """
        Compute predicted class probabilities for a set of observations X_d
        based on multiple samples from its continuous posterior (computed via
        self.IN2), passed through the label generator (i.e. self.PN2).
        """
        # make a function for computing the output of the label generator
        func = theano.function([self.Xd, self.Xc, self.Xm], \
            outputs=self.Yp2_proto)
        input_count = X_d.shape[0]
        X_c = 0.0 * X_d
        X_m = 0.0 * X_d
        # compute the expected output for X_d
        Y_p = None
        for i in range(samples):
            preds = func(X_d, X_c, X_m)
            preds = preds[0:input_count,:]
            if Y_p == None:
                Y_p = preds
            else:
                Y_p += preds
        Y_p = Y_p / float(samples)
        Y_p = np.exp(Y_p) / np.sum(np.exp(Y_p), axis=1, keepdims=True)
        return Y_p

def mnist_prob_embed(X, Y):
    """
    Embed the predicted class probabilities in Y in the digits images in X.
    """
    obs_count = X.shape[0]
    class_count = Y.shape[1]
    Xy = np.zeros(X.shape)
    for i in range(obs_count):
        x_sq = X[i,:].reshape((28,28))
        for j in range(class_count):
            x_sq[(2*j):(2*j+2),0:3] = Y[i,j]
            x_sq[(2*j):(2*j+2),3] = 0.33
        x_sq[2*class_count,0:3] = 0.33
        Xy[i,:] = x_sq.flatten()
    return Xy

if __name__=="__main__":
    from load_data import load_udm, load_udm_ss, load_mnist
    import utils as utils
    import PeaNet as PNet
    import InfNet as INet
    import GenNet as GNet
    # Initialize a source of randomness
    rng = np.random.RandomState(1234)

    sup_count = 600
    # Load some data to train/validate/test with
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm_ss(dataset, sup_count, rng, zero_mean=False)
    Xtr_su = datasets[0][0].get_value(borrow=False)
    Ytr_su = datasets[0][1].get_value(borrow=False)
    Xtr_un = datasets[1][0].get_value(borrow=False)
    Ytr_un = datasets[1][1].get_value(borrow=False)
    # get the unlabeled data
    Xtr_un = np.vstack([Xtr_su, Xtr_un]).astype(theano.config.floatX)
    Ytr_un = np.vstack([Ytr_su[:,np.newaxis], Ytr_un[:,np.newaxis]]).astype(np.int32)
    Ytr_un = 0 * Ytr_un
    # get the labeled data
    Xtr_su = Xtr_su.astype(theano.config.floatX)
    Ytr_su = Ytr_su[:,np.newaxis].astype(np.int32)
    # get observations and labels for the validation set
    Xva = datasets[2][0].get_value(borrow=False).astype(theano.config.floatX)
    Yva = datasets[2][1].get_value(borrow=False).astype(np.int32)
    Yva = Yva[:,np.newaxis] # numpy is dumb
    # get size information for the data
    un_samples = Xtr_un.shape[0]
    su_samples = Xtr_su.shape[0]
    va_samples = Xva.shape[0]

    # setup more stuff
    Xp = T.matrix('Xp_base')
    Xd = T.matrix('Xd_base')
    Xc = T.matrix('Xc_base')
    Xm = T.matrix('Xm_base')
    Yd = T.icol('Yd_base')
    data_dim = Xtr_un.shape[1]
    label_dim = 10
    prior_dim = 64
    prior_sigma = 1.0
    batch_size = 200

    # load the latent inferencer network and the observation generator
    # network from disk...
    gn_fname = "MMS_RESULTS/pt60k_params_GN.pkl"
    in_fname = "MMS_RESULTS/pt60k_params_IN.pkl"
    IN = INet.load_infnet_from_file(f_name=in_fname, rng=rng, Xd=Xd, Xc=Xc, Xm=Xm)
    GN = GNet.load_gennet_from_file(f_name=gn_fname, rng=rng, Xp=Xp)

    # choose some parameters for the categorical inferencer
    pn_params = {}
    pc0 = [prior_dim, 512, label_dim]
    pn_params['proto_configs'] = [pc0]
    # Set up some spawn networks
    sc0 = {'proto_key': 0, 'input_noise': 0.0, 'bias_noise': 0.1, 'do_dropout': True}
    pn_params['spawn_configs'] = [sc0]
    pn_params['spawn_weights'] = [1.0]
    # Set remaining params
    pn_params['activation'] = relu_actfun
    pn_params['init_scale'] = 0.5
    pn_params['lam_l2a'] = 1e-3
    pn_params['vis_drop'] = 0.0
    pn_params['hid_drop'] = 0.5
    # initialize a PeaNet ith the desired parameters
    PN = PeaNet(rng=rng, Xd=Xd, params=pn_params)
    PN.init_biases(0.1)

    # Initialize the GIStack
    GIS = GIStack(rng=rng, \
            Xd=Xd, Yd=Yd, Xc=Xc, Xm=Xm, \
            g_net=GN, i_net=IN, p_net=PN, \
            data_dim=data_dim, prior_dim=prior_dim, \
            label_dim=label_dim, params=None)
    # set weighting parameters for the various costs...
    GIS.set_lam_nll(1.0)
    GIS.set_lam_kld(1.0)
    GIS.set_lam_cat(1.0)
    GIS.set_lam_pea(lam_pea_su=0.0, lam_pea_un=2.0)
    GIS.set_lam_l2w(1e-4)

    # set initial learning rate and basic SGD hyper parameters
    out_name = 'GIS_TEST_RESULT.txt'

    out_file = open(out_name, 'wb')
    out_file.write("**TODO: More informative output, and maybe a real log**\n")
    out_file.write("sup_count: {0:d}\n".format(sup_count))
    out_file.flush()

    learn_rate = 0.0005
    cost_un = [0. for i in range(20)]
    cost_su = [0. for i in range(20)]
    for i in range(500000):
        # start with some updates only for the VAE (InfNet and GenNet)
        scale = float(min(i+1, 25000)) / 25000.0
        if ((i+1 % 100000) == 0):
            learn_rate = learn_rate * 0.75
        # do a minibatch update using unlabeled data
        if True:
            # get some unlabeled data to train with
            un_idx = npr.randint(low=0,high=un_samples,size=(batch_size,))
            Xd_un = Xtr_un.take(un_idx, axis=0)
            Yd_un = Ytr_un.take(un_idx, axis=0)
            Xc_un = 0.0 * Xd_un
            Xm_un = 0.0 * Xd_un
            # do a minibatch update of the model, and compute some costs
            GIS.set_all_sgd_params(learn_rate=0.0005, mom_1=0.9, mom_2=0.999)
            GIS.set_pn_sgd_params(learn_rate=0.001)
            GIS.set_lam_nll(1.0)
            GIS.set_lam_kld(3.0)
            GIS.set_lam_cat(100.0)
            GIS.set_lam_pea(lam_pea_su=0.0, lam_pea_un=200.0)
            outputs = GIS.train_joint(Xd_un, Xc_un, Xm_un, Yd_un)
            cost_un = [(cost_un[k] + 1.*outputs[k]) for k in range(len(outputs))]
        # do another minibatch update incorporating label information
        if True:
            # get some labeled data to train with
            su_idx = npr.randint(low=0,high=su_samples,size=(batch_size,))
            Xd_su = Xtr_su.take(su_idx, axis=0)
            Yd_su = Ytr_su.take(su_idx, axis=0)
            Xc_su = 0.0 * Xd_su
            Xm_su = 0.0 * Xd_su
            # update only based on the label-based classification cost
            GIS.set_all_sgd_params(learn_rate=0.0005, mom_1=0.9, mom_2=0.999)
            GIS.set_pn_sgd_params(learn_rate=0.001)
            GIS.set_lam_nll(0.01) # turn down observation generation term
            GIS.set_lam_kld(0.03) # turn down latent posterior kld term
            GIS.set_lam_cat(100.0)
            GIS.set_lam_pea(lam_pea_su=0.0, lam_pea_un=200.0)
            outputs = GIS.train_joint(Xd_su, Xc_su, Xm_su, Yd_su)
            cost_su = [(cost_su[k] + 1.*outputs[k]) for k in range(len(outputs))]
        if ((i % 500) == 0):
            cost_un = [(v / 500.) for v in cost_un]
            cost_su = [(v / 500.) for v in cost_su]
            o_str_un = "batch: {0:d}, joint: {1:.4f}, nll: {2:.4f}, kld: {3:.4f}, cat: {4:.4f}, pea: {5:.4f}, other_reg: {6:.4f}".format( \
                    i, cost_un[0], cost_un[1], cost_un[2], cost_un[3], cost_un[4], cost_un[5])
            o_str_su = "-----         joint: {1:.4f}, nll: {2:.4f}, kld: {3:.4f}, cat: {4:.4f}, pea: {5:.4f}, other_reg: {6:.4f}".format( \
                    i, cost_su[0], cost_su[1], cost_su[2], cost_su[3], cost_su[4], cost_su[5])
            print(o_str_un)
            print(o_str_su)
            out_file.write("{}\n".format(o_str_un))
            out_file.write("{}\n".format(o_str_su))
            if ((i % 2000) == 0):
                # check classification error on training and validation set
                train_err = GIS.classification_error(Xtr_su, Ytr_su)
                va_err = GIS.classification_error(Xva, Yva)
                o_str = "    tr_err: {0:.4f}, va_err: {1:.4f}".format(train_err, va_err)
                print(o_str)
                out_file.write("{}\n".format(o_str))
            out_file.flush()
        if ((i % 5000) == 0):
            file_name = "GIS_SAMPLES_b{0:d}.png".format(i)
            tr_idx = npr.randint(low=0,high=un_samples,size=(5,))
            va_idx = npr.randint(low=0,high=va_samples,size=(5,))
            Xd_samps = np.vstack([Xtr_un[tr_idx,:], Xva[va_idx,:]])
            Xd_samps = np.repeat(Xd_samps, 3, axis=0)
            sample_lists = GIS.sample_gis_from_data(Xd_samps, loop_iters=20)
            Xs = np.vstack(sample_lists["data samples"])
            Ys = GIS.class_probs(Xs)
            Xs = mnist_prob_embed(Xs, Ys)
            utils.visualize_samples(Xs, file_name, num_rows=20)
    out_file.close()
    print("TESTING COMPLETE!")




##############
# EYE BUFFER #
##############

