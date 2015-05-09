#############################################################################
# Code for managing and training a variational Iterative Refinement Model.  #
# This one does some sort of semi-supervised learning things.               #
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
                      apply_mask, to_fX, safe_softmax
from InfNet import InfNet
from PeaNet import PeaNet
from DKCode import get_adam_updates, get_adadelta_updates
from LogPDFs import log_prob_bernoulli, log_prob_gaussian2, gaussian_kld

####################
# IMPLEMENTANATION #
####################

class ClassModel(object):
    """
    Controller for training a fancy pseudo-bayesian classifier.

    Parameters:
        rng: numpy.random.RandomState (for reproducibility)
        x_in: the input data to encode
        y_in: int labels >= 1 for x_in when available, otherwise 0.
        q_z_given_x: InfNet for z given x
        class_count: number of classes to classify into
        z_dim: dimension of the "initial" latent space
        use_samples: whether to use z samples or just z mean
    """
    def __init__(self, rng=None, \
            x_in=None, y_in=None, \
            q_z_given_x=None, \
            class_count=None, \
            z_dim=None, \
            use_samples=None):
        # setup a rng for this GIPair
        self.rng = RandStream(rng.randint(100000))

        # record the dimensions of various spaces relevant to this model
        self.class_count = class_count
        self.z_dim = z_dim
        self.shared_dim = q_z_given_x.shared_layers[-1].out_dim
        self.use_samples = use_samples

        # grab handles to the relevant InfNets
        self.q_z_given_x = q_z_given_x

        # record the symbolic variables that will provide inputs to the
        # computation graph created to describe this MultiStageModel
        self.x_in = x_in
        self.y_in = y_in

        # setup switching variable for changing between sampling/training
        zero_ary = to_fX( np.zeros((1,)) )
        # setup a variable for controlling dropout noise
        self.drop_rate = theano.shared(value=zero_ary, name='cm_drop_rate')
        self.set_drop_rate(0.0)

        # initialize classification layer parameters
        init_mat = to_fX(0.01 * npr.randn(self.shared_dim, self.class_count))
        init_vec = to_fX( np.zeros((self.class_count,)) )
        self.W_class = theano.shared(value=init_mat, name='cm_W_class')
        self.b_class = theano.shared(value=init_vec, name='cm_b_class')
        # initialize "optimizable" parameters specific to this CM
        init_vec = to_fX( np.zeros((self.z_dim,)) )
        self.p_z_mean = theano.shared(value=init_vec, name='cm_p_z_mean')
        self.p_z_logvar = theano.shared(value=init_vec, name='cm_p_z_logvar')

        #################
        # Setup self.z. #
        #################
        self.q_z_mean, self.q_z_logvar, self.q_z_samples = \
                self.q_z_given_x.apply(self.x_in, do_samples=True)
        self.q_z_samples = self.q_z_given_x.apply_shared(self.x_in)

        # get a drop mask that drops things with probability p
        drop_scale = 1. / (1. - self.drop_rate[0])
        drop_rnd = self.rng.uniform(size=self.q_z_samples.shape, \
                low=0.0, high=1.0, dtype=theano.config.floatX)
        drop_mask = drop_scale * (drop_rnd > self.drop_rate[0])

        # get a droppy version of either z mean or z samples
        # if self.use_samples:
        #     self.z = self.q_z_samples * drop_mask
        # else:
        #     self.z = self.q_z_mean * drop_mask
        self.z = self.q_z_samples * drop_mask

        # compute class predictions
        self.y_out = T.dot(self.z, self.W_class) + self.b_class

        # compute KLds for training via variational free-energy
        self.kld_z_q2ps = gaussian_kld(self.q_z_mean, self.q_z_logvar, \
                                       self.p_z_mean, self.p_z_logvar)
        self.kld_z_p2qs = gaussian_kld(self.p_z_mean, self.p_z_logvar, \
                                       self.q_z_mean, self.q_z_logvar)

        ######################################################################
        # ALL SYMBOLIC VARS NEEDED FOR THE OBJECTIVE SHOULD NOW BE AVAILABLE #
        ######################################################################

        # shared var learning rate for generator and inferencer
        zero_ary = to_fX( np.zeros((1,)) )
        self.lr_1 = theano.shared(value=zero_ary, name='cm_lr_1')
        self.lr_2 = theano.shared(value=zero_ary, name='cm_lr_2')
        # shared var momentum parameters for generator and inferencer
        self.mom_1 = theano.shared(value=zero_ary, name='cm_mom_1')
        self.mom_2 = theano.shared(value=zero_ary, name='cm_mom_2')
        # init parameters for controlling learning dynamics
        self.set_sgd_params()
        # init shared var for weighting nll of data given posterior sample
        self.lam_nll = theano.shared(value=zero_ary, name='cm_lam_nll')
        self.set_lam_nll(lam_nll=1.0)
        # init shared var for weighting prior kld against reconstruction
        self.lam_kld_q2p = theano.shared(value=zero_ary, name='cm_lam_kld_q2p')
        self.lam_kld_p2q = theano.shared(value=zero_ary, name='cm_lam_kld_p2q')
        self.set_lam_kld(lam_kld_q2p=0.9, lam_kld_p2q=0.0)
        # init shared var for controlling l2 regularization on params
        self.lam_l2w = theano.shared(value=zero_ary, name='cm_lam_l2w')
        self.set_lam_l2w(1e-5)

        # Grab all of the "optimizable" parameters
        self.joint_params = [self.p_z_mean, self.p_z_logvar, \
                             self.W_class, self.b_class]
        self.joint_params.extend(self.q_z_given_x.mlp_params)

        #################################
        # CONSTRUCT THE NLL-BASED COSTS #
        #################################
        self.nll_costs = self.lam_nll[0] * self._construct_nll_costs(self.y_in)
        self.nll_cost = T.mean(self.nll_costs)
        #################################
        # CONSTRUCT THE KLD-BASED COSTS #
        #################################
        self.kld_z_q2p, self.kld_z_p2q = self._construct_kld_costs(p=1.0)
        self.kld_costs = (self.lam_kld_q2p[0] * self.kld_z_q2p) + \
                         (self.lam_kld_p2q[0] * self.kld_z_p2q)
        self.kld_cost = T.mean(self.kld_costs)
        ##################################
        # CONSTRUCT THE FINAL JOINT COST #
        ##################################
        param_reg_cost = self._construct_reg_costs()
        self.reg_cost = self.lam_l2w[0] * param_reg_cost
        self.joint_cost = self.nll_cost + self.kld_cost + self.reg_cost
        ##############################
        # CONSTRUCT A PER-INPUT COST #
        ##############################
        self.obs_costs = self.nll_costs + self.kld_costs

        # Get the gradient of the joint cost for all optimizable parameters
        print("Computing gradients of self.joint_cost...")
        self.joint_grads = OrderedDict()
        grad_list = T.grad(self.joint_cost, self.joint_params)
        for i, p in enumerate(self.joint_params):
            self.joint_grads[p] = grad_list[i]

        # Construct the updates for the model parameters
        self.joint_updates = get_adam_updates(params=self.joint_params, \
                grads=self.joint_grads, alpha=self.lr_1, \
                beta1=self.mom_1, beta2=self.mom_2, \
                mom2_init=1e-3, smoothing=1e-4, max_grad_norm=10.0)

        # Construct a function for jointly training the generator/inferencer
        print("Compiling training function...")
        self.train_joint = self._construct_train_joint()
        print("Compiling class error estimator...")
        self.class_error = self._construct_class_error()
        print("Compiling free-energy sampler...")
        self.compute_fe_terms = self._construct_compute_fe_terms()
        # make easy access points for some interesting parameters
        self.inf_weights = self.q_z_given_x.shared_layers[0].W
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

    def set_lam_kld(self, lam_kld_q2p=1.0, lam_kld_p2q=1.0):
        """
        Set the relative weight of various KL-divergences.
        """
        zero_ary = np.zeros((1,))
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

    def set_drop_rate(self, drop_rate=0.0):
        """
        Set the weight for shaping penalty on conditional priors over zt.
        """
        zero_ary = np.zeros((1,))
        new_val = zero_ary + drop_rate
        self.drop_rate.set_value(to_fX(new_val))
        return

    def _construct_nll_costs(self, yi):
        """
        Construct the categorical log-likelihood part of cost.
        """
        y_prob = safe_softmax(self.y_out)
        row_idx = T.arange(yi.shape[0])
        col_idx = yi.flatten() - 1
        row_mask = T.neq(yi, 0).reshape((yi.shape[0], 1))
        wacky_mat = (y_prob * row_mask) + (1. - row_mask)
        flat_nlls = -T.log(wacky_mat[row_idx,col_idx])
        class_nlls = flat_nlls.reshape((yi.shape[0], 1))
        return class_nlls

    def _construct_kld_costs(self, p=1.0):
        """
        Construct the z KLd part of cost.
        """
        kld_z_q2p = T.sum(self.kld_z_q2ps**p, axis=1, keepdims=True)
        kld_z_p2q = T.sum(self.kld_z_p2qs**p, axis=1, keepdims=True)
        return kld_z_q2p, kld_z_p2q

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
        inputs = [self.x_in, self.y_in]
        # collect the outputs to return from this function
        outputs = [self.joint_cost, self.nll_cost, self.kld_cost, \
                   self.reg_cost, self.obs_costs]
        # compile the theano function
        func = theano.function(inputs=inputs, outputs=outputs, \
                               updates=self.joint_updates)
        return func

    def _construct_compute_fe_terms(self):
        """
        Construct a function for computing terms in variational free energy.
        """
        # construct values to output
        nll = self._construct_nll_costs(self.y_in)
        kld = self.kld_z_q2p
        # compile theano function for a one-sample free-energy estimate
        fe_term_sample = theano.function(inputs=[self.x_in, self.y_in], \
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

    def _construct_class_error(self):
        """
        Compute classification error for a set of observations xi with known
        labels yi, based on multiple passes through noisy initial model.
        """
        # make a function for computing self.y_out
        y_func = theano.function([self.x_in], outputs=self.y_out)
        def multi_sample_error(xi, yi, samples=20):
            # compute self.y_out for the observations in xi
            xi = to_fX(xi)
            yp = y_func(xi)
            for i in range(samples-1):
                yp += y_func(xi)
            yp = yp / float(samples)
            # get the implied class labels
            yc = np.argmax(yp, axis=1).flatten()
            yi = yi.flatten()
            mask = 1.0 * (yi != 0)
            yi = yi - 1
            # compute the classification error for points with valid labels
            err_rate = np.sum(((yi != yc) * mask)) / np.sum(mask)
            return err_rate, yp
        return multi_sample_error

if __name__=="__main__":
    print("Hello world!")







##############
# EYE BUFFER #
##############
