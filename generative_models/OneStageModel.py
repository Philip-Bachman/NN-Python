################################################################
# Code for managing and training a generator/inferencer pair.  #
################################################################

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
from HelperFuncs import apply_mask, to_fX
from DKCode import get_adam_updates, get_adadelta_updates
from LogPDFs import log_prob_bernoulli, log_prob_gaussian2, gaussian_kld


##################
# Implementation #
##################

class OneStageModel(object):
    """
    Controller for training a basic one step VAE.

    Parameters:
        rng: numpy.random.RandomState (for reproducibility)
        x_in: symbolic "data" input to this VAE
        p_x_given_z: HydraNet for x given z
        q_z_given_x: InfNet for z given x
        x_dim: dimension of the "observation" variables
        z_dim: dimension of the "latent" variables
        params:
            x_type: can be "bernoulli" or "gaussian"
            xt_transform: optional transform for gaussian means
            logvar_bound: optional bound on gaussian output logvar
                          -- this can be useful for preventing output
                             distributions that are too "spiky" and thus
                             destabilize training.
    """
    def __init__(self, rng=None, x_in=None, \
            p_x_given_z=None, q_z_given_x=None, \
            x_dim=None, z_dim=None, \
            params=None):
        # setup a rng for this GIPair
        self.rng = RandStream(rng.randint(100000))

        # grab the user-provided parameters
        if params is None:
            self.params = {}
        else:
            self.params = params
        if 'xt_transform' in self.params:
            assert((self.params['xt_transform'] == 'sigmoid') or \
                    (self.params['xt_transform'] == 'none'))
            if self.params['xt_transform'] == 'sigmoid':
                self.xt_transform = lambda x: T.nnet.sigmoid(x)
            else:
                self.xt_transform = lambda x: x
        else:
            self.xt_transform = lambda x: T.nnet.sigmoid(x)
        if 'logvar_bound' in self.params:
            self.logvar_bound = self.params['logvar_bound']
        else:
            self.logvar_bound = 10.0
        #
        # x_type: this tells if we're using bernoulli or gaussian model for
        #         the observations
        #
        self.x_type = self.params['x_type']
        assert((self.x_type == 'bernoulli') or (self.x_type == 'gaussian'))

        # record the dimensions of various spaces relevant to this model
        self.z_dim = z_dim
        self.x_dim = x_dim

        # set parameters for the isotropic Gaussian prior over z
        self.prior_mean = 0.0
        self.prior_logvar = 0.0

        # record the symbolic variables that will provide inputs to the
        # computation graph created to describe this OneStageModel
        self.x_in = x_in
        
        #####################################################################
        # Setup the computation graph that provides values in our objective #
        #####################################################################
        # inferencer model for latent variables given observations
        self.q_z_given_x = q_z_given_x
        self.z_mean, self.z_logvar, self.z = \
                self.q_z_given_x.apply(self.x_in, do_samples=True)
        # generator model for observations given latent variables
        self.p_x_given_z = p_x_given_z
        outputs = self.p_x_given_z.apply(self.z)
        self.xt = outputs[-1]

        # construct the final output of generator, conditioned on z
        if self.x_type == 'bernoulli':
            self.xg = T.nnet.sigmoid(self.xt)
        else:
            self.xg = self.xt_transform(self.xt)

        # self.output_logvar modifies the output distribution
        self.output_logvar = self.p_x_given_z.output_layers[0].b
        self.bounded_logvar = self.logvar_bound * \
                    T.tanh(self.output_logvar[0] / self.logvar_bound)

        ######################################################################
        # ALL SYMBOLIC VARS NEEDED FOR THE OBJECTIVE SHOULD NOW BE AVAILABLE #
        ######################################################################

        # shared var learning rate for generator and inferencer
        zero_ary = to_fX( np.zeros((1,)) )
        self.lr = theano.shared(value=zero_ary, name='osm_lr')
        # shared var momentum parameters for generator and inferencer
        self.mom_1 = theano.shared(value=zero_ary, name='osm_mom_1')
        self.mom_2 = theano.shared(value=zero_ary, name='osm_mom_2')
        # init parameters for controlling learning dynamics
        self.set_sgd_params()
        # init shared var for weighting nll of data given posterior sample
        self.lam_nll = theano.shared(value=zero_ary, name='osm_lam_nll')
        self.set_lam_nll(lam_nll=1.0)
        # init shared var for weighting controlling KL(q(z|x) || p(z))
        self.lam_kld_1 = theano.shared(value=zero_ary, name='osm_lam_kld_1')
        self.lam_kld_2 = theano.shared(value=zero_ary, name='osm_lam_kld_2')
        self.kld_z_mean = theano.shared(value=zero_ary, name='osm_kld_z_mean')
        self.set_lam_kld(lam_kld_1=1.0, lam_kld_2=0.0)
        # init shared var for controlling l2 regularization on params
        self.lam_l2w = theano.shared(value=zero_ary, name='osm_lam_l2w')
        self.set_lam_l2w(1e-4)

        # grab a list of all the parameters to optimize
        self.joint_params = []
        self.joint_params.extend(self.q_z_given_x.mlp_params)
        self.joint_params.extend(self.p_x_given_z.mlp_params)

        ###################################
        # CONSTRUCT THE COSTS TO OPTIMIZE #
        ###################################
        # first, do NLL
        self.nll_costs = self.lam_nll[0] * self._construct_nll_costs()
        self.nll_cost = T.mean(self.nll_costs)
        # second, do KLd
        self.kld_costs_1, self.kld_costs_2 = self._construct_kld_costs()
        self.kld_costs = (self.lam_kld_1[0] * self.kld_costs_1) + \
                         (self.lam_kld_2[0] * self.kld_costs_2)
        self.kld_cost = T.mean(self.kld_costs)
        self.kld_z_mean_new = (0.95 * self.kld_z_mean) + \
                              (0.05 * T.mean(self.kld_costs_1))
        # third, do regularization
        self.reg_cost = self.lam_l2w[0] * self._construct_reg_costs()
        # finally, combine them for the joint cost.
        self.joint_cost = self.nll_cost + self.kld_cost + self.reg_cost

        # Get the gradient of the joint cost for all optimizable parameters
        print("Computing gradients of self.joint_cost...")
        self.joint_grads = OrderedDict()
        grad_list = T.grad(self.joint_cost, self.joint_params, \
                           consider_constant=[self.kld_z_mean])
        for i, p in enumerate(self.joint_params):
            self.joint_grads[p] = grad_list[i]

        # Construct the updates for the generator and inferencer networks
        self.joint_updates = get_adam_updates(params=self.joint_params, \
                grads=self.joint_grads, alpha=self.lr, \
                beta1=self.mom_1, beta2=self.mom_2, \
                mom2_init=1e-3, smoothing=1e-4, max_grad_norm=10.0)
        self.joint_updates[self.kld_z_mean] = self.kld_z_mean_new

        # Construct a function for jointly training the generator/inferencer
        print("Compiling self.train_joint...")
        self.train_joint = self._construct_train_joint()
        print("Compiling self.compute_fe_terms...")
        self.compute_fe_terms = self._construct_compute_fe_terms()
        print("Compiling self.compute_post_klds...")
        self.compute_post_klds = self._construct_compute_post_klds()
        print("Compiling self.sample_from_prior...")
        self.sample_from_prior = self._construct_sample_from_prior()
        self.transform_x_to_z = theano.function(inputs=[self.x_in], \
                                                outputs=self.z_mean)
        self.transform_z_to_x = theano.function(inputs=[self.z], \
                                                outputs=self.xg)
        self.inf_weights = self.q_z_given_x.shared_layers[0].W
        self.gen_weights = self.p_x_given_z.output_layers[-1].W
        return

    def set_sgd_params(self, lr=0.01, mom_1=0.9, mom_2=0.999):
        """
        Set learning rate and momentum parameter for all updates.
        """
        zero_ary = np.zeros((1,))
        # set learning rates
        new_lr = zero_ary + lr
        self.lr.set_value(to_fX(new_lr))
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

    def set_lam_kld(self, lam_kld_1=1.0, lam_kld_2=0.0):
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

    def set_kld_z_mean(self, x):
        """
        Compute mean of KL(q(z|x) || p(z)) for the observations in x, and
        then use it to reset self.kld_z_mean.
        """
        nll, kld = self.compute_fe_terms(x, 10)
        old_mean = self.kld_z_mean.get_value(borrow=False)
        new_mean = (0.0 * old_mean) + np.mean(kld)
        self.kld_z_mean.set_value(to_fX(new_mean))
        return

    def _construct_nll_costs(self):
        """
        Construct the negative log-likelihood part of cost to minimize.
        """
        if self.x_type == 'bernoulli':
            ll_cost = log_prob_bernoulli(self.x_in, self.xg)
        else:
            ll_cost = log_prob_gaussian2(self.x_in, self.xg, \
                    log_vars=self.bounded_logvar)
        nll_cost = -ll_cost
        return nll_cost

    def _construct_kld_costs(self):
        """
        Construct the posterior KL-d from prior part of cost to minimize.
        """
        # compute the KLds between posteriors and priors. we compute the KLd
        # independently for each input and each latent variable dimension
        kld_z = gaussian_kld(self.z_mean, self.z_logvar, \
                             self.prior_mean, self.prior_logvar)
        # compute the batch-wise L1 and L2 penalties on per-dim KLds
        kld_l1_costs = T.sum(kld_z, axis=1, keepdims=True)
        kld_l2_costs = (kld_l1_costs - self.kld_z_mean[0])**2.0
        return [kld_l1_costs, kld_l2_costs]

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
        br = T.lscalar()
        # collect the values to output with each function evaluation
        outputs = [self.joint_cost, self.nll_cost, self.kld_cost, \
                   self.reg_cost, self.nll_costs, self.kld_costs]
        func = theano.function(inputs=[ xi, br ], \
                outputs=outputs, \
                givens={ self.x_in: xi.repeat(br, axis=0) }, \
                updates=self.joint_updates)
        return func

    def _construct_compute_post_klds(self):
        """
        Construct theano function to compute the info about the variational
        approximate posteriors for some inputs.
        """
        # setup some symbolic variables for theano to deal with
        all_klds = gaussian_kld(self.z_mean, self.z_logvar, \
                                self.prior_mean, self.prior_logvar)
        # compile theano function for a one-sample free-energy estimate
        kld_func = theano.function(inputs=[self.x_in], outputs=all_klds)
        return kld_func

    def _construct_compute_fe_terms(self):
        """
        Construct theano function to compute the log-likelihood and posterior
        KL-divergence terms for the variational free-energy.
        """
        # construct values to output
        if self.x_type == 'bernoulli':
            ll_term = log_prob_bernoulli(self.x_in, self.xg)
        else:
            ll_term = log_prob_gaussian2(self.x_in, self.xg, \
                    log_vars=self.bounded_logvar)
        all_klds = gaussian_kld(self.z_mean, self.z_logvar, \
                                self.prior_mean, self.prior_logvar)
        kld_term = T.sum(all_klds, axis=1)
        # compile theano function for a one-sample free-energy estimate
        fe_term_sample = theano.function(inputs=[self.x_in], \
                                         outputs=[ll_term, kld_term])
        # construct a wrapper function for multi-sample free-energy estimate
        def fe_term_estimator(X, sample_count):
            X = to_fX(X)
            ll_sum = np.zeros((X.shape[0],))
            kld_sum = np.zeros((X.shape[0],))
            for i in range(sample_count):
                result = fe_term_sample(X)
                ll_sum = ll_sum + result[0].ravel()
                kld_sum = kld_sum + result[1].ravel()
            mean_nll = -ll_sum / float(sample_count)
            mean_kld = kld_sum / float(sample_count)
            return [mean_nll, mean_kld]
        return fe_term_estimator

    def _construct_sample_from_prior(self):
        """
        Construct a function for drawing independent samples from the
        distribution generated by this OneStageModel.
        """
        z_sym = T.matrix()
        oputs = self.xg
        sample_func = theano.function(inputs=[z_sym], outputs=oputs, \
                                      givens={ self.z: z_sym })
        def prior_sampler(samp_count):
            z_samps = npr.randn(samp_count, self.z_dim)
            z_samps = (np.exp(0.5 * self.prior_logvar) * z_samps) + \
                      self.prior_mean
            z_samps =to_fX(z_samps)
            model_samps = sample_func(z_samps)
            return model_samps
        return prior_sampler

    def sample_from_chain(self, X_d, X_c=None, X_m=None, loop_iters=5, \
                          sigma_scale=None):
        """
        Sample for several rounds through the I<->G loop, initialized with the
        the "data variable" samples in X_d.
        """
        data_samples = []
        prior_samples = []
        if X_c is None:
            X_c = 0.0 * X_d
        if X_m is None:
            X_m = 0.0 * X_d
        if sigma_scale is None:
            sigma_scale = 1.0
        # set sigma_scale on our InfNet
        old_scale = self.q_z_given_x.sigma_scale.get_value(borrow=False)
        self.q_z_given_x.set_sigma_scale(sigma_scale)
        for i in range(loop_iters):
            # apply mask, mixing foreground and background data
            X_d = apply_mask(Xd=X_d, Xc=X_c, Xm=X_m)
            # record the data samples for this iteration
            data_samples.append(1.0 * X_d)
            # sample from their inferred posteriors
            X_p = self.q_z_given_x.sample_posterior(X_d)
            # record the sampled points (in the "prior space")
            prior_samples.append(1.0 * X_p)
            # get next data samples by transforming the prior-space points
            X_d = self.transform_z_to_x(X_p)
        # reset sigma_scale on our InfNet
        self.q_z_given_x.set_sigma_scale(old_scale[0])
        result = {"data samples": data_samples, "prior samples": prior_samples}
        return result

if __name__=="__main__":
    print("TESTING COMPLETE!")







##############
# EYE BUFFER #
##############