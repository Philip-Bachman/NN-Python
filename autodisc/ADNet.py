############################################################
# Semi-supervised DEV-regularized multilayer perceptron.   #
# -- Now with 100% more denoising autoencoding!            #
############################################################

import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse
import theano.tensor.shared_randomstreams

from output_losses import LogRegSS, MCL2HingeSS

#####################################################################
# NON-LINEARITIES: Some activation functions, for your convenience. #
#####################################################################

def row_normalize(x):
    """Normalize rows of matrix x to unit (L2) length."""
    x_normed = x / T.sqrt(T.sum(x**2.,axis=1,keepdims=1)+1e-6)
    return x_normed

def col_normalize(x):
    """Normalize cols of matrix x to unit (L2) length."""
    x_normed = x / T.sqrt(T.sum(x**2.,axis=0,keepdims=1)+1e-6)
    return x_normed

def rehu_actfun(x):
    """Compute rectified huberized activation for x."""
    M_quad = (x > 0.0) - (x >= 0.5)
    M_line = (x >= 0.5)
    x_rehu = (M_quad * x**2.) + (M_line * (x - 0.25))
    return x_rehu

def relu_actfun(x):
    """Compute rectified linear activation for x."""
    x_relu = T.maximum(0., x)
    return x_relu

def noop_actfun(x):
    """Do nothing activation. For output layer probably."""
    return x

def safe_softmax(x):
    """Softmax that shouldn't overflow."""
    e_x = T.exp(x - T.max(x, axis=1, keepdims=True))
    x_sm = e_x / T.sum(e_x, axis=1, keepdims=True)
    return x_sm

def smooth_softmax(x):
    """Softmax that shouldn't overflow, with fake Laplace smoothing."""
    eps = 0.0001
    e_x = T.exp(x - T.max(x, axis=1, keepdims=True))
    p = e_x / T.sum(e_x, axis=1, keepdims=True)
    p_sm = (p + eps) / T.sum((p + eps), axis=1, keepdims=True)
    return p_sm

def smooth_entropy(p):
    """Measure the entropy of distribution p, after converting it from an
    encoding in terms of relative log-likelihoods into an encoding as a
    sum-to-one distribution."""
    p_sm = smooth_softmax(p)
    e_sm = -T.sum((T.log(p_sm) * p_sm), axis=1, keepdims=True)
    return kl_sm

def smooth_kl_divergence(p, q):
    """Measure the KL-divergence from "approximate" distribution q to "true"
    distribution p. Use smoothed softmax to convert p and q from encodings
    in terms of relative log-likelihoods into sum-to-one distributions."""
    p_sm = smooth_softmax(p)
    q_sm = smooth_softmax(q)
    # This term is: cross_entropy(p, q) - entropy(p)
    kl_sm = T.sum(((T.log(p_sm) - T.log(q_sm)) * p_sm), axis=1, keepdims=True)
    return kl_sm

def smooth_cross_entropy(p, q):
    """Measure the cross-entropy between "approximate" distribution q and
    "true" distribution p. Use smoothed softmax to convert p and q from
    encodings in terms of relative log-likelihoods into sum-to-one dists."""
    p_sm = smooth_softmax(p)
    q_sm = smooth_softmax(q)
    # This term is: entropy(p) + kl_divergence(p, q)
    ce_sm = -T.sum((p_sm * T.log(q_sm)), axis=1, keepdims=True)
    return ce_sm

def smooth_kld_sym(p, q):
    """Measure the symmetrized KL-divergence."""
    p_sm = smooth_softmax(p)
    q_sm = smooth_softmax(q)
    kl_pq = T.sum(((T.log(p_sm) - T.log(q_sm)) * p_sm), axis=1, keepdims=True)
    kl_qp = T.sum(((T.log(q_sm) - T.log(p_sm)) * q_sm), axis=1, keepdims=True)
    kl_sym = (kl_pq + kl_qp) / 2.0
    return kl_sym

def smooth_xent_sym(p, q):
    """Measure the symmetrized cross-entropy."""
    p_sm = smooth_softmax(p)
    q_sm = smooth_softmax(q)
    ce_pq = -T.sum((p_sm * T.log(q_sm)), axis=1, keepdims=True)
    ce_qp = -T.sum((q_sm * T.log(p_sm)), axis=1, keepdims=True)
    ce_sym = (ce_pq + ce_qp) / 2.0
    return ce_sym

################################################################################
# HIDDEN LAYER IMPLEMENTATIONS: We've implemented a standard feedforward layer #
# with non-linear activation transform and a max-pooling (a.k.a. Maxout) layer #
# which is currently fixed to operate over disjoint pools of linear filters.   #
################################################################################

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, \
                 activation=None, pool_size=4, drop_rate=0., \
                 W=None, b=None, \
                 use_bias=True):

        # Setup a shared random generator for this layer
        self.srng = theano.tensor.shared_randomstreams.RandomStreams( \
                rng.randint(100000))

        # Set some basic layer properties
        self.activation = activation
        self.in_dim = n_in
        self.out_dim = n_out
        self.drop_rate = theano.shared(value=np.asarray(drop_rate, \
                dtype=theano.config.floatX), name='drop_rate')
        self.noise_std = theano.shared(value=np.asarray(0.0, \
                dtype=theano.config.floatX), name='noise_std')

        # Get the raw input to this layer and apply fuzzing + masking noise
        self.raw_input = input
        self.input, self.drop_mask = self._mask_noise( \
                self._gauss_noise(self.raw_input, self.noise_std), self.drop_rate)

        # Get some random initial weights and biases, if not given
        if W is None:
            W_init = np.asarray(0.01 * rng.standard_normal( \
                      size=(n_in, n_out)), dtype=theano.config.floatX)
            W = theano.shared(value=W_init, name='W')
        if b is None:
            b_init = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_init, name='b')

        # Set layer weights and biases, rescaled to account for dropping
        self.W = W
        self.b = b

        # Compute linear "pre-activation" for this layer and then apply
        # a non-linearity to get the final activations
        if use_bias:
            self.linear_output = T.dot(self.input, self.W) + self.b
        else:
            self.linear_output = T.dot(self.input, self.W)
        self.output = self.activation(self.linear_output)

        # Compute some sums of the activations, to use as regularizers
        self.act_l2_sum = T.sum(self.output**2.) / self.output.size
        self.row_l1_sum = T.sum(abs(row_normalize(self.output))) / self.output.shape[0]
        self.col_l1_sum = T.sum(abs(col_normalize(self.output))) / self.output.shape[1]

        # Conveniently package layer parameters
        if use_bias:
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]

    def set_bias_noise(self, noise_lvl=0.):
        """Set stdev of noise on the biases for this layer."""
        self.noise_std.set_value(noise_lvl)
        return 1

    def set_drop_rate(self, drop_rate=0.):
        """Set stdev of noise on the biases for this layer."""
        self.drop_rate.set_value(drop_rate)
        return 1

    def _mask_noise(self, input, p):
        """p is the probability of dropping elements of input."""
        # p=1-p because 1's indicate keep and p is prob of dropping
        noise_mask = self.srng.binomial(n=1, p=1-p, size=input.shape, dtype=theano.config.floatX)
        noise_scale = 1.0 / (1.0 - p)
        noisy_input = noise_scale * (input * noise_mask)
        return [noisy_input, noise_mask]

    def _gauss_noise(self, input, noise_std):
        noisy_input = input + \
                (noise_std * self.srng.normal(size=input.shape, dtype=theano.config.floatX))
        return noisy_input

    def _noisy_W(self, noise_lvl=0.):
        """Noise weights, like blurring the energy surface."""
        W_nz = self.W + self.srng.normal(size=self.W.shape, avg=0., std=noise_lvl)
        return W_nz

##########################
# NETWORK IMPLEMENTATION #
##########################

class AD_NET(object):
    """A multipurpose layer-based feedforward net.

    This class is capable of standard backprop training, training with
    dropout, and training with Dropout Ensemble Variance regularization.
    """
    def __init__(self,
            rng,
            input,
            params):
        # First, setup a shared random number generator for this layer
        self.srng = theano.tensor.shared_randomstreams.RandomStreams( \
            rng.randint(100000))
        # Setup simple activation function for this net.
        self.act_fun = lambda x: relu_actfun(x)
        #self.act_fun = lambda x: T.nnet.sigmoid(x)
        ################################################
        # Process user-suplied parameters for this net #
        ################################################
        layer_sizes = params['layer_sizes']
        lam_l2a = theano.shared(value=np.asarray(params['lam_l2a'], \
                dtype=theano.config.floatX), name='lam_l2a')
        use_bias = params['use_bias']
        self.is_semisupervised = 0
        # DEV-related parameters are as follows:
        #   dev_types: the transform to apply to the activations of each layer
        #              prior to computing dropout ensemble variance
        #   dev_lams: the weight for each layer's DEV regulariation
        self.dev_types = params['dev_types']
        dev_lams = np.asarray(params['dev_lams'], dtype=theano.config.floatX)
        self.dev_lams_sum = np.sum(dev_lams)
        self.dev_lams = theano.shared(value=dev_lams, name='dev_lams')
        # Make a dict to tell which parameters are norm-boundable
        self.clip_params = {}
        # Set up all the hidden layers
        layer_connect_dims = zip(layer_sizes, layer_sizes[1:])
        self.mlp_layers = []
        self.dev_layers_1 = []
        self.dev_layers_2 = []
        # Initialize "next inputs", to be piped into new layers
        self.input = input
        next_raw_input = self.input
        next_dev1_input = self.input
        next_dev2_input = self.input
        # Construct one drop-free MLP and two droppy child networks
        layer_num = 0
        for n_in, n_out in layer_connect_dims:
            last_layer = (layer_num == (len(layer_connect_dims)-1))
            drop_prob = 0.2 if (layer_num == 0) else 0.5
            activation = (lambda x: noop_actfun(x)) if last_layer else self.act_fun
            # Add a new layer to the regular model
            self.mlp_layers.append(HiddenLayer(rng=rng, \
                    input=next_raw_input, \
                    activation=activation, \
                    drop_rate=0., pool_size=4, \
                    n_in=n_in, n_out=n_out, use_bias=use_bias))
            next_raw_input = self.mlp_layers[-1].output
            # Add a new layer to each perturbed model
            self.dev_layers_1.append(HiddenLayer(rng=rng, \
                    input=next_dev1_input, \
                    activation=activation, \
                    drop_rate=drop_prob, pool_size=4, \
                    n_in=n_in, n_out=n_out, use_bias=use_bias, \
                    W=self.mlp_layers[-1].W, \
                    b=self.mlp_layers[-1].b))
            self.dev_layers_2.append(HiddenLayer(rng=rng, \
                    input=next_dev2_input, \
                    activation=activation, \
                    drop_rate=drop_prob, pool_size=4, \
                    n_in=n_in, n_out=n_out, use_bias=use_bias, \
                    W=self.mlp_layers[-1].W, \
                    b=self.mlp_layers[-1].b))
            next_dev1_input = self.dev_layers_1[-1].output
            next_dev2_input = self.dev_layers_2[-1].output
            # Set the parameters of these layers to be clipped
            self.clip_params[self.mlp_layers[-1].W] = 1
            self.clip_params[self.mlp_layers[-1].b] = 0
            layer_num = layer_num + 1
        # Mash all the parameters together, listily
        self.mlp_params = [p for l in self.mlp_layers for p in l.params]
        self.class_params = [p for p in self.mlp_layers[-1].params]
        self.layer_count = len(self.mlp_layers)
        self.layers = self.mlp_layers

        # Build loss functions for auto-discriminator training.
        self._construct_ad_layers(rng, lam_l1=0.01, lam_l2=1e-2, nz_lvl=0.3)

        # Use the negative log likelihood of the logistic regression layer of
        # the RAW net as the standard optimization objective.
        self.raw_out_func = MCL2HingeSS(self.mlp_layers[-1])
        self.raw_class_loss = self.raw_out_func.loss_func
        self.raw_reg_loss = lam_l2a * T.sum([lay.act_l2_sum for lay in self.mlp_layers])
        self.dev_reg_loss = lambda y: self.dev_cost(y, joint_loss=0)
        self.class_errors = self.raw_out_func.errors

        # Use the negative log likelihood of the logistic regression layer of
        # the first DEV clone as dropout optimization objective.
        self.dev_out_func_1 = MCL2HingeSS(self.dev_layers_1[-1])
        self.dev_out_func_2 = MCL2HingeSS(self.dev_layers_2[-1])
        self.sde_class_loss = self.dev_out_func_1.loss_func
        self.sde_reg_loss = lam_l2a * T.sum([lay.act_l2_sum for lay in self.dev_layers_1])
        self.sde_cost = lambda y: (self.sde_class_loss(y) + self.sde_reg_loss)
        self.ss_dev_class_loss = lambda y: (self.dev_out_func_1.loss_func(y) + \
                self.dev_out_func_2.loss_func(y)) / 2.0

    def dev_cost(self, y, joint_loss=1):
        """Wrapper for optimization with Theano."""
        if (self.dev_lams_sum > 1e-5):
            # Use a DEV-regularized cost if some DEV lams are > 0
            class_loss = self.ss_dev_class_loss(y) if self.is_semisupervised \
                    else self.raw_class_loss(y)
            dev_losses = []
            for i in range(self.layer_count):
                if (i < (self.layer_count - 1)):
                    # DEV loss at hidden layers
                    x1 = self.mlp_layers[i].output
                    #x1 = self.dev_layers_2[i].output
                    x2 = self.dev_layers_1[i].output
                else:
                    # DEV loss at output layer
                    x1 = self.mlp_layers[i].linear_output
                    #x1 = self.dev_layers_2[i].linear_output
                    x2 = self.dev_layers_1[i].linear_output
                dev_type = self.dev_types[i]
                dev_loss = self.dev_lams[i] * self._dev_loss(x1, x2, y, dev_type)
                dev_losses.append(dev_loss)
            reg_loss = T.sum(dev_losses) + self.sde_reg_loss
        else:
            # Otherwise, use a standard feedforward MLP loss
            class_loss = self.raw_out_func.loss_func(y)
            reg_loss = self.raw_reg_loss
        if (joint_loss == 1):
            # Return classification loss + DEV regularization loss
            L = class_loss + reg_loss
        else:
            # Return only DEV regularization cost (for diagnostics probably)
            L = reg_loss
        return L

    def _dev_loss(self, X1, X2, Y, dev_type):
        """Compute the Pseudo-Ensemble Variance regularizer.

        Regularization is applied to the transformed activities of each
        layer in the network, with the preceeding layers' activities subject
        to dropout noise. The DEV regularizer is applied only to observations
        with class label 0 (in y), for use in semisupervised learning. To use
        DEV regularization on the labeled data, just pass it through the net
        both with and without a label.
        """
        if not self.is_semisupervised:
            # Compute DEV regularizer using _all_ observations, not just those
            # with class label 0. (assume -1 is not a class label...)
            ss_mask = T.neq(Y, -1).reshape((Y.shape[0], 1))
        else:
            # Compute DEV regularizer only for observations with class label 0
            ss_mask = T.eq(Y, 0).reshape((Y.shape[0], 1))
        var_fun = lambda x1, x2: T.sum(((x1 - x2) * ss_mask)**2.) / T.sum(ss_mask)
        tanh_fun = lambda x1, x2: var_fun(T.tanh(x1), T.tanh(x2))
        norm_fun = lambda x1, x2: var_fun(row_normalize(x1), row_normalize(x2))
        sigm_fun = lambda x1, x2: var_fun(T.nnet.sigmoid(x1), T.nnet.sigmoid(x2))
        bent_fun = lambda p, q: T.sum(ss_mask * T.nnet.binary_crossentropy( \
                T.nnet.sigmoid(xo), T.nnet.sigmoid(xt))) / T.sum(ss_mask)
        ment_fun = lambda p, q: T.sum(ss_mask * smooth_xent_sym(p, q)) / T.sum(ss_mask)
        kl_fun = lambda p, q: T.sum(ss_mask * smooth_kld_sym(p, q)) / T.sum(ss_mask)
        if (dev_type == 1):
            # Unit-normalized variance (like fake cosine distance)
            dev_fun = norm_fun
        elif (dev_type == 2):
            # Tanh-transformed variance
            dev_fun = tanh_fun
        elif (dev_type == 3):
            # Sigmoid-transformed variance
            dev_fun = sigm_fun
        elif (dev_type == 4):
            # Binary cross-entropy
            dev_fun = bent_fun
        elif (dev_type == 5):
            # Multinomial cross-entropy
            dev_fun = ment_fun
        elif (dev_type == 6):
            # Multinomial KL-divergence
            dev_fun = kl_fun
        else:
            dev_fun = var_fun
        return dev_fun(X1, X2)

    def _ent_loss(self, X, Y, ent_type=0):
        """Compute the entropy regularizer. Either binary or multinomial."""
        if not self.is_semisupervised:
            ss_mask = T.neq(Y, -1).reshape((Y.shape[0], 1))
        else:
            ss_mask = T.eq(Y, 0).reshape((Y.shape[0], 1))
        bent_fun = lambda x: T.sum(ss_mask * T.nnet.binary_crossentropy( \
                T.nnet.sigmoid(x), T.nnet.sigmoid(x)) / T.sum(ss_mask))
        ment_fun = lambda x: T.sum(ss_mask * smooth_cross_entropy(x, x)) / T.sum(ss_mask)
        if (ent_type == 0):
            # Binary cross-entropy
            ent_fun = bent_fun
        else:
            # Multinomial cross-entropy
            ent_fun = ment_fun
        return ent_fun(X)

    def _construct_ad_layers(self, rng, lam_l1=1e-3, lam_l2=1e-3, nz_lvl=0.3):
        """Build both dropless and droppy autodiscriminator layers on top of
        each layer in this network. Input to the AD built on layer i is the
        'noised' output of layer i-1 in the dropless version of this network.
        Dropout in AD training occurs only local to the activations of the
        encoder part of each AD (for now)."""
        # Construct an AD for each hidden layer in this network.
        self.ad_layers = []
        self.ad_params = []
        self.ad_costs = []
        for i in range(len(self.mlp_layers)-1):
            # Construct the AD layer object
            ad_layer = ADLayer(rng=rng, \
                       source_layer=self.dev_layers_1[i], \
                       input_noise=nz_lvl)
            self.clip_params[ad_layer.b_h] = 0
            self.clip_params[ad_layer.b_v] = 0
            self.clip_params[ad_layer.A] = 0
            self.clip_params[ad_layer.b] = 0
            self.ad_layers.append(ad_layer)
            self.ad_params.append(ad_layer.params)
            self.ad_costs.append(ad_layer.compute_costs(lam_l1, lam_l2))
        return 1

    def set_bias_noise(self, noise_lvl=0.):
        """Set stochastic noise rate on the biases."""
        for layer in self.mlp_layers:
            layer.set_bias_noise(noise_lvl)
        for layer in self.dev_layers_1:
            layer.set_bias_noise(noise_lvl)
        for layer in self.dev_layers_2:
            layer.set_bias_noise(noise_lvl)
        return 1

    def set_dev_lams(self, d_lams):
        """Set the DEV regularization weights."""
        d_lams = np.asarray(d_lams, dtype=theano.config.floatX)
        self.dev_lams.set_value(d_lams)
        return 1

    def _masking_noise(self, input, nz_lvl):
        """Apply masking noise to the input of some denoising autoencoder."""
        drop_mask = self.srng.binomial(n=1, p=1.-nz_lvl, size=input.shape)
        droppy_input = input * T.cast(drop_mask, theano.config.floatX)
        return [droppy_input, drop_mask]


#########################
# THE AUTODISCRIMINATOR #
#########################


class ADLayer(object):
    def __init__(self, rng, source_layer, input_noise):

        # Setup a shared random generator for this layer
        self.srng = theano.tensor.shared_randomstreams.RandomStreams( \
                    rng.randint(100000))

        # Set the source layer (from the parent MLP) for this ADLayer
        self.source_layer = source_layer
        self.input_noise = input_noise

        # Set some basic layer properties
        self.input = self.source_layer.raw_input
        self.activation = self.source_layer.activation
        self.in_dim = self.source_layer.in_dim
        self.out_dim = self.source_layer.out_dim

        # self.W, self.b_h, and self.b_v are parameters for the autoencoder
        # function of this AD layer.
        self.W = self.source_layer.W
        self.b_h = self.source_layer.b
        self.b_v = theano.shared(value=np.zeros((self.in_dim,), \
                   dtype=theano.config.floatX), name='b_v')

        # self.A and self.b are parameters for the autodiscrimnator function
        # of this layer
        A_init = np.asarray(0.05 * rng.standard_normal( \
                 size=(self.out_dim, self.out_dim)), dtype=theano.config.floatX)
        self.A = theano.shared(value=A_init, name='A')
        self.b = theano.shared(value=np.zeros((self.out_dim,), \
                 dtype=theano.config.floatX), name='b')

        self.params = [self.W, self.b_h, self.b_v, self.A, self.b]

    def compute_costs(self, lam_l1=0., lam_l2=0., lam_A=1e-3):
        """Compute autodiscrimination and activation sparsity costs."""
        # Get noisy input, for autoencoder computation
        noisy_input = self._gauss_noise(self.input, 0.05)
        noisy_input, noise_mask = self._mask_noise(noisy_input, self.input_noise)

        # Compute hidden and visible activations for autoencoder
        #A_v, A_h = self._compute_activations(noisy_input)
        # Compute autoencoder cost
        #ae_cost = 0.1 * (T.sum((self.input - A_v)**2.0) / A_v.shape[0])
        ae_cost = 1e-5 * T.sum(self.b_v**2.0)

        # Compute autodiscriminator predictions
        X = self.source_layer.output
        X_masked, mask = self._mask_noise(X, 0.33)
        F = T.dot(X_masked, self.A) + self.b
        # Compute autodiscriminator cost
        p_y = smooth_softmax(T.dot(((1.0 - mask) * F), X.T))
        idx = T.arange(p_y.shape[0])
        ad_cost = -(T.sum(T.log(p_y[idx,idx])) / p_y.shape[0]) + \
                  (lam_A * (T.sum(self.A**2.0) + T.sum(self.b**2.0)))

        # Compute general activation regularization penalties
        row_l1_sum = self.source_layer.row_l1_sum
        row_l2_sum = self.source_layer.act_l2_sum
        reg_cost = (lam_l1 * row_l1_sum) + (lam_l2 * row_l2_sum)
        return [(ae_cost + ad_cost), reg_cost]

    def _compute_activations(self, X):
        """Compute activations of autoencoder (@ hidden/visible layers)."""
        W_nz = self.W #self._noisy_W(self.W, 0.01)
        A_h = self.activation(T.dot(X, W_nz) + self.b_h)
        A_v = T.dot(A_h, W_nz.T) + self.b_v
        return [A_v, A_h]

    def _noisy_W(self, W, noise_lvl=0.):
        """Noise weights, like blurring the energy surface."""
        W_nz = W + self.srng.normal(size=W.shape, avg=0., std=noise_lvl, \
                dtype=theano.config.floatX)
        return W_nz

    def _mask_noise(self, input, p):
        """p is the probability of dropping elements of input."""
        # p=1-p because 1's indicate keep and p is prob of dropping
        noise_mask = self.srng.binomial(n=1, p=1-p, size=input.shape, dtype=theano.config.floatX)
        # Cast mask from int to float32, to keep things on GPU
        noisy_input = input * noise_mask
        return [noisy_input, noise_mask]

    def _gauss_noise(self, input, noise_std):
        noisy_input = input + \
                (noise_std * self.srng.normal(size=input.shape, dtype=theano.config.floatX))
        return noisy_input


##############
# EYE BUFFER #
##############
