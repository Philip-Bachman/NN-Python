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
    """Softmax that shouldn't overflow, with Laplacish smoothing."""
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

        # Use either droppy or undropped input, as determined by drop_rate
        if (drop_rate < 0.01):
            self.input = input
        else:
            self.input = self._drop_from_input(input, drop_rate)

        # Set some basic layer properties
        self.activation = activation
        self.in_dim = n_in
        self.out_dim = n_out
        self.noise_std = theano.shared(value=np.asarray(0.0, \
                dtype=theano.config.floatX), name='noise_std')

        # Don't use a bias for L2 pooling layers
        if (self.activation == 'l2_pool'):
            use_bias = False

        # Get some random initial weights and biases, if not given
        if W is None:
            #W_init = np.asarray(rng.uniform(
            #          low=-4 * np.sqrt(6. / (n_in + n_out)),
            #          high=4 * np.sqrt(6. / (n_in + n_out)),
            #          size=(n_in, n_out)), dtype=theano.config.floatX)
            W_init = np.asarray(0.01 * rng.standard_normal( \
                      size=(n_in, n_out)), dtype=theano.config.floatX)
            W = theano.shared(value=W_init, name='W')
        if b is None:
            b_init = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_init, name='b')

        # Set layer weights and biases, rescaled to account for dropping
        self.W = W if (drop_rate < 0.01) else ((1. / (1-drop_rate)) * W)
        self.b = b

        # Compute linear "pre-activation" for this layer
        if use_bias:
            self.linear_output = T.dot(self.input, self.W) + self.b
        else:
            self.linear_output = T.dot(self.input, self.W)
        # Get a "fuzzy" version of the linear output, with the degree of fuzz
        # determined by self.noise_std. For values of self.noise_std > 0, this
        # may be a useful regularizer during learning. But, self.noise_std
        # should probably be set to 0 at validation/test time.
        self.noisy_linear = self.linear_output  + \
                (self.noise_std * self.srng.normal(size=self.linear_output.shape, \
                dtype=theano.config.floatX))

        # Apply some non-linearity to compute "activation" for this layer
        if (self.activation == 'l2_pool'):
            self.output = T.sqrt( \
                    T.dot(self.input**2., abs(self.W)) + self.b)
        else:
            self.output = self.activation(self.noisy_linear)

        # Compute some sums of the activations, for regularizing
        self.act_l2_sum = T.sum(self.output**2.) / self.output.size
        self.row_l1_sum = T.sum(abs(row_normalize(self.output))) / self.output.shape[0]
        self.col_l1_sum = T.sum(abs(col_normalize(self.output))) / self.output.shape[1]

        # Conveniently package layer parameters
        if use_bias:
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]

    def _drop_from_input(self, input, p):
        """p is the probability of dropping elements of input."""
        # p=1-p because 1's indicate keep and p is prob of dropping
        drop_mask = self.srng.binomial(n=1, p=1-p, size=input.shape, dtype=theano.config.floatX)
        # Cast mask from int to float32, to keep things on GPU
        droppy_input = input * drop_mask
        return droppy_input

    def set_bias_noise(self, noise_lvl=0.):
        """Set stdev of noise on the biases for this layer."""
        self.noise_std.set_value(noise_lvl)
        return 1

    def _noisy_W(self, noise_lvl=0.):
        """Noise weights, like blurring the energy surface."""
        W_nz = self.W + self.srng.normal(size=self.W.shape, avg=0., std=noise_lvl)
        return W_nz

class MPLayer(object):
    """Max-pooling over disjoint sets of linear filters (a.k.a. Maxout)..."""
    def __init__(self, rng, input, n_in, n_out, \
                 activation=None, pool_size=4, drop_rate=0., \
                 W=None, b=None, \
                 use_bias=True):

        # Setup a shared random generator for this layer
        self.srng = theano.tensor.shared_randomstreams.RandomStreams( \
                rng.randint(100000))

        # Use either droppy or undropped input, as determined by drop_rate
        if (drop_rate < 0.01):
            self.input = input
        else:
            self.input = self._drop_from_input(input, drop_rate)

        # Set some basic layer properties
        self.in_dim = n_in
        self.out_dim = n_out
        self.pool_size = pool_size
        self.noise_std = theano.shared(value=np.asarray(0.0, \
                dtype=theano.config.floatX), name='noise_std')

        # Get some random initial weights and biases, if not given
        if W is None:
            W_init = np.asarray(0.01 * rng.standard_normal( \
                size=(n_in, n_out*pool_size)), dtype=theano.config.floatX)
            W = theano.shared(value=W_init, name='W')
        if b is None:
            b_init = np.zeros((n_out*pool_size,), dtype=theano.config.floatX)
            b = theano.shared(value=b_init, name='b')

        # Set layer weights and biases, rescaled to account for dropping
        self.W = W if (drop_rate < 0.01) else ((1. / (1-drop_rate)) * W)
        self.b = b

        # Compute linear "pre-activation" for this layer
        if use_bias:
            self.linear_output = T.dot(self.input, self.W) + self.b
        else:
            self.linear_output = T.dot(self.input, self.W)
        # Get a "fuzzy" version of the linear output, with the degree of fuzz
        # determined by self.noise_std. For values of self.noise_std > 0, this
        # may be a useful regularizer during learning. But, self.noise_std
        # should probably be set to 0 at validation/test time.
        self.noisy_linear = self.linear_output  + \
                (self.noise_std * self.srng.normal(size=self.linear_output.shape, \
                dtype=theano.config.floatX))

        # Compute output of this layer by max-pooling over noisified linear
        # filter responses.
        #mp_acts = 0.
        self.pre_pooling = self._drop_from_input(self.noisy_linear, 0.5)
        mp_acts = None
        for i in xrange(pool_size):
            cur = self.pre_pooling[:,(i*n_out):((i+1)*n_out)]
            if mp_acts is None:
                mp_acts = cur
            else:
                mp_acts = T.maximum(mp_acts, cur)
        self.output = mp_acts

        # Compute some sums of the activations, for regularizing
        self.act_l2_sum = T.sum(self.output**2.) / self.output.size
        self.row_l1_sum = T.sum(abs(row_normalize(self.output))) / self.output.shape[0]
        self.col_l1_sum = T.sum(abs(col_normalize(self.output))) / self.output.shape[1]

        # Conveniently package layer parameters
        if use_bias:
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]

    def _drop_from_input(self, input, p):
        """p is the probability of dropping elements of input."""
        # p=1-p because 1's indicate keep and p is prob of dropping
        drop_mask = self.srng.binomial(n=1, p=1-p, size=input.shape, dtype=theano.config.floatX)
        # Cast mask from int to float32, to keep things on GPU
        droppy_input = input * drop_mask
        return droppy_input

    def set_bias_noise(self, noise_lvl=0.):
        """Set stdev of noise on the biases for this layer."""
        self.noise_std.set_value(noise_lvl)
        return 1

##########################
# NETWORK IMPLEMENTATION #
##########################

class SS_DEV_NET(object):
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
        # Setup simple activation function for this net. If using the ReLU
        # activation, set self.using_sigmoid to 0. If using the sigmoid
        # activation, set self.using_sigmoid to 1.
        self.act_fun = lambda x: relu_actfun(x)
        #self.act_fun = lambda x: T.nnet.sigmoid(x)
        self.using_sigmoid = 0
        #self.using_sigmoid = 1
        ################################################
        # Process user-suplied parameters for this net #
        ################################################
        layer_sizes = params['layer_sizes']
        lam_l2a = theano.shared(value=np.asarray(params['lam_l2a'], \
                dtype=theano.config.floatX), name='dev_mix_rate')
        use_bias = params['use_bias']
        self.is_semisupervised = 0
        # DEV-related parameters are as follows:
        #   dev_types: the transform to apply to the activations of each layer
        #              prior to computing dropout ensemble variance
        #   dev_lams: the weight for each layer's DEV regulariation
        #   dev_mix_rate: the mixing ratio between the raw net output and the
        #                 droppy net output for computing classification loss
        #                 when training with DEV regularization.
        self.dev_types = params['dev_types']
        dev_lams = np.asarray(params['dev_lams'], dtype=theano.config.floatX)
        self.dev_lams_sum = np.sum(dev_lams)
        self.dev_lams = theano.shared(value=dev_lams, name='dev_lams')
        try:
            self.dev_mix_rate = theano.shared(value=np.asarray(params['dev_mix_rate'], \
                    dtype=theano.config.floatX), name='dev_mix_rate')
        except:
            self.dev_mix_rate = theano.shared(value=np.asarray(0.0, \
                    dtype=theano.config.floatX), name='dev_mix_rate')
        # Make a dict to tell which parameters are norm-boundable
        self.clip_params = {}
        # Set up all the hidden layers
        layer_connect_dims = zip(layer_sizes, layer_sizes[1:])
        self.mlp_layers = []
        self.dev_layers = []
        # Initialize "next inputs", to be piped into new layers
        self.input = input
        next_raw_input = self.input
        next_drop_input = self.input
        # Iteratively append layers to the RAW net and each of some number
        # of droppy DEV clones.
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
            # Add a new layer to the dropout model
            self.dev_layers.append(HiddenLayer(rng=rng, \
                    input=next_drop_input, \
                    activation=activation, \
                    drop_rate=drop_prob, pool_size=4, \
                    n_in=n_in, n_out=n_out, use_bias=use_bias, \
                    W=self.mlp_layers[-1].W, \
                    b=self.mlp_layers[-1].b))
            next_drop_input = self.dev_layers[-1].output
            # Set the parameters of these layers to be clipped
            self.clip_params[self.mlp_layers[-1].W] = 1
            self.clip_params[self.mlp_layers[-1].b] = 0
            layer_num = layer_num + 1
        # Mash all the parameters together, listily
        self.mlp_params = [p for l in self.mlp_layers for p in l.params]
        self.class_params = [p for p in self.mlp_layers[-1].params]
        self.layer_count = len(self.mlp_layers)
        self.layers = self.mlp_layers

        # Build loss functions for denoising autoencoder training. This sets
        # up a cost function for training each layer in this net as a DAE with
        # inputs determined by the output of the preceeding layer.
        self._construct_dae_layers(rng, lam_l1=0.2, nz_lvl=0.3)

        # Build layers and functions for computing finite-differences-based
        # regularization of functional curvature.
        self._construct_grad_layers(rng, lam_g1=0.1, lam_g2=0.1, step_len=0.2)

        # Use the negative log likelihood of the logistic regression layer of
        # the RAW net as the standard optimization objective.
        self.raw_out_func = MCL2HingeSS(self.mlp_layers[-1])
        self.raw_class_loss = self.raw_out_func.loss_func
        self.raw_reg_loss = lam_l2a * T.sum([lay.act_l2_sum for lay in self.mlp_layers])
        self.dev_reg_loss = lambda y: self.dev_cost(y, joint_loss=0)
        self.class_errors = self.raw_out_func.errors

        # Use the negative log likelihood of the logistic regression layer of
        # the first DEV clone as dropout optimization objective.
        self.sde_out_func = MCL2HingeSS(self.dev_layers[-1])
        self.sde_class_loss = self.sde_out_func.loss_func
        self.sde_reg_loss = lam_l2a * T.sum([lay.act_l2_sum for lay in self.dev_layers])
        self.sde_cost = lambda y: (self.sde_class_loss(y) + self.sde_reg_loss)

    def dev_cost(self, y, joint_loss=1):
        """Wrapper for optimization with Theano."""
        dmr = self.dev_mix_rate
        if (self.dev_lams_sum > 1e-5):
            # Use a DEV-regularized cost if some DEV lams are > 0
            class_loss = (dmr * self.raw_class_loss(y)) + ((1-dmr) * self.sde_class_loss(y))
            dev_losses = []
            for i in range(self.layer_count):
                if (i < (self.layer_count - 1)):
                    # DEV loss at hidden layers
                    x1 = self.mlp_layers[i].output
                    x2 = self.dev_layers[i].output
                else:
                    # DEV loss at output layer
                    x1 = self.mlp_layers[i].linear_output
                    x2 = self.dev_layers[i].linear_output
                dev_type = self.dev_types[i]
                dev_loss = self.dev_lams[i] * self._dev_loss(x1, x2, y, dev_type)
                dev_losses.append(dev_loss)
            reg_loss = T.sum(dev_losses) + (0.5 * (self.raw_reg_loss + self.sde_reg_loss))
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
        """Compute the Dropout Ensemble Variance regularizer.

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
        ment_fun = lambda p, q: T.sum(ss_mask * smooth_cross_entropy(p, q)) / T.sum(ss_mask)
        kl_fun = lambda p, q: T.sum(ss_mask * smooth_kl_divergence(p, q)) / T.sum(ss_mask)
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

    def _construct_dae_layers(self, rng, lam_l1=0., nz_lvl=0.25):
        """Build both dropless and droppy denoising autoencoders on top of
        each layer in this network. Input to the DAE built on layer i is the
        'noised' output of layer i-1 in the dropless version of this network.
        Dropout in DAE training occurs only local to the activations of the
        encoder part of each DAE (for now)."""
        # Construct a DAE for each hidden layer in this network.
        self.dae_layers = []
        self.dae_params = []
        self.dae_costs = []
        for i in range(len(self.mlp_layers)-1):
            W_mlp = self.mlp_layers[i].W
            b_mlp = self.mlp_layers[i].b
            input_mlp = self.mlp_layers[i].input
            obs_dim = self.mlp_layers[i].in_dim
            code_dim = self.mlp_layers[i].out_dim
            # Construct the DAE layer object
            dae_layer = DAELayer(rng=rng, \
                    input=input_mlp, \
                    n_in=obs_dim, n_out=code_dim, \
                    activation=self.act_fun, \
                    input_noise=nz_lvl, \
                    W=W_mlp, b_h=b_mlp, b_v=None)
            self.clip_params[dae_layer.b_h] = 0
            self.clip_params[dae_layer.b_v] = 0
            self.dae_layers.append(dae_layer)
            self.dae_params.append(dae_layer.params)
            self.dae_costs.append(dae_layer.compute_costs(lam_l1))
        return 1

    def _construct_grad_layers(self, rng, lam_g1=0., lam_g2=0., step_len=0.1):
        """Construct siamese networks for regularizing first and second order
        directional derivatives via stochastic finite differences."""
        self.left_layers = []
        self.right_layers = []
        for mlp_layer in self.mlp_layers:
            W = mlp_layer.W
            b = mlp_layer.b
            in_dim = mlp_layer.in_dim
            out_dim = mlp_layer.out_dim
            [left_input, right_input] = self._twin_displacement_noise( \
                    mlp_layer.input, step_len)
            # Construct left/right twins of the main MLP, through which points
            # will be passed for use in finite-difference-based regularization
            # of first and second-order functional curvature.
            left_layer = HiddenLayer(rng=rng, \
                        input=left_input, \
                        activation=mlp_layer.activation, \
                        drop_rate=0., \
                        W=W, b=b, \
                        n_in=in_dim, n_out=out_dim, use_bias=1)
            right_layer = HiddenLayer(rng=rng, \
                        input=right_input, \
                        activation=mlp_layer.activation, \
                        drop_rate=0., \
                        W=W, b=b, \
                        n_in=in_dim, n_out=out_dim, use_bias=1)
            self.left_layers.append(left_layer)
            self.right_layers.append(right_layer)
        # Construct the loss functions for first and second-order grads
        self.grad_losses = []
        for (l_num, mlp_layer) in enumerate(self.mlp_layers):
            if False: #(l_num < (self.layer_count - 1)):
                # Use the activation-transformed layer response
                left_vals = self.left_layers[l_num].output
                center_vals = mlp_layer.output
                right_vals = self.right_layers[l_num].output
            else:
                # Use the linear layer response (at output layer)
                left_vals = T.tanh(self.left_layers[l_num].linear_output)
                center_vals = T.tanh(mlp_layer.linear_output)
                right_vals = T.tanh(self.right_layers[l_num].linear_output)
            g1l_loss = T.sum(((left_vals - center_vals) / step_len)**2.)
            g1r_loss = T.sum(((right_vals - center_vals) / step_len)**2.)
            g1_loss = (g1l_loss + g1r_loss) / (2. * center_vals.shape[0])
            g2_loss = T.sum(((left_vals + right_vals - (2. * center_vals)) \
                    / step_len**2.)**2.) / center_vals.shape[0]
            self.grad_losses.append([g1_loss, g2_loss])
        return 1

    def set_bias_noise(self, noise_lvl=0.):
        """Set stochastic noise rate on the biases."""
        for layer in self.mlp_layers:
            layer.set_bias_noise(noise_lvl)
        for layer in self.dev_layers:
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

    def _twin_displacement_noise(self, input, step_len):
        """Compute a pair of points for each row in input such that the input
        row bisects its point pair (i.e. the three points are collinear), with
        the direction of the induced line selected uniformly at random, and
        with the distance between each distal point and the point from input
        equal to step_len. *For stochastic 1st/2nd-order grad regularizing."""
        step_dirs = self.srng.normal(size=input.shape,dtype=theano.config.floatX)
        step_dirs = row_normalize(step_dirs) * step_len
        l_twin = input - step_dirs
        r_twin = input + step_dirs
        return [l_twin, r_twin]


###########################################
# DENOISING AUTOENCODER IMPLEMENTATION... #
###########################################


class DAELayer(object):
    def __init__(self, rng, input, n_in, n_out, \
                 activation=None, input_noise=0., \
                 W=None, b_h=None, b_v=None):

        # Setup a shared random generator for this layer
        self.srng = theano.tensor.shared_randomstreams.RandomStreams( \
                rng.randint(100000))

        self.input = input
        self.noisy_input, self.noise_mask = self._get_noisy_input(input, input_noise)

        # Set some basic layer properties
        self.activation = activation
        self.in_dim = n_in
        self.out_dim = n_out

        # Get some random initial weights and biases, if not given
        if W is None:
            W_init = np.asarray(0.01 * rng.standard_normal( \
                      size=(n_in, n_out)), dtype=theano.config.floatX)
            W = theano.shared(value=W_init, name='W')
        if b_h is None:
            b_init = np.zeros((n_out,), dtype=theano.config.floatX)
            b_h = theano.shared(value=b_init, name='b_h')
        if b_v is None:
            b_init = np.zeros((n_in,), dtype=theano.config.floatX)
            b_v = theano.shared(value=b_init, name='b_v')

        self.W = W
        self.b_h = b_h
        self.b_v = b_v

        self.params = [self.W, self.b_h, self.b_v]

    def compute_costs(self, lam_l1=0.):
        """Compute reconstruction and activation sparsity costs."""
        # Get noisy weights
        W_nz = self._noisy_W(self.W, 0.01)
        # Compute hidden and visible activations
        A_v, A_h = self._compute_activations(self.noisy_input, W_nz, self.b_h, self.b_v)
        # Compute reconstruction error cost
        recon_cost = T.sum((self.input - A_v)**2.0) / self.input.shape[0]
        # Compute sparsity penalty
        row_l1_sum = T.sum(abs(row_normalize(A_h))) / A_h.shape[0]
        col_l1_sum = T.sum(abs(col_normalize(A_h))) / A_h.shape[1]
        sparse_cost = lam_l1 * (row_l1_sum + col_l1_sum)
        return [recon_cost, sparse_cost]

    def _compute_hidden_acts(self, X, W, b):
        """Compute activations at hidden layer."""
        A_h = self.activation(T.dot(X, W) + b)
        return A_h

    def _compute_activations(self, X, W, b_h, b_v):
        """Compute activations of decoder (at visible layer)."""
        A_h = self._compute_hidden_acts(X, W, b_h)
        A_v = T.dot(A_h, W.T) + b_v
        return [A_v, A_h]

    def _noisy_W(self, W, noise_lvl=0.):
        """Noise weights, like blurring the energy surface."""
        W_nz = W + self.srng.normal(size=W.shape, avg=0., std=noise_lvl, \
                dtype=theano.config.floatX)
        return W_nz

    def _get_noisy_input(self, input, p):
        """p is the probability of dropping elements of input."""
        # p=1-p because 1's indicate keep and p is prob of dropping
        noise_mask = self.srng.binomial(n=1, p=1-p, size=input.shape, dtype=theano.config.floatX)
        # Cast mask from int to float32, to keep things on GPU
        noisy_input = input * noise_mask
        return [noisy_input, noise_mask]


##############
# EYE BUFFER #
##############
