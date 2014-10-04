import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse
import theano.tensor.shared_randomstreams
from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams

###############################
# ACTIVATIONS AND OTHER STUFF #
###############################

def row_normalize(x):
    """Normalize rows of matrix x to unit (L2) norm."""
    x_normed = x / T.sqrt(T.sum(x**2.,axis=1,keepdims=1)+1e-6)
    return x_normed

def col_normalize(x):
    """Normalize cols of matrix x to unit (L2) norm."""
    x_normed = x / T.sqrt(T.sum(x**2.,axis=0,keepdims=1)+1e-6)
    return x_normed

def rehu_actfun(x):
    """Compute rectified huberized activation for x."""
    M_quad = (x > 0.0) * (x < 0.5)
    M_line = (x >= 0.5)
    x_rehu = (M_quad * x**2.) + (M_line * (x - 0.25))
    return x_rehu

def relu_actfun(x):
    """Compute rectified linear activation for x."""
    x_relu = T.maximum(0., x)
    return x_relu

def maxout_actfun(input, pool_size, filt_count):
    """Apply maxout over non-overlapping sets of values."""
    last_start = filt_count - pool_size
    mp_vals = None
    for i in xrange(pool_size):
        cur = input[:,i:(last_start+i+1):pool_size]
        if mp_vals is None:
            mp_vals = cur
        else:
            mp_vals = T.maximum(mp_vals, cur)
    return mp_vals

def normout_actfun(input, pool_size, filt_count):
    """Apply (L2) normout over non-overlapping sets of values."""
    l_start = filt_count - pool_size
    relu_vals = T.stack(\
        *[input[:,i:(l_start+i+1):pool_size] for i in range(pool_size)])
    pooled_vals = T.sqrt(T.mean(relu_vals**2.0, axis=0))
    return pooled_vals

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
    p = (e_x / T.sum(e_x, axis=1, keepdims=True)) + eps
    p_sm = p / T.sum(p, axis=1, keepdims=True)
    return p_sm

def smooth_entropy(p):
    """Measure the entropy of distribution p, after converting it from an
    encoding in terms of relative log-likelihoods into an encoding as a
    sum-to-one distribution."""
    p_sm = smooth_softmax(p)
    ent_sm = -T.sum((T.log(p_sm) * p_sm), axis=1, keepdims=True)
    return ent_sm

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

######################################
# BASIC FULLY-CONNECTED HIDDEN LAYER #
######################################

class HiddenLayer(object):
    def __init__(self, rng, input, in_dim, out_dim, \
                 activation=None, pool_size=0, \
                 drop_rate=0., input_noise=0., bias_noise=0., \
                 W=None, b=None, \
                 use_bias=True, name="", W_scale=1.0):

        # Setup a shared random generator for this layer
        #self.rng = theano.tensor.shared_randomstreams.RandomStreams( \
        #        rng.randint(100000))
        self.rng = CURAND_RandomStreams(rng.randint(1000000))

        self.clean_input = input

        # Add gaussian noise to the input (if desired)
        if (input_noise > 1e-4):
            self.fuzzy_input = input + self.rng.normal(size=input.shape, \
                    avg=0.0, std=input_noise, dtype=theano.config.floatX)
        else:
            self.fuzzy_input = input

        # Apply masking noise to the input (if desired)
        if (drop_rate > 1e-4):
            self.noisy_input = self._drop_from_input(self.fuzzy_input, drop_rate)
        else:
            self.noisy_input = self.fuzzy_input

        # Set some basic layer properties
        self.pool_size = pool_size
        self.in_dim = in_dim
        self.out_dim = out_dim
        if self.pool_size <= 1:
            self.filt_count = self.out_dim
        else:
            self.filt_count = self.out_dim * self.pool_size
        self.pool_count = self.filt_count / max(self.pool_size, 1)
        if activation:
            self.activation = activation
        else:
            if self.pool_size <= 1:
                self.activation = lambda x: relu_actfun(x)
            else:
                self.activation = lambda x: \
                        maxout_actfun(x, self.pool_size, self.filt_count)

        # Get some random initial weights and biases, if not given
        if W is None:
            if self.pool_size <= 1:
                # Generate random initial filters in a typical way
                W_init = 0.01 * np.asarray(rng.normal( \
                          size=(self.in_dim, self.filt_count)), \
                          dtype=theano.config.floatX)
            else:
                # Generate groups of random filters to pool over such that
                # intra-group correlations are stronger than inter-group
                # correlations, to encourage pooling over similar filters...
                filters = []
                f_size = (self.in_dim, 1)
                for g_num in range(self.pool_count):
                    g_filt = 0.01 * rng.normal(size=f_size)
                    for f_num in range(self.pool_size):
                        f_filt = g_filt + 0.003 * rng.normal(size=f_size)
                        filters.append(f_filt)
                W_init = np.hstack(filters).astype(theano.config.floatX)
            W = theano.shared(value=(W_scale*W_init), name="{0:s}_W".format(name))
        if b is None:
            b_init = np.zeros((self.filt_count,), dtype=theano.config.floatX)
            b = theano.shared(value=b_init, name="{0:s}_b".format(name))

        # Set layer weights and biases
        self.W = W
        self.b = b

        # Compute linear "pre-activation" for this layer
        if use_bias:
            self.linear_output = T.dot(self.noisy_input, self.W) + self.b
        else:
            self.linear_output = T.dot(self.noisy_input, self.W)

        # Add noise to the pre-activation features (if desired)
        if bias_noise > 1e-3:
            self.noisy_linear = self.linear_output  + \
                    self.rng.normal(size=self.linear_output.shape, \
                    avg=0.0, std=bias_noise, dtype=theano.config.floatX)
        else:
            self.noisy_linear = self.linear_output

        # Apply activation function
        self.output = self.activation(self.noisy_linear)

        # Compute some properties of the activations, probably to regularize
        self.act_l2_sum = T.sum(self.output**2.) / self.output.size
        self.row_l1_sum = T.sum(abs(row_normalize(self.output))) / \
                self.output.shape[0]
        self.col_l1_sum = T.sum(abs(col_normalize(self.output))) / \
                self.output.shape[1]

        # Conveniently package layer parameters
        if use_bias:
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]
        # Layer construction complete...
        return

    def _drop_from_input(self, input, p):
        """p is the probability of dropping elements of input."""
        # get a drop mask that drops things with probability p
        drop_rnd = self.rng.uniform(size=input.shape, low=0.0, high=1.0, \
                dtype=theano.config.floatX)
        drop_mask = drop_rnd > p
        # get a scaling factor to keep expectations fixed after droppage
        drop_scale = 1. / (1. - p)
        # apply dropout mask and rescaling factor to the input
        droppy_input = drop_scale * input * drop_mask
        return droppy_input

    def _noisy_params(self, P, noise_lvl=0.):
        """Noisy weights, like convolving energy surface with a gaussian."""
        P_nz = P + self.rng.normal(size=P.shape, avg=0.0, std=noise_lvl, \
                dtype=theano.config.floatX)
        return P_nz

######################################################
# SIMPLE LAYER FOR AVERAGING OUTPUTS OF OTHER LAYERS #
######################################################

class JoinLayer(object):
    """
    Simple layer that averages over "linear_output"s of other layers.

    Note: The list of layers to average over is the only parameter used.
    """
    def __init__(self, input_layers):
        print("making join layer over {0:d} output layers...".format( \
                len(input_layers)))
        il_los = [il.linear_output for il in input_layers]
        self.linear_output = T.mean(T.stack(*il_los), axis=0)
        return

#####################################################
# DISCRIMINATIVE LAYER (SINGLE-OUTPUT LINEAR LAYER) #
#####################################################

class DiscLayer(object):
    def __init__(self, rng, input, in_dim, W=None, b=None):
        # Setup a shared random generator for this layer
        self.rng = theano.tensor.shared_randomstreams.RandomStreams( \
                rng.randint(100000))
        #self.rng = CURAND_RandomStreams(rng.randint(1000000))

        self.input = input
        self.in_dim = in_dim

        # Get some random initial weights and biases, if not given
        if W is None:
            # Generate random initial filters in a typical way
            W_init = 0.01 * np.asarray(rng.normal( \
                      size=(self.in_dim, 1)), \
                      dtype=theano.config.floatX)
            W = theano.shared(value=W_init)
        if b is None:
            b_init = np.zeros((1,), dtype=theano.config.floatX)
            b = theano.shared(value=b_init)

        # Set layer weights and biases
        self.W = W
        self.b = b

        # Compute linear "pre-activation" for this layer
        self.linear_output = T.dot(self.input, self.W) + self.b

        # Apply activation function
        self.output = self.linear_output

        # Compute squared sum of outputs, for regularization
        self.act_l2_sum = T.sum(self.output**2.0) / self.output.shape[0]

        # Conveniently package layer parameters
        self.params = [self.W, self.b]
        # little layer construction complete...
        return

    def _noisy_params(self, P, noise_lvl=0.):
        """Noisy weights, like convolving energy surface with a gaussian."""
        P_nz = P + self.rng.normal(size=P.shape, avg=0.0, std=noise_lvl, \
                dtype=theano.config.floatX)
        return P_nz

##################################
# DENOISING AUTOENCODER LAYER... #
##################################

class DAELayer(object):
    def __init__(self, rng, clean_input=None, fuzzy_input=None, \
            in_dim=0, out_dim=0, activation=None, input_noise=0., \
            W=None, b_h=None, b_v=None):

        # Setup a shared random generator for this layer
        #self.rng = theano.tensor.shared_randomstreams.RandomStreams( \
        #        rng.randint(100000))
        self.rng = CURAND_RandomStreams(rng.randint(1000000))

        # Grab the layer input and perturb it with some sort of noise. This
        # is, afterall, a _denoising_ autoencoder...
        self.clean_input = clean_input
        self.noisy_input = self._get_noisy_input(fuzzy_input, input_noise)

        # Set some basic layer properties
        self.activation = activation
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Get some random initial weights and biases, if not given
        if W is None:
            W_init = np.asarray(0.01 * rng.standard_normal( \
                      size=(in_dim, out_dim)), dtype=theano.config.floatX)
            W = theano.shared(value=W_init, name='W')
        if b_h is None:
            b_init = np.zeros((out_dim,), dtype=theano.config.floatX)
            b_h = theano.shared(value=b_init, name='b_h')
        if b_v is None:
            b_init = np.zeros((in_dim,), dtype=theano.config.floatX)
            b_v = theano.shared(value=b_init, name='b_v')

        # Grab pointers to the now-initialized weights and biases
        self.W = W
        self.b_h = b_h
        self.b_v = b_v

        # Put the learnable/optimizable parameters into a list
        self.params = [self.W, self.b_h, self.b_v]
        # Beep boop... layer construction complete...
        return

    def compute_costs(self, lam_l1=None):
        """Compute reconstruction and activation sparsity costs."""
        # Get noise-perturbed encoder/decoder parameters
        W_nz = self._noisy_params(self.W, 0.01)
        b_nz = self.b_h #self._noisy_params(self.b_h, 0.05)
        # Compute hidden and visible activations
        A_v, A_h = self._compute_activations(self.noisy_input, \
                W_nz, b_nz, self.b_v)
        # Compute reconstruction error cost
        recon_cost = T.sum((self.clean_input - A_v)**2.0) / \
                self.clean_input.shape[0]
        # Compute sparsity penalty (over both population and lifetime)
        row_l1_sum = T.sum(abs(row_normalize(A_h))) / A_h.shape[0]
        col_l1_sum = T.sum(abs(col_normalize(A_h))) / A_h.shape[1]
        sparse_cost = lam_l1[0] * (row_l1_sum + col_l1_sum)
        return [recon_cost, sparse_cost]

    def _compute_hidden_acts(self, X, W, b_h):
        """Compute activations of encoder (at hidden layer)."""
        A_h = self.activation(T.dot(X, W) + b_h)
        return A_h

    def _compute_activations(self, X, W, b_h, b_v):
        """Compute activations of decoder (at visible layer)."""
        A_h = self._compute_hidden_acts(X, W, b_h)
        A_v = T.dot(A_h, W.T) + b_v
        return [A_v, A_h]

    def _noisy_params(self, P, noise_lvl=0.):
        """Noisy weights, like convolving energy surface with a gaussian."""
        if noise_lvl > 1e-3:
            P_nz = P + self.rng.normal(size=P.shape, avg=0.0, std=noise_lvl, \
                    dtype=theano.config.floatX)
        else:
            P_nz = P
        return P_nz

    def _get_noisy_input(self, input, p):
        """p is the probability of dropping elements of input."""
        drop_rnd = self.rng.uniform(input.shape, low=0.0, high=1.0, \
            dtype=theano.config.floatX)
        drop_mask = drop_rnd > p
        # Cast mask from int to float32, to keep things on GPU
        noisy_input = input * drop_mask
        return noisy_input
