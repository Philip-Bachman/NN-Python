import numpy as np
import numpy.random as npr
import theano
import theano.tensor as T
#from theano.tensor.shared_randomstreams import RandomStreams as RandStream
from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams as RandStream

#from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
#from pylearn2.sandbox.cuda_convnet.pool import MaxPool
#from theano.sandbox.cuda.basic_ops import gpu_contiguous

###############################
# ACTIVATIONS AND OTHER STUFF #
###############################

def row_normalize(x):
    """Normalize rows of matrix x to unit (L2) norm."""
    x_normed = x / T.sqrt(T.sum(x**2.,axis=1,keepdims=1) +  constFX(1e-8))
    return x_normed

def col_normalize(x):
    """Normalize cols of matrix x to unit (L2) norm."""
    x_normed = x / T.sqrt(T.sum(x**2.,axis=0,keepdims=1) + constFX(1e-8))
    return x_normed

def max_normalize(x, axis=1):
    """
    Normalize matrix x to max unit (L2) norm along some axis.
    """
    row_norms = T.sqrt(T.sum(x**2., axis=axis, keepdims=1))
    row_scales = T.maximum(1.0, row_norms)
    x_normed = x / row_scales
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

def softplus_actfun(x, scale=None):
    """Compute rescaled softplus activation for x."""
    if scale is None:
        x_softplus = (1.0 / 2.0) * T.nnet.softplus(2.0*x)
    else:
        x_softplus = (1.0 / scale) * T.nnet.softplus(scale*x)
    return x_softplus

def tanh_actfun(x, scale=None):
    """Compute  rescaled tanh activation for x."""
    if scale is None:
        x_tanh = T.tanh(x)
    else:
        x_tanh = scale * T.tanh(constFX(1/scale) * x)
    return x_tanh

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
    #x_sm = T.nnet.softmax(x)
    return x_sm

def smooth_softmax(x):
    """Softmax that shouldn't overflow, with Laplacish smoothing."""
    eps = 0.0001
    e_x = T.exp(x - T.max(x, axis=1, keepdims=True))
    p = (e_x / T.sum(e_x, axis=1, keepdims=True)) + constFX(eps)
    p_sm = p / T.sum(p, axis=1, keepdims=True)
    return p_sm

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

def apply_mask(Xd=None, Xc=None, Xm=None):
    """
    Apply a mask, like in the old days.
    """
    X_masked = ((1.0 - Xm) * Xd) + (Xm * Xc)
    return X_masked

def binarize_data(X):
    """
    Make a sample of bernoulli variables with probabilities given by X.
    """
    X_shape = X.shape
    probs = npr.rand(*X_shape)
    X_binary = 1.0 * (probs < X)
    return X_binary.astype(theano.config.floatX)

def row_shuffle(X, Y=None):
    """
    Return a copy of X with shuffled rows.
    """
    shuf_idx = np.arange(X.shape[0])
    npr.shuffle(shuf_idx)
    X_shuf = X[shuf_idx]
    if Y is None:
        result = X_shuf
    else:
        Y_shuf = Y[shuf_idx]
        result = [X_shuf, Y_shuf]
    return result

# code from the lasagne library
def ortho_matrix(shape=None, gain=1.0):
    """
    Orthogonal matrix initialization. For n-dimensional shapes where n > 2,
    the n-1 trailing axes are flattened. For convolutional layers, this
    corresponds to the fan-in, so this makes the initialization usable for
    both dense and convolutional layers.
    """
    # relu gain for elided activations
    if gain == 'relu':
        gain = np.sqrt(2)
    if len(shape) < 2:
        raise RuntimeError("Only shapes of length 2 or more are supported.")
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = npr.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v # pick the one with the correct shape
    q = q.reshape(shape)
    W = gain * q[:shape[0], :shape[1]]
    return W

######################################
# BASIC FULLY-CONNECTED HIDDEN LAYER #
######################################

class HiddenLayer(object):
    def __init__(self, rng, input, in_dim, out_dim, \
                 activation=None, pool_size=0, \
                 drop_rate=0., input_noise=0., bias_noise=0., \
                 W=None, b=None, b_in=None, s_in=None, \
                 name="", W_scale=1.0):

        # Setup a shared random generator for this layer
        self.rng = RandStream(rng.randint(1000000))

        # setup parameters for controlling 
        zero_ary = np.zeros((1,)).astype(theano.config.floatX)
        self.input_noise = theano.shared(value=(zero_ary+input_noise), \
                name="{0:s}_input_noise".format(name))
        self.bias_noise = theano.shared(value=(zero_ary+bias_noise), \
                name="{0:s}_bias_noise".format(name))
        self.drop_rate = theano.shared(value=(zero_ary+drop_rate), \
                name="{0:s}_drop_rate".format(name))

        # setup scale and bias params for the input
        if b_in is None:
            # input biases are always initialized to zero
            ary = np.zeros((in_dim,), dtype=theano.config.floatX)
            b_in = theano.shared(value=ary, name="{0:s}_b_in".format(name))
        if s_in is None:
            # input scales are always initialized to one
            ary = 0.541325 * np.ones((in_dim,), dtype=theano.config.floatX)
            s_in = theano.shared(value=ary, name="{0:s}_s_in".format(name))
        self.b_in = b_in
        self.s_in = s_in

        # Set some basic layer properties
        self.pool_size = pool_size
        self.in_dim = in_dim
        self.out_dim = out_dim
        if self.pool_size <= 1:
            self.filt_count = self.out_dim
        else:
            self.filt_count = self.out_dim * self.pool_size
        self.pool_count = self.filt_count / max(self.pool_size, 1)
        if activation is None:
            activation = relu_actfun
        if self.pool_size <= 1:
            self.activation = activation
        else:
            self.activation = lambda x: \
                    maxout_actfun(x, self.pool_size, self.filt_count)

        # Get some random initial weights and biases, if not given
        if W is None:
            # Generate initial filters using orthogonal random trick
            W_shape = (self.in_dim, self.filt_count)
            #W_scale = W_scale * (1.0 / np.sqrt(self.in_dim))
            #W_init = W_scale * npr.normal(0.0, 1.0, W_shape)
            W_init = ortho_matrix(shape=(self.in_dim, self.filt_count), \
                    gain=W_scale)
            #W_init = 0.01 * npr.normal(0.0, 1.0, W_shape)
            W_init = W_init.astype(theano.config.floatX)
            W = theano.shared(value=W_init, name="{0:s}_W".format(name))
        if b is None:
            b_init = np.zeros((self.filt_count,), dtype=theano.config.floatX)
            b = theano.shared(value=b_init, name="{0:s}_b".format(name))

        # Set layer weights and biases
        self.W = W
        self.b = b

        # Feedforward through the layer
        use_in = input_noise > 0.001
        use_bn = bias_noise > 0.001
        use_drop = drop_rate > 0.001
        self.linear_output, self.noisy_linear, self.output = \
                self.apply(input, use_in=use_in, use_bn=use_bn, \
                use_drop=use_drop)

        # Compute some properties of the activations, probably to regularize
        self.act_l2_sum = T.sum(self.noisy_linear**2.) / self.output.size

        # Conveniently package layer parameters
        self.params = [self.W, self.b, self.b_in, self.s_in]
        self.shared_param_dicts = { \
                'W': self.W, \
                'b': self.b, \
                'b_in': self.b_in, \
                's_in': self.s_in }
        # Layer construction complete...
        return

    def apply(self, input, use_in=False, use_bn=False, use_drop=False):
        """
        Apply feedforward to this input, returning several partial results.
        """
        # Add gaussian noise to the input (if desired)
        fancy_input = T.nnet.softplus(self.s_in) * (input + self.b_in)
        #fancy_input = input
        if use_in:
            fuzzy_input = fancy_input + self.input_noise[0] * \
                    self.rng.normal(size=fancy_input.shape, avg=0.0, std=1.0, \
                    dtype=theano.config.floatX)
        else:
            fuzzy_input = fancy_input
        # Apply masking noise to the input (if desired)
        if use_drop:
            noisy_input = self._drop_from_input(fuzzy_input, self.drop_rate[0])
        else:
            noisy_input = fuzzy_input
        self.noisy_input = noisy_input
        # Compute linear "pre-activation" for this layer
        linear_output = T.dot(noisy_input, self.W) + self.b
        # Add noise to the pre-activation features (if desired)
        if use_bn:
            noisy_linear = linear_output + self.bias_noise[0] * \
                    self.rng.normal(size=linear_output.shape, avg=0.0, \
                    std=1.0, dtype=theano.config.floatX)
        else:
            noisy_linear = linear_output
        # Apply activation function
        final_output = self.activation(noisy_linear)
        # package partial results for easy return
        results = [linear_output, noisy_linear, final_output]
        return results

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

##############################################
# COMBINED CONVOLUTION AND MAX-POOLING LAYER #
##############################################

COMMENT="""
class ConvPoolLayer(object):
    A simple convolution --> max-pooling layer.

    The (symbolic) input to this layer must be a theano.tensor.dtensor4 shaped
    like (batch_size, chan_count, im_dim_1, im_dim_2).

    filt_def should be a 4-tuple like (filt_count, in_chans, filt_def_1, filt_def_2)

    pool_def should be a 3-tuple like (pool_dim, pool_stride)
    def __init__(self, rng, input=None, filt_def=None, pool_def=(2, 2), \
            activation=None, drop_rate=0., input_noise=0., bias_noise=0., \
            W=None, b=None, name="", W_scale=1.0):

        # Setup a shared random generator for this layer
        self.rng = RandStream(rng.randint(100000))

        self.clean_input = input

        zero_ary = np.zeros((1,)).astype(theano.config.floatX)
        self.input_noise = theano.shared(value=(zero_ary+input_noise), \
                name="{0:s}_input_noise".format(name))
        self.bias_noise = theano.shared(value=(zero_ary+bias_noise), \
                name="{0:s}_bias_noise".format(name))
        self.drop_rate = theano.shared(value=(zero_ary+drop_rate), \
                name="{0:s}_bias_noise".format(name))

        # Add gaussian noise to the input (if desired)
        self.fuzzy_input = input + (self.input_noise[0] * \
                self.rng.normal(size=input.shape, avg=0.0, std=1.0, \
                dtype=theano.config.floatX))

        # Apply masking noise to the input (if desired)
        self.noisy_input = self._drop_from_input(self.fuzzy_input, \
                self.drop_rate[0])

        # Set the activation function for the conv filters
        if activation:
            self.activation = activation
        else:
            self.activation = lambda x: relu_actfun(x)

        # initialize weights with random weights
        W_init = 1.0 * np.asarray(rng.normal( \
                size=filt_def), dtype=theano.config.floatX)
        self.W = theano.shared(value=(W_scale*W_init), \
                name="{0:s}_W".format(name))

        # the bias is a 1D tensor -- one bias per output feature map
        b_init = np.zeros((filt_def[0],), dtype=theano.config.floatX) + 0.1
        self.b = theano.shared(value=b_init, name="{0:s}_b".format(name))

        # convolve input feature maps with filters
        input_c01b = self.noisy_input.dimshuffle(1, 2, 3, 0) # bc01 to c01b
        filters_c01b = self.W.dimshuffle(1, 2, 3, 0) # bc01 to c01b
        conv_op = FilterActs(stride=1, partial_sum=1)
        contig_input = gpu_contiguous(input_c01b)
        contig_filters = gpu_contiguous(filters_c01b)
        conv_out_c01b = conv_op(contig_input, contig_filters)

        if (bias_noise > 1e-4):
            noisy_conv_out_c01b = conv_out_c01b + self.rng.normal( \
                    size=conv_out_c01b.shape, avg=0.0, std=bias_noise, \
                    dtype=theano.config.floatX)
        else:
            noisy_conv_out_c01b = conv_out_c01b

        # downsample each feature map individually, using maxpooling
        pool_op = MaxPool(ds=pool_def[0], stride=pool_def[1])
        mp_out_c01b = pool_op(noisy_conv_out_c01b)
        mp_out_bc01 = mp_out_c01b.dimshuffle(3, 0, 1, 2) # c01b to bc01

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.noisy_linear_output = mp_out_bc01 + self.b.dimshuffle('x', 0, 'x', 'x')
        self.linear_output = self.noisy_linear_output
        self.output = self.activation(self.noisy_linear_output)

        # store parameters of this layer
        self.params = [self.W, self.b]

        return

    def _drop_from_input(self, input, p):
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
        P_nz = P + self.rng.normal(size=P.shape, avg=0.0, std=noise_lvl, \
                dtype=theano.config.floatX)
        return P_nz
"""

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
        self.output = T.mean(T.stack(*il_los), axis=0)
        self.linear_output = self.output
        self.noisy_linear_output = self.output
        return

#############################################
# RESHAPING LAYERS (FOR VECTORS<-->TENSORS) #
#############################################

class Reshape2D4DLayer(object):
	"""
	Reshape from flat vectors to image-y 3D tensors.
	"""
	def __init__(self, input=None, out_shape=None):
		assert(len(out_shape) == 3)
		self.input = input
		self.output = self.input.reshape((self.input.shape[0], \
			out_shape[0], out_shape[1], out_shape[2]))
		self.linear_output = self.output
		self.noisy_linear_output = self.output
		return

class Reshape4D2DLayer(object):
	"""
	Flatten from 3D image-y tensors to flat vectors.
	"""
	def __init__(self, input=None):
		self.input = input
		out_dim = T.prod(self.input.shape[1:])
		self.output = self.input.reshape((self.input.shape[0], out_dim))
		self.linear_output = self.output
		self.noisy_linear_output = self.output
		return

#####################################################
# DISCRIMINATIVE LAYER (SINGLE-OUTPUT LINEAR LAYER) #
#####################################################

class DiscLayer(object):
    def __init__(self, rng, input, in_dim, W=None, b=None, W_scale=1.0):
        # Setup a shared random generator for this layer
        self.rng = RandStream(rng.randint(1000000))

        self.input = input
        self.in_dim = in_dim

        # Get some random initial weights and biases, if not given
        if W is None:
            # Generate random initial filters in a typical way
            W_init = 1.0 * np.asarray(rng.normal( \
                      size=(self.in_dim, 1)), \
                      dtype=theano.config.floatX)
            W = theano.shared(value=(W_scale*W_init))
        if b is None:
            b_init = np.zeros((1,), dtype=theano.config.floatX)
            b = theano.shared(value=b_init)

        # Set layer weights and biases
        self.W = W
        self.b = b

        # Compute linear "pre-activation" for this layer
        self.linear_output = 20.0 * T.tanh((T.dot(self.input, self.W) + self.b) / 20.0)

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
        P_nz = P + DCG(self.rng.normal(size=P.shape, avg=0.0, std=noise_lvl, \
                dtype=theano.config.floatX))
        return P_nz

##################################
# DENOISING AUTOENCODER LAYER... #
##################################

class DAELayer(object):
    def __init__(self, rng, clean_input=None, fuzzy_input=None, \
            in_dim=0, out_dim=0, activation=None, input_noise=0., \
            W=None, b_h=None, b_v=None, W_scale=1.0):

        # Setup a shared random generator for this layer
        self.rng = RandStream(rng.randint(1000000))

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
            W_init = np.asarray(1.0 * DCG(rng.standard_normal( \
                      size=(in_dim, out_dim)), dtype=theano.config.floatX))
            W = theano.shared(value=(W_scale*W_init), name='W')
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
            P_nz = P + DCG(self.rng.normal(size=P.shape, avg=0.0, std=noise_lvl, \
                    dtype=theano.config.floatX))
        else:
            P_nz = P
        return P_nz

    def _get_noisy_input(self, input, p):
        """p is the probability of dropping elements of input."""
        drop_rnd = self.rng.uniform(input.shape, low=0.0, high=1.0, \
            dtype=theano.config.floatX)
        drop_mask = drop_rnd > p
        # Cast mask from int to float32, to keep things on GPU
        noisy_input = input * DCG(drop_mask)
        return noisy_input
