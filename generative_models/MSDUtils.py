#############################################################################
# Code for managing and training a variational Iterative Refinement Model.  #
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
                      apply_mask, constFX, to_fX, DCG

###################
# very basic LSTM #
###################

class SimpleLSTM(object):
    def __init__(self, in_dim, hid_dim, \
                 W_all=None, b_all=None, \
                 name="", W_scale=1.0):

        # Set some basic layer properties
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.joint_in_dim = self.in_dim + self.hid_dim
        self.joint_out_dim = 4 * self.hid_dim

        # Initialize layer params, using user-provided values if given
        if W_all is None:
            # Generate initial filters using orthogonal random trick
            W_shape = (self.joint_in_dim, self.joint_out_dim)
            W_scale = W_scale * (1.0 / np.sqrt(self.joint_in_dim))
            W_init = W_scale * npr.normal(0.0, 1.0, W_shape)
            #W_init = ortho_matrix(shape=W_shape, gain=W_scale)
            W_init = W_init.astype(theano.config.floatX)
            W_all = theano.shared(value=W_init, \
                    name="{0:s}_W_all".format(name))
        if b_all is None:
            b_init = np.zeros((self.joint_out_dim,), \
                    dtype=theano.config.floatX)
            b_all = theano.shared(value=b_init, \
                    name="{0:s}_b_all".format(name))
        # attach the weights to this object for later use
        self.W_all = W_all
        self.b_all = b_all

        # Conveniently package layer parameters
        self.mlp_params = [self.W_all, self.b_all]
        # Layer construction complete...
        return

    def apply(self, x_t, h_tm1, c_tm1):
        """
        Apply propagate the current input x_t and the previous exposed state
        and memory state h_tm1/c_tm1 through this LSTM layer.
        """
        hd = self.hid_dim
        # merge exogenous (i.e. x_t) and endogenous (i.e. h_tm1) inputs
        joint_input = T.horizontal_stack(x_t, h_tm1)
        joint_output = T.dot(joint_input, self.W_all) + self.b_all
        # compute transformed input to the layer
        g_t = T.tanh( joint_output[:,0:(1*hd)] )
        # compute input gate
        i_t = T.nnet.sigmoid( joint_output[:,(1*hd):(2*hd)] )
        # compute forget gate
        f_t = T.nnet.sigmoid( joint_output[:,(2*hd):(3*hd)] )
        # compute output gate
        o_t = T.nnet.sigmoid( joint_output[:,(3*hd):(4*hd)] )
        # compute current memory state
        c_t = (f_t * c_tm1) + (i_t * g_t)
        # compute current exposed state
        h_t = (o_t * T.tanh(c_t))
        return h_t, c_t

##################
# very basic MLP #
##################

class SimpleMLP(object):
    def __init__(self, in_dim, out_dim, \
                 W=None, b=None, \
                 name="", W_scale=1.0):

        # Set some basic layer properties
        self.in_dim = in_dim
        self.out_dim = out_dim

        # initialize weights and biases
        if W is None:
            # Generate initial filters using orthogonal random trick
            W_shape = (self.in_dim, self.out_dim)
            W_scale = W_scale * (1.0 / np.sqrt(self.in_dim))
            W_init = W_scale * npr.normal(0.0, 1.0, W_shape)
            #W_init = ortho_matrix(shape=W_shape, gain=W_scale)
            W_init = W_init.astype(theano.config.floatX)
            W = theano.shared(value=W_init, \
                    name="{0:s}_W".format(name))
        if b is None:
            b_init = np.zeros((self.out_dim,), \
                    dtype=theano.config.floatX)
            b = theano.shared(value=b_init, \
                    name="{0:s}_b".format(name))
        # grab handles for easy access
        self.W = W
        self.b = b

        # Conveniently package layer parameters
        self.mlp_params = [self.W, self.b]
        # Layer construction complete...
        return

    def get_bias(self):
        """
        Get the bias at output layer.
        """
        out_bias = self.b
        return out_bias

    def apply(self, x):
        """
        Apply this SimpleMLP to some input.
        """
        y = T.dot(x, self.W) + self.b
        return y

#####################
# very basic InfNet #
#####################

class SimpleInfNet(object):
    def __init__(self, rng, in_dim, out_dim, \
                 W_mean=None, b_mean=None, \
                 W_logvar=None, b_logvar=None, \
                 name="", W_scale=1.0):
        # setup a shared random generator for this network 
        self.rng = RandStream(rng.randint(1000000))

        # set some basic layer properties
        self.in_dim = in_dim
        self.out_dim = out_dim

        # initialize weights and biases for mean estimate
        if W_mean is None:
            # Generate initial filters using orthogonal random trick
            W_shape = (self.in_dim, self.out_dim)
            W_scale = W_scale * (1.0 / np.sqrt(self.in_dim))
            W_init = W_scale * npr.normal(0.0, 1.0, W_shape)
            #W_init = ortho_matrix(shape=W_shape, gain=W_scale)
            W_init = W_init.astype(theano.config.floatX)
            W_mean = theano.shared(value=W_init, \
                    name="{0:s}_W_mean".format(name))
        if b_mean is None:
            b_init = np.zeros((self.out_dim,), \
                    dtype=theano.config.floatX)
            b_mean = theano.shared(value=b_init, \
                    name="{0:s}_b_mean".format(name))
        # grab handles for easy access
        self.W_mean = W_mean
        self.b_mean = b_mean

        # initialize weights and biases for log-variance estimate
        if W_logvar is None:
            # Generate initial filters using orthogonal random trick
            W_shape = (self.in_dim, self.out_dim)
            W_scale = W_scale * (1.0 / np.sqrt(self.in_dim))
            W_init = W_scale * npr.normal(0.0, 1.0, W_shape)
            #W_init = ortho_matrix(shape=W_shape, gain=W_scale)
            W_init = W_init.astype(theano.config.floatX)
            W_logvar = theano.shared(value=W_init, \
                    name="{0:s}_W_logvar".format(name))
        if b_logvar is None:
            b_init = np.zeros((self.out_dim,), \
                    dtype=theano.config.floatX)
            b_logvar = theano.shared(value=b_init, \
                    name="{0:s}_b_logvar".format(name))
        # grab handles for easy access
        self.W_logvar = W_logvar
        self.b_logvar = b_logvar

        # Conveniently package layer parameters
        self.mlp_params = [self.W_mean, self.b_mean, \
                           self.W_logvar, self.b_logvar]
        # Layer construction complete...
        return

    def get_bias(self):
        """
        Get the bias at output layer.
        """
        out_bias = self.b_mean
        return out_bias

    def apply(self, x, do_samples=True):
        """
        Apply this SimpleInfNet to some input.
        """
        z_mean = T.dot(x, self.W_mean) + self.b_mean
        z_logvar = T.dot(x, self.W_logvar) + self.b_logvar
        z_samples = z_mean + ( (T.exp(0.5*z_logvar)) * \
                DCG(self.rng.normal(size=z_mean.shape, avg=0.0, std=1.0, \
                dtype=theano.config.floatX)) )
        # wrap them up for easy returnage
        result = [z_mean, z_logvar]
        if do_samples:
            result.append(z_samples)
        return result

####################################################
# Basic reader and writer, as in DRAW source paper #
####################################################

class SimpleReader(object):
    """
    Basic reader, as described in the DRAW source paper.
    """
    def __init__(self, x_dim, rnn_dim):
        self.x_dim = x_dim
        self.rnn_dim = rnn_dim
        self.out_dim = 2*self.x_dim
        self.mlp_params = [] # no params in this form of reader
        return

    def apply(self, x, x_hat, sim1_dec):
        """
        Just return the target x and the current residual.
        """
        output = T.concatenate([x, x_hat], axis=1)
        return output

class SimpleWriter(object):
    """
    Simple writer, as described in the DRAW source paper.
    """
    def __init__(self, in_dim, out_dim):

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mlp = SimpleMLP(in_dim, out_dim, \
                W=None, b=None, \
                name="writer_mlp", W_scale=1.0)

        self.mlp_params = self.mlp.mlp_params
        return

    def apply(self, si_dec):
        """
        Apply the MLP to the current decoder state.
        """
        output = self.mlp.apply(si_dec)
        return output

####################################################
# Fancy reader and writer, as in DRAW source paper #
####################################################

# class AttentionReader(Initializable):
#     def __init__(self, x_dim, dec_dim, height, width, N, **kwargs):
#         super(AttentionReader, self).__init__(name="reader", **kwargs)

#         self.img_height = height
#         self.img_width = width
#         self.N = N
#         self.x_dim = x_dim
#         self.dec_dim = dec_dim
#         self.output_dim = 2*N*N

#         self.zoomer = ZoomableAttentionWindow(height, width, N)
#         self.readout = MLP(activations=[Identity()], dims=[dec_dim, 5], **kwargs)

#         self.children = [self.readout]

#     def get_dim(self, name):
#         if name == 'input':
#             return self.dec_dim
#         elif name == 'x_dim':
#             return self.x_dim
#         elif name == 'output':
#             return self.output_dim
#         else:
#             raise ValueError
            
#     @application(inputs=['x', 'x_hat', 'h_dec'], outputs=['r'])
#     def apply(self, x, x_hat, h_dec):
#         l = self.readout.apply(h_dec)

#         center_y, center_x, delta, sigma, gamma = self.zoomer.nn2att(l)

#         w     = gamma * self.zoomer.read(x    , center_y, center_x, delta, sigma)
#         w_hat = gamma * self.zoomer.read(x_hat, center_y, center_x, delta, sigma)
        
#         return T.concatenate([w, w_hat], axis=1)

# class AttentionWriter(Initializable):
#     def __init__(self, input_dim, output_dim, width, height, N, **kwargs):
#         super(AttentionWriter, self).__init__(name="writer", **kwargs)

#         self.img_width = width
#         self.img_height = height
#         self.N = N
#         self.input_dim = input_dim
#         self.output_dim = output_dim

#         assert output_dim == width*height

#         self.zoomer = ZoomableAttentionWindow(height, width, N)
#         self.z_trafo = Linear(
#                 name=self.name+'_ztrafo',
#                 input_dim=input_dim, output_dim=5, 
#                 weights_init=self.weights_init, biases_init=self.biases_init,
#                 use_bias=True)

#         self.w_trafo = Linear(
#                 name=self.name+'_wtrafo',
#                 input_dim=input_dim, output_dim=N*N, 
#                 weights_init=self.weights_init, biases_init=self.biases_init,
#                 use_bias=True)

#         self.children = [self.z_trafo, self.w_trafo]

#     @application(inputs=['h'], outputs=['c_update'])
#     def apply(self, h):
#         w = self.w_trafo.apply(h)
#         l = self.z_trafo.apply(h)

#         center_y, center_x, delta, sigma, gamma = self.zoomer.nn2att(l)

#         c_update = 1./gamma * self.zoomer.write(w, center_y, center_x, delta, sigma)

#         return c_update

#     @application(inputs=['h'], outputs=['c_update', 'center_y', 'center_x', 'delta'])
#     def apply_detailed(self, h):
#         w = self.w_trafo.apply(h)
#         l = self.z_trafo.apply(h)

#         center_y, center_x, delta, sigma, gamma = self.zoomer.nn2att(l)

#         c_update = 1./gamma * self.zoomer.write(w, center_y, center_x, delta, sigma)

#         return c_update, center_y, center_x, delta

