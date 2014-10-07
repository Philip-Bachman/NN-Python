##################################################################
# Code for networks and whatnot to use in variationalish stuff.  #
##################################################################

# basic python
import numpy as np
import numpy.random as npr
from collections import OrderedDict

# theano business
import theano
import theano.tensor as T
from theano.ifelse import ifelse
import theano.tensor.shared_randomstreams
#from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams

# phil's sweetness
from NetLayers import HiddenLayer, DiscLayer

####################################
# INFERENCE NETWORK IMPLEMENTATION #
####################################


class INF_NET(object):
    """
    A net that tries to infer an approximate posterior for some observation,
    given some deep, directed generative model.

    Parameters:
        rng: a numpy.random RandomState object
        input_data: symbolic input matrix for inputting observed data
        input_infd: symbolic input matrix for inputting inferred data
        input_mask: symbolic input matrix for a mask on which values to take
                    from input_data and which to take from input_infer
        params: a dict of parameters describing the desired ensemble:
            lam_l2a: L2 regularization weight on neuron activations
            vis_drop: drop rate to use on inputs
            hid_drop: drop rate to use on hidden layer activations
                -- note: vis_drop/hid_drop are optional, with defaults 0.0/0.0
            bias_noise: standard dev for noise on the biases of hidden layers
            out_noise: standard dev for noise on the output of this net
            mlp_config: list of "layer descriptions"
    """
    def __init__(self,
            rng=None,
            input_data=None,
            input_infd=None,
            input_mask=None,
            params=None):
        # First, setup a shared random number generator for this layer
        self.rng = theano.tensor.shared_randomstreams.RandomStreams( \
            rng.randint(100000))
        # Grab the symbolic input matrix
        self.input_data = input_data
        self.input_infd = input_infd
        self.input_mask = input_mask
        #####################################################
        # Process user-supplied parameters for this network #
        #####################################################
        lam_l2a = params['lam_l2a']
        if 'vis_drop' in params:
            self.vis_drop = params['vis_drop']
        else:
            self.vis_drop = 0.0
        if 'hid_drop' in params:
            self.hid_drop = params['hid_drop']
        else:
            self.hid_drop = 0.0
        if 'bias_noise' in params:
            self.bias_noise = params['bias_noise']
        else:
            self.bias_noise = 0.0
        if 'out_noise' in params:
            self.out_noise = params['out_noise']
        else:
            self.out_noise = 0.0
        # Get the configuration/prototype for this network. The config is a
        # list of layer descriptions, including a description for the input
        # layer, which is typically just the dimension of the inputs. So, the
        # depth of the mlp is one less than the number of layer configs.
        self.mlp_config = params['mlp_config']
        self.mlp_depth = len(self.mlp_config) - 1
        self.latent_dim = self.mlp_config[0]
        self.data_dim = self.mlp_config[-1]
        ##########################
        # Initialize the network #
        ##########################
        self.clip_params = {}
        self.mlp_layers = []
        layer_def_pairs = zip(self.mlp_config[:-1],self.mlp_config[1:])
        layer_num = 0
        # Construct input by combining data input and inferred input, taking
        # masked values from inferred input and other values from data input
        next_input = ((1.0 - mask) * self.data_input) + \
                (mask * self.infd_input)
        for in_def, out_def in layer_def_pairs:
            first_layer = (layer_num == 0)
            last_layer = (layer_num == (len(layer_def_pairs) - 1))
            l_name = "in_layer_{0:d}".format(layer_num)
            if (type(in_def) is list) or (type(in_def) is tuple):
                # Receiving input from a poolish layer...
                in_dim = in_def[0]
            else:
                # Receiving input from a normal layer...
                in_dim = in_def
            if (type(out_def) is list) or (type(out_def) is tuple):
                # Applying some sort of pooling in this layer...
                out_dim = out_def[0]
                pool_size = out_def[1]
            else:
                # Not applying any pooling in this layer...
                out_dim = out_def
                pool_size = 0
            # Select the appropriate noise to add to this layer
            if first_layer:
                d_rate = self.vis_drop
            else:
                d_rate = self.hid_drop
            if last_layer:
                b_noise = self.out_noise
            else:
                b_noise = self.bias_noise
            # Add a new, well-configured layer to the regular model
            self.mlp_layers.append(HiddenLayer(rng=rng, \
                    input=next_input, activation=None, pool_size=pool_size, \
                    drop_rate=d_rate, input_noise=0., bias_noise=b_noise, \
                    in_dim=in_dim, out_dim=out_dim, \
                    name=l_name, W_scale=2.0))
            next_input = self.mlp_layers[-1].output
            # Set the non-bias parameters of this layer to be clipped
            self.clip_params[self.mlp_layers[-1].W] = 1
            # Acknowledge layer completion
            layer_num = layer_num + 1

        # set norms to which to clip various parameters
        self.clip_norms = {}

        # Mash all the parameters together, into a list.
        self.mlp_params = []
        for layer in self.mlp_layers:
            self.mlp_params.extend(layer.params)

        # The output of this inference network is given by the noisy output
        # of its final layer.
        self.output_noise = self.mlp_layers[-1].noisy_linear
        self.out_dim = self.mlp_layers[-1].out_dim
        # Get simple regularization penalty to moderate activation dynamics
        self.act_reg_cost = lam_l2a * self._act_reg_cost()
        self.output = self.output_noise
        return

    def _act_reg_cost(self):
        """Apply L2 regularization to the activations in each spawn-net."""
        act_sq_sums = []
        for layer in self.mlp_layers:
            act_sq_sums.append(layer.act_l2_sum)
        full_act_sq_sum = T.sum(act_sq_sums)
        return full_act_sq_sum

#####################################
# GENERATIVE NETWORK IMPLEMENTATION #
#####################################

class GEN_NET(object):
    """
    A net that transforms a simple distribution so that it matches some
    more complicated distribution, for some definition of match....

    Parameters:
        rng: a numpy.random RandomState object
        input_noise: symbolic input matrix for inputting latent noise
        input_data: symbolic input matrix for inputting real data
        params: a dict of parameters describing the desired ensemble:
            lam_l2a: L2 regularization weight on neuron activations
            vis_drop: drop rate to use on samples from the base distribution
            hid_drop: drop rate to use on activations of hidden layers
                -- note: vis_drop/hid_drop are optional, with defaults 0.0/0.0
            bias_noise: standard dev for noise on the biases of hidden layers
            out_noise: standard dev for noise on the output of this net
            mlp_config: list of "layer descriptions"
    """
    def __init__(self,
            rng=None,
            input_noise=None,
            input_data=None,
            params=None):
        # First, setup a shared random number generator for this layer
        self.rng = theano.tensor.shared_randomstreams.RandomStreams( \
            rng.randint(100000))
        # Grab the symbolic input matrix
        self.input_noise = input_noise
        self.input_data = input_data
        #####################################################
        # Process user-supplied parameters for this network #
        #####################################################
        lam_l2a = params['lam_l2a']
        if 'vis_drop' in params:
            self.vis_drop = params['vis_drop']
        else:
            self.vis_drop = 0.0
        if 'hid_drop' in params:
            self.hid_drop = params['hid_drop']
        else:
            self.hid_drop = 0.0
        if 'bias_noise' in params:
            self.bias_noise = params['bias_noise']
        else:
            self.bias_noise = 0.0
        if 'out_noise' in params:
            self.out_noise = params['out_noise']
        else:
            self.out_noise = 0.0
        # Get the configuration/prototype for this network. The config is a
        # list of layer descriptions, including a description for the input
        # layer, which is typically just the dimension of the inputs. So, the
        # depth of the mlp is one less than the number of layer configs.
        self.mlp_config = params['mlp_config']
        self.mlp_depth = len(self.mlp_config) - 1
        self.latent_dim = self.mlp_config[0]
        self.data_dim = self.mlp_config[-1]
        ##########################
        # Initialize the network #
        ##########################
        self.clip_params = {}
        self.mlp_layers = []
        layer_def_pairs = zip(self.mlp_config[:-1],self.mlp_config[1:])
        layer_num = 0
        next_input = self.input_noise
        for in_def, out_def in layer_def_pairs:
            first_layer = (layer_num == 0)
            last_layer = (layer_num == (len(layer_def_pairs) - 1))
            l_name = "gn_layer_{0:d}".format(layer_num)
            if (type(in_def) is list) or (type(in_def) is tuple):
                # Receiving input from a poolish layer...
                in_dim = in_def[0]
            else:
                # Receiving input from a normal layer...
                in_dim = in_def
            if (type(out_def) is list) or (type(out_def) is tuple):
                # Applying some sort of pooling in this layer...
                out_dim = out_def[0]
                pool_size = out_def[1]
            else:
                # Not applying any pooling in this layer...
                out_dim = out_def
                pool_size = 0
            # Select the appropriate noise to add to this layer
            if first_layer:
                d_rate = self.vis_drop
            else:
                d_rate = self.hid_drop
            if last_layer:
                b_noise = self.out_noise
            else:
                b_noise = self.bias_noise
            # Add a new, well-configured layer to the regular model
            self.mlp_layers.append(HiddenLayer(rng=rng, \
                    input=next_input, activation=None, pool_size=pool_size, \
                    drop_rate=d_rate, input_noise=0., bias_noise=b_noise, \
                    in_dim=in_dim, out_dim=out_dim, \
                    name=l_name, W_scale=2.0))
            next_input = self.mlp_layers[-1].output
            # Set the non-bias parameters of this layer to be clipped
            self.clip_params[self.mlp_layers[-1].W] = 1
            # Acknowledge layer completion
            layer_num = layer_num + 1

        # set norms to which to clip various parameters
        self.clip_norms = {}

        # Mash all the parameters together, into a list.
        self.mlp_params = []
        for layer in self.mlp_layers:
            self.mlp_params.extend(layer.params)

        # The output of this generator network is given by the noisy output
        # of its final layer. We will keep a running estimate of the mean and
        # covariance of the distribution induced by combining this network's
        # latent noise source with its deep non-linear transform. These will
        # be used to encourage the induced distribution to match the first and
        # second-order moments of the distribution we are trying to match.
        #self.output_noise = self.mlp_layers[-1].noisy_linear
        self.output_noise = T.nnet.sigmoid(self.mlp_layers[-1].noisy_linear)
        self.out_dim = self.mlp_layers[-1].out_dim
        C_init = np.zeros((self.out_dim,self.out_dim)).astype(theano.config.floatX)
        m_init = np.zeros((self.out_dim,)).astype(theano.config.floatX)
        self.dist_mean = theano.shared(m_init, name='gn_dist_mean')
        self.dist_cov = theano.shared(C_init, name='gn_dist_cov')
        # Get simple regularization penalty to moderate activation dynamics
        self.act_reg_cost = lam_l2a * self._act_reg_cost()
        # Joint the transformed noise output and the real data input
        self.output = T.vertical_stack(self.input_data, self.output_noise)
        return

    def _act_reg_cost(self):
        """Apply L2 regularization to the activations in each spawn-net."""
        act_sq_sums = []
        for layer in self.mlp_layers:
            act_sq_sums.append(layer.act_l2_sum)
        full_act_sq_sum = T.sum(act_sq_sums)
        return full_act_sq_sum

    def _batch_moments(self):
        """
        Compute covariance and mean of the current sample outputs.
        """
        mu = T.mean(self.output_noise, axis=0, keepdims=True)
        sigma = T.dot((self.output_noise.T - mu.T), (self.output_noise - mu))
        return [mu, sigma]

    def init_moments(self, X_noise):
        """
        Initialize the running mean and covariance estimates.
        """
        X_noise_sym = T.matrix()
        out_func = theano.function(inputs=[ X_noise_sym ], \
                outputs=[ self.output_noise ], \
                givens={self.input_noise: X_noise_sym})
        # Compute outputs for the input latent noise matrix
        X_out = out_func(X_noise.astype(theano.config.floatX))[0]
        # Compute mean and covariance of the outputs
        mu = np.mean(X_out, axis=0)
        X_out_minus_mu = X_out - mu
        sigma = np.dot(X_out_minus_mu.T,X_out_minus_mu) / X_out.shape[0]
        # Initialize the network's running estimates 
        self.dist_cov.set_value(sigma.astype(theano.config.floatX))
        self.dist_mean.set_value(mu.astype(theano.config.floatX))
        return



#############################################
# HELPER FUNCTION FOR 1st/2nd ORDER MOMENTS #
#############################################

def projected_moments(X, P, ary_type=None):
    """
    Compute 1st/2nd-order moments after linear transform.

    Return type is always a numpy array. Inputs should both be of the same
    type, which can be either numpy array or theano shared variable.
    """
    assert(not (ary_type is None))
    assert((ary_type == 'theano') or (ary_type == 'numpy'))
    proj_mean = None
    proj_cov = None
    if ary_type == 'theano':
        Xp = T.dot(X, P)
        Xp_mean = T.mean(Xp, axis=0)
        Xp_centered = Xp - Xp_mean
        Xp_cov = T.dot(Xp_centered.T, Xp_centered) / Xp.shape[0]
        proj_mean = Xp_mean.eval()
        proj_cov = Xp_cov.eval()
    else:
        Xp = np.dot(X, P)
        Xp_mean = np.mean(Xp, axis=0)
        Xp_centered = Xp - Xp_mean
        Xp_cov = np.dot(Xp_centered.T, Xp_centered) / Xp.shape[0]
        proj_mean = Xp_mean
        proj_cov = Xp_cov
    return [proj_mean, proj_cov]










##############
# EYE BUFFER #
##############