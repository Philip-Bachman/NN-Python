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

#####################################
# GENERATIVE NETWORK IMPLEMENTATION #
#####################################

class GEN_NET(object):
    """
    A net that transforms a simple distribution so that it matches some
    more complicated distribution, for some definition of match....

    Parameters:
        rng: a numpy.random RandomState object
        input_var: symbolic matrix for inputting latent variable samples
        params: a dict of parameters describing the desired network:
            lam_l2a: L2 regularization weight on neuron activations
            vis_drop: drop rate to use on the latent variable space
            hid_drop: drop rate to use on the hidden layer activations
                -- note: vis_drop/hid_drop are optional, with defaults 0.0/0.0
            bias_noise: standard dev for noise on the biases of hidden layers
            out_noise: standard dev for noise on the output of this net
            mlp_config: list of "layer descriptions"
        mlp_param_dicts: parameters for the MLP controlled by this GEN_NET
    """
    def __init__(self, \
            rng=None, \
            input_var=None, \
            params=None, \
            mlp_param_dicts=None):
        # First, setup a shared random number generator for this layer
        self.rng = theano.tensor.shared_randomstreams.RandomStreams( \
            rng.randint(100000))
        # Grab the symbolic input matrix
        self.input_var = input_var
        #####################################################
        # Process user-supplied parameters for this network #
        #####################################################
        assert(not (params is None))
        self.params = params
        lam_l2a = self.params['lam_l2a']
        if 'vis_drop' in self.params:
            # Drop rate on the latent variables
            self.vis_drop = self.params['vis_drop']
        else:
            self.vis_drop = 0.0
        if 'hid_drop' in self.params:
            # Drop rate on hidden layer activations
            self.hid_drop = self.params['hid_drop']
        else:
            self.hid_drop = 0.0
        if 'bias_noise' in self.params:
            # Noise sigma for hidden layer biases
            self.bias_noise = self.params['bias_noise']
        else:
            self.bias_noise = 0.0
        if 'out_noise' in self.params:
            # Noise sigma for the output/observable layer
            self.out_noise = self.params['out_noise']
        else:
            self.out_noise = 0.0
        # Check if the params for this net were given a priori. This option
        # will be used for creating "clones" of a generative network, with all
        # of the network parameters shared between clones.
        if mlp_param_dicts is None:
            # This is not a clone, and we will need to make a dict for
            # referring to the parameters of each network layer
            self.mlp_param_dicts = []
            self.is_clone = False
        else:
            # This is a clone, and its layer parameters can be found by
            # referring to the given param dict (i.e. mlp_param_dicts).
            self.mlp_param_dicts = mlp_param_dicts
            self.is_clone = True
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
        next_input = self.input_var
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
            if not self.is_clone:
                ##########################################
                # Initialize a layer with new parameters #
                ##########################################
                new_layer = HiddenLayer(rng=rng, \
                        input=next_input, activation=None, pool_size=pool_size, \
                        drop_rate=d_rate, input_noise=0., bias_noise=b_noise, \
                        in_dim=in_dim, out_dim=out_dim, \
                        name=l_name, W_scale=1.0)
                self.mlp_layers.append(new_layer)
                self.mlp_param_dicts.append({'W': new_layer.W, 'b': new_layer.b})
            else:
                ##################################################
                # Initialize a layer with some shared parameters #
                ##################################################
                init_params = self.mlp_param_dicts[layer_num]
                self.mlp_layers.append(HiddenLayer(rng=rng, \
                        input=next_input, activation=None, pool_size=pool_size, \
                        drop_rate=d_rate, input_noise=0., bias_noise=b_noise, \
                        in_dim=in_dim, out_dim=out_dim, \
                        W=init_params['W'], b=init_params['b'], \
                        name=l_name, W_scale=1.0))
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
        self.output = self.output_noise
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
                givens={self.input_var: X_noise_sym})
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

    def shared_param_clone(self, rng=None, input_var=None):
        """
        Return a clone of this network, with shared parameters but with a 
        different symbolic input var.

        This can be used for "unrolling" a generate->infer->generate->infer...
        loop. Then, we can do backprop through time for various objectives.
        """
        clone_net = GEN_NET(rng=rng, input_var=input_var, \
                params=self.params, mlp_param_dicts=self.mlp_param_dicts)
        return clone_net

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






if __name__=="__main__":
    # Do basic testing, to make sure classes aren't completely broken.
    input_var_1 = T.matrix('INPUT_1')
    # Initialize a source of randomness
    rng = np.random.RandomState(1234)
    # Choose some parameters for the generative network
    gn_params = {}
    gn_config = [400, 1200, 1200, 28*28]
    gn_params['mlp_config'] = gn_config
    gn_params['lam_l2a'] = 1e-3
    gn_params['vis_drop'] = 0.5
    gn_params['hid_drop'] = 0.0
    gn_params['bias_noise'] = 0.1
    gn_params['out_noise'] = 0.1
    # Make the starter network
    gn_1 = GEN_NET(rng=rng, input_var=input_var_1, params=gn_params, \
            mlp_param_dicts=None)
    # Make a clone of the network with a different symbolic input
    input_var_2 = T.matrix('INPUT_2')
    gn_2 = gn_1.shared_param_clone(rng=rng, input_var=input_var_2)
    print("TESTING COMPLETE")