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
    given some deep, directed generative model. The output of this network
    comprises two constructs: an approximate mean vector and an approximate
    standard deviation vector (i.e. diagonal matrix) for a Gaussian posterior.

    Parameters:
        rng: a numpy.random RandomState object
        input_cont: symbolic input matrix for inputting control data
        input_late: symbolic input matrix for inputting latent data
        input_mask: symbolic input matrix for a mask on which values to take
                    from input_cont and which to take from input_late
        params: a dict of parameters describing the desired ensemble:
            lam_l2a: L2 regularization weight on neuron activations
            vis_drop: drop rate to use on observable variables
            hid_drop: drop rate to use on hidden layer activations
                -- note: vis_drop/hid_drop are optional, with defaults 0.0/0.0
            bias_noise: standard dev for noise on the biases of hidden layers
            out_noise: standard dev for noise on the output of this net
            shared_config: list of "layer descriptions" for shared part
            mu_config: list of "layer descriptions" for mu part
            sigma_config: list of "layer descriptions" for sigma part
        mlp_param_dicts: parameters for the MLP controlled by this INF_NET
    """
    def __init__(self, \
            rng=None, \
            input_cont=None, \
            input_late=None, \
            input_mask=None, \
            params=None, \
            mlp_param_dicts=None):
        # First, setup a shared random number generator for this layer
        self.rng = theano.tensor.shared_randomstreams.RandomStreams( \
            rng.randint(100000))
        # Grab the symbolic input matrix
        self.input_cont = input_cont
        self.input_late = input_late
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
        if 'input_noise' in params:
            self.input_noise = params['input_noise']
        else:
            self.input_noise = 0.0
        # Check if the params for this net were given a priori. This option
        # will be used for creating "clones" of a generative network, with all
        # of the network parameters shared between clones.
        if mlp_param_dicts is None:
            # This is not a clone, and we will need to make a dict for
            # referring to the parameters of each network layer
            self.mlp_param_dicts = {'shared': [], 'mu': [], 'sigma': []}
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
        self.shared_config = params['shared_config']
        self.mu_config = params['mu_config']
        self.sigma_config = params['sigma_config']
        #########################################
        # Initialize the shared part of network #
        #########################################
        self.clip_params = {}
        self.shared_layers = []
        layer_def_pairs = zip(self.shared_config[:-1],self.shared_config[1:])
        layer_num = 0
        # Construct input by combining control input and latent input, taking
        # masked values from inferred input and other values from data input
        next_input = ((1.0 - self.input_mask) * self.input_cont) + \
                (self.input_mask * self.input_late)
        for in_def, out_def in layer_def_pairs:
            first_layer = (layer_num == 0)
            last_layer = (layer_num == (len(layer_def_pairs) - 1))
            l_name = "share_layer_{0:d}".format(layer_num)
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
            if first_layer:
                i_noise = self.input_noise
                b_noise = 0.0
            else:
                i_noise = 0.0
                b_noise = self.bias_noise
            if not self.is_clone:
                ##########################################
                # Initialize a layer with new parameters #
                ##########################################
                new_layer = HiddenLayer(rng=rng, \
                        input=next_input, activation=None, pool_size=pool_size, \
                        drop_rate=d_rate, input_noise=i_noise, bias_noise=b_noise, \
                        in_dim=in_dim, out_dim=out_dim, \
                        name=l_name, W_scale=1.0)
                self.shared_layers.append(new_layer)
                self.mlp_param_dicts['shared'].append({'W': new_layer.W, 'b': new_layer.b})
            else:
                ##################################################
                # Initialize a layer with some shared parameters #
                ##################################################
                init_params = self.mlp_param_dicts['shared'][layer_num]
                new_layer = HiddenLayer(rng=rng, \
                        input=next_input, activation=None, pool_size=pool_size, \
                        drop_rate=d_rate, input_noise=i_noise, bias_noise=b_noise, \
                        in_dim=in_dim, out_dim=out_dim, \
                        W=init_params['W'], b=init_params['b'], \
                        name=l_name, W_scale=1.0)
                self.shared_layers.append(new_layer)
            next_input = self.shared_layers[-1].output
            # Set the non-bias parameters of this layer to be clipped
            self.clip_params[self.shared_layers[-1].W] = 1
            # Acknowledge layer completion
            layer_num = layer_num + 1
        #####################################
        # Initialize the mu part of network #
        #####################################
        self.mu_layers = []
        layer_def_pairs = zip(self.mu_config[:-1],self.mu_config[1:])
        layer_num = 0
        # Take input from the output of the shared network
        next_input = self.shared_layers[-1].output
        for in_def, out_def in layer_def_pairs:
            first_layer = (layer_num == 0)
            last_layer = (layer_num == (len(layer_def_pairs) - 1))
            l_name = "mu_layer_{0:d}".format(layer_num)
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
            d_rate = self.hid_drop
            i_noise = 0.0
            b_noise = self.bias_noise
            if not self.is_clone:
                ##########################################
                # Initialize a layer with new parameters #
                ##########################################
                new_layer = HiddenLayer(rng=rng, \
                        input=next_input, activation=None, pool_size=pool_size, \
                        drop_rate=d_rate, input_noise=i_noise, bias_noise=b_noise, \
                        in_dim=in_dim, out_dim=out_dim, \
                        name=l_name, W_scale=1.0)
                self.mu_layers.append(new_layer)
                self.mlp_param_dicts['mu'].append({'W': new_layer.W, 'b': new_layer.b})
            else:
                ##################################################
                # Initialize a layer with some shared parameters #
                ##################################################
                init_params = self.mlp_param_dicts['mu'][layer_num]
                new_layer = HiddenLayer(rng=rng, \
                        input=next_input, activation=None, pool_size=pool_size, \
                        drop_rate=d_rate, input_noise=i_noise, bias_noise=b_noise, \
                        in_dim=in_dim, out_dim=out_dim, \
                        W=init_params['W'], b=init_params['b'], \
                        name=l_name, W_scale=1.0)
                self.mu_layers.append(new_layer)
            next_input = self.mu_layers[-1].output
            # Set the non-bias parameters of this layer to be clipped
            self.clip_params[self.mu_layers[-1].W] = 1
            # Acknowledge layer completion
            layer_num = layer_num + 1
        ########################################
        # Initialize the sigma part of network #
        ########################################
        self.sigma_layers = []
        layer_def_pairs = zip(self.sigma_config[:-1],self.sigma_config[1:])
        layer_num = 0
        # Take input from the output of the shared network
        next_input = self.shared_layers[-1].output
        for in_def, out_def in layer_def_pairs:
            first_layer = (layer_num == 0)
            last_layer = (layer_num == (len(layer_def_pairs) - 1))
            l_name = "sigma_layer_{0:d}".format(layer_num)
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
            d_rate = self.hid_drop
            i_noise = 0.0
            b_noise = self.bias_noise
            if not self.is_clone:
                ##########################################
                # Initialize a layer with new parameters #
                ##########################################
                new_layer = HiddenLayer(rng=rng, \
                        input=next_input, activation=None, pool_size=pool_size, \
                        drop_rate=d_rate, input_noise=i_noise, bias_noise=b_noise, \
                        in_dim=in_dim, out_dim=out_dim, \
                        name=l_name, W_scale=1.0)
                self.sigma_layers.append(new_layer)
                self.mlp_param_dicts['sigma'].append({'W': new_layer.W, 'b': new_layer.b})
            else:
                ##################################################
                # Initialize a layer with some shared parameters #
                ##################################################
                init_params = self.mlp_param_dicts['sigma'][layer_num]
                new_layer = HiddenLayer(rng=rng, \
                        input=next_input, activation=None, pool_size=pool_size, \
                        drop_rate=d_rate, input_noise=i_noise, bias_noise=b_noise, \
                        in_dim=in_dim, out_dim=out_dim, \
                        W=init_params['W'], b=init_params['b'], \
                        name=l_name, W_scale=1.0)
                self.sigma_layers.append(new_layer)
            next_input = self.sigma_layers[-1].output
            # Set the non-bias parameters of this layer to be clipped
            self.clip_params[self.sigma_layers[-1].W] = 1
            # Acknowledge layer completion
            layer_num = layer_num + 1

        # Mash all the parameters together, into a list.
        self.mlp_params = []
        for layer in self.shared_layers:
            self.mlp_params.extend(layer.params)
        for layer in self.mu_layers:
            self.mlp_params.extend(layer.params)
        for layer in self.sigma_layers:
            self.mlp_params.extend(layer.params)

        # The output of this inference network is given by the noisy output
        # of the final layers of its mu and sigma networks.
        self.output_mu = self.mu_layers[-1].noisy_linear
        self.output_sigma = T.log(1.0 + T.exp(self.sigma_layers[-1].noisy_linear))
        self.out_dim = self.sigma_layers[-1].out_dim
        # Get simple regularization penalty to moderate activation dynamics
        self.act_reg_cost = lam_l2a * self._act_reg_cost()
        return

    def _act_reg_cost(self):
        """
        Apply L2 regularization to the activations in each net.
        """
        act_sq_sums = []
        for layer in self.shared_layers:
            act_sq_sums.append(layer.act_l2_sum)
        for layer in self.mu_layers:
            act_sq_sums.append(layer.act_l2_sum)
        for layer in self.sigma_layers:
            act_sq_sums.append(layer.act_l2_sum)
        full_act_sq_sum = T.sum(act_sq_sums)
        return full_act_sq_sum

    def shared_param_clone(self, rng=None, input_cont=None, input_late=None, \
            input_mask=None):
        """
        Return a clone of this network, with shared parameters but with a 
        different symbolic input var.

        This can be used for "unrolling" a generate->infer->generate->infer...
        loop. Then, we can do backprop through time for various objectives.
        """
        clone_net = INF_NET(rng=rng, input_cont=input_cont, \
                input_late=input_late, input_mask=input_mask, \
                params=self.params, mlp_param_dicts=self.mlp_param_dicts)
        return clone_net

if __name__=="__main__":
    # Do basic testing, to make sure classes aren't completely broken.
    input_cont = T.matrix('CONTROL_INPUT')
    input_mask = T.matrix('MASK_INPUT')
    input_late_1 = T.matrix('LATE_INPUT_1')
    # Initialize a source of randomness
    rng = np.random.RandomState(1234)
    # Choose some parameters for the generative network
    in_params = {}
    shared_config = [28*28, 500]
    gauss_config = [shared_config[-1], 500, 100]
    mu_config = gauss_config
    sigma_config = gauss_config
    in_params['shared_config'] = shared_config
    in_params['mu_config'] = mu_config
    in_params['sigma_config'] = sigma_config
    in_params['lam_l2a'] = 1e-3
    in_params['vis_drop'] = 0.0
    in_params['hid_drop'] = 0.0
    in_params['bias_noise'] = 0.0
    in_params['input_noise'] = 0.0
    # Make the starter network
    in_1 = INF_NET(rng=rng, input_cont=input_cont, \
            input_late=input_late_1, input_mask=input_mask, \
            params=in_params, mlp_param_dicts=None)
    # Make a clone of the network with a different symbolic input
    input_late_2 = T.matrix('LATE_INPUT_2')
    in_2 = INF_NET(rng=rng, input_cont=input_cont, \
            input_late=input_late_2, input_mask=input_mask, \
            params=in_params, mlp_param_dicts=in_1.mlp_param_dicts)
    print("TESTING COMPLETE")