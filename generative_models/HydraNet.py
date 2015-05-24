##################################################################
# Code for networks and whatnot to use in variationalish stuff.  #
##################################################################

# basic python
import numpy as np
import numpy.random as npr
from collections import OrderedDict
import cPickle

# theano business
import theano
import theano.tensor as T
#from theano.tensor.shared_randomstreams import RandomStreams as RandStream
from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams as RandStream

# phil's sweetness
from NetLayers import HiddenLayer, relu_actfun, softplus_actfun
from HelperFuncs import constFX, to_fX

####################################
# INFREENCE NETWORK IMPLEMENTATION #
####################################

def row_normalize(x):
    """
    Normalize rows of matrix x to unit (L2) norm.
    """
    x_normed = x / T.sqrt(T.sum(x**2.,axis=1,keepdims=1) + constFX(1e-8))
    return x_normed

def soft_abs(x, smoothing=1e-5):
    """
    Soft absolute value function applied to x.
    """
    sa_x = T.sqrt(x**2. + constFX(smoothing))
    return sa_x

class HydraNet(object):
    """
    A net that turns one input into multiple outputs.

    All hidden layers are shared between the outputs. The outputs can be
    of different dimension. Each output is computed as a linear function of
    the final shared layer.

    Parameters:
        rng: a numpy.random RandomState object
        Xd: symbolic input matrix for inputs
        params: a dict of parameters describing the desired network:
            vis_drop: drop rate to use on observable variables
            hid_drop: drop rate to use on hidden layer activations
                -- note: vis_drop/hid_drop are optional, with defaults 0.0/0.0
            input_noise: standard dev for noise on the input of this net
            bias_noise: standard dev for noise on the biases of hidden layers
            shared_config: list of "layer descriptions" for shared part
            output_config: list of dimensions for the output layers
            activation: "function handle" for the desired non-linearity
            init_scale: scaling factor for hidden layer weights (__ * 0.01)
        shared_param_dicts: parameters for this HydraNet
    """
    def __init__(self, \
            rng=None, \
            Xd=None, \
            params=None, \
            shared_param_dicts=None):
        # Setup a shared random generator for this network 
        self.rng = RandStream(rng.randint(1000000))
        # Grab the symbolic input matrix
        self.Xd = Xd
        #####################################################
        # Process user-supplied parameters for this network #
        #####################################################
        self.params = params
        if 'build_theano_funcs' in params:
            self.build_theano_funcs = params['build_theano_funcs']
        else:
            self.build_theano_funcs = True
        if 'vis_drop' in params:
            self.vis_drop = params['vis_drop']
        else:
            self.vis_drop = 0.0
        if 'hid_drop' in params:
            self.hid_drop = params['hid_drop']
        else:
            self.hid_drop = 0.0
        if 'input_noise' in params:
            self.input_noise = params['input_noise']
        else:
            self.input_noise = 0.0
        if 'bias_noise' in params:
            self.bias_noise = params['bias_noise']
        else:
            self.bias_noise = 0.0
        if 'init_scale' in params:
            self.init_scale = params['init_scale']
        else:
            self.init_scale = 1.0
        # Check if the params for this net were given a priori. This option
        # will be used for creating "clones" of an inference network, with all
        # of the network parameters shared between clones.
        if shared_param_dicts is None:
            # This is not a clone, and we will need to make a dict for
            # referring to the parameters of each network layer
            self.shared_param_dicts = {'shared': [], 'output': []}
            self.is_clone = False
        else:
            # This is a clone, and its layer parameters can be found by
            # referring to the given param dict (i.e. shared_param_dicts).
            self.shared_param_dicts = shared_param_dicts
            self.is_clone = True
        # Get the configuration/prototype for this network. The config is a
        # list of layer descriptions, including a description for the input
        # layer, which is typically just the dimension of the inputs. So, the
        # depth of the mlp is one less than the number of layer configs.
        self.shared_config = params['shared_config']
        self.output_config = params['output_config']
        if 'activation' in params:
            self.activation = params['activation']
        else:
            self.activation = relu_actfun
        #########################################
        # Initialize the shared part of network #
        #########################################
        self.shared_layers = []
        layer_def_pairs = zip(self.shared_config[:-1],self.shared_config[1:])
        layer_num = 0
        # Construct input to the inference network
        next_input = self.Xd
        for in_def, out_def in layer_def_pairs:
            first_layer = (layer_num == 0)
            last_layer = (layer_num == (len(layer_def_pairs) - 1))
            l_name = "shared_layer_{0:d}".format(layer_num)
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
            # set in-bound weights to have norm self.init_scale
            i_scale = self.init_scale
            if not self.is_clone:
                ##########################################
                # Initialize a layer with new parameters #
                ##########################################
                new_layer = HiddenLayer(rng=rng, input=next_input, \
                        activation=self.activation, pool_size=pool_size, \
                        drop_rate=d_rate, input_noise=i_noise, bias_noise=b_noise, \
                        in_dim=in_dim, out_dim=out_dim, \
                        name=l_name, W_scale=i_scale)
                self.shared_layers.append(new_layer)
                self.shared_param_dicts['shared'].append( \
                        new_layer.shared_param_dicts)
            else:
                ##################################################
                # Initialize a layer with some shared parameters #
                ##################################################
                init_params = self.shared_param_dicts['shared'][layer_num]
                new_layer = HiddenLayer(rng=rng, input=next_input, \
                        activation=self.activation, pool_size=pool_size, \
                        drop_rate=d_rate, input_noise=i_noise, bias_noise=b_noise, \
                        in_dim=in_dim, out_dim=out_dim, \
                        W=init_params['W'], b=init_params['b'], \
                        b_in=init_params['b_in'], s_in=init_params['s_in'], \
                        name=l_name, W_scale=i_scale)
                self.shared_layers.append(new_layer)
            next_input = self.shared_layers[-1].output
            # Acknowledge layer completion
            layer_num = layer_num + 1
        ################################
        # Initialize the output layers #
        ################################
        self.output_layers = []
        # take input from the output of the shared network
        in_dim = self.shared_layers[-1].out_dim
        ol_input = self.shared_layers[-1].output
        for ol_num, out_dim in enumerate(self.output_config):
            ol_name = "output_layer_{0:d}".format(ol_num)
            # Select the appropriate noise to add to this layer
            pool_size = 0
            d_rate = self.hid_drop
            i_noise = 0.0
            b_noise = self.bias_noise
            i_scale = self.init_scale
            if not self.is_clone:
                ##########################################
                # Initialize a layer with new parameters #
                ##########################################
                new_layer = HiddenLayer(rng=rng, input=ol_input, \
                        activation=self.activation, pool_size=pool_size, \
                        drop_rate=d_rate, input_noise=i_noise, bias_noise=b_noise, \
                        in_dim=in_dim, out_dim=out_dim, \
                        name=l_name, W_scale=i_scale)
                self.output_layers.append(new_layer)
                self.shared_param_dicts['output'].append( \
                        new_layer.shared_param_dicts)
            else:
                ##################################################
                # Initialize a layer with some shared parameters #
                ##################################################
                init_params = self.shared_param_dicts['output'][ol_num]
                new_layer = HiddenLayer(rng=rng, input=ol_input, \
                        activation=self.activation, pool_size=pool_size, \
                        drop_rate=d_rate, input_noise=i_noise, bias_noise=b_noise, \
                        in_dim=in_dim, out_dim=out_dim, \
                        W=init_params['W'], b=init_params['b'], \
                        b_in=init_params['b_in'], s_in=init_params['s_in'], \
                        name=l_name, W_scale=i_scale)
                self.output_layers.append(new_layer)

        # mash all the parameters together, into a list.
        self.mlp_params = []
        for layer in self.shared_layers:
            self.mlp_params.extend(layer.params)
        for layer in self.output_layers:
            self.mlp_params.extend(layer.params)

        # create a symbolic handle for the outputs of this net
        self.outputs = self.apply(Xd)
        return

    def apply(self, X, use_bn=False, use_drop=False):
        """
        Pass input X through this HydraNet and get the resulting outputs.
        """
        # pass activations through the shared layers
        shared_acts = [X]
        for layer in self.shared_layers:
            _, _, layer_acts = layer.apply(shared_acts[-1], \
                                       use_bn=use_bn, use_drop=use_drop)
            shared_acts.append(layer_acts)
        shared_output = shared_acts[-1]
        # compute outputs of the output layers
        outputs = []
        for layer in self.output_layers:
            _, layer_acts, _ = layer.apply(shared_output, \
                                       use_bn=use_bn, use_drop=use_drop)
            outputs.append(layer_acts)
        return outputs

    def apply_shared(self, X):
        """
        Pass input X through this HydraNet's shared layers.
        """
        # pass activations through the shared layers
        shared_acts = [X]
        for layer in self.shared_layers:
            _, _, layer_acts = layer.apply(shared_acts[-1])
            shared_acts.append(layer_acts)
        shared_output = shared_acts[-1]
        return shared_output

    def set_bias_noise(self, bias_noise=0.0):
        """
        Set the bias noise in all hidden layers to the given value.
        """
        new_ary = np.zeros((1,)) + bias_noise
        new_bn = to_fX( new_ary )
        for layer in self.shared_layers:
            layer.bias_noise.set_value(new_bn)
        for layer in self.output_layers:
            layer.bias_noise.set_value(new_bn)
        return

    def init_biases(self, b_init=0.0, b_std=1e-2):
        """
        Initialize the biases in all shred layers to some constant.
        """
        for layer in self.shared_layers:
            b_vec = (0.0 * layer.b.get_value(borrow=False)) + b_init
            b_vec = b_vec + (b_std * npr.randn(*b_vec.shape))
            layer.b.set_value(to_fX(b_vec))
        return

    def shared_param_clone(self, rng=None, Xd=None):
        """
        Return a clone of this network, with shared parameters but with
        different symbolic input variables.
        """
        clone_net = HydraNet(rng=rng, Xd=Xd, params=self.params, \
                shared_param_dicts=self.shared_param_dicts)
        return clone_net

    def forked_param_clone(self, rng=None, Xd=None):
        """
        Return a clone of this network, with forked copies of the current
        shared parameters of this HydraNet, with different symbolic inputs.
        """
        new_spds = {}
        old_spds = self.shared_param_dicts
        # shared param dicts is nested like: dict of list of dicts
        # i.e., spd[k] is a list and spd[k][i] is a dict
        for k1 in old_spds:
            new_spds[k1] = []
            for i in range(len(old_spds[k1])):
                new_spds[k1].append({})
                for k2 in old_spds[k1][i]:
                    old_sp = old_spds[k1][i][k2]
                    old_sp_forked = old_sp.get_value(borrow=False)
                    new_sp = theano.shared(value=old_sp_forked)
                    new_spds[k1][i][k2] = new_sp
        clone_net = HydraNet(rng=rng, Xd=Xd, params=self.params, \
                shared_param_dicts=new_spds)
        return clone_net

    def save_to_file(self, f_name=None):
        """
        Dump important stuff to a Python pickle, so that we can reload this
        model later. We'll pickle everything required to create a clone of
        this model given the pickle and the rng/Xd params to the cloning
        function: "HydraNet.shared_param_clone()".
        """
        assert(not (f_name is None))
        f_handle = file(f_name, 'wb')
        # dump the dict self.params, which just holds "simple" python values
        cPickle.dump(self.params, f_handle, protocol=-1)
        # make a copy of self.shared_param_dicts, with numpy arrays in place
        # of the theano shared variables
        numpy_param_dicts = {'shared': [], 'output': []}
        for layer_group in ['shared', 'output']:
            for shared_dict in self.shared_param_dicts[layer_group]:
                numpy_dict = {}
                for key in shared_dict:
                    numpy_dict[key] = shared_dict[key].get_value(borrow=False)
                numpy_param_dicts[layer_group].append(numpy_dict)
        # dump the numpy version of self.shared_param_dicts to pickle file
        cPickle.dump(numpy_param_dicts, f_handle, protocol=-1)
        f_handle.close()
        return

    def save_to_dict(self):
        """
        Dump important stuff to a dict capable of rebooting the model.
        """
        model_dict = {}
        # dump the dict self.params, which just holds "simple" python values
        model_dict['params'] = self.params
        # make a copy of self.shared_param_dicts, with numpy arrays in place
        # of the theano shared variables
        numpy_param_dicts = {'shared': [], 'output': []}
        for layer_group in ['shared', 'output']:
            for shared_dict in self.shared_param_dicts[layer_group]:
                numpy_dict = {}
                for key in shared_dict:
                    numpy_dict[key] = shared_dict[key].get_value(borrow=False)
                numpy_param_dicts[layer_group].append(numpy_dict)
        # dump the numpy version of self.shared_param_dicts to the dict
        model_dict['numpy_param_dicts'] = numpy_param_dicts
        return model_dict

def load_hydranet_from_file(f_name=None, rng=None, Xd=None, \
                            new_params=None):
    """
    Load a clone of some previously trained model.
    """
    assert(not (f_name is None))
    pickle_file = open(f_name)
    # load basic parameters
    self_dot_params = cPickle.load(pickle_file)
    if not (new_params is None):
        for k in new_params:
            self_dot_params[k] = new_params[k]
    # load numpy arrays that will be converted to Theano shared arrays
    self_dot_numpy_param_dicts = cPickle.load(pickle_file)
    self_dot_shared_param_dicts = {'shared': [], 'output': []}
    for layer_group in ['shared', 'output']:
        # go over the list of parameter dicts in this layer group
        for numpy_dict in self_dot_numpy_param_dicts[layer_group]:
            shared_dict = {}
            for key in numpy_dict:
                # convert each numpy array to a Theano shared array
                val = to_fX(numpy_dict[key])
                shared_dict[key] = theano.shared(val)
            self_dot_shared_param_dicts[layer_group].append(shared_dict)
    # now, create a HydraNet with the configuration we just unpickled
    clone_net = HydraNet(rng=rng, Xd=Xd, params=self_dot_params, \
                         shared_param_dicts=self_dot_shared_param_dicts)
    # helpful output
    print("==================================================")
    print("LOADED HydraNet WITH PARAMS:")
    for k in self_dot_params:
        print("    {0:s}: {1:s}".format(str(k), str(self_dot_params[k])))
    print("==================================================")
    return clone_net

def load_hydranet_from_dict(model_dict, rng=None, Xd=None, \
                            new_params=None):
    """
    Load a clone of some previously trained model.
    """
    # load basic parameters
    self_dot_params = model_dict['params']
    if not (new_params is None):
        for k in new_params:
            self_dot_params[k] = new_params[k]
    # load numpy arrays that will be converted to Theano shared arrays
    self_dot_numpy_param_dicts = model_dict['numpy_param_dicts']
    self_dot_shared_param_dicts = {'shared': [], 'output': []}
    for layer_group in ['shared', 'output']:
        # go over the list of parameter dicts in this layer group
        for numpy_dict in self_dot_numpy_param_dicts[layer_group]:
            shared_dict = {}
            for key in numpy_dict:
                # convert each numpy array to a Theano shared array
                val = to_fX(numpy_dict[key])
                shared_dict[key] = theano.shared(val)
            self_dot_shared_param_dicts[layer_group].append(shared_dict)
    # now, create a HydraNet with the configuration we just unpacked
    clone_net = HydraNet(rng=rng, Xd=Xd, params=self_dot_params, \
                         shared_param_dicts=self_dot_shared_param_dicts)
    # helpful output
    print("==================================================")
    print("LOADED HydraNet WITH PARAMS:")
    for k in self_dot_params:
        print("    {0:s}: {1:s}".format(str(k), str(self_dot_params[k])))
    print("==================================================")
    return clone_net







if __name__=="__main__":
    # Derp
    print("NO TEST/DEMO CODE FOR NOW.")
