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
from NetLayers import HiddenLayer, DiscLayer, relu_actfun, \
                      softplus_actfun, safe_log

####################################
# INFREENCE NETWORK IMPLEMENTATION #
####################################


class InfNet(object):
    """
    A net that tries to infer an approximate posterior for some observation,
    given some deep, directed generative model. The output of this network
    comprises two constructs: an approximate mean vector and an approximate
    standard deviation vector (i.e. diagonal matrix) for a Gaussian posterior.

    Parameters:
        rng: a numpy.random RandomState object
        Xd: symbolic input matrix for inputting observable data
        Xc: symbolic input matrix for inputting control data
        Xm: symbolic input matrix for a mask on which values to take
                    from Xc and which to take from Xd
        prior_sigma: standard deviation of isotropic Gaussian prior that our
                     inferred posteriors will be penalized for deviating from.
        params: a dict of parameters describing the desired ensemble:
            lam_l2a: L2 regularization weight on neuron activations
            vis_drop: drop rate to use on observable variables
            hid_drop: drop rate to use on hidden layer activations
                -- note: vis_drop/hid_drop are optional, with defaults 0.0/0.0
            input_noise: standard dev for noise on the input of this net
            bias_noise: standard dev for noise on the biases of hidden layers
            shared_config: list of "layer descriptions" for shared part
            mu_config: list of "layer descriptions" for mu part
            sigma_config: list of "layer descriptions" for sigma part
            activation: "function handle" for the desired non-linearity
            init_scale: scaling factor for hidden layer weights (__ * 0.01)
        shared_param_dicts: parameters for the MLP controlled by this InfNet
    """
    def __init__(self, \
            rng=None, \
            Xd=None, \
            Xc=None, \
            Xm=None, \
            prior_sigma=None, \
            params=None, \
            shared_param_dicts=None):
        # Setup a shared random generator for this network 
        self.rng = RandStream(rng.randint(1000000))
        # Grab the symbolic input matrix
        self.Xd = Xd
        self.Xc = Xc
        self.Xm = Xm
        self.prior_sigma = prior_sigma
        #####################################################
        # Process user-supplied parameters for this network #
        #####################################################
        self.params = params
        self.lam_l2a = params['lam_l2a']
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
        if 'encoder' in params:
            self.encoder = params['encoder']
            self.use_encoder = True
            self.Xc_encoded = self.encoder(self.Xc)
            self.Xd_encoded = self.encoder(self.Xd)
        else:
            self.encoder = lambda x: x
            self.use_encoder = False
            self.Xc_encoded = self.encoder(self.Xc)
            self.Xd_encoded = self.encoder(self.Xd)
        # Check if the params for this net were given a priori. This option
        # will be used for creating "clones" of an inference network, with all
        # of the network parameters shared between clones.
        if shared_param_dicts is None:
            # This is not a clone, and we will need to make a dict for
            # referring to the parameters of each network layer
            self.shared_param_dicts = {'shared': [], 'mu': [], 'sigma': []}
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
        self.mu_config = params['mu_config']
        self.sigma_config = params['sigma_config']
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
        # Construct input by combining data input and control input, taking
        # unmasked values from data input and others from the control input
        masked_input = ((1.0 - self.Xm) * self.Xd) + \
                (self.Xm * self.Xc)
        if self.use_encoder:
            next_input = self.encoder(masked_input)
        else:
            next_input = masked_input
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
            # set in-bound weights to have norm self.init_scale
            i_scale = (100.0 / np.sqrt(in_dim)) * self.init_scale
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
                self.shared_param_dicts['shared'].append({'W': new_layer.W, 'b': new_layer.b})
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
                        name=l_name, W_scale=i_scale)
                self.shared_layers.append(new_layer)
            next_input = self.shared_layers[-1].output
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
            # set in-bound weights to have norm self.init_scale
            i_scale = (100.0 / np.sqrt(in_dim)) * self.init_scale
            if not self.is_clone:
                ##########################################
                # Initialize a layer with new parameters #
                ##########################################
                new_layer = HiddenLayer(rng=rng, input=next_input, \
                        activation=self.activation, pool_size=pool_size, \
                        drop_rate=d_rate, input_noise=i_noise, bias_noise=b_noise, \
                        in_dim=in_dim, out_dim=out_dim, \
                        name=l_name, W_scale=i_scale)
                self.mu_layers.append(new_layer)
                self.shared_param_dicts['mu'].append({'W': new_layer.W, 'b': new_layer.b})
            else:
                ##################################################
                # Initialize a layer with some shared parameters #
                ##################################################
                init_params = self.shared_param_dicts['mu'][layer_num]
                new_layer = HiddenLayer(rng=rng, input=next_input, \
                        activation=self.activation, pool_size=pool_size, \
                        drop_rate=d_rate, input_noise=i_noise, bias_noise=b_noise, \
                        in_dim=in_dim, out_dim=out_dim, \
                        W=init_params['W'], b=init_params['b'], \
                        name=l_name, W_scale=i_scale)
                self.mu_layers.append(new_layer)
            next_input = self.mu_layers[-1].output
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
            # set in-bound weights to have norm self.init_scale
            i_scale = (100.0 / np.sqrt(in_dim)) * self.init_scale
            if last_layer:
                # set in-bound weights for logvar predictions to 0
                i_scale = 0.0
            if not self.is_clone:
                ##########################################
                # Initialize a layer with new parameters #
                ##########################################
                new_layer = HiddenLayer(rng=rng, input=next_input, \
                        activation=self.activation, pool_size=pool_size, \
                        drop_rate=d_rate, input_noise=i_noise, bias_noise=b_noise, \
                        in_dim=in_dim, out_dim=out_dim, \
                        name=l_name, W_scale=i_scale)
                self.sigma_layers.append(new_layer)
                self.shared_param_dicts['sigma'].append({'W': new_layer.W, 'b': new_layer.b})
            else:
                ##################################################
                # Initialize a layer with some shared parameters #
                ##################################################
                init_params = self.shared_param_dicts['sigma'][layer_num]
                new_layer = HiddenLayer(rng=rng, input=next_input, \
                        activation=self.activation, pool_size=pool_size, \
                        drop_rate=d_rate, input_noise=i_noise, bias_noise=b_noise, \
                        in_dim=in_dim, out_dim=out_dim, \
                        W=init_params['W'], b=init_params['b'], \
                        name=l_name, W_scale=i_scale)
                self.sigma_layers.append(new_layer)
            next_input = self.sigma_layers[-1].output
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
        self.output_logvar = self.sigma_layers[-1].noisy_linear
        self.output_sigma = T.exp(0.5 * self.output_logvar)
        # We'll also construct an output containing a single samples from each
        # of the distributions represented by the rows of self.output_mu and
        # self.output_sigma.
        self.output = self._construct_post_samples()
        self.out_dim = self.sigma_layers[-1].out_dim
        # Get simple regularization penalty to moderate activation dynamics
        self.act_reg_cost = self.lam_l2a * self._act_reg_cost()
        # Construct a function for penalizing KL divergence between the
        # approximate posteriors produced by this model and some isotropic
        # Gaussian distribution.
        self.kld_cost = self._construct_kld_cost()
        # Construct a theano function for sampling from the approximate
        # posteriors inferred by this model for some collection of points
        # in the "data space".
        self.sample_posterior = self._construct_sample_posterior()
        self.mean_posterior = theano.function([self.Xd, self.Xc, self.Xm], \
                outputs=self.output_mu)
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

    def _construct_post_samples(self):
        """
        Draw a single sample from each of the approximate posteriors encoded
        in self.output_mu and self.output_sigma.
        """
        post_samples = self.output_mu + (self.output_sigma * \
                self.rng.normal(size=self.output_sigma.shape, avg=0.0, std=1.0, \
                dtype=theano.config.floatX))
        return post_samples

    def _construct_kld_cost(self):
        """
        Compute (analytically) the KL divergence between each approximate
        posterior encoded by self.mu/self.sigma and the isotropic Gaussian
        distribution with mean 0 and standard deviation self.prior_sigma.
        """
        prior_sigma_sq = self.prior_sigma**2.0
        prior_log_sigma_sq = np.log(prior_sigma_sq)
        kld_cost = 0.5 * T.sum(((self.output_mu**2.0 / prior_sigma_sq) + \
                (T.exp(self.output_logvar) / prior_sigma_sq) - \
                (self.output_logvar - prior_log_sigma_sq) - 1.0), axis=1, keepdims=True)
        return kld_cost

    def _construct_sample_posterior(self):
        """
        Construct a sampler that draws a single sample from the inferred
        posterior for some set of inputs.
        """
        psample = theano.function([self.Xd, self.Xc, self.Xm], \
                outputs=self.output)
        return psample

    def init_biases(self, b_init=0.0):
        """
        Initialize the biases in all hidden layers to some constant.
        """
        for layer in self.shared_layers:
            b_vec = (0.0 * layer.b.get_value(borrow=False)) + b_init
            layer.b.set_value(b_vec)
        for layer in self.mu_layers[:-1]:
            b_vec = (0.0 * layer.b.get_value(borrow=False)) + b_init
            layer.b.set_value(b_vec)
        for layer in self.sigma_layers[:-1]:
            b_vec = (0.0 * layer.b.get_value(borrow=False)) + b_init
            layer.b.set_value(b_vec)
        return

    def shared_param_clone(self, rng=None, Xd=None, Xc=None, Xm=None):
        """
        Return a clone of this network, with shared parameters but with
        different symbolic input variables.

        This can be used for "unrolling" a generate->infer->generate->infer...
        loop. Then, we can do backprop through time for various objectives.
        """
        clone_net = InfNet(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, \
                prior_sigma=self.prior_sigma, params=self.params, \
                shared_param_dicts=self.shared_param_dicts)
        return clone_net

    def save_to_file(self, f_name=None):
        """
        Dump important stuff to a Python pickle, so that we can reload this
        model later. We'll pickle everything required to create a clone of
        this model given the pickle and the rng/Xd params to the cloning
        function: "InfNet.shared_param_clone()".
        """
        assert(not (f_name is None))
        f_handle = file(f_name, 'wb')
        # dump the "simple" python value in self.prior_sigma
        cPickle.dump(self.prior_sigma, f_handle, protocol=-1)
        # dump the dict self.params, which just holds "simple" python values
        cPickle.dump(self.params, f_handle, protocol=-1)
        # make a copy of self.shared_param_dicts, with numpy arrays in place
        # of the theano shared variables
        numpy_param_dicts = {'shared': [], 'mu': [], 'sigma': []}
        for layer_group in ['shared', 'mu', 'sigma']:
            for shared_dict in self.shared_param_dicts[layer_group]:
                numpy_dict = {}
                for key in shared_dict:
                    numpy_dict[key] = shared_dict[key].get_value(borrow=False)
                numpy_param_dicts[layer_group].append(numpy_dict)
        # dump the numpy version of self.shared_param_dicts
        cPickle.dump(numpy_param_dicts, f_handle, protocol=-1)
        f_handle.close()
        return

def load_infnet_from_file(f_name=None, rng=None, Xd=None, Xc=None, Xm=None):
    """
    Load a clone of some previously trained model.
    """
    assert(not (f_name is None))
    pickle_file = open(f_name)
    self_dot_prior_sigma = cPickle.load(pickle_file)
    self_dot_params = cPickle.load(pickle_file)
    self_dot_numpy_param_dicts = cPickle.load(pickle_file)
    self_dot_shared_param_dicts = {'shared': [], 'mu': [], 'sigma': []}
    for layer_group in ['shared', 'mu', 'sigma']:
        for numpy_dict in self_dot_numpy_param_dicts[layer_group]:
            shared_dict = {}
            for key in numpy_dict:
                val = numpy_dict[key].astype(theano.config.floatX)
                shared_dict[key] = theano.shared(val)
            self_dot_shared_param_dicts[layer_group].append(shared_dict)
    # now, create a PeaNet with the configuration we just unpickled
    clone_net = InfNet(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, \
            prior_sigma=self_dot_prior_sigma, params=self_dot_params, \
            shared_param_dicts=self_dot_shared_param_dicts)
    return clone_net

if __name__=="__main__":
    # TEST CODE FOR MODEL SAVING AND LOADING
    from load_data import load_udm, load_udm_ss, load_mnist
    from NetLayers import relu_actfun, softplus_actfun, \
                          safe_softmax, safe_log
    
    # Simple test code, to check that everything is basically functional.
    print("TESTING...")

    # Initialize a source of randomness
    rng = np.random.RandomState(1234)

    # Load some data to train/validate/test with
    dataset = 'data/mnist.pkl.gz'
    datasets = load_udm(dataset, zero_mean=False)
    Xtr = datasets[0][0]
    Xtr = Xtr.get_value(borrow=False)
    Xva = datasets[1][0]
    Xva = Xva.get_value(borrow=False)
    print("Xtr.shape: {0:s}, Xva.shape: {1:s}".format(str(Xtr.shape),str(Xva.shape)))

    # get and set some basic dataset information
    tr_samples = Xtr.shape[0]
    data_dim = Xtr.shape[1]
    batch_size = 128
    prior_dim = 50
    prior_sigma = 1.0
    Xtr_mean = np.mean(Xtr, axis=0, keepdims=True)
    Xtr_mean = (0.0 * Xtr_mean) + np.mean(Xtr)
    Xc_mean = np.repeat(Xtr_mean, batch_size, axis=0).astype(theano.config.floatX)

    # Symbolic inputs
    Xd = T.matrix(name='Xd')
    Xc = T.matrix(name='Xc')
    Xm = T.matrix(name='Xm')
    Xt = T.matrix(name='Xt')
    Xp = T.matrix(name='Xp')

    ############################
    # Setup inferencer network #
    ############################
    # choose some parameters for the continuous inferencer
    in_params = {}
    shared_config = [data_dim, (200, 4), (200, 4)]
    top_config = [shared_config[-1], prior_dim]
    in_params['shared_config'] = shared_config
    in_params['mu_config'] = top_config
    in_params['sigma_config'] = top_config
    in_params['activation'] = softplus_actfun
    in_params['init_scale'] = 2.0
    in_params['lam_l2a'] = 1e-2
    in_params['vis_drop'] = 0.0
    in_params['hid_drop'] = 0.0
    in_params['bias_noise'] = 0.1
    in_params['input_noise'] = 0.0
    IN = InfNet(rng=rng, Xd=Xd, Xc=Xc, Xm=Xm, \
            prior_sigma=prior_sigma, params=in_params)
    IN.init_biases(0.0)

    pkl_file_name = "TEST_PKL_FILE.pkl"
    print("Saving model:")
    IN.save_to_file(f_name=pkl_file_name)
    print("Loading model:")
    IN_clone = load_infnet_from_file(f_name=pkl_file_name, rng=rng, \
            Xd=Xd, Xc=Xc, Xm=Xm)
    print("DONE!")
