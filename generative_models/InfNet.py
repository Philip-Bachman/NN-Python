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
from LogPDFs import gaussian_kld

####################################
# INFREENCE NETWORK IMPLEMENTATION #
####################################

def row_normalize(x):
    """
    Normalize rows of matrix x to unit (L2) norm.
    """
    x_normed = x / T.sqrt(T.sum(x**2.,axis=1,keepdims=1)+1e-8)
    return x_normed

def soft_abs(x, smoothing=1e-5):
    """
    Soft absolute value function applied to x.
    """
    sa_x = T.sqrt(x**2. + smoothing)
    return sa_x


class InfNet(object):
    """
    A net that tries to infer an approximate posterior for some observation,
    given some deep, directed generative model. The output of this network
    comprises two constructs: an approximate mean vector and an approximate
    standard deviation vector (i.e. diagonal matrix) for a Gaussian posterior.

    Parameters:
        rng: a numpy.random RandomState object
        Xd: symbolic input matrix for inputs
        prior_sigma: standard deviation of isotropic Gaussian prior that our
                     inferred posteriors will be penalized for deviating from.
        params: a dict of parameters describing the desired network:
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
            encoder: a function that will be applied to inputs prior to
                     passing them through the network. this can be used for
                     in-lining, e.g., PCA preprocessing on training data
        shared_param_dicts: parameters for the MLP controlled by this InfNet
    """
    def __init__(self, \
            rng=None, \
            Xd=None, \
            prior_sigma=None, \
            params=None, \
            shared_param_dicts=None):
        # Setup a shared random generator for this network 
        self.rng = RandStream(rng.randint(1000000))
        # Grab the symbolic input matrix
        self.Xd = Xd
        self.prior_sigma = prior_sigma
        #####################################################
        # Process user-supplied parameters for this network #
        #####################################################
        self.params = params
        self.lam_l2a = params['lam_l2a']
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
        if 'encoder' in params:
            self.encoder = params['encoder']
            self.decoder = params['decoder']
            self.use_encoder = True
            self.Xd_encoded = self.encoder(self.Xd)
        else:
            self.encoder = lambda x: x
            self.decoder = lambda x: x
            self.use_encoder = False
            self.Xd_encoded = self.encoder(self.Xd)
        if 'kld2_scale' in params:
            self.kld2_scale = params['kld2_scale']
        else:
            self.kld2_scale = 0.0
        if 'sigma_init_scale' in params:
            self.sigma_init_scale = params['sigma_init_scale']
        else:
            self.sigma_init_scale = 1.0
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
        # Construct input to the inference network
        if self.use_encoder:
            next_input = self.encoder(self.Xd)
        else:
            next_input = self.Xd
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
                        {'W': new_layer.W, 'b': new_layer.b, \
                         'b_in': new_layer.b_in, 's_in': new_layer.s_in})
            else:
                ##################################################
                # Initialize a layer with some shared parameters #
                ##################################################
                init_params = self.shared_param_dicts['shared'][layer_num]
                if not (('b_in' in init_params) and ('s_in' in init_params)):
                    init_params['b_in'] = None
                    init_params['s_in'] = None
                new_layer = HiddenLayer(rng=rng, input=next_input, \
                        activation=self.activation, pool_size=pool_size, \
                        drop_rate=d_rate, input_noise=i_noise, bias_noise=b_noise, \
                        in_dim=in_dim, out_dim=out_dim, \
                        W=init_params['W'], b=init_params['b'], \
                        b_in=init_params['b_in'], s_in=init_params['s_in'], \
                        name=l_name, W_scale=i_scale)
                self.shared_layers.append(new_layer)
                if ((init_params['b_in'] is None) or (init_params['s_in'] is None)):
                    init_params['b_in'] = new_layer.b_in
                    init_params['s_in'] = new_layer.s_in
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
                self.mu_layers.append(new_layer)
                self.shared_param_dicts['mu'].append( \
                        {'W': new_layer.W, 'b': new_layer.b, \
                         'b_in': new_layer.b_in, 's_in': new_layer.s_in})
            else:
                ##################################################
                # Initialize a layer with some shared parameters #
                ##################################################
                init_params = self.shared_param_dicts['mu'][layer_num]
                if not (('b_in' in init_params) and ('s_in' in init_params)):
                    init_params['b_in'] = None
                    init_params['s_in'] = None
                new_layer = HiddenLayer(rng=rng, input=next_input, \
                        activation=self.activation, pool_size=pool_size, \
                        drop_rate=d_rate, input_noise=i_noise, bias_noise=b_noise, \
                        in_dim=in_dim, out_dim=out_dim, \
                        W=init_params['W'], b=init_params['b'], \
                        b_in=init_params['b_in'], s_in=init_params['s_in'], \
                        name=l_name, W_scale=i_scale)
                self.mu_layers.append(new_layer)
                if ((init_params['b_in'] is None) or (init_params['s_in'] is None)):
                    init_params['b_in'] = new_layer.b_in
                    init_params['s_in'] = new_layer.s_in
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
            i_scale = self.init_scale
            if last_layer:
                # set in-bound weights for logvar predictions to 0
                i_scale = 0.0 * i_scale
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
                self.shared_param_dicts['sigma'].append( \
                        {'W': new_layer.W, 'b': new_layer.b, \
                         'b_in': new_layer.b_in, 's_in': new_layer.s_in})
            else:
                ##################################################
                # Initialize a layer with some shared parameters #
                ##################################################
                init_params = self.shared_param_dicts['sigma'][layer_num]
                if not (('b_in' in init_params) and ('s_in' in init_params)):
                    init_params['b_in'] = None
                    init_params['s_in'] = None
                new_layer = HiddenLayer(rng=rng, input=next_input, \
                        activation=self.activation, pool_size=pool_size, \
                        drop_rate=d_rate, input_noise=i_noise, bias_noise=b_noise, \
                        in_dim=in_dim, out_dim=out_dim, \
                        W=init_params['W'], b=init_params['b'], \
                        b_in=init_params['b_in'], s_in=init_params['s_in'], \
                        name=l_name, W_scale=i_scale)
                self.sigma_layers.append(new_layer)
                if ((init_params['b_in'] is None) or (init_params['s_in'] is None)):
                    init_params['b_in'] = new_layer.b_in
                    init_params['s_in'] = new_layer.s_in
            next_input = self.sigma_layers[-1].output
            # Acknowledge layer completion
            layer_num = layer_num + 1

        # Create a shared parameter for rescaling posterior "sigmas" to allow
        # control over the velocity of the markov chain generated by repeated
        # cycling through the INF -> GEN loop.
        if not ('sigma_scale' in self.shared_param_dicts['sigma'][-1]):
            # we use a hack-ish check to remain compatible with loading models
            # that were saved before the addition of the sigma_scale param.
            zero_ary = np.zeros((1,)).astype(theano.config.floatX)
            self.sigma_scale = theano.shared(value=zero_ary)
            new_dict = {'sigma_scale': self.sigma_scale}
            self.shared_param_dicts['sigma'].append(new_dict)
            self.set_sigma_scale(1.0)
        else:
            # this is a clone of some other InfNet, and that InfNet was made
            # after adding the sigma_scale param, so use its sigma_scale
            self.sigma_scale = \
                    self.shared_param_dicts['sigma'][-1]['sigma_scale']

        # Create a shared parameter for maintaining an exponentially decaying
        # estimate of the population mean of posterior KL divergence.
        if not ('kld_mean' in self.shared_param_dicts['sigma'][-1]):
            # add a kld_mean if none was already present
            zero_ary = np.zeros((1,)).astype(theano.config.floatX) + 100.0
            self.kld_mean = theano.shared(value=zero_ary)
            self.shared_param_dicts['sigma'][-1]['kld_mean'] = self.kld_mean
        else:
            # use a kld_mean that's already present
            self.kld_mean = self.shared_param_dicts['sigma'][-1]['kld_mean']

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
        self.output_mean = self.mu_layers[-1].linear_output
        self.output_logvar = self.sigma_layers[-1].linear_output
        self.output_sigma = self.sigma_init_scale * self.sigma_scale[0] * \
                T.exp(0.5 * self.output_logvar)

        # We'll also construct an output containing a single samples from each
        # of the distributions represented by the rows of self.output_mean and
        # self.output_sigma.
        self.output = self._construct_post_samples()
        self.out_dim = self.sigma_layers[-1].out_dim
        # Get simple regularization penalty to moderate activation dynamics
        self.act_reg_cost = self.lam_l2a * self._act_reg_cost()
        # Construct a function for penalizing KL divergence between the
        # approximate posteriors produced by this model and some isotropic
        # Gaussian distribution.
        self.kld_cost = self._construct_kld_cost()
        self.kld_mean_update = T.cast((0.98 * self.kld_mean) + \
                (0.02 * T.mean(self.kld_cost)), 'floatX')
        self.kld_func = self._construct_kld_func() # function computing KLds
        # Construct a theano function for sampling from the approximate
        # posteriors inferred by this model for some collection of points
        # in the "data space".
        if self.build_theano_funcs:
            self.sample_posterior = self._construct_sample_posterior()
            self.mean_posterior = theano.function([self.Xd], \
                    outputs=self.output_mean)
        else:
            self.sample_posterior = None
            self.mean_posterior = None

        ########################################################
        # CONSTRUCT FUNCTIONS FOR RICA PRETRAINING INPUT LAYER #
        ########################################################
        self.rica_func = None
        self.W_rica = self.shared_layers[0].W
        return

    def train_rica(self, X, lr, lam):
        """
        CONSTRUCT FUNCTIONS FOR RICA PRETRAINING INPUT LAYER
        """
        if self.rica_func is None:
            l_rate = T.scalar()
            lam_l1 = T.scalar()
            X_in = T.matrix('in_X_in')
            W_in = self.W_rica + self.rng.normal(size=self.W_rica.shape, \
                avg=0.0, std=0.01, dtype=theano.config.floatX)
            X_enc = self.encoder(X_in)
            H_rec = T.dot(X_enc, W_in)
            X_rec = T.dot(H_rec, W_in.T)
            recon_cost = T.sum((X_enc - X_rec)**2.0) / X_enc.shape[0]
            spars_cost = lam_l1 * (T.sum(soft_abs(H_rec)) / H_rec.shape[0])
            rica_cost = recon_cost + spars_cost
            dW = T.grad(rica_cost, self.W_rica)
            rica_updates = {self.W_rica: self.W_rica - (l_rate * dW)}
            rica_outputs = [rica_cost, recon_cost, spars_cost]
            self.rica_func = theano.function([X_in, l_rate, lam_l1], \
                    outputs=rica_outputs, \
                    updates=rica_updates)
        outputs = self.rica_func(X, lr, lam)
        return outputs

    def set_sigma_scale(self, sigma_scale=1.0):
        """
        Set the posterior sigma rescaling shared parameter to some value.
        """
        zero_ary = np.zeros((1,))
        new_scale = zero_ary + sigma_scale
        self.sigma_scale.set_value(new_scale.astype(theano.config.floatX))
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
        in self.output_mean and self.output_sigma.
        """
        post_samples = self.output_mean + (self.output_sigma * \
                self.rng.normal(size=self.output_sigma.shape, avg=0.0, std=1.0, \
                dtype=theano.config.floatX))
        return post_samples

    def _construct_kld_cost(self):
        """
        Compute (analytically) the KL divergence between each approximate
        posterior encoded by self.mu/self.sigma and the isotropic Gaussian
        distribution with mean 0 and standard deviation self.prior_sigma.
        """
        # construct a penalty that is L2-like near 0 and L1-like away from 0.
        huber_pen = lambda x, d: \
                ((1.0 / (2.0 * d)) * ((T.abs_(x) < d) * (x**2.0))) + \
                ((T.abs_(x) >= d) * (T.abs_(x) - (d / 2.0)))
        prior_mu = 0.0
        prior_logvar = np.log(self.prior_sigma**2.0)
        post_klds = gaussian_kld(self.output_mean, self.output_logvar, \
                prior_mu, prior_logvar)
        kld_cost = T.sum(huber_pen(post_klds, 0.2), axis=1, keepdims=True)
        return kld_cost

    def _construct_sample_posterior(self):
        """
        Construct a sampler that draws a single sample from the inferred
        posterior for some set of inputs.
        """
        psample = theano.function([self.Xd], \
                outputs=self.output)
        return psample

    def _construct_kld_func(self):
        """
        Construct a function for computing posterior KLds for some inputs.
        """
        kd_func = theano.function([self.Xd], \
                outputs=self.kld_cost)
        return kd_func

    def init_biases(self, b_init=0.0, b_std=1e-2):
        """
        Initialize the biases in all hidden layers to some constant.
        """
        for layer in self.shared_layers:
            b_vec = (0.0 * layer.b.get_value(borrow=False)) + b_init
            b_vec = b_vec + (b_std * npr.randn(*b_vec.shape))
            layer.b.set_value(b_vec.astype(theano.config.floatX))
        for layer in self.mu_layers[:-1]:
            b_vec = (0.0 * layer.b.get_value(borrow=False)) + b_init
            b_vec = b_vec + (b_std * npr.randn(*b_vec.shape))
            layer.b.set_value(b_vec.astype(theano.config.floatX))
        for layer in self.sigma_layers[:-1]:
            b_vec = (0.0 * layer.b.get_value(borrow=False)) + b_init
            b_vec = b_vec + (b_std * npr.randn(*b_vec.shape))
            layer.b.set_value(b_vec.astype(theano.config.floatX))
        return

    def shared_param_clone(self, rng=None, Xd=None):
        """
        Return a clone of this network, with shared parameters but with
        different symbolic input variables.

        This can be used for "unrolling" a generate->infer->generate->infer...
        loop. Then, we can do backprop through time for various objectives.
        """
        clone_net = InfNet(rng=rng, Xd=Xd, \
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

def load_infnet_from_file(f_name=None, rng=None, Xd=None, \
                          new_params=None):
    """
    Load a clone of some previously trained model.
    """
    assert(not (f_name is None))
    pickle_file = open(f_name)
    self_dot_prior_sigma = cPickle.load(pickle_file)
    self_dot_params = cPickle.load(pickle_file)
    if not (new_params is None):
        for k in new_params:
            self_dot_params[k] = new_params[k]
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
    clone_net = InfNet(rng=rng, Xd=Xd, \
            prior_sigma=self_dot_prior_sigma, params=self_dot_params, \
            shared_param_dicts=self_dot_shared_param_dicts)
    # helpful output
    print("==================================================")
    print("LOADED InfNet WITH PARAMS:")
    for k in self_dot_params:
        print("    {0:s}: {1:s}".format(str(k), str(self_dot_params[k])))
    print("==================================================")
    return clone_net

if __name__=="__main__":
    # Derp
    print("NO TEST/DEMO CODE FOR NOW.")
