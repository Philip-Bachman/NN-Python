###################################################################
# Semi-supervised EA-Regularized multilayer perceptron ensembles. #
###################################################################

import numpy as np
import numpy.random as npr
import theano
import theano.tensor as T
from theano.ifelse import ifelse
#from theano.tensor.shared_randomstreams import RandomStreams as RandStream
from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams as RandStream

from NetLayers import HiddenLayer, JoinLayer, DAELayer

#####################################################################
# NON-LINEARITIES: Some activation functions, for your convenience. #
#####################################################################

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

def smooth_js_divergence(p, q):
    """
    Measure the Jensen-Shannon divergence between (log-space) p and q.
    """
    p_sm = smooth_softmax(p)
    q_sm = smooth_softmax(q)
    mean_dist = (p_sm + q_sm) / 2.0
    js_1 = T.sum(p_sm * (T.log(p_sm) - T.log(mean_dist)), axis=1, keepdims=True)
    js_2 = T.sum(q_sm * (T.log(q_sm) - T.log(mean_dist)), axis=1, keepdims=True)
    js_div = (js_1 + js_2) / 2.0
    return js_div

def smooth_cross_entropy(p, q):
    """Measure the cross-entropy between "approximate" distribution q and
    "true" distribution p. Use smoothed softmax to convert p and q from
    encodings in terms of relative log-likelihoods into sum-to-one dists."""
    p_sm = smooth_softmax(p)
    q_sm = smooth_softmax(q)
    # This term is: entropy(p) + kl_divergence(p, q)
    ce_sm = -T.sum((p_sm * T.log(q_sm)), axis=1, keepdims=True)
    return ce_sm


##########################
# NETWORK IMPLEMENTATION #
##########################

class PeaNet(object):
    """
    A multi-purpose ensemble of noise-perturbed neural networks. This class
    constructs and manages the computation graph for a pseudo-ensemble, and
    provides costs for imposing pseudo-ensemble agreement regularization on
    the pseudo-ensemble. (i.e. droppy fuzzy networks)


    Parameters:
        rng: a numpy.random RandomState object
        Xd: Theano symbolic matrix for "observation" inputs to this PeaNet
        params: a dict of parameters describing the desired ensemble:
            lam_l2a: L2 regularization weight on neuron activations
            vis_drop: drop rate to use on input layers (when desired)
            hid_drop: drop rate to use on hidden layers (when desired)
                -- note: vis_drop/hid_drop are optional, with defaults 0.2/0.5
            ear_type: type of Ensemble Agreement Regularization (EAR) to use
                -- note: defns of _ear_cost() and _ear_loss() give more info
            activation: non-linearity to apply in hidden layers
            proto_configs: list of lists, where each sublist gives the number
                           of neurons to put in each hidden layer one of the
                           proto-networks underlying this ensemble. Sub-lists
                           need not be the same length, but their first values
                           should all match, as should their last values. This
                           is because the proto-nets all take the same input
                           and output predictions over the same classes.
            spawn_configs: list of dicts, where each dict describes the basic
                           values needed for spawning a noise-perturbed net
                           from some proto-net. The dict should contain keys:
                           proto_key: which proto-net to spawn from
                           input_noise: amount of noise on layer inputs
                           bias_noise: amount of noise on layer biases
                           do_dropout: whether to apply dropout
        shared_param_dicts: parameters for the MLP controlled by this PeaNet
    """
    def __init__(self,
            rng=None, \
            Xd=None, \
            params=None, \
            shared_param_dicts=None):
        # First, setup a shared random number generator for this layer
        self.rng = RandStream(rng.randint(100000))
        ################################################
        # Process user-suplied parameters for this net #
        ################################################
        assert(not (params is None))
        assert(len(params['proto_configs']) == 1) # permit only one proto-net
        assert(len(params['spawn_configs']) <= 2) # use one or two spawn nets
        self.Xd = Xd # symbolic input to this computation graph
        self.params = params
        lam_l2a = params['lam_l2a']
        if 'vis_drop' in params:
            self.vis_drop = params['vis_drop']
        else:
            self.vis_drop = 0.2
        if 'hid_drop' in params:
            self.hid_drop = params['hid_drop']
        else:
            self.hid_drop = 0.5
        if 'activation' in params:
            self.activation = params['activation']
        else:
            self.activation = relu_actfun
        self.proto_configs = params['proto_configs']
        self.spawn_configs = params['spawn_configs']
        self.ear_type = params['ear_type']
        # Compute some "structural" properties of this ensemble
        self.max_proto_depth = max([(len(pc)-1) for pc in self.proto_configs])
        self.spawn_count = len(self.spawn_configs)
        # Check if the params for this net were given a priori. This option
        # will be used for creating "clones" of a generative network, with all
        # of the network parameters shared between clones.
        if shared_param_dicts is None:
            # This is not a clone, and we will need to make a dict for
            # referring to the parameters of each network layer
            self.shared_param_dicts = []
            self.is_clone = False
        else:
            # This is a clone, and its layer parameters can be found by
            # referring to the given param dict (i.e. shared_param_dicts).
            self.shared_param_dicts = shared_param_dicts
            self.is_clone = True
        ########################################
        # Initialize all of the proto-networks #
        ########################################
        self.clip_params = {}
        self.proto_nets = []
        # Construct the proto-networks from which to generate spawn-sembles
        for (pn_num, proto_config) in enumerate(self.proto_configs):
            layer_defs = [ld for ld in proto_config]
            layer_connect_defs = zip(layer_defs[:-1], layer_defs[1:])
            layer_num = 0
            proto_net = []
            next_input = self.Xd
            for in_def, out_def in layer_connect_defs:
                last_layer = (layer_num == (len(layer_connect_defs) - 1))
                pnl_name = "pn{0:d}l{1:d}".format(pn_num, layer_num)
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
                # Add a new layer to the regular model
                if not self.is_clone:
                    ##########################################
                    # Initialize a layer with new parameters #
                    ##########################################
                    new_layer = HiddenLayer(rng=rng, input=next_input, \
                            activation=None, pool_size=pool_size, \
                            drop_rate=0., input_noise=0., bias_noise=0., \
                            in_dim=in_dim, out_dim=out_dim, \
                            name=pnl_name, W_scale=2.0)
                    proto_net.append(new_layer)
                    self.shared_param_dicts.append( \
                            {'W': new_layer.W, 'b': new_layer.b})
                else:
                    ##################################################
                    # Initialize a layer with some shared parameters #
                    ##################################################
                    init_params = self.shared_param_dicts[layer_num]
                    new_layer = HiddenLayer(rng=rng, input=next_input, \
                            activation=None, pool_size=pool_size, \
                            drop_rate=0., input_noise=0., bias_noise=0., \
                            in_dim=in_dim, out_dim=out_dim, \
                            W=init_params['W'], b=init_params['b'], \
                            name=pnl_name, W_scale=2.0)
                    proto_net.append(new_layer)
                next_input = proto_net[-1].output
                # Set the non-bias parameters of this layer to be clipped
                self.clip_params[proto_net[-1].W] = 1
                layer_num = layer_num + 1
            # Add this network to the list of proto-networks, and add its
            # param dict to the list of pro-net param dicts, if not a clone
            self.proto_nets.append(proto_net)
        #################################################################
        # Initialize all of the spawned (i.e. noise-perturbed) networks #
        #################################################################
        self.spawn_nets = []
        self.proto_keys = []
        for spawn_config in self.spawn_configs:
            proto_key = spawn_config['proto_key']
            self.proto_keys.append(proto_key)
            print("spawned from proto-net: {0:d} (of {1:d})".format(proto_key, \
                    len(self.proto_nets)))
            input_noise = spawn_config['input_noise']
            bias_noise = spawn_config['bias_noise']
            do_dropout = spawn_config['do_dropout']
            assert((proto_key >= 0) and (proto_key < len(self.proto_nets)))
            # Get info about the proto-network to spawn from
            layer_num = 0
            spawn_net = []
            next_input = self.Xd
            proto_net = self.proto_nets[proto_key]
            for proto_layer in proto_net:
                last_layer = (layer_num == (len(proto_net) - 1))
                layer_in = input_noise if (layer_num == 0) else 0.0
                d_prob = self.vis_drop if (layer_num == 0) else self.hid_drop
                drop_prob = d_prob if do_dropout else 0.0
                # Get important properties from the relevant proto-layer
                actfun = proto_layer.activation
                pool_size = proto_layer.pool_size
                in_dim = proto_layer.in_dim
                out_dim = proto_layer.out_dim
                # Add a new layer to the regular model
                spawn_net.append(HiddenLayer(rng=rng, \
                        input=next_input, activation=actfun, \
                        pool_size=pool_size, drop_rate=drop_prob, \
                        input_noise=layer_in, bias_noise=bias_noise, \
                        W=proto_layer.W, b=proto_layer.b, \
                        in_dim=in_dim, out_dim=out_dim))
                next_input = spawn_net[-1].output
                layer_num = layer_num + 1
            # Add this network to the list of spawn-networks
            self.spawn_nets.append(spawn_net)

        # TODO: implement adjustable norm clipping
        self.clip_norms = {}

        # Mash all the parameters together, into a list. Also make a list
        # comprising only parameters located in final/classification layers
        # of the proto-networks (for use in fine-tuning, probably).
        self.proto_params = []
        self.class_params = []
        for pn in self.proto_nets:
            for (i, pl) in enumerate(pn):
                self.proto_params.extend(pl.params)
                if (i == (len(pn) - 1)):
                    self.class_params.extend(pl.params)

        # Build loss functions for denoising autoencoder training. This sets up
        # a cost function for each possible layer, as determined by the maximum
        # number of layers in any proto-network. The DAE cost for layer i will
        # be the mean DAE cost over all i'th layers in the proto-networks.
        self.dae_lam_l1 = theano.shared( \
            value=np.asarray([0.2]).astype(theano.config.floatX))
        self._construct_dae_layers(rng, lam_l1=self.dae_lam_l1, nz_lvl=0.25)

        # create symbolic "hooks" for observing the output of this network,
        # either without perturbations or subject to perturbations
        self.output_proto = self.proto_nets[0][-1].linear_output
        self.output_spawn = [sn[-1].linear_output for sn in self.spawn_nets]

        # get a cost function for encouraging "pseudo-ensemble agreement"
        self.pea_reg_cost = self._ear_cost(self.ear_type)
        # get a cost function for penalizing/rewarding prediction entropy
        self.ent_reg_cost = self._ent_cost()
        self.act_reg_cost = lam_l2a * self._act_reg_cost()
        # construct a function for sampling from a categorical
        self.sample_posterior = self._construct_sample_posterior()
        return

    def _act_reg_cost(self):
        """
        Apply L2 regularization to the activations in each spawn-net.
        """
        act_sq_sums = []
        for i in range(self.spawn_count):
            sn = self.spawn_nets[i]
            for snl in sn:
                act_sq_sums.append(snl.act_l2_sum)
        full_act_sq_sum = T.sum(act_sq_sums) / self.spawn_count
        return full_act_sq_sum

    def _ear_cost(self, ear_type):
        """
        Compute the cost of pseudo-ensemble agreement regularization.
        """
        if self.spawn_count == 1:
            x1 = self.spawn_nets[0][-1].linear_output
            ear_loss = 0.0 * smooth_js_divergence(x1, x1)
        else:
            x1 = self.spawn_nets[0][-1].linear_output
            x2 = self.spawn_nets[1][-1].linear_output
            ear_loss = smooth_js_divergence(x1, x2)
            #ear_loss = (smooth_kl_divergence(x1, x2) + \
            #        smooth_kl_divergence(x2, x1)) / 2.0
        return ear_loss

    def _ent_cost(self, ent_type=1):
        """
        Compute cost for entropy regularization.
        """
        if ent_type == 0:
            # binary cross-entropy
            ent_fun = lambda x: T.sum(T.nnet.binary_crossentropy( \
                    T.nnet.sigmoid(x), T.nnet.sigmoid(x)), axis=1, keepdims=True)
        else:
            # multinomial cross-entropy
            ent_fun = lambda x: smooth_cross_entropy(x, x)
        if self.spawn_count == 1:
            x = self.spawn_nets[0][-1].linear_output
            ent_loss = ent_fun(x)
        else:
            x1 = self.spawn_nets[0][-1].linear_output
            x2 = self.spawn_nets[1][-1].linear_output
            ent_loss = (ent_fun(x1) + ent_fun(x2)) / 2.0
        return ent_loss

    def _construct_dae_layers(self, rng, lam_l1=None, nz_lvl=0.25):
        """
        Build cost functions for training DAEs defined for all hidden layers
        of the proto-networks making up this generalized ensemble. That is,
        construct a DAE for every proto-layer that isn't a classification
        layer. Inputs to each DAE are taken from the clean and post-fuzzed
        inputs of the spawn-net layer for the 'first' spawn-net spawned from
        any given proto-net.
        """
        self.dae_params = []
        self.dae_costs = []
        ACT_FUN = lambda x: relu_actfun(x)
        # The number of hidden layers in each proto-network is depth-1, where
        # depth is the total number of layers in the network. This is because
        # we count the output layer as well as the hidden layers.
        for d in range(self.max_proto_depth - 1):
            d_params = []
            d_costs = []
            for pn_key in range(len(self.proto_nets)):
                # Get the "first" spawn-net spawned from this proto-net
                sn_key = self.proto_keys.index(pn_key)
                sn = self.spawn_nets[sn_key]
                if (d < (len(sn) - 1)):
                    # Construct a DAE for this proto/spawn-net hidden layer
                    W_sn = sn[d].W
                    b_sn = sn[d].b
                    ci_sn = sn[d].clean_input # the input to be reconstructed
                    fi_sn = sn[d].fuzzy_input # the input to reconstruct from
                    vis_dim = sn[d].in_dim
                    hid_dim = sn[d].filt_count
                    # Construct the DAE layer object
                    dae_layer = DAELayer(rng=rng, \
                            clean_input=ci_sn, \
                            fuzzy_input=fi_sn, \
                            in_dim=vis_dim, out_dim=hid_dim, \
                            activation=ACT_FUN, \
                            input_noise=nz_lvl, \
                            W=W_sn, b_h=b_sn, b_v=None)
                    d_params.extend(dae_layer.params)
                    d_costs.append(dae_layer.compute_costs(lam_l1))
            # Record the set of all DAE params to-be-optimized at depth d
            self.dae_params.append(d_params)
            # Record the sum of reconstruction costs for DAEs at depth d (in
            # self.dae_costs[d][0]) and the sum of sparse regularization costs
            # for DAEs at depth d (in self.dae_costs[d][1]).
            self.dae_costs.append([T.sum([c[0] for c in d_costs]), \
                    T.sum([c[1] for c in d_costs])])
        return

    def _construct_sample_posterior(self):
        """
        Construct a function for sampling from the categorical distribution
        resulting from taking a softmax of the output of this PeaNet.
        """
        func = theano.function([self.Xd], \
                outputs=safe_softmax(self.output_proto))
        # this function is based on "roulette wheel" sampling
        def sampler(x):
            y_probs = func(x)
            y_cumsum = np.cumsum(y_probs, axis=1)
            rand_vals = npr.rand(y_probs.shape[0],1)
            y_bin = np.zeros(y_probs.shape)
            for row in range(y_bin.shape[0]):
                for col in range(y_bin.shape[1]):
                    if y_cumsum[row,col] > rand_vals[row]:
                        y_bin[row,col] = 1.0
            y_bin = y_bin.astype(theano.config.floatX)
            return y_bin
        return sampler

    def init_biases(self, b_init=0.0):
        """
        Initialize the biases in all hidden layers to some constant.
        """
        for layer in self.proto_nets[0][:-1]:
            b_init = (0.0 * layer.b.get_value(borrow=False)) + b_init
            layer.b.set_value(b_init)
        return

    def shared_param_clone(self, rng=None, Xd=None):
        """
        Return a clone of this network, with shared parameters but with
        different symbolic input variables.
        """
        clone_net = PeaNet(rng=rng, Xd=Xd, params=self.params, \
                shared_param_dicts=self.shared_param_dicts)
        return clone_net











##############
# EYE BUFFER #
##############
