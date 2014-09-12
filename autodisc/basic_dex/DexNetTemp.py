###################################################################
# Semi-supervised EA-Regularized multilayer perceptron ensembles. #
###################################################################

import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse
import theano.tensor.shared_randomstreams

from output_losses import LogRegSS, MCL2HingeSS

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

################################################################################
# HIDDEN LAYER IMPLEMENTATIONS: We've implemented a standard feedforward layer #
# with non-linear activation transform and a max-pooling (a.k.a. Maxout) layer #
# which is currently fixed to operate over disjoint pools of linear filters.   #
#                                                                              #
# TODO: Port the maxout layer implementation to the EAR format.                #
################################################################################

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, \
                 activation=None, pool_size=4, \
                 drop_rate=0., input_noise=0., bias_noise=0., \
                 W=None, b=None, \
                 use_bias=True, name=""):

        # Setup a shared random generator for this layer
        self.srng = theano.tensor.shared_randomstreams.RandomStreams( \
                rng.randint(100000))

        self.clean_input = input

        # Add gaussian noise to the input (if desired)
        if (input_noise > 1e-4):
            self.fuzzy_input = input + \
                    (input_noise * self.srng.normal(size=input.shape, \
                    dtype=theano.config.floatX))
        else:
            self.fuzzy_input = input

        # Apply masking noise to the input (if desired)
        if (drop_rate > 1e-4):
            self.noisy_input = self._drop_from_input(self.fuzzy_input, drop_rate)
        else:
            self.noisy_input = self.fuzzy_input

        # Set some basic layer properties
        self.activation = activation
        self.in_dim = n_in
        self.out_dim = n_out

        # Get some random initial weights and biases, if not given
        if W is None:
            W_init = np.asarray(0.01 * rng.standard_normal( \
                      size=(n_in, n_out)), dtype=theano.config.floatX)
            W = theano.shared(value=W_init, name="{0:s}_W".format(name))
        if b is None:
            b_init = np.zeros((n_out,), dtype=theano.config.floatX)
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
        self.noisy_linear = self.linear_output  + \
                (bias_noise * self.srng.normal(size=self.linear_output.shape, \
                dtype=theano.config.floatX))

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
        drop_mask = self.srng.binomial(n=1, p=1-p, size=input.shape, \
                dtype=theano.config.floatX)
        # get a scaling factor to keep expectations fixed after droppage
        drop_scale = 1. / (1. - p)
        # apply dropout mask and rescaling factor to the input
        droppy_input = drop_scale * input * drop_mask
        return droppy_input

    def _noisy_W(self, noise_lvl=0.):
        """Noisy weights, like convolving energy surface with a gaussian."""
        W_nz = self.W + \
                self.srng.normal(size=self.W.shape, avg=0., std=noise_lvl)
        return W_nz

class JoinLayer(object):
    """Simple layer that averages over "linear_output"s of other layers.

    Note: The list of layers to average over is the only parameter used.
    """
    def __init__(self, input_layers):
        print("making join layer over {0:d} output layers...".format( \
                len(input_layers)))
        il_los = [il.linear_output for il in input_layers]
        self.linear_output = T.mean(T.stack(*il_los), axis=0)
        return


##########################
# NETWORK IMPLEMENTATION #
##########################

class DEX_NET(object):
    """A multipurpose ensemble of noise-perturbed neural networks.

    Parameters:
        rng: a numpy.random RandomState object
        input: Theano symbolic matrix representing inputs to this ensemble
        params: a dict of parameters describing the desired ensemble:
            use_bias: whether to uses biases in hidden and output layers
            lam_l2a: L2 regularization weight on neuron activations
            vis_drop: drop rate to use on input layers (when desired)
            hid_drop: drop rate to use on hidden layers (when desired)
                -- note: vis_drop/hid_drop are optional, with defaults 0.2/0.5
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
            spawn_weights: the weight to multiply the classification loss of
                           each spawned-network by when computing the loss to
                           optimize for this generalized spawn-semble.
    """
    def __init__(self,
            rng,
            input,
            params):
        # First, setup a shared random number generator for this layer
        self.srng = theano.tensor.shared_randomstreams.RandomStreams( \
            rng.randint(100000))
        # Set the activation function
        self.act_fun = lambda x: relu_actfun(x)
        ################################################
        # Process user-suplied parameters for this net #
        ################################################
        lam_l2a = params['lam_l2a']
        use_bias = params['use_bias']
        if 'vis_drop' in params:
            self.vis_drop = params['vis_drop']
        else:
            self.vis_drop = 0.2
        if 'hid_drop' in params:
            self.hid_drop = params['hid_drop']
        else:
            self.hid_drop = 0.5
        self.proto_configs = params['proto_configs']
        self.spawn_configs = params['spawn_configs']
        self.reg_all_obs = 1
        self.spawn_weights = theano.shared(\
                value=np.asarray(params['spawn_weights'], \
                dtype=theano.config.floatX), name='spawn_weights')
        # Compute some "structural" properties of this ensemble
        self.max_proto_depth = max([(len(pc)-1) for pc in self.proto_configs])
        self.spawn_count = len(self.spawn_configs)
        ########################################
        # Initialize all of the proto-networks #
        ########################################
        self.clip_params = {}
        self.proto_nets = []
        self.input = input
        # Construct the proto-networks from which to generate spawn-sembles
        for (pn_num, proto_config) in enumerate(self.proto_configs):
            layer_sizes = [ls for ls in proto_config]
            layer_connect_dims = zip(layer_sizes, layer_sizes[1:])
            layer_num = 0
            proto_net = []
            next_input = self.input
            for n_in, n_out in layer_connect_dims:
                last_layer = (layer_num == (len(layer_connect_dims) - 1))
                pnl_name = "pn{0:d}l{1:d}".format(pn_num, layer_num)
                activation = (lambda x: noop_actfun(x)) if last_layer \
                        else self.act_fun
                # Add a new layer to the regular model
                proto_net.append(HiddenLayer(rng=rng, \
                        input=next_input, \
                        activation=activation, \
                        pool_size=4, \
                        drop_rate=0., input_noise=0., bias_noise=0., \
                        n_in=n_in, n_out=n_out, use_bias=use_bias, \
                        name=pnl_name))
                next_input = proto_net[-1].output
                # Set the non-bias parameters of this layer to be clipped
                self.clip_params[proto_net[-1].W] = 1
                layer_num = layer_num + 1
            # Add this network to the list of proto-networks
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
            next_input = self.input
            proto_net = self.proto_nets[proto_key]
            for proto_layer in proto_net:
                last_layer = (layer_num == (len(proto_net) - 1))
                d_prob = self.vis_drop if (layer_num == 0) else self.hid_drop
                drop_prob = d_prob if do_dropout else 0.0
                # Get important properties from the relevant proto-layer
                activation = proto_layer.activation
                in_dim = proto_layer.in_dim
                out_dim = proto_layer.out_dim
                # Add a new layer to the regular model
                spawn_net.append(HiddenLayer(rng=rng, \
                        input=next_input, \
                        activation=activation, \
                        pool_size=4, drop_rate=drop_prob, \
                        input_noise=input_noise, bias_noise=bias_noise, \
                        W=proto_layer.W, b=proto_layer.b, \
                        n_in=in_dim, n_out=out_dim, use_bias=use_bias))
                next_input = spawn_net[-1].output
                layer_num = layer_num + 1
            # Add this network to the list of spawn-networks
            self.spawn_nets.append(spawn_net)

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

        # DERP DORP
        self._construct_dex_layers(rng)

        # Get metrics for tracking performance over the mean of the outputs
        # of the proto-nets underlying this ensemble.
        self.proto_class_loss, self.proto_class_errors = self._proto_metrics()
        # Get loss functions to optimize based on the spawned-nets in this
        # generalized spawn-semble.
        self.spawn_class_cost = lambda y: self._spawn_class_cost(y)
        self.spawn_dex_cost = lambda idx: self._spawn_dex_cost(idx)
        self.act_reg_cost = lam_l2a * self._act_reg_cost()
        return

    def _proto_metrics(self):
        """Compute classification loss and error rate over proto-nets."""
        # Create a layer that joins the linear outputs of the proto-networks
        if len(self.proto_nets) > 1:
            proto_out_layers = [pn[-1] for pn in self.proto_nets]
        else:
            proto_out_layers = [self.proto_nets[0][-1], self.proto_nets[0][-1]]
        proto_out_func = MCL2HingeSS(JoinLayer(proto_out_layers))
        return [proto_out_func.loss_func, proto_out_func.errors]

    def _act_reg_cost(self):
        """Apply L2 regularization to the activations in each spawn-net."""
        act_sq_sums = []
        for i in range(self.spawn_count):
            sn = self.spawn_nets[i]
            for snl in sn:
                act_sq_sums.append(snl.act_l2_sum)
        full_act_sq_sum = T.sum(act_sq_sums) / self.spawn_count
        return full_act_sq_sum

    def _spawn_class_cost(self, y):
        """Compute the weighted sum of class losses over the spawn-nets.

        Classification cost/loss is computed for each spawn-net in this
        generalized pseudo-ensemble."""
        spawn_class_losses = []
        for i in range(self.spawn_count):
            spawn_net = self.spawn_nets[i]
            spawn_out_func = MCL2HingeSS(spawn_net[-1])
            spawn_class_loss = \
                    self.spawn_weights[i] * spawn_out_func.loss_func(y)
            spawn_class_losses.append(spawn_class_loss)
        total_loss = T.sum(spawn_class_losses)
        return total_loss

    def _spawn_dex_cost(self, idx):
        """DERP DORP."""
        dex_costs = []
        for dex_layer in self.dex_layers:
            dex_costs.append(dex_layer.cost(idx))

        mean_cost = T.mean(dex_costs)
        return mean_cost

    def _construct_dex_layers(self, rng):
        """DERP DORP."""
        self.dae_params = []
        self.dae_costs = []
        # The number of hidden layers in each proto-network is depth-1, where
        # depth is the total number of layers in the network. This is because
        # we count the output layer as well as the hidden layers.
        for d in range(self.max_proto_depth - 1):
            d_params = []
            s_params = []
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
                    hid_dim = sn[d].out_dim
                    # Construct the DAE layer object
                    dae_layer = DAELayer(rng=rng, \
                            clean_input=ci_sn, \
                            fuzzy_input=fi_sn, \
                            n_in=vis_dim, n_out=hid_dim, \
                            activation=self.act_fun, \
                            input_noise=nz_lvl, \
                            W=W_sn, b_h=b_sn, b_v=None)
                    d_params.extend(dae_layer.params)
                    s_params.extend([W_sn, b_sn])
                    d_costs.append(dae_layer.compute_costs(lam_l1))
            # Record the set of all DAE params to-be-optimized at depth d
            self.dae_params.append(d_params)
            # Record the sum of reconstruction costs for DAEs at depth d (in
            # self.dae_costs[d][0]) and the sum of sparse regularization costs
            # for DAEs at depth d (in self.dae_costs[d][1]).
            self.dae_costs.append([T.sum([c[0] for c in d_costs]), \
                    T.sum([c[1] for c in d_costs])])
        return

#################################
# EXEMPLAR LAYER IMPLEMENTATION #
#################################

class XMPLayer(object):
    def __init__(self, rng, X=None, I=None, \
            vec_dim=0, max_key=0, W=None, b=None):

        # Setup a shared random generator for this layer
        self.srng = theano.tensor.shared_randomstreams.RandomStreams( \
                rng.randint(100000))

        self.X = X

        # Set some basic layer properties
        self.vec_dim = vec_dim
        self.max_key = max_key

        # Get some random initial weights and biases, if not given
        if W is None:
            W_init = np.asarray(0.01 * rng.standard_normal( \
                      size=(max_key+10, vec_dim)), dtype=theano.config.floatX)
            W = theano.shared(value=W_init, name='W')
        if b is None:
            b_init = np.zeros((max_key+10, 1), dtype=theano.config.floatX)
            b = theano.shared(value=b_init, name='b')

        # Grab pointers to the now-initialized weights and biases
        self.W = W
        self.b = b
        self.W_I = T.take(self.W, I, axis=0)
        self.b_I = T.take(self.b, I, axis=0)

        # Put the learnable/optimizable parameters into a list
        self.params = [self.W, self.b]
        # Beep boop... layer construction complete...
        return









##############
# EYE BUFFER #
##############
