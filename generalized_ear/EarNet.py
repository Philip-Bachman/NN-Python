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
    """Normalize rows of matrix x to unit (L2) length."""
    x_normed = x / T.sqrt(T.sum(x**2.,axis=1,keepdims=1)+1e-6)
    return x_normed

def col_normalize(x):
    """Normalize cols of matrix x to unit (L2) length."""
    x_normed = x / T.sqrt(T.sum(x**2.,axis=0,keepdims=1)+1e-6)
    return x_normed

def rehu_actfun(x):
    """Compute rectified huberized activation for x."""
    M_quad = (x > 0.0) - (x >= 0.5)
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
    p = e_x / T.sum(e_x, axis=1, keepdims=True)
    p_sm = (p + eps) / T.sum((p + eps), axis=1, keepdims=True)
    return p_sm

def smooth_entropy(p):
    """Measure the entropy of distribution p, after converting it from an
    encoding in terms of relative log-likelihoods into an encoding as a
    sum-to-one distribution."""
    p_sm = smooth_softmax(p)
    e_sm = -T.sum((T.log(p_sm) * p_sm), axis=1, keepdims=True)
    return kl_sm

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
                 drop_rate=0., input_noise=0., preact_noise=0., \
                 W=None, b=None, \
                 use_bias=True, name=""):

        # Setup a shared random generator for this layer
        self.srng = theano.tensor.shared_randomstreams.RandomStreams( \
                rng.randint(100000))

        # Add gaussian noise to the input (if desired)
        if (input_noise > 1e-4):
            fuzzy_input = input + (input_noise * self.srng.normal(size=input.shape, \
                    dtype=theano.config.floatX))
        else:
            fuzzy_input = input

        # Apply masking noise to the input (if desired)
        if (drop_rate > 1e-4):
            self.input = self._drop_from_input(fuzzy_input, drop_rate)
        else:
            self.input = fuzzy_input

        # Apply noise in reverse order (i.e. masking -> fuzzing)
        COMMENT = """
        # Apply masking noise to the input (if desired)
        if (drop_rate > 1e-4):
            droppy_input = self._drop_from_input(input, drop_rate)
        else:
            droppy_input = input

        # Add gaussian noise to the masked input (if desired)
        if (input_noise > 1e-4):
            self.input = droppy_input + (input_noise * \
                    self.srng.normal(size=input.shape, dtype=theano.config.floatX))
        else:
            self.input = droppy_input
        """

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
            self.linear_output = T.dot(self.input, self.W) + self.b
        else:
            self.linear_output = T.dot(self.input, self.W)

        # Add noise to the pre-activation features (if desired)
        self.noisy_linear = self.linear_output  + \
                (preact_noise * self.srng.normal(size=self.linear_output.shape, \
                dtype=theano.config.floatX))

        # Apply activation function
        self.output = self.activation(self.noisy_linear)

        # Compute some sums of the activations, for regularizing
        self.act_l2_sum = T.sum(self.output**2.) / self.output.size
        self.row_l1_sum = T.sum(abs(row_normalize(self.output))) / self.output.shape[0]
        self.col_l1_sum = T.sum(abs(col_normalize(self.output))) / self.output.shape[1]

        # Conveniently package layer parameters
        if use_bias:
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]

    def _drop_from_input(self, input, p):
        """p is the probability of dropping elements of input."""
        # p=1-p because 1's indicate keep and p is prob of dropping
        drop_mask = self.srng.binomial(n=1, p=1-p, size=input.shape, dtype=theano.config.floatX)
        drop_scale = 1. / (1. - p)
        # Cast mask from int to float32, to keep things on GPU
        droppy_input = drop_scale * input * drop_mask
        return droppy_input

    def _noisy_W(self, noise_lvl=0.):
        """Noise weights, like blurring the energy surface."""
        W_nz = self.W + self.srng.normal(size=self.W.shape, avg=0., std=noise_lvl)
        return W_nz

class JoinLayer(object):
    def __init__(self, input_layers):
        print("making join layer over {0:d} layers...".format(len(input_layers)))
        il_los = [il.linear_output for il in input_layers]
        self.linear_output = T.mean(T.stack(*il_los), axis=0)
        return


##########################
# NETWORK IMPLEMENTATION #
##########################

class EAR_NET(object):
    """A multipurpose ensemble of noise-perturbed neural networks.

    TODO: Add documentation.
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
        self.proto_configs = params['proto_configs']
        self.spawn_configs = params['spawn_configs']
        self.is_semisupervised = 0
        self.ear_type = params['ear_type']
        self.ear_lam = theano.shared(value=np.asarray([params['ear_lam']], \
                dtype=theano.config.floatX), name='ear_lam')
        self.spawn_weights = theano.shared(value=np.asarray(params['spawn_weights'], \
                dtype=theano.config.floatX), name='spawn_weights')
        # Compute some "structural" properties of this ensemble
        self.max_proto_depth = max([len(pc) for pc in self.proto_configs])
        self.spawn_count = len(self.spawn_configs)
        self.ear_pairs = self.spawn_count * (self.spawn_count - 1)
        # Make a dict to tell which parameters are norm-boundable
        self.clip_params = {}
        # Initialize all of the proto-networks
        self.proto_nets = []
        self.input = input
        # Construct the proto-networks from which to generate spawn-sembles
        for (pn_num, proto_config) in enumerate(self.proto_configs):
            layer_sizes = [ls for ls in proto_config]
            layer_connect_dims = zip(layer_sizes, layer_sizes[1:])
            layer_num = 0
            proto_net = []
            pn_name = "pn{0:d}".format(pn_num)
            next_input = self.input
            for n_in, n_out in layer_connect_dims:
                last_layer = (layer_num == (len(layer_connect_dims) - 1))
                drop_prob = 0.2 if (layer_num == 0) else 0.5
                activation = (lambda x: noop_actfun(x)) if last_layer else self.act_fun
                # Add a new layer to the regular model
                proto_net.append(HiddenLayer(rng=rng, \
                        input=next_input, \
                        activation=activation, \
                        pool_size=4, \
                        drop_rate=0., input_noise=0., preact_noise=0., \
                        n_in=n_in, n_out=n_out, use_bias=use_bias, name=pn_name))
                next_input = proto_net[-1].output
                # Set the non-bias parameters of this layer to be clipped
                self.clip_params[proto_net[-1].W] = 1
                layer_num = layer_num + 1
            # Add this network to the list of proto-networks
            self.proto_nets.append(proto_net)
        # Create a layer that joins the linear outputs of the proto-networks
        if len(self.proto_nets) > 1:
            self.proto_join_layer = JoinLayer([pn[-1] for pn in self.proto_nets])
        else:
            self.proto_join_layer = JoinLayer([self.proto_nets[0][-1], \
                    self.proto_nets[0][-1]])
        # Initialize all of the spawned (i.e. noise-perturbed) networks
        self.spawn_nets = []
        for spawn_config in self.spawn_configs:
            proto_key = spawn_config['proto_key']
            print("spawned from proto-net: {0:d} (of {1:d})".format(proto_key, len(self.proto_nets)))
            input_noise = spawn_config['input_noise']
            preact_noise = spawn_config['preact_noise']
            do_dropout = spawn_config['do_dropout']
            assert((proto_key >= 0) and (proto_key < len(self.proto_nets)))
            # Get info about the proto-network to spawn from
            layer_num = 0
            spawn_net = []
            next_input = self.input
            proto_net = self.proto_nets[proto_key]
            for proto_layer in proto_net:
                last_layer = (layer_num == (len(proto_net) - 1))
                drop_prob = (0.2 if (layer_num == 0) else 0.5) if do_dropout else 0.0
                # Get important properties from the relevant proto-layer
                activation = proto_layer.activation
                in_dim = proto_layer.in_dim
                out_dim = proto_layer.out_dim
                # Add a new layer to the regular model
                spawn_net.append(HiddenLayer(rng=rng, \
                        input=next_input, \
                        activation=activation, \
                        pool_size=4, drop_rate=drop_prob, \
                        input_noise=input_noise, preact_noise=preact_noise, \
                        W=proto_layer.W, b=proto_layer.b, \
                        n_in=in_dim, n_out=out_dim, use_bias=use_bias))
                next_input = spawn_net[-1].output
                layer_num = layer_num + 1
            # Add this network to the list of spawn-networks
            self.spawn_nets.append(spawn_net)

        # Mash all the parameters together, listily. Keep an extra list
        # comprising only parameters located in final/classification layers
        # of some proto-network (for use in fine-tuning, probably).
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
        # be the mean DAE cost over all i'th layers in any proto-network.
        self._construct_dae_layers(rng, lam_l1=0.15, nz_lvl=0.30)

        # Use a combination of some classification loss, at the output layer of
        # each spawn network, and some regularization terms as an objective
        # to optimize.
        self.proto_out_func = MCL2HingeSS(self.proto_join_layer)
        self.proto_class_loss = self.proto_out_func.loss_func
        self.proto_class_errors = self.proto_out_func.errors

        self.spawn_out_funcs = [MCL2HingeSS(sn[-1]) for sn in self.spawn_nets]
        self.spawn_class_costs = []
        for (i, spawn_out_func) in enumerate(self.spawn_out_funcs):
            spawn_class_cost = lambda y: \
                    self.spawn_weights[i] * spawn_out_func.loss_func(y)
            self.spawn_class_costs.append(spawn_class_cost)
        self.spawn_class_cost = lambda y: T.sum([scc(y) for scc in self.spawn_class_costs])
        self.spawn_reg_cost = lambda y: self.ear_cost(y)


    def ear_cost(self, y):
        """Compute the cost of ensemble agreement regularization."""
        ear_losses = []
        for i in range(self.spawn_count):
            for j in range(self.spawn_count):
                if not (i == j):
                    x1 = self.spawn_nets[i][-1].linear_output
                    x2 = self.spawn_nets[j][-1].linear_output
                    ear_loss = self.ear_lam[0] * self._ear_loss(x1, x2, y, self.ear_type)
                    ear_losses.append(ear_loss)
        total_loss = T.sum(ear_losses) / self.ear_pairs
        return ear_loss

    def _ear_loss(self, X1, X2, Y, ear_type):
        """Compute Ensemble Agreement Regularization cost for outputs X1/X2.
        """
        if not self.is_semisupervised:
            # Compute EAR regularizer using _all_ observations, not just those
            # with class label 0. (assume -1 is not a class label...)
            ss_mask = T.neq(Y, -1).reshape((Y.shape[0], 1))
        else:
            # Compute EAR regularizer only for observations with class label 0
            ss_mask = T.eq(Y, 0).reshape((Y.shape[0], 1))
        var_fun = lambda x1, x2: T.sum(((x1 - x2) * ss_mask)**2.) / T.sum(ss_mask)
        tanh_fun = lambda x1, x2: var_fun(T.tanh(x1), T.tanh(x2))
        norm_fun = lambda x1, x2: var_fun(row_normalize(x1), row_normalize(x2))
        sigm_fun = lambda x1, x2: var_fun(T.nnet.sigmoid(x1), T.nnet.sigmoid(x2))
        bent_fun = lambda p, q: T.sum(ss_mask * T.nnet.binary_crossentropy( \
                T.nnet.sigmoid(xo), T.nnet.sigmoid(xt))) / T.sum(ss_mask)
        ment_fun = lambda p, q: T.sum(ss_mask * smooth_cross_entropy(p, q)) / T.sum(ss_mask)
        kl_fun = lambda p, q: T.sum(ss_mask * smooth_kl_divergence(p, q)) / T.sum(ss_mask)
        if (ear_type == 1):
            # Unit-normalized variance (like fake cosine distance)
            ear_fun = norm_fun
        elif (ear_type == 2):
            # Tanh-transformed variance
            ear_fun = tanh_fun
        elif (ear_type == 3):
            # Sigmoid-transformed variance
            ear_fun = sigm_fun
        elif (ear_type == 4):
            # Binary cross-entropy
            ear_fun = bent_fun
        elif (ear_type == 5):
            # Multinomial cross-entropy
            ear_fun = ment_fun
        elif (ear_type == 6):
            # Multinomial KL-divergence
            ear_fun = kl_fun
        else:
            ear_fun = var_fun
        return ear_fun(X1, X2)

    def _construct_dae_layers(self, rng, lam_l1=0., nz_lvl=0.25):
        """Build cost functions for training DAEs defined for all hidden
        layers of the proto-networks making up this generalized ensemble. That
        is, construct a DAE for every proto-layer that isn't a classification
        layer."""
        # Construct a DAE for each hidden layer in this network.
        self.dae_params = []
        self.dae_costs = []
        for d in range(self.max_proto_depth - 2):
            d_params = []
            d_costs = []
            for pn in self.proto_nets:
                if (d < (len(pn) - 1)):
                    # Construct a DAE for this proto-net hidden layer
                    W_pn = pn[d].W
                    b_pn = pn[d].b
                    input_pn = pn[d].input
                    vis_dim = pn[d].in_dim
                    hid_dim = pn[d].out_dim
                    # Construct the DAE layer object
                    dae_layer = DAELayer(rng=rng, \
                            input=input_pn, \
                            n_in=vis_dim, n_out=hid_dim, \
                            activation=self.act_fun, \
                            input_noise=nz_lvl, \
                            W=W_pn, b_h=b_pn, b_v=None)
                    d_params.extend(dae_layer.params)
                    d_costs.append(dae_layer.compute_costs(lam_l1))
            # Record the set of all DAE params to-be-optimized at depth d
            self.dae_params.append(d_params)
            # Record the sum of reconstruction costs for DAEs at depth d (in
            # self.dae_costs[d][0]) and the sum of sparse regularization costs
            # for DAEs at depth d (in self.dae_costs[d][1]).
            self.dae_costs.append([T.sum([c[0] for c in d_costs]), T.sum([c[1] for c in d_costs])])
        return

    def set_ear_lam(self, e_lam):
        """Set the Ensemble Agreement Regularization weight."""
        self.ear_lam.set_value(np.asarray([e_lam], dtype=theano.config.floatX))
        return


###########################################
# DENOISING AUTOENCODER IMPLEMENTATION... #
###########################################


class DAELayer(object):
    def __init__(self, rng, input, n_in, n_out, \
                 activation=None, input_noise=0., \
                 W=None, b_h=None, b_v=None):

        # Setup a shared random generator for this layer
        self.srng = theano.tensor.shared_randomstreams.RandomStreams( \
                rng.randint(100000))

        self.input = input
        self.noisy_input, self.noise_mask = self._get_noisy_input(input, input_noise)

        # Set some basic layer properties
        self.activation = activation
        self.in_dim = n_in
        self.out_dim = n_out

        # Get some random initial weights and biases, if not given
        if W is None:
            W_init = np.asarray(0.01 * rng.standard_normal( \
                      size=(n_in, n_out)), dtype=theano.config.floatX)
            W = theano.shared(value=W_init, name='W')
        if b_h is None:
            b_init = np.zeros((n_out,), dtype=theano.config.floatX)
            b_h = theano.shared(value=b_init, name='b_h')
        if b_v is None:
            b_init = np.zeros((n_in,), dtype=theano.config.floatX)
            b_v = theano.shared(value=b_init, name='b_v')

        self.W = W
        self.b_h = b_h
        self.b_v = b_v

        self.params = [self.W, self.b_h, self.b_v]

    def compute_costs(self, lam_l1=0.):
        """Compute reconstruction and activation sparsity costs."""
        # Get noisy weights
        W_nz = self._noisy_W(self.W, 0.01)
        # Compute hidden and visible activations
        A_v, A_h = self._compute_activations(self.noisy_input, W_nz, self.b_h, self.b_v)
        # Compute reconstruction error cost
        recon_cost = T.sum((self.input - A_v)**2.0) / self.input.shape[0]
        # Compute sparsity penalty
        row_l1_sum = T.sum(abs(row_normalize(A_h))) / A_h.shape[0]
        col_l1_sum = T.sum(abs(col_normalize(A_h))) / A_h.shape[1]
        sparse_cost = lam_l1 * (row_l1_sum + col_l1_sum)
        return [recon_cost, sparse_cost]

    def _compute_hidden_acts(self, X, W, b):
        """Compute activations at hidden layer."""
        A_h = self.activation(T.dot(X, W) + b)
        return A_h

    def _compute_activations(self, X, W, b_h, b_v):
        """Compute activations of decoder (at visible layer)."""
        A_h = self._compute_hidden_acts(X, W, b_h)
        A_v = T.dot(A_h, W.T) + b_v
        return [A_v, A_h]

    def _noisy_W(self, W, noise_lvl=0.):
        """Noise weights, like blurring the energy surface."""
        W_nz = W + self.srng.normal(size=W.shape, avg=0., std=noise_lvl, \
                dtype=theano.config.floatX)
        return W_nz

    def _get_noisy_input(self, input, p):
        """p is the probability of dropping elements of input."""
        # p=1-p because 1's indicate keep and p is prob of dropping
        noise_mask = self.srng.binomial(n=1, p=1-p, size=input.shape, dtype=theano.config.floatX)
        # Cast mask from int to float32, to keep things on GPU
        noisy_input = input * noise_mask
        return [noisy_input, noise_mask]

##############
# EYE BUFFER #
##############
