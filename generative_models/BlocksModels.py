from __future__ import division, print_function

import sys
sys.path.append("./lib")

import logging
import theano
import numpy
import cPickle

from theano import tensor
from collections import OrderedDict

from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.bricks.base import application, _Brick, Brick, lazy
from blocks.bricks.recurrent import BaseRecurrent, recurrent
from blocks.initialization import Constant, IsotropicGaussian, Orthogonal
from blocks.bricks import Random, MLP, Linear, Tanh, Softmax, Sigmoid, Initializable
from blocks.bricks import Tanh, Identity, Activation, Feedforward
from blocks.bricks.cost import BinaryCrossEntropy
from blocks.utils import shared_floatx_nans
from blocks.roles import add_role, WEIGHT, BIAS, PARAMETER, AUXILIARY

from BlocksAttention import ZoomableAttentionWindow
from DKCode import get_adam_updates
from HelperFuncs import constFX, to_fX

##################################
# Probability distribution stuff #
##################################

def log_prob_bernoulli(p_true, p_approx, mask=None):
    """
    Compute log probability of some binary variables with probabilities
    given by p_true, for probability estimates given by p_approx. We'll
    compute joint log probabilities over row-wise groups.
    """
    if mask is None:
        mask = tensor.ones((1, p_approx.shape[1]))
    log_prob_1 = p_true * tensor.log(p_approx+1e-6)
    log_prob_0 = (1.0 - p_true) * tensor.log((1.0 - p_approx)+1e-6)
    log_prob_01 = log_prob_1 + log_prob_0
    row_log_probs = tensor.sum((log_prob_01 * mask), axis=1, keepdims=True)
    #row_log_probs = -1.0 * tensor.sum( \
    #    (tensor.nnet.binary_crossentropy(p_approx, p_true) * mask), \
    #    axis=1, keepdims=True)
    return row_log_probs

def gaussian_kld(mu_left, logvar_left, mu_right, logvar_right):
    """
    Compute KL divergence between a bunch of univariate Gaussian distributions
    with the given means and log-variances.
    We do KL(N(mu_left, logvar_left) || N(mu_right, logvar_right)).
    """
    gauss_klds = 0.5 * (logvar_right - logvar_left + \
            (tensor.exp(logvar_left) / tensor.exp(logvar_right)) + \
            ((mu_left - mu_right)**2.0 / tensor.exp(logvar_right)) - 1.0)
    return gauss_klds

def bernoulli_kld(p_left, p_right):
    """
    Compute KL divergence between a pair of bernoulli distributions.
    """
    term_a = p_left * (tensor.log(p_left) - tensor.log(p_right))
    term_b = (1.0 - p_left) * \
             (tensor.log(1.0 - p_left) - tensor.log(1.0 - p_right))
    bern_kld = term_a + term_b
    return bern_kld


################################
# Softplus activation function #
################################

class Softplus(Activation):
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return tensor.nnet.softplus(input_)

class BiasedLSTM(BaseRecurrent, Initializable):
    @lazy(allocation=['dim'])
    def __init__(self, dim, ig_bias=0.0, fg_bias=0.0, og_bias=0.0, 
                 activation=None, **kwargs):
        super(BiasedLSTM, self).__init__(**kwargs)
        self.dim = dim
        self.ig_bias = constFX(ig_bias) # input gate bias
        self.fg_bias = constFX(fg_bias) # forget gate bias
        self.og_bias = constFX(og_bias) # output gate bias

        if not activation:
            activation = Tanh()
        self.children = [activation]
        return

    def get_dim(self, name):
        if name == 'inputs':
            return self.dim * 4
        if name in ['states', 'cells']:
            return self.dim
        if name == 'mask':
            return 0
        return super(BiasedLSTM, self).get_dim(name)

    def _allocate(self):
        self.W_state = shared_floatx_nans((self.dim, 4*self.dim),
                                          name='W_state')
        self.W_cell_to_in = shared_floatx_nans((self.dim,),
                                               name='W_cell_to_in')
        self.W_cell_to_forget = shared_floatx_nans((self.dim,),
                                                   name='W_cell_to_forget')
        self.W_cell_to_out = shared_floatx_nans((self.dim,),
                                                name='W_cell_to_out')
        add_role(self.W_state, WEIGHT)
        add_role(self.W_cell_to_in, WEIGHT)
        add_role(self.W_cell_to_forget, WEIGHT)
        add_role(self.W_cell_to_out, WEIGHT)

        self.params = [self.W_state, self.W_cell_to_in, self.W_cell_to_forget,
                       self.W_cell_to_out]
        return

    def _initialize(self):
        for w in self.params:
            self.weights_init.initialize(w, self.rng)
        return

    @recurrent(sequences=['inputs', 'mask'], states=['states', 'cells'],
               contexts=[], outputs=['states', 'cells'])
    def apply(self, inputs, states, cells, mask=None):
        """Apply the Long Short Term Memory transition.
        Parameters
        ----------
        states : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of current states in the shape
            (batch_size, features). Required for `one_step` usage.
        cells : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of current cells in the shape
            (batch_size, features). Required for `one_step` usage.
        inputs : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of inputs in the shape (batch_size,
            features * 4).
        mask : :class:`~tensor.TensorVariable`
            A 1D binary array in the shape (batch,) which is 1 if there is
            data available, 0 if not. Assumed to be 1-s only if not given.
        Returns
        -------
        states : :class:`~tensor.TensorVariable`
            Next states of the network.
        cells : :class:`~tensor.TensorVariable`
            Next cell activations of the network.
        """
        def slice_last(x, no):
            return x.T[no*self.dim: (no+1)*self.dim].T
        nonlinearity = self.children[0].apply

        activation = tensor.dot(states, self.W_state) + inputs
        in_gate = tensor.nnet.sigmoid(slice_last(activation, 0) +
                                      (cells * self.W_cell_to_in) + 
                                      self.ig_bias)
        forget_gate = tensor.nnet.sigmoid(slice_last(activation, 1) +
                                          (cells * self.W_cell_to_forget) +
                                          self.fg_bias)
        next_cells = (forget_gate * cells +
                      in_gate * nonlinearity(slice_last(activation, 2)))
        out_gate = tensor.nnet.sigmoid(slice_last(activation, 3) +
                                       (next_cells * self.W_cell_to_out) + 
                                       self.og_bias)
        next_states = out_gate * nonlinearity(next_cells)

        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
            next_cells = (mask[:, None] * next_cells +
                          (1 - mask[:, None]) * cells)

        return next_states, next_cells

###################################################
# Diagonal Gaussian conditional density estimator #
###################################################

class CondNet(Initializable, Feedforward):
    """A simple multi-layer perceptron for diagonal Gaussian conditionals.

    Note -- For now, we require both activations and dims to be specified.

    Parameters
    ----------
    activations : list of :class:`.Brick`, :class:`.BoundApplication`,
                  or ``None``
        A list of activations to apply after each linear transformation.
        Give ``None`` to not apply any activation. It is assumed that the
        application method to use is ``apply``. Required for
        :meth:`__init__`. The length of this list should be two less than
        the length of dims, as first dim is the input dim and the last dim
        is the dim of the output Gaussian.
    dims : list of ints
        A list of input dimensions, as well as the output dimension of the
        last layer. Required for :meth:`~.Brick.allocate`.
    """
    def __init__(self, activations=None, dims=None, **kwargs):
        if activations is None:
            raise ValueError("activations must be specified.")
        if dims is None:
            raise ValueError("dims must be specified.")
        if not (len(dims) == (len(activations) + 2)):
            raise ValueError("len(dims) != len(activations) + 2.")
        super(CondNet, self).__init__(**kwargs)
        
        self.dims = dims
        self.shared_acts = activations

        # construct the shared linear transforms for feedforward
        self.shared_linears = []
        for i in range(len(dims)-2):
            self.shared_linears.append( \
                Linear(dims[i], dims[i+1], name='shared_linear_{}'.format(i)))

        self.mean_linear = Linear(dims[-2], dims[-1], name='mean_linear')
        self.logvar_linear = Linear(dims[-2], dims[-1], name='logvar_linear',
                                    weights_init=Constant(0.))

        self.children = self.shared_linears + self.shared_acts
        self.children.append(self.mean_linear)
        self.children.append(self.logvar_linear)
        return

    def get_dim(self, name):
        if name == 'input':
            return self.dims[0]
        elif name == 'output':
            return self.dims[-1]
        else:
            raise ValueError("Invalid dim name: {}".format(name))
        return

    @property
    def input_dim(self):
        return self.dims[0]

    @property
    def output_dim(self):
        return self.dims[-1]

    @application(inputs=['x', 'u'], outputs=['z_mean', 'z_logvar', 'z'])
    def apply(self, x, u):
        f = [ x ]
        for linear, activation in zip(self.shared_linears, self.shared_acts):
            f.append( activation.apply(linear.apply(f[-1])) )
        z_mean = self.mean_linear.apply(f[-1])
        z_logvar = self.logvar_linear.apply(f[-1])
        z = z_mean + (u * tensor.exp(0.5 * z_logvar))
        return z_mean, z_logvar, z

class TanhMLPwFFBP(Initializable, Feedforward):
    def __init__(self, in_dim, layer_dims, out_dim, **kwargs):
        super(TanhMLPwFFBP, self).__init__(**kwargs)
        self.in_dim = in_dim
        self.layer_dims = layer_dims
        self.out_dim = out_dim
        return

    def _allocate(self):
        self.W_list = []
        self.b_list = []
        in_dims = [self.in_dim] + self.layer_dims
        out_dims = self.layer_dims + [self.out_dim]
        for i, in_dim in enumerate(in_dims):
            out_dim = out_dims[i]
            W_name = "W_{}2{}".format(i, i+1)
            b_name = "b_{}2{}".format(i, i+1)
            Wi = shared_floatx_nans((in_dim, out_dim), \
                                    name=W_name)
            bi = shared_floatx_nans((out_dim, ), \
                                    name=b_name)
            self.W_list.append(Wi)
            self.b_list.append(bi)
            add_role(Wi, WEIGHT)
            add_role(bi, BIAS)
        self.params = self.W_list + self.b_list
        return

    def _initialize(self):
        for W in self.W_list:
            self.weights_init.initialize(W, self.rng)
        for b in self.b_list:
            b_nan = b.get_value(borrow=False)
            b_zeros = numpy.zeros(b_nan.shape)
            b.set_value(b_zeros.astype(theano.config.floatX))
        return

    def get_dim(self, name):
        if name in ['input', 'grad_on_input']:
            return self.in_dim
        elif name in ['output', 'target']:
            return self.out_dim
        else:
            raise ValueError("Invalid dim name: {}".format(name))
        return

    @property
    def input_dim(self):
        return self.in_dim

    @property
    def output_dim(self):
        return self.out_dim

    @application(inputs=['input'], outputs=['output'])
    def apply(self, input):
        """
        Compute feedforward pass only.
        """
        # feedforward through the network
        ff_acts = [input]
        for i in range(len(self.W_list)):
            Wi = self.W_list[i]
            bi = self.b_list[i]
            line_act = tensor.dot(ff_acts[i], Wi) + bi
            if i < (len(self.W_list) - 1):
                # hidden layer activations
                tanh_act = tensor.tanh(line_act)
                ff_acts.append(tanh_act)
            else:
                # output layer activations
                ff_acts.append(line_act)
        output = ff_acts[-1]
        return output

    @application(inputs=['input', 'target'], \
                 outputs=['output', 'grad_on_input'])
    def apply_ffbp(self, input, target):
        """
        Compute forward and backward passes.
        """
        # feedforward through the network
        ff_acts = [input]
        for i in range(len(self.W_list)):
            Wi = self.W_list[i]
            bi = self.b_list[i]
            line_act = tensor.dot(ff_acts[i], Wi) + bi
            if i < (len(self.W_list) - 1):
                # hidden layer activations
                tanh_act = tensor.tanh(line_act)
                ff_acts.append(tanh_act)
            else:
                # output layer activations
                ff_acts.append(line_act)
        output = ff_acts[-1]
        # compute grad on output, assuming a bernoulli generative model
        sigm_act = tensor.nnet.sigmoid(output)
        grad_nll = target - sigm_act # grad on post-sigmoid activation
        sigm_grad = grad_nll * ((1.0 - sigm_act) * sigm_act) # bp through sigmoid
        # backpropagate through the network
        bp_grads = [grad_nll]
        for i in range(len(self.W_list)):
            Wi = self.W_list[-(i+1)]
            # backprop through the linear part of this layer
            semi_bp_grad = tensor.dot(bp_grads[i], Wi.T)
            if i < (len(self.W_list) - 1):
                # backprop through the tanh part of this layer
                tanh_act = ff_acts[-(i+2)]
                tanh_grad = 1.0 - tanh_act**2.0
                bp_grads.append(tanh_grad * semi_bp_grad)
            else:
                # no tanh backprop to perform at the input layer
                bp_grads.append(semi_bp_grad)
        grad_on_input = bp_grads[-1]
        return output, grad_on_input

###########################################
# QSampler for a single diagonal Gaussian #
###########################################

class Qsampler(Initializable, Random):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(Qsampler, self).__init__(**kwargs)

        self.mean_transform = Linear(
                name=self.name+'_mean',
                input_dim=input_dim, output_dim=output_dim, 
                weights_init=self.weights_init, biases_init=self.biases_init,
                use_bias=True)

        self.logvar_transform = Linear(
                name=self.name+'_logvar',
                input_dim=input_dim, output_dim=output_dim, 
                weights_init=self.weights_init, biases_init=self.biases_init,
                use_bias=True)

        self.children = [self.mean_transform, self.logvar_transform]
        return
    
    def get_dim(self, name):
        if name == 'input':
            return self.mean_transform.get_dim('input')
        elif name == 'output':
            return self.mean_transform.get_dim('output')
        else:
            raise ValueError
        return

    @application(inputs=['x', 'u'], outputs=['z', 'all_klds'])
    def sample(self, x, u):
        """
        Return samples and the corresponding KL term

        Parameters
        ----------
        x : input features for estimating mean and log-variance
        u : standard Normal samples to scale and shift

        Returns
        -------
        z : tensor.matrix
            Samples drawn from Q(z|x)
        kl : tensor.vector
             KL(Q(z|x) || P_z)
        
        """
        mean = self.mean_transform.apply(x)
        logvar = self.logvar_transform.apply(x)

        # ... and scale/translate samples
        z = mean + tensor.exp(0.5 * logvar) * u

        # Calculate KL
        all_klds = gaussian_kld(mean, logvar, 0.0, 0.0)
        return z, all_klds

    @application(inputs=['u'], outputs=['z_prior'])
    def sample_from_prior(self, u):
        """
        Sample z from the prior distribution P_z.

        Parameters
        ----------
        u : tensor.matrix
            gaussian random source 

        Returns
        -------
        z : tensor.matrix
            samples 

        """
        z_prior = u
        return z_prior

#-----------------------------------------------------------------------------

class Reader(Initializable):
    def __init__(self, x_dim, dec_dim, **kwargs):
        super(Reader, self).__init__(name="reader", **kwargs)

        self.x_dim = x_dim
        self.dec_dim = dec_dim
        self.output_dim = 2*x_dim

    def get_dim(self, name):
        if name == 'input':
            return self.dec_dim
        elif name == 'x_dim':
            return self.x_dim
        elif name == 'output':
            return self.output_dim
        else:
            raise ValueError

    @application(inputs=['x', 'x_hat', 'h_dec'], outputs=['r'])
    def apply(self, x, x_hat, h_dec):
        return tensor.concatenate([x, x_hat], axis=1)

class AttentionReader(Initializable):
    def __init__(self, x_dim, dec_dim, height, width, N, **kwargs):
        super(AttentionReader, self).__init__(name="reader", **kwargs)

        self.img_height = height
        self.img_width = width
        self.N = N
        self.x_dim = x_dim
        self.dec_dim = dec_dim
        self.output_dim = 2*N*N

        self.pre_trafo = Linear(
                name=self.name+'_pretrafo',
                input_dim=dec_dim, output_dim=dec_dim, 
                weights_init=self.weights_init, biases_init=self.biases_init,
                use_bias=True)

        self.zoomer = ZoomableAttentionWindow(height, width, N)
        self.readout = MLP(activations=[Identity()], dims=[dec_dim, 5], **kwargs)

        self.children = [self.pre_trafo, self.readout]

    def get_dim(self, name):
        if name == 'input':
            return self.dec_dim
        elif name == 'x_dim':
            return self.x_dim
        elif name == 'output':
            return self.output_dim
        else:
            raise ValueError
            
    @application(inputs=['x', 'x_hat', 'h_dec'], outputs=['r'])
    def apply(self, x, x_hat, h_dec):
        p = self.pre_trafo.apply(h_dec)
        l = self.readout.apply(p)

        center_y, center_x, delta, sigma, gamma = self.zoomer.nn2att(l)

        w     = gamma * self.zoomer.read(x    , center_y, center_x, delta, sigma)
        w_hat = gamma * self.zoomer.read(x_hat, center_y, center_x, delta, sigma)
        
        return tensor.concatenate([w, w_hat], axis=1)

#-----------------------------------------------------------------------------

class Writer(Initializable):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(Writer, self).__init__(name="writer", **kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.transform = Linear(
                name=self.name+'_transform',
                input_dim=input_dim, output_dim=output_dim, 
                weights_init=self.weights_init, biases_init=self.biases_init,
                use_bias=True)

        self.children = [self.transform]

    @application(inputs=['h'], outputs=['c_update'])
    def apply(self, h):
        return self.transform.apply(h)

class AttentionWriter(Initializable):
    def __init__(self, input_dim, output_dim, width, height, N, **kwargs):
        super(AttentionWriter, self).__init__(name="writer", **kwargs)

        self.img_width = width
        self.img_height = height
        self.N = N
        self.input_dim = input_dim
        self.output_dim = output_dim

        assert output_dim == width*height

        self.zoomer = ZoomableAttentionWindow(height, width, N)

        self.pre_trafo = Linear(
                name=self.name+'_pretrafo',
                input_dim=input_dim, output_dim=input_dim, 
                weights_init=self.weights_init, biases_init=self.biases_init,
                use_bias=True)  

        self.z_trafo = Linear(
                name=self.name+'_ztrafo',
                input_dim=input_dim, output_dim=5, 
                weights_init=self.weights_init, biases_init=self.biases_init,
                use_bias=True)

        self.w_trafo = Linear(
                name=self.name+'_wtrafo',
                input_dim=input_dim, output_dim=N*N, 
                weights_init=self.weights_init, biases_init=self.biases_init,
                use_bias=True)

        self.children = [self.pre_trafo, self.z_trafo, self.w_trafo]

    @application(inputs=['h'], outputs=['c_update'])
    def apply(self, h):
        p = self.pre_trafo.apply(h)
        w = self.w_trafo.apply(p)
        l = self.z_trafo.apply(p)

        center_y, center_x, delta, sigma, gamma = self.zoomer.nn2att(l)

        c_update = 1./gamma * self.zoomer.write(w, center_y, center_x, delta, sigma)

        return c_update

    @application(inputs=['h'], outputs=['c_update', 'center_y', 'center_x', 'delta'])
    def apply_detailed(self, h):
        p = self.pre_trafo.apply(h)
        w = self.w_trafo.apply(p)
        l = self.z_trafo.apply(p)

        center_y, center_x, delta, sigma, gamma = self.zoomer.nn2att(l)

        c_update = 1./gamma * self.zoomer.write(w, center_y, center_x, delta, sigma)

        return c_update, center_y, center_x, delta

##########################################################
# Generalized DRAW model, with infinite mixtures and RL. #
#    -- this only works open-loopishly                   #
##########################################################

class IMoOLDrawModels(BaseRecurrent, Initializable, Random):
    def __init__(self, n_iter, step_type, mix_enc_mlp, mix_dec_mlp,
                    reader_mlp, enc_mlp_in, enc_rnn, enc_mlp_out,
                    dec_mlp_in, dec_rnn, dec_mlp_out, writer_mlp,
                    **kwargs):
        super(IMoOLDrawModels, self).__init__(**kwargs)
        if not ((step_type == 'add') or (step_type == 'jump')):
            raise ValueError('step_type must be jump or add')
        # record the desired step count
        self.n_iter = n_iter
        self.step_type = step_type
        # grab handles for mixture stuff
        self.mix_enc_mlp = mix_enc_mlp
        self.mix_dec_mlp = mix_dec_mlp
        # grab handles for IMoOLDRAW model stuff
        self.reader_mlp = reader_mlp
        self.enc_mlp_in = enc_mlp_in
        self.enc_rnn = enc_rnn
        self.enc_mlp_out = enc_mlp_out
        self.dec_mlp_in = dec_mlp_in
        self.dec_rnn = dec_rnn
        self.dec_mlp_out = dec_mlp_out
        self.writer_mlp = writer_mlp

        # record the sub-models that underlie this model
        self.children = [self.mix_enc_mlp, self.mix_dec_mlp, self.reader_mlp,
                         self.enc_mlp_in, self.enc_rnn, self.enc_mlp_out,
                         self.dec_mlp_in, self.dec_rnn, self.dec_mlp_out,
                         self.writer_mlp]
        return

    def _allocate(self):
        c_dim = self.get_dim('c')
        zm_dim = self.get_dim('z_mix')
        # self.c_0 provides the initial state of the canvas
        self.c_0 = shared_floatx_nans((c_dim,), name='c_0')
        # self.zm_mean provides the mean of z_mix
        self.zm_mean = shared_floatx_nans((zm_dim,), name='zm_mean')
        # self.zm_logvar provides the logvar of z_mix
        self.zm_logvar = shared_floatx_nans((zm_dim,), name='zm_logvar')
        add_role(self.c_0, PARAMETER)
        add_role(self.zm_mean, PARAMETER)
        add_role(self.zm_logvar, PARAMETER)
        # add the theano shared variables to our parameter lists
        self.params.extend([ self.c_0, self.zm_mean, self.zm_logvar ])
        return

    def _initialize(self):
        # initialize to all parameters zeros...
        for p in self.params:
            p_nan = p.get_value(borrow=False)
            p_zeros = numpy.zeros(p_nan.shape)
            p.set_value(p_zeros.astype(theano.config.floatX))
        return
 
    def get_dim(self, name):
        if name == 'c':
            return self.reader_mlp.get_dim('x_dim')
        elif name == 'z_mix':
            return self.mix_enc_mlp.get_dim('output')
        elif name == 'h_enc':
            return self.enc_rnn.get_dim('states')
        elif name == 'c_enc':
            return self.enc_rnn.get_dim('cells')
        elif name == 'z_gen':
            return self.enc_mlp_out.get_dim('output')
        elif name == 'h_dec':
            return self.dec_rnn.get_dim('states')
        elif name == 'c_dec':
            return self.dec_rnn.get_dim('cells')
        elif name in ['nll', 'kl_q2p', 'kl_p2q']:
            return 0
        elif name == 'center_y':
            return 0
        elif name == 'center_x':
            return 0
        elif name == 'delta':
            return 0
        else:
            super(IMoOLDrawModels, self).get_dim(name)
        return

    #------------------------------------------------------------------------

    @recurrent(sequences=['u'], contexts=['x'],
               states=['c', 'h_enc', 'c_enc', 'h_dec', 'c_dec', 'nll', 'kl_q2p', 'kl_p2q'],
               outputs=['c', 'h_enc', 'c_enc', 'h_dec', 'c_dec', 'nll', 'kl_q2p', 'kl_p2q'])
    def iterate(self, u, c, h_enc, c_enc, h_dec, c_dec, nll, kl_q2p, kl_p2q, x):
        if self.step_type == 'add':
            # additive steps use c as a "direct workspace", which means it's
            # already directly comparable to x.
            c = c
        else:
            # non-additive steps use c_dec as a "latent workspace", which means
            # it needs to be transformed before being comparable to x.
            c = self.writer_mlp.apply(c_dec)
        # get the current "reconstruction error"
        x_hat = x - tensor.nnet.sigmoid(c)
        r_enc = self.reader_mlp.apply(x, x_hat, h_dec)
        # update the encoder RNN state
        i_enc = self.enc_mlp_in.apply(tensor.concatenate([r_enc, h_dec], axis=1))
        h_enc, c_enc = self.enc_rnn.apply(states=h_enc, cells=c_enc,
                                          inputs=i_enc, iterate=False)
        # estimate encoder conditional over z given h_enc
        q_gen_mean, q_gen_logvar, q_z_gen = \
                self.enc_mlp_out.apply(h_enc, u)
        # estimate decoder conditional over z given h_dec
        p_gen_mean, p_gen_logvar, p_z_gen = \
                self.dec_mlp_out.apply(h_dec, u)
        # update the decoder RNN state
        z_gen = q_z_gen # use samples from q while training
        i_dec = self.dec_mlp_in.apply(tensor.concatenate([z_gen], axis=1))
        h_dec, c_dec = self.dec_rnn.apply(states=h_dec, cells=c_dec, \
                                          inputs=i_dec, iterate=False)
        # additive steps use c as the "workspace"
        if self.step_type == 'add':
            c = c + self.writer_mlp.apply(h_dec)
        else:
            c = self.writer_mlp.apply(c_dec)
        # compute the NLL of the reconstructiion as of this step
        c_as_x = tensor.nnet.sigmoid(c)
        nll = -1.0 * tensor.flatten(log_prob_bernoulli(x, c_as_x))
        # compute KL(q || p) and KL(p || q) for this step
        kl_q2p = tensor.sum(gaussian_kld(q_gen_mean, q_gen_logvar, \
                            p_gen_mean, p_gen_logvar), axis=1)
        kl_p2q = tensor.sum(gaussian_kld(p_gen_mean, p_gen_logvar, \
                            q_gen_mean, q_gen_logvar), axis=1)
        return c, h_enc, c_enc, h_dec, c_dec, nll, kl_q2p, kl_p2q

    @recurrent(sequences=['u'], contexts=[],
               states=['c', 'h_dec', 'c_dec'],
               outputs=['c', 'h_dec', 'c_dec'])
    def decode(self, u, c, h_dec, c_dec):
        batch_size = c.shape[0]
        # sample z from p(z | h_dec) -- we used q(z | h_enc) during training
        p_gen_mean, p_gen_logvar, p_z_gen = \
                self.dec_mlp_out.apply(h_dec, u)
        z_gen = p_z_gen
        # update the decoder RNN state
        i_dec = self.dec_mlp_in.apply(tensor.concatenate([z_gen], axis=1))
        h_dec, c_dec = self.dec_rnn.apply(
                    states=h_dec, cells=c_dec,
                    inputs=i_dec, iterate=False)
        # additive steps use c as the "workspace"
        if self.step_type == 'add':
            c = c + self.writer_mlp.apply(h_dec)
        else:
            c = self.writer_mlp.apply(c_dec)
        return c, h_dec, c_dec

    #------------------------------------------------------------------------

    @application(inputs=['x_in', 'x_out'], 
                 outputs=['recons', 'nll', 'kl_q2p', 'kl_p2q'])
    def reconstruct(self, x_in, x_out):
        # get important size and shape information
        batch_size = x_in.shape[0]
        z_mix_dim = self.get_dim('z_mix')
        z_gen_dim = self.get_dim('z_gen')
        ce_dim = self.get_dim('c_enc')
        cd_dim = self.get_dim('c_dec')
        he_dim = self.get_dim('h_enc')
        hd_dim = self.get_dim('h_dec')

        # sample zero-mean, unit std. Gaussian noise for mixture init
        u_mix = self.theano_rng.normal(
                    size=(batch_size, z_mix_dim),
                    avg=0., std=1.)
        # transform ZMUV noise based on q(z_mix | x_in)
        z_mix_mean, z_mix_logvar, z_mix = \
                self.mix_enc_mlp.apply(x_in, u_mix)
        # transform samples from q(z_mix | x_in) into initial generator state
        mix_init = self.mix_dec_mlp.apply(z_mix)
        cd0 = mix_init[:, :cd_dim]
        hd0 = mix_init[:, cd_dim:(cd_dim+hd_dim)]
        ce0 = mix_init[:, (cd_dim+hd_dim):(cd_dim+hd_dim+ce_dim)]
        he0 = mix_init[:, (cd_dim+hd_dim+ce_dim):]
        c0 = tensor.zeros_like(x_out) + self.c_0

        # compute KL-divergence information for the mixture init step
        kl_q2p_mix = tensor.sum(gaussian_kld(z_mix_mean, z_mix_logvar, \
                                self.zm_mean, self.zm_logvar), axis=1)
        kl_p2q_mix = tensor.sum(gaussian_kld(self.zm_mean, self.zm_logvar, \
                                z_mix_mean, z_mix_logvar), axis=1)
        kl_q2p_mix = kl_q2p_mix.reshape((1, batch_size))
        kl_p2q_mix = kl_p2q_mix.reshape((1, batch_size))

        # get zero-mean, unit-std. Gaussian noise for use in scan op
        u_gen = self.theano_rng.normal(
                    size=(self.n_iter, batch_size, z_gen_dim),
                    avg=0., std=1.)

        # run the multi-stage guided generative process
        c, _, _, _, _, step_nlls, kl_q2p_gen, kl_p2q_gen = \
                self.iterate(u=u_gen, c=c0, h_enc=he0, c_enc=ce0, \
                             h_dec=hd0, c_dec=cd0, x=x_out)

        # grab the observations generated by the multi-stage process
        recons = tensor.nnet.sigmoid(c[-1,:,:])
        recons.name = "recons"
        # get the NLL after the final update for each example
        nll = step_nlls[-1]
        nll.name = "nll"
        # group up the klds from mixture init and multi-stage generation
        kl_q2p = tensor.vertical_stack(kl_q2p_mix, kl_q2p_gen)
        kl_q2p.name = "kl_q2p"
        kl_p2q = tensor.vertical_stack(kl_p2q_mix, kl_p2q_gen)
        kl_p2q.name = "kl_p2q"
        return recons, nll, kl_q2p, kl_p2q

    @application(inputs=['n_samples'], outputs=['samples'])
    def sample(self, n_samples):
        """Sample from model.

        Returns 
        -------

        samples : tensor3 (n_samples, n_iter, x_dim)
        """
        z_mix_dim = self.get_dim('z_mix')
        z_gen_dim = self.get_dim('z_gen')
        cd_dim = self.get_dim('c_dec')
        hd_dim = self.get_dim('h_dec')
        ce_dim = self.get_dim('c_enc')
        he_dim = self.get_dim('h_enc')
        c_dim = self.get_dim('c')

        # sample zero-mean, unit-std. Gaussian noise for the mixture init
        u_mix = self.theano_rng.normal(
                    size=(n_samples, z_mix_dim),
                    avg=0., std=1.)
        # transform noise based on learned mean and logvar
        z_mix = self.zm_mean + (u_mix * tensor.exp(0.5 * self.zm_logvar))
        # transform the sample from p(z_mix) into an initial generator state
        mix_init = self.mix_dec_mlp.apply(z_mix)
        cd0 = mix_init[:, :cd_dim]
        hd0 = mix_init[:, cd_dim:(cd_dim+hd_dim)]
        c0 = tensor.alloc(0.0, n_samples, c_dim) + self.c_0

        # sample from zero-mean unit-std. Gaussian for use in scan op
        u_gen = self.theano_rng.normal(
                    size=(self.n_iter, n_samples, z_gen_dim),
                    avg=0., std=1.)

        c, _, _, = self.decode(u=u_gen, c=c0, h_dec=hd0, c_dec=cd0)
        #c, _, _, center_y, center_x, delta = self.decode(u)
        return tensor.nnet.sigmoid(c)

    def build_model_funcs(self):
        """
        Build the symbolic costs and theano functions relevant to this model.
        """
        # some symbolic vars to represent various inputs/outputs
        x_in_sym = tensor.matrix('x_in_sym')
        x_out_sym = tensor.matrix('x_out_sym')

        # collect reconstructions of x produced by the IMoOLDRAW model
        _, nll, kl_q2p, kl_p2q = self.reconstruct(x_in_sym, x_out_sym)

        # get the expected NLL part of the VFE bound
        self.nll_term = nll.mean()
        self.nll_term.name = "nll_term"

        # get KL(q || p) and KL(p || q)
        self.kld_q2p_term = kl_q2p.sum(axis=0).mean()
        self.kld_q2p_term.name = "kld_q2p_term"
        self.kld_p2q_term = kl_p2q.sum(axis=0).mean()
        self.kld_p2q_term.name = "kld_p2q_term"

        # get the proper VFE bound on NLL
        self.nll_bound = self.nll_term + self.kld_q2p_term
        self.nll_bound.name = "nll_bound"

        # grab handles for all the optimizable parameters in our cost
        self.cg = ComputationGraph([self.nll_bound])
        self.joint_params = VariableFilter(roles=[PARAMETER])(self.cg.variables)

        # apply some l2 regularization to the model parameters
        self.reg_term = (1e-5 * sum([tensor.sum(p**2.0) for p in self.joint_params]))
        self.reg_term.name = "reg_term"

        # compute the full cost w.r.t. which we will optimize
        self.joint_cost = self.nll_term + (0.9 * self.kld_q2p_term) + \
                          (0.1 * self.kld_p2q_term) + self.reg_term
        self.joint_cost.name = "joint_cost"

        # Get the gradient of the joint cost for all optimizable parameters
        print("Computing gradients of joint_cost...")
        self.joint_grads = OrderedDict()
        grad_list = tensor.grad(self.joint_cost, self.joint_params)
        for i, p in enumerate(self.joint_params):
            self.joint_grads[p] = grad_list[i]
        
        # shared var learning rate for generator and inferencer
        zero_ary = to_fX( numpy.zeros((1,)) )
        self.lr = theano.shared(value=zero_ary, name='tbm_lr')
        # shared var momentum parameters for generator and inferencer
        self.mom_1 = theano.shared(value=zero_ary, name='tbm_mom_1')
        self.mom_2 = theano.shared(value=zero_ary, name='tbm_mom_2')
        # construct the updates for the generator and inferencer networks
        self.joint_updates = get_adam_updates(params=self.joint_params, \
                grads=self.joint_grads, alpha=self.lr, \
                beta1=self.mom_1, beta2=self.mom_2, \
                mom2_init=1e-4, smoothing=1e-6, max_grad_norm=10.0)

        # collect the outputs to return from this function
        outputs = [self.joint_cost, self.nll_bound, self.nll_term, \
                   self.kld_q2p_term, self.kld_p2q_term, self.reg_term]

        # compile the theano function
        print("Compiling model training/update function...")
        self.train_joint = theano.function(inputs=[x_in_sym, x_out_sym], \
                                outputs=outputs, updates=self.joint_updates)
        print("Compiling NLL bound estimator function...")
        self.compute_nll_bound = theano.function(inputs=[x_in_sym, x_out_sym], \
                                                 outputs=outputs)
        print("Compiling model sampler...")
        n_samples = tensor.iscalar("n_samples")
        samples = self.sample(n_samples)
        self.do_sample = theano.function([n_samples], outputs=samples, \
                                         allow_input_downcast=True)
        return

    def get_model_params(self, ary_type='numpy'):
        """
        Get the optimizable parameters in this model. This returns a list
        and, to reload this model's parameters, the list must stay in order.

        This can provide shared variables or numpy arrays.
        """
        if self.cg is None:
            self.build_model_funcs()
        joint_params = VariableFilter(roles=[PARAMETER])(self.cg.variables)
        if ary_type == 'numpy':
            for i, p in enumerate(joint_params):
                joint_params[i] = p.get_value(borrow=False)
        return joint_params

    def set_model_params(self, numpy_param_list):
        """
        Set the optimizable parameters in this model. This requires a list
        and, to reload this model's parameters, the list must be in order.
        """
        if self.cg is None:
            self.build_model_funcs()
        # grab handles for all the optimizable parameters in our cost
        joint_params = VariableFilter(roles=[PARAMETER])(self.cg.variables)
        for i, p in enumerate(joint_params):
            joint_params[i].set_value(to_fX(numpy_param_list[i]))
        return joint_params

    def save_model_params(self, f_name=None):
        """
        Save model parameters to a pickle file, in numpy form.
        """
        numpy_params = self.get_model_params(ary_type='numpy')
        f_handle = file(f_name, 'wb')
        # dump the dict self.params, which just holds "simple" python values
        cPickle.dump(numpy_params, f_handle, protocol=-1)
        f_handle.close()
        return

    def load_model_params(self, f_name=None):
        """
        Load model parameters from a pickle file, in numpy form.
        """
        pickle_file = open(f_name)
        numpy_params = cPickle.load(pickle_file)
        self.set_model_params(numpy_params)
        pickle_file.close()
        return

##########################################################
# Generalized DRAW model, with infinite mixtures and RL. #
#    -- also modified to operate closed-loopishly        #
##########################################################

class IMoCLDrawModels(BaseRecurrent, Initializable, Random):
    def __init__(self, n_iter, step_type,
                    mix_enc_mlp, mix_dec_mlp, mix_var_mlp,
                    reader_mlp, writer_mlp,
                    enc_mlp_in, enc_rnn, enc_mlp_out,
                    dec_mlp_in, dec_rnn,
                    var_mlp_in, var_rnn, var_mlp_out,
                    **kwargs):
        super(IMoCLDrawModels, self).__init__(**kwargs)
        if not ((step_type == 'add') or (step_type == 'jump')):
            raise ValueError('step_type must be jump or add')
        # record the desired step count
        self.n_iter = n_iter
        self.step_type = step_type
        # grab handles for mixture stuff
        self.mix_enc_mlp = mix_enc_mlp
        self.mix_dec_mlp = mix_dec_mlp
        self.mix_var_mlp = mix_var_mlp
        # grab handles for shared read/write models
        self.reader_mlp = reader_mlp
        self.writer_mlp = writer_mlp
        # grab handles for sequential read/write models
        self.enc_mlp_in = enc_mlp_in
        self.enc_rnn = enc_rnn
        self.enc_mlp_out = enc_mlp_out
        self.dec_mlp_in = dec_mlp_in
        self.dec_rnn = dec_rnn
        self.var_mlp_in = var_mlp_in
        self.var_rnn = var_rnn
        self.var_mlp_out = var_mlp_out
        # setup a "null pointer" that will point to the computation graph
        # for this model, which can be built by self.build_model_funcs()...
        self.cg = None

        # record the sub-models that underlie this model
        self.children = [self.mix_enc_mlp, self.mix_dec_mlp, self.mix_var_mlp,
                         self.reader_mlp, self.writer_mlp,
                         self.enc_mlp_in, self.enc_rnn, self.enc_mlp_out,
                         self.dec_mlp_in, self.dec_rnn,
                         self.var_mlp_in, self.var_rnn, self.var_mlp_out]
        return

    def _allocate(self):
        # allocate shared arrays to hold parameters owned by this model
        c_dim = self.get_dim('c')
        # self.c_0 provides the initial state of the canvas
        self.c_0 = shared_floatx_nans((c_dim,), name='c_0')
        add_role(self.c_0, PARAMETER)
        # add the theano shared variables to our parameter lists
        self.params.extend([ self.c_0 ])
        return

    def _initialize(self):
        # initialize all parameters to zeros...
        for p in self.params:
            p_nan = p.get_value(borrow=False)
            p_zeros = numpy.zeros(p_nan.shape)
            p.set_value(p_zeros.astype(theano.config.floatX))
        return
 
    def get_dim(self, name):
        if name == 'c':
            return self.reader_mlp.get_dim('x_dim')
        elif name == 'z_mix':
            return self.mix_enc_mlp.get_dim('output')
        elif name == 'z_gen':
            return self.enc_mlp_out.get_dim('output')
        elif name == 'h_enc':
            return self.enc_rnn.get_dim('states')
        elif name == 'c_enc':
            return self.enc_rnn.get_dim('cells')
        elif name == 'h_dec':
            return self.dec_rnn.get_dim('states')
        elif name == 'c_dec':
            return self.dec_rnn.get_dim('cells')
        elif name == 'h_var':
            return self.var_rnn.get_dim('states')
        elif name == 'c_var':
            return self.var_rnn.get_dim('cells')
        elif name in ['nll', 'kl_q2p', 'kl_p2q']:
            return 0
        elif name == 'center_y':
            return 0
        elif name == 'center_x':
            return 0
        elif name == 'delta':
            return 0
        else:
            super(IMoCLDrawModels, self).get_dim(name)
        return

    #------------------------------------------------------------------------

    @recurrent(sequences=['u'], contexts=['x', 'm'],
               states=['c', 'h_enc', 'c_enc', 'h_dec', 'c_dec', 'h_var', 'c_var', 'nll', 'kl_q2p', 'kl_p2q'],
               outputs=['c', 'h_enc', 'c_enc', 'h_dec', 'c_dec', 'h_var', 'c_var', 'nll', 'kl_q2p', 'kl_p2q'])
    def iterate(self, u, c, h_enc, c_enc, h_dec, c_dec, h_var, c_var, nll, kl_q2p, kl_p2q, x, m):
        if self.step_type == 'add':
            # additive steps use c as a "direct workspace", which means it's
            # already directly comparable to x.
            c_as_x = tensor.nnet.sigmoid(c)
        else:
            # non-additive steps use c_dec as a "latent workspace", which means
            # it needs to be transformed before being comparable to x.
            c_as_x = tensor.nnet.sigmoid(self.writer_mlp.apply(c_dec))
        # apply a mask for mixing observed and imputed parts of x. c_as_x
        # gives the current reconstruction of x, for all dimensions. m will
        # use 1 to indicate known values, and 0 to indicate values to impute.
        x_m = (m * x) + ((1.0 - m) * c_as_x) # when m==0 everywhere, this will
                                             # contain no information about x.
        # get the feedback available for use by the guide and primary policy
        x_hat_var = x - c_as_x   # provides LL grad w.r.t. c_as_x everywhere
        x_hat_enc = x_m - c_as_x # provides LL grad w.r.t. c_as_x where m==1
        # update the guide RNN state
        r_var = self.reader_mlp.apply(x, x_hat_var, h_dec)
        i_var = self.var_mlp_in.apply(tensor.concatenate([r_var, h_dec], axis=1))
        h_var, c_var = self.var_rnn.apply(states=h_var, cells=c_var,
                                          inputs=i_var, iterate=False)
        # update the encoder RNN state
        r_enc = self.reader_mlp.apply(x_m, x_hat_enc, h_dec)
        i_enc = self.enc_mlp_in.apply(tensor.concatenate([r_enc, h_dec], axis=1))
        h_enc, c_enc = self.enc_rnn.apply(states=h_enc, cells=c_enc,
                                          inputs=i_enc, iterate=False)
        # estimate guide conditional over z given h_var
        q_zg_mean, q_zg_logvar, q_zg = \
                self.var_mlp_out.apply(h_var, u)
        # estimate primary conditional over z given h_enc
        p_zg_mean, p_zg_logvar, p_zg = \
                self.enc_mlp_out.apply(h_enc, u)
        # update the decoder RNN state, using guidance from the guide
        i_dec = self.dec_mlp_in.apply(tensor.concatenate([q_zg], axis=1))
        #i_dec = self.dec_mlp_in.apply(tensor.concatenate([q_zg, h_enc], axis=1))
        h_dec, c_dec = self.dec_rnn.apply(states=h_dec, cells=c_dec, \
                                          inputs=i_dec, iterate=False)
        # update the "workspace" (stored in c)
        if self.step_type == 'add':
            c = c + self.writer_mlp.apply(h_dec)
        else:
            c = self.writer_mlp.apply(c_dec)
        # compute the NLL of the reconstruction as of this step
        c_as_x = tensor.nnet.sigmoid(c)
        m_inv = 1.0 - m
        nll = -1.0 * tensor.flatten(log_prob_bernoulli(x, c_as_x, mask=m_inv))
        # compute KL(q || p) and KL(p || q) for this step
        kl_q2p = tensor.sum(gaussian_kld(q_zg_mean, q_zg_logvar, \
                            p_zg_mean, p_zg_logvar), axis=1)
        kl_p2q = tensor.sum(gaussian_kld(p_zg_mean, p_zg_logvar, \
                            q_zg_mean, q_zg_logvar), axis=1)
        return c, h_enc, c_enc, h_dec, c_dec, h_var, c_var, nll, kl_q2p, kl_p2q

    @recurrent(sequences=['u'], contexts=['x', 'm'],
               states=['c', 'h_enc', 'c_enc', 'h_dec', 'c_dec'],
               outputs=['c', 'h_enc', 'c_enc', 'h_dec', 'c_dec'])
    def decode(self, u, c, h_enc, c_enc, h_dec, c_dec, x, m):
        # get current state of the reconstruction/imputation
        if self.step_type == 'add':
            c_as_x = tensor.nnet.sigmoid(c)
        else:
            c_as_x = tensor.nnet.sigmoid(self.writer_mlp.apply(c_dec))
        x_m = (m * x) + ((1.0 - m) * c_as_x) # mask the known/imputed vals
        x_hat_enc = x_m - c_as_x             # get feedback used by encoder
        # update the encoder RNN state
        r_enc = self.reader_mlp.apply(x_m, x_hat_enc, h_dec)
        i_enc = self.enc_mlp_in.apply(tensor.concatenate([r_enc, h_dec], axis=1))
        h_enc, c_enc = self.enc_rnn.apply(states=h_enc, cells=c_enc,
                                          inputs=i_enc, iterate=False)
        # estimate primary conditional over z given h_enc
        p_zg_mean, p_zg_logvar, p_zg = \
                self.enc_mlp_out.apply(h_enc, u)
        # update the decoder RNN state, using guidance from the guide
        i_dec = self.dec_mlp_in.apply(tensor.concatenate([p_zg], axis=1))
        #i_dec = self.dec_mlp_in.apply(tensor.concatenate([p_zg, h_enc], axis=1))
        h_dec, c_dec = self.dec_rnn.apply(states=h_dec, cells=c_dec, \
                                          inputs=i_dec, iterate=False)
        # update the "workspace" (stored in c)
        if self.step_type == 'add':
            c = c + self.writer_mlp.apply(h_dec)
        else:
            c = self.writer_mlp.apply(c_dec)
        return c, h_enc, c_enc, h_dec, c_dec

    #------------------------------------------------------------------------

    @application(inputs=['x', 'm'], 
                 outputs=['recons', 'nll', 'kl_q2p', 'kl_p2q'])
    def reconstruct(self, x, m):
        # get important size and shape information
        batch_size = x.shape[0]
        z_mix_dim = self.get_dim('z_mix')
        z_gen_dim = self.get_dim('z_gen')
        ce_dim = self.get_dim('c_enc')
        cd_dim = self.get_dim('c_dec')
        cv_dim = self.get_dim('c_var')
        he_dim = self.get_dim('h_enc')
        hd_dim = self.get_dim('h_dec')
        hv_dim = self.get_dim('h_var')

        # get initial state of the reconstruction/imputation
        c0 = tensor.zeros_like(x) + self.c_0
        c_as_x = tensor.nnet.sigmoid(c0)
        x_m = (m * x) + ((1.0 - m) * c_as_x)

        # sample zero-mean, unit std. Gaussian noise for mixture init
        u_mix = self.theano_rng.normal(
                    size=(batch_size, z_mix_dim),
                    avg=0., std=1.)
        # transform ZMUV noise based on q(z_mix | x)
        q_zm_mean, q_zm_logvar, q_zm = \
                self.mix_var_mlp.apply(x, u_mix)   # use full x info
        p_zm_mean, p_zm_logvar, p_zm = \
                self.mix_enc_mlp.apply(x_m, u_mix) # use masked x info
        # transform samples from q(z_mix | x) into initial generator state
        mix_init = self.mix_dec_mlp.apply(q_zm)
        cd0 = mix_init[:, :cd_dim]
        hd0 = mix_init[:, cd_dim:(cd_dim+hd_dim)]
        ce0 = mix_init[:, (cd_dim+hd_dim):(cd_dim+hd_dim+ce_dim)]
        he0 = mix_init[:, (cd_dim+hd_dim+ce_dim):(cd_dim+hd_dim+ce_dim+he_dim)]
        cv0 = mix_init[:, (cd_dim+hd_dim+ce_dim+he_dim):(cd_dim+hd_dim+ce_dim+he_dim+cv_dim)]
        hv0 = mix_init[:, (cd_dim+hd_dim+ce_dim+he_dim+cv_dim):]

        # compute KL-divergence information for the mixture init step
        kl_q2p_mix = tensor.sum(gaussian_kld(q_zm_mean, q_zm_logvar, \
                                p_zm_mean, p_zm_logvar), axis=1)
        kl_p2q_mix = tensor.sum(gaussian_kld(p_zm_mean, p_zm_logvar, \
                                p_zm_mean, p_zm_logvar), axis=1)
        kl_q2p_mix = kl_q2p_mix.reshape((1, batch_size))
        kl_p2q_mix = kl_p2q_mix.reshape((1, batch_size))

        # get zero-mean, unit-std. Gaussian noise for use in scan op
        u_gen = self.theano_rng.normal(
                    size=(self.n_iter, batch_size, z_gen_dim),
                    avg=0., std=1.)

        # run the multi-stage guided generative process
        c, _, _, _, _, _, _, step_nlls, kl_q2p_gen, kl_p2q_gen = \
                self.iterate(u=u_gen, c=c0, \
                             h_enc=he0, c_enc=ce0, \
                             h_dec=hd0, c_dec=cd0, \
                             h_var=hv0, c_var=cv0, \
                             x=x, m=m)

        # grab the observations generated by the multi-stage process
        c_as_x = tensor.nnet.sigmoid(c[-1,:,:])
        recons = (m * x) + ((1.0 - m) * c_as_x)
        recons.name = "recons"
        # get the NLL after the final update for each example
        nll = step_nlls[-1]
        nll.name = "nll"
        # group up the klds from mixture init and multi-stage generation
        kl_q2p = tensor.vertical_stack(kl_q2p_mix, kl_q2p_gen)
        kl_q2p.name = "kl_q2p"
        kl_p2q = tensor.vertical_stack(kl_p2q_mix, kl_p2q_gen)
        kl_p2q.name = "kl_p2q"
        return recons, nll, kl_q2p, kl_p2q

    @application(inputs=['x', 'm'], outputs=['recons'])
    def sample(self, x, m):
        """
        Sample from model. Sampling can be performed either with or
        without partial control (i.e. conditioning for imputation).

        Returns 
        -------

        samples : tensor3 (n_samples, n_iter, x_dim)
        """
        # get important size and shape information
        batch_size = x.shape[0]
        z_mix_dim = self.get_dim('z_mix')
        z_gen_dim = self.get_dim('z_gen')
        ce_dim = self.get_dim('c_enc')
        cd_dim = self.get_dim('c_dec')
        cv_dim = self.get_dim('c_var')
        he_dim = self.get_dim('h_enc')
        hd_dim = self.get_dim('h_dec')
        hv_dim = self.get_dim('h_var')

        # get initial state of the reconstruction/imputation
        c0 = tensor.zeros_like(x) + self.c_0
        c_as_x = tensor.nnet.sigmoid(c0)
        x_m = (m * x) + ((1.0 - m) * c_as_x)

        # sample zero-mean, unit std. Gaussian noise for mixture init
        u_mix = self.theano_rng.normal(
                    size=(batch_size, z_mix_dim),
                    avg=0., std=1.)
        # transform ZMUV noise based on q(z_mix | x)
        p_zm_mean, p_zm_logvar, p_zm = \
                self.mix_enc_mlp.apply(x_m, u_mix) # use masked x info
        # transform samples from q(z_mix | x) into initial generator state
        mix_init = self.mix_dec_mlp.apply(p_zm)
        cd0 = mix_init[:, :cd_dim]
        hd0 = mix_init[:, cd_dim:(cd_dim+hd_dim)]
        ce0 = mix_init[:, (cd_dim+hd_dim):(cd_dim+hd_dim+ce_dim)]
        he0 = mix_init[:, (cd_dim+hd_dim+ce_dim):(cd_dim+hd_dim+ce_dim+he_dim)]
        cv0 = mix_init[:, (cd_dim+hd_dim+ce_dim+he_dim):(cd_dim+hd_dim+ce_dim+he_dim+cv_dim)]
        hv0 = mix_init[:, (cd_dim+hd_dim+ce_dim+he_dim+cv_dim):]

        # get zero-mean, unit-std. Gaussian noise for use in scan op
        u_gen = self.theano_rng.normal(
                    size=(self.n_iter, batch_size, z_gen_dim),
                    avg=0., std=1.)
        # run the sequential generative policy from given initial states
        c, _, _, _, _ = self.decode(u=u_gen, c=c0, h_enc=he0, c_enc=ce0, \
                                    h_dec=hd0, c_dec=cd0, x=x, m=m)
        # convert output into the desired form, and apply masking
        c_as_x = tensor.nnet.sigmoid(c)
        recons = (m * x) + ((1.0 - m) * c_as_x)
        recons.name = "recons"
        return recons

    def build_model_funcs(self):
        """
        Build the symbolic costs and theano functions relevant to this model.
        """
        # some symbolic vars to represent various inputs/outputs
        x_sym = tensor.matrix('x_sym')
        m_sym = tensor.matrix('m_sym')

        # collect reconstructions of x produced by the IMoCLDRAW model
        _, nll, kl_q2p, kl_p2q = self.reconstruct(x_sym, m_sym)

        # get the expected NLL part of the VFE bound
        self.nll_term = nll.mean()
        self.nll_term.name = "nll_term"

        # get KL(q || p) and KL(p || q)
        self.kld_q2p_term = kl_q2p.sum(axis=0).mean()
        self.kld_q2p_term.name = "kld_q2p_term"
        self.kld_p2q_term = kl_p2q.sum(axis=0).mean()
        self.kld_p2q_term.name = "kld_p2q_term"

        # get the proper VFE bound on NLL
        self.nll_bound = self.nll_term + self.kld_q2p_term
        self.nll_bound.name = "nll_bound"

        # grab handles for all the optimizable parameters in our cost
        self.cg = ComputationGraph([self.nll_bound])
        self.joint_params = VariableFilter(roles=[PARAMETER])(self.cg.variables)

        # apply some l2 regularization to the model parameters
        self.reg_term = (1e-5 * sum([tensor.sum(p**2.0) for p in self.joint_params]))
        self.reg_term.name = "reg_term"

        # compute the full cost w.r.t. which we will optimize
        self.joint_cost = self.nll_term + (0.9 * self.kld_q2p_term) + \
                          (0.1 * self.kld_p2q_term) + self.reg_term
        self.joint_cost.name = "joint_cost"

        # Get the gradient of the joint cost for all optimizable parameters
        print("Computing gradients of joint_cost...")
        self.joint_grads = OrderedDict()
        grad_list = tensor.grad(self.joint_cost, self.joint_params)
        for i, p in enumerate(self.joint_params):
            self.joint_grads[p] = grad_list[i]
        
        # shared var learning rate for generator and inferencer
        zero_ary = to_fX( numpy.zeros((1,)) )
        self.lr = theano.shared(value=zero_ary, name='tbm_lr')
        # shared var momentum parameters for generator and inferencer
        self.mom_1 = theano.shared(value=zero_ary, name='tbm_mom_1')
        self.mom_2 = theano.shared(value=zero_ary, name='tbm_mom_2')
        # construct the updates for the generator and inferencer networks
        self.joint_updates = get_adam_updates(params=self.joint_params, \
                grads=self.joint_grads, alpha=self.lr, \
                beta1=self.mom_1, beta2=self.mom_2, \
                mom2_init=1e-4, smoothing=1e-6, max_grad_norm=10.0)

        # collect the outputs to return from this function
        outputs = [self.joint_cost, self.nll_bound, self.nll_term, \
                   self.kld_q2p_term, self.kld_p2q_term, self.reg_term]

        # compile the theano function
        print("Compiling model training/update function...")
        self.train_joint = theano.function(inputs=[x_sym, m_sym], \
                                outputs=outputs, updates=self.joint_updates)
        print("Compiling NLL bound estimator function...")
        self.compute_nll_bound = theano.function(inputs=[x_sym, m_sym], \
                                                 outputs=outputs)
        print("Compiling model sampler...")
        samples = self.sample(x_sym, m_sym)
        self.do_sample = theano.function([x_sym, m_sym], outputs=samples, \
                                         allow_input_downcast=True)
        return

    def get_model_params(self, ary_type='numpy'):
        """
        Get the optimizable parameters in this model. This returns a list
        and, to reload this model's parameters, the list must stay in order.

        This can provide shared variables or numpy arrays.
        """
        if self.cg is None:
            self.build_model_funcs()
        joint_params = VariableFilter(roles=[PARAMETER])(self.cg.variables)
        if ary_type == 'numpy':
            for i, p in enumerate(joint_params):
                joint_params[i] = p.get_value(borrow=False)
        return joint_params

    def set_model_params(self, numpy_param_list):
        """
        Set the optimizable parameters in this model. This requires a list
        and, to reload this model's parameters, the list must be in order.
        """
        if self.cg is None:
            self.build_model_funcs()
        # grab handles for all the optimizable parameters in our cost
        joint_params = VariableFilter(roles=[PARAMETER])(self.cg.variables)
        for i, p in enumerate(joint_params):
            joint_params[i].set_value(to_fX(numpy_param_list[i]))
        return joint_params

    def save_model_params(self, f_name=None):
        """
        Save model parameters to a pickle file, in numpy form.
        """
        numpy_params = self.get_model_params(ary_type='numpy')
        f_handle = file(f_name, 'wb')
        # dump the dict self.params, which just holds "simple" python values
        cPickle.dump(numpy_params, f_handle, protocol=-1)
        f_handle.close()
        return

    def load_model_params(self, f_name=None):
        """
        Load model parameters from a pickle file, in numpy form.
        """
        pickle_file = open(f_name)
        numpy_params = cPickle.load(pickle_file)
        self.set_model_params(numpy_params)
        pickle_file.close()
        return

######################################################################
# Generalized DRAW model, with infinite mixtures and early stopping. #
######################################################################

class IMoESDrawModels(BaseRecurrent, Initializable, Random):
    def __init__(self, n_iter, step_type, mix_enc_mlp, mix_dec_mlp,
                 reader_mlp, enc_mlp_in, enc_rnn, enc_mlp_out, enc_mlp_stop,
                 dec_mlp_in, dec_rnn, dec_mlp_out, dec_mlp_stop, writer_mlp,
                 **kwargs):
        super(IMoESDrawModels, self).__init__(**kwargs)
        if not ((step_type == 'add') or (step_type == 'jump')):
            raise ValueError('step_type must be jump or add')
        # record the desired step count
        self.n_iter = n_iter
        self.step_type = step_type
        # grab handles for mixture stuff
        self.mix_enc_mlp = mix_enc_mlp
        self.mix_dec_mlp = mix_dec_mlp
        # grab handles for IMoESDRAW model stuff
        self.reader_mlp = reader_mlp
        self.writer_mlp = writer_mlp
        self.enc_mlp_in = enc_mlp_in
        self.enc_rnn = enc_rnn
        self.enc_mlp_out = enc_mlp_out
        self.enc_mlp_stop = enc_mlp_stop
        self.dec_mlp_in = dec_mlp_in
        self.dec_rnn = dec_rnn
        self.dec_mlp_out = dec_mlp_out
        self.dec_mlp_stop = dec_mlp_stop

        # record the sub-models that underlie this model
        self.children = [self.mix_enc_mlp, self.mix_dec_mlp,
                         self.reader_mlp, self.writer_mlp,
                         self.enc_mlp_in, self.enc_rnn, 
                         self.enc_mlp_out, self.enc_mlp_stop,
                         self.dec_mlp_in, self.dec_rnn, 
                         self.dec_mlp_out, self.dec_mlp_stop]
        return

    def _allocate(self):
        c_dim = self.get_dim('c')
        zm_dim = self.get_dim('z_mix')
        # self.c_0 provides the initial state of the canvas
        self.c_0 = shared_floatx_nans((c_dim,), name='c_0')
        # self.zm_mean provides the mean of z_mix
        self.zm_mean = shared_floatx_nans((zm_dim,), name='zm_mean')
        # self.zm_logvar provides the logvar of z_mix
        self.zm_logvar = shared_floatx_nans((zm_dim,), name='zm_logvar')
        add_role(self.c_0, PARAMETER)
        add_role(self.zm_mean, PARAMETER)
        add_role(self.zm_logvar, PARAMETER)
        # add the theano shared variables to our parameter lists
        self.params.extend([ self.c_0, self.zm_mean, self.zm_logvar ])
        return

    def _initialize(self):
        # initialize to all parameters zeros...
        for p in self.params:
            p_nan = p.get_value(borrow=False)
            p_zeros = numpy.zeros(p_nan.shape)
            p.set_value(p_zeros.astype(theano.config.floatX))
        return
 
    def get_dim(self, name):
        if name == 'c':
            return self.reader_mlp.get_dim('x_dim')
        elif name == 'z_mix':
            return self.mix_enc_mlp.get_dim('output')
        elif name == 'h_enc':
            return self.enc_rnn.get_dim('states')
        elif name == 'c_enc':
            return self.enc_rnn.get_dim('cells')
        elif name == 'z_gen':
            return self.enc_mlp_out.get_dim('output')
        elif name == 'h_dec':
            return self.dec_rnn.get_dim('states')
        elif name == 'c_dec':
            return self.dec_rnn.get_dim('cells')
        elif name in ['esp_enc', 'esp_all']:
            return 0
        elif name in ['nll', 'nll_all']:
            return 0
        elif name in ['kl_esp', 'kl_z', 'kl_esp_all', 'kl_z_all']:
            return 0
        elif name in ['cost', 'cost_all']:
            return 0
        elif name == 'center_y':
            return 0
        elif name == 'center_x':
            return 0
        elif name == 'delta':
            return 0
        else:
            super(IMoESDrawModels, self).get_dim(name)
        return

    #------------------------------------------------------------------------

    @recurrent(sequences=['u'], contexts=['x'],
               states=['c', 'h_enc', 'c_enc', 'h_dec', 'c_dec', 'esp_enc', 'nll', 'kl_esp', 'kl_z'],
               outputs=['c', 'h_enc', 'c_enc', 'h_dec', 'c_dec', 'esp_enc', 'nll', 'kl_esp', 'kl_z'])
    def iterate(self, u, c, h_enc, c_enc, h_dec, c_dec, esp_enc, nll, kl_esp, kl_z, x):
        if self.step_type == 'add':
            # additive steps use c as a "direct workspace", which means it's
            # already directly comparable to x.
            c = c
        else:
            # non-additive steps use c_dec as a "latent workspace", which means
            # it needs to be transformed before being comparable to x.
            c = self.writer_mlp.apply(c_dec)
        # get the NLL gradient
        nll_grad = x - tensor.nnet.sigmoid(c)
        # update the encoder RNN state
        r = self.reader_mlp.apply(x, nll_grad, h_dec)
        i_enc = self.enc_mlp_in.apply(tensor.concatenate([r, h_dec], axis=1))
        h_enc, c_enc = self.enc_rnn.apply(states=h_enc, cells=c_enc,
                                          inputs=i_enc, iterate=False)
        # estimate encoder conditional over z given h_enc
        q_gen_mean, q_gen_logvar, q_z_gen = \
                self.enc_mlp_out.apply(h_enc, u)
        # estimate decoder conditional over z given h_dec
        p_gen_mean, p_gen_logvar, p_z_gen = \
                self.dec_mlp_out.apply(h_dec, u)
        # update the decoder RNN state
        z_gen = q_z_gen # use samples from q while training
        i_dec = self.dec_mlp_in.apply(tensor.concatenate([z_gen], axis=1))
        h_dec, c_dec = self.dec_rnn.apply(states=h_dec, cells=c_dec, \
                                          inputs=i_dec, iterate=False)
        # update the workspace base on the updated decoder state
        if self.step_type == 'add':
            # additive steps use c as a "direct workspace", which means it's
            # already directly comparable to x.
            c = c + self.writer_mlp.apply(h_dec)
        else:
            # non-additive steps use c_dec as a "latent workspace", which means
            # it needs to be transformed before being comparable to x.
            c = self.writer_mlp.apply(c_dec)
        # compute the encoder and decoder probabilities of stopping after
        # the update which we just performed.
        c_as_x = tensor.nnet.sigmoid(c)
        nll_grad = x - c_as_x
        esp_enc = self.enc_mlp_stop.apply( \
                tensor.concatenate([h_dec, nll_grad], axis=1)).flatten()
        esp_dec = self.dec_mlp_stop.apply( \
                tensor.concatenate([h_dec], axis=1)).flatten()
        esp_enc = tensor.nnet.sigmoid(esp_enc - 2.0)
        esp_dec = tensor.nnet.sigmoid(esp_dec - 2.0)
        # compute nll for this step (post update)
        nll = -1.0 * tensor.flatten(log_prob_bernoulli(x, c_as_x))
        # compute KLd between encoder and decoder stopping probabilities
        kl_esp = bernoulli_kld(esp_enc, esp_dec)
        # compute KLd between the encoder and decoder distributions over z
        kl_z = tensor.sum(gaussian_kld(q_gen_mean, q_gen_logvar, \
                                       p_gen_mean, p_gen_logvar), axis=1)
        return c, h_enc, c_enc, h_dec, c_dec, esp_enc, nll, kl_esp, kl_z

    @recurrent(sequences=['u'], contexts=[],
               states=['c', 'h_dec', 'c_dec'],
               outputs=['c', 'h_dec', 'c_dec'])
    def decode(self, u, c, h_dec, c_dec):
        batch_size = c.shape[0]
        # sample z from p(z | h_dec) -- we used q(z | h_enc) during training
        p_gen_mean, p_gen_logvar, p_z_gen = \
                self.dec_mlp_out.apply(h_dec, u)
        z_gen = p_z_gen
        # update the decoder RNN state
        i_dec = self.dec_mlp_in.apply(tensor.concatenate([z_gen], axis=1))
        h_dec, c_dec = self.dec_rnn.apply(
                    states=h_dec, cells=c_dec,
                    inputs=i_dec, iterate=False)
        # additive steps use c as the "workspace"
        if self.step_type == 'add':
            c = c + self.writer_mlp.apply(h_dec)
        else:
            c = self.writer_mlp.apply(c_dec)
        return c, h_dec, c_dec

    #------------------------------------------------------------------------

    @application(inputs=['x_in', 'x_out'], 
                 outputs=['cost', 'cost_all'])
    def reconstruct(self, x_in, x_out):
        # get important size and shape information
        batch_size = x_in.shape[0]
        z_mix_dim = self.get_dim('z_mix')
        z_gen_dim = self.get_dim('z_gen')
        ce_dim = self.get_dim('c_enc')
        cd_dim = self.get_dim('c_dec')
        he_dim = self.get_dim('h_enc')
        hd_dim = self.get_dim('h_dec')

        # sample zero-mean, unit std. Gaussian noise for mixture init
        u_mix = self.theano_rng.normal(
                    size=(batch_size, z_mix_dim),
                    avg=0., std=1.)
        # transform ZMUV noise based on q(z_mix | x_in)
        z_mix_mean, z_mix_logvar, z_mix = \
                self.mix_enc_mlp.apply(x_in, u_mix)
        # transform samples from q(z_mix | x_in) into initial generator state
        mix_init = self.mix_dec_mlp.apply(z_mix)
        cd0 = mix_init[:, :cd_dim]
        hd0 = mix_init[:, cd_dim:(cd_dim+hd_dim)]
        ce0 = mix_init[:, (cd_dim+hd_dim):(cd_dim+hd_dim+ce_dim)]
        he0 = mix_init[:, (cd_dim+hd_dim+ce_dim):]
        c0 = tensor.zeros_like(x_out) + self.c_0

        # get the initial reconstruction after mixture initialization
        if self.step_type == 'add':
            c_as_x = tensor.nnet.sigmoid(c0)
        else:
            c_as_x = tensor.nnet.sigmoid(self.writer_mlp.apply(cd0))
        nll_grad = x_out - c_as_x
        # compute the encoder and decoder probabilities of stopping after
        # performing only the mixture initialization step
        esp_mix_enc = self.enc_mlp_stop.apply( \
                tensor.concatenate([hd0, nll_grad], axis=1)).flatten()
        esp_mix_dec = self.dec_mlp_stop.apply( \
                tensor.concatenate([hd0], axis=1)).flatten()
        esp_mix_enc = tensor.nnet.sigmoid(esp_mix_enc - 2.0)
        esp_mix_dec = tensor.nnet.sigmoid(esp_mix_dec - 2.0)
        # compute nll for this step (post update)
        nll_mix = -1.0 * tensor.flatten(log_prob_bernoulli(x_out, c_as_x))
        # compute KLd for the encoder/decoder early stopping probabilities
        kl_esp_mix = bernoulli_kld(esp_mix_enc, esp_mix_dec)
        # compute KLd for the encoder/decoder distributions over z
        kl_z_mix = tensor.sum(gaussian_kld(z_mix_mean, z_mix_logvar, \
                                       self.zm_mean, self.zm_logvar), axis=1)
        # reshape for easy stacking with the outputs of scan op
        esp_mix = esp_mix_enc.reshape((1, batch_size))
        nll_mix = nll_mix.reshape((1, batch_size))
        kl_esp_mix = kl_esp_mix.reshape((1, batch_size))
        kl_z_mix = kl_z_mix.reshape((1, batch_size))

        # get zero-mean, unit-std. Gaussian noise for use in scan op
        u_gen = self.theano_rng.normal(
                    size=(self.n_iter, batch_size, z_gen_dim),
                    avg=0., std=1.)

        # run the multi-stage guided generative process using scan
        c, _, _, _, _, esp_scan, nll_scan, kl_esp_scan, kl_z_scan  = \
                self.iterate(u=u_gen, c=c0, h_enc=he0, c_enc=ce0, \
                             h_dec=hd0, c_dec=cd0, x=x_out)

        # stack up esps, nlls, and kls to get full per-step cost components
        esp_all = tensor.vertical_stack(esp_mix, esp_scan)
        nll_all = tensor.vertical_stack(nll_mix, nll_scan)
        kl_esp_all = tensor.vertical_stack(kl_esp_mix, kl_esp_scan)
        kl_z_all = tensor.vertical_stack(kl_z_mix, kl_z_scan)
        nesp_all = 1.0 - esp_all # step-wise probabilities of continuation

        ###############################################
        # compute the VFE with early stopping allowed #
        ###############################################

        # get cumulative product of probabilities of continuation
        nesp_prods = tensor.extra_ops.cumprod(nesp_all, axis=0)
        # get cumulative sums of KLd, for total KLd up to a given step
        kl_sums = tensor.extra_ops.cumsum(kl_esp_all, axis=0) + \
                  tensor.extra_ops.cumsum(kl_z_all, axis=0)
        # do stacking extensions to allow loop-free cost computation
        ones_row = tensor.alloc(1.0, 1, batch_size)
        nesp_stack = tensor.vertical_stack(ones_row, nesp_prods)
        esp_stack = tensor.vertical_stack(esp_all, ones_row)
        shitty_shim_row_kl = kl_sums[-1].reshape((1, batch_size))
        kl_stack = tensor.vertical_stack(kl_sums, shitty_shim_row_kl)
        shitty_shim_row_nll = nll_all[-1].reshape((1, batch_size))
        nll_stack = tensor.vertical_stack(nll_all, shitty_shim_row_nll)
        # compute step-wise costs per training example
        cost_all = (esp_stack * nesp_stack) * (nll_stack + kl_stack)
        # to compute the final cost: sum over steps and average over trials
        cost = tensor.sum(cost_all, axis=0).mean()
        return cost, cost_all

    @application(inputs=['n_samples'], outputs=['samples'])
    def sample(self, n_samples):
        """Sample from model.

        Returns 
        -------

        samples : tensor3 (n_samples, n_iter, x_dim)
        """
        z_mix_dim = self.get_dim('z_mix')
        z_gen_dim = self.get_dim('z_gen')
        cd_dim = self.get_dim('c_dec')
        hd_dim = self.get_dim('h_dec')
        ce_dim = self.get_dim('c_enc')
        he_dim = self.get_dim('h_enc')
        c_dim = self.get_dim('c')

        # sample zero-mean, unit-std. Gaussian noise for the mixture init
        u_mix = self.theano_rng.normal(
                    size=(n_samples, z_mix_dim),
                    avg=0., std=1.)
        # transform noise based on learned mean and logvar
        z_mix = self.zm_mean + (u_mix * tensor.exp(0.5 * self.zm_logvar))
        # transform the sample from p(z_mix) into an initial generator state
        mix_init = self.mix_dec_mlp.apply(z_mix)
        cd0 = mix_init[:, :cd_dim]
        hd0 = mix_init[:, cd_dim:(cd_dim+hd_dim)]
        c0 = tensor.alloc(0.0, n_samples, c_dim) + self.c_0

        # sample from zero-mean unit-std. Gaussian for use in scan op
        u_gen = self.theano_rng.normal(
                    size=(self.n_iter, n_samples, z_gen_dim),
                    avg=0., std=1.)

        c, _, _, = self.decode(u=u_gen, c=c0, h_dec=hd0, c_dec=cd0)
        #c, _, _, center_y, center_x, delta = self.decode(u)
        return tensor.nnet.sigmoid(c)

##############################
# Dot-matrix image generator #
##############################

class DotMatrix(BaseRecurrent, Initializable, Random):
    """Infinite mixture of LSTM generators, like variationally.

    We 'encode' to Gaussian posteriors in some latent space using a CondNet
    and then 'decode' using an LSTM that generates binary images column-wise
    from left-to-right. More or less.
    """
    def __init__(self, 
                 enc_x_to_z,
                 enc_z_to_mix,
                 dec_rnn,
                 dec_mlp_in,
                 dec_mlp_out,
                 im_shape,
                 mix_dim,
                 **kwargs):
        super(DotMatrix, self).__init__(**kwargs)
        # grab handles for underlying models
        self.enc_x_to_z = enc_x_to_z     # go from x to z
        self.enc_z_to_mix = enc_z_to_mix # go from z to mixture params
        self.dec_rnn = dec_rnn           # LSTM for statefulness
        self.dec_mlp_in = dec_mlp_in     # provide i_dec
        self.dec_mlp_out = dec_mlp_out   # predict from h_dec
        self.im_shape = im_shape
        self.mix_dim = mix_dim

        # record the sub-models that underlie this model
        self.children = [self.enc_x_to_z, self.enc_z_to_mix, \
                         self.dec_rnn, self.dec_mlp_in, self.dec_mlp_out]
        return
 
    def get_dim(self, name):
        if name in ['x_obs', 'x_hat']:
            return self.im_shape[0]
        elif name == 'h_dec':
            return self.dec_rnn.get_dim('states')
        elif name == 'c_dec':
            return self.dec_rnn.get_dim('cells')
        elif name == 's_mix':
            return self.mix_dim
        elif name == 'z_mix':
            return self.enc_x_to_z.get_dim('output')
        else:
            super(DotMatrix, self).get_dim(name)
        return

    #------------------------------------------------------------------------

    @recurrent(sequences=['x_obs'], contexts=[],
               states=['x_hat', 'h_dec', 'c_dec'],
               outputs=['x_hat', 'h_dec', 'c_dec'])
    def iterate(self, x_obs, x_hat, h_dec, c_dec):
        # compute predictions for this time step
        x_log = self.dec_mlp_out.apply(h_dec)
        x_hat = tensor.nnet.sigmoid(x_log)
        # update rnn state using current observation and previous state
        i_mlp = tensor.concatenate([x_obs, h_dec], axis=1)
        i_dec = self.dec_mlp_in.apply(i_mlp)
        h_dec, c_dec = self.dec_rnn.apply(
                    states=h_dec, cells=c_dec,
                    inputs=i_dec, iterate=False)
        return x_hat, h_dec, c_dec

    @recurrent(sequences=['u'], contexts=[], 
               states=['x_hat', 'h_dec', 'c_dec'],
               outputs=['x_hat', 'h_dec', 'c_dec'])
    def decode(self, u, x_hat, h_dec, c_dec):
        # update the rnn state using previous state information
        i_mlp = tensor.concatenate([x_hat, h_dec], axis=1)
        i_dec = self.dec_mlp_in.apply(i_mlp)
        h_dec, c_dec = self.dec_rnn.apply(
                    states=h_dec, cells=c_dec,
                    inputs=i_dec, iterate=False)
        # get predicted probablities at this time step
        x_log = self.dec_mlp_out.apply(h_dec)
        x_prob = tensor.nnet.sigmoid(x_log)
        # sample binary pixels using the uniform random values in u
        x_hat = u < x_prob
        x_hat = tensor.cast(x_hat, 'floatX')
        return x_hat, h_dec, c_dec

    #------------------------------------------------------------------------

    @application(inputs=['x_in', 'x_out'], outputs=['x_hat', 'kl_mix'])
    def reconstruct(self, x_in, x_out):
        batch_size = x_in.shape[0]
        z_mix_dim = self.get_dim('z_mix')
        cd_dim = self.get_dim('c_dec')
        hd_dim = self.get_dim('h_dec')
        im_rows = self.im_shape[0]
        im_cols = self.im_shape[1]

        # get noise to transform for q(z | x)
        u_mix = self.theano_rng.normal(
                    size=(batch_size, z_mix_dim),
                    avg=0., std=1.)

        # compute conditional over z given x and sample from it
        z_mix_mean, z_mix_logvar, z_mix = \
                self.enc_x_to_z.apply(x_in, u_mix)
        akl_mix = gaussian_kld(z_mix_mean, z_mix_logvar, 0.0, 0.0)
        kl_mix = tensor.sum(akl_mix, axis=1)
        kl_mix.name = "kl_mix"

        # transform samples from q(z|x) into some seed state info
        mix_init = self.enc_z_to_mix.apply(z_mix)
        cd0 = mix_init[:, :cd_dim]
        hd0 = mix_init[:, cd_dim:]

        # reshape target outputs for scanning over columns
        x_out = x_out.reshape((batch_size, im_rows, im_cols), ndim=3)
        x_obs = x_out.dimshuffle(2, 0, 1)

        # scan over pixels column-wise for prediction log-likelihood
        x_hat, h_dec, c_dec = self.iterate( \
                x_obs=x_obs, h_dec=hd0, c_dec=cd0)

        # grab the predicted pixel probabilities in flattened form
        x_hat = x_hat.dimshuffle(1, 2, 0).reshape((batch_size, (im_rows*im_cols)))
        x_hat.name = "x_hat"
        return x_hat, kl_mix

    @application(inputs=['n_samples'], outputs=['samples'])
    def sample(self, n_samples):
        """Sample from model.

        Returns 
        -------

        samples : tensor3 (n_samples, obs_dim)
        """
        z_mix_dim = self.get_dim('z_mix')
        cd_dim = self.get_dim('c_dec')
        hd_dim = self.get_dim('h_dec')
        ce_dim = self.get_dim('c_enc')
        he_dim = self.get_dim('h_enc')
        im_rows = self.im_shape[0]
        im_cols = self.im_shape[1]

        # get noise to from the anchor for q(z | x)
        z_mix = self.theano_rng.normal(
                    size=(n_samples, z_mix_dim),
                    avg=0., std=1.)

        # transform samples from q(z|x) into some seed state info
        mix_init = self.enc_z_to_mix.apply(z_mix)
        cd0 = mix_init[:, :cd_dim]
        hd0 = mix_init[:, cd_dim:]

        # generate some uniform random values to use for pixel sampling
        u = self.theano_rng.uniform(
                    size=(im_cols, n_samples, im_rows))

        samples, _, _, = self.decode(u=u, h_dec=hd0, c_dec=cd0)
        samples = samples.dimshuffle(1, 2, 0)
        samples = samples.reshape((n_samples, (im_rows*im_cols)))
        return samples














##########################################################
# This model generates observations by running a drift   #
# diffusion process over a latent space, where the state #
# of the process is maintained by an LSTM.               #
#                                                        #
# This model expects a model for transforming from LSTM  #
# state to the observation space that has an explicit    #
# method for simultaneous forward and back propagation.  #
##########################################################

class DriftDiffModel(BaseRecurrent, Initializable, Random):
    def __init__(self, n_iter, mix_enc_mlp, mix_dec_mlp,
                    enc_mlp_in, enc_rnn, enc_mlp_out,
                    dec_mlp_in, dec_rnn, dec_mlp_out,
                    s2x_mlp,
                    **kwargs):
        super(DriftDiffModel, self).__init__(**kwargs)
        # record the desired step count
        self.n_iter = n_iter
        # grab handles for mixture stuff
        self.mix_enc_mlp = mix_enc_mlp
        self.mix_dec_mlp = mix_dec_mlp
        # grab handles for sequential generation stuff
        self.enc_mlp_in = enc_mlp_in
        self.enc_rnn = enc_rnn
        self.enc_mlp_out = enc_mlp_out
        self.dec_mlp_in = dec_mlp_in
        self.dec_rnn = dec_rnn
        self.dec_mlp_out = dec_mlp_out
        self.s2x_mlp = s2x_mlp

        # record the sub-models that underlie this model
        self.children = [self.mix_enc_mlp, self.mix_dec_mlp,
                         self.enc_mlp_in, self.enc_rnn, self.enc_mlp_out,
                         self.dec_mlp_in, self.dec_rnn, self.dec_mlp_out,
                         self.s2x_mlp]
        return

    def _allocate(self):
        c_dim = self.get_dim('c')
        zm_dim = self.get_dim('z_mix')
        # self.c_0 provides the initial state of the canvas
        self.c_0 = shared_floatx_nans((c_dim,), name='c_0')
        # self.zm_mean provides the mean of z_mix
        self.zm_mean = shared_floatx_nans((zm_dim,), name='zm_mean')
        # self.zm_logvar provides the logvar of z_mix
        self.zm_logvar = shared_floatx_nans((zm_dim,), name='zm_logvar')
        add_role(self.c_0, PARAMETER)
        add_role(self.zm_mean, PARAMETER)
        add_role(self.zm_logvar, PARAMETER)
        # add the theano shared variables to our parameter lists
        self.params.extend([ self.c_0, self.zm_mean, self.zm_logvar ])
        return

    def _initialize(self):
        # initialize to all parameters zeros...
        for p in self.params:
            p_nan = p.get_value(borrow=False)
            p_zeros = numpy.zeros(p_nan.shape)
            p.set_value(p_zeros.astype(theano.config.floatX))
        return
 
    def get_dim(self, name):
        if name == 'c':
            return self.s2x_mlp.get_dim('output')
        elif name == 'z_mix':
            return self.mix_enc_mlp.get_dim('output')
        elif name == 'h_enc':
            return self.enc_rnn.get_dim('states')
        elif name == 'c_enc':
            return self.enc_rnn.get_dim('cells')
        elif name == 'z_gen':
            return self.enc_mlp_out.get_dim('output')
        elif name == 'h_dec':
            return self.dec_rnn.get_dim('states')
        elif name == 'c_dec':
            return self.dec_rnn.get_dim('cells')
        elif name in ['nll', 'kl_q2p', 'kl_p2q']:
            return 0
        else:
            super(DriftDiffModel, self).get_dim(name)
        return

    #------------------------------------------------------------------------

    @recurrent(sequences=['u'], contexts=['x'],
               states=['c', 'h_enc', 'c_enc', 'h_dec', 'c_dec', 'nll', 'kl_q2p', 'kl_p2q'],
               outputs=['c', 'h_enc', 'c_enc', 'h_dec', 'c_dec', 'nll', 'kl_q2p', 'kl_p2q'])
    def iterate(self, u, c, h_enc, c_enc, h_dec, c_dec, nll, kl_q2p, kl_p2q, x):
        # get the NLL grad w.r.t. to the encoder LSTM space
        _, grad_wrt_dec = self.s2x_mlp.apply_ffbp(c_dec, x)
        # update the encoder RNN state
        i_enc = self.enc_mlp_in.apply(tensor.concatenate([x, grad_wrt_dec, h_dec], axis=1))
        h_enc, c_enc = self.enc_rnn.apply(states=h_enc, cells=c_enc,
                                          inputs=i_enc, iterate=False)
        # estimate encoder conditional over z given h_enc
        q_gen_mean, q_gen_logvar, q_z_gen = \
                self.enc_mlp_out.apply(h_enc, u)
        # estimate decoder conditional over z given h_dec
        p_gen_mean, p_gen_logvar, p_z_gen = \
                self.dec_mlp_out.apply(h_dec, u)
        # update the decoder RNN state
        z_gen = q_z_gen # use samples from q while training
        i_dec = self.dec_mlp_in.apply(tensor.concatenate([z_gen], axis=1))
        h_dec, c_dec = self.dec_rnn.apply(states=h_dec, cells=c_dec, \
                                          inputs=i_dec, iterate=False)
        # compute the NLL of the reconstruction as of this step
        c = self.s2x_mlp.apply(c_dec)
        c_as_x = tensor.nnet.sigmoid(c)
        nll = -1.0 * tensor.flatten(log_prob_bernoulli(x, c_as_x))
        # compute KL(q || p) and KL(p || q) for this step
        kl_q2p = tensor.sum(gaussian_kld(q_gen_mean, q_gen_logvar, \
                            p_gen_mean, p_gen_logvar), axis=1)
        kl_p2q = tensor.sum(gaussian_kld(p_gen_mean, p_gen_logvar, \
                            q_gen_mean, q_gen_logvar), axis=1)
        return c, h_enc, c_enc, h_dec, c_dec, nll, kl_q2p, kl_p2q

    @recurrent(sequences=['u'], contexts=[],
               states=['c', 'h_dec', 'c_dec'],
               outputs=['c', 'h_dec', 'c_dec'])
    def decode(self, u, c, h_dec, c_dec):
        batch_size = c.shape[0]
        # sample z from p(z | h_dec) -- we used q(z | h_enc) during training
        p_gen_mean, p_gen_logvar, p_z_gen = \
                self.dec_mlp_out.apply(h_dec, u)
        z_gen = p_z_gen
        # update the decoder RNN state
        i_dec = self.dec_mlp_in.apply(tensor.concatenate([z_gen], axis=1))
        h_dec, c_dec = self.dec_rnn.apply(
                    states=h_dec, cells=c_dec,
                    inputs=i_dec, iterate=False)
        # compute transformed output
        c = self.s2x_mlp.apply(c_dec)
        return c, h_dec, c_dec

    #------------------------------------------------------------------------

    @application(inputs=['x_in', 'x_out'], 
                 outputs=['recons', 'nll', 'kl_q2p', 'kl_p2q'])
    def reconstruct(self, x_in, x_out):
        # get important size and shape information
        batch_size = x_in.shape[0]
        z_mix_dim = self.get_dim('z_mix')
        z_gen_dim = self.get_dim('z_gen')
        ce_dim = self.get_dim('c_enc')
        cd_dim = self.get_dim('c_dec')
        he_dim = self.get_dim('h_enc')
        hd_dim = self.get_dim('h_dec')

        # sample zero-mean, unit std. Gaussian noise for mixture init
        u_mix = self.theano_rng.normal(
                    size=(batch_size, z_mix_dim),
                    avg=0., std=1.)
        # transform ZMUV noise based on q(z_mix | x_in)
        z_mix_mean, z_mix_logvar, z_mix = \
                self.mix_enc_mlp.apply(x_in, u_mix)
        # transform samples from q(z_mix | x_in) into initial generator state
        mix_init = self.mix_dec_mlp.apply(z_mix)
        cd0 = mix_init[:, :cd_dim]
        hd0 = mix_init[:, cd_dim:(cd_dim+hd_dim)]
        ce0 = mix_init[:, (cd_dim+hd_dim):(cd_dim+hd_dim+ce_dim)]
        he0 = mix_init[:, (cd_dim+hd_dim+ce_dim):]
        c0 = self.s2x_mlp.apply(cd0)

        # compute KL-divergence information for the mixture init step
        kl_q2p_mix = tensor.sum(gaussian_kld(z_mix_mean, z_mix_logvar, \
                                self.zm_mean, self.zm_logvar), axis=1)
        kl_p2q_mix = tensor.sum(gaussian_kld(self.zm_mean, self.zm_logvar, \
                                z_mix_mean, z_mix_logvar), axis=1)
        kl_q2p_mix = kl_q2p_mix.reshape((1, batch_size))
        kl_p2q_mix = kl_p2q_mix.reshape((1, batch_size))

        # get zero-mean, unit-std. Gaussian noise for use in scan op
        u_gen = self.theano_rng.normal(
                    size=(self.n_iter, batch_size, z_gen_dim),
                    avg=0., std=1.)

        # run the multi-stage guided generative process
        c, _, _, _, _, step_nlls, kl_q2p_gen, kl_p2q_gen = \
                self.iterate(u=u_gen, c=c0, h_enc=he0, c_enc=ce0, \
                             h_dec=hd0, c_dec=cd0, x=x_out)

        # grab the observations generated by the multi-stage process
        recons = tensor.nnet.sigmoid(c[-1,:,:])
        recons.name = "recons"
        # get the NLL after the final update for each example
        nll = step_nlls[-1]
        nll.name = "nll"
        # group up the klds from mixture init and multi-stage generation
        kl_q2p = tensor.vertical_stack(kl_q2p_mix, kl_q2p_gen)
        kl_q2p.name = "kl_q2p"
        kl_p2q = tensor.vertical_stack(kl_p2q_mix, kl_p2q_gen)
        kl_p2q.name = "kl_p2q"
        return recons, nll, kl_q2p, kl_p2q

    @application(inputs=['n_samples'], outputs=['samples'])
    def sample(self, n_samples):
        """Sample from model.

        Returns 
        -------

        samples : tensor3 (n_samples, n_iter, x_dim)
        """
        z_mix_dim = self.get_dim('z_mix')
        z_gen_dim = self.get_dim('z_gen')
        cd_dim = self.get_dim('c_dec')
        hd_dim = self.get_dim('h_dec')
        ce_dim = self.get_dim('c_enc')
        he_dim = self.get_dim('h_enc')
        c_dim = self.get_dim('c')

        # sample zero-mean, unit-std. Gaussian noise for the mixture init
        u_mix = self.theano_rng.normal(
                    size=(n_samples, z_mix_dim),
                    avg=0., std=1.)
        # transform noise based on learned mean and logvar
        z_mix = self.zm_mean + (u_mix * tensor.exp(0.5 * self.zm_logvar))
        # transform the sample from p(z_mix) into an initial generator state
        mix_init = self.mix_dec_mlp.apply(z_mix)
        cd0 = mix_init[:, :cd_dim]
        hd0 = mix_init[:, cd_dim:(cd_dim+hd_dim)]
        c0 = self.s2x_mlp.apply(cd0)

        # sample from zero-mean unit-std. Gaussian for use in scan op
        u_gen = self.theano_rng.normal(
                    size=(self.n_iter, n_samples, z_gen_dim),
                    avg=0., std=1.)
        # run the sequential generative process...
        c, _, _, = self.decode(u=u_gen, c=c0, h_dec=hd0, c_dec=cd0)
        return tensor.nnet.sigmoid(c)

    def build_model_funcs(self):
        """
        Build the symbolic costs and theano functions relevant to this model.
        """
        # some symbolic vars to represent various inputs/outputs
        x_in_sym = tensor.matrix('x_in_sym')
        x_out_sym = tensor.matrix('x_out_sym')

        # collect reconstructions of x produced by the IMoOLDRAW model
        _, nll, kl_q2p, kl_p2q = self.reconstruct(x_in_sym, x_out_sym)

        # get the expected NLL part of the VFE bound
        self.nll_term = nll.mean()
        self.nll_term.name = "nll_term"

        # get KL(q || p) and KL(p || q)
        self.kld_q2p_term = kl_q2p.sum(axis=0).mean()
        self.kld_q2p_term.name = "kld_q2p_term"
        self.kld_p2q_term = kl_p2q.sum(axis=0).mean()
        self.kld_p2q_term.name = "kld_p2q_term"

        # get the proper VFE bound on NLL
        self.nll_bound = self.nll_term + self.kld_q2p_term
        self.nll_bound.name = "nll_bound"

        # grab handles for all the optimizable parameters in our cost
        self.cg = ComputationGraph([self.nll_bound])
        self.joint_params = VariableFilter(roles=[PARAMETER])(self.cg.variables)

        # apply some l2 regularization to the model parameters
        self.reg_term = (1e-5 * sum([tensor.sum(p**2.0) for p in self.joint_params]))
        self.reg_term.name = "reg_term"

        # compute the full cost w.r.t. which we will optimize
        self.joint_cost = self.nll_term + (0.9 * self.kld_q2p_term) + \
                          (0.1 * self.kld_p2q_term) + self.reg_term
        self.joint_cost.name = "joint_cost"

        # Get the gradient of the joint cost for all optimizable parameters
        print("Computing gradients of joint_cost...")
        self.joint_grads = OrderedDict()
        grad_list = tensor.grad(self.joint_cost, self.joint_params)
        for i, p in enumerate(self.joint_params):
            self.joint_grads[p] = grad_list[i]
        
        # shared var learning rate for generator and inferencer
        zero_ary = to_fX( numpy.zeros((1,)) )
        self.lr = theano.shared(value=zero_ary, name='tbm_lr')
        # shared var momentum parameters for generator and inferencer
        self.mom_1 = theano.shared(value=zero_ary, name='tbm_mom_1')
        self.mom_2 = theano.shared(value=zero_ary, name='tbm_mom_2')
        # construct the updates for the generator and inferencer networks
        self.joint_updates = get_adam_updates(params=self.joint_params, \
                grads=self.joint_grads, alpha=self.lr, \
                beta1=self.mom_1, beta2=self.mom_2, \
                mom2_init=1e-4, smoothing=1e-6, max_grad_norm=10.0)

        # collect the outputs to return from this function
        outputs = [self.joint_cost, self.nll_bound, self.nll_term, \
                   self.kld_q2p_term, self.kld_p2q_term, self.reg_term]

        # compile the theano function
        print("Compiling model training/update function...")
        self.train_joint = theano.function(inputs=[x_in_sym, x_out_sym], \
                                outputs=outputs, updates=self.joint_updates)
        print("Compiling NLL bound estimator function...")
        self.compute_nll_bound = theano.function(inputs=[x_in_sym, x_out_sym], \
                                                 outputs=outputs)
        print("Compiling model sampler...")
        n_samples = tensor.iscalar("n_samples")
        samples = self.sample(n_samples)
        self.do_sample = theano.function([n_samples], outputs=samples, \
                                         allow_input_downcast=True)
        return

    def get_model_params(self, ary_type='numpy'):
        """
        Get the optimizable parameters in this model. This returns a list
        and, to reload this model's parameters, the list must stay in order.

        This can provide shared variables or numpy arrays.
        """
        if self.cg is None:
            self.build_model_funcs()
        joint_params = VariableFilter(roles=[PARAMETER])(self.cg.variables)
        if ary_type == 'numpy':
            for i, p in enumerate(joint_params):
                joint_params[i] = p.get_value(borrow=False)
        return joint_params

    def set_model_params(self, numpy_param_list):
        """
        Set the optimizable parameters in this model. This requires a list
        and, to reload this model's parameters, the list must be in order.
        """
        if self.cg is None:
            self.build_model_funcs()
        # grab handles for all the optimizable parameters in our cost
        joint_params = VariableFilter(roles=[PARAMETER])(self.cg.variables)
        for i, p in enumerate(joint_params):
            joint_params[i].set_value(to_fX(numpy_param_list[i]))
        return joint_params

    def save_model_params(self, f_name=None):
        """
        Save model parameters to a pickle file, in numpy form.
        """
        numpy_params = self.get_model_params(ary_type='numpy')
        f_handle = file(f_name, 'wb')
        # dump the dict self.params, which just holds "simple" python values
        cPickle.dump(numpy_params, f_handle, protocol=-1)
        f_handle.close()
        return

    def load_model_params(self, f_name=None):
        """
        Load model parameters from a pickle file, in numpy form.
        """
        pickle_file = open(f_name)
        numpy_params = cPickle.load(pickle_file)
        self.set_model_params(numpy_params)
        pickle_file.close()
        return