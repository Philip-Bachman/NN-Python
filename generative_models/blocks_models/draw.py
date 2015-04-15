from __future__ import division, print_function

import sys
sys.path.append("./lib")

import logging
import theano
import theano.tensor as T
import numpy as np

from theano import tensor

from blocks.bricks.base import application, _Brick, Brick, lazy
from blocks.bricks.recurrent import BaseRecurrent, recurrent
from blocks.bricks import Random, MLP, Linear, Tanh, Softmax, Sigmoid, Initializable
from blocks.bricks import Tanh, Identity
from blocks.utils import shared_floatx_nans
from blocks.roles import add_role, WEIGHT, BIAS, PARAMETER

from attention import ZoomableAttentionWindow
from prob_layers import replicate_batch

####################################################
# KLd for pairs of diagonal Gaussian distributions #
####################################################

def gaussian_kld(mu_left, logvar_left, mu_right, logvar_right):
    """
    Compute KL divergence between a bunch of univariate Gaussian distributions
    with the given means and log-variances.
    We do KL(N(mu_left, logvar_left) || N(mu_right, logvar_right)).
    """
    gauss_klds = 0.5 * (logvar_right - logvar_left + \
            (T.exp(logvar_left) / T.exp(logvar_right)) + \
            ((mu_left - mu_right)**2.0 / T.exp(logvar_right)) - 1.0)
    return gauss_klds

###################################################
# More standard LSTM implementation, no peepholes #
###################################################

class BasicLSTM(BaseRecurrent, Initializable):
    u"""
    Basic LSTM, no peepholes and linear part of input transform is built in.

    ----------
    dim : int
        The dimension of the hidden state.
    activation : :class:`.Brick`, optional
        The activation function. The default and by far the most popular
        is :class:`.Tanh`.
    Notes
    -----
    See :class:`.Initializable` for initialization parameters.
    """
    @lazy(allocation=['dim'])
    def __init__(self, dim, activation=None, **kwargs):
        super(LSTM, self).__init__(**kwargs)
        self.dim = dim

        if not activation:
            activation = Tanh()
        self.children = [activation]

    def get_dim(self, name):
        if name == 'inputs':
            return self.dim * 4
        if name in ['states', 'cells']:
            return self.dim
        if name == 'mask':
            return 0
        return super(LSTM, self).get_dim(name)

    def _allocate(self):
        self.W_state = shared_floatx_nans((self.dim, 4*self.dim),
                                          name='W_state')
        self.W_cell_to_in = shared_floatx_nans((self.dim,),
                                               name='W_cell_to_in')
        self.W_cell_to_forget = shared_floatx_nans((self.dim,),
                                                   name='W_cell_to_forget')
        self.W_cell_to_out = shared_floatx_nans((self.dim,),
                                                name='W_cell_to_out')
        self.biases = shared_floatx_nans((4*self.dim,), name='biases')
        add_role(self.W_state, WEIGHT)
        add_role(self.W_cell_to_in, WEIGHT)
        add_role(self.W_cell_to_forget, WEIGHT)
        add_role(self.W_cell_to_out, WEIGHT)
        add_role(self.biases, BIAS)

        self.params = [self.W_state, self.W_cell_to_in, self.W_cell_to_forget,
                       self.W_cell_to_out, self.biases]

    def _initialize(self):
        self.biases_init.initialize(self.biases, self.rng)
        for w in self.params[:-1]:
            self.weights_init.initialize(w, self.rng)

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

        activation = tensor.dot(states, self.W_state) + inputs + self.biases
        in_gate = tensor.nnet.sigmoid(slice_last(activation, 0) +
                                      cells * self.W_cell_to_in)
        forget_gate = tensor.nnet.sigmoid(slice_last(activation, 1) +
                                          cells * self.W_cell_to_forget)
        next_cells = (forget_gate * cells +
                      in_gate * nonlinearity(slice_last(activation, 2)))
        out_gate = tensor.nnet.sigmoid(slice_last(activation, 3) +
                                       next_cells * self.W_cell_to_out)
        next_states = out_gate * nonlinearity(next_cells)

        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
            next_cells = (mask[:, None] * next_cells +
                          (1 - mask[:, None]) * cells)

        return next_states, next_cells

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
        z = mean + T.exp(0.5 * logvar) * u

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

############################################
# Qsampler for pairs of diagonal Gaussians #
############################################

class Qsampler2(Initializable, Random):
    def __init__(self, input_p_dim, input_q_dim, z_dim, **kwargs):
        super(Qsampler2, self).__init__(**kwargs)

        self.mean_p_transform = Linear(
                name=self.name+'_mean_p',
                input_dim=input_p_dim, output_dim=z_dim, 
                weights_init=self.weights_init, biases_init=self.biases_init,
                use_bias=True)

        self.logvar_p_transform = Linear(
                name=self.name+'_logvar_p',
                input_dim=input_p_dim, output_dim=z_dim, 
                weights_init=self.weights_init, biases_init=self.biases_init,
                use_bias=True)

        self.mean_q_transform = Linear(
                name=self.name+'_mean_q',
                input_dim=input_q_dim, output_dim=z_dim, 
                weights_init=self.weights_init, biases_init=self.biases_init,
                use_bias=True)

        self.logvar_q_transform = Linear(
                name=self.name+'_logvar_q',
                input_dim=input_q_dim, output_dim=z_dim, 
                weights_init=self.weights_init, biases_init=self.biases_init,
                use_bias=True)

        self.children = [self.mean_p_transform, self.logvar_p_transform, \
                         self.mean_q_transform, self.logvar_q_transform]
        return
    
    def get_dim(self, name):
        if name == 'input':
            return self.mean_transform.get_dim('input')
        elif name == 'output':
            return self.mean_transform.get_dim('output')
        else:
            raise ValueError
        return

    @application(inputs=['x_p', 'x_q', 'u'], outputs=['z_p', 'z_q', 'all_klds'])
    def sample(self, x_p, x_q, u):
        """
        Return a samples and the corresponding KL term

        Parameters
        ----------
        x_p : input features for estimating p's mean and log-variance
        x_q : input features for estimating q's mean and log-variance
        u : standard Normal samples to scale and shift

        Returns
        -------
        z : tensor.matrix
            Samples drawn from Q(z|x) 
        kl : tensor.vector
            KL(Q(z|x) || P_z)
        
        """
        mean_p = self.mean_p_transform.apply(x_p)
        logvar_p = self.logvar_p_transform.apply(x_p)
        mean_q = self.mean_p_transform.apply(x_q)
        logvar_q = self.logvar_p_transform.apply(x_q)

        # ... and scale/translate samples
        z_p = mean_p + T.exp(0.5 * logvar_p) * u
        z_q = mean_q + T.exp(0.5 * logvar_q) * u

        # Calculate KL
        all_klds = gaussian_kld(mean_p, logvar_p, mean_q, logvar_q)
        return z_p, z_q, all_klds

    @application(inputs=['x_p', 'u'], outputs=['z_prior'])
    def sample_from_prior(self, x_p, u):
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
        mean_p = self.mean_p_transform.apply(x_p)
        logvar_p = self.logvar_p_transform.apply(x_p)
        z_prior = mean_p + T.exp(0.5 * logvar_p) * u
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
        return T.concatenate([x, x_hat], axis=1)

class AttentionReader(Initializable):
    def __init__(self, x_dim, dec_dim, height, width, N, **kwargs):
        super(AttentionReader, self).__init__(name="reader", **kwargs)

        self.img_height = height
        self.img_width = width
        self.N = N
        self.x_dim = x_dim
        self.dec_dim = dec_dim
        self.output_dim = 2*N*N

        self.zoomer = ZoomableAttentionWindow(height, width, N)
        self.readout = MLP(activations=[Identity()], dims=[dec_dim, 5], **kwargs)

        self.children = [self.readout]

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
        l = self.readout.apply(h_dec)

        center_y, center_x, delta, sigma, gamma = self.zoomer.nn2att(l)

        w     = gamma * self.zoomer.read(x    , center_y, center_x, delta, sigma)
        w_hat = gamma * self.zoomer.read(x_hat, center_y, center_x, delta, sigma)
        
        return T.concatenate([w, w_hat], axis=1)

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

        self.children = [self.z_trafo, self.w_trafo]

    @application(inputs=['h'], outputs=['c_update'])
    def apply(self, h):
        w = self.w_trafo.apply(h)
        l = self.z_trafo.apply(h)

        center_y, center_x, delta, sigma, gamma = self.zoomer.nn2att(l)

        c_update = 1./gamma * self.zoomer.write(w, center_y, center_x, delta, sigma)

        return c_update

    @application(inputs=['h'], outputs=['c_update', 'center_y', 'center_x', 'delta'])
    def apply_detailed(self, h):
        w = self.w_trafo.apply(h)
        l = self.z_trafo.apply(h)

        center_y, center_x, delta, sigma, gamma = self.zoomer.nn2att(l)

        c_update = 1./gamma * self.zoomer.write(w, center_y, center_x, delta, sigma)

        return c_update, center_y, center_x, delta



#-----------------------------------------------------------------------------


class DrawModel(BaseRecurrent, Initializable, Random):
    def __init__(self, n_iter, reader, 
                    encoder_mlp, encoder_rnn, sampler, 
                    decoder_mlp, decoder_rnn, writer, **kwargs):
        super(DrawModel, self).__init__(**kwargs)   
        self.n_iter = n_iter

        self.reader = reader
        self.encoder_mlp = encoder_mlp 
        self.encoder_rnn = encoder_rnn
        self.sampler = sampler
        self.decoder_mlp = decoder_mlp 
        self.decoder_rnn = decoder_rnn
        self.writer = writer

        self.children = [self.reader, self.encoder_mlp, self.encoder_rnn, self.sampler, 
                         self.writer, self.decoder_mlp, self.decoder_rnn]
        return
 
    def get_dim(self, name):
        if name == 'c':
            return self.reader.get_dim('x_dim')
        elif name == 'h_enc':
            return self.encoder_rnn.get_dim('states')
        elif name == 'c_enc':
            return self.encoder_rnn.get_dim('cells')
        elif name in ['z', 'z_mean', 'z_log_sigma']:
            return self.sampler.get_dim('output')
        elif name == 'h_dec':
            return self.decoder_rnn.get_dim('states')
        elif name == 'c_dec':
            return self.decoder_rnn.get_dim('cells')
        elif name == 'kl':
            return 0
        elif name == 'center_y':
            return 0
        elif name == 'center_x':
            return 0
        elif name == 'delta':
            return 0
        else:
            super(DrawModel, self).get_dim(name)
        return

    #------------------------------------------------------------------------

    @recurrent(sequences=['u'], contexts=['x'], 
               states=['c', 'h_enc', 'c_enc', 'z', 'kl', 'h_dec', 'c_dec'],
               outputs=['c', 'h_enc', 'c_enc', 'z', 'kl', 'h_dec', 'c_dec'])
    def iterate(self, u, c, h_enc, c_enc, z, kl, h_dec, c_dec, x):
        x_hat = x-T.nnet.sigmoid(c)
        r = self.reader.apply(x, x_hat, h_dec)
        i_enc = self.encoder_mlp.apply(T.concatenate([r, h_dec], axis=1))
        h_enc, c_enc = self.encoder_rnn.apply(states=h_enc, cells=c_enc, inputs=i_enc, iterate=False)
        z, all_klds = self.sampler.sample(h_enc, u)
        kl = T.sum(all_klds, axis=1)

        i_dec = self.decoder_mlp.apply(z)
        h_dec, c_dec = self.decoder_rnn.apply(states=h_dec, cells=c_dec, inputs=i_dec, iterate=False)
        c = c + self.writer.apply(h_dec)
        return c, h_enc, c_enc, z, kl, h_dec, c_dec

    @recurrent(sequences=['u'], contexts=[], 
               states=['c', 'h_dec', 'c_dec'],
               outputs=['c', 'h_dec', 'c_dec'])
    def decode(self, u, c, h_dec, c_dec):
        batch_size = c.shape[0]

        z = self.sampler.sample_from_prior(u)
        i_dec = self.decoder_mlp.apply(z)
        h_dec, c_dec = self.decoder_rnn.apply(
                    states=h_dec, cells=c_dec, 
                    inputs=i_dec, iterate=False)
        c = c + self.writer.apply(h_dec)
        return c, h_dec, c_dec

    #------------------------------------------------------------------------

    @application(inputs=['features'], outputs=['recons', 'kl'])
    def reconstruct(self, features):
        batch_size = features.shape[0]
        dim_z = self.get_dim('z')

        # Sample from mean-zeros std.-one Gaussian
        u = self.theano_rng.normal(
                    size=(self.n_iter, batch_size, dim_z),
                    avg=0., std=1.)

        c, h_enc, c_enc, z, kl, h_dec, c_dec = \
            rvals = self.iterate(x=features, u=u)

        x_recons = T.nnet.sigmoid(c[-1,:,:])
        x_recons.name = "reconstruction"

        kl.name = "kl"

        return x_recons, kl

    @application(inputs=['n_samples'], outputs=['samples'])
    def sample(self, n_samples):
        """Sample from model.

        Returns 
        -------

        samples : tensor3 (n_samples, n_iter, x_dim)
        """
    
        # Sample from mean-zeros std.-one Gaussian
        u_dim = self.sampler.mean_transform.get_dim('output')
        u = self.theano_rng.normal(
                    size=(self.n_iter, n_samples, u_dim),
                    avg=0., std=1.)

        c, _, _, = self.decode(u)
        #c, _, _, center_y, center_x, delta = self.decode(u)
        return T.nnet.sigmoid(c)


############################################
# Generalized DRAW model, new and improved #
############################################

class IMoDrawModels(BaseRecurrent, Initializable, Random):
    def __init__(self, n_iter, mix_enc_mlp, mix_sampler, mix_dec_mlp,
                    reader, encoder_mlp, encoder_rnn, draw_sampler, 
                    decoder_mlp, decoder_rnn, writer, **kwargs):
        super(IMoDrawModels, self).__init__(**kwargs)
        # record the desired step count
        self.n_iter = n_iter
        # grab handles for mixture stuff
        self.mix_enc_mlp = mix_enc_mlp
        self.mix_sampler = mix_sampler
        self.mix_dec_mlp = mix_dec_mlp
        # grab handles for DRAW model stuff
        self.reader = reader
        self.encoder_mlp = encoder_mlp
        self.encoder_rnn = encoder_rnn
        self.draw_sampler = draw_sampler
        self.decoder_mlp = decoder_mlp
        self.decoder_rnn = decoder_rnn
        self.writer = writer

        # record the sub-models that underlie this model
        self.children = [self.reader, self.mix_enc_mlp, self.mix_sampler,
                         self.mix_dec_mlp, self.encoder_mlp, self.encoder_rnn,
                         self.draw_sampler, self.writer, self.decoder_mlp,
                         self.decoder_rnn]
        return

    def _allocate(self):
        c_dim = self.get_dim('c')
        h_enc_dim = self.get_dim('h_enc')
        h_dec_dim = self.get_dim('h_dec')
        c_enc_dim = self.get_dim('c_enc')
        c_dec_dim = self.get_dim('c_dec')

        self.c_0 = shared_floatx_nans((1, c_dim),
                                          name='c_0')
        self.c_enc_0 = shared_floatx_nans((1, c_enc_dim),
                                               name='c_enc_0')
        self.c_dec_0 = shared_floatx_nans((1, c_dec_dim),
                                               name='c_dec_0')
        self.h_enc_0 = shared_floatx_nans((1, h_enc_dim),
                                               name='h_enc_0')
        self.h_dec_0 = shared_floatx_nans((1, h_dec_dim),
                                               name='h_dec_0')
        add_role(self.c_0, PARAMETER)
        add_role(self.c_enc_0, PARAMETER)
        add_role(self.c_dec_0, PARAMETER)
        add_role(self.h_enc_0, PARAMETER)
        add_role(self.h_dec_0, PARAMETER)
        # add the theano shared variables to our parameter lists
        self.params.extend([self.c_0, self.c_enc_0, self.c_dec_0, \
                            self.h_enc_0, self.h_dec_0])
        return

    def _initialize(self):
        # initialize to zeros...
        for p in self.params:
            p_nan = p.get_value(borrow=False)
            p_zeros = np.zeros(p_nan.shape)
            p.set_value(p_zeros.astype(theano.config.floatX))
        return
 
    def get_dim(self, name):
        if name == 'c':
            return self.reader.get_dim('x_dim')
        elif name == 'z_mix':
            return self.mix_sampler.get_dim('output')
        elif name == 'h_enc':
            return self.encoder_rnn.get_dim('states')
        elif name == 'c_enc':
            return self.encoder_rnn.get_dim('cells')
        elif name in ['z_gen', 'z_mean', 'z_log_sigma']:
            return self.draw_sampler.get_dim('output')
        elif name == 'h_dec':
            return self.decoder_rnn.get_dim('states')
        elif name == 'c_dec':
            return self.decoder_rnn.get_dim('cells')
        elif name == 'kl':
            return 0
        elif name == 'center_y':
            return 0
        elif name == 'center_x':
            return 0
        elif name == 'delta':
            return 0
        else:
            super(IMoDrawModels, self).get_dim(name)
        return

    #------------------------------------------------------------------------

    @recurrent(sequences=['u'], contexts=['x', 's_mix'],
               states=['c', 'h_enc', 'c_enc', 'z_gen', 'kl', 'h_dec', 'c_dec'],
               outputs=['c', 'h_enc', 'c_enc', 'z_gen', 'kl', 'h_dec', 'c_dec'])
    def iterate(self, u, c, h_enc, c_enc, z_gen, kl, h_dec, c_dec, x, s_mix):
        x_hat = x - T.nnet.sigmoid(c)
        r = self.reader.apply(x, x_hat, h_dec)
        i_enc = self.encoder_mlp.apply(T.concatenate([r, h_dec, s_mix], axis=1))
        h_enc, c_enc = self.encoder_rnn.apply(states=h_enc, cells=c_enc, inputs=i_enc, iterate=False)
        z_gen, akl = self.draw_sampler.sample(h_enc, u)
        kl = T.sum(akl, axis=1)

        i_dec = self.decoder_mlp.apply(T.concatenate([z_gen, s_mix], axis=1))
        h_dec, c_dec = self.decoder_rnn.apply(states=h_dec, cells=c_dec, inputs=i_dec, iterate=False)
        c = c + self.writer.apply(h_dec)
        return c, h_enc, c_enc, z_gen, kl, h_dec, c_dec

    @recurrent(sequences=['u'], contexts=['s_mix'], 
               states=['c', 'h_dec', 'c_dec'],
               outputs=['c', 'h_dec', 'c_dec'])
    def decode(self, u, c, h_dec, c_dec, s_mix):
        batch_size = c.shape[0]
        z = self.draw_sampler.sample_from_prior(u)
        i_dec = self.decoder_mlp.apply(T.concatenate([z, s_mix], axis=1))
        h_dec, c_dec = self.decoder_rnn.apply(
                    states=h_dec, cells=c_dec, 
                    inputs=i_dec, iterate=False)
        c = c + self.writer.apply(h_dec)
        return c, h_dec, c_dec

    #------------------------------------------------------------------------

    @application(inputs=['features'], outputs=['recons', 'kl'])
    def reconstruct(self, features):
        batch_size = features.shape[0]
        z_mix_dim = self.get_dim('z_mix')
        z_gen_dim = self.get_dim('z_gen')
        ce_dim = self.get_dim('c_enc')
        cd_dim = self.get_dim('c_dec')
        he_dim = self.get_dim('h_enc')
        hd_dim = self.get_dim('h_dec')

        # deal with the infinite mixture intialization
        u_mix = self.theano_rng.normal(
                    size=(batch_size, z_mix_dim),
                    avg=0., std=1.)
        f_mix = self.mix_enc_mlp.apply(features)
        z_mix, akl_mix = self.mix_sampler.sample(f_mix, u_mix)
        mix_init = self.mix_dec_mlp.apply(z_mix)
        cd0 = mix_init[:, :cd_dim]
        hd0 = mix_init[:, cd_dim:(cd_dim+hd_dim)]
        ce0 = mix_init[:, (cd_dim+hd_dim):(cd_dim+hd_dim+ce_dim)]
        he0 = mix_init[:, (cd_dim+hd_dim+ce_dim):(cd_dim+hd_dim+ce_dim+he_dim)]
        sm0 = mix_init[:, (cd_dim+hd_dim+ce_dim+he_dim):]

        # Sample from mean-zeros std.-one Gaussian
        u_draw = self.theano_rng.normal(
                    size=(self.n_iter, batch_size, z_gen_dim),
                    avg=0., std=1.)

        c, h_enc, c_enc, z, kl_draw, h_dec, c_dec = \
            rvals = self.iterate(u=u_draw, h_enc=he0, c_enc=ce0, \
                                 h_dec=hd0, c_dec=cd0, x=features, s_mix=sm0)

        # grab the observations generated by the DRAW model
        x_recons = T.nnet.sigmoid(c[-1,:,:])
        x_recons.name = "reconstruction"
        # group up the klds from mixture and DRAW models
        kl_mix = T.sum(akl_mix, axis=1)
        klm_row = kl_mix.reshape((1, batch_size), ndim=2)
        kl = T.vertical_stack(klm_row, kl_draw)
        kl.name = "kl"
        return x_recons, kl

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

        # deal with the infinite mixture intialization
        sm0 = self.theano_rng.normal(
                    size=(n_samples, z_mix_dim),
                    avg=0., std=1.)

        mix_init = self.mix_dec_mlp.apply(sm0)
        cd0 = mix_init[:, 0:cd_dim]
        hd0 = mix_init[:, cd_dim:(cd_dim+hd_dim)]

        # Sample from zero-mean unit-std. Gaussian
        u = self.theano_rng.normal(
                    size=(self.n_iter, n_samples, z_gen_dim),
                    avg=0., std=1.)

        c, _, _, = self.decode(u=u, h_dec=hd0, c_dec=cd0, s_mix=sm0)
        #c, _, _, center_y, center_x, delta = self.decode(u)
        return T.nnet.sigmoid(c)


############################################
# Multi-stage infinite mixture of mixtures #
############################################

class MultiStageModel(BaseRecurrent, Initializable, Random):
    def __init__(self, p_hi_given_si, p_sip1_given_si_hi,
                 q_z_given_x, q_hi_given_x_si, \
                 n_iter, **kwargs):
        super(MultiStageModel, self).__init__(**kwargs)
        # record the desired step count
        self.n_iter = n_iter
        # grab handles for models in the generator
        self.p_hi_given_si = p_hi_given_si
        self.p_sip1_given_si_hi = p_sip1_given_si_hi
        # grab handles for models in the guide function
        self.q_z_given_x = q_z_given_x
        self.q_hi_given_x_si = q_hi_given_x_si

        # all of these should be CondNets

        # record the sub-models that underlie this model
        self.children = [self.p_hi_given_si, self.p_sip1_given_si_hi,
                         self.q_z_given_x, self.q_hi_given_x_si]
        return

    def _allocate(self):
        # we only have the output/canvas bias
        c_dim = self.get_dim('c')
        self.c_0 = shared_floatx_nans((1, c_dim),
                                          name='c_0')
        add_role(self.c_0, PARAMETER)
        # add the theano shared variables to our parameter lists
        self.params.extend([ self.c_0 ])
        return

    def _initialize(self):
        # initialize to zeros...
        for p in self.params:
            p_nan = p.get_value(borrow=False)
            p_zeros = np.zeros(p_nan.shape)
            p.set_value(p_zeros.astype(theano.config.floatX))
        return
 
    def get_dim(self, name):
        if name == 'c':
            return self.q_z_given_x.get_dim('input')
        elif name in ['z_mix', 'u_mix']:
            return self.q_z_given_x.get_dim('output')
        elif name in ['z_gen', 'u_gen']:
            return self.p_hi_given_si.get_dim('output')
        elif name == 'kl':
            return 0
        else:
            super(MultiStageModel, self).get_dim(name)
        return

    #------------------------------------------------------------------------

    @recurrent(sequences=['u_gen'], contexts=['x_out', 's_mix'],
               states=['c', 'kl'],
               outputs=['c', 'kl'])
    def iterate(self, u_gen, c, kl, x_out, s_mix):
        # get gradient of log-likelihood w.r.t. x_out given c
        c_obs = T.nnet.sigmoid(c)
        x_hat = x_out - c_obs
        # compute distribution over next step for the guide
        q_in = T.concatenate([x_hat, c_obs, s_mix], axis=1)
        z_q, z_q_mean, z_q_logvar = self.q_hi_given_x_si.apply(q_in, u_gen)
        # compute distribution over next step for the generator
        p_in = T.concatenate([c_obs, s_mix], axis=1)
        z_p, z_p_mean, z_p_logvar = self.p_hi_given_si.apply(p_in, u_gen)
        # get KL(guide || generator)
        akl = gaussian_kld(z_q_mean, z_q_logvar, z_p_mean, z_p_logvar)
        kl = T.sum(akl, axis=1)
        # get additive update from the generator
        c_step = self.p_sip1_given_hi.apply(z_q)
        c = c + c_step
        return c, kl

    @recurrent(sequences=['u_gen'], contexts=['s_mix'], 
               states=['c'],
               outputs=['c'])
    def decode(self, u_gen, c, s_mix):
        # get the information on which to condition the step distribution
        c_obs = T.nnet.sigmoid(c)
        p_in = T.concatenate([c_obs, s_mix], axis=1)
        # scale and shift the ZMUV Gaussian samples in z
        z_p, z_p_mean, z_p_logvar = self.p_hi_given_si.apply(p_in, u_gen)
        # get steps sampled from the conditional step distribution
        c_step = self.p_sip1_given_hi.apply(z_p)
        c = c + c_step
        return c

    #------------------------------------------------------------------------

    @application(inputs=['x_in', 'x_out'], outputs=['recons', 'kl'])
    def reconstruct(self, x_in, x_out):
        batch_size = x_in.shape[0]
        u_mix_dim = self.get_dim('u_mix')
        u_gen_dim = self.get_dim('u_gen')

        # deal with the infinite mixture intialization
        u_mix = self.theano_rng.normal(
                    size=(batch_size, u_mix_dim),
                    avg=0., std=1.)
        z_mix, z_mix_mean, z_mix_logvar = self.q_z_given_x.apply(x_in, u_mix)
        akl_mix = gaussian_kld(z_mix_mean, z_mix_logvar, 0.0, 0.0)
        kl_mix = T.sum(akl_mix, axis=1).reshape((1, batch_size), ndim=2)
        s_mix = 2.0 * T.tanh(0.5 * z_mix)

        # get some zero-mean, unit std. Gaussian noise for the loop
        u_gen = self.theano_rng.normal(
                    size=(self.n_iter, batch_size, u_gen_dim),
                    avg=0., std=1.)

        # run the iterative refinement loop
        c, kl_gen = self.iterate(u_gen=u_gen, c=c0, x_out=x_out, \
                                 s_mix=s_mix)

        # grab the observations generated by the refinement model
        x_recons = T.nnet.sigmoid(c[-1,:,:])
        x_recons.name = "reconstruction"
        # group up the klds from mixture and refinement models
        kl = T.vertical_stack(klm_row, kl_draw)
        kl.name = "kl"
        return x_recons, kl

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

        # deal with the infinite mixture intialization
        sm0 = self.theano_rng.normal(
                    size=(n_samples, z_mix_dim),
                    avg=0., std=1.)

        mix_init = self.mix_dec_mlp.apply(sm0)
        cd0 = mix_init[:, 0:cd_dim]
        hd0 = mix_init[:, cd_dim:(cd_dim+hd_dim)]

        # Sample from zero-mean unit-std. Gaussian
        u = self.theano_rng.normal(
                    size=(self.n_iter, n_samples, z_gen_dim),
                    avg=0., std=1.)

        c, _, _, = self.decode(u=u, h_dec=hd0, c_dec=cd0, s_mix=sm0)
        #c, _, _, center_y, center_x, delta = self.decode(u)
        return T.nnet.sigmoid(c)