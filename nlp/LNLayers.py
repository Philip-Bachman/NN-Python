from time import clock
import numpy as np
import numpy.random as npr
from scipy import signal as signal

#
# Each LNLayer class provides several methods that are intended for use by
# an external controller.
#

#######################
# K-MAX POOLING LAYER #
#######################

class KMaxLayer:
    def __init__(self, in_layer=False):
        # self.kmax contains the desired k for each incoming sequence. Note
        # that this will need to be set by an "external controller", e.g. a
        # KMaxNet, which will know how to control feedforward and backprop
        # through sequences of KMaxLayers...
        self.kmax = []
        # self.kmax_idx will hold a reverse lookup table, for inverting the
        # kmax operation
        self.kmax_idx = []
        self.comp_time = 0.0
        # Set stuff common to all layer types
        self.has_params = False
        self.X = []
        self.Y = []
        self.dLdX = []
        self.dLdY = []
        # Set the input source for this layer, and inform the input source of
        # its output destination
        self.input_layer = in_layer
        if self.input_layer:
            self.input_layer.output_layer = self
        # Don't set self.output_layer, as it will by the layer that receives
        # this layer's output as input (see above).
        self.output_layer = []
        return

    def _find_kmax_idx(self, x, k):
        """Find the ordered indices of the k max rows for each column of x."""
        km_idx = np.argsort(x, axis=0)
        km_idx = km_idx[-k:,:]
        km_idx.sort(axis=0)
        return km_idx

    def _apply_kmax(self, x, km_idx):
        """Apply kmax using the kmax indices in km_idx."""
        y = np.zeros(km_idx.shape)
        for c in range(km_idx.shape[1]):
            y[:,c] = x[km_idx[:,c],c]
        return y

    def _unapply_kmax(self, x, dldy, km_idx):
        """Unapply kmax using the kmax indices in km_idx."""
        dldx = np.zeros(x.shape)
        for c in range(km_idx.shape[1]):
            dldx[km_idx[:,c],c] = dldy[:,c]
        return dldx

    def feedforward(self, input, auto_prop=False):
        """Perform feedforward through this layer.
        """
        t1 = clock()
        # Roughly check that self.kmax was reasonably set prior to
        # attempting feedforward
        assert (len(input) == len(self.kmax))
        # Cleanup debris from any previous feedforward
        self.cleanup()
        # Do feedforward
        self.X = input
        # Compute the indices of kmax elements for each input sequence
        self.kmax_idx = [self._find_kmax_idx(x, k) for (x, k) \
                in zip(self.X, self.kmax)]
        # Use the indices to construct the kmaxed output sequences
        self.Y = [self._apply_kmax(x, km_idx) for (x, km_idx) \
                in zip(self.X, self.kmax_idx)]
        t2 = clock()
        self.comp_time = self.comp_time + (t2 - t1)
        # Pay it forward
        if auto_prop and self.output_layer:
            self.output_layer.feedforward(self.Y, True)
        return self.Y

    def backprop(self, dLdY_bp, auto_prop=False):
        """Perform backprop through this layer.

        This augments self.dLdY by the given dLdY_bp, then computes self.dLdX
        by backpropping through ReLU via self.dYdX, and then pushes self.dLdX
        onto self.input_layer for further backpropping.
        """
        assert (len(dLdY_bp) == len(self.Y))
        for (y, dldy) in zip(self.Y, dLdY_bp):
            assert (y.shape == dldy.shape)

        # Backprop through the k-max activation for each sequence
        t1 = clock()
        self.dLdY = dLdY_bp
        self.dLdX = [self._unapply_kmax(x, dldy, km_idx) for (x, dldy, km_idx) \
                in zip(self.X, self.dLdY, self.kmax_idx)]
        t2 = clock()
        self.comp_time = self.comp_time + (t2 - t1)
        # Pay it backward
        if auto_prop and self.input_layer:
            self.input_layer.backprop(self.dLdX, True)
        return self.dLdX

    def cleanup(self, auto_prop=False):
        """Clear all temp variables for this layer."""
        self.X = []
        self.Y = []
        self.kmax_idx = []
        self.dLdY = []
        self.dLdX = []
        if auto_prop:
            self.output_layer.cleanup(True)
        return

class KMaxNormLayer:
    def __init__(self, in_layer=False):
        # self.kmax contains the desired k for each incoming sequence. Note
        # that this will need to be set by an "external controller", e.g. a
        # KMaxNet, which will know how to control feedforward and backprop
        # through sequences of KMaxLayers...
        self.kmax = []
        # self.kmax_idx will hold a reverse lookup table, for inverting the
        # kmax operation
        self.kmax_idx = []
        self.comp_time = 0.0
        # Set stuff common to all layer types
        self.has_params = False
        self.X = []
        self.Y = []
        self.dLdX = []
        self.dLdY = []
        # Set the input source for this layer, and inform the input source of
        # its output destination
        self.input_layer = in_layer
        if self.input_layer:
            self.input_layer.output_layer = self
        # Don't set self.output_layer, as it will by the layer that receives
        # this layer's output as input (see above).
        self.output_layer = []
        return

    def _find_kmax_idx(self, x, k):
        """Find the ordered indices of the k max rows for each column of x."""
        x_norms = np.sqrt(np.sum(x**2.0,axis=1))
        x_norms = x_norms.reshape((x.shape[0], 1))
        km_idx = x_norms.argsort(axis=0)
        km_idx = km_idx[-k:,0]
        km_idx.sort(axis=0)
        return km_idx

    def _apply_kmax(self, x, km_idx):
        """Apply kmax using the kmax indices in km_idx."""
        y = x[km_idx,:]
        return y

    def _unapply_kmax(self, x, dldy, km_idx):
        """Unapply kmax using the kmax indices in km_idx."""
        dldx = np.zeros(x.shape)
        for i in range(km_idx.shape[0]):
            dldx[km_idx[i],:] = dldy[i,:]
        return dldx

    def feedforward(self, input, auto_prop=False):
        """Perform feedforward through this layer.
        """
        t1 = clock()
        # Roughly check that self.kmax was reasonably set prior to
        # attempting feedforward
        assert (len(input) == len(self.kmax))
        # Cleanup debris from any previous feedforward
        self.cleanup()
        # Do feedforward
        self.X = input
        # Compute the indices of kmax elements for each input sequence
        self.kmax_idx = [self._find_kmax_idx(x, k) for (x, k) \
                in zip(self.X, self.kmax)]
        # Use the indices to construct the kmaxed output sequences
        self.Y = [self._apply_kmax(x, km_idx) for (x, km_idx) \
                in zip(self.X, self.kmax_idx)]
        t2 = clock()
        self.comp_time = self.comp_time + (t2 - t1)
        # Pay it forward
        if auto_prop and self.output_layer:
            self.output_layer.feedforward(self.Y, True)
        return self.Y

    def backprop(self, dLdY_bp, auto_prop=False):
        """Perform backprop through this layer.

        This augments self.dLdY by the given dLdY_bp, then computes self.dLdX
        by backpropping through ReLU via self.dYdX, and then pushes self.dLdX
        onto self.input_layer for further backpropping.
        """
        assert (len(dLdY_bp) == len(self.Y))
        for (y, dldy) in zip(self.Y, dLdY_bp):
            assert (y.shape == dldy.shape)

        # Backprop through the k-max activation for each sequence
        t1 = clock()
        self.dLdY = dLdY_bp
        self.dLdX = [self._unapply_kmax(x, dldy, km_idx) for (x, dldy, km_idx) \
                in zip(self.X, self.dLdY, self.kmax_idx)]
        t2 = clock()
        self.comp_time = self.comp_time + (t2 - t1)
        # Pay it backward
        if auto_prop and self.input_layer:
            self.input_layer.backprop(self.dLdX, True)
        return self.dLdX

    def cleanup(self, auto_prop=False):
        """Clear all temp variables for this layer."""
        self.X = []
        self.Y = []
        self.kmax_idx = []
        self.dLdY = []
        self.dLdX = []
        if auto_prop:
            self.output_layer.cleanup(True)
        return

#########################
# RELU ACTIVATION LAYER #
#########################

class ReluLayer:
    def __init__(self, in_layer=False):
        # Set stufff required for managing this type of layer
        self.dYdX = []
        # Set stuff common to all layer types
        self.has_params = False
        self.X = []
        self.Y = []
        self.dLdX = []
        self.dLdY = []
        # Set the input source for this layer, and inform the input source of
        # its output destination
        self.input_layer = in_layer
        if self.input_layer:
            self.input_layer.output_layer = self
        # Don't set self.output_layer, as it will by the layer that receives
        # this layer's output as input (see above).
        self.output_layer = []
        return

    def feedforward(self, input, auto_prop=False):
        """Perform feedforward through this layer.

        This sets self.X to the given input, self.Y to the ReLU-transformed
        values of the given input, self.dYdX to the ReLU derivatives, and
        initializes self.dLdX/self.dLdY to prepare for backprop. Then, it
        pushes self.Y into self.output_layer for further feedforward.
        """
        # Cleanup detritus from any previous feedforward
        self.cleanup()
        # Do new feedforward
        self.X = input
        if (type(self.X) is list):
            # Respond to a list list of gparrays
            self.dYdX = []
            self.Y = []
            self.dLdX = []
            self.dLdY = []
            for x in self.X:
                relu_mask = (x > 0.0)
                y = x * relu_mask
                self.dYdX.append(relu_mask)
                self.Y.append(y)
                self.dLdY.append(np.zeros(y.shape))
        else:
            # Respond to a single gparray
            self.dYdX = (self.X > 0.0)
            self.Y = self.X * self.dYdX
            self.dLdX = []
            self.dLdY = np.zeros(self.Y.shape)
        if auto_prop and self.output_layer:
            self.output_layer.feedforward(self.Y, True)
        return self.Y

    def backprop(self, dLdY_bp, auto_prop=False):
        """Perform backprop through this layer.

        This augments self.dLdY by the given dLdY_bp, then computes self.dLdX
        by backpropping through ReLU via self.dYdX, and then pushes self.dLdX
        onto self.input_layer for further backpropping.
        """
        assert (type(dLdY_bp) is type(self.Y))
        if (type(dLdY_bp) is list):
            # Respond to a list of gparrays
            self.dLdX = []
            for (i, dldy_bp) in enumerate(dLdY_bp):
                self.dLdY[i] = self.dLdY[i] + dldy_bp
                self.dLdX.append(self.dLdY[i] * self.dYdX[i])
        else:
            # Respond to a single gparray
            self.dLdY = self.dLdY + dLdY_bp
            self.dLdX = self.dLdY * self.dYdX
        if auto_prop and self.input_layer:
            self.input_layer.backprop(self.dLdX, True)
        return self.dLdX

    def cleanup(self, auto_prop=False):
        """Clear all temp variables for this layer."""
        self.X = []
        self.Y = []
        self.dYdX = []
        self.dLdY = []
        self.dLdX = []
        if auto_prop:
            self.output_layer.cleanup(True)
        return

#################
# DROPOUT LAYER #
#################

class DropLayer:
    def __init__(self, drop_rate=0.0, in_layer=False):
        # Set stufff required for managing this type of layer
        self.dYdX = []
        self.drop_rate = drop_rate
        self.drop_scale = 1.0 / (1.0 - drop_rate)
        # Set stuff common to all layer types
        self.has_params = False
        self.X = []
        self.Y = []
        self.dLdX = []
        self.dLdY = []
        # Set the input source for this layer, and inform the input source of
        # its output destination
        self.input_layer = in_layer
        if self.input_layer:
            self.input_layer.output_layer = self
        # Don't set self.output_layer, as it will by the layer that receives
        # this layer's output as input (see above).
        self.output_layer = []
        return

    def set_drop_rate(self, drop_rate):
        """Set the drop rate for this drop layer."""
        self.drop_rate = drop_rate
        self.drop_scale = 1.0 / (1.0 - drop_rate)
        return

    def feedforward(self, input, auto_prop=False):
        """Perform feedforward through this layer.
        """
        # Cleanup detritus from any previous feedforward
        self.cleanup()
        # Do new feedforward
        self.X = input
        if (type(self.X) is list):
            # Respond to a list list of gparrays
            self.dYdX = []
            self.Y = []
            self.dLdX = []
            for x in self.X:
                drop_mask = self.drop_scale * \
                        (npr.rand(x.shape[0], x.shape[1]) > self.drop_rate)
                self.dYdX.append(drop_mask)
                self.Y.append(drop_mask * x)
        else:
            # Respond to a single gparray
            drop_mask = self.drop_scale * \
                    (npr.rand(self.X.shape[0], self.X.shape[1]) > self.drop_rate)
            self.dYdX = drop_mask
            self.Y = drop_mask * self.X
        if auto_prop and self.output_layer:
            self.output_layer.feedforward(self.Y, True)
        return self.Y

    def backprop(self, dLdY_bp, auto_prop=False):
        """Perform backprop through this layer.
        """
        assert (type(dLdY_bp) is type(self.Y))
        if (type(dLdY_bp) is list):
            # Respond to a list of gparrays
            self.dLdX = []
            for (i, dldy_bp) in enumerate(dLdY_bp):
                self.dLdX.append(dldy_bp * self.dYdX[i])
        else:
            # Respond to a single gparray
            self.dLdX = dLdY_bp * self.dYdX
        if auto_prop and self.input_layer:
            self.input_layer.backprop(self.dLdX, True)
        return self.dLdX

    def cleanup(self, auto_prop=False):
        """Clear all temp variables for this layer."""
        self.X = []
        self.Y = []
        self.dYdX = []
        self.dLdX = []
        if auto_prop:
            self.output_layer.cleanup(True)
        return

###################
# RESHAPING LAYER #
###################

class ML2VMLayer:
    def __init__(self, in_shape=(0,0), out_shape=(0,0), in_layer=False):
        # Set stufff required for managing this type of layer
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.dYdX = []
        # Set stuff common to all layer types
        self.has_params = False
        self.X = []
        self.Y = []
        self.dLdX = []
        self.dLdY = []
        # Set the input source for this layer, and inform the input source of
        # its output destination
        self.input_layer = in_layer
        if self.input_layer:
            self.input_layer.output_layer = self
        # Don't set self.output_layer, as it will by the layer that receives
        # this layer's output as input (see above).
        self.output_layer = []
        return

    def feedforward(self, input, auto_prop=False):
        """Perform feedforward through this layer.
        """
        # Cleanup detritus from any previous feedforward
        self.cleanup()
        # Reshape...
        assert (type(input) is list)
        self.X = input
        self.Y = np.zeros((len(self.X), self.out_shape[1]))
        for (i, x) in enumerate(self.X):
            assert (x.shape == self.in_shape)
            self.Y[i,:] = x.reshape(self.out_shape)
        if auto_prop and self.output_layer:
            self.output_layer.feedforward(self.Y, True)
        return self.Y

    def backprop(self, dLdY_bp, auto_prop=False):
        """Perform backprop through this layer.
        """
        # Reshape...
        self.dLdY = dLdY_bp
        self.dLdX = []
        for i in range(self.dLdY.shape[0]):
            self.dLdX.append(self.dLdY[i,:].reshape(self.in_shape))
        if auto_prop and self.input_layer:
            self.input_layer.backprop(self.dLdX, True)
        return self.dLdX

    def cleanup(self, auto_prop=False):
        """Clear all temp variables for this layer."""
        self.X = []
        self.Y = []
        self.dYdX = []
        self.dLdX = []
        self.dLdY = []
        if auto_prop:
            self.output_layer.cleanup(True)
        return

#########################
# FULLY-CONNECTED LAYER #
#########################

class FullLayer:
    def __init__(self, in_layer=False, in_dim=0, out_dim=0):
        # Set stuff for managing this type of layer
        self.dim_input = in_dim
        self.dim_output = out_dim
        self.params = {}
        self.params['W'] = npr.randn(in_dim, out_dim)
        self.params['b'] = np.zeros((1, out_dim))
        self.param_grads = {}
        self.param_grads['W'] = np.zeros((in_dim, out_dim))
        self.param_grads['b'] = np.zeros((1, out_dim))
        self.max_norm = 10.0
        self.comp_time = 0.0
        # Set common stuff for all types layers
        self.has_params = True
        self.X = []
        self.Y = []
        self.dLdX = []
        self.dLdY = []
        # Set the input source for this layer, and inform the input source of
        # its output destination
        self.input_layer = in_layer
        if self.input_layer:
            self.input_layer.output_layer = self
        # Don't set self.output_layer, as it will be set by the layer that
        # receives this layer's output as input (see above).
        self.output_layer = False
        return

    def init_params(self, w_scale=0.01, b_scale=0.0):
        """Randomly initialize the weights in this layer."""
        self.params['W'] = w_scale * npr.randn(self.dim_input, self.dim_output)
        self.param_grads['W'] = np.zeros((self.dim_input, self.dim_output))
        self.params['b'] = np.zeros((1, self.dim_output))
        self.param_grads['b'] = np.zeros((1, self.dim_output))
        return

    def clip_params(self):
        """Bound L2 (column-wise) norm of self.params['W'] by wt_bnd."""
        EPS = 1e-5
        W = self.params['W']
        # Compute L2 norm of weights inbound to each node in this layer
        w_norms = np.sqrt(np.sum(W**2.0,axis=0) + EPS)
        # Compute scales based on norms and the upperbound set by wt_bnd
        w_scales = self.max_norm / w_norms
        mask = (w_scales < 1.0)
        w_scales = (w_scales * mask) + (1.0 - mask)
        w_scales = w_scales[np.newaxis,:]
        # Rescale weights to meet the bound set by wt_bnd
        W = W * w_scales
        return

    def feedforward(self, input, auto_prop=False):
        """Run feedforward for this layer.
        """
        t1 = clock()
        # Cleanup detritus from any previous feedforward
        self.cleanup()
        # Do new feedforward...
        self.X = input
        self.Y = np.dot(self.X, self.params['W']) + self.params['b']
        self.dLdY = np.zeros(self.Y.shape)
        t2 = clock()
        self.comp_time = self.comp_time + (t2 - t1)
        if auto_prop and self.output_layer:
            self.output_layer.feedforward(self.Y, True)
        return self.Y

    def backprop(self, dLdY_bp, auto_prop=False):
        """Backprop through this layer.
        """
        t1 = clock()
        self.dLdY = self.dLdY + dLdY_bp
        # Compute gradient with respect to layer parameters
        dLdW = np.dot(self.X.T, self.dLdY)
        dLdb = np.sum(self.dLdY, axis=0, keepdims=True)
        self.param_grads['W'] = self.param_grads['W'] + dLdW
        self.param_grads['b'] = self.param_grads['b'] + dLdb
        # Compute gradient with respect to layer input
        self.dLdX = np.dot(self.dLdY, self.params['W'].T)
        t2 = clock()
        self.comp_time = self.comp_time + (t2 - t1)
        if auto_prop and self.input_layer:
            self.input_layer.backprop(self.dLdX, True)
        return self.dLdX

    def reset_grads(self, shrink=0.0):
        """Reset the gradient accumulators for this layer."""
        self.param_grads['W'] = shrink * self.param_grads['W']
        self.param_grads['b'] = shrink * self.param_grads['b']
        return

    def cleanup(self, auto_prop=False):
        """Cleanup temporary feedforward/backprop stuff."""
        self.X = []
        self.Y = []
        self.dLdX = []
        self.dLdY = []
        if auto_prop and self.output_layer:
            self.output_layer.cleanup(True)
        return


#####################################
# ONE-DIMENSIONAL CONVOLUTION LAYER #
#####################################

class C1DLayer:
    """This layer does one-dimensional convolution over sequences of vectors.

    The input to this layer should be provided as a list of matrices. Each
    matrix in the list should have the same column count, but they may differ
    in row count. Each matrix will be treated as a one-dimensional sequence of
    vectors, with each row presenting a vector in the sequence. The output of
    this layer is produced by convolving a bank of filters with the vector
    sequence in each input matrix. Each filter has the same number of columns
    as the input matrices, and their row count is set when the object is first
    created.
    """

    def __init__(self, num_filt, filt_len, filt_dim, in_layer=False):
        # Set stuff for managing this type of layer
        self.num_filt = num_filt
        self.filt_len = filt_len
        self.filt_dim = filt_dim
        self.filt_size = filt_len * filt_dim
        self.params = {}
        self.param_grads = {}
        self.params['W'] = npr.randn(self.filt_size, self.num_filt)
        self.params['b'] = np.zeros((1, self.num_filt))
        self.param_grads['W'] = np.zeros((self.filt_size, self.num_filt))
        self.param_grads['b'] = np.zeros((1, self.num_filt))
        self.max_norm = 10.0
        self.conv_pad = np.zeros((self.filt_len-1, self.filt_dim))
        self.comp_time = 0.0
        self.Xc = []
        # Set common stuff for all types layers
        self.has_params = True
        self.X = []
        self.Y = []
        self.dLdX = []
        self.dLdY = []
        # Set the input source for this layer to False
        self.input_layer = in_layer
        if self.input_layer:
            self.input_layer.output_layer = self
        # Don't set self.output_layer, as it will be set by the layer that
        # receives this layer's output as input (see above).
        self.output_layer = False
        return

    def init_params(self, w_scale=0.01):
        """Randomly initialize the weights in this layer."""
        self.params['W'] = w_scale * npr.randn(self.filt_size, self.num_filt)
        self.param_grads['W'] = np.zeros((self.filt_size, self.num_filt))
        self.params['b'] = np.zeros((1, self.num_filt))
        self.param_grads['b'] = np.zeros((1, self.num_filt))
        return

    def clip_params(self):
        """Bound L2 (row-wise) norm of each filter by wt_bnd."""
        EPS = 1e-5
        for filt_num in range(self.num_filt):
            Wf = self.params['W'][:,filt_num].reshape((self.filt_len, self.filt_dim))
            # Compute L2 norm of per-word weights of each filter
            wf_norms = np.sqrt(np.sum(Wf**2.0,axis=1) + EPS)
            # Compute scales based on norms and the upperbound set by wt_bnd
            wf_scales = self.max_norm / wf_norms
            mask = (wf_scales < 1.0)
            wf_scales = (wf_scales * mask) + (1.0 - mask)
            wf_scales = wf_scales[:,np.newaxis]
            # Rescale weights to meet the bound set by wt_bnd
            Wf = Wf * wf_scales
            self.params['W'][:,filt_num] = Wf.ravel()
        return

    def _conv_len(self, S):
        """Compute the length of the result of convolving one of this layer's
        filters with the given vector sequence.
        """
        conv_len = S.shape[0] + self.filt_len - 1
        return conv_len

    def _conv_mat(self, S):
        """Get the convolution-friendly chunked matrix for sequence x."""
        conv_len = self._conv_len(S)
        Sp = np.concatenate([self.conv_pad, S, self.conv_pad], axis=0)
        Sc = np.zeros((conv_len, self.filt_size))
        for i in range(conv_len):
            Sc[i,:] = Sp[i:(i+self.filt_len),:].reshape((1, self.filt_size))
        return Sc

    def _conv_1d(self, Sc):
        """Compute the 1d-conv of current filters with vector sequence S."""
        # Simple matrix product, cuz we're using conv-chunk matrix format
        Yc = np.dot(Sc, self.params['W'])
        # Add the per-filter biases to complete the result
        Yc = Yc + self.params['b']
        return Yc

    def _deconv_1d(self, S, Sc, dLdY):
        """Backprop gradients dLdY through 1d-conv with sequence S."""
        conv_len = Sc.shape[0]
        seq_len = S.shape[0]
        # Compute gradients with respect to filter weights and biases
        self.param_grads['W'] += np.dot(Sc.T, dLdY)
        self.param_grads['b'] += np.sum(dLdY, axis=0, keepdims=True)
        # Compute gradients with respect to conv-chunk matrix
        dLdSc = np.dot(dLdY, self.params['W'].T)
        # Unroll and accumulate gradients over the padded version of S
        dLdSp = np.zeros(((seq_len + 2*(self.filt_len-1)), self.filt_dim))
        for i in range(conv_len):
            dLdSp[i:(i+self.filt_len),:] += dLdSc[i,:].reshape((self.filt_len, self.filt_dim))
        # Extract portion of sequence gradient derived from unpadded S
        dLdS = dLdSp[(self.filt_len - 1):((self.filt_len - 1) + seq_len),:]
        return dLdS

    def feedforward(self, input, auto_prop=False):
        """Run feedforward for this layer.

        The input passed to feedforward here should be either a single list
        of integer indices into the look-up table or a list of lut index lists.
        """
        t1 = clock()
        # Cleanup detritus from any previous feedforward
        self.cleanup()
        self.X = input
        # Generate the conv-chunk matrix for each sequence
        self.Xc = [self._conv_mat(x) for x in self.X]
        # Convolve filters with each vector sequence in the input list
        self.Y = [self._conv_1d(xc) for xc in self.Xc]
        # Stop timer
        t2 = clock()
        self.comp_time = self.comp_time + (t2 - t1)
        # Pay it forward
        if auto_prop and self.output_layer:
            self.output_layer.feedforward(self.Y, True)
        return self.Y

    def backprop(self, dLdY_bp, auto_prop=False):
        """Backprop through this layer.
        """
        # Check that the shape of the incoming gradients is valid
        assert (len(dLdY_bp) == len(self.Y))
        for (y, dldy) in zip(self.Y, dLdY_bp):
            assert (y.shape == dldy.shape)
        t1 = clock()
        self.dLdY = dLdY_bp
        # Compute gradients w.r.t. input sequences, note that this also
        # performs updates to self.param_grads['W'] and self.param_grads['b']
        # while computing gradients w.r.t. input sequences.
        self.dLdX = [self._deconv_1d(x, xc, dldy) for (x, xc, dldy) \
                in zip(self.X, self.Xc, self.dLdY)]
        t2 = clock()
        self.comp_time = self.comp_time + (t2 - t1)
        # Pay it backward
        if auto_prop and self.input_layer:
            self.input_layer.backprop(self.dLdX, True)
        return

    def reset_grads(self, shrink=0.0):
        """Reset the gradient accumulators for this layer."""
        self.param_grads['W'] = shrink * self.param_grads['W']
        self.param_grads['b'] = shrink * self.param_grads['b']
        return

    def cleanup(self, auto_prop=False):
        """Cleanup temporary feedforward/backprop stuff."""
        self.Xc = []
        self.X = []
        self.Y = []
        self.dLdX = []
        self.dLdY = []
        if auto_prop and self.output_layer:
            self.output_layer.cleanup(True)
        return

#######################
# LOOK-UP TABLE LAYER #
#######################

class LUTLayer:
    def __init__(self, key_count, embed_dim):
        # Set stuff for managing this type of layer
        self.comp_time = 0.0
        self.params = {}
        self.params['W'] = npr.randn(key_count, embed_dim)
        self.param_grads = {}
        self.param_grads['W'] = np.zeros(self.params['W'].shape)
        self.key_count = key_count
        self.embed_dim = embed_dim
        self.max_norm = 10.0
        # Set common stuff for all types layers
        self.has_params = True
        self.X = []
        self.Y = []
        self.dLdX = []
        self.dLdY = []
        # Set the input source for this layer to False
        self.input_layer = False
        # Don't set self.output_layer, as it will be set by the layer that
        # receives this layer's output as input (see above).
        self.output_layer = False
        return

    def init_params(self, w_scale=0.01):
        """Randomly initialize the weights in this layer."""
        self.params['W'] = w_scale * npr.randn(self.key_count, self.embed_dim)
        self.param_grads['W'] = np.zeros((self.key_count, self.embed_dim))
        return

    def clip_params(self):
        """Bound L2 (row-wise) norm of self.params['W'] by wt_bnd."""
        EPS = 1e-5
        W = self.params['W']
        # Compute L2 norm of weights inbound to each node in this layer
        w_norms = np.sqrt(np.sum(W**2.0,axis=1) + EPS)
        # Compute scales based on norms and the upperbound set by wt_bnd
        w_scales = self.max_norm / w_norms
        mask = (w_scales < 1.0)
        w_scales = (w_scales * mask) + (1.0 - mask)
        w_scales = w_scales[:,np.newaxis]
        # Rescale weights to meet the bound set by wt_bnd
        W = W * w_scales
        # Store clipped parameters
        self.params['W'] = W
        return

    def feedforward(self, input, auto_prop=False):
        """Run feedforward for this layer.

        The input passed to feedforward here should be either a single list
        of integer indices into the look-up table or a list of lut index lists.
        """
        # Cleanup detritus from any previous feedforward
        t1 = clock()
        self.cleanup()
        if type(input[0]) is int:
            # List-of-listsify any single list of lut indices
            self.X = [input]
        else:
            self.X = input
        # Verify input type and lut index range
        for idx_seq in self.X:
            for lut_idx in idx_seq:
                assert (type(lut_idx) is int)
                assert ((lut_idx >= 0) and (lut_idx < self.key_count))
        # Use look-up table to generate the desired sequences
        W = self.params['W']
        for idx_seq in self.X:
            # Convert this lut index sequence to a vector sequence
            vec_seq = W[idx_seq,:]
            self.Y.append(vec_seq)
        t2 = clock()
        self.comp_time = self.comp_time + (t2 - t1)
        if auto_prop and self.output_layer:
            self.output_layer.feedforward(self.Y, True)
        return self.Y

    def backprop(self, dLdY_bp, auto_prop=False):
        """Backprop through this layer.
        """
        # Check that the shape of the incoming gradients is valid
        t1 = clock()
        assert (len(dLdY_bp) == len(self.Y))
        for (out_seq, bp_seq) in zip(self.Y, dLdY_bp):
            assert (out_seq.shape == bp_seq.shape)
        self.dLdY = dLdY_bp
        dLdW = np.zeros(self.param_grads['W'].shape)
        for (dldy, idx_seq) in zip(self.dLdY, self.X):
            for (seq_idx, lut_idx) in enumerate(idx_seq):
                dLdW[lut_idx,:] = dLdW[lut_idx,:] + dldy[seq_idx,:]
        # Add the gradients to the gradient accumulator
        self.param_grads['W'] = self.param_grads['W'] + dLdW
        t2 = clock()
        self.comp_time = self.comp_time + (t2 - t1)
        return dLdW

    def reset_grads(self, shrink=0.0):
        """Reset the gradient accumulators for this layer."""
        self.param_grads['W'] = shrink * self.param_grads['W']
        return

    def cleanup(self, auto_prop=False):
        """Cleanup temporary feedforward/backprop stuff."""
        self.X = []
        self.Y = []
        self.dLdX = []
        self.dLdY = []
        if auto_prop and self.output_layer:
            self.output_layer.cleanup(True)
        return


#######################
# RANDOM KNICK-KNACKS #
#######################

def rand_idx_list(max_idx, samples):
    """Sample "samples" random ints between 0 and "max_idx"."""
    idx_list = [npr.randint(0, high=max_idx) for i in range(samples)]
    return idx_list


if __name__ == '__main__':
    obs_count = 500
    obs_dim = 32
    X = npr.randn(obs_count, obs_dim)
    b = npr.randn(obs_dim, 1)
    b = b / npr.sqrt(np.sum(b**2.0))
    Y = np.dot(X, b)
    # Construct a sequence of layers
    layers = []
    # Add a fully connected layer
    layers.append(FullLayer(in_dim=obs_dim, out_dim=100))
    layers[-1].init_params(w_scale=0.01)
    # Pass through a ReLU layer and dropout layer
    layers.append(ReluLayer(in_layer=layers[-1]))
    drop_layer = DropLayer(in_layer=layers[-1])
    layers.append(drop_layer)
    # Add another fully connected layer
    layers.append(FullLayer(in_dim=100, out_dim=Y.shape[1], in_layer=layers[-1]))
    layers[-1].init_params(w_scale=0.01)
    # Construct a look-up table layer, for testing purposes
    key_count = obs_count
    embed_dim = obs_dim
    lut_layer = LUTLayer(key_count, embed_dim)
    lut_layer.params['W'] = X
    batch_size = 25
    batch_count = 32
    # Stupid, simple test if C1DLayer class is somewhere near functional
    num_filt_1 = 16
    filt_len_1 = 5
    num_filt_2 = 32
    filt_len_2 = 5
    conv_layer_1 = C1DLayer(num_filt_1, filt_len_1, obs_dim, lut_layer)
    km_layer_1 = KMaxLayer(conv_layer_1)
    conv_layer_2 = C1DLayer(num_filt_2, filt_len_2, num_filt_1, km_layer_1)
    km_layer_2 = KMaxLayer(conv_layer_2)
    print("Feeding batches through conv layers:")
    t1 = clock()
    for i in range(5):
        idx_batches = [rand_idx_list(key_count, batch_size) for b in range(batch_count)]
        km_layer_1.kmax = [10 for idx_batch in idx_batches]
        km_layer_2.kmax = [5 for idx_batch in idx_batches]
        lut_layer.feedforward(idx_batches, True)
        Y = km_layer_2.Y
        dLdY = []
        for y in Y:
            dLdY.append(np.zeros(y.shape))
        km_layer_2.backprop(dLdY, True)
        print("-- completed batch {0:d}.".format(i))
    t2 = clock()
    e_time = t2 - t1
    print("Elapsed time: {0:.4f}".format(e_time))
    print("lut time: {0:.4f}".format(lut_layer.comp_time))
    print("conv_1 time: {0:.4f}, km_1 time: {1:.4f}".format(conv_layer_1.comp_time, km_layer_1.comp_time))
    print("conv_2 time: {0:.4f}, km_2 time: {1:.4f}".format(conv_layer_2.comp_time, km_layer_2.comp_time))



##############
# EYE BUFFER #
##############
