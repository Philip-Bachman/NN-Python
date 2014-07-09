from time import clock
import numpy as np
import numpy.random as npr
import gnumpy as gp
import random

###########################
# NEGATIVE SAMPLING LAYER #
###########################

class NSLayer:
    def __init__(self, key_count=0, in_dim=0):
        # Set stuff for managing this type of layer
        self.dim_input = in_dim
        self.key_count = key_count
        self.params = {}
        self.params['W'] = npr.randn(in_dim, key_count)
        self.params['b'] = np.zeros((1, key_count))
        self.param_grads = {}
        self.param_grads['W'] = np.zeros((in_dim, key_count))
        self.param_grads['b'] = np.zeros((1, key_count))
        self.param_moms = {}
        self.param_moms['W'] = np.zeros((in_dim, key_count))
        self.param_moms['b'] = np.zeros((1, key_count))
        self.max_norm = 10.0
        self.comp_time = 0.0
        # Set common stuff for all types layers
        self.has_params = True
        self.X = []
        self.Y = []
        self.samp_keys = []
        self.dLdX = []
        self.dLdY = []
        self.grad_idx = set()
        self.trained_idx = set()
        return

    def init_params(self, w_scale=0.01, b_scale=0.0):
        """Randomly initialize the weights in this layer."""
        self.params['W'] = w_scale * npr.randn(self.dim_input, self.key_count)
        self.param_grads['W'] = np.zeros((self.dim_input, self.key_count))
        self.params['b'] = np.zeros((1, self.key_count))
        self.param_grads['b'] = np.zeros((1, self.key_count))
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

    def feedforward(self, X, pos_samples, neg_samples):
        """Run feedforward for this layer.

        Parameter pos_samples should be a single column vector of integer
        indices into this look-up-table, and neg_samples should be a matrix
        whose columns are lut indices for some negative samples.
        """
        assert(X.shape[1] == self.params['W'].shape[0])
        assert(pos_samples.shape[0] == X.shape[0])
        assert(neg_samples.shape[0] == X.shape[0])
        assert(pos_samples.shape[1] == 1)
        t1 = clock()
        # Cleanup detritus from any previous feedforward
        self._cleanup()
        # Do new feedforward...
        pos_samples = np.maximum(pos_samples, self.key_count-1)
        neg_samples = np.maximum(neg_samples, self.key_count-1)
        self.X = X
        self.samp_keys = np.hcat([pos_samples, neg_samples]).astype(np.int32)
        W = self.params['W']
        b = self.params['b']
        Y = np.zeros((X.shape[0], self.samp_keys.shape[1]))
        for i in range(self.samp_keys.shape[1]):
            keys = self.samp_keys[:,i]
            Y[:,i] = np.sum(X.T * W[:,keys], axis=0) + b[keys]
        self.Y = Y
        t2 = clock()
        self.comp_time = self.comp_time + (t2 - t1)
        return self.Y

    def backprop(self):
        """Backprop through this layer, based on most recent feedforward.
        """
        t1 = clock()
        X = self.X
        W = self.params['W']
        dW = self.param_grads['W']
        db = self.param_grads['b']
        samp_keys = self.samp_keys
        samp_sign = np.ones(samp_keys.shape)
        samp_sign[:,0] = -1.0
        dLdY = np.log(1.0 + np.exp(samp_sign * self.Y))
        dLdX = np.zeros(self.X.shape)
        self.grad_idx.update(samp_keys.ravel())
        for i in range(dLdY.shape[0]):
            for j in range(dLdY.shape[1]):
                dldy = dLdY[i,j]
                s_key = samp_keys[i,j]
                db[0,s_key] += dldy
                dW[:,s_key] += dldy * X[i,:]
                dLdX[i,:] += dldy * W[:,s_key]
        self.dLdY = dLdY
        self.dLdX = dLdX
        t2 = clock()
        self.comp_time = self.comp_time + (t2 - t1)
        return self.dLdX

    def l2_regularize(self, lam_l2=1e-5):
        """Add gradients for l2 regularization. And compute loss."""
        t1 = clock()
        self.param_grads['W'] += lam_l2 * self.params['W']
        self.param_grads['b'] += lam_l2 * self.params['b']
        L = 0.5 * lam_l2 * (np.sum(self.params['W']**2.0) + \
                np.sum(self.params['b']**2.0))
        t2 = clock()
        self.comp_time = self.comp_time + (t2 - t1)
        return L

    def apply_grads(self, learn_rate=1e-2, ada_smooth=1e-3):
        """Apply the current accumulated gradients, with adagrad."""
        t1 = clock()
        self.trained_idx.update(self.grad_idx)
        nz_idx = np.asarray([i for i in self.grad_idx])
        self.param_moms['W'][nz_idx,:] += self.param_grads['W'][nz_idx,:]**2.0
        self.params['W'][nz_idx,:] -= learn_rate * (self.param_grads['W'][nz_idx,:] / \
                (np.sqrt(self.param_moms['W'][nz_idx,:]) + ada_smooth))
        self.param_moms['b'][nz_idx,:] += self.param_grads['b'][nz_idx,:]**2.0
        self.params['b'][nz_idx,:] -= learn_rate * (self.param_grads['b'][nz_idx,:] / \
                (np.sqrt(self.param_moms['b'][nz_idx,:]) + ada_smooth))
        self.reset_grads()
        t2 = clock()
        self.comp_time = self.comp_time + (t2 - t1)
        return

    def reset_grads(self):
        """Reset the gradient accumulators for this layer."""
        self.param_grads['W'] = 0.0 * self.param_grads['W']
        self.param_grads['b'] = 0.0 * self.param_grads['b']
        self.grad_idx = set()
        return

    def reset_moms(self, ada_init=1e-3):
        """Reset the gradient accumulators for this layer."""
        self.param_moms['W'] = (0.0 * self.param_moms['W']) + ada_init
        self.param_moms['b'] = (0.0 * self.param_moms['b']) + ada_init
        return

    def _cleanup(self, auto_prop=False):
        """Cleanup temporary feedforward/backprop stuff."""
        self.X = []
        self.Y = []
        self.samp_keys = []
        self.dLdX = []
        self.dLdY = []
        return

##############################
# HIERARCHICAL SOFTMAX LAYER #
##############################

class HSMLayer:
    def __init__(self, in_dim=0, code_vecs=0, max_code_len=0):
        # Set stuff for managing this type of layer
        self.dim_input = in_dim
        self.code_vecs = code_vecs
        self.max_code_len = max_code_len
        self.params = {}
        self.params['W'] = npr.randn(in_dim, code_vecs)
        self.params['b'] = np.zeros((1, code_vecs))
        self.param_grads = {}
        self.param_grads['W'] = np.zeros((in_dim, code_vecs))
        self.param_grads['b'] = np.zeros((1, code_vecs))
        self.param_moms = {}
        self.param_moms['W'] = np.zeros((in_dim, code_vecs))
        self.param_moms['b'] = np.zeros((1, code_vecs))
        self.max_norm = 10.0
        self.comp_time = 0.0
        # Set common stuff for all types layers
        self.has_params = True
        self.X = []
        self.code_idx = []
        self.code_sign = []
        self.Y = []
        self.dLdX = []
        self.dLdY = []
        return

    def init_params(self, w_scale=0.01, b_scale=0.0):
        """Randomly initialize the weights in this layer."""
        self.params['W'] = w_scale * npr.randn(self.dim_input, self.code_vecs)
        self.param_grads['W'] = np.zeros((self.dim_input, self.code_vecs))
        self.params['b'] = np.zeros((1, self.code_vecs))
        self.param_grads['b'] = np.zeros((1, self.code_vecs))
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

    def feedforward(self, X, code_idx, code_sign):
        """Run feedforward for this layer.
        """
        t1 = clock()
        # Cleanup detritus from any previous feedforward
        self._cleanup()
        # Do new feedforward...
        self.X = X
        self.code_idx = code_idx
        self.code_sign = code_sign
        W = self.params['W']
        b = self.params['b']
        Y = np.zeros((X.shape[0], code_idx.shape[1]))
        for i in range(code_idx.shape[1]):
            Y[:,i] = np.sum(X.T * W[:,code_idx[:,i]], axis=0) + b[0,code_idx[:,i]]
        self.Y = Y
        t2 = clock()
        self.comp_time = self.comp_time + (t2 - t1)
        return self.Y

    def backprop(self):
        """Backprop through this layer, based on most recent feedforward.
        """
        t1 = clock()
        X = self.X
        code_idx = self.code_idx
        W = self.params['W']
        b = self.params['b']
        dW = self.param_grads['W']
        db = self.param_grads['b']
        dLdY = np.log(1.0 + np.exp(-1.0 * (self.Y * self.code_sign)))
        dLdX = np.zeros(self.X.shape)
        for i in range(self.X.shape[0]):
            ci = code_idx[i,:]
            dW[:,ci] += np.outer(X[i,:], dLdY[i,:])
            db[0,ci] += dLdY[i,:]
            dLdX[i,:] = np.dot(dLdY[i,:], W[:,ci].T)
        self.dLdY = dLdY
        self.dLdX = dLdX
        t2 = clock()
        self.comp_time = self.comp_time + (t2 - t1)
        return self.dLdX

    def l2_regularize(self, lam_l2=1e-5):
        """Add gradients for l2 regularization. And compute loss."""
        self.param_grads['W'] += lam_l2 * self.params['W']
        self.param_grads['b'] += lam_l2 * self.params['b']
        L = 0.5 * lam_l2 * (np.sum(self.params['W']**2.0) + \
                np.sum(self.params['b']**2.0))
        return L

    def apply_grads(self, learn_rate=1e-2, ada_smooth=1e-3):
        """Apply the current accumulated gradients, with adagrad."""
        t1 = clock()
        self.param_moms['W'] += self.param_grads['W']**2.0
        self.param_moms['b'] += self.param_grads['b']**2.0
        self.params['W'] -= learn_rate * (self.param_grads['W'] / \
                (np.sqrt(self.param_moms['W']) + ada_smooth))
        self.params['b'] -= learn_rate * (self.param_grads['b'] / \
                (np.sqrt(self.param_moms['b']) + ada_smooth))
        t2 = clock()
        self.comp_time = self.comp_time + (t2 - t1)
        self.reset_grads()
        return

    def reset_grads(self):
        """Reset the gradient accumulators for this layer."""
        self.param_grads['W'] = 0.0 * self.param_grads['W']
        self.param_grads['b'] = 0.0 * self.param_grads['b']
        return

    def reset_moms(self, ada_init=1e-3):
        """Reset the gradient accumulators for this layer."""
        self.param_moms['W'] = (0.0 * self.param_moms['W']) + ada_init
        self.param_moms['b'] = (0.0 * self.param_moms['b']) + ada_init
        return

    def _cleanup(self, auto_prop=False):
        """Cleanup temporary feedforward/backprop stuff."""
        self.X = []
        self.code_idx = []
        self.code_sign = []
        self.Y = []
        self.dLdX = []
        self.dLdY = []
        return

##################################
# FULLY-CONNECTED SOFTMAX LAYERS #
##################################

class GPUFullLayer:
    def __init__(self, in_layer=False, in_dim=0, out_dim=0):
        # Set stuff for managing this type of layer
        self.dim_input = in_dim
        self.dim_output = out_dim
        self.params = {}
        self.params['W'] = gp.randn((in_dim, out_dim))
        self.params['b'] = gp.zeros((1, out_dim))
        self.param_grads = {}
        self.param_grads['W'] = gp.zeros((in_dim, out_dim))
        self.param_grads['b'] = gp.zeros((1, out_dim))
        self.param_moms = {}
        self.param_moms['W'] = gp.zeros((in_dim, out_dim))
        self.param_moms['b'] = gp.zeros((1, out_dim))
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
        self.params['W'] = w_scale * gp.randn((self.dim_input, self.dim_output))
        self.param_grads['W'] = gp.zeros((self.dim_input, self.dim_output))
        self.params['b'] = gp.zeros((1, self.dim_output))
        self.param_grads['b'] = gp.zeros((1, self.dim_output))
        return

    def clip_params(self):
        """Bound L2 (column-wise) norm of self.params['W'] by wt_bnd."""
        EPS = 1e-5
        W = self.params['W']
        # Compute L2 norm of weights inbound to each node in this layer
        w_norms = gp.sqrt(gp.sum(W**2.0,axis=0) + EPS)
        # Compute scales based on norms and the upperbound set by wt_bnd
        w_scales = self.max_norm / w_norms
        mask = (w_scales < 1.0)
        w_scales = (w_scales * mask) + (1.0 - mask)
        w_scales = w_scales[gp.newaxis,:]
        # Rescale weights to meet the bound set by wt_bnd
        W = W * w_scales
        return

    def feedforward(self, X, auto_prop=False):
        """Run feedforward for this layer.
        """
        t1 = clock()
        # Cleanup detritus from any previous feedforward
        self._cleanup()
        # Do new feedforward...
        self.X = gp.garray(X)
        self.Y = gp.dot(self.X, self.params['W']) + self.params['b']
        t2 = clock()
        self.comp_time = self.comp_time + (t2 - t1)
        if auto_prop and self.output_layer:
            self.output_layer.feedforward(self.Y, True)
        return self.Y

    def backprop(self, dLdY, auto_prop=False):
        """Backprop through this layer.
        """
        self.dLdY = gp.garray(dLdY)
        # Compute gradient with respect to layer parameters
        dLdW = gp.dot(self.X.T, self.dLdY)
        dLdb = gp.sum(self.dLdY, axis=0)
        dLdb = dLdb[gp.newaxis,:]
        self.param_grads['W'] += dLdW
        self.param_grads['b'] += dLdb
        # Compute gradient with respect to layer input
        self.dLdX = gp.dot(self.dLdY, self.params['W'].T)
        if auto_prop and self.input_layer:
            self.input_layer.backprop(self.dLdX, True)
        return self.dLdX

    def backprop_sm(self, Y_cat, auto_prop=False):
        """Backprop through this layer.
        """
        t1 = clock()
        Y_cat = Y_cat.astype(np.int32)
        Y_ind = np.zeros(self.Y.shape)
        Y_ind[np.arange(Y_ind.shape[0]), Y_cat] = 1.0
        dLdY_bp = self.cross_entropy(self.Y, Y_ind)
        self.backprop(dLdY_bp, auto_prop)
        t2 = clock()
        self.comp_time = self.comp_time + (t2 - t1)
        return self.dLdX

    def safe_softmax(self, Y):
        """Compute a relatively (numerically) safe softmax."""
        Y_max = gp.max(Y, axis=1)
        Y_max = Y_max[:,gp.newaxis]
        Y_exp = gp.exp(Y - Y_max)
        Y_sum = gp.sum(Y_exp, axis=1)
        Y_sum = Y_sum[:,gp.newaxis]
        Y_sm = Y_exp / Y_sum
        return Y_sm

    def cross_entropy(self, Yh, Y_ind):
        """Cross-entropy loss/grad for predictions Yh and true classes Y."""
        Y_ind = gp.garray(Y_ind)
        Yh_sm = self.safe_softmax(Yh)
        dLdYh = Yh_sm - Y_ind
        return dLdYh

    def check_loss(self, Yh, Y_cat):
        """Cross-entropy loss/grad for predictions Yh and true classes Y."""
        Y_ind = np.zeros(Yh.shape)
        Y_ind[np.arange(Y_ind.shape[0]), Y_cat] = 1.0
        Y_ind = gp.garray(Y_ind)
        Yh_sm = self.safe_softmax(Yh)
        L = -gp.sum((Y_ind * gp.log(Yh_sm)))
        return L

    def l2_regularize(self, lam_l2=1e-5):
        """Add gradients for l2 regularization. And compute loss."""
        self.param_grads['W'] += lam_l2 * self.params['W']
        self.param_grads['b'] += lam_l2 * self.params['b']
        L = 0.5 * lam_l2 * (gp.sum(self.params['W']**2.0) + \
                gp.sum(self.params['b']**2.0))
        return L

    def apply_grads(self, learn_rate=1e-2, ada_smooth=1e-3):
        """Apply the current accumulated gradients, with adagrad."""
        self.param_moms['W'] += self.param_grads['W']**2.0
        self.param_moms['b'] += self.param_grads['b']**2.0
        self.params['W'] -= learn_rate * (self.param_grads['W'] / \
                (gp.sqrt(self.param_moms['W']) + ada_smooth))
        self.params['b'] -= learn_rate * (self.param_grads['b'] / \
                (gp.sqrt(self.param_moms['b']) + ada_smooth))
        self.reset_grads()
        return

    def reset_grads(self):
        """Reset the gradient accumulators for this layer."""
        self.param_grads['W'] = 0.0 * self.param_grads['W']
        self.param_grads['b'] = 0.0 * self.param_grads['b']
        return

    def reset_moms(self, ada_init=1e-3):
        """Reset the gradient accumulators for this layer."""
        self.param_moms['W'] = (0.0 * self.param_moms['W']) + ada_init
        self.param_moms['b'] = (0.0 * self.param_moms['b']) + ada_init
        return

    def _cleanup(self, auto_prop=False):
        """Cleanup temporary feedforward/backprop stuff."""
        self.X = []
        self.Y = []
        self.dLdX = []
        self.dLdY = []
        return

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
        self.param_moms = {}
        self.param_moms['W'] = np.zeros((in_dim, out_dim))
        self.param_moms['b'] = np.zeros((1, out_dim))
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

    def feedforward(self, X, auto_prop=False):
        """Run feedforward for this layer.
        """
        t1 = clock()
        # Cleanup detritus from any previous feedforward
        self._cleanup()
        # Do new feedforward...
        self.X = X
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
        self.dLdY = self.dLdY + dLdY_bp
        # Compute gradient with respect to layer parameters
        dLdW = np.dot(self.X.T, self.dLdY)
        dLdb = np.sum(self.dLdY, axis=0, keepdims=True)
        self.param_grads['W'] = self.param_grads['W'] + dLdW
        self.param_grads['b'] = self.param_grads['b'] + dLdb
        # Compute gradient with respect to layer input
        self.dLdX = np.dot(self.dLdY, self.params['W'].T)
        if auto_prop and self.input_layer:
            self.input_layer.backprop(self.dLdX, True)
        return self.dLdX

    def backprop_sm(self, Y_cat, auto_prop=False):
        """Backprop through this layer.
        """
        t1 = clock()
        Y_cat = Y_cat.astype(np.int32)
        Y_ind = np.zeros(self.Y.shape)
        Y_ind[np.arange(Y_ind.shape[0]), Y_cat] = 1.0
        dLdY_bp = self.cross_entropy(self.Y, Y_ind)
        self.backprop(dLdY_bp, auto_prop)
        t2 = clock()
        self.comp_time = self.comp_time + (t2 - t1)
        return self.dLdX

    def safe_softmax(self, Y):
        """Compute a relatively (numerically) safe softmax."""
        Y_exp = np.exp(Y - np.max(Y, axis=1, keepdims=True))
        Y_sm = Y_exp / np.sum(Y_exp, axis=1, keepdims=True)
        return Y_sm

    def cross_entropy(self, Yh, Y_ind):
        """Cross-entropy loss/grad for predictions Yh and true classes Y."""
        Yh_sm = self.safe_softmax(Yh)
        dLdYh = Yh_sm - Y_ind
        return dLdYh

    def check_loss(self, Yh, Y_cat):
        """Cross-entropy loss/grad for predictions Yh and true classes Y."""
        Y_ind = np.zeros(Yh.shape)
        Y_ind[np.arange(Y_ind.shape[0]), Y_cat] = 1.0
        Yh_sm = self.safe_softmax(Yh)
        L = -np.sum((Y_ind * np.log(Yh_sm)))
        return L

    def l2_regularize(self, lam_l2=1e-5):
        """Add gradients for l2 regularization. And compute loss."""
        self.param_grads['W'] += lam_l2 * self.params['W']
        self.param_grads['b'] += lam_l2 * self.params['b']
        L = 0.5 * lam_l2 * (np.sum(self.params['W']**2.0) + \
                np.sum(self.params['b']**2.0))
        return L

    def apply_grads(self, learn_rate=1e-2, ada_smooth=1e-3):
        """Apply the current accumulated gradients, with adagrad."""
        self.param_moms['W'] += self.param_grads['W']**2.0
        self.param_moms['b'] += self.param_grads['b']**2.0
        self.params['W'] -= learn_rate * (self.param_grads['W'] / \
                (np.sqrt(self.param_moms['W']) + ada_smooth))
        self.params['b'] -= learn_rate * (self.param_grads['b'] / \
                (np.sqrt(self.param_moms['b']) + ada_smooth))
        self.reset_grads()
        return

    def reset_grads(self):
        """Reset the gradient accumulators for this layer."""
        self.param_grads['W'] = 0.0 * self.param_grads['W']
        self.param_grads['b'] = 0.0 * self.param_grads['b']
        return

    def reset_moms(self, ada_init=1e-3):
        """Reset the gradient accumulators for this layer."""
        self.param_moms['W'] = (0.0 * self.param_moms['W']) + ada_init
        self.param_moms['b'] = (0.0 * self.param_moms['b']) + ada_init
        return

    def _cleanup(self, auto_prop=False):
        """Cleanup temporary feedforward/backprop stuff."""
        self.X = []
        self.Y = []
        self.dLdX = []
        self.dLdY = []
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
        self.param_moms = {}
        self.param_moms['W'] = np.zeros(self.params['W'].shape)
        self.grad_idx = set()
        self.trained_idx = set()
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

    def feedforward(self, X, auto_prop=False):
        """Run feedforward for this layer.

        The input passed to feedforward here should be either a single list
        of integer indices into the look-up table or a list of lut index lists.
        """
        t1 = clock()
        # Cleanup detritus from any previous feedforward
        self._cleanup()
        # Record the incoming list of row indices to extract
        self.X = X.astype(np.int32)
        # Use look-up table to generate the desired sequences
        self.Y = self.params['W'][self.X,:]
        t2 = clock()
        self.comp_time = self.comp_time + (t2 - t1)
        if auto_prop and self.output_layer:
            self.output_layer.feedforward(self.Y, True)
        return self.Y

    def backprop(self, dLdY_bp, auto_prop=False):
        """Backprop through this layer.
        """
        t1 = clock()
        self.dLdY = dLdY_bp
        # Add the gradients to the gradient accumulator
        for i in range(self.dLdY.shape[0]):
            self.grad_idx.add(int(self.X[i]))
            self.param_grads['W'][self.X[i],:] += self.dLdY[i,:]
        t2 = clock()
        self.comp_time = self.comp_time + (t2 - t1)
        return self.param_grads['W']

    def l2_regularize(self, lam_l2=1e-5):
        """Add gradients for l2 regularization. And compute loss."""
        t1 = clock()
        self.param_grads['W'] += lam_l2 * self.params['W']
        L = 0.5 * lam_l2 * np.sum(self.params['W']**2.0)
        t2 = clock()
        self.comp_time = self.comp_time + (t2 - t1)
        return L

    def apply_grads(self, learn_rate=1e-2, ada_smooth=1e-3):
        """Apply the current accumulated gradients, with adagrad."""
        t1 = clock()
        self.trained_idx.update(self.grad_idx)
        nz_idx = np.asarray([i for i in self.grad_idx])
        self.param_moms['W'][nz_idx,:] += self.param_grads['W'][nz_idx,:]**2.0
        self.params['W'][nz_idx,:] -= learn_rate * (self.param_grads['W'][nz_idx,:] / \
                (np.sqrt(self.param_moms['W'][nz_idx,:]) + ada_smooth))
        self.reset_grads()
        t2 = clock()
        self.comp_time = self.comp_time + (t2 - t1)
        return

    def reset_grads(self):
        """Reset the gradient accumulators for this layer."""
        self.param_grads['W'] = 0.0 * self.param_grads['W']
        self.grad_idx = set()
        return

    def reset_moms(self, ada_init=1e-3):
        """Reset the gradient accumulators for this layer."""
        self.param_moms['W'] = (0.0 * self.param_moms['W']) + ada_init
        return

    def _cleanup(self):
        """Cleanup temporary feedforward/backprop stuff."""
        self.X = []
        self.Y = []
        self.dLdX = []
        self.dLdY = []
        return

#############################
# CONTEXTUAL MODIFIER LAYER #
#############################

class CMLayer:
    def __init__(self, key_count=0, source_dim=0, bias_dim=0):
        # Set stuff for managing this type of layer
        self.comp_time = 0.0
        self.params = {}
        self.params['W'] = np.zeros((key_count, source_dim))
        self.params['b'] = np.zeros((key_count, bias_dim))
        self.param_grads = {}
        self.param_grads['W'] = np.zeros(self.params['W'].shape)
        self.param_grads['b'] = np.zeros(self.params['b'].shape)
        self.param_moms = {}
        self.param_moms['W'] = np.zeros(self.params['W'].shape)
        self.param_moms['b'] = np.zeros(self.params['b'].shape)
        self.grad_idx = set()
        self.trained_idx = set()
        self.key_count = key_count
        self.source_dim = source_dim
        self.bias_dim = bias_dim
        self.max_norm = 10.0
        # Set common stuff for all types layers
        self.has_params = True
        self.X = []
        self.C = []
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
        self.params['W'] = w_scale * npr.randn(self.key_count, self.source_dim)
        self.param_grads['W'] = np.zeros(self.params['W'].shape)
        self.params['b'] = w_scale * npr.randn(self.key_count, self.bias_dim)
        self.param_grads['b'] = np.zeros(self.params['b'].shape)
        return

    def clip_params(self):
        """Bound L2 (row-wise) norm of W and b by wt_bnd."""
        EPS = 1e-5
        M = self.params['W']
        # Compute L2 norm of weights inbound to each node in this layer
        m_norms = np.sqrt(np.sum(M**2.0,axis=1) + EPS)
        # Compute scales based on norms and the upperbound set by wt_bnd
        m_scales = self.max_norm / m_norms
        mask = (m_scales < 1.0)
        m_scales = (m_scales * mask) + (1.0 - mask)
        m_scales = m_scales[:,np.newaxis]
        # Rescale weights to meet the bound set by wt_bnd
        M = M * m_scales
        # Store clipped parameters
        self.params['W'] = M
        # Do it again
        M = self.params['b']
        m_scales = self.max_norm / np.sqrt(np.sum(M**2.0,axis=1) + EPS)
        mask = (m_scales < 1.0)
        m_scales = (m_scales * mask) + (1.0 - mask)
        self.params['b'] = M * m_scales[:,np.newaxis]
        return

    def feedforward(self, X, C, auto_prop=False):
        """Run feedforward for this layer.
        """
        t1 = clock()
        # Cleanup detritus from any previous feedforward
        self._cleanup()
        # Record the incoming list of row indices to extract
        self.X = X
        self.C = C
        # Get the feature re-weighting and bias adjustment parameters
        W_exp = np.exp(self.params['W'][C,:])
        W_sig = W_exp / (1.0 + W_exp)
        # Modify X by scaling and augmenting
        self.Y = np.zeros((X.shape[0], (self.bias_dim + self.source_dim)))
        self.Y[:,0:self.bias_dim] = self.params['b'][C,:]
        self.Y[:,self.bias_dim:] = X * W_sig
        t2 = clock()
        self.comp_time = self.comp_time + (t2 - t1)
        if auto_prop and self.output_layer:
            self.output_layer.feedforward(self.Y, True)
        return self.Y

    def backprop(self, dLdY, auto_prop=False):
        """Backprop through this layer.
        """
        t1 = clock()
        # Add the gradients to the gradient accumulators
        self.dLdY = dLdY
        dLdYb = dLdY[:,0:self.bias_dim]
        dLdYw = dLdY[:,self.bias_dim:]
        C = self.C
        W_exp = np.exp(self.params['W'][C,:])
        W_sig = W_exp / (1.0 + W_exp)
        dLdW = (W_sig / W_exp) * self.X * dLdYw
        dW = self.param_grads['W']
        db = self.param_grads['b']
        self.grad_idx.update([int(i) for i in C])
        for i in range(dLdY.shape[0]):
            dW[C[i],:] += dLdW[i,:]
            db[C[i],:] += dLdYb[i,:]
        dLdX = W_sig * dLdYw
        # timing stuff
        t2 = clock()
        self.comp_time = self.comp_time + (t2 - t1)
        return dLdX

    def l2_regularize(self, lam_l2=1e-5):
        """Add gradients for l2 regularization. And compute loss."""
        t1 = clock()
        self.param_grads['W'] += lam_l2 * self.params['W']
        L = 0.5 * lam_l2 * np.sum(self.params['W']**2.0)
        t2 = clock()
        self.comp_time = self.comp_time + (t2 - t1)
        return L

    def apply_grads(self, learn_rate=1e-2, ada_smooth=1e-3):
        """Apply the current accumulated gradients, with adagrad."""
        t1 = clock()
        self.trained_idx.update(self.grad_idx)
        nz_idx = np.asarray([i for i in self.grad_idx])
        self.param_moms['W'][nz_idx,:] += self.param_grads['W'][nz_idx,:]**2.0
        self.params['W'][nz_idx,:] -= learn_rate * (self.param_grads['W'][nz_idx,:] / \
                (np.sqrt(self.param_moms['W'][nz_idx,:]) + ada_smooth))
        self.param_moms['b'][nz_idx,:] += self.param_grads['b'][nz_idx,:]**2.0
        self.params['b'][nz_idx,:] -= learn_rate * (self.param_grads['b'][nz_idx,:] / \
                (np.sqrt(self.param_moms['b'][nz_idx,:]) + ada_smooth))
        t2 = clock()
        self.comp_time = self.comp_time + (t2 - t1)
        self.reset_grads()
        return

    def reset_grads(self):
        """Reset the gradient accumulators for this layer."""
        self.param_grads['W'] = 0.0 * self.param_grads['W']
        self.param_grads['b'] = 0.0 * self.param_grads['b']
        self.grad_idx = set()
        return

    def reset_moms(self, ada_init=1e-3):
        """Reset the gradient accumulators for this layer."""
        self.param_moms['W'] = (0.0 * self.param_moms['W']) + ada_init
        self.param_moms['b'] = (0.0 * self.param_moms['b']) + ada_init
        return

    def _cleanup(self):
        """Cleanup temporary feedforward/backprop stuff."""
        self.X = []
        self.Y = []
        self.dLdX = []
        self.dLdY = []
        return

#########################
# NOISE INJECTION LAYER #
#########################

class NoiseLayer:
    def __init__(self, drop_rate=0.0, fuzz_scale=0.0, in_layer=False):
        # Set stuff required for managing this type of layer
        self.dYdX = []
        self.drop_rate = drop_rate
        self.drop_scale = 1.0 / (1.0 - drop_rate)
        self.fuzz_scale = fuzz_scale
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

    def set_noise_params(self, drop_rate=0.0, fuzz_scale=0.0):
        """Set the drop rate for this drop layer."""
        self.drop_rate = drop_rate
        self.drop_scale = 1.0 / (1.0 - drop_rate)
        self.fuzz_scale = fuzz_scale
        return

    def feedforward(self, X, auto_prop=False):
        """Perform feedforward through this layer.
        """
        # Cleanup detritus from any previous feedforward
        self._cleanup()
        # Record (a pointer to) the passed input
        self.X = X
        # Generate and apply a dropout mask to the input
        if (self.drop_rate > 1e-4):
            drop_mask = self.drop_scale * \
                    (npr.rand(self.X.shape[0], self.X.shape[1]) > self.drop_rate)
        else:
            drop_mask = np.ones((self.X.shape[0], self.X.shape[1]))
        self.dYdX = drop_mask
        if (self.fuzz_scale > 1e-4):
            fuzz_bump = (self.fuzz_scale / self.drop_scale) * \
                    npr.randn(self.X.shape[0], self.X.shape[1])
            self.Y = drop_mask * (self.X + fuzz_bump)
        else:
            self.Y = drop_mask * self.X
        if auto_prop and self.output_layer:
            self.output_layer.feedforward(self.Y, True)
        return self.Y

    def backprop(self, dLdY, auto_prop=False):
        """Perform backprop through this layer.
        """
        # Backprop is just multiplication by the mask from feedforward
        self.dLdX = dLdY * self.dYdX
        if auto_prop and self.input_layer:
            self.input_layer.backprop(self.dLdX, True)
        return self.dLdX

    def _cleanup(self):
        """Clear all temp variables for this layer."""
        self.X = []
        self.Y = []
        self.dYdX = []
        self.dLdX = []
        return

#########################
# TANH ACTIVATION LAYER #
#########################

class TanhLayer:
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

    def feedforward(self, X, auto_prop=False):
        """Perform feedforward through this layer.
        """
        # Cleanup detritus from any previous feedforward
        self._cleanup()
        # Record (a pointer to) the passed input
        self.X = X
        # Apply tanh to the input
        self.Y = np.tanh(self.X)
        if auto_prop and self.output_layer:
            self.output_layer.feedforward(self.Y, True)
        return self.Y

    def backprop(self, dLdY, auto_prop=False):
        """Perform backprop through this layer.
        """
        # Backprop is just multiplication by tanh grads, and we have tanh
        # of self.X already stored in self.Y...
        self.dLdX = dLdY * (1.0 - self.Y**2.0)
        if auto_prop and self.input_layer:
            self.input_layer.backprop(self.dLdX, True)
        return self.dLdX

    def _cleanup(self):
        """Clear all temp variables for this layer."""
        self.X = []
        self.Y = []
        self.dYdX = []
        self.dLdX = []
        return

#######################
# RANDOM KNICK-KNACKS #
#######################

def rand_idx_list(max_idx, samples):
    """Sample "samples" random ints between 0 and "max_idx"."""
    idx_list = npr.randint(0, high=max_idx, size=(samples,))
    return idx_list

def rand_word_pairs(phrase_list, pair_count, context_size):
    """Sample random anchor/context pairs for skip-gram training.

    Given a list of phrases, where each phrase is described by a list of
    indices into a look-up-table, sample random pairs of anchor word and
    context word for training a skip-gram model. The skip-gram objective is
    to predict the context word given the anchor word. The "context_size"
    determines the max separation between sampled context words and their
    corresponding anchor.
    """
    phrase_count = len(phrase_list)
    anchor_idx = np.zeros((pair_count,), dtype=np.int32)
    context_idx = np.zeros((pair_count,), dtype=np.int32)
    phrase_idx = np.zeros((pair_count,), dtype=np.int32)
    for i in range(pair_count):
        phrase_idx[i] = random.randint(0, phrase_count-1)
        phrase = phrase_list[phrase_idx[i]]
        phrase_len = len(phrase)
        a_idx = random.randint(0, phrase_len-1)
        c_max = min((a_idx+context_size), phrase_len-1)
        c_min = max((a_idx-context_size), 0)
        c_idx = random.randint(c_min, c_max)
        anchor_idx[i] = a_idx
        context_idx[i] = c_idx
    return [anchor_idx, context_idx, phrase_idx]

if __name__ == '__main__':
    batch_count = 10
    batch_size = 500
    context_size = 5
    word_count = 20000
    embed_dim = 100
    hsm_vecs = 20000
    hsm_depth = 20

    # Create a lookup table for word representations
    word_lut = LUTLayer(word_count, embed_dim)
    tanh_layer = TanhLayer(in_layer=word_lut)
    noise_layer = NoiseLayer(in_layer=tanh_layer, drop_rate=0.0, fuzz_scale=0.0)

    # Create a full/softmax layer for classification
    #class_layer = FullLayer(in_layer=False, in_dim=embed_dim, out_dim=word_count)
    class_layer = HSMLayer(in_dim=embed_dim, code_vecs=hsm_vecs, max_code_len=hsm_depth)

    # Initialize params for the LUT and softmax classifier
    word_lut.init_params(0.05)
    class_layer.init_params(0.05)

    table_time = 0.0
    print("Processing batches:")
    t1 = clock()
    for b in range(batch_count):
        # Sample a batch of random anchor/context prediction pairs for
        # training a skip-gram model.
        fake_phrases = [[random.randint(0,word_count) for i in range(50)] for i in range(10)]
        a_idx, c_idx, xxx = rand_word_pairs(fake_phrases, batch_size, context_size)
        code_idx = npr.randint(0, high=hsm_vecs, size=(batch_size,hsm_depth))
        code_sign = np.sign(npr.randn(batch_size,hsm_depth))

        tt = clock()
        # Feedforward through word look-up, tanh, and noise
        word_lut.feedforward(a_idx, auto_prop=True)
        Xb = noise_layer.Y
        table_time += clock() - tt

        # Feedforward through classification/prediction layer
        #Yb = class_layer.feedforward(Xb)
        class_layer.feedforward(Xb, code_idx, code_sign)

        # Backprop through classification/prediction layer
        #dLdXb = class_layer.backprop_sm(c_idx)
        dLdXb = class_layer.backprop()

        # Update softmax/prediction params based on batch gradients
        class_layer.apply_grads(learn_rate=1e-2, ada_smooth=1e-3)

        tt = clock()
        # Backprop through word look-up-table, tanh, and noise
        noise_layer.backprop(gp.as_numpy_array(dLdXb), auto_prop=True)
        # Update look-up-table based on gradients for this batch
        word_lut.apply_grads(learn_rate=1e-2, ada_smooth=1e-3)
        table_time += clock() - tt
        print(".")

    t2 = clock()
    e_time = t2 - t1
    print("Word count: {0:d}, word dim: {1:d}".format(word_count, embed_dim))
    print("elapsed time: {0:.4f}".format(e_time))
    print("look-up time: {0:.4f}".format(table_time))
    print("softmax time: {0:.4f}".format(class_layer.comp_time))
    print("Words per second: {0:.4f}".format((1.0*batch_count*batch_size /
        e_time)))


##############
# EYE BUFFER #
##############
