from __future__ import absolute_import

# Imports of public stuff
import numpy as np
import numpy.random as npr
import gnumpy as gp

# Imports of my stuff
import HelperFuncs as hf
from HelperFuncs import w2v_ff_bp, nsl_ff, nsl_bp, lut_bp, ag_update_2d, \
                        ag_update_1d, lut_bp, catch_oov_words

###########################
# NEGATIVE SAMPLING LAYER #
###########################

class NSLayer:
    def __init__(self, in_dim=0, max_out_key=0):
        # Record the required input dimension and output dimension. The
        # required ouput dimension is one more than the max output key.
        self.dim_input = in_dim
        self.key_count = max_out_key + 1
        self.params = {}
        self.params['W'] = 0.01 * npr.randn(self.key_count, in_dim)
        self.params['b'] = np.zeros((self.key_count,))
        self.grads = {}
        self.grads['W'] = np.zeros((self.key_count, in_dim))
        self.grads['b'] = np.zeros((self.key_count,))
        self.moms = {}
        self.moms['W'] = np.zeros((self.key_count, in_dim))
        self.moms['b'] = np.zeros((self.key_count,))
        # Set temp vars to use in feedforward/backprop
        self.X = []
        self.Y = []
        self.dLdX = []
        self.dLdY = []
        self.samp_keys = []
        self.grad_idx = set()
        return

    def init_params(self, w_scale=0.01, b_scale=0.0):
        """Randomly initialize the weights in this layer."""
        self.params['W'] = w_scale * npr.randn(self.key_count, self.dim_input)
        self.grads['W'] = np.zeros((self.key_count, self.dim_input))
        self.params['b'] = np.zeros((self.key_count,))
        self.grads['b'] = np.zeros((self.key_count,))
        return

    def clip_params(self, max_norm=10.0):
        """Bound L2 (column-wise) norm of self.params['W'] by wt_bnd."""
        W = self.params['W']
        # Compute L2 norm of weights inbound to each node in this layer
        w_norms = np.sqrt(np.sum(W**2.0,axis=1) + 1e-5)
        # Compute scales based on norms and the upperbound set by wt_bnd
        w_scales = max_norm / w_norms
        mask = (w_scales < 1.0)
        w_scales = (w_scales * mask) + (1.0 - mask)
        w_scales = w_scales[:,np.newaxis]
        # Rescale weights to meet the bound set by wt_bnd
        W = W * w_scales
        return

    def feedforward(self, X, pos_samples, neg_samples):
        """Run feedforward for this layer.

        Parameter pos_samples should be a single column vector of integer
        indices into this look-up-table, and neg_samples should be a matrix
        whose columns are lut indices for some negative samples.
        """
        assert(X.shape[1] == self.params['W'].shape[1])
        assert(pos_samples.shape[0] == X.shape[0])
        assert(neg_samples.shape[0] == X.shape[0])
        # Cleanup debris from any previous feedforward
        self._cleanup()
        # Record inputs and keys for positive/negative sample examples
        pos_samples = pos_samples[:,np.newaxis]
        self.X = X
        self.samp_keys = np.hstack((pos_samples, neg_samples))
        self.samp_keys = self.samp_keys.astype(np.int32)
        # Do the feedforward
        self.Y = np.zeros((X.shape[0], self.samp_keys.shape[1]))
        nsl_ff(self.samp_keys, self.X, self.params['W'], \
               self.params['b'], self.Y)
        # Using the outputs for these positive and negative samples, compute
        # loss and gradients for pseudo noise-contrastive training.
        samp_sign = np.ones(self.samp_keys.shape)
        samp_sign[:,0] = -1.0
        exp_ss_y = np.exp(samp_sign * self.Y)
        L = np.sum(np.log(1.0 + exp_ss_y))
        self.dLdY = samp_sign * (exp_ss_y / (1.0 + exp_ss_y))
        return L

    def backprop(self):
        """Backprop through this layer, based on most recent feedforward.
        """
        self.dLdX = np.zeros(self.X.shape)
        self.grad_idx.update(self.samp_keys.ravel())
        nsl_bp(self.samp_keys, self.X, self.params['W'], self.dLdY, \
                    self.dLdX, self.grads['W'], self.grads['b'])
        return self.dLdX

    def l2_regularize(self, lam_l2=1e-5):
        """Add gradients for l2 regularization. And compute loss."""
        self.params['W'] -= lam_l2 * self.params['W']
        self.params['b'] -= lam_l2 * self.params['b']
        return 1

    def apply_grad_reg(self, learn_rate=1e-2, ada_smooth=1e-3, lam_l2=0.0):
        """Apply the current accumulated gradients, with adagrad."""
        nz_idx = np.asarray([i for i in self.grad_idx]).astype(np.int32)
        ag_update_2d(nz_idx, self.params['W'], self.grads['W'], \
                     self.moms['W'], learn_rate, ada_smooth, lam_l2)
        ag_update_1d(nz_idx, self.params['b'], self.grads['b'], \
                     self.moms['b'], learn_rate, ada_smooth, lam_l2)
        self.grad_idx = set()
        return

    def reset_moms(self, ada_init=1e-3):
        """Reset the gradient accumulators for this layer."""
        self.moms['W'] = (0.0 * self.moms['W']) + ada_init
        self.moms['b'] = (0.0 * self.moms['b']) + ada_init
        return

    def _cleanup(self):
        """Cleanup temporary feedforward/backprop stuff."""
        self.X = []
        self.Y = []
        self.samp_keys = []
        self.dLdX = []
        self.dLdY = []
        return

#################################################
# HIERARCHICAL SOFTMAX LAYER -- VERY INCOMPLETE #
#################################################

class HSMLayer:
    def __init__(self, in_dim=0, code_vecs=0, max_code_len=0):
        # Set stuff for managing this type of layer
        self.dim_input = in_dim
        self.code_vecs = code_vecs
        self.max_code_len = max_code_len
        self.params = {}
        self.params['W'] = npr.randn(in_dim, code_vecs)
        self.params['b'] = np.zeros((1, code_vecs))
        self.grads = {}
        self.grads['W'] = np.zeros((in_dim, code_vecs))
        self.grads['b'] = np.zeros((1, code_vecs))
        self.moms = {}
        self.moms['W'] = np.zeros((in_dim, code_vecs))
        self.moms['b'] = np.zeros((1, code_vecs))
        # Set common stuff for all types layers
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
        self.grads['W'] = np.zeros((self.dim_input, self.code_vecs))
        self.params['b'] = np.zeros((1, self.code_vecs))
        self.grads['b'] = np.zeros((1, self.code_vecs))
        return

    def clip_params(self, max_norm=10.0):
        """Bound L2 (column-wise) norm of self.params['W'] by wt_bnd."""
        EPS = 1e-5
        W = self.params['W']
        # Compute L2 norm of weights inbound to each node in this layer
        w_norms = np.sqrt(np.sum(W**2.0,axis=0) + EPS)
        # Compute scales based on norms and the upperbound set by wt_bnd
        w_scales = max_norm / w_norms
        mask = (w_scales < 1.0)
        w_scales = (w_scales * mask) + (1.0 - mask)
        w_scales = w_scales[np.newaxis,:]
        # Rescale weights to meet the bound set by wt_bnd
        W = W * w_scales
        return

    def feedforward(self, X, code_idx, code_sign):
        """Run feedforward for this layer.
        """
        # Cleanup debris from any previous feedforward
        self._cleanup()
        # Do new feedforward...
        self.X = X
        self.code_idx = code_idx.astype(np.int32)
        self.code_sign = code_sign
        self.trained_idx.update(self.code_idx.ravel())
        W = self.params['W']
        b = self.params['b']
        Y = np.zeros((X.shape[0], code_idx.shape[1]))
        for i in range(code_idx.shape[1]):
            Y[:,i] = np.sum(X.T * W[:,code_idx[:,i]], axis=0) + b[0,code_idx[:,i]]
        self.Y = Y
        return self.Y

    def backprop(self):
        """Backprop through this layer, based on most recent feedforward.
        """
        X = self.X
        code_idx = self.code_idx
        W = self.params['W']
        b = self.params['b']
        dW = self.grads['W']
        db = self.grads['b']
        dLdY = np.log(1.0 + np.exp(-1.0 * (self.Y * self.code_sign)))
        dLdX = np.zeros(self.X.shape)
        for i in range(self.X.shape[0]):
            ci = code_idx[i,:]
            dW[:,ci] += np.outer(X[i,:], dLdY[i,:])
            db[0,ci] += dLdY[i,:]
            dLdX[i,:] = np.dot(dLdY[i,:], W[:,ci].T)
        self.dLdY = dLdY
        self.dLdX = dLdX
        return self.dLdX

    def l2_regularize(self, lam_l2=1e-5):
        """Add gradients for l2 regularization. And compute loss."""
        self.params['W'] -= lam_l2 * self.params['W']
        self.params['b'] -= lam_l2 * self.params['b']
        return 1

    def apply_grad_reg(self, learn_rate=1e-2, ada_smooth=1e-3, lam_l2=0.0):
        """Apply the current accumulated gradients, with adagrad."""
        self.grads['W'] += lam_l2 * self.params['W']
        self.grads['b'] += lam_l2 * self.params['b']
        self.moms['W'] += self.grads['W']**2.0
        self.moms['b'] += self.grads['b']**2.0
        self.params['W'] -= learn_rate * (self.grads['W'] / \
                (np.sqrt(self.moms['W']) + ada_smooth))
        self.params['b'] -= learn_rate * (self.grads['b'] / \
                (np.sqrt(self.moms['b']) + ada_smooth))
        self.reset_grads()
        return

    def reset_grads(self):
        """Reset the gradient accumulators for this layer."""
        self.grads['W'] = 0.0 * self.grads['W']
        self.grads['b'] = 0.0 * self.grads['b']
        return

    def reset_moms(self, ada_init=1e-3):
        """Reset the gradient accumulators for this layer."""
        self.moms['W'] = (0.0 * self.moms['W']) + ada_init
        self.moms['b'] = (0.0 * self.moms['b']) + ada_init
        return

    def _cleanup(self):
        """Cleanup temporary feedforward/backprop stuff."""
        self.X = []
        self.code_idx = []
        self.code_sign = []
        self.Y = []
        self.dLdX = []
        self.dLdY = []
        return

#################################
# FULLY-CONNECTED SOFTMAX LAYER #
#################################

class FullLayer:
    def __init__(self, in_dim=0, max_out_key=0):
        # Set dimension of incoming vectors and the number of outcomes for
        # which to perform prediction. Increment the requested prediction size
        # by 1, to accommodate 0 indexing.
        out_dim = max_out_key + 1
        self.dim_input = in_dim
        self.dim_output = out_dim
        # Initialize parameters, gradients, and adagrad "momentums"
        self.params = {}
        self.params['W'] = 0.01 * gp.randn((in_dim, out_dim))
        self.params['b'] = gp.zeros((1, out_dim))
        self.grads = {}
        self.grads['W'] = gp.zeros((in_dim, out_dim))
        self.grads['b'] = gp.zeros((1, out_dim))
        self.moms = {}
        self.moms['W'] = gp.zeros((in_dim, out_dim))
        self.moms['b'] = gp.zeros((1, out_dim))
        # Initialize temp vars to use during feedforward/backpropagation
        self.X = []
        self.Y = []
        self.Y_cat = []
        return

    def init_params(self, w_scale=0.01, b_scale=0.0):
        """Randomly initialize the weights in this layer."""
        self.params['W'] = w_scale * gp.randn((self.dim_input, self.dim_output))
        self.grads['W'] = gp.zeros((self.dim_input, self.dim_output))
        self.params['b'] = gp.zeros((1, self.dim_output))
        self.grads['b'] = gp.zeros((1, self.dim_output))
        return

    def clip_params(self, max_norm=10.0):
        """Bound l2 (row-wise) norm of self.params['W'] by max_norm."""
        W = self.params['W']
        w_norms = gp.sqrt(gp.sum(W**2.0,axis=0) + 1e-5)
        w_scales = max_norm / w_norms
        mask = (w_scales < 1.0)
        w_scales = (w_scales * mask) + (1.0 - mask)
        w_scales = w_scales[gp.newaxis,:]
        W = W * w_scales
        return

    def feedforward(self, X):
        """Run feedforward for this layer."""
        # Cleanup debris from any previous feedforward
        self._cleanup()
        # Do new feedforward...
        self.X = gp.garray(X)
        self.Y = gp.dot(self.X, self.params['W']) + self.params['b']
        return self.Y

    def backprop(self, Y_cat, return_on_gpu=False):
        """Backprop through softmax using the given target predictions."""
        Y_cat = Y_cat.astype(np.int32)
        self.Y_cat = Y_cat
        # Convert from categorical classes to "one-hot" vectors
        Y_ind = np.zeros(self.Y.shape)
        Y_ind[np.arange(Y_ind.shape[0]), Y_cat] = 1.0
        # Compute gradient of cross-entropy objective, based on the given
        # target predictions and the most recent feedforward information.
        dLdY = self.cross_entropy_grad(self.Y, Y_ind)
        # Backprop cross-ent grads to get grads w.r.t. layer parameters
        dLdW = gp.dot(self.X.T, dLdY)
        dLdb = gp.sum(dLdY, axis=0)
        dLdb = dLdb[gp.newaxis,:]
        self.grads['W'] += dLdW
        self.grads['b'] += dLdb
        # Backprop cross-ent grads to get grads w.r.t. layer input
        dLdX = gp.dot(dLdY, self.params['W'].T)
        # Return gradients w.r.t. to input, either on or off the GPU
        if not return_on_gpu:
            dLdX = gp.as_numpy_array(dLdX)
        return dLdX

    def safe_softmax(self, Y):
        """Compute a reasonably (numerically) safe softmax."""
        Y_max = gp.max(Y, axis=1)
        Y_max = Y_max[:,gp.newaxis]
        Y_exp = gp.exp(Y - Y_max)
        Y_sum = gp.sum(Y_exp, axis=1)
        Y_sum = Y_sum[:,gp.newaxis]
        Y_sm = Y_exp / Y_sum
        return Y_sm

    def cross_entropy_grad(self, Yh, Y_ind):
        """Cross-entropy gradient for predictions Yh given targets Y_ind."""
        # Push one-hot target vectors to GPU if not already there
        Y_ind = gp.garray(Y_ind)
        # Compute softmax and cross-entropy gradients
        Yh_sm = self.safe_softmax(Yh)
        dLdYh = Yh_sm - Y_ind
        return dLdYh

    def cross_entropy_loss(self, Yh, Y_cat):
        """Cross-entropy loss for predictions Yh given targets Y_cat."""
        # Convert from categorical classes to "one-hot" target vectors
        Y_ind = np.zeros(Yh.shape)
        Y_ind[np.arange(Y_ind.shape[0]), Y_cat] = 1.0
        # Push one-hot targets vectors to the GPU
        Y_ind = gp.garray(Y_ind)
        # Compute softmax and then cross-entropy loss
        Yh_sm = self.safe_softmax(Yh)
        L = -gp.sum((Y_ind * gp.log(Yh_sm)))
        return L

    def l2_regularize(self, lam_l2=1e-5):
        """Apply some amount of l2 "shrinkage" to weights and biases."""
        self.params['W'] -= lam_l2 * self.params['W']
        self.params['b'] -= lam_l2 * self.params['b']
        return

    def apply_grad_reg(self, learn_rate=1e-2, ada_smooth=1e-3, lam_l2=0.0):
        """Apply the current accumulated gradients, with adagrad."""
        # Add l2 regularization effect to the gradients
        self.grads['W'] += lam_l2 * self.params['W']
        self.grads['b'] += lam_l2 * self.params['b']
        # Update the adagrad "momentums"
        self.moms['W'] += self.grads['W']**2.0
        self.moms['b'] += self.grads['b']**2.0
        # Apply adagrad-style updates using current grads and moms
        self.params['W'] -= learn_rate * (self.grads['W'] / \
                (gp.sqrt(self.moms['W']) + ada_smooth))
        self.params['b'] -= learn_rate * (self.grads['b'] / \
                (gp.sqrt(self.moms['b']) + ada_smooth))
        # Reset gradient accumulators
        self.reset_grads()
        return

    def reset_grads(self):
        """Reset the gradient accumulators for this layer."""
        self.grads['W'] = 0.0 * self.grads['W']
        self.grads['b'] = 0.0 * self.grads['b']
        return

    def reset_moms(self, ada_init=1e-3):
        """Reset the adagrad "momentums" for this layer."""
        self.moms['W'] = (0.0 * self.moms['W']) + ada_init
        self.moms['b'] = (0.0 * self.moms['b']) + ada_init
        return

    def _cleanup(self):
        """Cleanup temp vars used during feedforward/backprop."""
        self.X = []
        self.Y = []
        self.Y_cat = []
        return

#######################
# LOOK-UP TABLE LAYER #
#######################

class LUTLayer:
    def __init__(self, max_key, embed_dim):
        # Set stuff for managing this type of layer
        self.key_count = max_key + 1 # add 1 to accommodate 0 indexing
        self.params = {}
        self.params['W'] = 0.01 * npr.randn(self.key_count, embed_dim)
        self.grads = {}
        self.grads['W'] = np.zeros(self.params['W'].shape)
        self.moms = {}
        self.moms['W'] = np.zeros(self.params['W'].shape)
        self.grad_idx = set()
        self.embed_dim = embed_dim
        self.X = []
        self.Y = []
        return

    def init_params(self, w_scale=0.01):
        """Randomly initialize the weights in this layer."""
        self.params['W'] = w_scale * npr.randn(self.key_count, self.embed_dim)
        self.grads['W'] = np.zeros((self.key_count, self.embed_dim))
        return

    def clip_params(self, max_norm=10.0):
        """Bound L2 (row-wise) norm of self.params['W'] by wt_bnd."""
        W = self.params['W']
        # Compute L2 norm of weights inbound to each node in this layer
        w_norms = np.sqrt(np.sum(W**2.0,axis=1) + 1e-5)
        # Compute scales based on norms and the upperbound set by wt_bnd
        w_scales = max_norm / w_norms
        mask = (w_scales < 1.0)
        w_scales = (w_scales * mask) + (1.0 - mask)
        w_scales = w_scales[:,np.newaxis]
        # Rescale weights to meet the bound set by wt_bnd
        W = W * w_scales
        # Store clipped parameters
        self.params['W'] = W
        return

    def feedforward(self, X):
        """Run feedforward for this layer.

        The input passed to feedforward here should be either a single list
        of integer indices into the look-up table or a list of lut index lists.
        """
        # Cleanup debris from any previous feedforward
        self._cleanup()
        # Record the incoming list of row indices to extract
        self.X = X.astype(np.int32)
        # Use look-up table to generate the desired sequences
        self.Y = self.params['W'][self.X,:]
        return self.Y

    def backprop(self, dLdY):
        """Backprop through this layer.
        """
        self.grad_idx.update(self.X.ravel())
        # Add the gradients to the gradient accumulator
        lut_bp(self.X, dLdY, self.grads['W'])
        return 1

    def l2_regularize(self, lam_l2=1e-5):
        """Add gradients for l2 regularization. And compute loss."""
        self.params['W'] -= lam_l2 * self.params['W']
        return 1

    def apply_grad_reg(self, learn_rate=1e-2, ada_smooth=1e-3, lam_l2=0.0):
        """Apply the current accumulated gradients, with adagrad."""
        nz_idx = np.asarray([i for i in self.grad_idx]).astype(np.int32)
        ag_update_2d(nz_idx, self.params['W'], self.grads['W'], \
                     self.moms['W'], learn_rate, ada_smooth, lam_l2)
        self.grad_idx = set()
        return

    def reset_moms(self, ada_init=1e-3):
        """Reset the gradient accumulators for this layer."""
        self.moms['W'] = (0.0 * self.moms['W']) + ada_init
        return

    def _cleanup(self):
        """Cleanup temporary feedforward/backprop stuff."""
        self.X = []
        self.Y = []
        return

##########################
# CONTEXT MODIFIER LAYER #
##########################

class CMLayer:
    def __init__(self, max_key=0, source_dim=0, bias_dim=0, do_rescale=False):
        # Set stuff for managing this type of layer
        self.key_count = max_key + 1 # add 1 to accommodate 0 indexing
        self.source_dim = source_dim
        self.bias_dim = bias_dim
        self.do_rescale = do_rescale # set to True for magical fun
        self.params = {}
        self.params['W'] = np.zeros((self.key_count, source_dim))
        self.params['b'] = np.zeros((self.key_count, bias_dim))
        self.grads = {}
        self.grads['W'] = np.zeros(self.params['W'].shape)
        self.grads['b'] = np.zeros(self.params['b'].shape)
        self.moms = {}
        self.moms['W'] = np.zeros(self.params['W'].shape)
        self.moms['b'] = np.zeros(self.params['b'].shape)
        self.grad_idx = set()
        # Set common stuff for all types layers
        self.X = []
        self.C = []
        self.W = []
        self.Y = []
        self.dLdX = []
        self.dLdY = []
        return

    def init_params(self, w_scale=0.01):
        """Randomly initialize the weights in this layer."""
        self.params['W'] = w_scale * npr.randn(self.key_count, self.source_dim)
        self.grads['W'] = np.zeros(self.params['W'].shape)
        self.params['b'] = w_scale * npr.randn(self.key_count, self.bias_dim)
        self.grads['b'] = np.zeros(self.params['b'].shape)
        return

    def clip_params(self, max_norm=10.0):
        """Bound L2 (row-wise) norm of W and b by wt_bnd."""
        # Rescale some parameters to unit norm
        for param in ['W', 'b']:
            M = self.params[param]
            m_scales = self.max_norm / np.sqrt(np.sum(M**2.0,axis=1) + 1e-5)
            mask = (m_scales < 1.0)
            m_scales = (m_scales * mask) + (1.0 - mask)
            self.params[param] = M * m_scales[:,np.newaxis]
        return

    def norm_info(self, param_name='W'):
        """Diagnostic info about norms of W's rows."""
        M = self.params[param_name]
        row_norms = np.sqrt(np.sum(M**2.0, axis=1))
        men_n = np.mean(row_norms)
        min_n = np.min(row_norms)
        med_n = np.median(row_norms)
        max_n = np.max(row_norms)
        info = {'mean': men_n, 'min': min_n, 'median': med_n, 'max': max_n}
        return info

    def feedforward(self, X, C):
        """Run feedforward for this layer.
        """
        # Cleanup debris from any previous feedforward
        self._cleanup()
        # Record the incoming list of row indices to extract
        self.X = X
        self.C = C.astype(np.int32)
        # Get the feature re-weighting and bias adjustment parameters
        self.W = self.params['W'][C,:]
        if self.do_rescale:
            W_exp = np.exp(self.W)
            W_sig = W_exp / (1.0 + W_exp)
        else:
            W_sig = np.ones(X.shape)
        # Modify X by scaling and augmenting
        self.Y = np.hstack((self.params['b'][C,:], (X * W_sig)))
        return self.Y

    def backprop(self, dLdY):
        """Backprop through this layer.
        """
        # Add the gradients to the gradient accumulators
        self.grad_idx.update(self.C.ravel())
        self.dLdY = dLdY
        dLdYb, dLdYw = np.hsplit(dLdY, [self.bias_dim])
        if self.do_rescale:
            W_exp = np.exp(self.W)
            W_sig = W_exp / (1.0 + W_exp)
            dLdW = (W_sig / W_exp) * self.X * dLdYw
        else:
            W_sig = np.ones(dLdYw.shape)
            dLdW = np.zeros(dLdYw.shape)
        lut_bp(self.C, dLdW, self.grads['W'])
        lut_bp(self.C, dLdYb, self.grads['b'])
        dLdX = W_sig * dLdYw
        return dLdX

    def l2_regularize(self, lam_l2=1e-5):
        """Add gradients for l2 regularization. And compute loss."""
        self.params['W'] -= lam_l2 * self.params['W']
        return 1

    def apply_grad_reg(self, learn_rate=1e-2, ada_smooth=1e-3, lam_l2=0.0):
        """Apply the current accumulated gradients, with adagrad."""
        nz_idx = np.asarray([i for i in self.grad_idx]).astype(np.int32)
        ag_update_2d(nz_idx, self.params['W'], self.grads['W'], \
                     self.moms['W'], learn_rate, ada_smooth, lam_l2)
        ag_update_2d(nz_idx, self.params['b'], self.grads['b'], \
                     self.moms['b'], learn_rate, ada_smooth, lam_l2)
        self.grad_idx = set()
        return

    def reset_moms(self, ada_init=1e-3):
        """Reset the gradient accumulators for this layer."""
        self.moms['W'] = (0.0 * self.moms['W']) + ada_init
        self.moms['b'] = (0.0 * self.moms['b']) + ada_init
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
    def __init__(self, drop_rate=0.0, fuzz_scale=0.0):
        # Set stuff required for managing this type of layer
        self.drop_rate = drop_rate
        self.drop_scale = 1.0 / (1.0 - drop_rate)
        self.fuzz_scale = fuzz_scale
        self.X = []
        self.dYdX = []
        return

    def set_noise_params(self, drop_rate=0.0, fuzz_scale=0.0):
        """Set the drop rate for this drop layer."""
        self.drop_rate = drop_rate
        self.drop_scale = 1.0 / (1.0 - drop_rate)
        self.fuzz_scale = fuzz_scale
        return

    def feedforward(self, X):
        """Perform feedforward through this layer.
        """
        # Cleanup debris from any previous feedforward
        self._cleanup()
        # Keep a pointer to the given input
        self.X = X
        # Generate a dropout mask for the input
        if (self.drop_rate > 1e-4):
            drop_mask = self.drop_scale * \
                    (npr.rand(self.X.shape[0], self.X.shape[1]) > self.drop_rate)
        else:
            drop_mask = np.ones((self.X.shape[0], self.X.shape[1]))
        self.dYdX = drop_mask
        # Add gaussian fuzz to the input and apply the dropout mask
        if (self.fuzz_scale > 1e-4):
            fuzz_bump = (self.fuzz_scale / self.drop_scale) * \
                    npr.randn(self.X.shape[0], self.X.shape[1])
            Y = drop_mask * (self.X + fuzz_bump)
        else:
            Y = drop_mask * self.X
        return Y

    def backprop(self, dLdY):
        """Perform backprop through this layer.
        """
        # Backprop is just multiplication by the mask from feedforward
        dLdX = dLdY * self.dYdX
        return dLdX

    def _cleanup(self):
        """Clear all temp variables for this layer."""
        self.X = []
        self.dYdX = []
        return

#########################
# TANH ACTIVATION LAYER #
#########################

class TanhLayer:
    def __init__(self):
        # Initialize the temp vars used in feedforward/backprop
        self.X = []
        self.Y = []
        return

    def feedforward(self, X):
        """Perform feedforward through this layer.
        """
        # Cleanup debris from any previous feedforward
        self._cleanup()
        # Record (a pointer to) the passed input
        self.X = X
        # Apply tanh to the input
        self.Y = np.tanh(self.X)
        return self.Y

    def backprop(self, dLdY):
        """Perform backprop through this layer.
        """
        # Backprop is just multiplication by tanh grads, and we have tanh
        # of self.X already stored in self.Y, so backprop is easy.
        dLdX = dLdY * (1.0 - self.Y**2.0)
        return dLdX

    def _cleanup(self):
        """Clear all temp variables for this layer."""
        self.X = []
        self.Y = []
        return

################################
# WORD-2-VEC IN A SINGLE LAYER #
################################

class W2VLayer:
    def __init__(self, max_word_key=0, word_dim=0, lam_l2=1e-3):
        # Set basic layer parameters. The max_word_key passed as an argument
        # is incremented by 1 to accommodate 0 indexing.
        self.word_dim = word_dim
        self.word_count = max_word_key + 1
        # Initialize arrays for tracking parameters, gradients, and
        # adagrad "momentums" (i.e. sums of squared gradients).
        self.params = {}
        self.params['Wa'] = 0.01 * npr.randn(self.word_count, word_dim)
        self.params['Wc'] = 0.01 * npr.randn(self.word_count, word_dim)
        self.params['b'] = np.zeros((self.word_count,))
        self.grads = {}
        self.grads['Wa'] = np.zeros((self.word_count, word_dim))
        self.grads['Wc'] = np.zeros((self.word_count, word_dim))
        self.grads['b'] = np.zeros((self.word_count,))
        self.moms = {}
        self.moms['Wa'] = np.zeros((self.word_count, word_dim))
        self.moms['Wc'] = np.zeros((self.word_count, word_dim))
        self.moms['b'] = np.zeros((self.word_count,))
        # Set l2 regularization parameter
        self.lam_l2 = lam_l2
        # Initialize sets for tracking which words we have trained
        self.trained_Wa = set()
        self.trained_Wc = set()
        return

    def init_params(self, w_scale=0.01, b_scale=0.0):
        """Randomly initialize the weights in this layer."""
        self.params['Wa'] = w_scale * npr.randn(self.word_count, self.word_dim)
        self.grads['Wa'] = np.zeros((self.word_count, self.word_dim))
        self.moms['Wa'] = np.zeros((self.word_count, self.word_dim)) + 1e-3
        self.params['Wc'] = w_scale * npr.randn(self.word_count, self.word_dim)
        self.grads['Wc'] = np.zeros((self.word_count, self.word_dim))
        self.moms['Wc'] = np.zeros((self.word_count, self.word_dim)) + 1e-3
        self.params['b'] = np.zeros((self.word_count,))
        self.grads['b'] = np.zeros((self.word_count,))
        self.moms['b'] = np.zeros((self.word_count,)) + 1e-3
        return

    def batch_train(self, anc_idx, pos_idx, neg_idx, learn_rate=1e-3):
        """Perform a batch update of all parameters based on the given sets
        of anchor/positive example/negative examples indices.
        """
        ada_smooth = 1e-3
        # Force incoming LUT indices to the right type (i.e. np.int32)
        anc_idx = anc_idx.astype(np.int32)
        pos_idx = pos_idx[:,np.newaxis]
        pn_idx = np.hstack((pos_idx, neg_idx)).astype(np.int32)
        pn_sign = np.ones(pn_idx.shape)
        pn_sign[:,0] = -1.0
        L = np.zeros((1,))
        # Do feedforward and backprop through the predictor/predictee tables
        w2v_ff_bp(anc_idx, pn_idx, pn_sign, self.params['Wa'], \
                self.params['Wc'], self.params['b'], self.grads['Wa'], \
                self.grads['Wc'], self.grads['b'], L)
        L = L[0]
        # Apply gradients to (touched only) look-up-table parameters
        a_mod_idx = np.unique(anc_idx.ravel())
        c_mod_idx = np.unique(pn_idx.ravel())
        ag_update_2d(a_mod_idx, self.params['Wa'], self.grads['Wa'], \
                self.moms['Wa'], learn_rate, ada_smooth, self.lam_l2)
        ag_update_2d(c_mod_idx, self.params['Wc'], self.grads['Wc'], \
                self.moms['Wc'], learn_rate, ada_smooth, self.lam_l2)
        ag_update_1d(c_mod_idx, self.params['b'], self.grads['b'], \
                self.moms['b'], learn_rate, ada_smooth, self.lam_l2)
        return L

    def batch_test(self, anc_idx, pos_idx, neg_idx):
        """Perform a batch update of all parameters based on the given sets
        of anchor/positive example/negative examples indices.
        """
        anc_idx = anc_idx.astype(np.int32)
        pos_idx = pos_idx[:,np.newaxis]
        pn_idx = np.hstack((pos_idx, neg_idx)).astype(np.int32)
        pn_sign = np.ones(pn_idx.shape)
        pn_sign[:,0] = -1.0
        L = np.zeros((1,))
        # Do feedforward and backprop through the predictor/predictee tables
        w2v_ff_bp(anc_idx, pn_idx, pn_sign, self.params['Wa'], \
               self.params['Wc'], self.params['b'], self.grads['Wa'], \
               self.grads['Wc'], self.grads['b'], L)
        L = L[0]
        return L

    def reset_moms(self, ada_init=1e-3):
        """Reset the gradient accumulators for this layer."""
        self.moms['Wa'] = (0.0 * self.moms['Wa']) + ada_init
        self.moms['Wc'] = (0.0 * self.moms['Wc']) + ada_init
        self.moms['b'] = (0.0 * self.moms['b']) + ada_init
        return

###################################
# TEST BASIC MODULE FUNCTIONALITY #
###################################

def run_stb_test():
    import StanfordTrees as st
     # Load tree data
    tree_dir = './trees'
    stb_data = st.SimpleLoad(tree_dir)

    # Get the lists of full train and test phrases
    tr_phrases = stb_data['train_full_phrases']
    te_phrases = stb_data['dev_full_phrases']
    # Get the list of all word occurrences in the training phrases
    tr_words = []
    for phrase in tr_phrases:
        tr_words.extend(phrase)
    tr_words = np.asarray(tr_words).astype(np.int32)
    tr_phrases = [np.asarray(p).astype(np.int32) for p in tr_phrases]
    te_phrases = [np.asarray(p).astype(np.int32) for p in te_phrases]

    max_word_key = max(stb_data['words_to_keys'].values())
    batch_count = 100001
    batch_size = 256
    context_size = 5
    embed_dim = 200
    lam_l2 = 1e-3

    # Create a lookup table for word representations
    w2v_layer = W2VLayer(max_word_key, embed_dim)

    # Initialize params for the LUT and softmax classifier
    w2v_layer.init_params(0.05)

    print("Processing batches:")
    L = 0.0
    for b in range(batch_count):
        # Sample a batch of random anchor/context prediction pairs for
        # training a skip-gram model.
        [a_idx, p_idx, n_idx, phrase_idx] = \
            hf.rand_pos_neg(tr_phrases, tr_words, batch_size, context_size, 8)

        # Compute and apply the updates for this batch
        L += w2v_layer.batch_train(a_idx, p_idx, n_idx, learn_rate=2e-4)

        # Reset adagrad smoothing factors from time-to-time
        if ((b > 1) and ((b % 10000) == 0)):
            w2v_layer.reset_moms()
        # Display loss on the training set from time-to-time
        if ((b % 200) == 0):
            obs_count = batch_size * 200.0
            print("Batch {0:d}, loss {1:.4f}".format(b, (L / obs_count)))
            L = 0.0
        # Occasionally compute loss on the validation set
        if ((b > 1) and ((b % 1000) == 0)):
            obs_count = 1000
            [a_idx, p_idx, n_idx, phrase_idx] = \
                hf.rand_pos_neg(te_phrases, tr_words, obs_count, context_size, 8)
            Lv = w2v_layer.batch_test(a_idx, p_idx, n_idx)
            print("    test loss: {0:.4f}".format(Lv / obs_count))


if __name__ == '__main__':
    run_stb_test()










##############
# EYE BUFFER #
##############
