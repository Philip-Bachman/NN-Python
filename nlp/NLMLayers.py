from __future__ import absolute_import

# Imports of public stuff
import numpy as np
import numpy.random as npr
import gnumpy as gp
import numexpr as ne

# Imports of my stuff
from HelperFuncs import randn, ones, zeros
from CythonFuncs import w2v_ff_bp, nsl_ff_bp, hsm_ff_bp, lut_bp, \
                        ag_update_2d, ag_update_1d

ADA_EPS = 1e-3

###########################
# NEGATIVE SAMPLING LAYER #
###########################

class NSLayer:
    def __init__(self, in_dim=0, max_out_key=0):
        # Record and initialize layer parameters
        self.dim_input = in_dim
        self.key_count = max_out_key + 1 # assume 0 is a key
        self.params = {}
        self.params['W'] = 0.01 * randn((self.key_count, in_dim))
        self.params['b'] = zeros((self.key_count,))
        self.grads = {}
        self.grads['W'] = zeros((self.key_count, in_dim))
        self.grads['b'] = zeros((self.key_count,))
        self.moms = {}
        self.moms['W'] = zeros((self.key_count, in_dim))
        self.moms['b'] = zeros((self.key_count,))
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
        self.params['W'] = w_scale * randn((self.key_count, self.dim_input))
        self.grads['W'] = zeros((self.key_count, self.dim_input))
        self.params['b'] = zeros((self.key_count,))
        self.grads['b'] = zeros((self.key_count,))
        return

    def clip_params(self, max_norm=5.0):
        """Bound L2 (row-wise) norm of W by max_norm."""
        M = self.params['W']
        m_scales = max_norm / np.sqrt(np.sum(M**2.0,axis=1) + 1e-5)
        mask = (m_scales < 1.0)
        mask = mask.astype(np.float32) # why is explicit cast needed?
        m_scales = (m_scales * mask) + (1.0 - mask)
        self.params['W'] = M * m_scales[:,np.newaxis]
        return

    def ff_bp(self, X, pos_samples, neg_samples, do_grad=True):
        """Perform feedforward and then backprop for this layer."""
        assert(X.shape[1] == self.params['W'].shape[1])
        assert(pos_samples.shape[0] == X.shape[0])
        assert(neg_samples.shape[0] == X.shape[0])
        do_grad = 1 if do_grad else 0 # change to int for cython code
        # Cleanup debris from any previous feedforward
        self._cleanup()
        # Record inputs and keys for positive/negative sample examples
        pos_samples = pos_samples[:,np.newaxis]
        samp_keys = np.hstack((pos_samples, neg_samples)).astype(np.uint32)
        samp_sign = -1.0 * ones(samp_keys.shape)
        samp_sign[:,0] = 1.0
        # Do feedforward and backprop all in one go
        L = zeros((1,))
        dLdX = zeros(X.shape)
        nsl_ff_bp(samp_keys, samp_sign, X, self.params['W'], self.params['b'], \
                  dLdX, self.grads['W'], self.grads['b'], L, do_grad)
        # Derp dorp
        L = L[0]
        if do_grad:
            self.grad_idx.update(samp_keys.ravel())
        return [dLdX, L]

    def l2_regularize(self, lam_l2=1e-5):
        """Add gradients for l2 regularization. And compute loss."""
        self.params['W'] -= lam_l2 * self.params['W']
        self.params['b'] -= lam_l2 * self.params['b']
        return 1

    def apply_grad(self, learn_rate=1e-2):
        """Apply the current accumulated gradients, with adagrad."""
        nz_idx = np.asarray([i for i in self.grad_idx]).astype(np.uint32)
        ag_update_2d(nz_idx, self.params['W'], self.grads['W'], \
                     self.moms['W'], learn_rate)
        ag_update_1d(nz_idx, self.params['b'], self.grads['b'], \
                     self.moms['b'], learn_rate)
        self.grad_idx = set()
        return

    def reset_moms(self, ada_init=1e-3):
        """Reset the gradient accumulators for this layer."""
        self.moms['W'] = (0.0 * self.moms['W']) + ada_init
        self.moms['b'] = (0.0 * self.moms['b']) + ada_init
        return

    def reset_grads_and_moms(self, ada_init=1e-3):
        """Reset the gradient accumulators for this layer."""
        self.grads['W'] = (0.0 * self.grads['W'])
        self.grads['b'] = (0.0 * self.grads['b'])
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
    def __init__(self, in_dim=0, max_out_key=0):
        # Record and initialize some layer parameters
        self.dim_input = in_dim
        self.key_count = max_out_key + 1 # assume 0 is a key
        self.params = {}
        self.params['W'] = 0.01 * randn((self.key_count, in_dim))
        self.params['b'] = zeros((self.key_count,))
        self.grads = {}
        self.grads['W'] = zeros((self.key_count, in_dim))
        self.grads['b'] = zeros((self.key_count,))
        self.moms = {}
        self.moms['W'] = zeros((self.key_count, in_dim))
        self.moms['b'] = zeros((self.key_count,))
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
        self.params['W'] = w_scale * randn((self.key_count, self.dim_input))
        self.grads['W'] = zeros((self.key_count, self.dim_input))
        self.params['b'] = zeros((self.key_count,))
        self.grads['b'] = zeros((self.key_count,))
        return

    def clip_params(self, max_norm=5.0):
        """Bound L2 (row-wise) norm of W by max_norm."""
        M = self.params['W']
        m_scales = max_norm / np.sqrt(np.sum(M**2.0,axis=1) + 1e-5)
        mask = (m_scales < 1.0)
        mask = mask.astype(np.float32) # why is explicit cast needed?
        m_scales = (m_scales * mask) + (1.0 - mask)
        self.params['W'] = M * m_scales[:,np.newaxis]
        return

    def ff_bp(self, X, code_keys, code_signs, do_grad=True):
        """Perform feedforward and then backprop for this layer."""
        do_grad = 1 if do_grad else 0 # change to int for cython code
        # Cleanup debris from any previous feedforward
        self._cleanup()
        # Do feedforward and backprop all in one go
        L = zeros((1,))
        dLdX = zeros(X.shape)
        hsm_ff_bp(code_keys, code_signs, X, self.params['W'], self.params['b'], \
                  dLdX, self.grads['W'], self.grads['b'], L, do_grad)
        # Derp dorp
        L = L[0]
        if do_grad:
            self.grad_idx.update(code_keys.ravel())
        return [dLdX, L]

    def l2_regularize(self, lam_l2=1e-5):
        """Add gradients for l2 regularization. And compute loss."""
        self.params['W'] -= lam_l2 * self.params['W']
        self.params['b'] -= lam_l2 * self.params['b']
        return 1

    def apply_grad(self, learn_rate=1e-2):
        """Apply the current accumulated gradients, with adagrad."""
        nz_idx = np.asarray([i for i in self.grad_idx if i >= 0]).astype(np.uint32)
        ag_update_2d(nz_idx, self.params['W'], self.grads['W'], \
                     self.moms['W'], learn_rate)
        ag_update_1d(nz_idx, self.params['b'], self.grads['b'], \
                     self.moms['b'], learn_rate)
        self.grad_idx = set()
        return

    def reset_moms(self, ada_init=1e-3):
        """Reset the gradient accumulators for this layer."""
        self.moms['W'] = (0.0 * self.moms['W']) + ada_init
        self.moms['b'] = (0.0 * self.moms['b']) + ada_init
        return

    def reset_grads_and_moms(self, ada_init=1e-3):
        """Reset the gradient accumulators for this layer."""
        self.grads['W'] = (0.0 * self.grads['W'])
        self.grads['b'] = (0.0 * self.grads['b'])
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
        """Bound L2 (row-wise) norm of W by max_norm."""
        M = self.params['W']
        m_scales = max_norm / gp.sqrt(gp.sum(M**2.0,axis=1) + 1e-5)
        mask = (m_scales < 1.0) # with gnumpy, this already comes as float32
        m_scales = (m_scales * mask) + (1.0 - mask)
        self.params['W'] = M * m_scales[:,gp.newaxis]
        return

    def feedforward(self, X):
        """Run feedforward for this layer."""
        # Cleanup debris from any previous feedforward
        self._cleanup()
        # Do new feedforward...
        self.X = gp.garray(X)
        self.Y = gp.dot(self.X, self.params['W']) + self.params['b']
        return self.Y

    def backprop(self, Y_cat, L_ary=None, return_on_gpu=False):
        """Backprop through softmax using the given target predictions."""
        # Compute gradient of cross-entropy objective, based on the given
        # target predictions and the most recent feedforward information.
        L, dLdY = self.xent_loss_and_grad(self.Y, Y_cat.astype(np.uint32))
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
            dLdX = gp.as_numpy_array(dLdX).astype(np.float32)
        # Write loss into L_ary if it was given
        L_ary[0] = L
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

    def xent_loss_and_grad(self, Yh, Y_cat):
        """Cross-entropy loss for predictions Yh given targets Y_cat."""
        # Convert from categorical classes to "one-hot" target vectors
        Y_ind = zeros(Yh.shape)
        Y_ind[np.arange(Y_ind.shape[0]), Y_cat] = 1.0
        # Push one-hot targets vectors to the GPU
        Y_ind = gp.garray(Y_ind)
        # Compute softmax and then cross-entropy loss
        Yh_sm = self.safe_softmax(Yh)
        L = -gp.sum((Y_ind * gp.log(Yh_sm)))
        dLdYh = Yh_sm - Y_ind
        return [L, dLdYh]

    def l2_regularize(self, lam_l2=1e-5):
        """Apply some amount of l2 "shrinkage" to weights and biases."""
        self.params['W'] -= lam_l2 * self.params['W']
        self.params['b'] -= lam_l2 * self.params['b']
        return

    def apply_grad(self, learn_rate=1e-2,):
        """Apply the current accumulated gradients, with adagrad."""
        # Update the adagrad "momentums"
        self.moms['W'] = (0.95 * self.moms['W']) + (0.05 * self.grads['W']**2.0)
        self.moms['b'] = (0.95 * self.moms['b']) + (0.05 * self.grads['b']**2.0)
        # Apply adagrad-style updates using current grads and moms
        self.params['W'] -= learn_rate * (self.grads['W'] / \
                (gp.sqrt(self.moms['W']) + ADA_EPS))
        self.params['b'] -= learn_rate * (self.grads['b'] / \
                (gp.sqrt(self.moms['b']) + ADA_EPS))
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
    def __init__(self, max_key, embed_dim, n_gram=1):
        # Set stuff for managing this type of layer
        self.key_count = max_key + 1 # add 1 to accommodate 0 indexing
        self.params = {}
        self.params['W'] = 0.01 * randn((self.key_count, embed_dim))
        self.grads = {}
        self.grads['W'] = zeros(self.params['W'].shape)
        self.moms = {}
        self.moms['W'] = zeros(self.params['W'].shape)
        self.grad_idx = set()
        self.embed_dim = embed_dim
        self.n_gram = n_gram
        self.X = []
        self.Y = []
        return

    def init_params(self, w_scale=0.01):
        """Randomly initialize the weights in this layer."""
        self.params['W'] = w_scale * randn((self.key_count, self.embed_dim))
        self.grads['W'] = zeros((self.key_count, self.embed_dim))
        return

    def clip_params(self, max_norm=5.0):
        """Bound L2 (row-wise) norm of W by max_norm."""
        M = self.params['W']
        m_scales = max_norm / np.sqrt(np.sum(M**2.0,axis=1) + 1e-5)
        mask = (m_scales < 1.0)
        mask = mask.astype(np.float32) # why is explicit cast needed?
        m_scales = (m_scales * mask) + (1.0 - mask)
        self.params['W'] = M * m_scales[:,np.newaxis]
        return

    def feedforward(self, X):
        """Run feedforward for this layer.

        The input passed to feedforward here should be either a single list
        of integer indices into the look-up table or a list of lut index lists.
        """
        # Cleanup debris from any previous feedforward
        self._cleanup()
        # Record the incoming list of row indices to extract
        self.X = X.astype(np.uint32)
        # Use look-up table to generate the desired sequences
        if (self.n_gram == 1):
            self.Y = self.params['W'].take(self.X, axis=0)
        else:
            self.Y = zeros((self.X.shape[0], (self.n_gram * self.embed_dim)))
            for i in range(self.n_gram):
                s_idx = i * self.embed_dim
                e_idx = s_idx + self.embed_dim
                self.Y[:,s_idx:e_idx] = self.params['W'].take(self.X[:,i], axis=0)
        return self.Y

    def backprop(self, dLdY):
        """Backprop through this layer.
        """
        self.grad_idx.update(self.X.ravel())
        # Add the gradients to the gradient accumulator
        if (self.n_gram == 1):
            lut_bp(self.X, dLdY, self.grads['W'])
        else:
            # Backprop for each of the predictor words
            for i in range(self.n_gram):
                s_idx = i * self.embed_dim
                e_idx = s_idx + self.embed_dim
                lut_bp(self.X[:,i], dLdY[:,s_idx:e_idx], self.grads['W'])
        return 1

    def l2_regularize(self, lam_l2=1e-5):
        """Add gradients for l2 regularization. And compute loss."""
        self.params['W'] -= lam_l2 * self.params['W']
        return 1

    def apply_grad(self, learn_rate=1e-2):
        """Apply the current accumulated gradients, with adagrad."""
        nz_idx = np.asarray([i for i in self.grad_idx]).astype(np.uint32)
        ag_update_2d(nz_idx, self.params['W'], self.grads['W'], \
                     self.moms['W'], learn_rate)
        self.grad_idx = set()
        return

    def reset_moms(self, ada_init=1e-3):
        """Reset the gradient accumulators for this layer."""
        self.moms['W'] = (0.0 * self.moms['W']) + ada_init
        return

    def reset_grads_and_moms(self, ada_init=1e-3):
        """Reset the gradient accumulators for this layer."""
        self.grads['W'] = (0.0 * self.grads['W'])
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
        self.params['Wm'] = zeros((self.key_count, source_dim))
        self.params['Wb'] = zeros((self.key_count, bias_dim))
        self.grads = {}
        self.grads['Wm'] = zeros(self.params['Wm'].shape)
        self.grads['Wb'] = zeros(self.params['Wb'].shape)
        self.moms = {}
        self.moms['Wm'] = zeros(self.params['Wm'].shape)
        self.moms['Wb'] = zeros(self.params['Wb'].shape)
        self.grad_idx = set()
        # Set common stuff for all types layers
        self.X = []
        self.C = []
        self.Wm_exp = []
        self.Wm_sig = []
        self.Y = []
        self.dLdX = []
        self.dLdY = []
        return

    def init_params(self, w_scale=0.01):
        """Randomly initialize the weights in this layer."""
        self.params['Wm'] = w_scale * randn((self.key_count, self.source_dim))
        self.grads['Wm'] = zeros(self.params['Wm'].shape)
        self.params['Wb'] = w_scale * randn((self.key_count, self.bias_dim))
        self.grads['Wb'] = zeros(self.params['Wb'].shape)
        return

    def clip_params(self, Wm_norm=5.0, Wb_norm=5.0):
        """Bound L2 (row-wise) norm of Wm and Wb by max_norm."""
        for (param, max_norm) in zip(['Wm','Wb'],[Wm_norm, Wb_norm]):
            M = self.params[param]
            m_scales = max_norm / np.sqrt(np.sum(M**2.0,axis=1) + 1e-5)
            mask = (m_scales < 1.0)
            mask = mask.astype(np.float32) # why is explicit cast needed?
            m_scales = (m_scales * mask) + (1.0 - mask)
            self.params[param] = M * m_scales[:,np.newaxis]
        return

    def norm_info(self, param_name='Wm'):
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
        assert ((self.bias_dim >= 5) or (self.source_dim >= 5))
        # Record the incoming list of row indices to extract
        self.X = X
        self.C = C.astype(np.uint32)
        # Extract the relevant bias parameter rows
        Wb = self.params['Wb'].take(C, axis=0)
        if (self.bias_dim < 5):
            # No context-adaptive bias term should be applied if self.bias_dim
            # is < 5. I.e. only information coming up from the word LUT, and
            # possibly rescaled by this layer, should be used in prediction.
            Wb = zeros(Wb.shape)
        # Get the feature re-weighting and bias adjustment parameters
        if self.do_rescale:
            Wm = self.params['Wm'].take(C, axis=0)
            self.Wm_exp = ne.evaluate('exp(Wm)', optimization='aggressive')
            self.Wm_sig = self.Wm_exp / (1.0 + self.Wm_exp)
            if (self.source_dim < 5):
                # Information from the word LUT should not pass through this
                # layer. When source_dim < 5, we assume that we are meant to
                # do prediction using only the context-adaptive biases.
                self.Wm_exp = ones(Wm.shape)
                self.Wm_sig = zeros(Wm.shape)
        else:
            self.Wm_sig = ones(X.shape)
        # Modify X by augmenting a multi-dimensional bias and rescaling
        self.Y = np.hstack((Wb, (X * self.Wm_sig)))
        return self.Y

    def backprop(self, dLdY):
        """Backprop through this layer.
        """
        # Add the gradients to the gradient accumulators
        self.grad_idx.update(self.C.ravel())
        self.dLdY = dLdY
        dLdYb, dLdYw = np.hsplit(dLdY, [self.bias_dim])
        dLdYb = dLdYb.copy() # copy, because hsplit leaves the new arrays in
                             # the same memory as the split array, which is
                             # not good for the BLAS calls used by the Cython
                             # version of lut_bp, which expect input arrays
                             # that are in contiguous memory
        if self.do_rescale:
            dLdW = (self.Wm_sig / self.Wm_exp) * self.X * dLdYw
            lut_bp(self.C, dLdW, self.grads['Wm'])
        lut_bp(self.C, dLdYb, self.grads['Wb'])
        dLdX = self.Wm_sig * dLdYw
        return dLdX

    def apply_grad(self, learn_rate=1e-2):
        """Apply the current accumulated gradients, with adagrad."""
        nz_idx = np.asarray([i for i in self.grad_idx]).astype(np.uint32)
        # Information from the word LUT should not pass through this
        # layer when source_dim < 5. In this case, we assume that we
        # will do prediction using only the context-adaptive biases.
        if self.do_rescale:
            m_rate = learn_rate if (self.source_dim >= 5) else 0.0
            ag_update_2d(nz_idx, self.params['Wm'], self.grads['Wm'], \
                         self.moms['Wm'], m_rate)
        # No context-adaptive bias term should be applied if self.bias_dim
        # is < 5. I.e. only information coming up from the word LUT, and
        # possibly rescaled by this layer, should be used in prediction.
        b_rate = learn_rate if (self.bias_dim >= 5) else 0.0
        ag_update_2d(nz_idx, self.params['Wb'], self.grads['Wb'], \
                     self.moms['Wb'], b_rate)
        self.grad_idx = set()
        return

    def l2_regularize(self, lam_Wm=1e-5, lam_Wb=1e-5):
        """Add gradients for l2 regularization."""
        self.params['Wm'] -= lam_Wm * self.params['Wm']
        self.params['Wb'] -= lam_Wb * self.params['Wb']
        return 1

    def reset_moms(self, ada_init=1e-3):
        """Reset the gradient accumulators for this layer."""
        self.moms['Wm'] = (0.0 * self.moms['Wm']) + ada_init
        self.moms['Wb'] = (0.0 * self.moms['Wb']) + ada_init
        return

    def reset_grads_and_moms(self, ada_init=1e-3):
        """Reset the gradient accumulators for this layer."""
        self.grads['Wm'] = (0.0 * self.grads['Wm'])
        self.grads['Wb'] = (0.0 * self.grads['Wb'])
        self.moms['Wm'] = (0.0 * self.moms['Wm']) + ada_init
        self.moms['Wb'] = (0.0 * self.moms['Wb']) + ada_init
        return

    def _cleanup(self):
        """Cleanup temporary feedforward/backprop stuff."""
        self.X = []
        self.Y = []
        self.Wm_exp = []
        self.Wm_sig = []
        self.dLdX = []
        self.dLdY = []
        return

##########################
# NOISE INJECTION LAYERS #
##########################

class GPUNoiseLayer:
    def __init__(self, drop_rate=0.0, fuzz_scale=0.0):
        # Set stuff required for managing this type of layer
        self.dYdX = []
        self.drop_rate = drop_rate
        self.drop_scale = 1.0 / (1.0 - drop_rate)
        self.fuzz_scale = fuzz_scale
        # Set stuff common to all layer types
        self.X = []
        self.Y = []
        self.dLdY = []
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
        # Record (a pointer to) the passed input
        self.X = gp.garray(X)
        # Generate and apply a dropout mask to the input
        if (self.drop_rate > 1e-4):
            drop_mask = self.drop_scale * \
                    (gp.rand((self.X.shape[0], self.X.shape[1])) > self.drop_rate)
        else:
            drop_mask = gp.ones((self.X.shape[0], self.X.shape[1]))
        self.dYdX = drop_mask
        if (self.fuzz_scale > 1e-4):
            fuzz_bump = (self.fuzz_scale / self.drop_scale) * \
                    gp.randn((self.X.shape[0], self.X.shape[1]))
            self.Y = drop_mask * (self.X + fuzz_bump)
        else:
            self.Y = drop_mask * self.X
        return self.Y

    def backprop(self, dLdY, return_on_gpu=False):
        """Perform backprop through this layer.
        """
        # Backprop is just multiplication by the mask from feedforward
        dLdX = gp.garray(dLdY) * self.dYdX
        if not return_on_gpu:
            dLdX = gp.as_numpy_array(dLdX).astype(np.float32)
        return dLdX

    def _cleanup(self):
        """Clear all temp variables for this layer."""
        self.X = []
        self.Y = []
        self.dYdX = []
        return

class NoiseLayer:
    def __init__(self, drop_rate=0.0, fuzz_scale=0.0):
        # Set stuff required for managing this type of layer
        self.dYdX = []
        self.drop_rate = drop_rate
        self.drop_scale = 1.0 / (1.0 - drop_rate)
        self.fuzz_scale = fuzz_scale
        # Set stuff common to all layer types
        self.X = []
        self.Y = []
        self.dLdY = []
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
        # Record (a pointer to) the passed input
        self.X = X
        # Generate and apply a dropout mask to the input
        if (self.drop_rate > 1e-4):
            drop_mask = self.drop_scale * \
                    (npr.rand(self.X.shape[0], self.X.shape[1]) > self.drop_rate)
        else:
            drop_mask = ones((self.X.shape[0], self.X.shape[1]))
        self.dYdX = drop_mask
        if (self.fuzz_scale > 1e-4):
            fuzz_bump = (self.fuzz_scale / self.drop_scale) * \
                    npr.randn(self.X.shape[0], self.X.shape[1])
            self.Y = drop_mask * (self.X + fuzz_bump)
        else:
            self.Y = drop_mask * self.X
        return self.Y.astype(np.float32)

    def backprop(self, dLdY):
        """Perform backprop through this layer.
        """
        # Backprop is just multiplication by the mask from feedforward
        dLdX = dLdY * self.dYdX
        return dLdX.astype(np.float32)

    def _cleanup(self):
        """Clear all temp variables for this layer."""
        self.X = []
        self.Y = []
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
        self.Y = ne.evaluate('tanh(X)', optimization='aggressive')
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
        self.params['Wa'] = 0.01 * randn((self.word_count, word_dim))
        self.params['Wc'] = 0.01 * randn((self.word_count, word_dim))
        self.params['b'] = zeros((self.word_count,))
        self.grads = {}
        self.grads['Wa'] = zeros((self.word_count, word_dim))
        self.grads['Wc'] = zeros((self.word_count, word_dim))
        self.grads['b'] = zeros((self.word_count,))
        self.moms = {}
        self.moms['Wa'] = zeros((self.word_count, word_dim))
        self.moms['Wc'] = zeros((self.word_count, word_dim))
        self.moms['b'] = zeros((self.word_count,))
        # Set l2 regularization parameter
        self.lam_l2 = lam_l2
        # Initialize sets for tracking which words we have trained
        self.trained_Wa = set()
        self.trained_Wc = set()
        return

    def init_params(self, w_scale=0.01, b_scale=0.0):
        """Randomly initialize the weights in this layer."""
        self.params['Wa'] = w_scale * randn((self.word_count, self.word_dim))
        self.grads['Wa'] = zeros((self.word_count, self.word_dim))
        self.moms['Wa'] = zeros((self.word_count, self.word_dim)) + 1e-3
        self.params['Wc'] = w_scale * randn((self.word_count, self.word_dim))
        self.grads['Wc'] = zeros((self.word_count, self.word_dim))
        self.moms['Wc'] = zeros((self.word_count, self.word_dim)) + 1e-3
        self.params['b'] = zeros((self.word_count,))
        self.grads['b'] = zeros((self.word_count,))
        self.moms['b'] = zeros((self.word_count,)) + 1e-3
        return

    def clip_params(self, max_norm=5.0):
        """Bound L2 (row-wise) norm of Wa and Wc by max_norm."""
        for param in ['Wa', 'Wc']:
            M = self.params[param]
            m_scales = max_norm / np.sqrt(np.sum(M**2.0,axis=1) + 1e-5)
            mask = (m_scales < 1.0)
            mask = mask.astype(np.float32) # why is explicit cast needed?
            m_scales = (m_scales * mask) + (1.0 - mask)
            self.params[param] = M * m_scales[:,np.newaxis]
        return

    def batch_train(self, anc_idx, pos_idx, neg_idx, learn_rate=1e-3):
        """Perform a batch update of all parameters based on the given sets
        of anchor/positive example/negative examples indices.
        """
        # Force incoming LUT indices to the right type (i.e. np.uint32)
        anc_idx = anc_idx.astype(np.uint32)
        pos_idx = pos_idx[:,np.newaxis]
        pn_idx = np.hstack((pos_idx, neg_idx)).astype(np.uint32)
        pn_sign = -1.0 * ones(pn_idx.shape)
        pn_sign[:,0] = 1.0
        L = zeros((1,))
        # Do feedforward and backprop through the predictor/predictee tables
        w2v_ff_bp(anc_idx, pn_idx, pn_sign, self.params['Wa'], \
                  self.params['Wc'], self.params['b'], self.grads['Wa'], \
                  self.grads['Wc'], self.grads['b'], L, 1)
        L = L[0]
        # Apply gradients to (touched only) look-up-table parameters
        a_mod_idx = np.unique(anc_idx.ravel())
        c_mod_idx = np.unique(pn_idx.ravel())
        ag_update_2d(a_mod_idx, self.params['Wa'], self.grads['Wa'], \
                self.moms['Wa'], learn_rate)
        ag_update_2d(c_mod_idx, self.params['Wc'], self.grads['Wc'], \
                self.moms['Wc'], learn_rate)
        ag_update_1d(c_mod_idx, self.params['b'], self.grads['b'], \
                self.moms['b'], learn_rate)
        return L

    def batch_test(self, anc_idx, pos_idx, neg_idx):
        """Perform a batch update of all parameters based on the given sets
        of anchor/positive example/negative examples indices.
        """
        anc_idx = anc_idx.astype(np.uint32)
        pos_idx = pos_idx[:,np.newaxis]
        pn_idx = np.hstack((pos_idx, neg_idx)).astype(np.uint32)
        pn_sign = ones(pn_idx.shape)
        pn_sign[:,0] = -1.0
        L = zeros((1,))
        # Do feedforward and backprop through the predictor/predictee tables
        w2v_ff_bp(anc_idx, pn_idx, pn_sign, self.params['Wa'], \
               self.params['Wc'], self.params['b'], self.grads['Wa'], \
               self.grads['Wc'], self.grads['b'], L, 0)
        self.grads['Wa'] = 0.0 * self.grads['Wa']
        self.grads['Wc'] = 0.0 * self.grads['Wc']
        self.grads['b'] = 0.0 * self.grads['b']
        L = L[0]
        return L

    def l2_regularize(self, lam_l2=1e-5):
        """Add gradients for l2 regularization."""
        self.params['Wa'] -= lam_l2 * self.params['Wa']
        self.params['Wc'] -= lam_l2 * self.params['Wc']
        return

    def reset_moms(self, ada_init=1e-3):
        """Reset the gradient accumulators for this layer."""
        self.moms['Wa'] = (0.0 * self.moms['Wa']) + ada_init
        self.moms['Wc'] = (0.0 * self.moms['Wc']) + ada_init
        self.moms['b'] = (0.0 * self.moms['b']) + ada_init
        return

    def reset_grads_and_moms(self, ada_init=1e-3):
        """Reset the gradient accumulators for this layer."""
        self.grads['Wa'] = (0.0 * self.grads['Wa']) + ada_init
        self.grads['Wc'] = (0.0 * self.grads['Wc']) + ada_init
        self.grads['b'] = (0.0 * self.grads['b']) + ada_init
        self.moms['Wa'] = (0.0 * self.moms['Wa']) + ada_init
        self.moms['Wc'] = (0.0 * self.moms['Wc']) + ada_init
        self.moms['b'] = (0.0 * self.moms['b']) + ada_init
        return

###################################
# TEST BASIC MODULE FUNCTIONALITY #
###################################

def run_test():
    #########################################################
    # TODO: write new tests that don't depend on STB files. #
    #########################################################
    print("TODO: WRITE TEST FOR Word2Vec.py")


if __name__ == '__main__':
    run_test()










##############
# EYE BUFFER #
##############
