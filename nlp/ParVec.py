from __future__ import absolute_import

# Imports of public stuff
import numpy as np
import numpy.random as npr
import gnumpy as gp

# Imports of my stuff
import HelperFuncs as hf
from HelperFuncs import randn, ones, zeros
from NumbaFuncs import ag_update_2d, ag_update_1d, lut_bp
from CythonFuncs import ag_update_2d, ag_update_1d, lut_bp

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
        L, dLdY = self.xent_loss_and_grad(self.Y, Y_cat.astype(np.int32))
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

    def apply_grad(self, learn_rate=1e-2, ada_smooth=1e-3):
        """Apply the current accumulated gradients, with adagrad."""
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

###################################################
# CONTEXT LAYER (STACKS CONTEXT AND WORD VECTORS) #
###################################################

class ContextLayer:
    def __init__(self, max_word_key, word_dim, max_context_key, context_dim):
        #  Add 1s to accommodate 0 indexing.
        word_keys = max_word_key + 1
        context_keys = max_context_key + 1
        # We need param vectors for each trainable word and each trainable
        # context, as well as their gradients and adagrad "momentums". Note
        # that trainable words/contexts are those with LUT keys referencing
        # rows up to (and including) the penultimate rows of their LUTs.
        self.params = {}
        self.params['W'] = 0.01 * randn((word_keys, word_dim))
        self.params['C'] = 0.01 * randn((context_keys, context_dim))
        self.grads = {}
        self.grads['W'] = zeros(self.params['W'].shape)
        self.grads['C'] = zeros(self.params['C'].shape)
        self.moms = {}
        self.moms['W'] = zeros(self.params['W'].shape)
        self.moms['C'] = zeros(self.params['C'].shape)
        # Record the sizes of our word and context LUTs
        self.word_keys = word_keys
        self.word_dim = word_dim
        self.cont_keys = context_keys
        self.cont_dim = context_dim
        # Create sets to track which word/context vectors we have trained
        self.grad_idx_w = set()
        self.grad_idx_c = set()
        # Set temp vars to use during feedforward and backprop
        self.Iw = []
        self.Ic = []
        self.Y = []
        return

    def init_params(self, w_scale=0.01):
        """Randomly initialize the weights in this layer."""
        self.params['W'] = w_scale * randn((self.word_keys, self.word_dim))
        self.params['C'] = w_scale * randn((self.cont_keys, self.cont_dim))
        self.grads['W'] = zeros(self.params['W'].shape)
        self.grads['C'] = zeros(self.params['C'].shape)
        return

    def clip_params(self, W_norm=5.0, C_norm=5.0):
        """Bound L2 (row-wise) norm of W & C by W_norm & C_norm."""
        for (param, max_norm) in zip(['W', 'C'], [W_norm, C_norm]):
            M = self.params[param]
            m_scales = max_norm / np.sqrt(np.sum(M**2.0,axis=1) + 1e-5)
            mask = (m_scales < 1.0)
            mask = mask.astype(np.float32) # why is explicit cast needed?
            m_scales = (m_scales * mask) + (1.0 - mask)
            self.params[param] = M * m_scales[:,np.newaxis]
        return

    def feedforward(self, Iw, Ic):
        """Run feedforward for this layer. Using sacks of LUT keys.
        """
        # Cleanup debris from any previous feedforward
        self._cleanup()
        obs_count, pre_words = Iw.shape
        # Record the incoming lists of rows to extract from each LUT
        self.Iw = Iw.astype(np.int32)
        self.Ic = Ic.astype(np.int32)
        # Construct the output of this layer using table look-ups
        self.Y = zeros((obs_count,self.cont_dim+(pre_words*self.word_dim)))
        self.Y[:,0:self.cont_dim] = self.params['C'][self.Ic,:]
        for i in range(pre_words):
            s_idx = self.cont_dim + (i * self.word_dim)
            e_idx = s_idx + self.word_dim
            self.Y[:,s_idx:e_idx] = self.params['W'][self.Iw[:,i]]
        return self.Y

    def backprop(self, dLdY):
        """Backprop through this layer.
        """
        obs_count, pre_words = self.Iw.shape
        self.grad_idx_w.update(self.Iw.ravel())
        self.grad_idx_c.update(self.Ic.ravel())
        # Backprop for the context vectors
        lut_bp(self.Ic, dLdY[:,0:self.cont_dim], self.grads['C'])
        # Backprop for each of the predictor words
        for i in range(pre_words):
            s_idx = self.cont_dim + (i * self.word_dim)
            e_idx = s_idx + self.word_dim
            lut_bp(self.Iw[:,i], dLdY[:,s_idx:e_idx], self.grads['W'])
        return

    def apply_grad(self, learn_rate=1e-3, train_context=False, \
                   train_other=False, ada_smooth=1e-3):
        """Apply the current accumulated gradients, adagrad style."""
        # Find which LUT keys point to params with pending updates
        nz_idx_w = np.asarray([i for i in self.grad_idx_w]).astype(np.int32)
        nz_idx_c = np.asarray([i for i in self.grad_idx_c]).astype(np.int32)
        # Update the params for words/contexts with pending updates
        other_rate = 0.0
        context_rate = 0.0
        if train_other:
            other_rate = learn_rate
        if train_context:
            context_rate = learn_rate
        ag_update_2d(nz_idx_w, self.params['W'], self.grads['W'], \
                     self.moms['W'], other_rate, ada_smooth)
        ag_update_2d(nz_idx_c, self.params['C'], self.grads['C'], \
                     self.moms['C'], context_rate, ada_smooth)
        self.grad_idx_w = set()
        self.grad_idx_c = set()
        return

    def l2_regularize(self, lam_w=1e-5, lam_c=1e-5):
        """Apply some amount of l2 "shrinkage" to word/context params."""
        self.params['W'] -= lam_w * self.params['W']
        self.params['C'] -= lam_c * self.params['C']
        return

    def reset_moms(self, ada_init=1e-3):
        """Reset the adagrad "momentums" for this layer."""
        self.moms['W'] = (0.0 * self.moms['W']) + ada_init
        self.moms['C'] = (0.0 * self.moms['C']) + ada_init
        return

    def reset_grads(self):
        """Reset the gradient accumulators for this layer."""
        self.grads['W'] = (0.0 * self.grads['W'])
        self.grads['C'] = (0.0 * self.grads['C'])
        self.grad_idx_w = set()
        self.grad_idx_c = set()
        return

    def _cleanup(self):
        """Cleanup temp vars used in feedforward/backprop."""
        self.Iw = []
        self.Ic = []
        self.Y = []
        return

#########################
# NOISE INJECTION LAYER #
#########################

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
        self.X = gp.garray(X)
        # Generate and apply a dropout mask to the input
        if (self.drop_rate > 1e-4):
            drop_mask = self.drop_scale * \
                    (gp.rand(self.X.shape[0], self.X.shape[1]) > self.drop_rate)
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

###################################
# TEST BASIC MODULE FUNCTIONALITY #
###################################

def run_test():
    #########################################################
    # TODO: write new tests that don't depend on STB files. #
    #########################################################
    print("TODO: WRITE TEST FOR ParVec.py")

if __name__ == '__main__':
    run_test()



##############
# EYE BUFFER #
##############
