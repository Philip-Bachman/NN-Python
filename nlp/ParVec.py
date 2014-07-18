from __future__ import absolute_import

import numpy as np
import numpy.random as npr
import gnumpy as gp
import threading
import numba
from math import exp, log, pow as mpow, sqrt

from time import clock
from numba import jit, void, i4, f8
from ctypes import pythonapi, c_void_p

########################################
# MULTITHREADING HELPER-FUNC AND DEFNS #
########################################

THREAD_NUM = 6

savethread = pythonapi.PyEval_SaveThread
savethread.argtypes = []
savethread.restype = c_void_p

restorethread = pythonapi.PyEval_RestoreThread
restorethread.argtypes = [c_void_p]
restorethread.restype = None

def make_multithread(inner_func, numthreads):
    def func_mt(*args):
        length = len(args[0])
        sp_idx = np.arange(0,length).astype(np.int32)
        chunklen = (length + 1) // numthreads
        chunkargs = [(sp_idx[i*chunklen:(i+1)*chunklen],)+args for i in range(numthreads)]
        # Start a thread for all but the last chunk of work
        threads = [threading.Thread(target=inner_func, args=cargs)
                   for cargs in chunkargs[:-1]]
        for thread in threads:
            thread.start()
        # Give the last chunk of work to the main thread
        inner_func(*chunkargs[-1])
        for thread in threads:
            thread.join()
        return 1
    return func_mt

##############################
# NUMBA FUNCTION DEFINITIONS #
##############################

def ag_update_2d_sp(sp_idx, row_idx, W, dW, mW, learn_rate, ada_smooth, lam_l2):
    """Row-wise partial update ala adagrad.

    This updates params and adagrad "momentums", and then zeros the grads.
    """
    threadstate = savethread()
    row_count = sp_idx.shape[0]
    vec_dim = W.shape[1]
    for spi in range(row_count):
        idx = row_idx[sp_idx[spi]]
        for j in range(vec_dim):
            dW[idx,j] += (lam_l2 * W[idx,j])
            mW[idx,j] += mpow(dW[idx,j], 4)
            W[idx,j] -= (learn_rate * (dW[idx,j] / (sqrt(mW[idx,j]) + ada_smooth)))
            dW[idx,j] = 0.0
    restorethread(threadstate)
    return
fn_sig_4 = void(i4[:], i4[:], f8[:,:], f8[:,:], f8[:,:], f8, f8, f8)
ag_update_2d_st = jit(fn_sig_4, nopython=True)(ag_update_2d_sp)
ag_update_2d = make_multithread(ag_update_2d_st, THREAD_NUM)

@numba.jit("void(i4[:], f8[:], f8[:], f8[:], f8, f8, f8)")
def ag_update_1d(row_idx, W, dW, mW, learn_rate, ada_smooth, lam_l2):
    """Element-wise partial update ala adagrad.

    This updates params and adagrad "momentums", and then zeros the grads.
    """
    row_count = row_idx.shape[0]
    for i in range(row_count):
        idx = row_idx[i]
        dW[idx] += lam_l2 * W[idx]
        mW[idx] += mpow(dW[idx], 2)
        W[idx] -= learn_rate * (dW[idx] / (sqrt(mW[idx]) + ada_smooth))
        dW[idx] = 0.0
    return

def lut_sp(sp_idx, row_idx, dLdY, dW):
    """Simple row-wise updates for adjusting dW with dLdY.

    This adds each row of dLdY to some row of dW. The row of dW to adjust
    is given by the corresponding item in row_idx.
    """
    threadstate = savethread()
    row_count = sp_idx.shape[0]
    vec_dim = dW.shape[1]
    for i in range(row_count):
        idx = row_idx[sp_idx[i]]
        for j in range(vec_dim):
            dW[idx,j] += dLdY[i,j]
    restorethread(threadstate)
    return
fn_sig_5 = void(i4[:], i4[:], f8[:,:], f8[:,:])
lut_st = jit(fn_sig_5, nopython=True)(lut_sp)
lut_bp = make_multithread(lut_st, THREAD_NUM)

###########################################
# WORD VALIDATION, FOR MANAGING OOV WORDS #
###########################################

def catch_oov_words(w_idx, v_idx, oov_idx):
    """w_idx is an ndarray, v_idx is a set, and oov_idx is an int32."""
    assert((w_idx.ndim == 1) or (w_idx.ndim == 2))
    # Brute-force convert untrained words to OOV word
    if (w_idx.ndim == 1):
        for i in range(w_idx.shape[0]):
            if not (w_idx[i] in v_idx):
                w_idx[i] = oov_idx
    else:
        for i in range(w_idx.shape[0]):
            for j in range(w_idx.shape[1]):
                if not (w_idx[i,j] in v_idx):
                    w_idx[i,j] = oov_idx
    return w_idx

##################################
# FULLY-CONNECTED SOFTMAX LAYERS #
##################################

class GPUFullLayer:
    def __init__(self, in_dim=0, out_dim=0):
        # Set stuff for managing this type of layer
        self.dim_input = in_dim
        self.dim_output = out_dim
        self.params = {}
        self.params['W'] = gp.randn((in_dim, out_dim))
        self.params['b'] = gp.zeros((1, out_dim))
        self.grads = {}
        self.grads['W'] = gp.zeros((in_dim, out_dim))
        self.grads['b'] = gp.zeros((1, out_dim))
        self.moms = {}
        self.moms['W'] = gp.zeros((in_dim, out_dim))
        self.moms['b'] = gp.zeros((1, out_dim))
        self.max_norm = 10.0
        self.comp_time = 0.0
        # Set common stuff for all types layers
        self.X = []
        self.Y = []
        self.dLdX = []
        self.dLdY = []
        self.trained_idx = set()
        return

    def init_params(self, w_scale=0.01, b_scale=0.0):
        """Randomly initialize the weights in this layer."""
        self.params['W'] = w_scale * gp.randn((self.dim_input, self.dim_output))
        self.grads['W'] = gp.zeros((self.dim_input, self.dim_output))
        self.params['b'] = gp.zeros((1, self.dim_output))
        self.grads['b'] = gp.zeros((1, self.dim_output))
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

    def feedforward(self, X):
        """Run feedforward for this layer."""
        # Cleanup debris from any previous feedforward
        self._cleanup()
        # Do new feedforward...
        self.X = gp.garray(X)
        self.Y = gp.dot(self.X, self.params['W']) + self.params['b']
        return self.Y

    def _backprop_(self, dLdY):
        """Backprop through this layer."""
        self.dLdY = gp.garray(dLdY)
        # Compute gradient with respect to layer parameters
        dLdW = gp.dot(self.X.T, self.dLdY)
        dLdb = gp.sum(self.dLdY, axis=0)
        dLdb = dLdb[gp.newaxis,:]
        self.grads['W'] += dLdW
        self.grads['b'] += dLdb
        # Compute gradient with respect to layer input
        self.dLdX = gp.dot(self.dLdY, self.params['W'].T)
        return self.dLdX

    def backprop(self, Y_cat):
        """Backprop through this layer."""
        Y_cat = Y_cat.astype(np.int32)
        Y_ind = np.zeros(self.Y.shape)
        Y_ind[np.arange(Y_ind.shape[0]), Y_cat] = 1.0
        dLdY_bp = self.cross_entropy(self.Y, Y_ind)
        self._backprop_(dLdY_bp)
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
        self.params['W'] -= lam_l2 * self.params['W']
        self.params['b'] -= lam_l2 * self.params['b']
        return

    def apply_grad_reg(self, learn_rate=1e-2, ada_smooth=1e-3, lam_l2=0.0):
        """Apply the current accumulated gradients, with adagrad."""
        self.grads['W'] += lam_l2 * self.params['W']
        self.grads['b'] += lam_l2 * self.params['b']
        self.moms['W'] += self.grads['W']**2.0
        self.moms['b'] += self.grads['b']**2.0
        self.params['W'] -= learn_rate * (self.grads['W'] / \
                (gp.sqrt(self.moms['W']) + ada_smooth))
        self.params['b'] -= learn_rate * (self.grads['b'] / \
                (gp.sqrt(self.moms['b']) + ada_smooth))
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
        self.Y = []
        self.dLdX = []
        self.dLdY = []
        return

class FullLayer:
    def __init__(self, in_dim=0, out_dim=0):
        # Set stuff for managing this type of layer
        self.dim_input = in_dim
        self.dim_output = out_dim
        self.params = {}
        self.params['W'] = npr.randn(in_dim, out_dim)
        self.params['b'] = np.zeros((1, out_dim))
        self.grads = {}
        self.grads['W'] = np.zeros((in_dim, out_dim))
        self.grads['b'] = np.zeros((1, out_dim))
        self.moms = {}
        self.moms['W'] = np.zeros((in_dim, out_dim))
        self.moms['b'] = np.zeros((1, out_dim))
        self.max_norm = 10.0
        self.comp_time = 0.0
        # Set common stuff for all types layers
        self.X = []
        self.Y = []
        self.dLdX = []
        self.dLdY = []
        return

    def init_params(self, w_scale=0.01, b_scale=0.0):
        """Randomly initialize the weights in this layer."""
        self.params['W'] = w_scale * npr.randn(self.dim_input, self.dim_output)
        self.grads['W'] = np.zeros((self.dim_input, self.dim_output))
        self.params['b'] = np.zeros((1, self.dim_output))
        self.grads['b'] = np.zeros((1, self.dim_output))
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

    def feedforward(self, X):
        """Run feedforward for this layer."""
        # Cleanup debris from any previous feedforward
        self._cleanup()
        # Do new feedforward...
        self.X = X
        self.Y = np.dot(self.X, self.params['W']) + self.params['b']
        self.dLdY = np.zeros(self.Y.shape)
        return self.Y

    def _backprop_(self, dLdY_bp):
        """Backprop through this layer."""
        self.dLdY = self.dLdY + dLdY_bp
        # Compute gradient with respect to layer parameters
        dLdW = np.dot(self.X.T, self.dLdY)
        dLdb = np.sum(self.dLdY, axis=0, keepdims=True)
        self.grads['W'] = self.grads['W'] + dLdW
        self.grads['b'] = self.grads['b'] + dLdb
        # Compute gradient with respect to layer input
        self.dLdX = np.dot(self.dLdY, self.params['W'].T)
        return self.dLdX

    def backprop_sm(self, Y_cat):
        """Backprop through this layer."""
        Y_cat = Y_cat.astype(np.int32)
        Y_ind = np.zeros(self.Y.shape)
        Y_ind[np.arange(Y_ind.shape[0]), Y_cat] = 1.0
        dLdY_bp = self.cross_entropy(self.Y, Y_ind)
        self._backprop_(dLdY_bp)
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
        self.params['W'] -= lam_l2 * self.params['W']
        self.params['b'] -= lam_l2 * self.params['b']
        return

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
        self.Y = []
        self.dLdX = []
        self.dLdY = []
        return

###################################################
# CONTEXT LAYER (STACKS CONTEXT AND WORD VECTORS) #
###################################################

class ContextLayer:
    def __init__(self, word_keys, word_dim, context_keys, context_dim):
        #  Add 1s to account for the OOV tokens.
        word_keys = word_keys + 1
        context_keys = context_keys + 1
        # We need param vectors for each possible word and each possible
        # context, as well as their gradients and adagrad "momentums".
        self.params = {}
        self.params['Vw'] = npr.randn(word_keys, word_dim)
        self.params['Vc'] = npr.randn(context_keys, context_dim)
        self.grads = {}
        self.grads['Vw'] = np.zeros(self.params['Vw'].shape)
        self.grads['Vc'] = np.zeros(self.params['Vc'].shape)
        self.moms = {}
        self.moms['Vw'] = np.zeros(self.params['Vw'].shape)
        self.moms['Vc'] = np.zeros(self.params['Vc'].shape)
        # Record the dimensions of our representation spaces
        self.word_keys = word_keys
        self.word_dim = word_dim
        self.cont_keys = context_keys
        self.cont_dim = context_dim
        # Create sets to track which keys' vectors have been trained
        self.grad_idx_w = set()
        self.trained_idx_w = set()
        self.grad_idx_c = set()
        self.trained_idx_c = set()
        self.max_norm = 10.0
        # Set temp vars to use during feedforward and backprop
        self.Iw = []
        self.Ic = []
        self.Y = []
        return

    def init_params(self, w_scale=0.01):
        """Randomly initialize the weights in this layer."""
        self.params['Vw'] = w_scale * npr.randn(self.word_keys, self.word_dim)
        self.params['Vc'] = w_scale * npr.randn(self.cont_keys, self.cont_dim)
        self.grads['Vw'] = np.zeros(self.params['Vw'].shape)
        self.grads['Vc'] = np.zeros(self.params['Vc'].shape)
        return

    def clip_params(self):
        """Bound L2 (row-wise) norm of self.params['W'] by wt_bnd."""
        EPS = 1e-5
        for param in ['Vw', 'Vc']:
            W = self.params[param]
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
            self.params[param] = W
        return

    def feedforward(self, Iw, Ic, test=False):
        """Run feedforward for this layer. Using sacks of LUT indices.
        """
        # Cleanup debris from any previous feedforward
        self._cleanup()
        obs_count, pre_words = Iw.shape
        # Record the incoming list of row indices to extract
        self.Iw = Iw.astype(np.int32)
        self.Ic = Ic.astype(np.int32)
        # Handle OOV if testing (for both words and contexts)
        if test:
            oov_idx = (self.word_keys-1) * np.ones((1,)).astype(np.int32)
            self.Iw = catch_oov_words(self.Iw, self.trained_idx_w, oov_idx[0])
            oov_idx = (self.cont_keys-1) * np.ones((1,)).astype(np.int32)
            self.Ic = catch_oov_words(self.Ic, self.trained_idx_c, oov_idx[0])
        # Construct the output of this layer using table look-ups
        self.Y = np.zeros((obs_count,self.cont_dim+(pre_words*self.word_dim)))
        self.Y[:,0:self.cont_dim] = self.params['Vc'][self.Ic,:]
        for i in range(pre_words):
            s_idx = self.cont_dim + (i * self.word_dim)
            e_idx = s_idx + self.word_dim
            self.Y[:,s_idx:e_idx] = self.params['Vw'][self.Iw[:,i]]
        return self.Y

    def backprop(self, dLdY):
        """Backprop through this layer.
        """
        obs_count, pre_words = self.Iw.shape
        self.grad_idx_w.update(self.Iw.ravel())
        self.grad_idx_c.update(self.Ic.ravel())
        # Backprop for the context vectors
        lut_bp(self.Ic, dLdY[:,0:self.cont_dim], self.grads['Vc'])
        # Backprop for each of the predictor words
        for i in range(pre_words):
            s_idx = self.cont_dim + (i * self.word_dim)
            e_idx = s_idx + self.word_dim
            lut_bp(self.Iw[:,i], dLdY[:,s_idx:e_idx], self.grads['Vw'])
        return

    def l2_regularize(self, lam_l2=1e-5):
        """Add gradients for l2 regularization. And compute loss."""
        self.params['Vw'] -= lam_l2 * self.params['Vw']
        self.params['Vc'] -= lam_l2 * self.params['Vc']
        return

    def apply_grad_reg(self, learn_rate=1e-2, ada_smooth=1e-3, lam_l2=0.0):
        """Apply the current accumulated gradients, with adagrad."""
        self.trained_idx_w.update(self.grad_idx_w)
        self.trained_idx_c.update(self.grad_idx_c)
        nz_idx = np.asarray([i for i in self.grad_idx_w]).astype(np.int32)
        ag_update_2d(nz_idx, self.params['Vw'], self.grads['Vw'], \
                     self.moms['Vw'], learn_rate, ada_smooth, lam_l2)
        self.params['Vw'][-1,:] = 0.0
        nz_idx = np.asarray([i for i in self.grad_idx_c]).astype(np.int32)
        ag_update_2d(nz_idx, self.params['Vc'], self.grads['Vc'], \
                     self.moms['Vc'], learn_rate, ada_smooth, lam_l2)
        self.params['Vc'][-1,:] = 0.0
        self.grad_idx_w = set()
        self.grad_idx_c = set()
        return

    def reset_moms(self, ada_init=1e-3):
        """Reset the gradient accumulators for this layer."""
        self.moms['Vw'] = (0.0 * self.moms['Vw']) + ada_init
        self.moms['Vc'] = (0.0 * self.moms['Vc']) + ada_init
        return

    def _cleanup(self):
        """Cleanup temporary feedforward/backprop stuff."""
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
        self.dLdX = []
        self.dLdY = []
        return

    def set_noise_params(self, drop_rate=0.0, fuzz_scale=0.0):
        """Set the drop rate for this drop layer."""
        self.drop_rate = drop_rate
        self.drop_scale = 1.0 / (1.0 - drop_rate)
        self.fuzz_scale = fuzz_scale
        return

    def feedforward(self, X, test=False):
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
            drop_mask = np.ones((self.X.shape[0], self.X.shape[1]))
        self.dYdX = drop_mask
        if (self.fuzz_scale > 1e-4):
            fuzz_bump = (self.fuzz_scale / self.drop_scale) * \
                    npr.randn(self.X.shape[0], self.X.shape[1])
            self.Y = drop_mask * (self.X + fuzz_bump)
        else:
            self.Y = drop_mask * self.X
        return self.Y

    def backprop(self, dLdY):
        """Perform backprop through this layer.
        """
        # Backprop is just multiplication by the mask from feedforward
        self.dLdX = dLdY * self.dYdX
        return self.dLdX

    def _cleanup(self):
        """Clear all temp variables for this layer."""
        self.X = []
        self.Y = []
        self.dYdX = []
        self.dLdX = []
        return

##################################
# WORD AND PHRASE SAMPLING STUFF #
##################################

def rand_word_seqs(phrase_list, seq_count, seq_len, non_idx):
    """Sample random sequences of contiguous words from a list of phrases.

    Given a list of phrases, where each phrase is described by a list of
    indices into a look-up-table, sample random subsequences of the phrases,
    for use in training a forward prediction n-gram model (or whatever).
    """
    phrase_count = len(phrase_list)
    seq_idx = np.zeros((seq_count, seq_len), dtype=np.int32)
    phrase_idx = np.zeros((seq_count,), dtype=np.int32)
    for i in range(seq_count):
        phrase_idx[i] = npr.randint(0, phrase_count)
        phrase = phrase_list[phrase_idx[i]]
        phrase_len = len(phrase)
        predictee_idx = npr.randint(0, phrase_len)
        seq_idx[i,-1] = predictee_idx
        for j in range(seq_len-1):
            predictor_idx = predictee_idx - seq_len + j + 1
            if (predictor_idx < 0):
                seq_idx[i,j] = non_idx
            else:
                seq_idx[i,j] = phrase[predictor_idx]
    return [seq_idx, phrase_idx]

def run_test():
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
    # Record maximum required keys for the context layer's tables
    max_word_key = max(stb_data['lut_keys'].values()) + 1
    max_context_key = len(tr_phrases) - 1

    # Set some simple hyperparameters for training
    non_word_key = max_word_key
    batch_count = 500001
    batch_size = 256
    pre_words = 5
    word_dim = 200
    context_dim = 100
    lam_l2 = 1e-3

    # Create a lookup table for word representations
    context_layer = ContextLayer(max_word_key, word_dim, max_context_key, context_dim)
    noise_layer = NoiseLayer(drop_rate=0.5, fuzz_scale=0.025)
    class_layer = GPUFullLayer(in_dim=(pre_words*word_dim + context_dim), \
                               out_dim=max_word_key)

    # Initialize params for the LUT and softmax classifier
    context_layer.init_params(0.05)
    class_layer.init_params(0.05)

    print("Processing batches:")
    L = 0.0
    for b in range(batch_count):
        # Sample a batch of random anchor/context prediction pairs for
        # training a skip-gram model.
        [seq_idx, phrase_idx] = \
            rand_word_seqs(tr_phrases, batch_size, pre_words+1, non_word_key)
        predictor_idx = seq_idx[:,:-1]
        predictee_idx = seq_idx[:,-1]

        # Feedforward through look-up-table and classifier layers
        Xb = context_layer.feedforward(predictor_idx, phrase_idx, test=False)
        Xn = noise_layer.feedforward(Xb)
        Yn = class_layer.feedforward(Xn)
        L += class_layer.check_loss(Yn, predictee_idx)

        # Backprop through classifier and look-up-table layers
        dLdXn = class_layer.backprop(predictee_idx).as_numpy_array()
        dLdXb = noise_layer.backprop(dLdXn)
        context_layer.backprop(dLdXb)

        # Apply gradients computed during backprop
        class_layer.apply_grad_reg(learn_rate=1e-4, ada_smooth=1e-3, lam_l2=lam_l2)
        context_layer.apply_grad_reg(learn_rate=1e-4, ada_smooth=1e-3, lam_l2=lam_l2)

        # Compute and display loss from time-to-time (for diagnostics)
        if ((b % 100) == 0):
            obs_count = 100.0 * batch_size
            print("Batch {0:d}, loss {1:.4f}".format(b, (L / obs_count)))
            L = 0.0

        # Reset adagrad smoothing factors from time-to-time
        if ((b > 1) and ((b % 10000) == 0)):
            class_layer.reset_moms()
            context_layer.reset_moms()

if __name__ == '__main__':
    run_test()



##############
# EYE BUFFER #
##############
