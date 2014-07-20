from __future__ import absolute_import

# Imports of public stuff
import numpy as np
import numpy.random as npr
import gnumpy as gp
import numba

# Imports of my stuff
import HelperFuncs as hf
from HelperFuncs import ag_update_2d, ag_update_1d, lut_bp, catch_oov_words

#################################
# FULLY-CONNECTED SOFTMAX LAYER #
#################################

class FullLayer:
    def __init__(self, in_dim=0, out_dim=0):
        # Set dimension of incoming vectors and the number of outcomes for
        # which to perform prediction. Increment the requested prediction size
        # by 1, to accommodate an OOV prediction.
        self.dim_input = in_dim
        self.dim_output = out_dim + 1
        # Initialize parameters, gradients, and adagrad "momentums"
        self.params = {}
        self.params['W'] = gp.randn((in_dim, out_dim))
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
        # self.trained_idx tracks which prediction vectors we have trained
        self.trained_idx = set()
        return

    def init_params(self, w_scale=0.01, b_scale=0.0):
        """Randomly initialize the weights in this layer."""
        self.params['W'] = w_scale * gp.randn((self.dim_input, self.dim_output))
        self.grads['W'] = gp.zeros((self.dim_input, self.dim_output))
        self.params['b'] = gp.zeros((1, self.dim_output))
        self.grads['b'] = gp.zeros((1, self.dim_output))
        # Zero-out parameters for the OOV prediction
        self.params['W'][-1,:] = 0.0
        self.params['b'][1,-1] = 0.0
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

    def feedforward(self, X, test=False):
        """Run feedforward for this layer."""
        # Cleanup debris from any previous feedforward
        self._cleanup()
        # Do new feedforward...
        self.X = gp.garray(X)
        self.Y = gp.dot(self.X, self.params['W']) + self.params['b']
        return self.Y

    def backprop(self, Y_cat, return_on_gpu=False, test=False):
        """Backprop through softmax using the given target predictions."""
        Y_cat = Y_cat.astype(np.int32)
        # Handle untrained (or otherwise unknown) targets when testing
        if test:
            oov_idx = (self.dim_output-1) * np.ones((1,)).astype(np.int32)
            Y_cat = catch_oov_words(Y_cat, self.trained_idx, oov_idx[0])
        else:
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
        # Update the set of trained prediction vectors
        self.trained_idx.update([i for i in self.Y_cat.ravel()])
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
        # Zero-out parameters for the OOV prediction
        self.params['W'][-1,:] = 0.0
        self.params['b'][1,-1] = 0.0
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
    def __init__(self, word_keys, word_dim, context_keys, context_dim):
        #  Add 1s to accommodate OOV tokens (for words _and_ contexts).
        word_keys = word_keys + 1
        context_keys = context_keys + 1
        # We need param vectors for each trainable word and each trainable
        # context, as well as their gradients and adagrad "momentums". Note
        # that trainable words/contexts are those with LUT keys referencing
        # rows up to (and including) the penultimate rows of their LUTs.
        self.params = {}
        self.params['Vw'] = npr.randn(word_keys, word_dim)
        self.params['Vc'] = npr.randn(context_keys, context_dim)
        self.grads = {}
        self.grads['Vw'] = np.zeros(self.params['Vw'].shape)
        self.grads['Vc'] = np.zeros(self.params['Vc'].shape)
        self.moms = {}
        self.moms['Vw'] = np.zeros(self.params['Vw'].shape)
        self.moms['Vc'] = np.zeros(self.params['Vc'].shape)
        # Record the sizes of our word and context LUTs
        self.word_keys = word_keys
        self.word_dim = word_dim
        self.cont_keys = context_keys
        self.cont_dim = context_dim
        # Create sets to track which word/context vectors we have trained
        self.grad_idx_w = set()
        self.trained_idx_w = set()
        self.grad_idx_c = set()
        self.trained_idx_c = set()
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
        # Zero-out the final word/context vector (for use as OOV)
        self.params['Vw'][-1,:] = 0.0
        self.params['Vc'][-1,:] = 0.0
        return

    def clip_params(self, Vw_norm=10.0, Vc_norm=10.0):
        """Bound L2 (row-wise) norm of self.params['W'] by wt_bnd."""
        for (param, max_norm) in zip(['Vw', 'Vc'], [Vw_norm, Vc_norm]):
            W = self.params[param]
            w_norms = np.sqrt(np.sum(W**2.0,axis=1) + 1e-5)
            w_scales = max_norm / w_norms
            mask = (w_scales < 1.0)
            w_scales = (w_scales * mask) + (1.0 - mask)
            w_scales = w_scales[:,np.newaxis]
            W = W * w_scales
            self.params[param] = W
        return

    def feedforward(self, Iw, Ic, test=False):
        """Run feedforward for this layer. Using sacks of LUT keys.
        """
        # Cleanup debris from any previous feedforward
        self._cleanup()
        obs_count, pre_words = Iw.shape
        # Record the incoming lists of rows to extract from each LUT
        self.Iw = Iw.astype(np.int32)
        self.Ic = Ic.astype(np.int32)
        # Handle OOV keys if testing (for words _and_ contexts)
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

    def backprop(self, dLdY, test=False):
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

    def apply_grad_reg(self, learn_rate=1e-2, ada_smooth=1e-3, lam_l2=0.0):
        """Apply the current accumulated gradients, adagrad style."""
        # Update the sets of LUT keys tracking which params we have trained
        self.trained_idx_w.update(self.grad_idx_w)
        self.trained_idx_c.update(self.grad_idx_c)
        # Find which LUT keys point to params with pending updates
        nz_idx_w = np.asarray([i for i in self.grad_idx_w]).astype(np.int32)
        nz_idx_c = np.asarray([i for i in self.grad_idx_c]).astype(np.int32)
        # Update the params for words/contexts with pending updates
        ag_update_2d(nz_idx_w, self.params['Vw'], self.grads['Vw'], \
                     self.moms['Vw'], learn_rate, ada_smooth, lam_l2)
        ag_update_2d(nz_idx_c, self.params['Vc'], self.grads['Vc'], \
                     self.moms['Vc'], learn_rate, ada_smooth, lam_l2)
        # Zero-out the final word/context vector (for use as OOV)
        self.params['Vw'][-1,:] = 0.0
        self.params['Vc'][-1,:] = 0.0
        # Reset the sets of LUT keys for parameters with pending updates
        self.grad_idx_w = set()
        self.grad_idx_c = set()
        return

    def l2_regularize(self, lam_l2=1e-5):
        """Apply some amount of l2 "shrinkage" to word/context params."""
        self.params['Vw'] -= lam_l2 * self.params['Vw']
        self.params['Vc'] -= lam_l2 * self.params['Vc']
        return

    def reset_moms(self, ada_init=1e-3):
        """Reset the adagrad "momentums" for this layer."""
        self.moms['Vw'] = (0.0 * self.moms['Vw']) + ada_init
        self.moms['Vc'] = (0.0 * self.moms['Vc']) + ada_init
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

###################################
# TEST BASIC MODULE FUNCTIONALITY #
###################################

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
    word_dim = 100
    context_dim = 50
    lam_l2 = 1e-3

    # Create a lookup table for word representations
    context_layer = ContextLayer(max_word_key, word_dim, max_context_key, context_dim)
    noise_layer = NoiseLayer(drop_rate=0.5, fuzz_scale=0.025)
    class_layer = FullLayer(in_dim=(pre_words*word_dim + context_dim), \
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
            hf.rand_word_seqs(tr_phrases, batch_size, pre_words+1, non_word_key)
        predictor_idx = seq_idx[:,:-1]
        predictee_idx = seq_idx[:,-1]

        # Feedforward through look-up-table and classifier layers
        Xb = context_layer.feedforward(predictor_idx, phrase_idx, test=False)
        Xn = noise_layer.feedforward(Xb)
        Yn = class_layer.feedforward(Xn)
        L += class_layer.cross_entropy_loss(Yn, predictee_idx)

        # Backprop through classifier and look-up-table layers
        dLdXn = class_layer.backprop(predictee_idx, return_on_gpu=False)
        dLdXb = noise_layer.backprop(dLdXn)
        context_layer.backprop(dLdXb)

        # Apply gradients computed during backprop
        class_layer.apply_grad_reg(learn_rate=1e-4, ada_smooth=1e-3, lam_l2=lam_l2)
        context_layer.apply_grad_reg(learn_rate=1e-4, ada_smooth=1e-3, lam_l2=lam_l2)

        # Compute and display loss from time-to-time (for diagnostics)
        if ((b % 10) == 0):
            obs_count = 10.0 * batch_size
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
