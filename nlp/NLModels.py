import numpy as np
import numpy.random as npr
import Word2Vec as w2v
import ParVec as pv
import HelperFuncs as hf

"""
TODO: write classes for managing training and testing of several types of
neural language models.

Note: For all models, it will be assumed that sampling LUT indices
corresponding to the word vectors needed for training and testing will be done
by outside code. I.e. these classes don't include tools for converting chunks
of text into sequences of look-up-table keys or for sampling from the LUT key
sequences to get sets of LUT keys suitable for training/testing.

Model 1: Basic Word2Vec style skip-gram model trained with negative sampling.

Model 2: Basic Paragraph Vector model, i.e. forward-prediction n-gram model
         with context-adaptive biases.

Model 3: Word2Vec style skip-gram model with added dropout, gaussian fuzzing,
         context-adaptive biases and context-adaptive feature reweighting.
"""

class PVModel:
    """
    Paragraph Vector model, as described in "Distributed Representations of
    Sentences and Documents" by Quoc Le and Tomas Mikolov (ICML 2014).

    Important Parameters (accessible via self.*):
      wv_dim: dimension of the LUT word vectors
      cv_dim: dimension of the context (a.k.a. paragraph) vectors
      max_wv_key: max key of a valid word in the word LUT
      max_cv_key: max key of a valid context in the context LUT
      pre_words: # of previous context words to use in forward-prediction
      lam_wv: l2 regularization parameter for word vectors
      lam_cv: l2 regularization parameter for context vectors
      lam_sm: l2 regularization parameter for weights in softmax layer

    Note: This implementation also passes the word/context vectors through
          an extra "noise layer" prior to the softmax layer. The noise layer
          adds the option of using dropout/masking noise and also Gaussian
          "weight fuzzing" noise for stronger regularization.
    """
    def __init__(self, wv_dim, cv_dim, max_wv_key, max_cv_key, \
                 pre_words=5, lam_wv=1e-4, lam_cv=1e-4, lam_sm=1e-4):
        # Record options/parameters
        self.wv_dim = wv_dim
        self.cv_dim = cv_dim
        self.max_wv_key = max_wv_key
        self.max_cv_key = max_cv_key
        self.pre_words = pre_words
        self.lam_wv = lam_wv
        self.lam_cv = lam_cv
        self.lam_sm = lam_sm
        # Set noise layer parameters (for better regularization)
        self.drop_rate = 0.0
        self.fuzz_scale = 0.0
        # Create layers to use during training
        self.context_layer = pv.ContextLayer(max_wv_key, wv_dim, max_cv_key, cv_dim)
        self.noise_layer = pv.NoiseLayer(drop_rate=self.drop_rate,
                                         fuzz_scale=self.fuzz_scale)
        self.softmax_layer = pv.FullLayer(in_dim=(pre_words*wv_dim + cv_dim), \
                                          out_dim=max_wv_key)
        return

    def init_params(self, weight_scale=0.05):
        """Reset weights in the context LUT and softmax layers."""
        self.context_layer.init_params(weight_scale)
        self.softmax_layer.init_params(weight_scale)
        return

    def reset_moms(self, ada_init=1e-3):
        """Reset the adagrad "momentums" in each layer."""
        self.context_layer.reset_moms(ada_init)
        self.softmax_layer.reset_moms(ada_init)
        return

    def batch_update(self, pre_keys, post_keys, phrase_keys, learn_rate, \
                    ada_smooth=1e-3, context_only=False):
        """Perform a single "minibatch" update of the model parameters.

        This uses the word key sequences stored in the rows of the matrix
        word_keys together with their corresponding context keys stored in the
        vector phrase_keys. Parameters learn_rate and ada_smooth control the
        step size and "smoothing" applied during the adagrad update.
        """
        # Feedforward through look-up-table, noise, and softmax layers
        Xb = self.context_layer.feedforward(pre_keys, phrase_keys, \
                                            test_w=False, \
                                            test_c=False)
        Xn = self.noise_layer.feedforward(Xb)
        Yn = self.softmax_layer.feedforward(Xn)
        # Compute loss at softmax layer
        L = self.softmax_layer.cross_entropy_loss(Yn, post_keys)

        # Backprop through the layers in reverse order
        dLdXn = self.softmax_layer.backprop(post_keys, return_on_gpu=True, \
                                            test=context_only)
        dLdXb = self.noise_layer.backprop(dLdXn, return_on_gpu=False)
        self.context_layer.backprop(dLdXb)
        if not context_only:
            # Apply gradients computed during backprop (to all parameters)
            self.softmax_layer.apply_grad_reg(learn_rate=learn_rate, \
                                              ada_smooth=ada_smooth, \
                                              lam_l2=self.lam_sm)
            self.context_layer.apply_grad_reg(learn_rate=learn_rate, \
                                              ada_smooth=ada_smooth, \
                                              lam_w=self.lam_wv, \
                                              lam_c=self.lam_cv)
        else:
            # Apply gradients only to the context vectors (for test-time)
            self.context_layer.update_context_vectors(learn_rate=learn_rate, \
                                                      ada_smooth=ada_smooth, \
                                                      lam_c=self.lam_cv)
        return L

    def train_all_params(self, phrase_list, batch_size, batch_count):
        """Train all parameters in the model using the given phrases.

        Parameters:
            phrase_list: list of 1d numpy arrays of LUT keys (i.e. phrases)
            batch_size: size of minibatches for each update
            batch_count: number of minibatch updates to perform
        """
        L = 0.0
        print("Training new context vectors:")
        for b in range(batch_count):
            [seq_keys, phrase_keys] = hf.rand_word_seqs(phrase_list, batch_size, \
                                    self.pre_words+1, self.max_wv_key)
            pre_keys = seq_keys[:,0:-1]
            post_keys = seq_keys[:,-1]
            L += self.batch_update(pre_keys, post_keys, phrase_keys, 1e-3, \
                                   ada_smooth=1e-3, context_only=False)
            if ((b % 100) == 0):
                obs_count = 100.0 * batch_size
                print("Batch {0:d}, loss {1:.4f}".format(b, L/obs_count))
                L = 0.0
        return

    def new_context_vectors(self, phrase_list, batch_size, batch_count):
        """Train context/paragraph vectors for each of the given phrases.

        Each phrase in phrase_list gives a sequence of keys into the word
        LUT that was (presumably) previously trained for this model. During
        test inference words not previously trained will be given the OOV
        vector (which should be zero, if ParVec.py is bug-free).
        """
        # Put a new context layer in place of self.context_layer, but keep
        # a pointer to self.context_layer around to restore later...
        max_cv_key = len(phrase_list) - 1
        new_context_layer = pv.ContextLayer(self.max_wv_key, self.wv_dim, \
                                            max_cv_key, self.cv_dim)
        new_context_layer.params['Vw'] = self.context_layer.params['Vw'].copy()
        prev_context_layer = self.context_layer
        self.context_layer = new_context_layer
        # Update the context vectors in the new context layer for some number
        # of minibatch update rounds
        L = 0.0
        print("Training new context vectors:")
        for b in range(batch_count):
            [seq_keys, phrase_keys] = hf.rand_word_seqs(phrase_list, batch_size, \
                                    self.pre_words+1, self.max_wv_key)
            pre_keys = seq_keys[:,0:-1]
            post_keys = seq_keys[:,-1]
            L += self.batch_update(pre_keys, post_keys, phrase_keys, 4e-3, \
                                   ada_smooth=1e-3, context_only=True)
            if ((b % 100) == 0):
                obs_count = 100.0 * batch_size
                print("Test batch {0:d}, loss {1:.4f}".format(b, L/obs_count))
                L = 0.0
        # Set self.context_layer back to what it was prior to retraining
        self.context_layer = prev_context_layer
        return new_context_layer


def stb_test_PVModel():
    """Hard-coded test on STB data, for development/debugging."""
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
    max_wv_key = max(stb_data['lut_keys'].values()) + 1
    max_cv_key = len(tr_phrases) - 1

    # Choose some simple hyperparameters for the model
    pre_words = 5
    wv_dim = 250
    cv_dim = 250
    lam_l2 = 1e-3

    pvm = PVModel(wv_dim, cv_dim, max_wv_key, max_cv_key, pre_words=pre_words, \
                  lam_wv=lam_l2, lam_cv=lam_l2, lam_sm=lam_l2)
    pvm.init_params(0.05)

    # Train all parameters using the training set phrases
    pvm.train_all_params(tr_phrases, 256, 100001)

    # Train new context vectors using the validation set phrases
    pvm.new_context_vectors(te_phrases, 256, 100001)
    return



if __name__ == '__main__':
    stb_test_PVModel()


##############
# EYE BUFFER #
##############
