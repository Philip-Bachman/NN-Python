import numpy as np
import numpy.random as npr
import NLMLayers as nlml
import cPickle as pickle
from HelperFuncs import zeros, ones, randn
import CorpusUtils as cu

class PVModel:
    """
    Paragraph Vector model, as described in "Distributed Representations of
    Sentences and Documents" by Quoc Le and Tomas Mikolov (ICML 2014).

    All training of this model is based on feeding it Python lists of phrases,
    where each element in a list is the representation of some phrase as a
    sequence of look-up-table keys stored in a 1d numpy array. The model is
    written to automatically add a "NULL" key that will be used when a sampled
    training n-gram extends beyond the scope of its source phrase. The model
    does _not_ deal with OOV words. It is up to the user to assign valid LUT
    keys to words not seen in training. In particular, no phrase passed to the
    moel should contain a LUT key greater than the parameter max_wv_key passed
    when the model was initialized.

    Predictions are trained based on a hierarchical softmax layer.

    Important Parameters (accessible via self.*):
      wv_dim: dimension of the word LUT vectors
      cv_dim: dimension of the context (a.k.a. paragraph) LUT vectors
      max_wv_key: max key of a valid word in the word LUT
      max_cv_key: max key of a valid context in the context LUT
      max_hs_key: max key used in any hierarchical softmax code
      pre_words: # of previous words to use in forward-prediction
      lam_wv: l2 regularization parameter for word vectors
      lam_cv: l2 regularization parameter for context vectors
      lam_sm: l2 regularization parameter for weights in softmax layer

    Note: This implementation also passes the word/context vectors through
          an extra "noise layer" prior to the HSM layer. The noise layer adds
          the option of using dropout/masking noise and Gaussian "fuzzing"
          noise for stronger regularization.
    """
    def __init__(self, wv_dim, cv_dim, max_wv_key, max_cv_key, max_hs_key, \
                 pre_words=5, lam_wv=1e-4, lam_cv=1e-4, lam_sm=1e-4):
        # Record options/parameters
        self.wv_dim = wv_dim
        self.cv_dim = cv_dim
        # Add 1 to the requested max word key, to accommodate the "NULL" key
        # that will be assigned to parts of an n-gram that don't exist.
        self.max_wv_key = max_wv_key + 1
        self.max_cv_key = max_cv_key
        self.pre_words = pre_words
        self.lam_wv = lam_wv
        self.lam_cv = lam_cv
        self.lam_sm = lam_sm
        # Set noise layer parameters (for better regularization)
        self.drop_rate = 0.0
        self.fuzz_scale = 0.0
        # Create layers to use during training
        self.word_layer = nlml.LUTLayer(self.max_wv_key, wv_dim, \
                                        n_gram=self.pre_words)
        self.context_layer = nlml.CMLayer(max_key=max_cv_key, \
                                          source_dim=wv_dim, \
                                          bias_dim=cv_dim, \
                                          do_rescale=False)
        self.noise_layer = nlml.GPUNoiseLayer(drop_rate=self.drop_rate, \
                                              fuzz_scale=self.fuzz_scale)
        assert(self.max_hs_key > 0)
        self.class_layer = nlml.HSMLayer(\
                in_dim=(self.cv_dim + (self.pre_words * self.wv_dim)), \
                max_hs_key=self.max_hs_key)
        return

    def set_noise(self, drop_rate=0.0, fuzz_scale=0.0):
        """Set params for the noise injection (i.e. perturbation) layer."""
        self.noise_layer.set_noise_params(drop_rate=drop_rate, \
                                          fuzz_scale=fuzz_scale)
        self.drop_rate = drop_rate
        self.fuzz_scale = fuzz_scale
        return

    def init_params(self, weight_scale=0.05):
        """Reset weights in the context LUT and softmax layers."""
        self.word_layer.init_params(weight_scale)
        self.context_layer.init_params(weight_scale)
        self.class_layer.init_params(weight_scale)
        return

    def reset_moms(self, ada_init=1e-3):
        """Reset the adagrad "momentums" in each layer."""
        self.word_layer.reset_moms(ada_init)
        self.context_layer.reset_moms(ada_init)
        self.class_layer.reset_moms(ada_init)
        return

    def batch_update(self, pre_keys, post_keys, phrase_keys, learn_rate=1e-3, \
                     train_context=False, train_other=False):
        """Perform a single "minibatch" update of the model parameters.

        Parameters:
            pre_keys: keys for the n-1 items in each n-gram to predict with
            post_keys: the n^th item in each n-gram to be predicted
            phrase_keys: key for the phrase from which each n-gram was taken
            learn_rate: learning rate to use in parameter updates
            train_context: train bias and modulator params in context layer
            train_other: train the basic word and classification params
        """
        L = zeros((1,))
        # Feedforward through look-up-table, noise, and prediction layers
        Xw = self.word_layer.feedforward(pre_keys)
        Xc = self.context_layer.feedforward(Xw, phrase_keys)
        Xn = self.noise_layer.feedforward(Xc, return_on_gpu=False)

        # TODO: deal with conversion to using HSM instead of full softmax

        # mostly, the use of hsm codes and signs needs to be built in, to
        # replace the reliance on just LUT keys for the prediction targets

        # Turn the corner with feedforward and backprop at class layer
        dLdXn, L = self.class_layer.ff_bp(Xn, param_1, param_2, do_grad=True)

        # Backprop through remaining layers
        dLdXc = self.noise_layer.backprop(dLdXn, return_on_gpu=False)
        dLdXw = self.context_layer.backprop(dLdXc)
        self.word_layer.backprop(dLdXw)

        # Apply the gradient updates computed during backprop
        if train_other:
            self.class_layer.apply_grad(learn_rate=learn_rate)
            self.word_layer.apply_grad(learn_rate=learn_rate)
        if train_context:
            self.context_layer.apply_grad(learn_rate=learn_rate)
        return L

    def train_all_params(self, phrase_list, batch_size, batch_count, \
                        learn_rate=1e-3):
        """Train all parameters in the model using the given phrases.

        Parameters:
            phrase_list: list of 1d numpy arrays of LUT keys (i.e. phrases)
            batch_size: size of minibatches for each update
            batch_count: number of minibatch updates to perform
            learn_rate: learning rate to use for updates
        """
        L = 0.0
        self.word_layer.reset_moms(ada_init=1.0)
        self.context_layer.reset_moms(ada_init=1.0)
        self.class_layer.reset_moms(ada_init=1.0)
        print("Training all parameters:")
        for b in range(batch_count):
            [seq_keys, phrase_keys] = rand_word_seqs(phrase_list, batch_size, \
                                      self.pre_words+1, self.max_wv_key)
            pre_keys = seq_keys[:,0:-1]
            post_keys = seq_keys[:,-1]
            L += self.batch_update(pre_keys, post_keys, phrase_keys, learn_rate, \
                                   train_context=True, train_other=True)
            if ((b % 200) == 0):
                obs_count = 200.0 * batch_size
                print("Batch {0:d}/{1:d}, loss {2:.4f}".format(b, batch_count, L/obs_count))
                L = 0.0
                self.word_layer.clip_params(max_norm=3.0)
                self.context_layer.clip_params(Wb_norm=3.0)
                self.class_layer.clip_params(max_norm=6.0)
        return

    def infer_context_vectors(self, phrase_list, batch_size, batch_count, \
                              learn_rate=1e-3):
        """Train context/paragraph vectors for each of the given phrases.

        Parameters:
            phrase_list: list of 1d numpy arrays of LUT keys (i.e. phrases)
            batch_size: batch size for minibatch updates
            batch_count: number of minibatch updates to perform
            learn_rate: learning rate for parameter updates
        """
        # Put a new context layer in place of self.context_layer, but keep
        # a pointer to self.context_layer around to restore later...
        max_cv_key = len(phrase_list) - 1
        new_context_layer = nlml.CMLayer(max_key=self.max_cv_key, \
                                         source_dim=self.wv_dim, \
                                         bias_dim=self.cv_dim, \
                                         do_rescale=False)
        new_context_layer.init_params(0.0)
        prev_context_layer = self.context_layer
        self.context_layer = new_context_layer
        # Update the context vectors in the new context layer for some number
        # of minibatch update rounds
        L = 0.0
        print("Training new context vectors:")
        for b in range(batch_count):
            [seq_keys, phrase_keys] = rand_word_seqs(phrase_list, batch_size, \
                                      self.pre_words+1, self.max_wv_key)
            pre_keys = seq_keys[:,0:-1]
            post_keys = seq_keys[:,-1]
            L += self.batch_update(pre_keys, post_keys, phrase_keys, learn_rate, \
                                   train_context=True, train_other=False)
            if ((b % 200) == 0):
                obs_count = 200.0 * batch_size
                print("Test batch {0:d}/{1:d}, loss {2:.4f}".format(b, batch_count, L/obs_count))
                L = 0.0
                self.context_layer.clip_params(max_norm=3.0)
        # Set self.context_layer back to what it was prior to retraining
        self.word_layer.reset_grads()
        self.context_layer = prev_context_layer
        self.class_layer.reset_grads()
        return new_context_layer


####################################
# Context-adaptive Skip-gram Model #
####################################

class CAModel:
    """
    Context-adaptive skip-gram model, as defined by Philip Bachman.

    Important Parameters (accessible via self.*):
      wv_dim: dimension of the LUT word vectors
      cv_dim: dimension of the context (a.k.a. paragraph) vectors
      max_wv_key: max key of a valid word in the word LUT
      max_cv_key: max key of a valid context in the context LUT
      use_ns: if True, use negative sanpling, otherwise use HSM
      lam_wv: l2 regularization parameter for word vectors
      lam_cv: l2 regularization parameter for context vectors
      lam_ns: l2 regularization parameter for negative sampling layer

    Note: This implementation also passes the word/context vectors through
          an extra "noise layer" prior to the negative sampling layer. The
          noise layer adds the option of using dropout/masking noise and also
          Gaussian "weight fuzzing" noise for stronger regularization.
    """
    def __init__(self, wv_dim, cv_dim, max_wv_key, max_cv_key, use_ns=True, \
                 max_hs_key=0, lam_wv=1e-4, lam_cv=1e-4, lam_cl=1e-4):
        # Record options/parameters
        self.wv_dim = wv_dim
        self.cv_dim = cv_dim
        self.max_wv_key = max_wv_key
        self.max_cv_key = max_cv_key
        self.use_ns = use_ns
        self.max_hs_key = max_hs_key
        self.lam_wv = lam_wv
        self.lam_cv = lam_cv
        self.lam_cl = lam_cl
        self.reg_freq = 20 # number of batches between regularization updates
        # Set noise layer parameters (for better regularization)
        self.drop_rate = 0.0
        self.fuzz_scale = 0.0
        # Create layers to use during training
        self.use_tanh = True
        self.tanh_layer = nlml.TanhLayer()
        self.word_layer = nlml.LUTLayer(max_wv_key, wv_dim)
        self.context_layer = nlml.CMLayer(max_key=max_cv_key, \
                                          source_dim=wv_dim, \
                                          bias_dim=cv_dim, \
                                          do_rescale=True)
        self.noise_layer = nlml.NoiseLayer(drop_rate=self.drop_rate, \
                                              fuzz_scale=self.fuzz_scale)
        if self.use_ns:
            self.class_layer = nlml.NSLayer(in_dim=(self.cv_dim+self.wv_dim), \
                                            max_out_key=self.max_wv_key)
        else:
            assert(self.max_hs_key > 0)
            self.class_layer = nlml.HSMLayer(in_dim=(self.cv_dim+self.wv_dim), \
                                             max_hs_key=self.max_hs_key)
        return

    def init_params(self, weight_scale=0.05):
        """Reset weights in the word/context LUTs and the prediciton layer."""
        self.word_layer.init_params(weight_scale)
        self.context_layer.init_params(0.0 * weight_scale)
        self.class_layer.init_params(weight_scale)
        return

    def reset_moms(self, ada_init=1e-3):
        """Reset the adagrad "momentums" in each layer."""
        self.word_layer.reset_moms(ada_init)
        self.context_layer.reset_moms(ada_init)
        self.class_layer.reset_moms(ada_init)
        return

    def set_noise(self, drop_rate=0.0, fuzz_scale=0.0):
        """Set params for the noise injection (i.e. perturbation) layer."""
        self.noise_layer.set_noise_params(drop_rate=drop_rate, \
                                          fuzz_scale=fuzz_scale)
        self.drop_rate = drop_rate
        self.fuzz_scale = fuzz_scale
        return

    def batch_update(self, anc_keys, param_1, param_2, phrase_keys, \
                     train_words=True, train_context=True, learn_rate=1e-3):
        """
        Perform a single "minibatch" update of the model parameters.

        Parameters:
            anc_keys: word LUT keys for the anchor words
            if self.use_ns:
                param_1: LUT keys for positive prediction targets
                param_2: LUT keys for negative prediction targets
            else:
                param_1: LUT keys for HSM codes
                param_2: Target classes (+1/-1) for HSM codes
            phrase_keys: phrase/context LUT keys for the phrases from which
                         the words to predict with (in anc_keys and param_1)
                         were sampled.
            train_words: set to True to update word and prediction LUTs
            train_context: set to True to update context layer
            learn_rate: learning rate for adagrad updates
        """
        # Feedforward through the various layers of this model
        Xb = self.word_layer.feedforward(anc_keys)
        Xc = self.context_layer.feedforward(Xb, phrase_keys)
        if self.use_tanh:
            Xt = self.tanh_layer.feedforward(Xc)
        else:
            Xt = Xc
        Xn = self.noise_layer.feedforward(Xt) #, return_on_gpu=False)

        # Turn the corner with feedforward and backprop at class layer
        dLdXn, L = self.class_layer.ff_bp(Xn, param_1, param_2, do_grad=True)

        # Backprop through layers based on feedforward result
        dLdXt = self.noise_layer.backprop(dLdXn) #, return_on_gpu=False)
        if self.use_tanh:
            dLdXc = self.tanh_layer.backprop(dLdXt)
        else:
            dLdXc = dLdXt
        dLdXb = self.context_layer.backprop(dLdXc)
        self.word_layer.backprop(dLdXb)

        # Update parameters using the gradients computed in backprop
        if train_words:
            # Update parameters that directly represent words
            self.word_layer.apply_grad(learn_rate=learn_rate)
            self.class_layer.apply_grad(learn_rate=learn_rate)
        if train_context:
            # Update parameters that control context-adaptivity
            self.context_layer.apply_grad(learn_rate=learn_rate)
        return L

    def train(self, pos_sampler, var_param, batch_size, batch_count, \
              train_words=True, train_context=True, learn_rate=1e-3):
        """
        Train all parameters in the model using the given phrases.

        Parameters:
            pos_sampler: sampler for generating positive prediction pairs
            if self.use_ns:
                var_param: sampler for generating negative prediction pairs
            else:
                var_param: dict containing HSM code keys and signs (in LUTs)
            batch_size: size of minibatches for each update
            batch_count: number of minibatch updates to perform
            train_words: whether or not to do updates for the word LUT and
                         classification layers
            train_context: whether or not to do updates for the context layer
            learn_rate: learning rate for adagrad updates
        """
        print("Training all parameters:")
        L = 0.0
        self.word_layer.reset_moms(1.0)
        self.context_layer.reset_moms(1.0)
        self.class_layer.reset_moms(1.0)
        for b in range(batch_count):
            anc_keys, pos_keys, phrase_keys = pos_sampler.sample(batch_size)
            if self.use_ns:
                param_1 = pos_keys
                param_2 = var_param.sample(batch_size)
            else:
                param_1 = var_param['keys_to_code_keys'].take(pos_keys,axis=0)
                param_2 = var_param['keys_to_code_signs'].take(pos_keys,axis=0)
            L += self.batch_update(anc_keys, param_1, param_2, phrase_keys, \
                                   train_words=train_words, \
                                   train_context=train_context, \
                                   learn_rate=learn_rate)
            # apply l2 regularization, but not every round (to save flops)
            if ((b > 1) and ((b % self.reg_freq) == 0)):
                reg_rate = learn_rate * self.reg_freq
                if train_words:
                    self.word_layer.l2_regularize(lam_l2=(reg_rate*self.lam_wv))
                    self.class_layer.l2_regularize(lam_l2=(reg_rate*self.lam_cl))
                if train_context:
                    self.context_layer.l2_regularize(lam_Wm=(reg_rate*self.lam_cv), \
                                                    lam_Wb=(reg_rate*self.lam_cv))
            # diagnostic display stuff...
            if ((b > 1) and ((b % 1000) == 0)):
                Wm_info = self.context_layer.norm_info('Wm')
                Wb_info = self.context_layer.norm_info('Wb')
                print("-- mean norms: Wm = {0:.4f}, Wb = {1:.4f}".format(Wm_info['mean'],Wb_info['mean']))
            if ((b % 500) == 0):
                obs_count = 500.0 * batch_size
                print("Batch {0:d}/{1:d}, loss {2:.4f}".format(b, batch_count, L/obs_count))
                L = 0.0
        return

    def infer_context_vectors(self, pos_sampler, var_param, batch_size, \
                              batch_count, learn_rate=1e-3):
        """
        Train a new set of context layer parameters using the given phrases.

        Parameters:
            pos_sampler: sampler for generating positive prediction pairs
            if self.use_ns:
                var_param: sampler for generating negative prediction pairs
            else:
                var_param: dict containing HSM code keys and signs (in LUTs)
            batch_size: size of minibatches for each update
            batch_count: number of minibatch updates to perform
        """
        # Put a new context layer in place of self.context_layer, but keep
        # a pointer to self.context_layer around to restore later...
        max_cv_key = pos_sampler.max_phrase_key
        new_context_layer = nlml.CMLayer(max_key=max_cv_key, \
                                         source_dim=self.wv_dim, \
                                         bias_dim=self.cv_dim, \
                                         do_rescale=True)
        new_context_layer.init_params(0.0)
        prev_context_layer = self.context_layer
        self.context_layer = new_context_layer
        self.context_layer.reset_moms(1.0)
        print("Training new context vectors:")
        L = 0.0
        for b in range(batch_count):
            anc_keys, pos_keys, phrase_keys = pos_sampler.sample(batch_size)
            if self.use_ns:
                param_1 = pos_keys
                param_2 = var_param.sample(batch_size)
            else:
                param_1 = var_param['keys_to_code_keys'].take(pos_keys,axis=0)
                param_2 = var_param['keys_to_code_signs'].take(pos_keys,axis=0)
            L += self.batch_update(anc_keys, param_1, param_2, phrase_keys, \
                                   train_words=False, \
                                   train_context=train_context, \
                                   learn_rate=learn_rate)
            if ((b > 1) and ((b % self.reg_freq) == 0)):
                reg_rate = learn_rate * self.reg_freq
                self.context_layer.l2_regularize(lam_Wm=(reg_rate*self.lam_cv), \
                                                 lam_Wb=(reg_rate*self.lam_cv))
            if ((b > 1) and ((b % 1000) == 0)):
                Wm_info = self.context_layer.norm_info('Wm')
                Wb_info = self.context_layer.norm_info('Wb')
                print("-- mean norms: Wm = {0:.4f}, Wb = {1:.4f}".format(Wm_info['mean'],Wb_info['mean']))
            if ((b % 500) == 0):
                obs_count = 500.0 * batch_size
                print("Batch {0:d}/{1:d}, loss {2:.4f}".format(b, batch_count, L/obs_count))
                L = 0.0
        # Set self.context_layer back to what it was previously
        self.context_layer = prev_context_layer
        # Reset gradients in all layers
        self.word_layer.reset_grads_and_moms()
        self.context_layer.reset_grads_and_moms()
        self.class_layer.reset_grads_and_moms()
        return new_context_layer

#######################
# Test scripting code #
#######################

def some_nearest_words(keys_to_words, sample_count, W1=None, W2=None):
    assert(not (W1 is None))
    if not (W2 is None):
        W = np.hstack((W1, W2))
    else:
        W = W1
    norms = np.sqrt(np.sum(W**2.0,axis=1,keepdims=1))
    W = W / (norms + 1e-5)
    # 
    source_keys = np.zeros((sample_count,)).astype(np.uint32)
    neighbor_keys = np.zeros((sample_count, 10)).astype(np.uint32)
    all_keys = np.asarray(keys_to_words.keys()).astype(np.uint32)
    for s in range(sample_count):
        i = npr.randint(0,all_keys.size)
        source_k = all_keys[i]
        neg_cos_sims = -1.0 * np.sum(W * W[source_k], axis=1)
        sorted_k = np.argsort(neg_cos_sims)
        source_keys[s] = source_k
        neighbor_keys[s,:] = sorted_k[1:11]
    source_words = []
    neighbor_words = []
    for s in range(sample_count):
        source_words.append(keys_to_words[source_keys[s]])
        neighbor_words.append([keys_to_words[k] for k in neighbor_keys[s]])
    return [source_keys, neighbor_keys, source_words, neighbor_words]

if __name__=="__main__":
    # import DataLoaders as dl
    #  # Load text phrase data
    # dataset = dl.Load1BWords('./training_text', file_count=4, min_freq=10)

    # # Get the lists of full train and test phrases
    # tr_phrases = dataset['train_key_phrases']
    # te_phrases = dataset['dev_key_phrases']
    # k2w = dataset['keys_to_words']
    # # Get the list of all word occurrences in the training phrases
    # tr_words = []
    # for phrase in tr_phrases:
    #     tr_words.extend([k for k in phrase])
    # tr_words = np.array(tr_words).astype(np.uint32)
    # # Record maximum required keys for the context layer's tables
    # max_wv_key = max(dataset['words_to_keys'].values())
    # max_cv_key = 100
    sentences = cu.SentenceFileIterator('./training_text')
    key_dicts = cu.build_vocab(sentences, min_count=3, compute_hs_tree=True, \
                            compute_ns_table=True, down_sample=0.0)
    w2k = key_dicts['words_to_keys']
    k2w = key_dicts['keys_to_words']
    neg_table = key_dicts['ns_table']
    unk_word = key_dicts['unk_word']
    sentences = cu.SentenceFileIterator('./training_text')
    tr_phrases = cu.sample_phrases(sentences, w2k, unk_word=unk_word, \
                                max_phrases=100000)
    max_cv_key = len(tr_phrases) + 1
    max_wv_key = max(w2k.values())


    # Choose some simple hyperparameters for the model
    sg_window = 6
    ns_count = 10
    wv_dim = 80
    cv_dim = 10
    lam_l2 = 2e-3

    cam = CAModel(wv_dim, cv_dim, max_wv_key, max_cv_key, \
                use_ns=True, max_hs_key=0, \
                lam_wv=lam_l2, lam_cv=lam_l2, lam_cl=lam_l2)
    cam.init_params(0.05)
    cam.set_noise(drop_rate=0.5, fuzz_scale=0.0)

    # Initialize samplers for training
    pos_sampler = cu.PosSampler(tr_phrases, sg_window)
    neg_sampler = cu.NegSampler(neg_table=neg_table, neg_count=ns_count)

    # Train all parameters using the training set phrases
    for i in range(50):
        cam.train(pos_sampler, neg_sampler, 300, 10001, train_words=True, \
                  train_context=False, learn_rate=1e-3)
        [s_keys, n_keys, s_words, n_words] = some_nearest_words( k2w, 10, \
                  W1=cam.word_layer.params['W'], W2=None)
        for w in range(10):
            print("{0:s}: {1:s}".format(s_words[w],", ".join(n_words[w])))
