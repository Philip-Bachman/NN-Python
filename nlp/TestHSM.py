import numpy as np
import numpy.random as npr
import NLMLayers as nlml
import cPickle as pickle
from HelperFuncs import zeros, ones, randn
import CorpusUtils as cu


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
        Xn = self.noise_layer.feedforward(Xc)

        # Turn the corner with feedforward and backprop at class layer
        dLdXn, L = self.class_layer.ff_bp(Xn, param_1, param_2, do_grad=True)

        # Backprop through layers based on feedforward result
        dLdXc = self.noise_layer.backprop(dLdXn)
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
            for i in range(batch_size):
                phrase_keys[i] = 0
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
    import DataLoaders as dl
     # Load text phrase data
    dataset = dl.Load1BWords('./training_text', file_count=4, min_freq=10)

    # Get the lists of full train and test phrases
    tr_phrases = dataset['train_key_phrases']
    te_phrases = dataset['dev_key_phrases']
    k2w = dataset['keys_to_words']
    # Get the list of all word occurrences in the training phrases
    tr_words = []
    for phrase in tr_phrases:
        tr_words.extend([k for k in phrase])
    tr_words = np.asarray(tr_words).astype(np.uint32)
    # Record maximum required keys for the context layer's tables
    max_wv_key = max(dataset['words_to_keys'].values())
    max_cv_key = 100

    # Choose some simple hyperparameters for the model
    sg_window = 6
    ns_count = 10
    wv_dim = 256
    cv_dim = 50
    lam_l2 = 1e-3

    cam = CAModel(wv_dim, cv_dim, max_wv_key, max_cv_key, \
                use_ns=True, max_hs_key=0, \
                lam_wv=lam_l2, lam_cv=lam_l2, lam_cl=lam_l2)
    cam.init_params(0.05)
    cam.set_noise(drop_rate=0.5, fuzz_scale=0.0)

    # Initialize samplers for training
    pos_sampler = cu.PosSampler(tr_phrases, sg_window)
    neg_sampler = cu.NegSampler(neg_table=tr_words, neg_count=ns_count)

    # Train all parameters using the training set phrases
    for i in range(50):
        cam.train(pos_sampler, neg_sampler, 300, 10001, train_words=True, \
                  train_context=False, learn_rate=1e-2)
        [s_keys, n_keys, s_words, n_words] = some_nearest_words( k2w, 10, \
                  W1=cam.word_layer.params['W'], W2=None)
        for w in range(10):
            print("{0:s}: {1:s}".format(s_words[w],", ".join(n_words[w])))
