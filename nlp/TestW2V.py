import numpy as np
import numpy.random as npr
import StanfordTrees as st
import Word2Vec as w2v
import random as random
from time import clock
from sys import stdout as stdout

@profile
def run_test():
    # Load tree data
    tree_dir = './trees'
    stb_data = st.SimpleLoad(tree_dir)
    max_lut_idx = max(stb_data['lut_keys'].values())

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

    batch_count = 1001
    batch_size = 256
    context_size = 6
    word_count = max_lut_idx + 1
    embed_dim = 300
    lam_l2 = 1e-3

    print("Word count: {0:d}, word dim: {1:d}".format(word_count, embed_dim))
    # Get a lookup table for word vectors and some noise to perturb them
    word_lut = w2v.LUTLayer(word_count, embed_dim)
    noise_layer = w2v.NoiseLayer(drop_rate=0.5, fuzz_scale=0.025)

    # Create a negative-sampling layer for classification
    class_layer = w2v.NSLayer(key_count=max_lut_idx, in_dim=embed_dim)

    # Initialize params for the LUT and NS layers
    word_lut.init_params(0.05)
    class_layer.init_params(0.05)

    print("Processing batches:")
    L = 0.0
    for b in range(batch_count):
        # Reset adagrad smoothing factors from time-to-time
        if ((b > 1) and ((b % 5000) == 0)):
            word_lut.reset_moms()
            class_layer.reset_moms()

        # Sample a batch of random anchor/context prediction pairs for
        # training a skip-gram model.
        [a_idx, p_idx, n_idx, phrase_idx] = \
            w2v.rand_pos_neg(tr_phrases, tr_words, batch_size, context_size, 8)

        # Feedforward through all of the layers
        Xb = word_lut.feedforward(a_idx)
        Xn = noise_layer.feedforward(Xb)
        L += class_layer.feedforward(Xn, p_idx, n_idx)

        # Backprop through all of the layers
        dLdXn = class_layer.backprop()
        dLdXb = noise_layer.backprop(dLdXn)
        word_lut.backprop(dLdXb)

        # Update parameters based on the gradients for this batch
        word_lut.apply_grad_reg(learn_rate=3e-4, ada_smooth=1e-3, lam_l2=lam_l2)
        class_layer.apply_grad_reg(learn_rate=3e-4, ada_smooth=1e-3, lam_l2=lam_l2)

        # Display training set loss from time-to-time (for diagnostics)
        if ((b % 100) == 0):
            obs_count = batch_size * 100.0
            print("Batch {0:d}, loss {1:.4f}".format(b, (L / obs_count)))
            L = 0.0
        # Occasionally compute validation set loss (for diagnostics)
        if ((b > 1) and ((b % 1000) == 0)):
            obs_count = 1000
            [a_idx, p_idx, n_idx, phrase_idx] = \
                w2v.rand_pos_neg(te_phrases, tr_words, obs_count, context_size, 8)
            Xb = word_lut.feedforward(a_idx, test=True)
            Xn = noise_layer.feedforward(Xb, test=True)
            Lv = class_layer.feedforward(Xn, p_idx, n_idx, test=True)
            print("    test loss: {0:.4f}".format(Lv / obs_count))


if __name__ == '__main__':
    run_test()


##############
# EYE BUFFER #
##############
