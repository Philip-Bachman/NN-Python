import numpy as np
import numpy.random as npr
import StanfordTrees as st
import Word2Vec as w2v
import random as random
from time import clock
from sys import stdout as stdout

def run_test():
    # Load tree data
    tree_dir = './trees'
    stb_data = st.SimpleLoad(tree_dir)
    max_lut_idx = max(stb_data['lut_keys'].values())

    # Get the lists of full train and test phrases
    tr_phrases = stb_data['train_full_phrases']
    te_phrases = stb_data['train_full_phrases']
    # Get the list of all word occurrences in the training phrases
    tr_words = []
    for phrase in tr_phrases:
        tr_words.extend(phrase)
    tr_words = np.asarray(tr_words).astype(np.int32)
    tr_phrases = [np.asarray(p).astype(np.int32) for p in tr_phrases]
    te_phrases = [np.asarray(p).astype(np.int32) for p in te_phrases]

    batch_count = 5001
    batch_size = 256
    context_size = 5
    word_count = max_lut_idx + 1
    embed_dim = 200
    lam_l2 = 1e-3

    # Create a lookup table for word representations
    print("Word count: {0:d}, word dim: {1:d}".format(word_count, embed_dim))
    w2v_layer = w2v.W2VLayer(word_count, embed_dim)

    # Initialize params for the LUT and softmax classifier
    w2v_layer.init_params(0.05)

    sample_time = 0.0
    update_time = 0.0
    print("Processing batches:")
    L = 0.0
    for b in range(batch_count):
        # Sample a batch of random anchor/context prediction pairs for
        # training a skip-gram model.
        [a_idx, p_idx, n_idx, phrase_idx] = \
            w2v.rand_pos_neg(tr_phrases, tr_words, batch_size, context_size, 8)

        # Compute and apply the updates for this batch
        L += w2v_layer.batch_update(a_idx, p_idx, n_idx, learn_rate=2e-4)

        # Compute and display loss from time-to-time (for diagnostics)
        if ((b % 100) == 0):
            print("Batch {0:d}, loss {1:.4f}".format(b, (L / 10.0)))
            L = 0.0

        # Reset adagrad smoothing factors from time-to-time
        if ((b % 5000) == 0):
            w2v_layer.reset_moms()

if __name__ == '__main__':
    run_test()


##############
# EYE BUFFER #
##############
