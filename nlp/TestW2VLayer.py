import numpy as np
import numpy.random as npr
import StanfordTrees as st
import Word2Vec as w2v
import random as random
from time import clock
from sys import stdout as stdout

def load_data():
    tree_dir = './trees'
    stb_data = st.SimpleLoad(tree_dir)
    return stb_data

def run_test(stb_data=None):

    # Load tree data
    if not stb_data:
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

    batch_count = 20001
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

    L = 0.0
    print("Processing batches:")
    for b in range(batch_count):
        # Sample a batch of random anchor/context prediction pairs for
        # training a skip-gram model.
        [a_idx, p_idx, n_idx, phrase_idx] = \
            w2v.rand_pos_neg(tr_phrases, tr_words, batch_size, context_size, 8)

        # Compute and apply the updates for this batch
        L += w2v_layer.batch_train(a_idx, p_idx, n_idx, learn_rate=1e-4)

        # Compute and display loss from time-to-time (for diagnostics)
        if ((b % 100) == 0):
            obs_count = batch_size * 100.0
            print("Batch {0:d}, loss {1:.4f}".format(b, (L / obs_count)))
            L = 0.0

        # Occasionally compute validation set loss
        if ((b % 500) == 0):
            obs_count = 1000
            [a_idx, p_idx, n_idx, phrase_idx] = \
                w2v.rand_pos_neg(te_phrases, tr_words, obs_count, context_size, 8)
            Lv = w2v_layer.batch_test(a_idx, p_idx, n_idx)
            print("    test loss: {0:.4f}".format(Lv / obs_count))

        # Reset adagrad smoothing factors from time-to-time
        if ((b > 1) and ((b % 5000) == 0)):
            w2v_layer.reset_moms()

if __name__ == '__main__':
    stb_data = load_data()
    run_test(stb_data)


##############
# EYE BUFFER #
##############
