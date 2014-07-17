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
    te_phrases = stb_data['test_full_phrases']
    # Get the list of all word occurrences in the training phrases
    tr_words = []
    for phrase in tr_phrases:
        tr_words.extend(phrase)
    tr_words = np.asarray(tr_words).astype(np.int32)
    tr_phrases = [np.asarray(p).astype(np.int32) for p in tr_phrases]
    te_phrases = [np.asarray(p).astype(np.int32) for p in te_phrases]

    batch_count = 501
    batch_size = 400
    context_size = 5
    word_count = max_lut_idx + 1
    embed_dim = 250
    bias_dim = 50
    lam_l2 = 1e-3

    # Create a lookup table for word representations
    print("Word count: {0:d}, word dim: {1:d}".format(word_count, embed_dim))
    word_lut = w2v.LUTLayer(word_count, embed_dim)
    tanh_layer = w2v.TanhLayer()
    noise_layer = w2v.NoiseLayer(drop_rate=0.0, fuzz_scale=0.0)
    phrase_layer = w2v.CMLayer(key_count=len(tr_phrases), source_dim=embed_dim, bias_dim=bias_dim)

    # Create a negative-sampling layer for classification
    class_layer = w2v.NSLayer(key_count=max_lut_idx, in_dim=(embed_dim+bias_dim))

    # Initialize params for the LUT and softmax classifier
    word_lut.init_params(0.05)
    class_layer.init_params(0.05)

    print("Processing batches:")
    L = 0.0
    for b in range(batch_count):
        # Reset adagrad smoothing factors from time-to-time
        if ((b > 1) and ((b % 5000) == 0)):
            word_lut.reset_moms()
            phrase_layer.reset_moms()
            class_layer.reset_moms()

        # Sample a batch of random anchor/context prediction pairs for
        # training a skip-gram model.
        [a_idx, p_idx, n_idx, phrase_idx] = \
            w2v.rand_pos_neg(tr_phrases, tr_words, batch_size, context_size, 8)

        # Feedforward through word look-up and phrase biasing/reweighting
        Xb = word_lut.feedforward(a_idx)
        Xp = phrase_layer.feedforward(Xb, phrase_idx)
        Xn = noise_layer.feedforward(Xp)

        # Feedforward through the prediction layer
        L += class_layer.feedforward(Xn, p_idx, n_idx)

        # Compute and display loss from time-to-time (for diagnostics)
        if ((b % 50) == 0):
            print("Batch {0:d}, loss {1:.4f}".format(b, (L / 10.0)))
            L = 0.0
        if ((b % 500) == 0):
            info = phrase_layer.norm_info('W')
            print("    ||W|| -- mean: {0:.4f}, min: {1:.4f}, median: {2:.4f}, max: {3:.4f}".format( \
                    info['mean'], info['min'], info['median'], info['max']))
            info = phrase_layer.norm_info('b')
            print("    ||b|| -- mean: {0:.4f}, min: {1:.4f}, median: {2:.4f}, max: {3:.4f}".format( \
                    info['mean'], info['min'], info['median'], info['max']))

        # Backprop through the composition of layers
        dLdXn = class_layer.backprop()
        dLdXp = noise_layer.backprop(dLdXn)
        dLdXb = phrase_layer.backprop(dLdXp)
        word_lut.backprop(dLdXb)

        # Update parameters based on the gradients for this batch
        word_lut.apply_grad_reg(learn_rate=2e-4, ada_smooth=1e-3, lam_l2=lam_l2)
        class_layer.apply_grad_reg(learn_rate=2e-4, ada_smooth=1e-3, lam_l2=lam_l2)
        phrase_layer.apply_grad_reg(learn_rate=2e-4, ada_smooth=1e-3, lam_l2=lam_l2)

if __name__ == '__main__':
    stb_data = load_data()
    run_test(stb_data)


##############
# EYE BUFFER #
##############
