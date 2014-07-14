import numpy as np
import numpy.random as npr
import gnumpy as gp
import StanfordTrees as st
import Word2Vec as w2v
import random as random
from time import clock
from sys import stdout as stdout

def simple_stb_test(tree_dir='./trees'):
    stb_data = st.SimpleLoad(tree_dir)
    return

if __name__ == '__main__':
    # Load tree data
    tree_dir = './trees'
    stb_data = st.SimpleLoad(tree_dir)
    max_lut_idx = max(stb_data['lut_keys'].values())

    # negligible change

    all_phrases = stb_data['train_full_phrases']

    batch_count = 50000
    batch_size = 256
    context_size = 5
    word_count = max_lut_idx + 1
    embed_dim = 300
    bias_dim = 100
    lam_l2 = 1e-5

    # Create a lookup table for word representations
    word_lut = w2v.LUTLayer(word_count, embed_dim)
    tanh_layer = w2v.TanhLayer(in_layer=word_lut)
    noise_layer = w2v.NoiseLayer(in_layer=tanh_layer, drop_rate=0.0, fuzz_scale=0.0)
    phrase_layer = w2v.CMLayer(key_count=len(all_phrases), source_dim=embed_dim, bias_dim=bias_dim)

    # Create a full/softmax layer for classification
    class_layer = w2v.GPUFullLayer(in_layer=False, in_dim=(bias_dim+embed_dim), out_dim=word_count)

    # Initialize params for the LUT and softmax classifier
    word_lut.init_params(0.05)
    class_layer.init_params(0.05)

    table_time = 0.0
    print("Processing batches:")
    t1 = clock()
    for b in range(batch_count):
        # Sample a batch of random anchor/context prediction pairs for
        # training a skip-gram model.
        [a_idx, c_idx, p_idx] = w2v.rand_word_pairs(all_phrases, batch_size, context_size)

        tt = clock()
        # Feedforward through word look-up and phrase biasing/reweighting
        Xb = word_lut.feedforward(a_idx)
        Xp = phrase_layer.feedforward(Xb, p_idx)
        table_time += clock() - tt

        # Feedforward and backprop through prediction layer
        Yb = class_layer.feedforward(Xp)

        # Compute and display loss from time-to-time (for diagnostics)
        if ((b % 20) == 0):
            L = class_layer.check_loss(Yb, c_idx)
            print("Batch {0:d}, loss {1:.4f}".format(b, L))
        if ((b % 1000) == 0):
            word_lut.reset_moms()
            class_layer.reset_moms()
            phrase_layer.reset_moms()

        # Backprop through prediction layer
        dLdXp = class_layer.backprop_sm(c_idx)

        tt = clock()
        # Backprop through phrase biasing and reweighting
        dLdXb = phrase_layer.backprop(gp.as_numpy_array(dLdXp))
        # Backprop through word look-up-table
        word_lut.backprop(dLdXb)
        table_time += clock() - tt

        # Update parameters based on the gradients for this batch
        word_lut.l2_regularize(lam_l2)
        class_layer.l2_regularize(lam_l2)
        phrase_layer.l2_regularize(lam_l2)
        word_lut.apply_grads(learn_rate=1e-3, ada_smooth=1e-3)
        class_layer.apply_grads(learn_rate=1e-3, ada_smooth=1e-3)
        phrase_layer.apply_grads(learn_rate=1e-3, ada_smooth=1e-3)

    t2 = clock()
    e_time = t2 - t1
    print("Word count: {0:d}, word dim: {1:d}".format(word_count, embed_dim))
    print("elapsed time: {0:.4f}".format(e_time))
    print("look-up time: {0:.4f}".format(table_time))
    print("softmax time: {0:.4f}".format(class_layer.comp_time))
    print("Words per second: {0:.4f}".format((1.0*batch_count*batch_size /
        e_time)))



##############
# EYE BUFFER #
##############
