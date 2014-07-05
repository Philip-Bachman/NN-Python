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

def rand_word_pairs(phrase_list, pair_count, context_size):
    """Sample random anchor/context pairs for skip-gram training.

    Given a list of phrases, where each phrase is described by a list of
    indices into a look-up-table, sample random pairs of anchor word and
    context word for training a skip-gram model. The skip-gram objective is
    to predict the context word given the anchor word. The "context_size"
    determines the max separation between sampled context words and their
    corresponding anchor.
    """
    phrase_count = len(phrase_list)
    anchor_idx = np.zeros((pair_count,), dtype=np.int32)
    context_idx = np.zeros((pair_count,), dtype=np.int32)
    for i in range(pair_count):
        phrase_idx = random.randint(0, phrase_count-1)
        phrase = phrase_list[phrase_idx]
        phrase_len = len(phrase)
        a_idx = random.randint(0, phrase_len-1)
        c_max = min((a_idx+context_size), phrase_len-1)
        c_min = max((a_idx-context_size), 0)
        c_idx = random.randint(c_min, c_max)
        anchor_idx[i] = a_idx
        context_idx[i] = c_idx
    return [anchor_idx, context_idx]

if __name__ == '__main__':
    # Load tree data
    tree_dir = './trees'
    stb_data = st.SimpleLoad(tree_dir)
    max_lut_idx = max(stb_data['lut_keys'].values())

    all_phrases = stb_data['train_full_phrases']

    batch_count = 1000
    batch_size = 500
    context_size = 5
    word_count = max_lut_idx + 1
    embed_dim = 100

    # Create a lookup table for word representations
    word_lut = w2v.LUTLayer(word_count, embed_dim)
    tanh_layer = w2v.TanhLayer(in_layer=word_lut)
    noise_layer = w2v.NoiseLayer(in_layer=tanh_layer, drop_rate=0.0, fuzz_scale=0.0)

    # Create a full/softmax layer for classification
    class_layer = w2v.GPUFullLayer(in_layer=False, in_dim=embed_dim, out_dim=word_count)

    # Initialize params for the LUT and softmax classifier
    word_lut.init_params(0.05)
    class_layer.init_params(0.05)

    table_time = 0.0
    print("Processing batches:")
    t1 = clock()
    for b in range(batch_count):
        # Sample a batch of random anchor/context prediction pairs for
        # training a skip-gram model.
        [a_idx, c_idx] = rand_word_pairs(all_phrases, batch_size, context_size)

        tt = clock()
        # Feedforward through word look-up, tanh, and noise
        word_lut.feedforward(a_idx, auto_prop=True)
        Xb = noise_layer.Y
        table_time += clock() - tt

        # Feedforward through classification/prediction layer
        Yb = class_layer.feedforward(Xb)
        Y_ind = np.zeros(Yb.shape)
        Y_ind[np.arange(Y_ind.shape[0]), c_idx] = 1.0

        # Get get gradients on output of classification/prediction layer
        dLdYb = class_layer.cross_entropy(Yb, Y_ind)
        if ((b % 10) == 0):
            L = class_layer.check_loss(Yb, Y_ind)
            print("Batch {0:d}, loss {1:.4f}".format(b, L))

        # Backprop through classification/prediction layer
        dLdXb = class_layer.backprop(dLdYb)
        # Update softmax/prediction params based on batch gradients
        class_layer.apply_grads(learn_rate=2e-3, ada_smooth=1e-3)

        tt = clock()
        # Backprop through word look-up-table, tanh, and noise
        noise_layer.backprop(gp.as_numpy_array(dLdXb), auto_prop=True)
        # Update look-up-table based on gradients for this batch
        word_lut.apply_grads(learn_rate=2e-3, ada_smooth=1e-3)
        table_time += clock() - tt

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
