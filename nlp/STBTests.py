import numpy as np
import numpy.random as npr
import StanfordTrees as st
import LNLayers as lnl
import LayerNets as ln
import random as random
from time import clock
from sys import stdout as stdout

def simple_stb_test(tree_dir='./trees'):
    stb_data = st.SimpleLoad(tree_dir)
    return

if __name__ == '__main__':
    tree_dir = './trees'
    stb_data = st.SimpleLoad(tree_dir)
    max_lut_idx = max(stb_data['lut_keys'].values())
    basic_opts = {}
    basic_opts['class_count'] = 5
    lut_opts = {}
    lut_opts['max_key'] = max_lut_idx
    lut_opts['embed_dim'] = 25
    lut_opts['max_norm'] = 2.0
    basic_opts['lut_layer'] = lut_opts

    # Initialize a network
    KMN = ln.KMaxNet(basic_opts)
    KMN.init_weights(w_scale=0.05, b_shift=0.1)

    # Get a "flattened" list of training phrases and classes
    train_phrases = []
    train_labels = []
    for (phrases, labels) in zip(stb_data['train_phrases'], stb_data['train_labels']):
        train_phrases.extend(phrases)
        train_labels.extend(labels)

    batch_size = 128
    epoch_batches = 2000
    learn_rate = 0.01
    train_pairs = [(phrase, label) for (phrase, label) in zip(train_phrases, train_labels)]
    train_phrases = []
    train_labels = []
    for e in range(500):
        print("Starting epoch {0:d}, {1:d} batches".format(e, len(train_pairs)/batch_size))
        stdout.flush()
        # Reset batch extraction indices and completed batch counter
        batch_start = 0
        batch_end = batch_start + batch_size
        completed_batches = 0
        # Perform batch updates for the current epoch
        L = 0.0
        t1 = clock()
        random.shuffle(train_pairs)
        if ((e % 5) == 0):
            KMN.reset_moms(ada_init=0.0, clear_moms=False)
        while ((batch_end < len(train_pairs)) and (completed_batches < epoch_batches)):
            # Extract the current training phrase/label batch
            batch_pairs = train_pairs[batch_start:batch_end]
            # Train on this batch, and count its completion
            Xb = [pair[0] for pair in batch_pairs]
            Yb = [pair[1] for pair in batch_pairs]
            L += KMN.process_training_batch(Xb, Yb, learn_rate, use_dropout=False)
            completed_batches += 1
            # Advance batch extraction indices
            batch_start = batch_start + batch_size
            batch_end = batch_start + batch_size
            # Print diagnostic info from time-to-time
            if ((completed_batches % 50) == 0):
                print("completed {0:d} updates, with average loss {1:.4f}".format( \
                        completed_batches, (L / 50.0)))
                L = 0.0
                t2 = clock()
                print("-- time: {0:.2f}".format(t2-t1))
                t1 = clock()
                stdout.flush()



##############
# EYE BUFFER #
##############
