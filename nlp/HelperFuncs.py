from __future__ import absolute_import

import numpy as np
import numpy.random as npr
import numba

###########################
# GENERATE TYPED MATRICES #
###########################

def randn(shape, dtype=np.float32):
    return npr.randn(shape[0], shape[1]).astype(dtype)

def ones(shape, dtype=np.float32):
    return np.ones(shape, dtype=dtype)

def zeros(shape, dtype=np.float32):
    return np.zeros(shape, dtype=dtype)


################################
# TRAINING DATA SAMPLING STUFF #
################################

def rand_word_seqs(phrase_list, seq_count, seq_len, null_key):
    """Sample LUT key n-grams from a list of phrases.

    Given a list of phrases, where each phrase is described by a list of
    keys into a look-up-table, sample random n-grams of keys from the
    sequences, where n is given by seq_len. The sampled n-grams are only
    constrained to have their final item inside the source phrase. When any
    of the first n-1 items in a sampled sequence are not in the source phrase,
    they are assigned the key given by null_key.
    """
    phrase_count = len(phrase_list)
    seq_keys = np.zeros((seq_count, seq_len), dtype=np.uint32)
    phrase_keys = np.zeros((seq_count,), dtype=np.uint32)
    for i in range(seq_count):
        phrase_keys[i] = npr.randint(0, phrase_count)
        phrase = phrase_list[phrase_keys[i]]
        phrase_len = len(phrase)
        final_key = npr.randint(0, phrase_len)
        seq_keys[i,-1] = phrase[final_key]
        for j in range(seq_len-1):
            preceding_key = final_key - seq_len + j + 1
            if (preceding_key < 0):
                seq_keys[i,j] = null_key
            else:
                seq_keys[i,j] = phrase[preceding_key]
    seq_keys = seq_keys.astype(np.uint32)
    phrase_keys = phrase_keys.astype(np.uint32)
    return [seq_keys, phrase_keys]

def rand_word_pairs(phrase_list, pair_count, context_size):
    """Sample anchor/context LUT key pairs for skip-gram training.

    Parameters:
        phrase_list: list of lists/vectors of LUT keys representing phrases
        pair_count: number of training pairs of LUT keys to sample
        context_size: half-width of context window to sample positives from
            NOTE: Samples are always drawn uniformly from within a context
                  window that was already clipped to fit the source phrase.
    Outputs:
        anchor_keys: vector of np.uint32 (samp_count,)
        context_keys: vector of np.uint32 (samp_count,)
        phrase_keys: vector of np.uint32 (samp_count,)
    """
    phrase_count = len(phrase_list)
    anchor_keys = np.zeros((pair_count,), dtype=np.uint32)
    context_keys = np.zeros((pair_count,), dtype=np.uint32)
    phrase_keys = np.zeros((pair_count,), dtype=np.uint32)
    for i in range(pair_count):
        phrase_keys[i] = npr.randint(0, phrase_count)
        phrase = phrase_list[phrase_keys[i]]
        phrase_len = len(phrase)
        a_idx = npr.randint(0, phrase_len)
        c_max = min((a_idx+context_size+1), phrase_len)
        c_min = max((a_idx-context_size), 0)
        c_idx = a_idx
        while (c_idx == a_idx):
            c_idx = npr.randint(c_min, c_max)
        anchor_keys[i] = phrase[a_idx]
        context_keys[i] = phrase[c_idx]
    anchor_keys = anchor_keys.astype(np.uint32)
    context_keys = context_keys.astype(np.uint32)
    phrase_keys =  phrase_keys.astype(np.uint32)
    return [anchor_keys, context_keys, phrase_keys]

def rand_pos_neg(phrase_list, all_words, samp_count, context_size, neg_count):
    """Sample LUT key tuples for skip-gram training via negative sampling.

    Parameters:
        phrase_list: list of lists/vectors of LUT keys representing phrases
        all_words: set of LUT keys to sample uniformly for negative examples
        samp_count: number of training tuples of LUT keys to sample
        context_size: half-width of context window to sample positives from
        neg_count: number of negative samples to draw for each positive one
    Outputs:
        anchor_keys: vector of np.uint32 (samp_count,)
        pos_keys: vector of np.uint32 (samp_count,)
        neg_keys: matrix of np.uint32 (samp_count, neg_count)
        phrase_keys: vector of np.uint32 (samp_count,)
    """
    phrase_count = len(phrase_list)
    #max_len = np.max(np.asarray([p.size for p in phrase_list]))
    #phrase_probs = np.asarray([float(p.size) / max_len for p in phrase_list])
    word_count = len(all_words)
    anchor_keys = np.zeros((samp_count,), dtype=np.uint32)
    pos_keys = np.zeros((samp_count,), dtype=np.uint32)
    neg_keys = np.zeros((samp_count,neg_count), dtype=np.uint32)
    phrase_keys = np.zeros((samp_count,), dtype=np.uint32)
    for i in range(samp_count):
        phrase_keys[i] = npr.randint(0, high=phrase_count)
        # rejection sample phrases in proprtion to their length
        #while (npr.rand() > phrase_probs[phrase_keys[i]]):
        #    phrase_keys[i] = npr.randint(0, high=phrase_count)
        phrase = phrase_list[phrase_keys[i]]
        phrase_len = len(phrase)
        a_idx = npr.randint(0, high=phrase_len)
        c_max = min((a_idx+context_size), phrase_len-1)
        c_min = max((a_idx-context_size), 0)
        c_idx = a_idx
        while (c_idx == a_idx):
            c_idx = npr.randint(c_min, high=(c_max+1))
        # Record the anchor word and its positive context word
        anchor_keys[i] = phrase[a_idx]
        pos_keys[i] = phrase[c_idx]
        # Sample a random negative example from the full word list
        n_idx = npr.randint(0, high=word_count, size=(1,neg_count))
        neg_keys[i,:] = all_words[n_idx]
    anchor_keys = anchor_keys.astype(np.uint32)
    pos_keys = pos_keys.astype(np.uint32)
    neg_keys = neg_keys.astype(np.uint32)
    phrase_keys = phrase_keys.astype(np.uint32)
    return [anchor_keys, pos_keys, neg_keys, phrase_keys]


@numba.jit("void(i4[:], i8, i8, i8, i4[:], i4[:], i4[:], i4[:])")
def fast_pair_sample(phrase, max_window, i, repeats, anc_keys, pos_keys, rand_pool, ri):
    phrase_len = phrase.size
    max_key_idx = anc_keys.shape[0]
    for r in range(repeats):
        j = i + r
        a_idx = rand_pool[ri[0]] % phrase_len
        ri[0] += 1
        red_win = (rand_pool[ri[0]] % max_window) + 1
        ri[0] += 1
        c_min = a_idx - red_win
        if (c_min < 0):
            c_min = 0
        c_max = a_idx + red_win
        if (c_max >= phrase_len):
            c_max = phrase_len - 1
        c_span = c_max - c_min + 1
        c_idx = a_idx
        while (c_idx == a_idx):
            c_idx = c_min + (rand_pool[ri[0]] % c_span)
            ri[0] += 1
        anc_keys[j] = phrase[a_idx]
        pos_keys[j] = phrase[c_idx]
    return

class PNSampler:
    """This samples words from a corpus for training via negative sampling.
    """
    def __init__(self, phrase_list, all_words, max_window, neg_count):
        # phrase_list contains the phrases to sample from 
        self.phrase_list = phrase_list
        self.phrase_lens = np.asarray([p.size for p in phrase_list])
        max_len = np.max(self.phrase_lens)
        self.phrase_probs = np.asarray([pl/max_len for pl in self.phrase_lens])
        self.phrase_count = len(self.phrase_list)
        #
        self.max_window = max_window
        # neg_table contains the words to sample as negative examples
        self.neg_table = all_words
        self.neg_table.reshape((self.neg_table.size,))
        self.neg_table_size = all_words.size
        self.neg_count = neg_count
        return

    def sample_negatives(self, sample_count):
        neg_keys = np.zeros((sample_count,self.neg_count), dtype=np.uint32)
        neg_idx = npr.randint(0, high=self.neg_table_size, size=neg_keys.shape)
        for i in range(neg_keys.shape[1]):
            neg_keys[:,i] = self.neg_table[neg_idx[:,i]]
        neg_keys = neg_keys.astype(np.uint32)
        return neg_keys

    def sample(self, sample_count):
        """Draw a sample."""
        anc_keys = np.zeros((sample_count,), dtype=np.uint32)
        pos_keys = np.zeros((sample_count,), dtype=np.uint32)
        phrase_keys = np.zeros((sample_count,), dtype=np.uint32)
        rand_pool = npr.randint(0, high=1e6, size=(10*sample_count,)).astype(np.uint32)
        ri = np.asarray([0]).astype(np.uint32)
        repeats = 5
        if not ((sample_count % repeats) == 0):
            repeats = 1
        for i in range(0, sample_count, repeats):
            phrase_keys[i] = npr.randint(0, high=self.phrase_count)
            fast_pair_sample(self.phrase_list[phrase_keys[i]], self.max_window, \
                             i, repeats, anc_keys, pos_keys, rand_pool, ri)
        # Sample negative examples from self.neg_table
        anc_keys = anc_keys.astype(np.uint32)
        pos_keys = pos_keys.astype(np.uint32)
        neg_keys = self.sample_negatives(sample_count)
        phrase_keys = phrase_keys.astype(np.uint32)
        return [anc_keys, pos_keys, neg_keys, phrase_keys]


#for j in range(c_min, c_max+1):
#    if (j == a_idx):
#        continue
#    else:
#        # Record the anchor word and its positive context word
#        phrase_keys[i] = phrase_idx
#        anchor_keys[i] = self.phrase_list[phrase_idx][a_idx]
#        pos_keys[i] = self.phrase_list[phrase_idx][c_min+c_loc]
#        i += 1
#    if (i >= sample_count):
#        break





##############
# EYE BUFFER #
##############
