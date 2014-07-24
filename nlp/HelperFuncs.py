from __future__ import absolute_import

import numpy as np
import numpy.random as npr
import threading
import numba
from math import exp, log, sqrt
from numba import jit, void, i4, f8
from ctypes import pythonapi, c_void_p

########################################
# MULTITHREADING HELPER-FUNC AND DEFNS #
########################################

THREAD_NUM = 2

savethread = pythonapi.PyEval_SaveThread
savethread.argtypes = []
savethread.restype = c_void_p

restorethread = pythonapi.PyEval_RestoreThread
restorethread.argtypes = [c_void_p]
restorethread.restype = None

def make_multithread(inner_func, numthreads):
    def func_mt(*args):
        length = len(args[0])
        sp_idx = np.arange(0,length).astype(np.int32)
        chunklen = (length + (numthreads-1)) // numthreads
        chunkargs = [(sp_idx[i*chunklen:(i+1)*chunklen],)+args for i in range(numthreads)]
        # Start a thread for all but the last chunk of work
        threads = [threading.Thread(target=inner_func, args=cargs)
                   for cargs in chunkargs[:-1]]
        for thread in threads:
            thread.start()
        # Give the last chunk of work to the main thread
        inner_func(*chunkargs[-1])
        for thread in threads:
            thread.join()
        return 1
    return func_mt

##############################
# NUMBA FUNCTION DEFINITIONS #
##############################

def w2v_ff_bp_sp(sp_idx, anc_idx, pn_idx, pn_sign, Wa, Wc, b, dWa, dWc, db, L):
    """Feedforward and backprop for unified (neg-sample) word-2-vec layer."""
    threadstate = savethread()
    sp_size = sp_idx.shape[0]
    cols = pn_idx.shape[1]
    vec_dim = Wa.shape[1]
    for sp_i in range(sp_size):
        i = sp_idx[sp_i]
        ai = anc_idx[i]
        for j in range(cols):
            ci = pn_idx[i,j]
            y = b[ci]
            for k in range(vec_dim):
                y += (Wa[ai,k] * Wc[ci,k])
            exp_pns_y = exp(pn_sign[i,j] * y)
            L[0] += log(1.0 + exp_pns_y)
            dLdy = pn_sign[i,j] * (exp_pns_y / (1.0 + exp_pns_y))
            db[ci] = db[ci] + dLdy
            for k in range(vec_dim):
                dWa[ai,k] += (dLdy * Wc[ci,k])
                dWc[ci,k] += (dLdy * Wa[ai,k])
    restorethread(threadstate)
    return
fn_sig_1 = void(i4[:], i4[:], i4[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:], f8[:,:], f8[:,:], f8[:], f8[:])
w2v_ff_bp_st = jit(fn_sig_1, nopython=True)(w2v_ff_bp_sp)
w2v_ff_bp = make_multithread(w2v_ff_bp_st, THREAD_NUM)

def nsl_bp_sp(sp_idx, table_idx, X, W, dLdY, dLdX, dW, db):
    """Backprop for NSLayer: main loop in Numba-friendly form."""
    threadstate = savethread()
    rows = sp_idx.shape[0]
    cols = dLdY.shape[1]
    vec_dim = X.shape[1]
    for spi in range(rows):
        i = sp_idx[spi]
        for j in range(cols):
            dldy = dLdY[i,j]
            idx = table_idx[i,j]
            db[idx] += dldy
            for k in range(vec_dim):
                dW[idx,k] += dldy * X[i,k]
                dLdX[i,k] += dldy * W[idx,k]
    restorethread(threadstate)
    return
fn_sig_2 = void(i4[:], i4[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:])
nsl_bp_st = jit(fn_sig_2, nopython=True)(nsl_bp_sp)
nsl_bp = make_multithread(nsl_bp_st, THREAD_NUM)

def nsl_ff_sp(sp_idx, table_idx, X, W, b, Y):
    """Feedforward for NSLayer: main loop in Numba-friendly form."""
    threadstate = savethread()
    rows = sp_idx.shape[0]
    cols = table_idx.shape[1]
    vec_dim = X.shape[1]
    for spi in range(rows):
        i = sp_idx[spi]
        for j in range(cols):
            idx = table_idx[i,j]
            Y[i,j] = b[idx]
            for k in range(vec_dim):
                Y[i,j] += X[i,k] * W[idx,k]
    restorethread(threadstate)
    return
fn_sig_3 = void(i4[:], i4[:,:], f8[:,:], f8[:,:], f8[:], f8[:,:])
nsl_ff_st = jit(fn_sig_3, nopython=True)(nsl_ff_sp)
nsl_ff = make_multithread(nsl_ff_st, THREAD_NUM)

def ag_update_2d_sp(sp_idx, row_idx, W, dW, mW, learn_rate, ada_smooth, lam_l2):
    """Element-wise partial update ala adagrad, with l2 regularization.

    For the entries indicated by row_idx, this first updates the grads in dW to
    account for l2 regularization with weight lam_l2, then updates the adagrad
    sums-of-squares in mW, then updates the params in W, and finally sets the
    grads in dW back to 0.
    """
    threadstate = savethread()
    row_count = sp_idx.shape[0]
    vec_dim = W.shape[1]
    for spi in range(row_count):
        idx = row_idx[sp_idx[spi]]
        for j in range(vec_dim):
            dW[idx,j] += (lam_l2 * W[idx,j])
            mW[idx,j] += dW[idx,j] * dW[idx,j]
            W[idx,j] -= (learn_rate * (dW[idx,j] / (sqrt(mW[idx,j]) + ada_smooth)))
            dW[idx,j] = 0.0
    restorethread(threadstate)
    return
fn_sig_4 = void(i4[:], i4[:], f8[:,:], f8[:,:], f8[:,:], f8, f8, f8)
ag_update_2d_st = jit(fn_sig_4, nopython=True)(ag_update_2d_sp)
ag_update_2d = make_multithread(ag_update_2d_st, THREAD_NUM)

@numba.jit("void(i4[:], f8[:], f8[:], f8[:], f8, f8, f8)")
def ag_update_1d(row_idx, W, dW, mW, learn_rate, ada_smooth, lam_l2):
    """Element-wise partial update ala adagrad, with l2 regularization.

    For the entries indicated by row_idx, this first updates the grads in dW to
    account for l2 regularization with weight lam_l2, then updates the adagrad
    sums-of-squares in mW, then updates the params in W, and finally sets the
    grads in dW back to 0.
    """
    row_count = row_idx.shape[0]
    for i in range(row_count):
        idx = row_idx[i]
        dW[idx] += lam_l2 * W[idx]
        mW[idx] += dW[idx] * dW[idx]
        W[idx] -= learn_rate * (dW[idx] / (sqrt(mW[idx]) + ada_smooth))
        dW[idx] = 0.0
    return

def lut_sp(sp_idx, row_idx, dLdY, dW):
    """Simple row-wise updates for adjusting dW with dLdY.

    This adds each row of dLdY to some row of dW. The row of dW to adjust
    is given by the corresponding item in row_idx."""
    threadstate = savethread()
    row_count = sp_idx.shape[0]
    vec_dim = dW.shape[1]
    for i in range(row_count):
        idx = row_idx[sp_idx[i]]
        for j in range(vec_dim):
            dW[idx,j] += dLdY[i,j]
    restorethread(threadstate)
    return
fn_sig_5 = void(i4[:], i4[:], f8[:,:], f8[:,:])
lut_st = jit(fn_sig_5, nopython=True)(lut_sp)
lut_bp = make_multithread(lut_st, THREAD_NUM)

###########################################
# WORD VALIDATION, FOR MANAGING OOV WORDS #
###########################################

def catch_oov_words(w_keys, v_keys, oov_key):
    """Set each entry in w_keys that is not in v_keys to oov_key.

    Note: w_keys should be a 1d or 2d numpy array of np.int32, v_keys should
          be a _set_ of np.int32, and oov_key should be a lone np.int32.
    """
    assert((w_keys.ndim == 1) or (w_keys.ndim == 2))
    if (w_keys.ndim == 1):
        for i in range(w_keys.shape[0]):
            if not (w_keys[i] in v_keys):
                w_keys[i] = oov_key
    else:
        for i in range(w_keys.shape[0]):
            for j in range(w_keys.shape[1]):
                if not (w_keys[i,j] in v_keys):
                    w_keys[i,j] = oov_key
    return w_keys

#######################
# RANDOM KNICK-KNACKS #
#######################

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
    seq_keys = np.zeros((seq_count, seq_len), dtype=np.int32)
    phrase_keys = np.zeros((seq_count,), dtype=np.int32)
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
    seq_keys = seq_keys.astype(np.int32)
    phrase_keys = phrase_keys.astype(np.int32)
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
        anchor_keys: vector of np.int32 (samp_count,)
        context_keys: vector of np.int32 (samp_count,)
        phrase_keys: vector of np.int32 (samp_count,)
    """
    phrase_count = len(phrase_list)
    anchor_keys = np.zeros((pair_count,), dtype=np.int32)
    context_keys = np.zeros((pair_count,), dtype=np.int32)
    phrase_keys = np.zeros((pair_count,), dtype=np.int32)
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
    anchor_keys = anchor_keys.astype(np.int32)
    context_keys = context_keys.astype(np.int32)
    phrase_keys =  phrase_keys.astype(np.int32)
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
        anchor_keys: vector of np.int32 (samp_count,)
        pos_keys: vector of np.int32 (samp_count,)
        neg_keys: matrix of np.int32 (samp_count, neg_count)
        phrase_keys: vector of np.int32 (samp_count,)
    """
    phrase_count = len(phrase_list)
    max_len = np.max(np.asarray([p.size for p in phrase_list]))
    phrase_probs = np.asarray([float(p.size) / max_len for p in phrase_list])
    word_count = len(all_words)
    anchor_keys = np.zeros((samp_count,), dtype=np.int32)
    pos_keys = np.zeros((samp_count,), dtype=np.int32)
    neg_keys = np.zeros((samp_count,neg_count), dtype=np.int32)
    phrase_keys = np.zeros((samp_count,), dtype=np.int32)
    for i in range(samp_count):
        phrase_keys[i] = npr.randint(0, high=phrase_count)
        # rejection sample phrases in proprtion to their length
        while (npr.rand() > phrase_probs[phrase_keys[i]]):
            phrase_keys[i] = npr.randint(0, high=phrase_count)
        phrase = phrase_list[phrase_keys[i]]
        phrase_len = len(phrase)
        a_idx = npr.randint(0, high=phrase_len)
        c_max = min((a_idx+context_size+1), phrase_len)
        c_min = max((a_idx-context_size), 0)
        c_idx = a_idx
        while (c_idx == a_idx):
            c_idx = npr.randint(c_min, high=c_max)
        # Record the anchor word and its positive context word
        anchor_keys[i] = phrase[a_idx]
        pos_keys[i] = phrase[c_idx]
        # Sample a random negative example from the full word list
        n_idx = npr.randint(0, high=word_count, size=(1,neg_count))
        neg_keys[i,:] = all_words[n_idx]
    anchor_keys = anchor_keys.astype(np.int32)
    pos_keys = pos_keys.astype(np.int32)
    neg_keys = neg_keys.astype(np.int32)
    phrase_keys = phrase_keys.astype(np.int32)
    return [anchor_keys, pos_keys, neg_keys, phrase_keys]









##############
# EYE BUFFER #
##############
