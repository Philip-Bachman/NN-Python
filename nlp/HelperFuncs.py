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
        chunklen = (length + 1) // numthreads
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

def catch_oov_words(w_idx, v_idx, oov_idx):
    """w_idx is an ndarray, v_idx is a set, and oov_idx is an int32."""
    assert((w_idx.ndim == 1) or (w_idx.ndim == 2))
    # Brute-force convert untrained words to OOV word
    if (w_idx.ndim == 1):
        for i in range(w_idx.shape[0]):
            if not (w_idx[i] in v_idx):
                w_idx[i] = oov_idx
    else:
        for i in range(w_idx.shape[0]):
            for j in range(w_idx.shape[1]):
                if not (w_idx[i,j] in v_idx):
                    w_idx[i,j] = oov_idx
    return w_idx

#######################
# RANDOM KNICK-KNACKS #
#######################

def rand_word_seqs(phrase_list, seq_count, seq_len, non_idx):
    """Sample random sequences of contiguous words from a list of phrases.

    Given a list of phrases, where each phrase is described by a list of
    indices into a look-up-table, sample random subsequences of the phrases,
    for use in training a forward prediction n-gram model (or whatever).
    """
    phrase_count = len(phrase_list)
    seq_idx = np.zeros((seq_count, seq_len), dtype=np.int32)
    phrase_idx = np.zeros((seq_count,), dtype=np.int32)
    for i in range(seq_count):
        phrase_idx[i] = npr.randint(0, phrase_count)
        phrase = phrase_list[phrase_idx[i]]
        phrase_len = len(phrase)
        predictee_idx = npr.randint(0, phrase_len)
        seq_idx[i,-1] = predictee_idx
        for j in range(seq_len-1):
            predictor_idx = predictee_idx - seq_len + j + 1
            if (predictor_idx < 0):
                seq_idx[i,j] = non_idx
            else:
                seq_idx[i,j] = phrase[predictor_idx]
    return [seq_idx, phrase_idx]

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
    phrase_idx = np.zeros((pair_count,), dtype=np.int32)
    for i in range(pair_count):
        phrase_idx[i] = npr.randint(0, phrase_count)
        phrase = phrase_list[phrase_idx[i]]
        phrase_len = len(phrase)
        a_idx = npr.randint(0, phrase_len)
        c_max = min((a_idx+context_size+1), phrase_len)
        c_min = max((a_idx-context_size), 0)
        c_idx = a_idx
        while (c_idx == a_idx):
            c_idx = npr.randint(c_min, c_max)
        anchor_idx[i] = phrase[a_idx]
        context_idx[i] = phrase[c_idx]
    return [anchor_idx, context_idx, phrase_idx]

def rand_pos_neg(phrase_list, all_words, pair_count, context_size, neg_count):
    """Sample random anchor/positive/negative tuples for training a skip-gram
    model using "negative sampling" (i.e. "fake" noise-contrastive...).
    """
    phrase_count = len(phrase_list)
    word_count = len(all_words)
    anchor_idx = np.zeros((pair_count,), dtype=np.int32)
    pos_idx = np.zeros((pair_count,), dtype=np.int32)
    neg_idx = np.zeros((pair_count,neg_count), dtype=np.int32)
    phrase_idx = np.zeros((pair_count,), dtype=np.int32)
    for i in range(pair_count):
        phrase_idx[i] = npr.randint(0, high=phrase_count)
        phrase = phrase_list[phrase_idx[i]]
        phrase_len = len(phrase)
        a_idx = npr.randint(0, high=phrase_len)
        c_max = min((a_idx+context_size+1), phrase_len)
        c_min = max((a_idx-context_size), 0)
        c_idx = a_idx
        while (c_idx == a_idx):
            c_idx = npr.randint(c_min, high=c_max)
        # Record the anchor word and its positive context word
        anchor_idx[i] = phrase[a_idx]
        pos_idx[i] = phrase[c_idx]
        # Sample a random negative example from the full word list
        n_idx = npr.randint(0, high=word_count, size=(1,neg_count))
        neg_idx[i,:] = all_words[n_idx]
    return [anchor_idx, pos_idx, neg_idx, phrase_idx]

##############
# EYE BUFFER #
##############
