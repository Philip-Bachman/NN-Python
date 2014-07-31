from __future__ import absolute_import

import numpy as np
import numpy.random as npr
import threading
import numba
from math import exp, log, sqrt
from numba import jit, void, i4, f4
from ctypes import pythonapi, c_void_p

ADA_EPS = 0.001

########################################
# MULTITHREADING HELPER-FUNC AND DEFNS #
########################################

THREAD_NUM = 4

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
    def func_st(*args):
        length = len(args[0])
        sp_idx = np.arange(0,length).astype(np.int32)
        sp_args = (sp_idx,) + args
        inner_fun(*sp_args)
    func = None
    if numthreads == 1:
        func = func_st
    else:
        func = func_mt
    return func_mt

##############################
# NUMBA FUNCTION DEFINITIONS #
##############################

def w2v_ff_bp_sp(sp_idx, anc_idx, pn_idx, pn_sign, Wa, Wc, b, dWa, dWc, db, L, do_grad):
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
            if (do_grad == 1):
                dLdy = pn_sign[i,j] * (exp_pns_y / (1.0 + exp_pns_y))
                db[ci] = db[ci] + dLdy
                for k in range(vec_dim):
                    dWa[ai,k] += (dLdy * Wc[ci,k])
                    dWc[ci,k] += (dLdy * Wa[ai,k])
    restorethread(threadstate)
    return
fn_sig_1 = void(i4[:], i4[:], i4[:,:], f4[:,:], f4[:,:], f4[:,:], f4[:], f4[:,:], f4[:,:], f4[:], f4[:], i4)
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
fn_sig_2 = void(i4[:], i4[:,:], f4[:,:], f4[:,:], f4[:,:], f4[:,:], f4[:,:], f4[:])
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
fn_sig_3 = void(i4[:], i4[:,:], f4[:,:], f4[:,:], f4[:], f4[:,:])
nsl_ff_st = jit(fn_sig_3, nopython=True)(nsl_ff_sp)
nsl_ff = make_multithread(nsl_ff_st, THREAD_NUM)

def ag_update_2d_sp(sp_idx, row_idx, W, dW, mW, learn_rate):
    """Element-wise partial update ala adagrad.

    For the entries indicated by row_idx, this first updates the adagrad sums
    of squares in mW, then updates the params in W, and finally sets the
    grads in dW back to 0.
    """
    threadstate = savethread()
    row_count = sp_idx.shape[0]
    vec_dim = W.shape[1]
    for spi in range(row_count):
        idx = row_idx[sp_idx[spi]]
        for j in range(vec_dim):
            mW[idx,j] = (0.95 * mW[idx,j]) + (0.05 * dW[idx,j] * dW[idx,j])
            W[idx,j] -= (learn_rate * (dW[idx,j] / (sqrt(mW[idx,j]) + ADA_EPS)))
            dW[idx,j] = 0.0
    restorethread(threadstate)
    return
fn_sig_4 = void(i4[:], i4[:], f4[:,:], f4[:,:], f4[:,:], f4)
ag_update_2d_st = jit(fn_sig_4, nopython=True)(ag_update_2d_sp)
ag_update_2d = make_multithread(ag_update_2d_st, THREAD_NUM)

@numba.jit("void(i4[:], f4[:], f4[:], f4[:], f4)")
def ag_update_1d(row_idx, W, dW, mW, learn_rate):
    """Element-wise partial update ala adagrad.

    For the entries indicated by row_idx, this first updates the adagrad sums
    of squares in mW, then updates the params in W, and finally sets the
    grads in dW back to 0.
    """
    row_count = row_idx.shape[0]
    for i in range(row_count):
        idx = row_idx[i]
        mW[idx] = (0.95 * mW[idx]) + (0.05 * dW[idx] * dW[idx])
        W[idx] -= learn_rate * (dW[idx] / (sqrt(mW[idx]) + ADA_EPS))
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
fn_sig_5 = void(i4[:], i4[:], f4[:,:], f4[:,:])
lut_st = jit(fn_sig_5, nopython=True)(lut_sp)
lut_bp = make_multithread(lut_st, THREAD_NUM)



##############
# EYE BUFFER #
##############
