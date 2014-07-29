#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# coding: utf-8
#

import cython
import numpy as np
cimport numpy as np

from libc.math cimport exp, log
from libc.string cimport memset

cdef extern from "voidptr.h":
    void* PyCObject_AsVoidPtr(object obj)

from scipy.linalg import blas

ctypedef np.float32_t REAL_t
ctypedef np.uint32_t UI32_t
ctypedef np.int32_t I32_t

DEF MAX_SENTENCE_LEN = 10000

ctypedef void (*scopy_ptr) (const int *N, const float *X, const int *incX, float *Y, const int *incY) nogil
ctypedef void (*saxpy_ptr) (const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil
ctypedef float (*sdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef double (*dsdot_ptr) (const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil
ctypedef double (*snrm2_ptr) (const int *N, const float *X, const int *incX) nogil
ctypedef void (*sscal_ptr) (const int *N, const float *alpha, const float *X, const int *incX) nogil


ctypedef void (*cy_w2v_ff_bp_ptr) (
    const int sp_size, const I32_t *sp_idx, const I32_t *anc_keys,
    const int pn_size, const I32_t *pn_keys, REAL_t *pn_sign,
    REAL_t *Wa, REAL_t *Wc, REAL_t *b, REAL_t *dWa, REAL_t *dWc, REAL_t *db,
    REAL_t *L, const int do_grad, const int vec_dim) nogil

ctypedef void (*cy_nsl_ff_bp_ptr) (
    const int sp_size, const I32_t *sp_idx,
    const int pn_size, const I32_t *pn_keys, REAL_t *pn_sign,
    REAL_t *X, REAL_t *W, REAL_t *b,
    REAL_t *dX, REAL_t *dW, REAL_t *db,
    REAL_t *L, const int vec_dim) nogil

cdef scopy_ptr scopy=<scopy_ptr>PyCObject_AsVoidPtr(blas.scopy._cpointer)  # y = x
cdef saxpy_ptr saxpy=<saxpy_ptr>PyCObject_AsVoidPtr(blas.saxpy._cpointer)  # y += alpha * x
cdef sdot_ptr sdot=<sdot_ptr>PyCObject_AsVoidPtr(blas.sdot._cpointer)  # float = dot(x, y)
cdef dsdot_ptr dsdot=<dsdot_ptr>PyCObject_AsVoidPtr(blas.sdot._cpointer)  # double = dot(x, y)
cdef snrm2_ptr snrm2=<snrm2_ptr>PyCObject_AsVoidPtr(blas.snrm2._cpointer)  # sqrt(x^2)
cdef sscal_ptr sscal=<sscal_ptr>PyCObject_AsVoidPtr(blas.sscal._cpointer) # x = alpha * x

cdef cy_w2v_ff_bp_ptr cy_w2v_ff_bp
cdef cy_nsl_ff_bp_ptr cy_nsl_ff_bp

# define some useful constants
cdef int ONE = 1
cdef REAL_t ONEF = <REAL_t>1.0

#############
# W2V_FF_BP #
#############

cdef void cy_w2v_ff_bp0(
    const int sp_size, const I32_t *sp_idx, const I32_t *anc_keys,
    const int pn_size, const I32_t *pn_keys, REAL_t *pn_sign,
    REAL_t *Wa, REAL_t *Wc, REAL_t *b, REAL_t *dWa, REAL_t *dWc, REAL_t *db,
    REAL_t *L, const int do_grad, const int vec_dim) nogil:

    # declarations
    cdef long long row1, row2
    cdef REAL_t label, y, exp_pns_y, g
    cdef I32_t a_key, c_key
    cdef int sp_i, i, j

    # update loop
    for sp_i in range(sp_size):
        i = <int>sp_idx[sp_i]
        a_key = anc_keys[i] # get the LUT key for the anchor word
        row1 = a_key * vec_dim # get the starting index of anchor word's row
        for j in range(pn_size):
            c_key = pn_keys[i*pn_size + j] # get the LUT key for the context word
            row2 = c_key * vec_dim # get the starting index of context word's row
            label = pn_sign[i*pn_size + j] # get the +1/-1 label for the context word
            # compute prediction y as np.dot(a_vec, c_vec.T) + b[c_key]
            y = <REAL_t>dsdot(&vec_dim, &Wa[row1], &ONE, &Wc[row2], &ONE) + b[c_key]
            exp_pns_y = <REAL_t>exp(label * y) # this is used for loss/grad computations
            L[0] = L[0] + log(1.0 + exp_pns_y) # add the loss on this a/c pair
            if (do_grad == 1):
                # Compute gradient and update parameter gradient accumulators
                g = label * (exp_pns_y / (1.0 + exp_pns_y))
                saxpy(&vec_dim, &g, &Wa[row1], &ONE, &dWc[row2], &ONE)
                saxpy(&vec_dim, &g, &Wc[row2], &ONE, &dWa[row1], &ONE)
                db[c_key] = db[c_key] + g
    return

cdef void cy_w2v_ff_bp1(
    const int sp_size, const I32_t *sp_idx, const I32_t *anc_keys,
    const int pn_size, const I32_t *pn_keys, REAL_t *pn_sign,
    REAL_t *Wa, REAL_t *Wc, REAL_t *b, REAL_t *dWa, REAL_t *dWc, REAL_t *db,
    REAL_t *L, const int do_grad, const int vec_dim) nogil:

    # declarations
    cdef long long row1, row2
    cdef REAL_t label, y, exp_pns_y, g
    cdef I32_t a_key, c_key
    cdef int sp_i, i, j

    # update loop
    for sp_i in range(sp_size):
        i = <int>sp_idx[sp_i]
        a_key = anc_keys[i] # get the LUT key for the anchor word
        row1 = a_key * vec_dim # get the starting index of anchor word's row
        for j in range(pn_size):
            c_key = pn_keys[i*pn_size + j] # get the LUT key for the context word
            row2 = c_key * vec_dim # get the starting index of context word's row
            label = pn_sign[i*pn_size + j] # get the +1/-1 label for the context word
            # compute prediction y as np.dot(a_vec, c_vec.T) + b[c_key]
            y = <REAL_t>sdot(&vec_dim, &Wa[row1], &ONE, &Wc[row2], &ONE) + b[c_key]
            exp_pns_y = <REAL_t>exp(label * y) # this is used for loss/grad computations
            L[0] = L[0] + log(1.0 + exp_pns_y) # add the loss on this a/c pair
            if (do_grad == 1):
                # Compute gradient and update parameter gradient accumulators
                g = label * (exp_pns_y / (1.0 + exp_pns_y))
                saxpy(&vec_dim, &g, &Wa[row1], &ONE, &dWc[row2], &ONE)
                saxpy(&vec_dim, &g, &Wc[row2], &ONE, &dWa[row1], &ONE)
                db[c_key] = db[c_key] + g
    return

def w2v_ff_bp_pyx(sp_idx_p, anc_keys_p, pn_keys_p, pn_sign_p, Wa_p, Wc_p, b_p,
                  dWa_p, dWc_p, db_p, L_p, do_grad_p):
    # Define and cast minibatch problem parameters
    cdef int sp_size = <int>sp_idx_p.shape[0]
    cdef int pn_size = <int>pn_keys_p.shape[1]
    cdef int do_grad = <int>do_grad_p
    cdef int vec_dim = <int>Wa_p.shape[1]
    cdef I32_t *sp_idx = <I32_t *>(np.PyArray_DATA(sp_idx_p))
    cdef I32_t *anc_keys = <I32_t *>(np.PyArray_DATA(anc_keys_p))
    cdef I32_t *pn_keys = <I32_t *>(np.PyArray_DATA(pn_keys_p))
    cdef REAL_t *pn_sign = <REAL_t *>(np.PyArray_DATA(pn_sign_p))
    cdef REAL_t *Wa = <REAL_t *>(np.PyArray_DATA(Wa_p))
    cdef REAL_t *Wc = <REAL_t *>(np.PyArray_DATA(Wc_p))
    cdef REAL_t *b = <REAL_t *>(np.PyArray_DATA(b_p))
    cdef REAL_t *dWa = <REAL_t *>(np.PyArray_DATA(dWa_p))
    cdef REAL_t *dWc = <REAL_t *>(np.PyArray_DATA(dWc_p))
    cdef REAL_t *db = <REAL_t *>(np.PyArray_DATA(db_p))
    cdef REAL_t *L = <REAL_t *>(np.PyArray_DATA(L_p))

    with nogil:
        cy_w2v_ff_bp(sp_size, sp_idx, anc_keys, pn_size, pn_keys, pn_sign,
                     Wa, Wc, b, dWa, dWc, db, L, do_grad, vec_dim)
    return

#############
# NSL_FF_BP #
#############

cdef void cy_nsl_ff_bp0(
    const int sp_size, const I32_t *sp_idx,
    const int pn_size, const I32_t *pn_keys, REAL_t *pn_sign,
    REAL_t *X, REAL_t *W, REAL_t *b,
    REAL_t *dX, REAL_t *dW, REAL_t *db,
    REAL_t *L, const int vec_dim) nogil:

    # declarations
    cdef long long row1, row2
    cdef REAL_t label, y, exp_pns_y, g
    cdef I32_t X_key, W_key
    cdef int sp_i, i, j

    # update loop
    for sp_i in range(sp_size):
        X_key = sp_idx[sp_i]
        row1 = X_key * vec_dim # get the starting index of input row (in X)
        for j in range(pn_size):
            W_key = pn_keys[X_key*pn_size + j] # get the LUT key for target word
            row2 = W_key * vec_dim # get the starting index of target row (in W)
            label = pn_sign[X_key*pn_size + j] # get the +1/-1 label for target
            # compute prediction y as np.dot(X[X_key], W[W_key].T) + b[W_key]
            y = <REAL_t>dsdot(&vec_dim, &X[row1], &ONE, &W[row2], &ONE) + b[W_key]
            exp_pns_y = <REAL_t>exp(label * y) # this is used for loss/grad
            L[0] = L[0] + log(1.0 + exp_pns_y) # record the loss
            # Compute gradient and update gradient accumulators
            g = label * (exp_pns_y / (1.0 + exp_pns_y))
            saxpy(&vec_dim, &g, &X[row1], &ONE, &dW[row2], &ONE)
            saxpy(&vec_dim, &g, &W[row2], &ONE, &dX[row1], &ONE)
            db[W_key] = db[W_key] + g
    return


cdef void cy_nsl_ff_bp1(
    const int sp_size, const I32_t *sp_idx,
    const int pn_size, const I32_t *pn_keys, REAL_t *pn_sign,
    REAL_t *X, REAL_t *W, REAL_t *b,
    REAL_t *dX, REAL_t *dW, REAL_t *db,
    REAL_t *L, const int vec_dim) nogil:

    # declarations
    cdef long long row1, row2
    cdef REAL_t label, y, exp_pns_y, g
    cdef I32_t X_key, W_key
    cdef int sp_i, i, j

    # update loop
    for sp_i in range(sp_size):
        X_key = sp_idx[sp_i]
        row1 = X_key * vec_dim # get the starting index of input row (in X)
        for j in range(pn_size):
            W_key = pn_keys[X_key*pn_size + j] # get the LUT key for target word
            row2 = W_key * vec_dim # get the starting index of target row (in W)
            label = pn_sign[X_key*pn_size + j] # get the +1/-1 label for target
            # compute prediction y as np.dot(X[X_key], W[W_key].T) + b[W_key]
            y = <REAL_t>sdot(&vec_dim, &X[row1], &ONE, &W[row2], &ONE) + b[W_key]
            exp_pns_y = <REAL_t>exp(label * y) # this is used for loss/grad
            L[0] = L[0] + log(1.0 + exp_pns_y) # record the loss
            # Compute gradient and update gradient accumulators
            g = label * (exp_pns_y / (1.0 + exp_pns_y))
            saxpy(&vec_dim, &g, &X[row1], &ONE, &dW[row2], &ONE)
            saxpy(&vec_dim, &g, &W[row2], &ONE, &dX[row1], &ONE)
            db[W_key] = db[W_key] + g
    return

def nsl_ff_bp_pyx(sp_idx_p, pn_keys_p, pn_sign_p, X_p, W_p, b_p,
                  dX_p, dW_p, db_p, L_p):
    # Define and cast minibatch problem parameters
    cdef int sp_size = <int>sp_idx_p.shape[0]
    cdef int pn_size = <int>pn_keys_p.shape[1]
    cdef int vec_dim = <int>W_p.shape[1]
    cdef I32_t *sp_idx = <I32_t *>(np.PyArray_DATA(sp_idx_p))
    cdef I32_t *pn_keys = <I32_t *>(np.PyArray_DATA(pn_keys_p))
    cdef REAL_t *pn_sign = <REAL_t *>(np.PyArray_DATA(pn_sign_p))
    cdef REAL_t *X = <REAL_t *>(np.PyArray_DATA(X_p))
    cdef REAL_t *W = <REAL_t *>(np.PyArray_DATA(W_p))
    cdef REAL_t *b = <REAL_t *>(np.PyArray_DATA(b_p))
    cdef REAL_t *dX = <REAL_t *>(np.PyArray_DATA(dX_p))
    cdef REAL_t *dW = <REAL_t *>(np.PyArray_DATA(dW_p))
    cdef REAL_t *db = <REAL_t *>(np.PyArray_DATA(db_p))
    cdef REAL_t *L = <REAL_t *>(np.PyArray_DATA(L_p))

    with nogil:
        cy_nsl_ff_bp(sp_size, sp_idx, pn_size, pn_keys, pn_sign,
                     X, W, b, dX, dW, db, L, vec_dim)
    return

################
# AG_UPDATE_2D #
################

cdef void cy_ag_update_2d(
    const int sp_size, const I32_t *sp_idx, const I32_t *row_idx,
    REAL_t *W, REAL_t *dW, REAL_t *mW, REAL_t alpha, REAL_t ada_eps,
    const int vec_dim) nogil:

    # declarations
    cdef long long row_ptr
    cdef int sp_i, vec_bytes
    cdef I32_t i, row_key

    # update loop
    vec_bytes = cython.sizeof(REAL_t) * vec_dim
    for sp_i in range(sp_size):
        i = sp_idx[sp_i]
        row_key = row_idx[i]
        row_ptr = row_key * vec_dim
        saxpy(&vec_dim, &alpha, &dW[row_ptr], &ONE, &W[row_ptr], &ONE)
        memset(&dW[row_ptr], 0, vec_bytes)
    return

def ag_update_2d_pyx(sp_idx_p, row_idx_p, W_p, dW_p, mW_p, alpha_p, ada_eps_p):
    # Define and cast minibatch problem parameters
    cdef int sp_size = <int>sp_idx_p.shape[0]
    cdef int vec_dim = <int>W_p.shape[1]
    cdef I32_t *sp_idx = <I32_t *>(np.PyArray_DATA(sp_idx_p))
    cdef I32_t *row_idx = <I32_t *>(np.PyArray_DATA(row_idx_p))
    cdef REAL_t *W = <REAL_t *>(np.PyArray_DATA(W_p))
    cdef REAL_t *dW = <REAL_t *>(np.PyArray_DATA(dW_p))
    cdef REAL_t *mW = <REAL_t *>(np.PyArray_DATA(mW_p))
    cdef REAL_t alpha = <REAL_t>(-1.0 * alpha_p)
    cdef REAL_t ada_eps = <REAL_t>ada_eps_p

    with nogil:
        cy_ag_update_2d(sp_size, sp_idx, row_idx, W, dW, mW, alpha, ada_eps, vec_dim)
    return

################
# AG_UPDATE_1D #
################

cdef void cy_ag_update_1d(
    const int sp_size, const I32_t *sp_idx, const I32_t *row_idx, REAL_t *W,
    REAL_t *dW, REAL_t *mW, REAL_t alpha, REAL_t ada_eps) nogil:

    # declarations
    cdef int sp_i
    cdef I32_t i, row_key

    # update loop
    for sp_i in range(sp_size):
        i = sp_idx[sp_i]
        row_key = row_idx[i]
        W[row_key] = W[row_key] + (alpha * dW[row_key])
        dW[row_key] = 0.0
    return

def ag_update_1d_pyx(sp_idx_p, row_idx_p, W_p, dW_p, mW_p, alpha_p, ada_eps_p):
    # Define and cast minibatch problem parameters
    cdef int sp_size = <int>sp_idx_p.shape[0]
    cdef I32_t *sp_idx = <I32_t *>(np.PyArray_DATA(sp_idx_p))
    cdef I32_t *row_idx = <I32_t *>(np.PyArray_DATA(row_idx_p))
    cdef REAL_t *W = <REAL_t *>(np.PyArray_DATA(W_p))
    cdef REAL_t *dW = <REAL_t *>(np.PyArray_DATA(dW_p))
    cdef REAL_t *mW = <REAL_t *>(np.PyArray_DATA(mW_p))
    cdef REAL_t alpha = <REAL_t>(-1.0 * alpha_p)
    cdef REAL_t ada_eps = <REAL_t>ada_eps_p

    with nogil:
        cy_ag_update_1d(sp_size, sp_idx, row_idx, W, dW, mW, alpha, ada_eps)
    return

##########
# LUT_BP #
##########

cdef void cy_lut_bp(
    const int sp_size, const I32_t *sp_idx, const I32_t *row_idx,
    REAL_t *dLdY, REAL_t *dW, const int vec_dim) nogil:

    # declarations
    cdef long long row1, row2
    cdef int sp_i
    cdef I32_t i, j

    # update loop
    for sp_i in range(sp_size):
        i = sp_idx[sp_i] # row key for dLdY
        j = row_idx[i] # row key for dW
        row1 = i * vec_dim
        row2 = j * vec_dim
        saxpy(&vec_dim, &ONEF, &dLdY[row1], &ONE, &dW[row2], &ONE)
    return

def lut_bp_pyx(sp_idx_p, row_idx_p, dLdY_p, dW_p):
    # Define and cast minibatch problem parameters
    cdef int sp_size = <int>sp_idx_p.shape[0]
    cdef int vec_dim = <int>dLdY_p.shape[1]
    cdef I32_t *sp_idx = <I32_t *>(np.PyArray_DATA(sp_idx_p))
    cdef I32_t *row_idx = <I32_t *>(np.PyArray_DATA(row_idx_p))
    cdef REAL_t *dLdY = <REAL_t *>(np.PyArray_DATA(dLdY_p))
    cdef REAL_t *dW = <REAL_t *>(np.PyArray_DATA(dW_p))

    with nogil:
        cy_lut_bp(sp_size, sp_idx, row_idx, dLdY, dW, vec_dim)
    return

###############
# INIT, INNIT #
###############

def init():
    """
    Bleep bloop: computer compute.
    """
    global cy_w2v_ff_bp
    global cy_nsl_ff_bp

    cdef float *x = [<float>10.0]
    cdef float *y = [<float>0.01]
    cdef float expected = <float>0.1
    cdef int size = 1
    cdef double d_res
    cdef float *p_res

    # check whether sdot returns double or float
    d_res = dsdot(&size, x, &ONE, y, &ONE)
    p_res = <float *>&d_res
    if (abs(d_res - expected) < 0.0001):
        cy_w2v_ff_bp = cy_w2v_ff_bp0
        cy_nsl_ff_bp = cy_nsl_ff_bp0
        return 0  # double
    elif (abs(p_res[0] - expected) < 0.0001):
        cy_w2v_ff_bp = cy_w2v_ff_bp1
        cy_nsl_ff_bp = cy_nsl_ff_bp1
        return 1  # float
    else:
        # dsdot returns neither single nor double precision!?
        assert False
        return 2

FAST_VERSION = init()  # initialize the module