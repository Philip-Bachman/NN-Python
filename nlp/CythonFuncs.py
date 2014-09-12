from __future__ import absolute_import

# try to compile and use the faster cython version
import os
from numpy import get_include
import pyximport
models_dir = os.path.dirname(__file__) or os.getcwd()
pyximport.install(setup_args={"include_dirs": [models_dir, get_include()]})
from CythonFuncsPyx import w2v_ff_bp_pyx, ag_update_2d_pyx, ag_update_1d_pyx, \
                           lut_bp_pyx, nsl_ff_bp_pyx, acl_ff_bp_pyx, DO_INIT

import numpy as np
import numpy.random as npr
import threading
from ctypes import pythonapi, c_void_p

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
        sp_idx = np.arange(0,length).astype(np.uint32)
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
        sp_idx = np.arange(0,length).astype(np.uint32)
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

w2v_ff_bp = make_multithread(w2v_ff_bp_pyx, THREAD_NUM)
hsm_ff_bp = make_multithread(nsl_ff_bp_pyx, THREAD_NUM)
nsl_ff_bp = make_multithread(nsl_ff_bp_pyx, THREAD_NUM)
lut_bp = make_multithread(lut_bp_pyx, THREAD_NUM)

ag_update_2d = make_multithread(ag_update_2d_pyx, THREAD_NUM)
ag_update_1d = make_multithread(ag_update_1d_pyx, 1)


##############
# EYE BUFFER #
##############
