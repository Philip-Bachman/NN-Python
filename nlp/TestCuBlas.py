'''
This example uses cuBLAS gemm routine to perform matrix-matrix multiplication.
Please refer to the documentation for details of how to use the gemm routine
  http://docs.continuum.io/numbapro/cudalib.html#blas-level-2
  
Note: cuBLAS uses Fortran layout
'''

import numbapro.cudalib.cublas as cublas
from numbapro import cuda
import numpy as np
import numpy.random as npr
from timeit import default_timer as timer
import gnumpy as gp

N = 500     # no. of rows/cols

def gemm_v1():
    '''
    Note that all arrays are in Fortran order.
    '''
    print("Version 1".center(80, '='))
    # Prepare arrays for input
    A = np.array(np.arange(N ** 2, dtype=np.float32).reshape(N, N), order='F')
    B = np.array(np.arange(N) + 10, dtype=A.dtype, order='F')
    D = np.zeros_like(A, order='F')

    # NumPy
    start = timer()
    E = np.dot(A, np.diag(B))
    numpy_time = timer() - start
    print("Numpy took %f seconds" % numpy_time)

    # cuBLAS
    blas = cublas.Blas()

    stream = cuda.stream()
    cuda.to_device(A, stream=stream)
    stream.synchronize()
    
    start = timer()
    blas.gemm('N', 'N', N, N, N, 1.0, A, np.diag(B), 1.0, D)
    cuda_time = timer() - start

    print("CUBLAS took %f seconds" % cuda_time)
    diff = np.abs(D - E)
    print("Maximum error %f" % np.max(diff))


def gemm_v2():
    """
    Let GEMM transpose the input matrices so that they can be in C order,
    originally.  Note that the output matrix is still in Fortran array.
    The string arguments in gemm tells it to apply transformation on the input
    matrices.
    
    See argument description in:
        http://docs.continuum.io/numbapro/cudalib.html#blas-level-2
    """
    print("Version 2".center(80, '='))
    # Prepare arrays for input
    A = np.array(np.arange(N ** 2, dtype=np.float32).reshape(N, N))
    B = np.array(np.arange(N) + 10, dtype=A.dtype)
    D = np.zeros_like(A, order='F')

    # NumPy
    start = timer()
    E = np.dot(A, np.diag(B))
    numpy_time = timer() - start
    print("Numpy took %f seconds" % numpy_time)

    # cuBLAS
    blas = cublas.Blas()
    
    stream = cuda.stream()
    cuda.to_device(A, stream=stream)
    stream.synchronize()

    start = timer()
    blas.gemm('T', 'T', N, N, N, 1.0, A, np.diag(B), 1.0, D)
    cuda_time = timer() - start

    print("CUBLAS took %f seconds" % cuda_time)
    diff = np.abs(D - E)
    print("Maximum error %f" % np.max(diff))


def main():
    gemm_v1()
    gemm_v2()

if __name__ == '__main__':
    main()
    start = timer()
    A = npr.randn(256, 1500)
    for i in range(1000):
        B = gp.garray(A)
        B = B + B
        A = npr.rand(256, 1500)
    berk_time = timer() - start
    print("Berk time: {0:.4f}".format(berk_time))
    print("  @ {0:.4f} transfers/second".format(1000.0 / berk_time))