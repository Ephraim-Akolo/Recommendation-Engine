
import numpy as np
cimport numpy
cimport cython
# from cpython.mem cimport PyMem_Malloc, PyMem_Free
# from libc.stdio cimport printf

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def merge_recommendations_(numpy.ndarray[numpy.uint32_t, ndim=1] arr1, numpy.ndarray[numpy.uint32_t, ndim=1] arr2, unsigned int items_len, unsigned int max_count):
    '''
    Merge two recommendation arrays of type numpy.uint32.
    '''
    cdef numpy.ndarray[numpy.uint8_t, ndim=1] recommended
    cdef numpy.ndarray[numpy.uint32_t, ndim=1] recommendations
    recommended = np.zeros(items_len, dtype=np.uint8)
    recommendations = np.ndarray(max_count, dtype=np.uint32)
    cdef unsigned int n, m, i, j
    n = 0
    m = 0
    for _ in range(max_count):
        i = arr1[m]
        j = arr2[m]
        if i == j:
            if not recommended[i]:
                recommendations[n] = i + 1
                recommended[i] = 1
                n += 1
        else:
            if not recommended[i]:
                recommendations[n] = i + 1
                recommended[i] = 1
                n += 1
            if not recommended[j]:
                recommendations[n] = j + 1
                recommended[j] = 1
                n += 1
        m += 1
        if not n < max_count:
            break
    return recommendations