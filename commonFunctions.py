from numpy import dot
from numpy.random import randint
from itertools import ifilter

def squareFrobeniusNormOfSparse(M):
    """
    Computes the square of the Frobenius norm
    """
    rows, cols = M.nonzero()
    norm = 0
    for i in range(len(rows)):
        norm += M[rows[i],cols[i]] ** 2
    return norm

def fitNorm(row, col, Xi, ARk, A):   
    """
    Computes i,j element of the squared Frobenius norm of the fitting matrix
    """
    ARAtValue = dot(ARk[row,:], A[col,:])
    return (Xi[row, col] - ARAtValue)**2

def reservoir(it, k):
    ls = [next(it) for _ in range(k)]
    for i, x in enumerate(it, k + 1):
        j = randint(0, i)
        if j < k:
            ls[j] = x
    return ls  
