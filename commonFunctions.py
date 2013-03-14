from numpy import dot
from numpy.random import randint
from numpy.random import random_integers
from scipy.sparse import lil_matrix
import numpy as np

def squareFrobeniusNormOfSparse(M):
    """
    Computes the square of the Frobenius norm
    """
    rows, cols = M.nonzero()
    norm = 0
    for i in range(len(rows)):
        norm += M[rows[i],cols[i]] ** 2
    return norm

def trace(M):
    """ Compute the trace of a sparse matrix
    """
    return sum(M.diagonal())

def fitNorm(X, A, R):   
    """
    Computes the squared Frobenius norm of the fitting matrix || X - A*R*A^T ||,
    where X is a sparse matrix
    """
    AtA = dot(A.T, A)
    secondTerm = dot(A.T, dot(X.dot(A), R.T))
    thirdTerm = dot(dot(AtA, R), dot(AtA, R.T))
    return squareFrobeniusNormOfSparse(X) - 2 * trace(secondTerm) + np.trace(thirdTerm)

def reservoir(it, k):
    ls = [next(it) for _ in range(k)]
    for i, x in enumerate(it, k + 1):
        j = randint(0, i)
        if j < k:
            ls[j] = x
    return ls  

def checkingIndices(M, ratio = 1):
    """
    Returns the indices for computing fit values
    based on non-zero values as well as sample indices
    (the sample size is proportional to the given ratio ([0,1]) and number of matrix columns)
    """
    rowSize, colSize = M.shape
    nonzeroRows, nonzeroCols = M.nonzero()
    nonzeroIndices = [(nonzeroRows[i], nonzeroCols[i]) for i in range(len(nonzeroRows))]                
    sampledRows = random_integers(0, rowSize - 1, round(ratio*colSize))
    sampledCols = random_integers(0, colSize - 1, round(ratio*colSize))
    sampledIndices = zip(sampledRows, sampledCols)
    indices = list(set(sampledIndices + nonzeroIndices))
    return indices