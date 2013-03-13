from numpy import dot
from numpy.random import randint
from numpy.random import random_integers

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