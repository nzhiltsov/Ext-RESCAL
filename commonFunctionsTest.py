from numpy import ones, dot
import numpy as np
from scipy.sparse import coo_matrix
from commonFunctions import squareFrobeniusNormOfSparse, fitNorm, reservoir
from numpy.linalg.linalg import norm
from nose.tools import assert_almost_equal
from itertools import product

def testSquareFrobeniusNorm():
    zeroCount = 2
    rowIndices = np.array([1, 2])
    colIndices = np.array([0, 0])
    rowSize = 6
    colSize = 6
    M = coo_matrix((ones(zeroCount),(rowIndices, colIndices)), shape=(rowSize, colSize), dtype=np.uint8).tolil()
    assert squareFrobeniusNormOfSparse(M) == 2
    
def testFitNorm():
    X = coo_matrix((ones(4),([0, 1, 2, 2], [1, 1, 0, 1])), shape=(3, 3), dtype=np.uint8).tolil()
    n = X.shape[0]
    A = np.array([[0.9, 0.1],
         [0.8, 0.2],
         [0.1, 0.9]])
    R = np.array([[0.9, 0.1],
         [0.1, 0.9]])
    expectedNorm = norm(X - dot(A,dot(R, A.T)))**2
    ARk = dot(A, R)
    fits = []
    for i in xrange(n):
        for j in xrange(n):
            fits.append(fitNorm(i, j, X, ARk, A))
    assert_almost_equal(sum(fits), expectedNorm)  
    
def testSampling():
    xs = range(0, 3)
    ys = range(0, 4)
    size = int(0.9 * len(xs) * len(ys))
    sampledElements = reservoir(product(xs, ys), size)
    assert len(sampledElements) == size
    checkedElements = [] 
    for i in xrange(size):
        assert checkedElements.count(sampledElements[i]) == 0
        checkedElements.append(sampledElements[i])
    assert len(checkedElements) == len(sampledElements)
    