from numpy import ones
import numpy as np
from scipy.sparse import coo_matrix
from commonFunctions import squareFrobeniusNormOfSparse

def test():
    zeroCount = 2
    rowIndices = np.array([1, 2])
    colIndices = np.array([0, 0])
    rowSize = 6
    colSize = 6
    M = coo_matrix((ones(zeroCount),(rowIndices, colIndices)), shape=(rowSize, colSize), dtype=np.uint8).tolil()
    assert squareFrobeniusNormOfSparse(M) == 2
    
    