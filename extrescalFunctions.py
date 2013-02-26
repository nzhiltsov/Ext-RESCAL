import numpy as np
from numpy import dot, zeros, eye, empty
from numpy.linalg import inv
from commonFunctions import reservoir
from itertools import product
from numpy.random import random_integers

def updateA(X, A, R, V, D, lmbda):
    n, rank = A.shape
    F = zeros((n,rank), dtype=np.float64)
    E = zeros((rank, rank), dtype=np.float64)

    AtA = dot(A.T, A)
    for i in range(len(X)):
        ar = dot(A, R[i])
        art = dot(A, R[i].T)
        F += X[i].dot(art) + X[i].T.dot(ar)
        E += dot(R[i], dot(AtA, R[i].T)) + dot(R[i].T, dot(AtA, R[i]))
    A = dot(F  + D.dot(V.T), inv(lmbda * eye(rank) + E + dot(V, V.T)))
    return A

def updateV(A, D, lmbda):
    n, rank = A.shape    
    At = A.T
    invPart = empty((1, 1))
    if lmbda == 0:
        invPart = inv(dot(At, A))
    else :
        invPart = inv(dot(At, A) + lmbda * eye(rank))
    return dot(invPart, At) * D

def matrixFitNormElement(i, j, D, A, V):
    """
    Computes i,j element of the fitting matrix Frobenius norm ||D - A*V||
    """ 
    return (D[i,j] - dot(A[i,:], V[:, j]))**2

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

    
