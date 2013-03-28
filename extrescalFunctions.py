import numpy as np
from numpy import dot, zeros, eye, empty
from numpy.linalg import inv
from commonFunctions import trace, squareFrobeniusNormOfSparse
from scipy.sparse import lil_matrix

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

def matrixFitNorm(D, A, V):
    """
    Computes the Frobenius norm of the fitting matrix ||D - A*V||,
    where D is a sparse matrix
    """ 
    return squareFrobeniusNormOfSparse(D) + matrixFitNormWithoutNormD(D, A, V)

def matrixFitNormWithoutNormD(D, A, V):
    thirdTerm = dot(dot(V, V.T), dot(A.T, A))
    secondTerm = dot(A.T, D.dot(V.T))
    return np.trace(thirdTerm) - 2 * trace(secondTerm) 


    
