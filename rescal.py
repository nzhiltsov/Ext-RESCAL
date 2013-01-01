import logging, time, argparse
from numpy import dot, zeros, kron, array, eye, argmax, argmin, ones, linalg, sqrt, savetxt, loadtxt
from numpy.linalg import qr, pinv, norm, inv 
from scipy.linalg import eigh
from numpy.random import rand
from numpy.random import random_integers
from scipy import sparse
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
import numpy as np
import os
import fnmatch
import carray as ca
import handythread
import operator
import itertools
from multiprocessing import Pool, Process, Manager, Array

__version__ = "0.1" 
__all__ = ['rescal', 'rescal_with_random_restarts']

__DEF_MAXITER = 500
__DEF_INIT = 'nvecs'
__DEF_PROJ = True
__DEF_CONV = 1e-2
__DEF_LMBDA = 0
__DEF_SAMPLESIZE = 1e-2

logging.basicConfig(filename='rescal.log',filemode='w', level=logging.DEBUG)
_log = logging.getLogger('RESCAL') 
ARk = ca.zeros((1,1))
Aglobal = ca.zeros((1,1))
Xiglobal = ca.zeros((1,1))

def rescal_with_random_restarts(X, rank, restarts=10, **kwargs):
    """
    Restarts RESCAL multiple time from random starting point and 
    returns factorization with best fit.
    """
    models = []
    fits = []
    for i in range(restarts):
        res = rescal(X, rank, init='random', **kwargs)
        models.append(res)
        fits.append(res[2])
    return models[argmin(fits)]

def squareFrobeniusNormOfSparse(M):
    """
    Computes the square of the Frobenius norm
    """
    norm = sum(M.dot(M.transpose()).diagonal())
    return norm

def minus(L, R):
    """
    Compute L - R for cArray matrices
    """
    l1, l2 = L.shape
    r1, r2 = R.shape
    if l1 != r1 or l2 != r2:
        raise 'Both the matrices must have the same shape.'
    matrix = L.copy()
    for i in range(l1):
        for j in range(l2):
            matrix[i,j] = matrix[i,j] - R[i,j]
    return matrix

def dotAsCArray(L, R):
    """
    Computes the dot product as a cArray
    """
    l1, l2 = L.shape
    r1, r2 = R.shape
    
    matrix = ca.zeros((l1, r2))
    for i in range(l1):
        for j in range(r2):
            matrix[i,j] = dot(L[i,:],R[:,j])
    return matrix

def squareOfMatrix(M):
    """
    Computes A^T * A, i.e., the square of a given matrix
    """
    n,r = M.shape
    matrix = ca.zeros((r, r))
    for i in range(r):
        for j in range(r):
            matrix[i,j] = dot(M[:,i], M[:,j])
    return matrix

def ARAtFunc(j, ARki, A):
    """
    Computes the j-th row of the matrix ARk * A^T
    """
    return dot(ARki, A[j,:])

def fitNorm(i):   
    """
    Computes the squared Frobenius norm of the i-th fitting matrix row
    """
    n, r = A.shape
    ARAtValues = handythread.parallel_map2(ARAtFunc, range(n), ARk[i,:], Aglobal, threads=7)
    return norm(Xiglobal.getrow(i).todense() - ARAtValues)**2

def rescal(X, rank, **kwargs):
    """
    RESCAL 

    Factors a three-way tensor X such that each frontal slice 
    X_k = A * R_k * A.T. The frontal slices of a tensor are 
    N x N matrices that correspond to the adjecency matrices 
    of the relational graph for a particular relation.

    For a full description of the algorithm see: 
      Maximilian Nickel, Volker Tresp, Hans-Peter-Kriegel, 
      "A Three-Way Model for Collective Learning on Multi-Relational Data",
      ICML 2011, Bellevue, WA, USA

    Parameters
    ----------
    X : list
        List of frontal slices X_k of the tensor X. The shape of each X_k is ('N', 'N')
    rank : int 
        Rank of the factorization
    lmbda : float, optional 
        Regularization parameter for A and R_k factor matrices. 0 by default 
    init : string, optional
        Initialization method of the factor matrices. 'nvecs' (default) 
        initializes A based on the eigenvectors of X. 'random' initializes 
        the factor matrices randomly.
    proj : boolean, optional 
        Whether or not to use the QR decomposition when computing R_k.
        True by default 
    samplesize: float, optional
        The proportion of sample tensor values involved in computing the fitting value.
        0.01 by default
    maxIter : int, optional 
        Maximium number of iterations of the ALS algorithm. 500 by default. 
    conv : float, optional 
        Stop when residual of factorization is less than conv. 1e-5 by default

    Returns 
    -------
    A : ndarray 
        array of shape ('N', 'rank') corresponding to the factor matrix A
    R : list
        list of 'M' arrays of shape ('rank', 'rank') corresponding to the factor matrices R_k 
    f : float 
        function value of the factorization 
    iter : int 
        number of iterations until convergence 
    exectimes : ndarray 
        execution times to compute the updates in each iteration
    """

    # init options
    ainit = kwargs.pop('init', __DEF_INIT)
    proj = kwargs.pop('proj', __DEF_PROJ)
    maxIter = kwargs.pop('maxIter', __DEF_MAXITER)
    conv = kwargs.pop('conv', __DEF_CONV)
    lmbda = kwargs.pop('lmbda', __DEF_LMBDA)
    samplesize = kwargs.pop('samplesize', __DEF_SAMPLESIZE)

    if not len(kwargs) == 0:
        raise ValueError( 'Unknown keywords (%s)' % (kwargs.keys()) )
   
    sz = X[0].shape
    dtype = X[0].dtype 
    n = sz[0]
    k = len(X) 
    
    _log.debug('[Config] rank: %d | maxIter: %d | conv: %7.1e | lmbda: %7.1e' % (rank, 
        maxIter, conv, lmbda))
    _log.debug('[Config] dtype: %s' % dtype)
    
    # precompute norms of X 
    normX = [squareFrobeniusNormOfSparse(M) for M in X]
    _log.debug('[Config] finished precomputing norms')
    sumNormX = sum(normX)
    
    # initialize A
    A = ca.zeros((n,rank), dtype=np.float64)
    if ainit == 'random':
#        A = array(rand(n, rank), dtype=np.float64)
         for k in range(n/1000):
             A[k*1000:(k+1)*1000,0:rank] = rand(1000, rank)
    elif ainit == 'nvecs':
        S = coo_matrix((n, n), dtype=np.float64)
        T = coo_matrix((n, n), dtype=dtype)
        for i in range(k):
            T = X[i]
            S = S + T + T.T
        evals, A = eigsh(S,k=rank)
    else :
        raise 'Unknown init option ("%s")' % ainit

    # initialize R
    if proj:
        Q, A2 = qr(A)
        X2 = __projectSlices(X, Q)
        R = __updateR(X2, A2, lmbda)
    else :
        raise 'Projection via QR decomposition is required; pass proj=true'
#        R = __updateR(X, A, lmbda)

    # compute factorization
    fit = fitchange = fitold = 0
    exectimes = []

    for iter in xrange(maxIter):
        tic = time.clock()
        
        A = __updateA(X, A, R, lmbda)
        global Aglobal
        Aglobal = A
        if proj:
            Q, A2 = qr(A)
            X2 = __projectSlices(X, Q)
            R = __updateR(X2, A2, lmbda)
        else :
            raise 'Projection via QR decomposition is required; pass proj=true'
#            R = __updateR(X, A, lmbda)

        # compute fit values
        regularizedFit = 0
        if lmbda != 0:
            regRFit = 0 
            for i in range(len(R)):
                regRFit += norm(R[i])**2
            regularizedFit = lmbda*(norm(A)**2) + lmbda*regRFit
            
        fit = 0
        for i in range(len(R)):
            global ARk
            ARk = dotAsCArray(A, R[i])
            global Xiglobal
            Xiglobal = X[i]           
            p = Pool(4)
            fits = p.map(fitNorm, range(n))
            fit += sum(fits)           
        fit *= 0.5
        fit += regularizedFit
        fit /= sumNormX 
                
            
        toc = time.clock()
        exectimes.append( toc - tic )
        fitchange = abs(fitold - fit)
        if lmbda != 0:
            _log.debug('[%3d] approxFit: %7.1e | regularized fit: %7.1e | approxFit delta: %7.1e | secs: %.5f' % (iter, 
        fit, regularizedFit, fitchange, exectimes[-1]))
        else :
            _log.debug('[%3d] approxFit: %7.1e | approxFit delta: %7.1e | secs: %.5f' % (iter, 
        fit, fitchange, exectimes[-1]))
            
        fitold = fit
#            if iter > 1 and fitchange < conv:
#                break
    return A, R, fit, iter+1, array(exectimes)

def __updateA(X, A, R, lmbda):
    n, rank = A.shape
#    F = zeros((n, rank), dtype=np.float64)
    F = coo_matrix((n,rank), dtype=np.float64)
    E = zeros((rank, rank), dtype=np.float64)

    AtA = squareOfMatrix(A)
    for i in range(len(X)):
        ar = dotAsCArray(A, R[i])
        art = dotAsCArray(A, R[i].T)
        F = F + X[i].dot(art) + X[i].T.dot(ar)
        E = E + dotAsCArray(R[i], dotAsCArray(AtA, R[i].T)) + dotAsCArray(R[i].T, dotAsCArray(AtA, R[i]))
    A = dotAsCArray(F, inv(lmbda * eye(rank) + E))
    return A

def __updateR(X, A, lmbda):
    r = A.shape[1]
    R = []
    At = A.T    
    if lmbda == 0:
        ainv = dot(pinv(dot(At, A)), At)
        for i in range(len(X)):
            R.append( dot(ainv, X[i].dot(ainv.T)) )
    else :
        AtA = dot(At, A)
        tmp = inv(kron(AtA, AtA) + lmbda * eye(r**2))
        for i in range(len(X)):
            AtXA = dot(At, X[i].dot(A)) 
            R.append( dot(AtXA.flatten(), tmp).reshape(r, r) )
    return R

def __projectSlices(X, Q):
    q = Q.shape[1]
    X2 = []
    for i in range(len(X)):
        X2.append( dot(Q.T, X[i].dot(Q)) )
    return X2

parser = argparse.ArgumentParser()
parser.add_argument("--latent", type=int, help="number of latent components")
parser.add_argument("--samplesize", type=float, help="proportion (in percents) of sampled tensor elements while computing the fit value")
args = parser.parse_args()
numLatentComponents = args.latent
samplesizeVal = args.samplesize

dim = 0
with open('./data2/entity-ids') as entityIds:
    for line in entityIds:
          dim += 1
print 'The number of entities: %d' % dim          

numSlices = 0
X = []
for file in os.listdir('./data2'):
    if fnmatch.fnmatch(file, '*-rows'):
        numSlices += 1
        row = loadtxt('./data2/' + file, dtype=np.int32)
        if row.size == 1: 
            row = np.atleast_1d(row)
        col = loadtxt('./data2/' + file.replace("rows", "cols"), dtype=np.int32)
        if col.size == 1: 
            col = np.atleast_1d(col)
        A = coo_matrix((ones(row.size),(row,col)), shape=(dim,dim), dtype=np.uint8)
        X.append(A)
        
print 'The number of slices: %d' % numSlices

result = rescal(X, numLatentComponents, init='random', samplesize=samplesizeVal)
print 'Objective function value: %.5f' % result[2]
print '# of iterations: %d' % result[3] 
#print the matrix of latent embeddings
A = result[0]
savetxt("latent-embeddings.csv", A)

