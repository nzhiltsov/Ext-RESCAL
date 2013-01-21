import logging, time, argparse
from numpy import dot, zeros, empty, kron, array, eye, argmin, ones, savetxt, loadtxt
from numpy.linalg import qr, pinv, norm, inv 
from numpy.random import rand
from scipy import sparse
from scipy.sparse import coo_matrix
import numpy as np
import os
import fnmatch

__version__ = "0.1" 
__all__ = ['rescal', 'rescal_with_random_restarts']

__DEF_MAXITER = 50
__DEF_PREHEATNUM = 40
__DEF_INIT = 'nvecs'
__DEF_PROJ = True
__DEF_CONV = 1e-5
__DEF_LMBDA = 0

logging.basicConfig(filename='extrescal.log',filemode='w', level=logging.DEBUG)
_log = logging.getLogger('RESCAL') 

def rescal_with_random_restarts(X, D, rank, restarts=10, **kwargs):
    """
    Restarts RESCAL multiple time from random starting point and 
    returns factorization with best fit.
    """
    models = []
    fits = []
    for i in range(restarts):
        res = rescal(X, D, rank, **kwargs)
        models.append(res)
        fits.append(res[2])
    return models[argmin(fits)]

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

def rescal(X, D, rank, **kwargs):
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
    D : matrix
        A sparse matrix involved in the tensor factorization (aims to incorporate
        the entity-term matrix aka document-term matrix)
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
    preheatnum = kwargs.pop('preheatnum', __DEF_PREHEATNUM)

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
    A = zeros((n,rank), dtype=np.float64)
    if ainit == 'random':
        A = array(rand(n, rank), dtype=np.float64)
    else :
        raise 'This type of initialization is not supported, please use random'

    # initialize R
    if proj:
        Q, A2 = qr(A)
        X2 = __projectSlices(X, Q)
        R = __updateR(X2, A2, lmbda)
    else :
        raise 'Projection via QR decomposition is required; pass proj=true'
#        R = __updateR(X, A, lmbda)
    
    # initialize V
    Drow, Dcol = D.shape
    V = array(rand(rank, Dcol), dtype=np.float64)
    
    # compute factorization
    fit = fitchange = fitold = 0
    exectimes = []

    for iter in xrange(maxIter):
        tic = time.clock()
        
        A = __updateA(X, A, R, V, lmbda)
        if proj:
            Q, A2 = qr(A)
            X2 = __projectSlices(X, Q)
            R = __updateR(X2, A2, lmbda)
        else :
            raise 'Projection via QR decomposition is required; pass proj=true'
#            R = __updateR(X, A, lmbda)

        V = __updateV(A, D, lmbda)
        # compute fit values
        fit = 0
        regularizedFit = 0
        if iter > preheatnum:
            if lmbda != 0:
                regRFit = 0 
                for i in range(len(R)):
                    regRFit += norm(R[i])**2
                regularizedFit = lmbda*(norm(A)**2) + lmbda*regRFit
        
            extendedFit = 0
            if lmbda != 0:
                extendedFit += norm(D - dot(A, V))**2 + lmbda*(norm(V)**2)
            else :
                extendedFit += norm(D - dot(A, V))**2    
        
            for i in range(len(R)):
                ARk = dot(A, R[i])       
                Xrow, Xcol = X[i].nonzero()
                fits = []
                for rr in range(len(Xrow)):
                    fits.append(fitNorm(Xrow[rr], Xcol[rr], X[i], ARk, A))
                fit += sum(fits)           
            fit *= 0.5
            fit += regularizedFit
            fit += extendedFit
            fit /= sumNormX 
        else :
            _log.debug('Preheating is going on.')        
            
        toc = time.clock()
        exectimes.append( toc - tic )
        fitchange = abs(fitold - fit)
        if lmbda != 0:
            _log.debug('[%3d] approxFit: %.20f | regularized fit: %.20f | approxFit delta: %.20f | secs: %.5f' % (iter, 
        fit, regularizedFit, fitchange, exectimes[-1]))
        else :
            _log.debug('[%3d] approxFit: %.20f | approxFit delta: %.20f | secs: %.5f' % (iter, 
        fit, fitchange, exectimes[-1]))
            
        fitold = fit
        if iter > preheatnum and fitchange < conv:
            break
    return A, R, fit, iter+1, array(exectimes), V

def __updateA(X, A, R, V, lmbda):
    n, rank = A.shape
    F = zeros((n,rank))
    E = zeros((rank, rank), dtype=np.float64)

    AtA = dot(A.T, A)
    for i in range(len(X)):
        ar = dot(A, R[i])
        art = dot(A, R[i].T)
        F = F + X[i].dot(art) + X[i].T.dot(ar) + D.dot(V.T)
        E = E + dot(R[i], dot(AtA, R[i].T)) + dot(R[i].T, dot(AtA, R[i]))
    A = dot(F, inv(lmbda * eye(rank) + E + dot(V, V.T)))
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

def __updateV(A, D, lmbda):
    n, rank = A.shape    
    At = A.T
    invPart = empty((1, 1))
    if lmbda == 0:
        invPart = inv(dot(At, A))
    else :
        invPart = inv(dot(At, A) + lmbda * eye(rank))
    return dot(invPart, At) * D
        

def __projectSlices(X, Q):
    q = Q.shape[1]
    X2 = []
    for i in range(len(X)):
        X2.append( dot(Q.T, X[i].dot(Q)) )
    return X2

parser = argparse.ArgumentParser()
parser.add_argument("--latent", type=int, help="number of latent components", required=True)
parser.add_argument("--lmbda", type=float, help="regularization parameter", required=True)
parser.add_argument("--input", type=str, help="the directory, where the input data are stored", required=True)
parser.add_argument("--outputentities", type=str, help="the file, where the latent embedding for entities will be output", required=True)
parser.add_argument("--outputterms", type=str, help="the file, where the latent embedding for terms will be output", required=True)
args = parser.parse_args()
numLatentComponents = args.latent
inputDir = args.input
regularizationParam = args.lmbda
outputEntities = args.outputentities
outputTerms = args.outputterms

dim = 0
with open('./%s/entity-ids' % inputDir) as entityIds:
    for line in entityIds:
          dim += 1
print 'The number of entities: %d' % dim          

numSlices = 0
numNonzeroTensorEntries = 0
X = []
for file in os.listdir('./%s' % inputDir):
    if fnmatch.fnmatch(file, '*-rows'):
        numSlices += 1
        row = loadtxt('./%s/%s' % (inputDir, file), dtype=np.int32)
        if row.size == 1: 
            row = np.atleast_1d(row)
        col = loadtxt('./%s/%s' % (inputDir, file.replace("rows", "cols")), dtype=np.int32)
        if col.size == 1: 
            col = np.atleast_1d(col)
        Xi = coo_matrix((ones(row.size),(row,col)), shape=(dim,dim), dtype=np.uint8).tolil()
        numNonzeroTensorEntries += row.size
        X.append(Xi)
        
print 'The number of tensor slices: %d' % numSlices
print 'The number of non-zero values in the tensor: %d' % numNonzeroTensorEntries

extDim = 0
with open('./%s/words' % inputDir) as words:
    for line in words:
          extDim += 1
print 'The number of words: %d' % extDim

extRow = loadtxt('./%s/ext-matrix-rows' % inputDir, dtype=np.int32)
if extRow.size == 1: 
    extRow = np.atleast_1d(extRow)
extCol = loadtxt('./%s/ext-matrix-cols' % inputDir, dtype=np.int32)
if extCol.size == 1: 
    extCol = np.atleast_1d(extCol)
D = coo_matrix((ones(extRow.size),(extRow,extCol)), shape=(dim,extDim), dtype=np.uint8).tocsr()

print 'The number of non-zero values in the additional matrix: %d' % extRow.size         

result = rescal(X, D, numLatentComponents, init='random', lmbda=regularizationParam)
print 'Objective function value: %.30f' % result[2]
print '# of iterations: %d' % result[3] 
#print the matrices of latent embeddings
A = result[0]
savetxt(outputEntities, A)
V = result[5]
savetxt(outputTerms, V.T)

