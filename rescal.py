import logging, time
from numpy import dot, zeros, kron, array, eye, argmax, argmin, ones, linalg, sqrt, savetxt
from numpy.linalg import qr, pinv, norm, inv 
from scipy.linalg import eigh
from numpy.random import rand
from numpy.random import random_integers
from scipy import sparse
from scipy.sparse import coo_matrix

__version__ = "0.1" 
__all__ = ['rescal', 'rescal_with_random_restarts']

__DEF_MAXITER = 500
__DEF_INIT = 'nvecs'
__DEF_PROJ = True
__DEF_CONV = 1e-5
__DEF_LMBDA = 0

logging.basicConfig(filename='rescal.log',filemode='w', level=logging.DEBUG)
_log = logging.getLogger('RESCAL') 

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
    Xflat = [M for M in X]
    sumNormX = sum(normX)
    
    # initialize A
    if ainit == 'random':
        A = array(rand(n, rank), dtype=dtype)
    elif ainit == 'nvecs':
        S = zeros((n, n), dtype=dtype)
        T = zeros((n, n), dtype=dtype)
        for i in range(k):
            T = X[i]
            S = S + T + T.T
        evals, A = eigh(S,eigvals=(n-rank,n-1))
    else :
        raise 'Unknown init option ("%s")' % ainit

    # initialize R
    if proj:
        Q, A2 = qr(A)
        X2 = __projectSlices(X, Q)
        R = __updateR(X2, A2, lmbda)
    else :
        R = __updateR(X, A, lmbda)

    # compute factorization
    fit = fitchange = fitold = f = 0
    exectimes = []
    ARAt = zeros((n,n), dtype=dtype)
    for iter in xrange(maxIter):
        tic = time.clock()
        fitold = fit
        A = __updateA(X, A, R, lmbda)
        if proj:
            Q, A2 = qr(A)
            X2 = __projectSlices(X, Q)
            R = __updateR(X2, A2, lmbda)
        else :
            R = __updateR(X, A, lmbda)

        # compute fit value
        f = lmbda*(norm(A)**2)
        for i in range(k):
            f += normX[i] + norm(ARAt)**2 - 2*Xflat[i].multiply(ARAt).sum() + lmbda*(R[i].flatten()**2).sum()
        f *= 0.5
        
        fit = 1 - f / sumNormX
        fitchange = abs(fitold - fit)
        
        toc = time.clock()
        exectimes.append( toc - tic )
        _log.debug('[%3d] fit: %.5f | delta: %7.1e | secs: %.5f' % (iter, 
            fit, fitchange, exectimes[-1]))
        if iter > 1 and fitchange < conv:
            break
    return A, R, f, iter+1, array(exectimes)

def __updateA(X, A, R, lmbda):
    n, rank = A.shape
    F = zeros((n, rank), dtype=X[0].dtype)
    E = zeros((rank, rank), dtype=X[0].dtype)

    AtA = dot(A.T,A)
    for i in range(len(X)):
        ar = dot(A, R[i])
        summand = X[i].T.dot(ar)
        F += X[i].dot(dot(A, R[i].T)) + summand
        E += dot(R[i], dot(AtA, R[i].T)) + dot(R[i].T, dot(AtA, R[i]))
    A = dot(F, inv(lmbda * eye(rank) + E))
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

X = []
#X.append(array([[0, 1, 0, 1], [0, 1, 0, 1], [1, 1, 1, 1], [0, 0, 0, 0]]))
#X.append(array([[1, 0, 1, 1], [1, 0, 1, 1], [0, 1, 1, 1], [1, 0, 0, 0]]))
dim = 1000
numSlices = 5
for i in range(numSlices-1):
 row = random_integers(0,dim-1,0.2*dim*dim)
 col = random_integers(0,dim-1,0.2*dim*dim)
 A = coo_matrix((ones(row.size),(row,col)), shape=(dim,dim))
 X.append(A)

result = rescal(X, 4, lmbda=0.1)
#A, R, f, iter+1, array(exectimes)
print('Objective function value:')
print(result[2])
print('# of iterations:')
print(result[3])
#print('Matrix of latent embeddings:')
A = result[0]
savetxt("latent-embeddings.csv", A)

