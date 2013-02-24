from numpy import dot

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