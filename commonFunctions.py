

def squareFrobeniusNormOfSparse(M):
    """
    Computes the square of the Frobenius norm
    """
    rows, cols = M.nonzero()
    norm = 0
    for i in range(len(rows)):
        norm += M[rows[i],cols[i]] ** 2
    return norm