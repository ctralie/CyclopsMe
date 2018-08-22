import numpy as np 
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_dilation
import scipy.sparse as sparse
import scipy.sparse.linalg as slinalg

def getBoundaryMask(mask):
    """
    Given a mask in which the inside region is a 1, return
    an image with 1s inside, 2s on the boundary, and -1s elsewhere.
    
    """
    win = np.ones((3, 3), dtype=int)
    boundin = binary_dilation(mask==0, win) & mask
    maskout = mask + boundin
    idx = -1*np.ones_like(mask)
    N = np.sum(maskout == 1)
    idx[maskout == 1] = np.arange(N)
    W = idx.shape[1] # Width
    H = idx.shape[0] # Height
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    X = X[maskout == 1]
    Y = Y[maskout == 1] 
    return {'mask':maskout, 'X':X, 'Y':Y, 'W':W, 'H':H, 'N':N, 'idx':idx}


def solve_poisson(source, dest, masksource, maskdest):
    # TODO: Pad image so the boundary is always inside of the image.
    # That will make things easier
    m1 = getBoundaryMask(masksource)
    m2 = getBoundaryMask(maskdest)
    N = m1['N']

    I = np.array([])
    J = np.array([])
    V = np.array([])
    b = np.zeros((N, 3))
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            nbX1 = m1['X'] + dx
            nbY1 = m1['Y'] + dy
            nbX2 = m2['X'] + dx
            nbY2 = m2['Y'] + dy
            idx = np.arange(N)
            # Step 1: Figure out how many neighbors there are in the selected region
            idx = idx[m1['mask'][nbY1, nbX1] == 1]
            # Update the neighbor count at each pixel accordingly
            I = np.concatenate((I, idx))
            J = np.concatenate((J, idx))
            V = np.concatenate((V, np.ones(idx.size)))

            # Step 2: Figure out the value of the laplacian at the source and add to b
            b[idx, :] += source[nbY1[idx], nbX1[idx], :] - source[m1['Y'][idx], m1['X'][idx], :]

            # Step 3: Figure out which neighbor pixels are on the boundary and which
            # are in the interior
            idxint = idx[m1['mask'][nbY1[idx], nbX1[idx]] == 1]
            idxbnd = idx[m1['mask'][nbY1[idx], nbX1[idx]] == 2]
            I = np.concatenate((I, idxint))
            J = np.concatenate((J, m2['idx'][nbY2[idxint], nbX2[idxint]]))
            V = np.concatenate((V, -1*np.ones(idxint.size)))
            b[idxbnd] += dest[m2['Y'][idxbnd], m2['X'][idxbnd], :]
    A = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
    res = np.zeros((N, 3))
    for c in range(3):
        print("Solving channel %i"%c)
        res[:, c] = slinalg.lsqr(A, b[:, c])[0]
    dest[m2['Y'], m2['X'], :] = res