"""Laplace solver. One solve per electrode at 1 V (others and boundary at 0 V).

Because Laplace is linear, any voltage set is then a weighted sum of these basis
solutions - exactly SIMION's "fast adjust" trick, which is what makes RF cheap.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def solve_basis(grid):
    n = grid.n
    fixed = grid.electrode > 0
    fixed[0, :] = fixed[-1, :] = fixed[:, 0] = fixed[:, -1] = True   # grounded boundary
    free = ~fixed
    idx = -np.ones((n, n), dtype=np.int64)
    idx[free] = np.arange(free.sum())
    nf = int(free.sum())

    # 5-point Laplacian on free nodes; neighbours that are fixed go to the RHS
    rows, cols, vals = [], [], []
    fi, fj = np.nonzero(free)
    me = idx[fi, fj]
    rows.append(me); cols.append(me); vals.append(np.full(nf, -4.0))
    neighbours = []
    for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        ni, nj = fi + di, fj + dj
        nb_free = free[ni, nj]
        rows.append(me[nb_free]); cols.append(idx[ni, nj][nb_free]); vals.append(np.ones(nb_free.sum()))
        neighbours.append((ni, nj, nb_free))
    A = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nf, nf))
    lu = spla.splu(A)

    basis = {}
    for k, name in enumerate(grid.names, start=1):
        boundary_v = (grid.electrode == k).astype(float)
        rhs = np.zeros(nf)
        for ni, nj, nb_free in neighbours:
            rhs -= np.where(~nb_free, boundary_v[ni, nj], 0.0)
        phi = boundary_v.copy()
        phi[free] = lu.solve(rhs)
        basis[name] = phi
    return basis
