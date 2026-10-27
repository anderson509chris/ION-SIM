"""Laplace solve on a structured grid (any dimension) with algebraic multigrid.

Electrodes are fixed-potential nodes. Each electrode carries a Drive:
    V(t) = dc + rf cos(wt + phase)  =  dc + c cos(wt) - s sin(wt),  c = rf cos(phase), s = rf sin(phase)
Laplace is linear, so three solves (dc, c, s values on every electrode at once) give the
field at any time, however many electrodes there are: E(t) = E_dc + cos(wt) E_c - sin(wt) E_s.
Alternating 0/180 deg RF needs only the dc and c solves.

The grid is built from link weights between neighbouring nodes (finite volumes), so the
same code solves 3D Cartesian (all weights 1) and axisymmetric r-z grids (weights ~ r).
Grid faces are either grounded (Dirichlet 0) or open (zero normal field, Neumann).
"""
import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass


@dataclass
class Drive:
    dc: float = 0.0      # V
    rf: float = 0.0      # V, zero-to-peak
    phase: float = 0.0   # deg


def drive_parts(names, drives):
    """Per-electrode boundary values for the solves that are needed.
    Returns {"dc"|"cos"|"sin": array indexed by electrode id (0 = free space)}."""
    n = len(names) + 1
    dc, c, s = np.zeros(n), np.zeros(n), np.zeros(n)
    for k, name in enumerate(names, start=1):
        d = drives.get(name, Drive())
        ph = np.radians(d.phase)
        dc[k], c[k], s[k] = d.dc, d.rf * np.cos(ph), d.rf * np.sin(ph)
    c[np.abs(c) < 1e-12] = 0.0
    s[np.abs(s) < 1e-12] = 0.0
    parts = {"dc": dc}
    if np.any(c):
        parts["cos"] = c
    if np.any(s):
        parts["sin"] = s
    return parts


def part_coefficients(parts, freq, t):
    w = 2 * np.pi * freq * t
    return np.array([{"dc": 1.0, "cos": np.cos(w), "sin": -np.sin(w)}[p] for p in parts])


class LaplaceSolver:
    """fixed: bool array of Dirichlet nodes. weights[a]: link weights between node k and k+1
    along axis a (that axis one shorter). Out-of-range neighbours are open (Neumann)."""

    def __init__(self, fixed, weights, tol=1e-9):
        import pyamg
        self.shape, self.fixed, self.tol = fixed.shape, fixed, tol
        size = fixed.size
        idx = np.full(size, -1, dtype=np.int64)
        free = ~fixed.ravel()
        nf = int(free.sum())
        idx[free] = np.arange(nf)
        idx = idx.reshape(self.shape)
        flat = np.arange(size, dtype=np.int64).reshape(self.shape)
        rows, cols, vals = [], [], []
        diag = np.zeros(nf)
        self.couplings = []
        for ax, W in enumerate(weights):
            lo = tuple(slice(0, -1) if a == ax else slice(None) for a in range(fixed.ndim))
            hi = tuple(slice(1, None) if a == ax else slice(None) for a in range(fixed.ndim))
            ia, ib, w = idx[lo].ravel(), idx[hi].ravel(), np.broadcast_to(W, idx[lo].shape).ravel()
            fa, fb = ia >= 0, ib >= 0
            both = fa & fb
            rows += [ia[both], ib[both]]; cols += [ib[both], ia[both]]; vals += [-w[both], -w[both]]
            diag += np.bincount(ia[fa], weights=w[fa], minlength=nf)
            diag += np.bincount(ib[fb], weights=w[fb], minlength=nf)
            m = fa & ~fb
            self.couplings.append((ia[m], flat[hi].ravel()[m], w[m]))
            m = fb & ~fa
            self.couplings.append((ib[m], flat[lo].ravel()[m], w[m]))
        rows.append(np.arange(nf)); cols.append(np.arange(nf)); vals.append(diag)
        self.A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nf, nf))
        self.nf = nf
        self.ml = pyamg.smoothed_aggregation_solver(self.A, max_coarse=2000)

    def solve(self, fixed_values):
        """fixed_values: array of grid shape, used on fixed nodes. Returns the full potential."""
        fv = fixed_values.ravel()
        b = np.zeros(self.nf)
        for fi, fx, w in self.couplings:
            b += np.bincount(fi, weights=w * fv[fx], minlength=self.nf)
        phi = np.where(self.fixed, fixed_values, 0.0).astype(float)
        if np.any(b):
            phi[~self.fixed] = self.ml.solve(b, tol=self.tol, maxiter=300, accel="cg")
        return phi


def solve_drives(solver, electrode, names, drives):
    """Potentials for each needed part: {"dc"|"cos"|"sin": phi}."""
    return {p: solver.solve(v[electrode]) for p, v in drive_parts(names, drives).items()}
