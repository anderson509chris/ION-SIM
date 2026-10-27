"""Field sources. Every source exposes efield(t, pos) -> E (N,3) in V/m.

New physics (e.g. an axial drag field) is added as another FieldSource and the
Simulation simply sums them.
"""
import numpy as np


def rf_dc(U, V, freq, phase=0.0):
    """Rod voltage U + V cos(2 pi f t + phase). V is zero-to-peak."""
    omega = 2 * np.pi * freq
    return lambda t: U + V * np.cos(omega * t + phase)


class FieldSource:
    def efield(self, t, pos):
        raise NotImplementedError

    def hit(self, pos):
        """Boolean (N,) - True where the ion is inside an electrode / outside the region."""
        return np.zeros(len(pos), dtype=bool)


class GridField2D(FieldSource):
    """Superposition of solved basis potentials, field taken as uniform along z."""

    def __init__(self, grid, basis, voltages):
        self.grid = grid
        self.voltages = voltages          # name -> float or callable(t)
        self.dx = grid.dx
        self.x0 = grid.x[0]
        self.n = grid.n
        self.basis = basis
        # E = -grad(phi), per unit volt, precomputed once
        self.Ex, self.Ey = {}, {}
        for name, phi in basis.items():
            gx, gy = np.gradient(phi, grid.dx, grid.dx)
            self.Ex[name], self.Ey[name] = -gx, -gy

    def _v(self, name, t):
        v = self.voltages.get(name, 0.0)
        return v(t) if callable(v) else v

    def potential(self, t):
        return sum(self._v(n, t) * p for n, p in self.basis.items())

    def _bilinear(self, arr, i, j, fx, fy):
        return (arr[i, j] * (1 - fx) * (1 - fy) + arr[i + 1, j] * fx * (1 - fy)
                + arr[i, j + 1] * (1 - fx) * fy + arr[i + 1, j + 1] * fx * fy)

    def efield(self, t, pos):
        gx = (pos[:, 0] - self.x0) / self.dx
        gy = (pos[:, 1] - self.x0) / self.dx
        i = np.clip(np.floor(gx).astype(int), 0, self.n - 2)
        j = np.clip(np.floor(gy).astype(int), 0, self.n - 2)
        fx, fy = gx - i, gy - j
        E = np.zeros_like(pos)
        for name in self.basis:
            v = self._v(name, t)
            if v == 0.0:
                continue
            E[:, 0] += v * self._bilinear(self.Ex[name], i, j, fx, fy)
            E[:, 1] += v * self._bilinear(self.Ey[name], i, j, fx, fy)
        return E

    def hit(self, pos):
        gx = np.rint((pos[:, 0] - self.x0) / self.dx).astype(int)
        gy = np.rint((pos[:, 1] - self.x0) / self.dx).astype(int)
        out = (gx <= 0) | (gy <= 0) | (gx >= self.n - 1) | (gy >= self.n - 1)
        gx = np.clip(gx, 0, self.n - 1); gy = np.clip(gy, 0, self.n - 1)
        return out | (self.grid.electrode[gx, gy] > 0)


class IdealQuadField(FieldSource):
    """Analytic hyperbolic quadrupole: phi = Phi0(t) (x^2 - y^2) / r0^2. Used for validation."""

    def __init__(self, r0, phi0):
        self.r0, self.phi0 = r0, phi0

    def efield(self, t, pos):
        p = self.phi0(t) if callable(self.phi0) else self.phi0
        E = np.zeros_like(pos)
        E[:, 0] = -2 * p * pos[:, 0] / self.r0 ** 2
        E[:, 1] = 2 * p * pos[:, 1] / self.r0 ** 2
        return E

    def hit(self, pos):
        return pos[:, 0] ** 2 + pos[:, 1] ** 2 >= self.r0 ** 2
