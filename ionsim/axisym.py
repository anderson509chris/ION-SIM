"""Axisymmetric (r, z) grid and field: exact for stacks of ring electrodes.

Laplace in cylindrical coordinates with finite volumes: the link weight between nodes
is the radius of the face between them (r_{i+1/2} radially, r_i axially, 1/8 on the
axis), which keeps the matrix symmetric and handles r = 0 without special cases.
The outer radius and the far z end are grounded; the entrance end (z min) is grounded
too unless `open_entrance`, where it has zero normal field, as if the space in front
were filled by more of the first electrode's surroundings (an inlet, not a grounded wall).
"""
import numpy as np
from .laplace import LaplaceSolver, solve_drives, part_coefficients


class AxisymGrid:
    def __init__(self, r_max, z_min, z_max, h, open_entrance=False):
        self.h, self.open_entrance = h, open_entrance
        self.nr = int(round(r_max / h)) + 1
        self.nz = int(round((z_max - z_min) / h)) + 1
        self.r = h * np.arange(self.nr)
        self.z = z_min + h * np.arange(self.nz)
        self.electrode = np.zeros((self.nr, self.nz), dtype=np.int16)   # 0 = free space
        self.names = []

    @property
    def nodes(self):
        return self.nr * self.nz

    def _id(self, name):
        if name not in self.names:
            self.names.append(name)
        return self.names.index(name) + 1

    def add_annulus(self, name, r_in, r_out, z0, z1):
        """Ring from radius r_in to r_out, z0..z1. Always at least one node thick."""
        i0 = int(np.ceil(r_in / self.h - 1e-9))
        i1 = max(i0, int(np.floor(r_out / self.h + 1e-9)))
        j0 = int(round((z0 - self.z[0]) / self.h))
        j1 = max(j0, int(round((z1 - self.z[0]) / self.h)))
        self.electrode[i0:i1 + 1, max(j0, 0):j1 + 1] = self._id(name)

    def solver(self):
        i = np.arange(self.nr, dtype=float)
        wr = (i[:-1] + 0.5)[:, None]                      # radial links
        wz = np.where(i == 0, 0.125, i)[:, None]           # axial links
        fixed = self.electrode > 0
        fixed[-1, :] = fixed[:, -1] = True                 # grounded can
        if not self.open_entrance:
            fixed[:, 0] = True
        return LaplaceSolver(fixed, [wr, wz])


class AxisymField:
    """Field of an AxisymGrid for a set of electrode Drives at RF frequency `freq`."""

    def __init__(self, grid, drives, freq, solver=None, guess=None):
        """guess: the phi dict of an earlier field on the same grid (warm start)."""
        self.grid, self.freq = grid, freq
        self.solver = solver or grid.solver()
        self.phi = solve_drives(self.solver, grid.electrode, grid.names, drives, guess)
        self.parts = list(self.phi)
        comps = []
        for p in self.parts:
            gr, gz = np.gradient(self.phi[p], grid.h, grid.h)
            comps += [-gr, -gz]
        self.table = np.stack(comps, axis=-1)              # (nr, nz, 2*parts): Er, Ez per part
        self.r_max = grid.r[-1]

    def _interp(self, pos):
        g = self.grid
        r = np.hypot(pos[:, 0], pos[:, 1])
        gr, gz = r / g.h, (pos[:, 2] - g.z[0]) / g.h
        i = np.clip(np.floor(gr).astype(int), 0, g.nr - 2)
        j = np.clip(np.floor(gz).astype(int), 0, g.nz - 2)
        fr, fz = (gr - i)[:, None], (gz - j)[:, None]
        T = self.table
        v = (T[i, j] * (1 - fr) * (1 - fz) + T[i + 1, j] * fr * (1 - fz)
             + T[i, j + 1] * (1 - fr) * fz + T[i + 1, j + 1] * fr * fz)
        return r, v.reshape(len(pos), len(self.parts), 2)

    def efield(self, t, pos):
        r, v = self._interp(pos)
        erz = np.einsum("npc,p->nc", v, part_coefficients(self.parts, self.freq, t))
        inv = np.divide(1.0, r, out=np.zeros_like(r), where=r > 0)
        return np.column_stack([erz[:, 0] * pos[:, 0] * inv, erz[:, 0] * pos[:, 1] * inv, erz[:, 1]])

    def hit(self, pos):
        g = self.grid
        ir = np.rint(np.hypot(pos[:, 0], pos[:, 1]) / g.h).astype(int)
        jz = np.rint((pos[:, 2] - g.z[0]) / g.h).astype(int)
        out = (ir >= g.nr - 1) | (jz <= 0) | (jz >= g.nz - 1)
        return out | (g.electrode[np.clip(ir, 0, g.nr - 1), np.clip(jz, 0, g.nz - 1)] > 0)

    # ---- for plots ----
    def sections(self, center=(0.0, 0.0)):
        """Same format as GridField3D.sections: the mirrored r-z plane serves as both x-z and y-z."""
        g = self.grid
        mirror = lambda a: np.concatenate([a[:0:-1], a], axis=0)
        sec = {"coords": np.concatenate([-g.r[:0:-1], g.r]), "z": g.z, "h": g.h, "mask": mirror(g.electrode > 0),
               "dc": mirror(self.phi["dc"]), "rf": [mirror(self.phi[p]) for p in self.parts if p != "dc"]}
        return {"x": sec, "y": sec}

    def rf_e2(self, pos):
        """Mean square RF field <E^2> at pos (for the pseudopotential)."""
        _, v = self._interp(pos)
        idx = [k for k, p in enumerate(self.parts) if p != "dc"]
        return 0.5 * np.sum(v[:, idx, :] ** 2, axis=(1, 2)) if idx else np.zeros(len(pos))
