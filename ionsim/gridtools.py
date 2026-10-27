"""Helpers shared by everything that lives on the electrostatic grid's nodes: insulators
painted beside the electrodes, mapping flight positions onto nodes, interpolation, and
depositing ion charge (cloud-in-cell) for space charge.

Grid kinds: "axisym" (nodes in r, z), "2d" (x, y cross-section, extruded along z) and
"3d" (x, y, z).
"""
import itertools
import numpy as np

EPS0 = 8.8541878128e-12


class Media:
    """Mixin for grids with an `electrode` id array: adds insulators (relative permittivity
    per node). Insulator nodes are solid (ions stop there) but not fixed in the solve."""
    eps = None
    insulator = None

    def paint_dielectric(self, mask, eps_r):
        if self.eps is None:
            self.eps = np.ones(self.electrode.shape, np.float32)
            self.insulator = np.zeros(self.electrode.shape, bool)
        self.eps[mask] = eps_r
        self.insulator[mask] = True

    @property
    def solid(self):
        el = self.electrode > 0
        return el if self.insulator is None else el | self.insulator


class GridMap:
    """Where the nodes of a grid are in flight coordinates (ions travel along +z)."""

    def __init__(self, kind, h, origin, shape, length=None):
        self.kind, self.h, self.shape = kind, h, tuple(shape)
        self.origin = np.asarray(origin, float)
        self.length = length            # "2d": extrusion length, to turn charge into charge per length

    @classmethod
    def of(cls, grid):
        """GridMap of an AxisymGrid or Grid3D."""
        if hasattr(grid, "r"):
            return cls("axisym", grid.h, (0.0, grid.z[0]), grid.electrode.shape)
        return cls("3d", grid.h, grid.origin, grid.shape)

    @property
    def dim(self):
        return len(self.shape)

    def coords(self, pos):
        """Fractional node indices of flight positions (N,3)."""
        if self.kind == "axisym":
            q = np.column_stack([np.hypot(pos[:, 0], pos[:, 1]), pos[:, 2]])
        elif self.kind == "2d":
            q = pos[:, :2]
        else:
            q = pos
        return (q - self.origin) / self.h

    def _corners(self, pos):
        gi = self.coords(pos)
        i0 = np.clip(np.floor(gi).astype(int), 0, np.array(self.shape) - 2)
        f = np.clip(gi - i0, 0.0, 1.0)
        for c in itertools.product((0, 1), repeat=self.dim):
            w = np.ones(len(pos))
            for ax, d in enumerate(c):
                w = w * (f[:, ax] if d else 1 - f[:, ax])
            yield tuple(i0[:, ax] + d for ax, d in enumerate(c)), w

    def interp(self, table, pos):
        """Linear interpolation of a node table (*shape, C) at flight positions -> (N, C)."""
        out = 0
        for idx, w in self._corners(pos):
            out = out + table[idx] * w[:, None]
        return out

    def gradient_table(self, phi, scale=-1.0):
        """scale * grad(phi) on the nodes, (*shape, dim) float32 (default: the field -grad phi)."""
        g = np.gradient(phi, self.h)
        return np.stack([scale * np.asarray(c, np.float32) for c in (g if self.dim > 1 else [g])], axis=-1)

    def to_xyz(self, v, pos):
        """Node-frame vectors (N, dim) to flight-frame (N, 3)."""
        out = np.zeros((len(pos), 3))
        if self.kind == "axisym":
            r = np.hypot(pos[:, 0], pos[:, 1])
            inv = np.divide(1.0, r, out=np.zeros_like(r), where=r > 0)
            out[:, 0], out[:, 1], out[:, 2] = v[:, 0] * pos[:, 0] * inv, v[:, 0] * pos[:, 1] * inv, v[:, 1]
        elif self.kind == "2d":
            out[:, :2] = v
        else:
            out[:] = v
        return out

    def vector(self, table, pos):
        return self.to_xyz(self.interp(table, pos), pos)

    def deposit(self, pos, weight):
        """Cloud-in-cell: spread weight (scalar or (N,)) at flight positions onto the nodes."""
        n = int(np.prod(self.shape))
        out = np.zeros(n)
        weight = np.broadcast_to(np.asarray(weight, float), (len(pos),))
        for idx, w in self._corners(pos):
            out += np.bincount(np.ravel_multi_index(idx, self.shape), weights=w * weight, minlength=n)
        return out.reshape(self.shape)

    def charge_source(self, q):
        """Node charges (C; C per ring for axisym) as the solver's source term. The solver's
        link weights are the physical eps0-free conductances divided by 2 pi h (axisym), h (3D)
        or 1 per unit length (2D)."""
        if self.kind == "axisym":
            return q / (2 * np.pi * self.h * EPS0)
        if self.kind == "2d":
            return q / (self.length * EPS0)
        return q / (self.h * EPS0)

    def flux_scale(self):
        """Physical area / length of a unit link weight (m), for fluxes through the grid."""
        return {"axisym": 2 * np.pi * self.h, "3d": self.h}.get(self.kind, self.length)


class Dwell:
    """Time each ion spends near each node, summed over a flight (cloud-in-cell). Positions
    are buffered and deposited in large batches."""

    def __init__(self, gmap, batch=2_000_000):
        self.gmap, self.batch = gmap, batch
        self.total = np.zeros(gmap.shape)
        self._pos, self._w, self._n = [], [], 0

    def add(self, pos, weight):
        if len(pos):
            self._pos.append(pos.copy()); self._w.append(np.full(len(pos), weight)); self._n += len(pos)
            if self._n >= self.batch:
                self._flush()

    def _flush(self):
        if self._pos:
            self.total += self.gmap.deposit(np.concatenate(self._pos), np.concatenate(self._w))
            self._pos, self._w, self._n = [], [], 0

    def result(self):
        self._flush()
        return self.total
