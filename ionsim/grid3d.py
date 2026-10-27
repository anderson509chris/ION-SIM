"""3D Cartesian grid and field, for geometry that is not a stack of rings: planar
(two-board) funnels and electrodes imported from CAD (see cad.py).

Electrodes are painted onto the nodes either as boxes or from triangle meshes. A mesh
is voxelised by ray parity along z (fills closed solids) plus a surface shell sampled
at h/2 (keeps thin parts such as PCB copper, and open surfaces, from slipping between
nodes). Grid faces are grounded unless listed in `open_faces` (zero normal field):
"x", "y", "z" for both faces of an axis, or one face such as "z-" (the z-min face).
Insulators (PCB substrate, spacers) are painted the same way with a relative permittivity.
"""
import numpy as np
from .laplace import LaplaceSolver, solve_drives, part_coefficients
from .gridtools import Media

CHUNK = 4_000_000       # max candidate points per vectorised batch


class Grid3D(Media):
    def __init__(self, bounds, h, open_faces=()):
        """bounds = (xmin, xmax, ymin, ymax, zmin, zmax) in metres, h = node spacing."""
        self.h = h
        self.axes = [b0 + h * np.arange(int(round((b1 - b0) / h)) + 1)
                     for b0, b1 in zip(bounds[0::2], bounds[1::2])]
        self.x, self.y, self.z = self.axes
        self.shape = tuple(len(a) for a in self.axes)
        self.origin = np.array([a[0] for a in self.axes])
        self.electrode = np.zeros(self.shape, dtype=np.int16)
        self.names = []
        self.open_faces = {open_faces} if isinstance(open_faces, str) else set(open_faces)

    @property
    def nodes(self):
        return int(np.prod(self.shape))

    def _id(self, name):
        if name not in self.names:
            self.names.append(name)
        return self.names.index(name) + 1

    def _range(self, ax, a0, a1):
        i0 = int(round((a0 - self.axes[ax][0]) / self.h))
        i1 = max(i0, int(round((a1 - self.axes[ax][0]) / self.h)))
        return slice(max(i0, 0), max(min(i1 + 1, self.shape[ax]), 0))

    def _box(self, x0, x1, y0, y1, z0, z1):
        return self._range(0, x0, x1), self._range(1, y0, y1), self._range(2, z0, z1)

    def add_box(self, name, x0, x1, y0, y1, z0, z1):
        """Axis-aligned box, at least one node thick in every direction."""
        self.electrode[self._box(x0, x1, y0, y1, z0, z1)] = self._id(name)

    def add_dielectric_box(self, x0, x1, y0, y1, z0, z1, eps_r):
        """Insulating box (relative permittivity eps_r); electrodes on the same nodes win."""
        m = np.zeros(self.shape, bool)
        m[self._box(x0, x1, y0, y1, z0, z1)] = True
        self.paint_dielectric(m, eps_r)

    def mesh_mask(self, tris):
        """Nodes of a triangle mesh (M,3,3) in metres: closed solids are filled, any surface is kept."""
        fill = self._fill(tris)
        near = fill.copy()                      # fill grown by one node: the shell only adds
        for ax in range(3):                     # nodes where a part is too thin to be filled
            lo = tuple(slice(0, -1) if a == ax else slice(None) for a in range(3))
            hi = tuple(slice(1, None) if a == ax else slice(None) for a in range(3))
            near[lo] |= fill[hi]; near[hi] |= fill[lo]
        return fill | (self._shell(tris) & ~near)

    def add_mesh(self, name, tris):
        """Electrode from triangles (M,3,3) in metres."""
        self.electrode[self.mesh_mask(tris)] = self._id(name)

    def add_dielectric_mesh(self, tris, eps_r):
        self.paint_dielectric(self.mesh_mask(tris), eps_r)

    def _fill(self, tris):
        h, (nx, ny, nz) = self.h, self.shape
        x0, y0, z0 = self.origin + np.array([1.234567e-4, 2.345678e-4, 0]) * h   # jitter rays off edges
        hits_col, hits_z = [], []
        txy = tris[:, :, :2]
        i0 = np.ceil((txy[:, :, 0].min(1) - x0) / h).astype(int).clip(0, nx)
        i1 = np.floor((txy[:, :, 0].max(1) - x0) / h).astype(int).clip(-1, nx - 1)
        j0 = np.ceil((txy[:, :, 1].min(1) - y0) / h).astype(int).clip(0, ny)
        j1 = np.floor((txy[:, :, 1].max(1) - y0) / h).astype(int).clip(-1, ny - 1)
        ni, nj = np.maximum(i1 - i0 + 1, 0), np.maximum(j1 - j0 + 1, 0)
        counts = ni * nj
        tri_ids = np.nonzero(counts)[0]
        start = 0
        while start < len(tri_ids):
            csum = np.cumsum(counts[tri_ids[start:]])
            stop = start + max(1, int(np.searchsorted(csum, CHUNK)))
            t = tri_ids[start:stop]
            c = counts[t]
            ti = np.repeat(t, c)
            off = np.arange(c.sum()) - np.repeat(np.cumsum(c) - c, c)
            ii = i0[ti] + off // nj[ti]
            jj = j0[ti] + off % nj[ti]
            px, py = x0 + ii * h, y0 + jj * h
            a, b, cc = tris[ti, 0], tris[ti, 1], tris[ti, 2]
            det = (b[:, 0] - a[:, 0]) * (cc[:, 1] - a[:, 1]) - (cc[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
            ok = np.abs(det) > 1e-30
            det = np.where(ok, det, 1.0)
            u = ((px - a[:, 0]) * (cc[:, 1] - a[:, 1]) - (cc[:, 0] - a[:, 0]) * (py - a[:, 1])) / det
            v = ((b[:, 0] - a[:, 0]) * (py - a[:, 1]) - (px - a[:, 0]) * (b[:, 1] - a[:, 1])) / det
            inside = ok & (u >= 0) & (v >= 0) & (u + v <= 1)
            zc = a[:, 2] + u * (b[:, 2] - a[:, 2]) + v * (cc[:, 2] - a[:, 2])
            hits_col.append((ii * ny + jj)[inside]); hits_z.append(zc[inside])
            start = stop
        fill = np.zeros(self.shape, dtype=bool)
        if not hits_col:
            return fill
        col, z = np.concatenate(hits_col), np.concatenate(hits_z)
        order = np.lexsort((z, col))
        col, z = col[order], z[order]
        first = np.r_[0, np.nonzero(np.diff(col))[0] + 1]
        rank = np.arange(len(col)) - np.repeat(first, np.diff(np.r_[first, len(col)]))
        n_in_col = np.repeat(np.diff(np.r_[first, len(col)]), np.diff(np.r_[first, len(col)]))
        keep = rank < n_in_col - (n_in_col % 2)          # drop an unpaired last crossing
        col, z, rank = col[keep], z[keep], rank[keep]
        kz = np.where(rank % 2 == 0, np.ceil((z - z0) / h), np.floor((z - z0) / h) + 1).astype(int).clip(0, nz)
        diff = np.zeros((nx * ny, nz + 1), dtype=np.int32)
        np.add.at(diff, (col, kz), np.where(rank % 2 == 0, 1, -1))
        return (np.cumsum(diff, axis=1)[:, :nz] > 0).reshape(self.shape)

    def _shell(self, tris):
        h = self.h
        shell = np.zeros(self.shape, dtype=bool)
        edge = np.linalg.norm(tris - np.roll(tris, 1, axis=1), axis=2).max(1)
        nseg = np.maximum(np.ceil(edge / (0.5 * h)).astype(int), 1)
        for n in np.unique(nseg):
            uu, vv = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
            m = uu + vv <= n
            w = np.column_stack([1 - (uu[m] + vv[m]) / n, uu[m] / n, vv[m] / n])    # (P,3)
            group = tris[nseg == n]
            step = max(1, CHUNK // len(w))
            for s in range(0, len(group), step):
                p = np.einsum("pk,mkd->mpd", w, group[s:s + step]).reshape(-1, 3)
                idx = np.rint((p - self.origin) / h).astype(int)
                ok = np.all((idx >= 0) & (idx < np.array(self.shape)), axis=1)
                shell[tuple(idx[ok].T)] = True
        return shell

    def solver(self):
        fixed = self.electrode > 0
        for ax, name in enumerate("xyz"):
            for end, face in ((0, name + "-"), (-1, name + "+")):
                if name not in self.open_faces and face not in self.open_faces:
                    fixed[tuple(end if a == ax else slice(None) for a in range(3))] = True
        return LaplaceSolver(fixed, self.link_weights(), eps=self.eps)

    def link_weights(self):
        return [np.ones(1)] * 3


class GridField3D:
    """Field of a Grid3D for a set of electrode Drives at RF frequency `freq`."""

    def __init__(self, grid, drives, freq, solver=None, guess=None):
        """guess: the phi dict of an earlier field on the same grid (warm start)."""
        self.grid, self.freq = grid, freq
        self.solver = solver or grid.solver()
        self.phi = solve_drives(self.solver, grid.electrode, grid.names, drives, guess)
        self.parts = list(self.phi)
        comps = []
        for p in self.parts:
            comps += [-g.astype(np.float32) for g in np.gradient(self.phi[p], grid.h)]
        self.table = np.stack(comps, axis=-1)     # (nx, ny, nz, 3*parts)
        self.solid = grid.solid                   # electrodes and insulators: ions stop there

    def _interp(self, pos):
        g = self.grid
        gi = (pos - g.origin) / g.h
        i0 = np.clip(np.floor(gi).astype(int), 0, np.array(g.shape) - 2)
        f = gi - i0
        T, out = self.table, 0
        for dx in (0, 1):
            wx = f[:, 0] if dx else 1 - f[:, 0]
            for dy in (0, 1):
                wy = f[:, 1] if dy else 1 - f[:, 1]
                for dz in (0, 1):
                    wz = f[:, 2] if dz else 1 - f[:, 2]
                    out = out + T[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz] * (wx * wy * wz)[:, None]
        return out.reshape(len(pos), len(self.parts), 3)

    def efield(self, t, pos):
        return np.einsum("npc,p->nc", self._interp(pos), part_coefficients(self.parts, self.freq, t)).astype(float)

    def hit(self, pos):
        g = self.grid
        idx = np.rint((pos - g.origin) / g.h).astype(int)
        out = np.any((idx <= 0) | (idx >= np.array(g.shape) - 1), axis=1)
        idx = np.clip(idx, 0, np.array(g.shape) - 1)
        return out | self.solid[idx[:, 0], idx[:, 1], idx[:, 2]]

    def rf_e2(self, pos):
        v = self._interp(pos)
        idx = [k for k, p in enumerate(self.parts) if p != "dc"]
        return 0.5 * np.sum(v[:, idx, :] ** 2, axis=(1, 2)) if idx else np.zeros(len(pos))

    def sections(self, center=(0.0, 0.0)):
        """{"x": x-z plane through y=center[1], "y": y-z plane through x=center[0]}, each a dict of
        transverse coords, z coords, grid step, electrode mask, dc potential and rf part potentials."""
        g = self.grid
        jy = int(np.clip(round((center[1] - g.y[0]) / g.h), 0, g.shape[1] - 1))
        ix = int(np.clip(round((center[0] - g.x[0]) / g.h), 0, g.shape[0] - 1))
        out = {}
        for key, sl, coords in (("x", (slice(None), jy), g.x), ("y", (ix, slice(None)), g.y)):
            out[key] = {"coords": coords, "z": g.z, "h": g.h, "mask": g.electrode[sl] > 0, "dc": self.phi["dc"][sl],
                        "insulator": None if g.insulator is None else g.insulator[sl],
                        "rf": [self.phi[p][sl] for p in self.parts if p != "dc"]}
        return out
