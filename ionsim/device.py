"""General devices built from CAD: electrodes, voltage patterns, grids, fields.

A Device is a list of electrodes (outlines from a DXF, or solid bodies from STEP / STL /
OBJ) plus how the drawing maps onto space:

  planar2d  2D cross-section, extruded along the ion axis (z). Quadrupoles, multipoles.
  axisym    half profile drawn with x = ion axis, y = radius; spun about the axis.
            Ring funnels, ring stacks, lenses, capillary exits.
  boards    top view of PCB pads (x = along the ion path, y = across), placed on two
            facing boards `board_gap` apart. TWAVE boards, planar funnels.
  cad3d     3D solids, one electrode per body.

Flight coordinates: ions travel along +z. planar2d: x, y = drawing x, y. axisym: z =
drawing x, r = radius. boards: z = drawing x, x = drawing y, y = normal to the boards.

Every electrode carries   V(t) = dc + rf cos(2 pi f t + phase)  [+ TWAVE amplitude when
its phase is on]. The field is the sum of a few solved "parts" (dc, cos, sin, one per
TWAVE phase), each a Laplace solve with those boundary values, so any time is a cheap
weighted sum and a voltage change re-solves only a few parts: a back-substitution on
2D grids, a warm-started multigrid solve in 3D.
"""
from dataclasses import dataclass, field, asdict
import numpy as np

from .laplace import LaplaceSolver
from .axisym import AxisymGrid
from .grid3d import Grid3D

KINDS = {
    "planar2d": "2D cross-section (extruded along the ion axis)",
    "axisym": "Axisymmetric half profile (drawing x = ion axis, y = radius)",
    "boards": "PCB boards: pads on two facing boards",
    "cad3d": "3D solids (STEP / STL / OBJ)",
}
QUALITY = {  # target cells across the largest dimension (2D) or total nodes (3D)
    "Quick": {"2d": 220, "3d": 400_000},
    "Detailed": {"2d": 900, "3d": 4_000_000},
}


@dataclass
class Electrode:
    name: str
    layer: str = ""
    shape: object = None        # dxf.Shape (2D kinds)
    tris: object = None         # (M,3,3) metres, flight coordinates (cad3d)
    dc: float = 0.0
    rf: float = 0.0             # V zero-to-peak
    phase: float = 0.0          # deg
    tw: int = -1                # TWAVE phase index, -1 = not a TWAVE electrode
    role: str = ""              # which pattern set it (shown in the table)

    def settings(self):
        return {"dc": self.dc, "rf": self.rf, "phase": self.phase, "tw": self.tw, "role": self.role}


@dataclass
class Twave:
    n_phases: int = 8
    n_on: int = 4
    amplitude: float = 20.0     # V added to the "on" electrodes
    speed: float = 200.0        # m/s
    pitch: float = 0.0          # m, electrode pitch along the path (0 = measured)

    def step_time(self):
        return self.pitch / self.speed if self.pitch > 0 and self.speed > 0 else np.inf


@dataclass
class Device:
    kind: str
    electrodes: list
    source: str = ""
    units: str = "mm"
    rf_freq: float = 1.0e6
    twave: Twave = field(default_factory=Twave)
    board_gap: float = 2.75e-3      # boards
    both_boards: bool = True        # boards: same pads on the facing board
    length: float = 0.1             # planar2d: length along the ion axis
    open_entrance: bool = True      # the z-min end is open (ions come from upstream)

    # ---------- geometry helpers ----------
    def names(self):
        return [e.name for e in self.electrodes]

    def get(self, name):
        return next(e for e in self.electrodes if e.name == name)

    def axis_pos(self, e):
        """Position of an electrode along the ion axis (used to order patterns)."""
        if self.kind == "cad3d":
            return float(e.tris[:, :, 2].mean())
        return float(e.shape.centroid[0])

    def extent(self):
        """(lo, hi) corners of the electrodes in flight coordinates (x, y, z)."""
        if self.kind == "cad3d":
            pts = np.concatenate([e.tris.reshape(-1, 3) for e in self.electrodes])
            return pts.min(0), pts.max(0)
        b = np.array([e.shape.bbox for e in self.electrodes])
        u0, v0, u1, v1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
        if self.kind == "planar2d":
            return np.array([u0, v0, 0.0]), np.array([u1, v1, self.length])
        if self.kind == "axisym":
            r = max(abs(v0), abs(v1))
            return np.array([-r, -r, u0]), np.array([r, r, u1])
        g = self.board_gap / 2
        return np.array([v0, -g, u0]), np.array([v1, g, u1])

    # ---------- patterns ----------
    @staticmethod
    def _ranks(pos, tol):
        """Rank of each position after merging positions closer than tol (same column)."""
        order = np.argsort(pos)
        ranks = np.zeros(len(pos), int)
        r, last = -1, None
        for i in order:
            if last is None or pos[i] - last > tol:
                r += 1
                last = pos[i]
            ranks[i] = r
        return ranks

    def _members(self, names):
        els = [self.get(n) for n in names]
        pos = np.array([self.axis_pos(e) for e in els])
        lo, hi = self.extent()
        tol = 1e-4 * float(np.max(hi - lo)) + 1e-9
        return els, pos, self._ranks(pos, tol)

    def apply_fixed(self, names, dc, rf=0.0, phase=0.0, label="fixed"):
        for e in (self.get(n) for n in names):
            e.dc, e.rf, e.phase, e.role = dc, rf, phase, label

    def apply_gradient(self, names, v_start, v_end, rf=0.0, alternate=True):
        """DC ramp from v_start (first electrode along the ion axis) to v_end (last),
        linear in position; optional RF alternating 0/180 deg from column to column."""
        els, pos, ranks = self._members(names)
        span = pos.max() - pos.min()
        for e, p, r in zip(els, pos, ranks):
            f = (p - pos.min()) / span if span > 0 else 0.0
            e.dc = v_start + (v_end - v_start) * f
            e.rf, e.phase = rf, (180.0 * (r % 2) if alternate else 0.0)
            e.role = "gradient"

    def apply_twave(self, names, start_phase=0, dc=None):
        """Assign TWAVE phases 0..N-1 in order along the ion axis (electrodes in the same
        column share a phase). Measures the pitch for the wave timing."""
        els, pos, ranks = self._members(names)
        n = self.twave.n_phases
        for e, r in zip(els, ranks):
            e.tw = int((start_phase + r) % n)
            if dc is not None:
                e.dc = dc
            e.role = f"TWAVE ph {e.tw + 1}"
        cols = np.array(sorted({round(p, 9) for p in pos}))
        if len(cols) > 1:
            self.twave.pitch = float(np.median(np.diff(cols)))

    def clear_twave(self, names):
        for e in (self.get(n) for n in names):
            e.tw = -1
            if e.role.startswith("TWAVE"):
                e.role = ""

    def has_twave(self):
        return any(e.tw >= 0 for e in self.electrodes)

    # ---------- grids ----------
    def default_h(self, quality="Quick"):
        lo, hi = self.extent()
        size = hi - lo
        if self.kind in ("planar2d", "axisym"):
            dims = size[[0, 1]] if self.kind == "planar2d" else np.array([size[0] / 2, size[2]])
            return float(dims.max() * 1.3 / QUALITY[quality]["2d"])
        if self.kind == "boards":
            size = size + np.array([0, 0, 0]) + np.array([2e-3, 0, 2e-3])
            size[1] = self.board_gap
        vol = float(np.prod(np.maximum(size * 1.2, 1e-6)))
        return float((vol / QUALITY[quality]["3d"]) ** (1 / 3))

    def build(self, h, pad=None):
        """Paint the electrodes onto a grid of spacing h. Returns (DeviceGrid, warnings)."""
        if pad is None:
            size = self.extent()[1] - self.extent()[0]
            size = size[:2] if self.kind == "planar2d" else size
            pad = max(8 * h, 0.1 * float(np.max(size)))
        return DeviceGrid(self, h, pad)

    # ---------- project file ----------
    def to_dict(self):
        return {"kind": self.kind, "source": self.source, "units": self.units, "rf_freq": self.rf_freq,
                "twave": asdict(self.twave), "board_gap": self.board_gap, "both_boards": self.both_boards,
                "length": self.length, "open_entrance": self.open_entrance,
                "electrodes": {e.name: e.settings() for e in self.electrodes}}

    def apply_dict(self, d):
        for k in ("rf_freq", "board_gap", "both_boards", "length", "open_entrance"):
            if k in d:
                setattr(self, k, d[k])
        if "twave" in d:
            self.twave = Twave(**d["twave"])
        missing = []
        for name, s in d.get("electrodes", {}).items():
            try:
                e = self.get(name)
            except StopIteration:
                missing.append(name)
                continue
            for k, v in s.items():
                setattr(e, k, v)
        return missing


def from_drawing(drawing, kind):
    els = [Electrode(s.name, s.layer, shape=s) for s in drawing.shapes]
    return Device(kind, els, drawing.path, drawing.units)


def from_bodies(bodies, source="", units="mm"):
    return Device("cad3d", [Electrode(n, "", tris=t) for n, t in bodies], source, units)


# ---------------------------------------------------------------------------------
class DeviceGrid:
    """Electrodes painted on a grid, with checks. Holds the (reusable) solver."""

    def __init__(self, dev, h, pad):
        self.dev, self.h, self.kind = dev, h, dev.kind
        self.warnings = []
        names = dev.names()
        lo, hi = dev.extent()
        if self.kind == "planar2d":
            m = int(np.ceil((max(abs(lo[0]), abs(hi[0]), abs(lo[1]), abs(hi[1])) + pad) / h))
            self.x = self.y = h * np.arange(-m, m + 1)
            self.electrode = np.zeros((len(self.x), len(self.y)), np.int16)
            X, Y = np.meshgrid(self.x, self.y, indexing="ij")
            self._paint2d(X, Y, lambda s, P: s.contains(P), (self.x[0], self.y[0]))
        elif self.kind == "axisym":
            self.g = AxisymGrid(hi[0] + pad, lo[2] - pad, hi[2] + pad, h, open_entrance=dev.open_entrance)
            self.g.names = list(names)
            R, Z = np.meshgrid(self.g.r, self.g.z, indexing="ij")
            self.electrode = self.g.electrode
            self._paint2d(Z, R, lambda s, P: s.contains(P) | s.contains(P * [1, -1]), (self.g.z[0], 0.0), swap=True)
        elif self.kind == "boards":
            gap = dev.board_gap / 2
            ym = gap + 4 * h
            self.g = Grid3D((lo[0] - pad, hi[0] + pad, -ym, ym, lo[2] - pad, hi[2] + pad), h,
                            open_faces=("z",) if dev.open_entrance else ("z+",))
            self.g.names = list(names)
            self.electrode = self.g.electrode
            X, Z = np.meshgrid(self.g.x, self.g.z, indexing="ij")
            jy = [int(np.argmin(np.abs(self.g.y - gap)))]
            if dev.both_boards:
                jy.append(int(np.argmin(np.abs(self.g.y + gap))))
            masks = [self._mask2d(e.shape, Z, X, (self.g.z[0], self.g.x[0]), swap=True) for e in dev.electrodes]
            for j in jy:
                self._paint_layer(self.electrode[:, j, :], masks, report=(j == jy[0]))
        else:
            self.g = Grid3D((lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad, lo[2] - pad, hi[2] + pad), h,
                            open_faces=("z-",) if dev.open_entrance else ())
            self.electrode = self.g.electrode
            for k, e in enumerate(dev.electrodes, start=1):
                before = self.electrode.copy()
                self.g.add_mesh(e.name, e.tris)
                new = self.electrode == k
                clobbered = (before > 0) & (before != k) & new
                if clobbered.any():
                    others = {names[i - 1] for i in np.unique(before[clobbered])}
                    self.warnings.append(f"'{e.name}' overlaps or nearly touches {', '.join(sorted(others))} on this grid.")
            for k, e in enumerate(dev.electrodes, start=1):
                self._check_small(e, int((self.electrode == k).sum()))
        self.nodes = int(self.electrode.size)
        self.names = names
        self._edge_check()
        self._solver = None

    # painting helpers
    def _mask2d(self, shape, U, V, origin, swap=False):
        """(nodes inside a shape, nodes nearest its outline). The outline nodes keep thin
        parts on the grid; they are only used where no other electrode is inside."""
        b = shape.bbox
        sel = (U >= b[0] - self.h) & (U <= b[2] + self.h) & (V >= b[1] - self.h) & (V <= b[3] + self.h)
        inside = np.zeros(U.shape, bool)
        if sel.any():
            inside[sel] = shape.contains(np.column_stack([U[sel], V[sel]]))
        m = np.zeros(U.shape, bool)
        for poly in [shape.outer] + list(shape.holes):
            seg = np.vstack([poly, poly[:1]])
            L = np.linalg.norm(np.diff(seg, axis=0), axis=1)
            for a, bb, l in zip(seg[:-1], seg[1:], L):
                n = max(int(np.ceil(l / (0.5 * self.h))), 1)
                p = a + (bb - a) * np.linspace(0, 1, n + 1)[:, None]
                iu = np.rint((p[:, 0] - origin[0]) / self.h).astype(int)
                iv = np.rint((p[:, 1] - origin[1]) / self.h).astype(int)
                if swap:
                    ok = (iu >= 0) & (iu < U.shape[1]) & (iv >= 0) & (iv < U.shape[0])
                    m[iv[ok], iu[ok]] = True
                else:
                    ok = (iu >= 0) & (iu < U.shape[0]) & (iv >= 0) & (iv < U.shape[1])
                    m[iu[ok], iv[ok]] = True
        return inside, m & ~inside

    def _paint2d(self, U, V, contains, origin, swap=False):
        masks = []
        for e in self.dev.electrodes:
            ins, edge = self._mask2d(e.shape, U, V, origin, swap)
            if self.kind == "axisym":          # outlines drawn below the axis count as their mirror
                i2, e2 = self._mask2d(_Mirror(e.shape), U, V, origin, swap)
                ins, edge = ins | i2, (edge | e2) & ~(ins | i2)
            masks.append((ins, edge))
        self._paint_layer(self.electrode, masks)

    def _paint_layer(self, arr, masks, report=True):
        """Interiors first (true overlaps are reported), then outline nodes where free;
        electrodes whose outlines collide are closer than the grid can separate."""
        overlaps, close = set(), set()
        for k, (ins, _) in enumerate(masks, start=1):
            clash = ins & (arr > 0)
            for i in np.unique(arr[clash]):
                overlaps.add(tuple(sorted((int(i), k))))
            arr[ins] = k
        interior = arr.copy()
        for k, (_, edge) in enumerate(masks, start=1):
            clash = edge & (arr > 0) & (arr != k)
            for i in np.unique(arr[clash]):
                pair = tuple(sorted((int(i), k)))
                if pair not in overlaps:
                    close.add(pair)
            arr[edge & (interior == 0)] = k
        if not report:
            return
        nm = lambda i: self.dev.electrodes[i - 1].name
        for a, b in sorted(overlaps):
            self.warnings.append(f"'{nm(a)}' and '{nm(b)}' overlap in the drawing.")
        if close:
            ex = ", ".join(f"{nm(a)}/{nm(b)}" for a, b in sorted(close)[:4])
            self.warnings.append(f"{len(close)} electrode pair(s) are closer than the grid spacing ({ex}"
                                 f"{', ...' if len(close) > 4 else ''}); on this grid they nearly touch. "
                                 "Use Detailed or a smaller spacing if the gap matters.")
        for k, (ins, edge) in enumerate(masks, start=1):
            self._check_small(self.dev.electrodes[k - 1], int((arr == k).sum()))

    def _check_small(self, e, n):
        if n == 0:
            self.warnings.append(f"'{e.name}' is not on the grid (smaller than a grid step?).")
            return
        if e.shape is not None:
            b = e.shape.bbox
            if min(b[2] - b[0], b[3] - b[1]) < 1.5 * self.h:
                self.warnings.append(f"'{e.name}' is thinner than 1.5 grid steps: use Detailed or a smaller spacing.")

    def _edge_check(self):
        el = self.electrode
        faces = [el[0], el[-1]] if el.ndim == 2 else [el[0], el[-1], el[:, 0], el[:, -1], el[:, :, 0], el[:, :, -1]]
        touching = set()
        for f in faces:
            touching |= set(np.unique(f[f > 0]).tolist())
        if self.kind == "axisym":
            touching -= set(np.unique(el[0][el[0] > 0]).tolist())   # the axis row is fine
        if touching:
            self.warnings.append("Touching the edge of the grid: " + ", ".join(self.names[i - 1] for i in sorted(touching))
                                 + ". Add space around the parts.")

    def solver(self):
        if self._solver is None:
            if self.kind == "planar2d":
                fixed = self.electrode > 0
                fixed[0, :] = fixed[-1, :] = fixed[:, 0] = fixed[:, -1] = True
                self._solver = LaplaceSolver(fixed, [np.ones(1), np.ones(1)])
            else:
                self._solver = self.g.solver()
        return self._solver


class _Mirror:
    """A dxf.Shape reflected in the drawing x axis (y -> -y)."""
    def __init__(self, s):
        from .dxf import Shape
        self.s = Shape(s.name, s.layer, s.outer * [1, -1], [h * [1, -1] for h in s.holes])

    def __getattr__(self, k):
        return getattr(self.s, k)


# ---------------------------------------------------------------------------------
def part_values(dev):
    """Boundary value per electrode id for every part: {"dc","cos","sin","tw0".."twN-1"}."""
    n = len(dev.electrodes) + 1
    parts = {"dc": np.zeros(n)}
    c, s = np.zeros(n), np.zeros(n)
    tw = np.zeros((dev.twave.n_phases, n))
    for k, e in enumerate(dev.electrodes, start=1):
        parts["dc"][k] = e.dc
        ph = np.radians(e.phase)
        c[k], s[k] = e.rf * np.cos(ph), e.rf * np.sin(ph)
        if 0 <= e.tw < dev.twave.n_phases:
            tw[e.tw, k] = 1.0
    c[np.abs(c) < 1e-12] = 0.0
    s[np.abs(s) < 1e-12] = 0.0
    if c.any():
        parts["cos"] = c
    if s.any():
        parts["sin"] = s
    for p in range(dev.twave.n_phases):
        if tw[p].any():
            parts[f"tw{p}"] = tw[p]
    return parts


class DeviceField:
    """Solved field of a DeviceGrid. Pass `previous` (an earlier DeviceField on the same
    grid) to warm-start the solves."""

    def __init__(self, dgrid, previous=None, progress=None):
        self.dg, self.dev, self.kind = dgrid, dgrid.dev, dgrid.kind
        self.freq = self.dev.rf_freq
        self.tw = Twave(**asdict(self.dev.twave))
        solver = dgrid.solver()
        vals = part_values(self.dev)
        self.parts = list(vals)
        guess = previous.phi if previous is not None and previous.dg is dgrid else {}
        el = dgrid.electrode
        self.phi = {}
        for i, (p, v) in enumerate(vals.items()):
            self.phi[p] = solver.solve(v[el], guess.get(p))
            if progress:
                progress((i + 1) / len(vals))
        h = dgrid.h
        comps = []
        for p in self.parts:
            comps += [-g.astype(np.float32) for g in np.gradient(self.phi[p], h)]
        self.table = np.stack(comps, axis=-1)
        self.dim = 3 if el.ndim == 3 else 2

    # ---- time dependence ----
    def coefficients(self, t):
        w = 2 * np.pi * self.freq * t
        step = np.floor(t / self.tw.step_time()) if np.isfinite(self.tw.step_time()) else 0
        out = []
        for p in self.parts:
            if p == "dc":
                out.append(1.0)
            elif p == "cos":
                out.append(np.cos(w))
            elif p == "sin":
                out.append(-np.sin(w))
            else:
                k = int(p[2:])
                on = ((k - step) % self.tw.n_phases) < self.tw.n_on
                out.append(self.tw.amplitude if on else 0.0)
        return np.array(out)

    # ---- interpolation ----
    def _grid_coords(self, pos):
        """Fractional grid indices for flight positions, and the axis map."""
        if self.kind == "planar2d":
            g = self.dg
            return np.column_stack([(pos[:, 0] - g.x[0]) / g.h, (pos[:, 1] - g.y[0]) / g.h])
        if self.kind == "axisym":
            g = self.dg.g
            r = np.hypot(pos[:, 0], pos[:, 1])
            return np.column_stack([r / g.h, (pos[:, 2] - g.z[0]) / g.h])
        g = self.dg.g
        return (pos - g.origin) / g.h

    def _interp(self, pos):
        gi = self._grid_coords(pos)
        shape = np.array(self.dg.electrode.shape)
        i0 = np.clip(np.floor(gi).astype(int), 0, shape - 2)
        f = gi - i0
        T, out = self.table, 0
        corners = [(a, b) for a in (0, 1) for b in (0, 1)] if self.dim == 2 else \
                  [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
        for cn in corners:
            w = np.ones(len(pos))
            idx = []
            for ax, d in enumerate(cn):
                w = w * (f[:, ax] if d else 1 - f[:, ax])
                idx.append(i0[:, ax] + d)
            out = out + T[tuple(idx)] * w[:, None]
        return out.reshape(len(pos), len(self.parts), self.dim)

    def efield(self, t, pos):
        v = np.einsum("npc,p->nc", self._interp(pos), self.coefficients(t))
        E = np.zeros((len(pos), 3))
        if self.kind == "planar2d":
            E[:, :2] = v
        elif self.kind == "axisym":
            r = np.hypot(pos[:, 0], pos[:, 1])
            inv = np.divide(1.0, r, out=np.zeros_like(r), where=r > 0)
            E[:, 0], E[:, 1], E[:, 2] = v[:, 0] * pos[:, 0] * inv, v[:, 0] * pos[:, 1] * inv, v[:, 1]
        else:
            E[:] = v
        return E

    def hit(self, pos):
        gi = np.rint(self._grid_coords(pos)).astype(int)
        shape = np.array(self.dg.electrode.shape)
        if self.kind == "axisym":
            out = (gi[:, 0] >= shape[0] - 1) | (gi[:, 1] <= 0) | (gi[:, 1] >= shape[1] - 1)
        elif self.kind == "planar2d":
            out = np.any((gi <= 0) | (gi >= shape - 1), axis=1)
        else:
            out = np.any((gi[:, :2] <= 0) | (gi[:, :2] >= shape[:2] - 1), axis=1) | (gi[:, 2] >= shape[2] - 1)
        gi = np.clip(gi, 0, shape - 1)
        return out | (self.dg.electrode[tuple(gi.T)] > 0)

    def potential(self, t=0.0):
        c = self.coefficients(t)
        return sum(ci * self.phi[p] for ci, p in zip(c, self.parts))

    def sections(self, t=0.0):
        """Plot sections: list of dicts (title, h, v coords, potential [v, h], mask [v, h], labels)."""
        phi, el = self.potential(t), self.dg.electrode
        mm = 1e3
        if self.kind == "planar2d":
            g = self.dg
            return [dict(title="cross-section", h=g.x * mm, v=g.y * mm, phi=phi.T, mask=el.T > 0,
                         hl="x (mm)", vl="y (mm)", ion="point")]
        if self.kind == "axisym":
            g = self.dg.g
            mir = lambda a: np.concatenate([a[:0:-1], a], axis=0)
            r = np.concatenate([-g.r[:0:-1], g.r])
            return [dict(title="r-z section", h=g.z * mm, v=r * mm, phi=mir(phi), mask=mir(el) > 0,
                         hl="z (mm, ion axis)", vl="r (mm)", ion="z")]
        g = self.dg.g
        jy = int(np.argmin(np.abs(g.y)))
        out = [dict(title="x-z section, y = 0" + (" (midplane between boards)" if self.kind == "boards" else ""),
                    h=g.z * mm, v=g.x * mm, phi=phi[:, jy, :], mask=el[:, jy, :] > 0,
                    hl="z (mm, ion axis)", vl="x (mm)", ion="z")]
        ix = int(np.argmin(np.abs(g.x - 0.5 * (g.x[0] + g.x[-1]))))
        out.append(dict(title="y-z section", h=g.z * mm, v=g.y * mm, phi=phi[ix], mask=el[ix] > 0,
                        hl="z (mm, ion axis)", vl="y (mm)", ion="z"))
        return out


def fly_device(fld, mz, K0, n, pressure_torr, temp_k=300.0, charge_state=1, center=(0.0, 0.0),
               radius=0.5e-3, z_start=None, z_end=None, energy_ev=0.0, gas_velocity=0.0, dt=None,
               t_max=None, record=40, rng=None, progress=None):
    """Fly n ions of one m/z through a DeviceField with gas (Langevin) dynamics.
    Near-vacuum works too: give a tiny pressure and an axial energy."""
    from .gasflight import disc_ions, fly
    from .twave import mobility
    from .constants import AMU, E_CHARGE
    lo, hi = fld.dev.extent()
    z_start = lo[2] if z_start is None else z_start
    z_end = hi[2] if z_end is None else z_end
    K = mobility(K0, max(pressure_torr, 1e-9), temp_k)
    freq = fld.freq
    has_rf = any(p in fld.parts for p in ("cos", "sin"))
    ions = disc_ions(n, mz * charge_state, K, charge_state, center, radius, z_start, temp_k,
                     birth_window=(1 / freq if has_rf else 0.0), rng=rng)
    if energy_ev > 0:
        ions.vel[:, 2] += np.sqrt(2 * energy_ev * E_CHARGE * charge_state / (mz * charge_state * AMU))
    if dt is None:
        cands = [1 / (freq * 30)] if has_rf else []
        if np.isfinite(fld.tw.step_time()) and any(p.startswith("tw") for p in fld.parts):
            cands.append(fld.tw.step_time() / 20)
        L = max(z_end - z_start, 1e-6)
        cands.append(L / 2000)
        tau = mz * charge_state * AMU * K / (charge_state * E_CHARGE)
        if not has_rf:
            cands.append(max(tau, 1e-9) * 50)
        dt = min(cands)
    if t_max is None:
        L = max(z_end - z_start, 1e-6)
        v = abs(gas_velocity) + (np.sqrt(2 * energy_ev * E_CHARGE / (mz * AMU)) if energy_ev > 0 else 0)
        if np.isfinite(fld.tw.step_time()) and any(p.startswith("tw") for p in fld.parts):
            v = max(v, 0.05 * fld.tw.speed)
        if pressure_torr > 0.01:
            v = max(v, K * 500.0)                    # ~5 V/cm drift as a floor
        t_max = 3 * L / max(v, 1.0)
    steps = t_max / dt
    if steps > 3e6:
        t_max = 3e6 * dt
    res = fly(fld, ions, dt, t_max, z_end, temp_k, gas_velocity, record=record,
              record_every=max(1, int(steps / 400)), rng=rng, progress=progress)
    return res, dict(dt=dt, t_max=t_max, z_start=z_start, z_end=z_end, K=K)
