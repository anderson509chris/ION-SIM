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
its phase is on], or is wired to an output of a Driver (drivers.py), whose waveform can
add excitation tones at other frequencies, its own TWAVE timing and edge rise times. A part with a relative permittivity (eps > 0) is an insulator instead
(PCB substrate, spacer): it shapes the field but carries no voltage. Boards can also get
an FR4 substrate behind the pads (board_thickness, board_eps), grounded on its back face. The field is the sum of a few solved "parts" (dc, cos, sin, one per
TWAVE phase), each a Laplace solve with those boundary values, so any time is a cheap
weighted sum and a voltage change re-solves only a few parts: a back-substitution on
2D grids, a warm-started multigrid solve in 3D.
"""
from dataclasses import dataclass, field, asdict
import numpy as np

from .laplace import LaplaceSolver
from .axisym import AxisymGrid
from .grid3d import Grid3D
from .gridtools import GridMap, Media
from .drivers import Driver, TwSpec

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
    eps: float = 0.0            # > 0: an insulator with this relative permittivity, not an electrode
    drive: str = ""             # "Driver / Output" when wired to a driver; overrides dc, rf, phase, tw

    def settings(self):
        return {"dc": self.dc, "rf": self.rf, "phase": self.phase, "tw": self.tw, "role": self.role, "eps": self.eps,
                "drive": self.drive}


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
    board_thickness: float = 0.0    # boards: substrate behind the pads (0 = none)
    board_eps: float = 4.4          # boards: substrate relative permittivity (FR4)
    drivers: dict = field(default_factory=dict)   # name -> Driver, for electrodes wired to drivers

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
        g = self.board_gap / 2 + self.board_thickness
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

    # ---------- drivers ----------
    def driver_of(self, e):
        """(Driver, Output) an electrode is wired to, or None."""
        if not e.drive or " / " not in e.drive:
            return None
        dn, on = e.drive.split(" / ", 1)
        d = self.drivers.get(dn)
        if d is None:
            return None
        try:
            return d, d.output(on)
        except KeyError:
            return None

    def tw_spec(self):
        """The device's own TWAVE timing (for electrodes not wired to a driver)."""
        t = self.twave
        return TwSpec("device", t.n_phases, t.n_on, t.amplitude, t.step_time(), 0.0)

    def components(self, e):
        """(dc, [(freq, amp, phase_deg)], (TwSpec, phase) or None) of one electrode."""
        w = self.driver_of(e)
        if w:
            return w[0].components(w[1])
        sines = [(self.rf_freq, e.rf, e.phase)] if e.rf else []
        tw = (self.tw_spec(), e.tw) if 0 <= e.tw < self.twave.n_phases else None
        return e.dc, sines, tw

    def pulse_of(self, e):
        """(PulseSpec, amplitude) of an electrode wired to a pulsed driver output, or None."""
        w = self.driver_of(e)
        return w[0].pulse_of(w[1]) if w else None

    def wave_of(self, e):
        """(Trace, scale) of an electrode wired to a driver output that carries an imported
        waveform, or None."""
        w = self.driver_of(e)
        return w[0].wave_of(w[1]) if w else None

    def wire_driver(self, names, drv, mode="one", output=None, start_phase=0, rf_alt="across"):
        """Wire electrodes to a driver. mode "one": all to `output`. mode "twave": TWAVE
        phases in order along the ion axis (electrodes side by side share a phase), each to
        the driver output carrying that phase; where the driver has RF+ and RF- versions of
        a phase, they alternate across the path (rf_alt="across"), along it ("along"), or
        RF+ only ("none")."""
        self.drivers[drv.name] = drv
        if mode == "one":
            for e in (self.get(n) for n in names):
                e.drive, e.role = f"{drv.name} / {output}", f"{drv.name}: {output}"
            return []
        els, pos, ranks = self._members(names)
        n = drv.tw_phases
        missing = set()
        # rank across the path among electrodes of the same column
        across = np.zeros(len(els), int)
        for r in set(ranks):
            idx = [i for i in range(len(els)) if ranks[i] == r]
            idx.sort(key=lambda i: self._across(els[i]))
            for j, i in enumerate(idx):
                across[i] = j
        for i, (e, r) in enumerate(zip(els, ranks)):
            ph = int((start_phase + r) % n)
            cands = [o for o in drv.outputs if o.tw == ph]
            if not cands:
                missing.add(ph + 1)
                continue
            pol = {"across": across[i] % 2, "along": r % 2}.get(rf_alt, 0)
            pick = [o for o in cands if (abs(((o.rf_phase + 90) % 360) - 90) > 90) == bool(pol)] or cands
            e.drive, e.role = f"{drv.name} / {pick[0].name}", f"{drv.name}: {pick[0].name}"
            e.tw = ph
        self.twave.n_phases = n
        cols = np.array(sorted({round(p, 9) for p in pos}))
        if len(cols) > 1:
            self.twave.pitch = float(np.median(np.diff(cols)))
        return sorted(missing)

    def _across(self, e):
        if self.kind == "cad3d":
            return float(e.tris[:, :, 0].mean())
        return float(e.shape.centroid[1])

    def unwire(self, names):
        for e in (self.get(n) for n in names):
            if e.drive:
                e.drive, e.role = "", ""

    # ---------- grids ----------
    def default_h(self, quality="Quick"):
        lo, hi = self.extent()
        size = hi - lo
        if self.kind in ("planar2d", "axisym"):
            dims = size[[0, 1]] if self.kind == "planar2d" else np.array([size[0] / 2, size[2]])
            return float(dims.max() * 1.3 / QUALITY[quality]["2d"])
        if self.kind == "boards":
            size = size + np.array([0, 0, 0]) + np.array([2e-3, 0, 2e-3])
            size[1] = self.board_gap + 2 * self.board_thickness
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
                "board_thickness": self.board_thickness, "board_eps": self.board_eps,
                "electrodes": {e.name: e.settings() for e in self.electrodes},
                "drivers": [d.to_dict() for n, d in self.drivers.items()
                            if any(e.drive.startswith(n + " / ") for e in self.electrodes)]}

    def apply_dict(self, d):
        for k in ("rf_freq", "board_gap", "both_boards", "length", "open_entrance", "board_thickness", "board_eps"):
            if k in d:
                setattr(self, k, d[k])
        if "twave" in d:
            self.twave = Twave(**d["twave"])
        for dd in d.get("drivers", []):
            drv = Driver.from_dict(dd)
            self.drivers[drv.name] = drv
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
class DeviceGrid(Media):
    """Electrodes painted on a grid, with checks. Holds the (reusable) solver.
    Insulators are painted like electrodes, then moved to the permittivity map."""

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
            ym = gap + (dev.board_thickness + 2 * h if dev.board_thickness > 0 else 4 * h)
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
            if dev.board_thickness > 0:
                yb = np.abs(self.g.y)
                sub = (yb > gap + 0.5 * h) & (yb <= gap + dev.board_thickness + 1e-12)
                sub &= np.sign(self.g.y) == np.sign(self.g.y[jy[0]]) if len(jy) == 1 else True
                m = np.zeros(self.electrode.shape, bool)
                m[:, sub, :] = True
                self.g.paint_dielectric(m, dev.board_eps)
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
        self._move_insulators()
        self._edge_check()
        self._solver = None

    def _move_insulators(self):
        media = getattr(self, "g", self)
        for k, e in enumerate(self.dev.electrodes, start=1):
            if e.eps > 0:
                m = self.electrode == k
                media.paint_dielectric(m, e.eps)
                self.electrode[m] = 0

    @property
    def media(self):
        """The object holding the permittivity map and solid mask."""
        return getattr(self, "g", self)

    def gridmap(self):
        if self.kind == "planar2d":
            return GridMap("2d", self.h, (self.x[0], self.y[0]), self.electrode.shape, length=self.dev.length)
        return GridMap.of(self.g)

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
                self._solver = LaplaceSolver(fixed, [np.ones(1), np.ones(1)], eps=self.eps)
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
    """Boundary value per electrode id for every part, and what each part is.
    Parts: "dc"; "cos"/"sin" at the device RF frequency and "cos@f"/"sin@f" at any other
    frequency (driver RF, excitation tones); "tw{p}" for the device's TWAVE phases and
    "tw@{driver}#{p}" for a driver's; "pulse@{driver}" for a driver's pulse and
    "wave@{driver}#{waveform}" for an imported waveform. Returns ({name: values}, {name: meta})."""
    n = len(dev.electrodes) + 1
    vals, meta = {"dc": np.zeros(n)}, {"dc": ("dc",)}

    def part(name, m):
        if name not in vals:
            vals[name], meta[name] = np.zeros(n), m
        return vals[name]

    for k, e in enumerate(dev.electrodes, start=1):
        dc, sines, tw = dev.components(e)
        vals["dc"][k] = dc
        for f, amp, ph in sines:
            tag = "" if np.isclose(f, dev.rf_freq, rtol=1e-9) else f"@{f:.9g}"
            part("cos" + tag, ("cos", f))[k] += amp * np.cos(np.radians(ph))
            part("sin" + tag, ("sin", f))[k] += amp * np.sin(np.radians(ph))
        if tw:
            spec, p = tw
            name = f"tw{p}" if spec.key == "device" else f"tw@{spec.key}#{p}"
            part(name, ("tw", spec, p))[k] = 1.0
        pu = dev.pulse_of(e)
        if pu:
            part(f"pulse@{pu[0].key}", ("pulse", pu[0]))[k] = pu[1]
        wv = dev.wave_of(e)
        if wv:
            drv = dev.driver_of(e)[0]
            part(f"wave@{drv.name}#{wv[0].name}", ("wave", wv[0]))[k] = wv[1]
    out_v, out_m = {}, {}
    for name, v in vals.items():
        v[np.abs(v) < 1e-12] = 0.0
        if name == "dc" or v.any():
            out_v[name], out_m[name] = v, meta[name]
    return out_v, out_m


class DeviceField:
    """Solved field of a DeviceGrid. Pass `previous` (an earlier DeviceField on the same
    grid) to warm-start the solves."""

    def __init__(self, dgrid, previous=None, progress=None):
        self.dg, self.dev, self.kind = dgrid, dgrid.dev, dgrid.kind
        self.freq = self.dev.rf_freq
        self.tw = Twave(**asdict(self.dev.twave))
        solver = dgrid.solver()
        vals, self.meta = part_values(self.dev)
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
        self.solid = dgrid.media.solid
        self.insulator = dgrid.media.insulator

    # ---- time dependence ----
    def coefficients(self, t):
        out = []
        for p in self.parts:
            m = self.meta[p]
            if m[0] == "dc":
                out.append(1.0)
            elif m[0] == "cos":
                out.append(np.cos(2 * np.pi * m[1] * t))
            elif m[0] == "sin":
                out.append(-np.sin(2 * np.pi * m[1] * t))
            elif m[0] in ("pulse", "wave"):
                out.append(float(m[1].level(t)))
            else:
                spec, k = m[1], m[2]
                out.append(spec.amp * float(spec.level(k, t)))
        return np.array(out)

    # ---- timing, for picking time steps ----
    def frequencies(self):
        return sorted({m[1] for m in self.meta.values() if m[0] in ("cos", "sin")})

    def tw_specs(self):
        return list({m[1] for m in self.meta.values() if m[0] == "tw"})

    def pulse_specs(self):
        return list({m[1] for m in self.meta.values() if m[0] == "pulse"})

    def waves(self):
        return [m[1] for m in self.meta.values() if m[0] == "wave"]

    def tw_speed(self):
        """Slowest wave speed (m/s) over the TWAVE sequencers in use, at the device pitch."""
        pitch = self.dev.twave.pitch
        sp = [pitch / s.step for s in self.tw_specs() if np.isfinite(s.step) and s.step > 0 and pitch > 0]
        return min(sp) if sp else 0.0

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
        return out | self.solid[tuple(gi.T)]

    def potential(self, t=0.0):
        c = self.coefficients(t)
        return sum(ci * self.phi[p] for ci, p in zip(c, self.parts))

    def sections(self, t=0.0):
        """Plot sections: list of dicts (title, h, v coords, potential [v, h], mask [v, h], labels)."""
        phi, el = self.potential(t), self.dg.electrode
        ins = self.insulator if self.insulator is not None else np.zeros(el.shape, bool)
        mm = 1e3
        if self.kind == "planar2d":
            g = self.dg
            return [dict(title="cross-section", h=g.x * mm, v=g.y * mm, phi=phi.T, mask=el.T > 0, ins=ins.T,
                         hl="x (mm)", vl="y (mm)", ion="point")]
        if self.kind == "axisym":
            g = self.dg.g
            mir = lambda a: np.concatenate([a[:0:-1], a], axis=0)
            r = np.concatenate([-g.r[:0:-1], g.r])
            return [dict(title="r-z section", h=g.z * mm, v=r * mm, phi=mir(phi), mask=mir(el) > 0, ins=mir(ins),
                         hl="z (mm, ion axis)", vl="r (mm)", ion="z")]
        g = self.dg.g
        jy = int(np.argmin(np.abs(g.y)))
        out = [dict(title="x-z section, y = 0" + (" (midplane between boards)" if self.kind == "boards" else ""),
                    h=g.z * mm, v=g.x * mm, phi=phi[:, jy, :], mask=el[:, jy, :] > 0, ins=ins[:, jy, :],
                    hl="z (mm, ion axis)", vl="x (mm)", ion="z")]
        ix = int(np.argmin(np.abs(g.x - 0.5 * (g.x[0] + g.x[-1]))))
        out.append(dict(title="y-z section", h=g.z * mm, v=g.y * mm, phi=phi[ix], mask=el[ix] > 0, ins=ins[ix],
                        hl="z (mm, ion axis)", vl="y (mm)", ion="z"))
        return out


def device_gas_flow(fld, slm, pressure_torr, temp_k, inlet_radius, center=(0.0, 0.0)):
    """Gas flow field (gasflow.GasFlow) through an axisymmetric or 3D device."""
    from .gasflow import GasFlow, volumetric_flow
    if fld.kind == "planar2d":
        raise ValueError("A 2D cross-section has no flow pattern along the axis: use a uniform gas speed")
    return GasFlow(fld.dg.g, volumetric_flow(slm, pressure_torr, temp_k), inlet_radius, center)


def fly_device(fld, mz, K0, n, pressure_torr, temp_k=300.0, charge_state=1, center=(0.0, 0.0),
               radius=0.5e-3, z_start=None, z_end=None, energy_ev=0.0, gas_velocity=0.0, dt=None,
               t_max=None, record=40, rng=None, progress=None, current=0.0, sc_iterations=4, heating=None):
    """Fly ions through a DeviceField with gas (Langevin) dynamics. mz, K0, charge_state and
    n are scalars (one kind of ion) or one value per species (a mixture, flown together;
    res.species gives each ion's species index, ions interleaved across species).
    Near-vacuum works too: give a tiny pressure and an axial energy.
    gas_velocity: uniform speed along z (m/s) or a GasFlow (device_gas_flow).
    current: ion beam current (A); non-zero iterates the beam's space charge.
    heating: a heating.HeatingModel to track ion heating (res.heat)."""
    from .gasflight import disc_ions, fly
    from .twave import mobility
    from .constants import AMU, E_CHARGE
    from .spacecharge import fly_with_space_charge
    rng = rng or np.random.default_rng()
    u_typ = 0.0 if hasattr(gas_velocity, "velocity") else abs(gas_velocity)   # a flow field mostly leaves sideways
    lo, hi = fld.dev.extent()
    z_start = lo[2] if z_start is None else z_start
    z_end = hi[2] if z_end is None else z_end
    from .ions import interleave
    mz_s = np.atleast_1d(np.asarray(mz, float))
    zc_s = np.broadcast_to(np.atleast_1d(np.asarray(charge_state, float)), mz_s.shape)
    K_s = mobility(np.broadcast_to(np.atleast_1d(np.asarray(K0, float)), mz_s.shape), max(pressure_torr, 1e-9), temp_k)
    sp = interleave(np.broadcast_to(np.atleast_1d(n), mz_s.shape))
    K = K_s.min()                                   # slowest species sets the flight time
    freqs = fld.frequencies()
    has_rf = bool(freqs)
    # ions are born spread over the longest period of the drive (RF, tones, or a repeating imported waveform)
    periods = [1 / f for f in freqs] + [w.cycle() for w in fld.waves() if w.repeat and w.cycle() > 0]
    steps_tw = [sp.step for sp in fld.tw_specs() if np.isfinite(sp.step) and sp.step > 0]
    if dt is None:
        cands = [1 / (max(freqs) * 30)] if has_rf else []
        cands += [st / 20 for st in steps_tw]
        cands += [sp.rise / 4 for sp in fld.tw_specs() if sp.rise > 0]
        cands += [r / 4 for sp in fld.pulse_specs() for r in (sp.rise, sp.fall) if r > 0]
        cands += [w.time_step() for w in fld.waves()]
        L = max(z_end - z_start, 1e-6)
        cands.append(L / 2000)
        tau = float(np.min(mz_s * AMU * K_s / E_CHARGE))
        if not has_rf:
            cands.append(max(tau, 1e-9) * 50)
        dt = min(cands)
    if t_max is None:
        L = max(z_end - z_start, 1e-6)
        v = u_typ + (np.sqrt(2 * energy_ev * E_CHARGE / (mz_s.max() * AMU)) if energy_ev > 0 else 0)
        if fld.tw_speed() > 0:
            v = max(v, 0.05 * fld.tw_speed())
        if pressure_torr > 0.01:
            v = max(v, K * 500.0)                    # ~5 V/cm drift as a floor
        t_max = 3 * L / max(v, 1.0)
    steps = t_max / dt
    if steps > 3e6:
        t_max = 3e6 * dt

    def fly_once(f, dwell=None, dwell_every=5, prog=progress):
        ions = disc_ions(len(sp), mz_s[sp] * zc_s[sp], K_s[sp], zc_s[sp], center, radius, z_start, temp_k,
                         birth_window=min(max(periods), 1e-4) if periods else 0.0, rng=rng)
        if energy_ev > 0:
            v_in = np.sqrt(2 * energy_ev * E_CHARGE / (mz_s[sp] * AMU))
            ions.vel[:, 2] += v_in
            ions.v0 = np.zeros_like(ions.vel)
            ions.v0[:, 2] = v_in - (0.0 if hasattr(gas_velocity, "velocity") else gas_velocity)
        return fly(f, ions, dt, t_max, z_end, temp_k, gas_velocity, record=record,
                   record_every=max(1, int(steps / 400)), rng=rng, progress=prog, dwell=dwell,
                   dwell_every=max(1, int(steps / 2000)), heating=heating)

    if current:
        res, _, _ = fly_with_space_charge(fly_once, fld, fld.dg.gridmap(), fld.dg.solver(), current,
                                          sc_iterations, progress=progress)
    else:
        res = fly_once(fld)
    res.species = sp
    return res, dict(dt=dt, t_max=t_max, z_start=z_start, z_end=z_end, K=K_s if len(K_s) > 1 else float(K_s[0]))
