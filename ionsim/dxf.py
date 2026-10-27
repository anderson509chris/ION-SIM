"""Electrode outlines from DXF drawings.

Every closed shape becomes an electrode: closed polylines, circles, ellipses, closed
splines, hatch boundaries, and loops made of separate LINE / ARC pieces that meet end to
end. A closed shape lying inside another closed shape on the same layer is a hole in it
(so a ring cross-section is two concentric circles). Blocks (INSERT) are exploded.

Names: a TEXT / MTEXT inside a shape names it; otherwise "<layer> <n>".
Units come from the drawing header ($INSUNITS); unitless drawings fall back to the
caller's choice. Open pieces that cannot be joined into loops are reported, not used.
"""
import os
from dataclasses import dataclass, field
import numpy as np

INSUNITS = {1: "inch", 2: "ft", 4: "mm", 5: "cm", 6: "m", 13: "µm"}
UNIT_M = {"mm": 1e-3, "cm": 1e-2, "m": 1.0, "µm": 1e-6, "inch": 25.4e-3, "ft": 0.3048}


@dataclass
class Shape:
    name: str
    layer: str
    outer: np.ndarray                          # (K,2) closed outline, metres
    holes: list = field(default_factory=list)  # [(K,2)]

    @property
    def centroid(self):
        return self.outer.mean(axis=0)

    @property
    def bbox(self):
        return np.r_[self.outer.min(0), self.outer.max(0)]

    def contains(self, pts):
        from matplotlib.path import Path
        inside = Path(self.outer).contains_points(pts)
        for h in self.holes:
            inside &= ~Path(h).contains_points(pts)
        return inside


@dataclass
class Drawing:
    path: str
    units: str
    shapes: list
    warnings: list

    @property
    def layers(self):
        return sorted({s.layer for s in self.shapes})

    @property
    def bbox(self):
        b = np.array([s.bbox for s in self.shapes])
        return np.r_[b[:, :2].min(0), b[:, 2:].max(0)]


def _area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _entities(msp):
    """Model-space entities with blocks exploded (recursively)."""
    for e in msp:
        if e.dxftype() == "INSERT":
            try:
                yield from _entities(list(e.virtual_entities()))
            except Exception:
                continue
        else:
            yield e


def load_dxf(path, default_units="mm"):
    import ezdxf
    from ezdxf import path as ezpath
    doc = ezdxf.readfile(path)
    code = doc.header.get("$INSUNITS", 0)
    units = INSUNITS.get(code, default_units)
    scale = UNIT_M[units]
    warnings = [] if code in INSUNITS else [f"Drawing has no units set; using {units}."]

    loops, opens, texts = [], [], []
    ents = list(_entities(doc.modelspace()))
    # flattening tolerance from the drawing size
    pts = []
    for e in ents:
        try:
            p = ezpath.make_path(e) if e.dxftype() != "HATCH" else None
            if p is not None and len(p):
                pts.append(np.array([(q.x, q.y) for q in p.control_vertices()]))
        except Exception:
            pass
    ext = np.ptp(np.concatenate(pts), axis=0).max() if pts else 1.0
    tol = max(ext, 1e-9) * 2e-4
    skipped = {}
    for e in ents:
        kind, layer = e.dxftype(), e.dxf.layer
        if kind in ("TEXT", "MTEXT"):
            txt = e.plain_text() if kind == "MTEXT" else e.dxf.text
            ins = e.dxf.insert
            texts.append((txt.strip(), np.array([ins[0], ins[1]])))
            continue
        try:
            paths = list(ezpath.from_hatch(e)) if kind == "HATCH" else [ezpath.make_path(e)]
        except Exception:
            skipped[kind] = skipped.get(kind, 0) + 1
            continue
        for p in paths:
            if not len(p) and kind != "CIRCLE":
                continue
            v = np.array([(q.x, q.y) for q in p.flattening(tol)])
            if len(v) < 2:
                continue
            closed = p.is_closed or np.linalg.norm(v[0] - v[-1]) <= tol
            if closed and len(v) >= 3:
                loops.append((layer, v[:-1] if np.allclose(v[0], v[-1]) else v))
            else:
                opens.append((layer, v))
    for k, n in skipped.items():
        warnings.append(f"Skipped {n} {k} entit{'y' if n == 1 else 'ies'} (not a shape).")

    # join open pieces end to end, per layer
    joined, left = _join(opens, tol * 5)
    loops += joined
    if left:
        by_layer = {}
        for layer, _ in left:
            by_layer[layer] = by_layer.get(layer, 0) + 1
        for layer, n in by_layer.items():
            warnings.append(f"Layer '{layer}': {n} open line(s) that do not form a closed shape were ignored.")

    # holes: loops inside a larger loop on the same layer
    loops = [(layer, v) for layer, v in loops if _area(v) > 0]
    loops.sort(key=lambda lv: -_area(lv[1]))
    from matplotlib.path import Path
    shapes, depth = [], []
    for layer, v in loops:
        parent = None
        for k in range(len(shapes) - 1, -1, -1):
            s = shapes[k]
            if s.layer == layer and Path(s.outer).contains_points(v[:1])[0] and depth[k] % 2 == 0:
                if not any(Path(h).contains_points(v[:1])[0] for h in s.holes):
                    parent = k
                    break
        if parent is None:
            shapes.append(Shape("", layer, v)); depth.append(0)
        else:
            shapes[parent].holes.append(v)
    # names, numbered along the drawing's x then y within each layer
    shapes.sort(key=lambda s: (s.layer, round(float(s.centroid[0]), 9), float(s.centroid[1])))
    count = {}
    for s in shapes:
        label = next((t for t, p in texts if t and s.contains(p[None])[0]), None)
        if label is None:
            count[s.layer] = count.get(s.layer, 0) + 1
            label = f"{s.layer} {count[s.layer]}"
        s.name = label
    names = [s.name for s in shapes]
    for k, s in enumerate(shapes):
        if names.count(s.name) > 1:
            s.name = f"{s.name} ({k + 1})"
    # to metres
    for s in shapes:
        s.outer = s.outer * scale
        s.holes = [h * scale for h in s.holes]
    if not shapes:
        warnings.append("No closed shapes found. Electrodes must be closed outlines (polylines, circles, or joined lines/arcs).")
    return Drawing(os.path.abspath(path), units, shapes, warnings)


def _join(opens, tol):
    """Greedy end-to-end joining of open polylines on the same layer.
    Returns (closed loops, pieces left open)."""
    used = [False] * len(opens)
    loops, left = [], []
    for i, (layer, v) in enumerate(opens):
        if used[i]:
            continue
        used[i] = True
        chain, members, grew = v.copy(), [i], True
        while grew and np.linalg.norm(chain[0] - chain[-1]) > tol:
            grew = False
            for j, (lj, w) in enumerate(opens):
                if used[j] or lj != layer:
                    continue
                if np.linalg.norm(chain[-1] - w[0]) <= tol:
                    chain = np.vstack([chain, w[1:]])
                elif np.linalg.norm(chain[-1] - w[-1]) <= tol:
                    chain = np.vstack([chain, w[::-1][1:]])
                elif np.linalg.norm(chain[0] - w[-1]) <= tol:
                    chain = np.vstack([w[:-1], chain])
                elif np.linalg.norm(chain[0] - w[0]) <= tol:
                    chain = np.vstack([w[::-1][:-1], chain])
                else:
                    continue
                used[j] = True
                members.append(j)
                grew = True
        if len(chain) >= 4 and np.linalg.norm(chain[0] - chain[-1]) <= tol:
            loops.append((layer, chain[:-1]))
        else:
            left += [opens[k] for k in members]
    return loops, left
