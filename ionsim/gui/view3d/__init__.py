"""Interactive 3D viewer: lit electrodes and ion paths, shown inside the app or saved as one web
page that works offline (three.js is bundled in viewer.js, MIT licence in THREE_LICENSE.txt).

Build a Scene in mm with y up, then show it in a View3D widget or write it with Scene.write():

    sc = Scene(title="Quadrupole")
    sc.mesh(triangles_mm, color="#c8773a", group="Rods", clip=True)
    sc.paths("transmitted", [path_mm, ...], color="#1f77b4")
    View3D().show_scene(sc)        # or sc.write("quad 3d.html")

Paths may carry time as a 4th column first, (t, x, y, z): the viewer can then colour them by time.
scene_from_traj() turns the recorded paths and solver grid of any tab into a scene.
"""
import base64
import json
import os
import tempfile
import webbrowser

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MM = 1e-3


def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a, np.float32).tobytes()).decode()


def _hex(c):
    from matplotlib.colors import to_hex
    return to_hex(c)


def voltage_color(v, vmax=10000.0):
    """Blue (negative) through light grey to red (positive), square-root scaled to vmax."""
    neg, mid, pos = np.array([0.12, 0.36, 0.85]), np.array([0.85, 0.83, 0.78]), np.array([0.85, 0.25, 0.16])
    t = float(np.sign(v) * np.sqrt(min(abs(v) / vmax, 1.0))) if vmax > 0 else 0.0
    c = mid + (neg - mid) * (-t) if t < 0 else mid + (pos - mid) * t
    return _hex(c)


def voltage_colorbar(vmin, vmax, label="electrodes coloured by voltage"):
    """Colour bar matching voltage_color between vmin and vmax (V)."""
    big = max(abs(vmin), abs(vmax), 1.0)
    vs = np.linspace(vmin, vmax, 9)
    stops = [[float((v - vmin) / (vmax - vmin or 1)), voltage_color(v, big)] for v in vs]
    fmt = (lambda v: f"{v / 1000:+.3g} kV") if big >= 1000 else (lambda v: f"{v:+.3g} V")
    return dict(stops=stops, ticks=[fmt(vmin), "0" if vmin < 0 < vmax else fmt((vmin + vmax) / 2), fmt(vmax)], label=label)


class Scene:
    """Everything the viewer draws, in mm with y up."""

    def __init__(self, title="", subtitle="", footer="", note=""):
        self.d = dict(title=title, subtitle=subtitle, footer=footer, note=note, objects=[], paths=[], groups=[],
                      views=[], colorbar=None, clip=None, stretch=None, nPaths=100, pathColor="group",
                      timeColors=["#2fd8ff", "#ff4fd8"])

    # ---- geometry ----
    def _obj(self, kind, group, color, opacity, clip, **kw):
        self._group(group)
        o = dict(type=kind, group=group, color=_hex(color), opacity=float(opacity), clip=bool(clip))
        o.update({k: v for k, v in kw.items() if v is not None})
        self.d["objects"].append(o)
        return o

    def _group(self, name, visible=True, toggle=True):
        for g in self.d["groups"]:
            if g["name"] == name:
                return g
        g = dict(name=name, visible=visible, toggle=toggle)
        self.d["groups"].append(g)
        return g

    def group(self, name, visible=True, toggle=True):
        """Declare a geometry group (one checkbox in the viewer), e.g. hidden at first."""
        g = self._group(name)
        g.update(visible=visible, toggle=toggle)

    def mesh(self, tris, color="#b8bcc4", opacity=1.0, group="Electrodes", clip=False, metal=0.3, rough=0.5, glow=0.0, edges=None):
        """tris: (n, 3, 3) triangle corners in mm."""
        tris = np.asarray(tris, float).reshape(-1, 3, 3)
        if len(tris):
            self._obj("mesh", group, color, opacity, clip, tris=_b64(tris), metal=metal, rough=rough, glow=glow, edges=edges)

    def quads(self, quads, **kw):
        """quads: (n, 4, 3) corners in order, in mm (split into two triangles each)."""
        q = np.asarray(quads, float).reshape(-1, 4, 3)
        self.mesh(np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]]), **kw)

    def box(self, x, y, z, color="#b8bcc4", opacity=1.0, group="Electrodes", clip=False, metal=0.3, rough=0.5, glow=0.0, edges=None):
        """Axis-aligned box: x, y, z are (lo, hi) in mm."""
        self._obj("box", group, color, opacity, clip, x=[float(v) for v in x], y=[float(v) for v in y],
                  z=[float(v) for v in z], metal=metal, rough=rough, glow=glow, edges=edges)

    def shape(self, polys, at, axis="y", color="#c8773a", opacity=1.0, group="Electrodes", clip=False, metal=0.35, rough=0.45, glow=0.0):
        """Flat polygons (with holes) in a plane: axis "y" -> polygons in (x, z) at y = at;
        "z" -> (x, y) at z = at; "x" -> (z, y) at x = at. polys: [{"outer": [[u, v], ...],
        "holes": [[[u, v], ...], ...]}]."""
        self._obj("shape", group, color, opacity, clip, polys=polys, axis=axis, at=float(at), metal=metal, rough=rough, glow=glow)

    def slab(self, outer, at, axis="y", holes=(), color="#5e6b3a", opacity=1.0, group="Boards", clip=False, metal=0.0, rough=0.85, glow=0.0, edges=None):
        """A polygon (with holes) extruded between at = (lo, hi) along axis (see shape)."""
        self._obj("slab", group, color, opacity, clip, outer=[[float(a), float(b)] for a, b in outer],
                  holes=[[[float(a), float(b)] for a, b in h] for h in holes], axis=axis,
                  at=[float(at[0]), float(at[1])], metal=metal, rough=rough, glow=glow, edges=edges)

    # ---- paths ----
    def paths(self, name, paths, color="#2fd8ff", width=1.5, opacity=0.85, legend=True):
        """One group of ion paths: list of (k, 3) arrays (x, y, z) or (k, 4) (t, x, y, z), in mm
        (t in any unit). color "time" colours by time. Groups of the same size are thinned with the
        same ions, so a path split into stages stays joined."""
        dim = 4 if any(np.asarray(p).ndim == 2 and np.asarray(p).shape[1] == 4 for p in paths) else 3
        pts, off = [], [0]
        for p in paths:
            p = np.asarray(p, float).reshape(-1, dim) if len(p) else np.zeros((0, dim))
            p = p[np.all(np.isfinite(p), axis=1)]
            pts.append(p); off.append(off[-1] + len(p))
        arr = np.concatenate(pts) if pts else np.zeros((0, dim))
        self.d["paths"].append(dict(name=name, color=color if color == "time" else _hex(color), width=float(width),
                                    opacity=float(opacity), legend=legend, dim=dim, off=off, pts=_b64(arr)))

    # ---- presentation ----
    def view(self, name, pos, target, fov=30):
        self.d["views"].append(dict(name=name, pos=[float(v) for v in pos], tgt=[float(v) for v in target], fov=fov))

    def set(self, **kw):
        """colorbar=dict(stops, ticks, label), clip=dict(normal, offset, label, on),
        stretch=dict(value, max, label), nPaths, pathColor ("group" or "time"), view (default name)."""
        self.d.update(kw)

    # ---- output ----
    def html(self):
        page = open(os.path.join(HERE, "viewer.html"), encoding="utf-8").read()
        app = open(os.path.join(HERE, "viewer.js"), encoding="utf-8").read()
        data = "window.SCENE=" + json.dumps(self.d).replace("</", "<\\/") + ";"
        title = (self.d.get("title") or "3D view").replace("<", "")
        return page.replace("/*TITLE*/", title).replace("/*SCENE*/", data).replace("/*APP*/", app.replace("</script", "<\\/script"))

    def write(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.html())
        return path


# ---------------------------------------------------------------- building blocks
def revolve(profile, n=48, axis="z"):
    """Triangles of a solid of revolution about the z axis (or "x"): profile is a closed polygon
    [(r, z), ...] in mm, r >= 0 (e.g. a tube wall: inner and outer radius between two z)."""
    pr = np.asarray(profile, float)
    pr = np.vstack([pr, pr[:1]]) if np.any(pr[0] != pr[-1]) else pr
    ph = np.linspace(0, 2 * np.pi, n + 1)
    c, s_ = np.cos(ph), np.sin(ph)
    tris = []
    for (r0, z0), (r1, z1) in zip(pr[:-1], pr[1:]):
        a = np.stack([r0 * c[:-1], r0 * s_[:-1], np.full(n, z0)], 1)
        b = np.stack([r0 * c[1:], r0 * s_[1:], np.full(n, z0)], 1)
        cc = np.stack([r1 * c[1:], r1 * s_[1:], np.full(n, z1)], 1)
        d = np.stack([r1 * c[:-1], r1 * s_[:-1], np.full(n, z1)], 1)
        tris += [np.stack([a, b, cc], 1), np.stack([a, cc, d], 1)]
    t = np.concatenate(tris) if tris else np.zeros((0, 3, 3))
    return t[:, :, [2, 1, 0]] if axis == "x" else t


def box_tris(x, y, z):
    """The 12 triangles of an axis-aligned box (x, y, z are (lo, hi)), for merging many boxes into one mesh."""
    (x0, x1), (y0, y1), (z0, z1) = x, y, z
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    f = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
         (3, 7, 6), (3, 6, 2), (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5)]
    return v[np.array(f)]


def fit_view(lo, hi, direction, aspect=1.6, fov=30.0, margin=1.08):
    """Camera (position, target) looking at the box lo..hi (mm) from `direction`, framed like the
    viewer's own views (z right, y up)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c = 0.5 * (lo + hi)
    d = np.asarray(direction, float); d /= np.linalg.norm(d)
    F = -d
    R = np.cross(F, [0.0, 1.0, 0.0])
    R = R / np.linalg.norm(R) if np.linalg.norm(R) > 1e-9 else np.array([0.0, 0.0, 1.0])
    U = np.cross(R, F)
    tv = np.tan(np.radians(fov / 2)); th = tv * aspect
    dist = 1e-6
    for x in (lo[0], hi[0]):
        for y in (lo[1], hi[1]):
            for z in (lo[2], hi[2]):
                p = np.array([x, y, z]) - c
                dist = max(dist, abs(p @ R) / th - p @ F, abs(p @ U) / tv - p @ F)
    return c + d * dist * margin, c


def _unb64(s, dim):
    return np.frombuffer(base64.b64decode(s), np.float32).reshape(-1, dim).astype(float)


def draw_simple(fig, sc, max_paths=40):
    """A matplotlib picture of a Scene (paths and outlines), for when the viewer can't be shown.
    Drawn like traj3d: z across, x into the page, y up."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig.clear()
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)

    def grow(p):
        nonlocal lo, hi
        if len(p):
            lo, hi = np.minimum(lo, p.min(0)), np.maximum(hi, p.max(0))

    for o in sc.d["objects"]:
        g = next((g for g in sc.d["groups"] if g["name"] == o["group"]), {})
        if not g.get("visible", True):
            continue
        if o["type"] == "box":
            t = box_tris(o["x"], o["y"], o["z"])
        elif o["type"] == "mesh":
            t = _unb64(o["tris"], 9).reshape(-1, 3, 3)
            if len(t) > 6000:
                t = t[:: int(np.ceil(len(t) / 6000))]
        else:
            continue
        grow(t.reshape(-1, 3))
        ax.add_collection3d(Poly3DCollection(t[:, :, [2, 0, 1]], facecolors=o["color"], edgecolors="none",
                                             alpha=min(0.5, o["opacity"]), zorder=1))
    for pg in sc.d["paths"]:
        arr = _unb64(pg["pts"], pg["dim"])[:, -3:]
        off = pg["off"]
        step = max(1, (len(off) - 1) // max_paths)
        for k in range(0, len(off) - 1, step):
            p = arr[off[k]:off[k + 1]]
            if len(p) > 1:
                ax.plot(p[:, 2], p[:, 0], p[:, 1], lw=0.7, color=pg["color"] if pg["color"] != "time" else "tab:blue",
                        zorder=5)
                grow(p)
    if np.all(np.isfinite(lo)):
        span = np.maximum(hi - lo, 1e-6)
        ax.set_box_aspect((1.0, max(span[0] / span[2], 0.15), max(span[1] / span[2], 0.15)))
        ax.set_xlim(lo[2], hi[2]); ax.set_ylim(lo[0], hi[0]); ax.set_zlim(lo[1], hi[1])
    ax.set_xlabel("z (mm)"); ax.set_ylabel("x (mm)"); ax.set_zlabel("y (mm)")
    ax.view_init(elev=22, azim=-62)
    if sc.d.get("title"):
        ax.set_title(sc.d["title"] + (f"\n{sc.d['subtitle']}" if sc.d.get("subtitle") else ""), fontsize=9)
    fig.scene3d = sc
    return ax


def _faces_xyz(ids, xe, ye, ze):
    from ..traj3d import faces
    verts, fid, _ = faces(ids, xe, ye, ze)            # plot order (z, x, y), mm
    return verts[:, :, [1, 2, 0]], fid


def scene_from_traj(traj, styles, grid=None, title="", subtitle="", z_end=None, labels=None):
    """The same picture as traj3d.draw_traj3d, for the viewer. traj: (n_samples, n_paths, 3) in m;
    styles: (colour, linestyle) per path; grid: the solver grid (any kind). labels: {colour: name}
    for the path groups (default: the colour). Electrodes near the ions are drawn, and the front
    half (x < 0) can be cut away in the viewer."""
    from ..traj3d import voxels, MAX_PATHS
    sc = Scene(title=title, subtitle=subtitle)
    traj = np.asarray(traj, float)
    n = traj.shape[1] if traj.ndim == 3 else 0
    pick = np.arange(n) if n <= MAX_PATHS * 2 else np.linspace(0, n - 1, MAX_PATHS * 2).astype(int)
    groups = {}
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for i in pick:
        p = traj[:, i]
        p = p[np.all(np.isfinite(p), axis=1)]
        if len(p) < 2:
            continue
        c, ls = styles[i]
        key = (_hex(c), ls)
        groups.setdefault(key, []).append(p / MM)
        lo, hi = np.minimum(lo, p.min(0)), np.maximum(hi, p.max(0))
    names = {"-": "", ":": " (lost)", "--": " (still flying)"}
    if labels is None:                                  # the colours of ion_table.path_style
        labels = {_hex("tab:blue"): "transmitted", _hex("tab:red"): "lost", _hex("tab:orange"): "still flying"}
        try:
            from ..ion_table import species_color
            for k in range(12):
                labels.setdefault(_hex(species_color(k)), f"species {k + 1}")
        except Exception:                               # noqa: BLE001
            pass
    for (c, ls), ps in groups.items():
        base = labels.get(c, "ions")
        sc.paths(base + names.get(ls, ""), ps, color=c, width=1.4 if ls == "-" else 1.0,
                 opacity=0.9 if ls == "-" else 0.6)
    if grid is not None:
        ids, xe, ye, ze = voxels(grid, (lo[2], hi[2]) if np.isfinite(lo[2]) else None)
        cx, cy, cz = (0.5 * (e[1:] + e[:-1]) for e in (xe, ye, ze))
        keep = np.ones(ids.shape, bool)
        if np.all(np.isfinite(lo)):
            pad = np.full(3, 0.8 * max(hi[0] - lo[0], hi[1] - lo[1]) + 4e-3)
            blo, bhi = lo - pad, hi + pad
            blo[2], bhi[2] = lo[2] - 0.03 * (hi[2] - lo[2]), hi[2] + 0.03 * (hi[2] - lo[2])
            keep &= ((cx >= blo[0]) & (cx <= bhi[0]))[:, None, None]
            keep &= ((cy >= blo[1]) & (cy <= bhi[1]))[None, :, None]
            keep &= ((cz >= blo[2]) & (cz <= bhi[2]))[None, None, :]
        ids = np.where(keep, ids, 0)
        verts, fid = _faces_xyz(ids, xe, ye, ze)
        for k in np.unique(fid):
            shade = 0.70 + 0.22 * ((k * 0.37) % 1.0)
            col = (shade * 0.98, shade * 0.93, shade * 0.86)
            sc.quads(verts[fid == k], color=col, opacity=0.9, group="Electrodes", clip=True, metal=0.25, rough=0.45)
        sc.set(clip=dict(normal=[1, 0, 0], offset=0.0, label="cut away the front half (x < 0)", on=True))
    if z_end is not None and np.all(np.isfinite(lo)):
        w = max(hi[0] - lo[0], 1e-3) / MM
        sc.box((lo[0] / MM - 0.2 * w, hi[0] / MM + 0.2 * w), (-0.01 * w, 0.01 * w), (z_end / MM - 0.05, z_end / MM + 0.05),
               color="#ffffff", opacity=0.25, group="Exit plane")
    return sc


# ---------------------------------------------------------------- Qt widget
def webengine_available():
    try:
        from PySide6 import QtWebEngineWidgets  # noqa: F401
        return True
    except Exception:          # noqa: BLE001 - missing module or missing Qt WebEngine libraries
        return False


def make_widget(parent=None):
    """The viewer widget for a plot tab, or None if Qt WebEngine is not installed."""
    if not webengine_available():
        return None
    from PySide6 import QtCore, QtWidgets
    from PySide6.QtWebEngineWidgets import QWebEngineView

    class View3D(QtWidgets.QWidget):
        """A web view showing a Scene, with Save and Open in browser buttons."""

        def __init__(self, parent=None):
            super().__init__(parent)
            v = QtWidgets.QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0)
            row = QtWidgets.QHBoxLayout()
            self.save_btn = QtWidgets.QPushButton("Save as web page…")
            self.open_btn = QtWidgets.QPushButton("Open in browser")
            for b in (self.save_btn, self.open_btn):
                b.setEnabled(False); row.addWidget(b)
            row.addStretch(1)
            self.note = QtWidgets.QLabel("Drag to turn, right-drag to move, scroll to zoom. The page saves "
                                         "with everything in it and opens in any browser, offline.")
            self.note.setStyleSheet("color: #666")
            row.addWidget(self.note)
            v.addLayout(row)
            self.web = QWebEngineView()
            v.addWidget(self.web, 1)
            self.scene = None
            self._file = None
            self.save_btn.clicked.connect(self._save)
            self.open_btn.clicked.connect(self._open)

        def show_scene(self, scene):
            self.scene = scene
            fd, path = tempfile.mkstemp(suffix=".html", prefix="ionsim3d_")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(scene.html())
            if self._file:
                try:
                    os.remove(self._file)
                except OSError:
                    pass
            self._file = path
            self.web.load(QtCore.QUrl.fromLocalFile(path))
            self.save_btn.setEnabled(True); self.open_btn.setEnabled(True)

        def _save(self):
            if not self.scene:
                return
            name = (self.scene.d.get("title") or "3D view").split(":")[0].strip() + " 3D.html"
            fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save the 3D view", name, "Web page (*.html)")
            if fn:
                self.scene.write(fn)

        def _open(self):
            if self.scene:
                fd, path = tempfile.mkstemp(suffix=".html", prefix="ionsim3d_")
                os.close(fd)
                self.scene.write(path)
                webbrowser.open("file://" + path)

    return View3D(parent)
