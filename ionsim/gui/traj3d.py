"""A rotatable 3D view of recorded ion paths with the electrodes around them.

Electrodes are drawn as shaded blocks taken straight from the solver grid, so it works for every
grid kind: 3D grids, axisymmetric grids (turned about the axis) and cross-section grids
(stretched along the ion axis). The grid is coarsened for drawing (a thin electrode is kept,
never dropped), and only the electrodes near the ions are shown. The front half (x < 0) is cut
away so the ions inside stay in sight. The beam axis z runs left to right, y is up.
Drag to turn the view, right-drag to zoom.
"""
import numpy as np
from matplotlib.ticker import MaxNLocator

MM = 1e-3
MAX_PATHS = 300
CELLS_ACROSS, CELLS_ALONG = 48, 240      # drawing resolution
LIGHT = {0: 0.80, 1: 1.0, 2: 0.65}       # brightness of faces facing along z, x, y (in plot order)


def _pool(a, f):
    """Block maximum of an integer array by whole factors f per axis (pads with 0)."""
    f = [max(int(k), 1) for k in f]
    pad = [(0, (-n) % k) for n, k in zip(a.shape, f)]
    a = np.pad(a, pad)
    shp = []
    for n, k in zip(a.shape, f):
        shp += [n // k, k]
    return a.reshape(shp).max(axis=tuple(range(1, 2 * a.ndim, 2)))


def voxels(grid, z_range=None):
    """Coarse electrode ids on a box grid: (ids (nx, ny, nz), x edges, y edges, z edges) in metres."""
    el = np.asarray(grid.electrode)
    if el.ndim == 3:                                    # Grid3D
        f = np.ceil(np.array(el.shape) / (CELLS_ACROSS, CELLS_ACROSS, CELLS_ALONG))
        ids = _pool(el, f)
        edges = [grid.origin[a] + grid.h * (np.arange(ids.shape[a] + 1) * f[a] - 0.5) for a in range(3)]
        return ids, *edges
    if hasattr(grid, "r"):                              # axisymmetric (r, z): turn about the axis
        fr = np.ceil(el.shape[0] / (CELLS_ACROSS / 2))
        fz = np.ceil(el.shape[1] / CELLS_ALONG)
        rz = _pool(el, (fr, fz))
        dr = grid.h * fr
        n = rz.shape[0]
        xe = dr * np.arange(-n, n + 1)
        xc = 0.5 * (xe[1:] + xe[:-1])
        ir = np.floor(np.hypot(*np.meshgrid(xc, xc, indexing="ij")) / dr).astype(int)
        ok = ir < n
        ids = np.where(ok[:, :, None], rz[np.minimum(ir, n - 1)], 0)
        ze = grid.z[0] + grid.h * (np.arange(rz.shape[1] + 1) * fz - 0.5)
        return ids, xe, xe.copy(), ze
    h = getattr(grid, "dx", getattr(grid, "h", None))   # cross-section (x, y), uniform along z
    f = np.ceil(np.array(el.shape) / CELLS_ACROSS)
    xy = _pool(el, f)
    xe = grid.x[0] + h * (np.arange(xy.shape[0] + 1) * f[0] - 0.5)
    ye = grid.y[0] + h * (np.arange(xy.shape[1] + 1) * f[1] - 0.5)
    z0, z1 = z_range if z_range is not None else (0.0, 0.1)
    ze = np.linspace(z0, z1, 25)
    return np.repeat(xy[:, :, None], len(ze) - 1, axis=2), xe, ye, ze


def faces(ids, xe, ye, ze):
    """Quads on every voxel face between an electrode and free space: (verts (n, 4, 3) as plot
    coordinates z, x, y in mm; electrode id per quad; axis of the face normal in plot order)."""
    on = ids > 0
    E = (xe / MM, ye / MM, ze / MM)
    verts, fid, fax = [], [], []
    for a in range(3):
        b, c = [k for k in range(3) if k != a]
        for s in (1, -1):
            nb = np.zeros_like(on)
            src = [slice(None)] * 3
            dst = [slice(None)] * 3
            if s == 1:
                src[a], dst[a] = slice(1, None), slice(None, -1)
            else:
                src[a], dst[a] = slice(None, -1), slice(1, None)
            nb[tuple(dst)] = on[tuple(src)]
            idx = np.argwhere(on & ~nb)
            if not len(idx):
                continue
            pa = E[a][idx[:, a] + (1 if s == 1 else 0)]
            b0, b1 = E[b][idx[:, b]], E[b][idx[:, b] + 1]
            c0, c1 = E[c][idx[:, c]], E[c][idx[:, c] + 1]
            q = np.zeros((len(idx), 4, 3))
            q[:, :, a] = pa[:, None]
            q[:, :, b] = np.column_stack([b0, b1, b1, b0])
            q[:, :, c] = np.column_stack([c0, c0, c1, c1])
            verts.append(q[:, :, [2, 0, 1]])            # (x, y, z) -> plot order (z, x, y)
            fid.append(ids[tuple(idx.T)])
            fax.append(np.full(len(idx), {2: 0, 0: 1, 1: 2}[a]))
    if not verts:
        return np.zeros((0, 4, 3)), np.zeros(0, int), np.zeros(0, int)
    return np.concatenate(verts), np.concatenate(fid), np.concatenate(fax)


def draw_traj3d(fig, traj, styles, grid=None, title="", cut=True, z_end=None):
    """traj: (n_samples, n_paths, 3) in metres; styles: (colour, linestyle) per path."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig.clear()
    ax = fig.add_subplot(111, projection="3d")
    n = traj.shape[1]
    pick = np.arange(n) if n <= MAX_PATHS else np.linspace(0, n - 1, MAX_PATHS).astype(int)
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for i in pick:
        p = traj[:, i]
        p = p[np.all(np.isfinite(p), axis=1)]
        if not len(p):
            continue
        c, ls = styles[i]
        ax.plot(p[:, 2] / MM, p[:, 0] / MM, p[:, 1] / MM, lw=0.7, color=c, ls=ls, zorder=5)
        lo, hi = np.minimum(lo, p.min(0)), np.maximum(hi, p.max(0))
    if grid is not None:
        ids, xe, ye, ze = voxels(grid, (lo[2], hi[2]) if np.isfinite(lo[2]) else None)
        cx, cy, cz = (0.5 * (e[1:] + e[:-1]) for e in (xe, ye, ze))
        keep = np.ones(ids.shape, bool)
        if np.all(np.isfinite(lo)):                     # the electrodes near the ions, not the whole box
            pad = np.full(3, 0.8 * max(hi[0] - lo[0], hi[1] - lo[1]) + 4e-3)
            box_lo, box_hi = lo - pad, hi + pad
            box_lo[2], box_hi[2] = lo[2] - 0.03 * (hi[2] - lo[2]), hi[2] + 0.03 * (hi[2] - lo[2])
            keep &= ((cx >= box_lo[0]) & (cx <= box_hi[0]))[:, None, None]
            keep &= ((cy >= box_lo[1]) & (cy <= box_hi[1]))[None, :, None]
            keep &= ((cz >= box_lo[2]) & (cz <= box_hi[2]))[None, None, :]
        if cut:
            keep &= (cx >= 0)[:, None, None]
        ids = np.where(keep, ids, 0)
        verts, fid, fax = faces(ids, xe, ye, ze)
        if len(verts):
            base = 0.55 + 0.25 * ((fid * 0.37) % 1.0)
            light = np.array([LIGHT[a] for a in range(3)])[fax]
            col = np.clip(base * light, 0, 1)
            rgba = np.column_stack([col * 0.95, col * 0.93, col * 0.88, np.full(len(col), 0.55)])
            ax.add_collection3d(Poly3DCollection(verts, facecolors=rgba, edgecolors="none", linewidths=0))
            on = np.argwhere(ids > 0)
            vlo = np.array([xe[on[:, 0]].min(), ye[on[:, 1]].min(), ze[on[:, 2]].min()])
            vhi = np.array([xe[on[:, 0] + 1].max(), ye[on[:, 1] + 1].max(), ze[on[:, 2] + 1].max()])
            lo, hi = np.minimum(lo, vlo), np.maximum(hi, vhi)
    if z_end is not None and np.all(np.isfinite(lo)):
        ax.plot([z_end / MM] * 2, [lo[0] / MM, hi[0] / MM], [0, 0], "k:", lw=1)
    if np.all(np.isfinite(lo)):
        span = np.maximum(hi - lo, 1e-6)
        lz = span[2]
        ax.set_box_aspect((1.0, max(span[0] / lz, 0.25), max(span[1] / lz, 0.25)))
        ax.set_xlim(lo[2] / MM, hi[2] / MM); ax.set_ylim(lo[0] / MM, hi[0] / MM); ax.set_zlim(lo[1] / MM, hi[1] / MM)
    ax.set_xlabel("z (mm)"); ax.set_ylabel("x (mm)"); ax.set_zlabel("y (mm)")
    ax.yaxis.set_major_locator(MaxNLocator(3)); ax.zaxis.set_major_locator(MaxNLocator(4))
    ax.view_init(elev=22, azim=-62)
    note = "drag to turn, right-drag to zoom" + ("; front half (x < 0) of the electrodes cut away" if cut and grid is not None else "")
    ax.set_title((title + "\n" if title else "") + note, fontsize=9)
    return ax
