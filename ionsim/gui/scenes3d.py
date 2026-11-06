"""3D scenes for the models whose own plots are flat: OA-TOF, FAIMS, TWAVE and the ESI inlet.

Each function returns a view3d.Scene in mm with y up. The models themselves are 1D or 2D (the
TWAVE and inlet models follow ions along their axis, FAIMS across the gap), so the scene puts the
ions where the model has them and says so in the subtitle.
"""
import numpy as np

from .view3d import Scene, voltage_color, voltage_colorbar, box_tris, revolve, fit_view
from .ion_table import species_color

MM = 1e-3


def _views(sc, lo, hi, extra=()):
    """The usual four views, fitted to the box lo..hi (mm), plus extra (name, direction, lo, hi)."""
    for name, d in (("3D", (-0.75, 0.55, -0.45)), ("top", (-1e-3, 1, 0)), ("side", (-1, 0.02, 0)),
                    ("end", (0, 0.02, -1))):
        pos, tgt = fit_view(lo, hi, d)
        sc.view(name, pos, tgt)
    for name, d, a, b in extra:
        pos, tgt = fit_view(a, b, d)
        sc.view(name, pos, tgt)


def _frame(x, y, z, w, axis="z"):
    """Triangles of a square ring (frame) of bar width w in the plane normal to `axis`."""
    (x0, x1), (y0, y1), (z0, z1) = x, y, z
    if axis == "z":
        bars = [((x0, x1), (y0, y0 + w)), ((x0, x1), (y1 - w, y1)), ((x0, x0 + w), (y0 + w, y1 - w)),
                ((x1 - w, x1), (y0 + w, y1 - w))]
        return np.concatenate([box_tris(a, b, (z0, z1)) for a, b in bars])
    raise ValueError(axis)


# ------------------------------------------------------------------ OA-TOF
def tof_scene(tof, beam, res, plane_names=None):
    """Pusher, grids, reflectron rings and detector with the recorded ion paths. The flight axis
    (x in the model) runs left to right, the beam direction (z) runs into the picture."""
    from .. import tof as T
    pl = tof.planes()
    t_on = tof.pulse.delay + 0.5 * tof.pulse.width
    volts = np.asarray(tof.potentials(t_on), float)
    vmax = max(float(np.max(np.abs(volts))), 1.0)
    dz = res.det_z if res is not None else (tof.det_z or 0.0)
    zs = [-beam.window / 2, beam.window / 2, dz - tof.det_width_z / 2, dz + tof.det_width_z / 2]
    zc = 0.5 * (min(zs) + max(zs))
    half_b = 0.5 * (max(zs) - min(zs)) / MM + 12           # plate half width along the beam (mm)
    half_y = tof.det_width_y / 2 / MM + 12
    n_ions = 0 if res is None else len(res.traj_ids)
    sc = Scene(title="OA-TOF" + (f": {n_ions} ion paths" if n_ions else ""),
               subtitle="flight axis left to right, beam into the picture; plates coloured by voltage during the push")
    X = lambda x: x / MM                                     # model x -> viewer z
    for k, (x, kind, name) in enumerate(pl):
        v = volts[k] if k < len(volts) else 0.0
        col = voltage_color(v, vmax)
        if kind == "det":
            sc.box(((dz - zc - tof.det_width_z / 2) / MM, (dz - zc + tof.det_width_z / 2) / MM),
                   (-tof.det_width_y / 2 / MM, tof.det_width_y / 2 / MM), (X(x) - 3, X(x)),
                   color="#2a2d33", group="Detector", metal=0.6, rough=0.35)
            continue
        if kind == "wall":
            th = 1.0 if x <= 0 else -1.0
            sc.box((-half_b, half_b), (-half_y, half_y), sorted((X(x), X(x) - th)), color=col, group="Plates",
                   metal=0.45, rough=0.4)
        else:
            sc.box((-half_b, half_b), (-half_y, half_y), (X(x) - 0.15, X(x) + 0.15), color=col, opacity=0.28,
                   group="Grids", metal=0.6, rough=0.3, edges="#ffffff")
    # reflectron rings between the entrance grid and the back plate, voltage graded along each stage
    if tof.reflectron != "none":
        xs = [p[0] for p in pl if p[2].startswith("reflectron")]
        vs = list(volts[-len(xs):])
        x_e, x_b = xs[0], xs[-1]
        n = int(np.clip((x_b - x_e) / 6e-3, 4, 40))
        for k in range(1, n):
            xr = x_e + (x_b - x_e) * k / n
            vr = float(np.interp(xr, xs, vs))
            sc.mesh(_frame((-half_b, half_b), (-half_y, half_y), (X(xr) - 0.5, X(xr) + 0.5), 3.0),
                    color=voltage_color(vr, vmax), opacity=0.85, group="Reflectron rings", metal=0.5, rough=0.4)
    # drift tube walls, faint
    x_d0 = pl[2][0]
    x_d1 = [p[0] for p in pl if p[2] == "reflectron entrance"][0] if tof.reflectron != "none" else pl[-1][0]
    for yy in (-half_y, half_y):
        sc.box((-half_b, half_b), (yy - 0.3, yy + 0.3), (X(x_d0), X(x_d1)), color=voltage_color(tof.v_drift, vmax),
               opacity=0.06, group="Drift tube")
    # the beam slice in the pusher
    x0 = beam.x0 if beam.x0 is not None else tof.push_gap / 2
    sc.box(((-beam.window / 2 - zc) / MM, (beam.window / 2 - zc) / MM), (-2 * beam.sigma_y / MM, 2 * beam.sigma_y / MM),
           ((x0 - 2 * beam.sigma_x) / MM, (x0 + 2 * beam.sigma_x) / MM), color="#ff9f1c", opacity=0.35, group="Beam",
           glow=0.4)
    # ion paths
    if res is not None and n_ions:
        groups = {}
        multi = len(res.names) > 1
        for k, (t, x, z) in enumerate(res.paths()):
            i = res.traj_ids[k]
            st = int(res.status[i])
            t_stop = res.t[i] if st == T.ST_DETECTED and np.isfinite(res.t[i]) else res.t_end[i]
            vy = (res.y_hit[i] - res.y0[i]) / t_stop if np.isfinite(t_stop) and t_stop > 0 else 0.0
            y = res.y0[i] + vy * t
            p = np.column_stack([t * 1e6, (z - zc) / MM, y / MM, x / MM])
            if multi:
                key = (res.names[int(res.species[i])], species_color(int(res.species[i])))
            else:
                key = ("detected", "tab:blue") if st == T.ST_DETECTED else (T.STATUS_NAMES.get(st, "lost"), "tab:red")
            groups.setdefault(key, []).append(p)
        for (name, col), ps in groups.items():
            sc.paths(name, ps, color=col, width=1.4)
    sc.set(colorbar=voltage_colorbar(min(volts.min(), 0), max(volts.max(), 0), "plates coloured by voltage (push on)"),
           nPaths=60)
    x_end = pl[-1][0]
    lo = np.array([-half_b, -half_y, -2.0]); hi = np.array([half_b, half_y, X(x_end) + 2])
    src_hi = np.array([half_b, half_y, X(pl[2][0]) + 5])
    det_x = X([p[0] for p in pl if p[1] == "det"][0])
    _views(sc, lo, hi, [("source", (-0.8, 0.45, -0.4), lo, src_hi),
                        ("detector", (-0.8, 0.45, 0.4), np.array([-half_b, -half_y, det_x - 30]),
                         np.array([half_b, half_y, det_x + 10]))])
    sc.set(view="3D")
    return sc


# ------------------------------------------------------------------ FAIMS
def faims_scene(f, r, species_name, cv, plate_width=None):
    """The two plates with the recorded ion paths (in the midplane: the model follows ions along
    the cell and across the gap)."""
    L, g = f.length / MM, f.gap / MM
    w = plate_width or max(10.0, 4 * g)
    t_pk = 0.0
    vmax = max(abs(f.dv), 1.0)
    sc = Scene(title=f"FAIMS: {species_name} at CV {cv:.2f} V",
               subtitle="each path is the ion's drift, averaged over one waveform period, drawn in the middle of the plates "
                        "(the model follows ions along the cell and across the gap)")
    sc.box((-w / 2, w / 2), (-g / 2 - 1.0, -g / 2), (0, L), color=voltage_color(0.0, vmax), group="Lower plate (ground)",
           metal=0.5, rough=0.35)
    sc.box((-w / 2, w / 2), (g / 2, g / 2 + 1.0), (0, L), color=voltage_color(f.dv, vmax), opacity=0.35,
           group="Upper plate (dispersion voltage)", metal=0.5, rough=0.35, edges="#ffffff")
    xy = r.traj["xy"] if r.traj else np.zeros((0, 0, 2))
    tt = r.traj.get("t", np.zeros(0)) if r.traj else np.zeros(0)
    nper = int(round(1 / f.freq / np.median(np.diff(tt)))) if len(tt) > 2 else 1
    tr, lost = [], []
    for k in range(xy.shape[1]):
        p = xy[:, k]
        p = p[np.all(np.isfinite(p), axis=1)]
        if len(p) < 2:
            continue
        if nper > 1 and len(p) > 3 * nper:                 # the drift: averaged over one waveform period
            ker = np.ones(nper) / nper
            p = np.column_stack([np.convolve(p[:, c], ker, mode="valid") for c in range(2)])
        if len(p) > 2000:
            p = p[np.unique(np.linspace(0, len(p) - 1, 2000).astype(int))]
        pts = np.column_stack([np.zeros(len(p)), p[:, 1] / MM, p[:, 0] / MM])
        (tr if p[-1, 0] >= 0.98 * f.length else lost).append(pts)
    if tr:
        sc.paths("through the cell", tr, color="tab:blue", width=1.3)
    if lost:
        sc.paths("hit a plate", lost, color="tab:red", width=1.1)
    # gas flow arrows at the entrance
    sc.paths("gas flow", [np.array([[0, yy, -6.0], [0, yy, -1.0]]) for yy in np.linspace(-g / 3, g / 3, 3)],
             color="#9aa4b2", width=2.0)
    sc.set(colorbar=voltage_colorbar(min(0, f.dv), max(0, f.dv), "plates coloured by peak voltage"), nPaths=40,
           stretch=dict(value=1, max=20, label="Gap stretch"))
    lo, hi = np.array([-w / 2, -g / 2 - 1, -6]), np.array([w / 2, g / 2 + 1, L])
    _views(sc, lo, hi, [("entrance", (-0.6, 0.3, -0.75), lo, np.array([w / 2, g / 2 + 1, min(L, 6 * g)]))])
    sc.set(view="3D")
    return sc


# ------------------------------------------------------------------ TWAVE
def twave_scene(tw, t_snap, z_snap, species_of, names, path, t_board=0.0, board_width=None, max_strips=4000):
    """The two boards with their electrodes (on/off at time t_board) and the ions' positions along
    the path at a few moments: each ion is a short upright dash at its place along the boards.
    z_snap: (n_snap, n_ions) positions (m, NaN once arrived); species_of: per ion."""
    g = tw.gap / MM
    bw = board_width or max(6.0, 2.5 * g)
    L = path / MM
    pitch, ew = tw.pitch / MM, tw.width / MM
    n_strips = int(np.ceil(L / pitch)) + 1
    sc = Scene(title=f"TWAVE: {', '.join(names)} along {path:g} m",
               subtitle="each tick is an ion where it is at one moment, one lane across the boards per moment "
                        "(the model follows ions along the boards)")
    sc.box((-bw / 2, bw / 2), (-g / 2 - 1.6, -g / 2 - 0.05), (0, L), color="#4d5a33", group="Boards", metal=0.0, rough=0.85)
    sc.group("Upper board", visible=False)
    sc.box((-bw / 2, bw / 2), (g / 2 + 0.05, g / 2 + 1.6), (0, L), color="#4d5a33", opacity=0.3, group="Upper board",
           metal=0.0, rough=0.85)
    if n_strips <= max_strips:
        z0 = np.arange(n_strips) * pitch
        on = np.asarray(tw.board_potential(z0 * MM - tw.speed * t_board) if hasattr(tw, "board_potential") else 0) > 0.5
        for state, col in ((True, "#d9893b"), (False, "#b7aa8e")):          # copper: on (amplitude) and off
            zz = z0[on == state]
            for grp, (y0, y1) in (("Boards", (-g / 2 - 0.05, -g / 2)), ("Upper board", (g / 2, g / 2 + 0.05))):
                tris = [box_tris((-bw / 2 + 0.3, bw / 2 - 0.3), (y0, y1), (z - ew / 2, z + ew / 2)) for z in zz]
                if tris:
                    sc.mesh(np.concatenate(tris), color=col, group=grp, metal=0.5, rough=0.35)
    else:
        sc.d["note"] = f"{n_strips} electrodes per board: drawn as plain boards"
    n_snap = len(t_snap)
    lanes = (np.arange(n_snap) + 0.5) / n_snap * (bw - 0.6) - (bw - 0.6) / 2   # one lane per moment, front to back
    lane_w = 0.7 * (bw - 0.6) / n_snap
    for s, name in enumerate(names):
        dashes = []
        for j in range(n_snap):
            zz = z_snap[j][(species_of == s) & np.isfinite(z_snap[j])] / MM
            rng = np.random.default_rng(j * 31 + s)
            yy = (rng.random(len(zz)) - 0.5) * 0.7 * g
            dashes += [np.array([[lanes[j] - lane_w / 2, y, z], [lanes[j] + lane_w / 2, y, z]]) for y, z in zip(yy, zz)]
        if dashes:
            sc.paths(name, dashes, color=species_color(s), width=2.0)
    sc.d["note"] = (sc.d.get("note", "") + "; " if sc.d.get("note") else "") + \
        "lanes front to back: " + ", ".join(f"{t * 1e3:.2f}" for t in t_snap) + " ms"
    sc.set(nPaths=100000, stretch=dict(value=1, max=20, label="Gap stretch"))
    lo, hi = np.array([-bw / 2, -g / 2 - 1.6, 0]), np.array([bw / 2, g / 2 + 1.6, L])
    zoom = min(L, 40 * tw.wavelength / MM)
    extra = [("start", (-0.7, 0.6, -0.4), lo, np.array([bw / 2, g / 2 + 1.6, zoom]))]
    last = z_snap[-1][np.isfinite(z_snap[-1])] / MM if n_snap else np.zeros(0)
    if len(last):                                  # the packets at the last moment, close up
        a, b = np.percentile(last, [1, 99])
        pad = max(0.15 * (b - a), 3 * tw.wavelength / MM)
        extra.append(("packets", (-0.7, 0.6, -0.4), np.array([-bw / 2, -g / 2 - 1.6, max(0, a - pad)]),
                      np.array([bw / 2, g / 2 + 1.6, min(L, b + pad)])))
    _views(sc, lo, hi, extra)
    sc.set(view="packets" if len(last) else "3D")
    return sc


# ------------------------------------------------------------------ ESI inlet
def inlet_scene(inl, res):
    """Emitter, inlet capillary (cut open), the free jet with its Mach disk, and the skimmer, with
    each ion's path along the axis coloured by time."""
    gap, L = inl.spray_gap / MM, inl.length / MM
    r_in = inl.diameter / 2 / MM
    r_out = max(0.8, 2 * r_in)                                 # 1/16 inch tubing at least
    x_md, skim = inl.x_mach_disk / MM, inl.skim_dist / MM
    sc = Scene(title="ESI inlet: spray to skimmer",
               subtitle="ions follow the axis in this model; their paths are coloured by time")
    # emitter needle: a thin tube ending in a cone at -gap
    tip = max(inl.tip_radius / MM, 0.02)
    sc.mesh(revolve([(0, -gap), (0.08 + tip, -gap - 0.6), (0.18, -gap - 2.0), (0.18, -gap - 12), (0, -gap - 12)], 32),
            color="#c9ccd2", group="Emitter", metal=0.7, rough=0.3)
    # capillary (front half cut away)
    sc.mesh(revolve([(r_in, 0), (r_out, 0), (r_out, L), (r_in, L)], 64), color="#b9bec7", group="Capillary",
            clip=True, metal=0.6, rough=0.35)
    # free jet: barrel shock to the Mach disk, translucent
    R = max(0.38 * x_md, 1.2 * r_in)                          # barrel shock about 0.75 x_md across, Mach disk half that
    fs = np.array([0, .15, .3, .45, .6, .75, .9, 1.0])
    fr = np.array([0, .55, .82, .96, 1, .95, .8, .66])
    prof = [(0, L)] + [(r_in + (R - r_in) * b, L + a * x_md) for a, b in zip(fs, fr)] + [(0, L + x_md)]
    sc.mesh(revolve(prof, 48), color="#7fc8ff", opacity=0.18, group="Free jet (to the Mach disk)", clip=True, glow=0.3)
    # skimmer cone
    sk_r = max(0.3, r_in)
    sc.mesh(revolve([(sk_r, L + skim), (sk_r + 3.5, L + skim + 4), (sk_r + 3.8, L + skim + 4), (sk_r + 0.25, L + skim)], 64),
            color="#9fa6b2", group="Skimmer", clip=True, metal=0.6, rough=0.35)
    for d in res:
        x = np.asarray(d["x"], float) / MM
        t = np.asarray(d.get("t", np.linspace(0, d["t_final"], len(x))), float)
        p = np.column_stack([t * 1e6, np.zeros(len(x)), np.zeros(len(x)), x])
        sc.paths(d["ion"].name, [p], color="time", width=2.4)
    sc.set(pathColor="time", clip=dict(normal=[1, 0, 0], offset=0.0, label="cut away the front half (x < 0)", on=True))
    lo, hi = np.array([-r_out - 2, -r_out - 2, -gap - 3]), np.array([r_out + 2, r_out + 2, L + skim + 5])
    jr = max(R, sk_r + 4) + 1
    _views(sc, lo, hi, [("spray", (-0.7, 0.4, -0.55), np.array([-2, -2, -gap - 3]), np.array([2, 2, 4])),
                        ("exit and jet", (-0.7, 0.4, -0.55), np.array([-jr, -jr, L - 4]), np.array([jr, jr, L + skim + 5]))])
    sc.set(view="exit and jet")
    return sc
