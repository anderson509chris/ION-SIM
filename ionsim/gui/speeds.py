"""Gas and ion speeds along the ion path, from each model's last run, for the whole-instrument view.

Each device tab keeps `speeds` after a run: {"span": (z0, z1) of its own axis in metres,
"gas": (z, u) gas speed along the axis or None for still gas, "ions": [(name, z, v)] median axial
ion speed in bins along z}. The Instrument tab lays the stages end to end (at their tagged place
when the stage came from the CAD model) and draws them on one axis.
"""
import numpy as np

MM = 1e-3
N_BINS = 60


def ion_speeds(res, species, z0=None, z1=None, nbins=N_BINS):
    """Median axial speed (m/s) of each species in bins along z, from the recorded paths."""
    out = []
    if res.traj is None or res.traj_t is None or res.traj.shape[0] < 2:
        return out
    tr = res.traj
    t = np.asarray(res.traj_t)
    z = tr[:, :, 2]
    zm = 0.5 * (z[1:] + z[:-1])
    vz = np.diff(z, axis=0) / np.diff(t)[:, None]
    lo = np.nanmin(z) if z0 is None else z0
    hi = np.nanmax(z) if z1 is None else z1
    edges = np.linspace(lo, hi, nbins + 1)
    rec_species = np.asarray(res.species)[:tr.shape[1]] if res.species is not None else np.zeros(tr.shape[1], int)
    for k, sp in enumerate(species):
        sel = rec_species == k
        if not sel.any():
            continue
        a, b = zm[:, sel].ravel(), vz[:, sel].ravel()
        ok = np.isfinite(a) & np.isfinite(b) & (a >= lo) & (a <= hi)
        a, b = a[ok], b[ok]
        if not len(a):
            continue
        idx = np.clip(np.digitize(a, edges) - 1, 0, nbins - 1)
        zc, vc = [], []
        for j in range(nbins):
            m = idx == j
            if m.sum() >= 3:
                zc.append(0.5 * (edges[j] + edges[j + 1])); vc.append(np.median(b[m]))
        if zc:
            out.append((sp.name, np.array(zc), np.array(vc)))
    return out


def gas_line(gas_u, start, z0, z1, n=200):
    """Gas speed along the axis line through `start` (x, y): from a flow field (anything with
    velocity(pos)), a uniform speed, or None for still gas."""
    z = np.linspace(z0, z1, n)
    if hasattr(gas_u, "velocity"):
        pos = np.column_stack([np.full(n, start[0]), np.full(n, start[1]), z])
        return z, np.linalg.norm(gas_u.velocity(pos), axis=1)
    if gas_u:
        return z, np.full(n, abs(float(gas_u)))
    return None


def draw_speeds(fig, rows):
    """rows: [(stage name, speeds, offset in m)] in order along the instrument."""
    from .ion_table import species_color
    fig.clear()
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    colors, shade = {}, ("0.93", "0.97")
    lo_v = []
    for k, (name, sp, off) in enumerate(rows):
        z0, z1 = (np.array(sp["span"]) + off) / MM
        for ax in (ax1, ax2):
            ax.axvspan(z0, z1, color=shade[k % 2], zorder=0)
        ax1.annotate(name, ((z0 + z1) / 2, 1.0), xycoords=("data", "axes fraction"), ha="center", va="bottom",
                     fontsize=8)
        if sp.get("gas") is not None:
            z, u = sp["gas"]
            ax1.plot((z + off) / MM, np.maximum(u, 1e-2), color="k", lw=1.2)
            lo_v.append(np.max(u))
        else:
            ax1.annotate("still gas", ((z0 + z1) / 2, 0.5), xycoords=("data", "axes fraction"), ha="center",
                         fontsize=7, color="0.4")
        for ion, z, v in sp["ions"]:
            c = colors.setdefault(ion, species_color(len(colors)))
            ax2.plot((z + off) / MM, np.maximum(np.abs(v), 1e-2), color=c, lw=1.2,
                     label=ion if ion not in [h.get_label() for h in ax2.lines[:-1]] else None)
    for ax, lab in ((ax1, "gas speed (m/s)"), (ax2, "ion speed along the axis (m/s)")):
        ax.set_yscale("log"); ax.set_ylabel(lab)
    if ax2.lines:
        ax2.legend(fontsize=7, loc="best")
    ax2.set_xlabel("position along the instrument (mm)")
    fig.suptitle("Gas and ion speeds along the instrument, from each model's last run "
                 "(ion speed: median of the recorded ions)", fontsize=9)
    return ax1, ax2
