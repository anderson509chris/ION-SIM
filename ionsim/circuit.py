"""The electrodes as a circuit load: capacitance, crosstalk, and the drive they really get.

The field solve already knows how much charge each electrode carries for a given set of
voltages, so it gives the Maxwell capacitance matrix C: Q_i = sum_j C_ij V_j. For each net
(the electrodes on one driver output, or one electrode) the net is set to 1 V and the rest
to 0 V, the Laplace problem is solved, and the charge on every net is the flux of
eps grad(phi) through its surface (summed over the grid links that touch it, so insulators
count with their permittivity). C_ii is a net's total capacitance; -C_ij (i != j) is the
mutual capacitance between nets i and j; the row sum is the capacitance to ground (the
grounded box around the device and anything held at 0 V).

From C:
  spice_subckt()  writes the load as a SPICE subcircuit (one pin per net, capacitors to
                  ground and between nets) to put after your own driver model;
  drive_netlist() writes a full netlist in which every net is driven by its ION-SIM waveform
                  through a source resistance and wiring inductance, and
  loaded_drive()  runs it with ngspice and returns what each electrode actually sees, the
                  drive current, and the crosstalk onto the other nets;
  apply_load()    hands the difference (electrode minus source) back to the device, where it
                  is one more field part per net on top of the drive (Device.load), so the
                  ions fly in the fields the electrodes really get.
"""
import os
import re
import tempfile
import numpy as np

from .gridtools import EPS0


# ---------------------------------------------------------------------------------
# capacitance from the field solve

def capacitance_matrix(solver, electrode, flux_scale, groups, progress=None):
    """Maxwell capacitance matrix (F) of groups of electrode ids on a solved grid.
    solver: LaplaceSolver; electrode: id per node (0 = free); flux_scale: physical size of a
    unit link weight (GridMap.flux_scale()); groups: [[ids], ...], one per net."""
    el = electrode.ravel()
    free_flat = np.flatnonzero(~solver.fixed.ravel())
    fi = np.concatenate([c[0] for c in solver.couplings])
    fx = np.concatenate([c[1] for c in solver.couplings])
    w = np.concatenate([c[2] for c in solver.couplings])
    gid = np.full(int(el.max()) + 1 if el.size else 1, -1)
    for k, ids in enumerate(groups):
        gid[np.asarray(ids, int)] = k
    owner = np.where(el[fx] > 0, gid[el[fx]], -1)          # which net each fixed end belongs to (-1: ground / boundary)
    n = len(groups)
    C = np.zeros((n, n))
    for k in range(n):
        v = np.zeros(gid.shape)
        v[np.asarray(groups[k], int)] = 1.0
        vals = v[electrode]
        phi = solver.solve(vals).ravel()
        flux = w * (phi[fx] - phi[free_flat[fi]])            # outward from the fixed node into free space
        on = owner >= 0
        C[:, k] = np.bincount(owner[on], weights=flux[on], minlength=n) * EPS0 * flux_scale
        if progress and progress((k + 1) / n):
            break
    return C


class Capacitance:
    """Capacitance of the nets of a device. names[i] is net i; members[i] its electrode names."""

    def __init__(self, names, members, C, note=""):
        self.names, self.members, self.note = list(names), [list(m) for m in members], note
        self.C = 0.5 * (np.asarray(C, float) + np.asarray(C, float).T)     # symmetric up to the grid error
        self.asym = float(np.max(np.abs(np.asarray(C) - np.asarray(C).T)) / max(np.max(np.abs(C)), 1e-30))

    @property
    def n(self):
        return len(self.names)

    def total(self):
        """Total capacitance of each net (F): what its driver charges when the others stay put."""
        return np.diag(self.C).copy()

    def to_ground(self):
        return self.C.sum(axis=1)

    def mutual(self):
        """Mutual capacitance between nets (F, >= 0), zero diagonal."""
        M = -self.C.copy()
        np.fill_diagonal(M, 0.0)
        return np.maximum(M, 0.0)

    def coupling(self):
        """Fraction of a voltage step on net j that appears on net i when net i floats
        (C_ij / C_ii, capacitive divider): the worst-case crosstalk, before the driver holds it."""
        M = self.mutual()
        return M / np.maximum(self.total(), 1e-30)[:, None]

    def worst_pairs(self, k=8):
        M = self.coupling()
        idx = np.dstack(np.unravel_index(np.argsort(-M, axis=None), M.shape))[0]
        return [(self.names[i], self.names[j], M[i, j], self.mutual()[i, j]) for i, j in idx[:k] if M[i, j] > 0]

    def table(self):
        rows = ["net, total (pF), to ground (pF), largest mutual (pF) to"]
        M = self.mutual()
        for i, nm in enumerate(self.names):
            j = int(np.argmax(M[i])) if self.n > 1 else i
            rows.append(f"{nm}, {self.C[i, i] * 1e12:.4g}, {self.to_ground()[i] * 1e12:.4g}, "
                        + (f"{M[i, j] * 1e12:.4g} {self.names[j]}" if self.n > 1 else "-"))
        return rows

    def to_dict(self):
        return dict(names=self.names, members=self.members, C=self.C.tolist(), note=self.note)


def nets(dev, by="output"):
    """[(net name, [electrode ids])] of a Device. by "output": electrodes wired to the same
    driver output are one net (they share a wire); electrodes not wired to a driver are one net
    each. by "electrode": every electrode its own net. Insulators are left out."""
    out, where = [], {}
    for k, e in enumerate(dev.electrodes, start=1):
        if e.eps > 0:
            continue
        key = e.drive if by == "output" and dev.driver_of(e) else f"#{e.name}"
        if key not in where:
            where[key] = len(out)
            out.append((key if not key.startswith("#") else e.name, []))
        out[where[key]][1].append(k)
    return out


def device_capacitance(dgrid, by="output", progress=None):
    """Capacitance of a DeviceGrid's nets (one Laplace solve per net, on the grid's own solver)."""
    dev = dgrid.dev
    ns = nets(dev, by)
    C = capacitance_matrix(dgrid.solver(), dgrid.electrode, dgrid.gridmap().flux_scale(), [m for _, m in ns], progress)
    names = [n for n, _ in ns]
    members = [[dev.electrodes[i - 1].name for i in m] for _, m in ns]
    note = {"planar2d": f"2D cross-section times the {dev.length * 1e3:g} mm length (no end effects)",
            "boards": "both boards' copies of a pad are one electrode" if dev.both_boards else ""}.get(dev.kind, "")
    return Capacitance(names, members, C, note)


# ---------------------------------------------------------------------------------
# SPICE

def spice_name(s):
    """A node name SPICE accepts."""
    s = s.replace("+", "p").replace("-", "m").replace("/", "_")      # RF+ / RF- stay apart
    n = re.sub(r"[^A-Za-z0-9_]+", "_", s).strip("_") or "net"
    return ("n_" + n) if n[0].isdigit() else n


def _unique_nodes(names):
    out, seen = [], set()
    for nm in names:
        n, k = spice_name(nm), 2
        while n.lower() in seen:
            n = f"{spice_name(nm)}_{k}"; k += 1
        seen.add(n.lower())
        out.append(n)
    return out


def _val(x):
    return f"{x:.6g}"


def spice_subckt(cap, name="IONSIM_LOAD", min_rel=1e-4, title=""):
    """The nets as a SPICE subcircuit: .subckt NAME <one pin per net>, each pin's capacitance
    to ground and the mutual capacitance to every other pin. Mutuals below min_rel of the
    largest total capacitance are left out."""
    nodes = _unique_nodes(cap.names)
    tg, M = cap.to_ground(), cap.mutual()
    floor = min_rel * max(cap.total().max(), 1e-30)
    lines = [f"* {title or 'Electrode load'} from ION-SIM: {cap.n} nets",
             "* pins, in order (net: electrodes):"]
    for nd, nm, mem in zip(nodes, cap.names, cap.members):
        lines.append(f"*   {nd} = {nm}: {', '.join(mem[:8])}{' ...' if len(mem) > 8 else ''}")
    if cap.note:
        lines.append(f"* {cap.note}")
    lines.append(f".subckt {name} " + " ".join(nodes))
    for i, nd in enumerate(nodes):
        if tg[i] > floor:
            lines.append(f"Cg_{nd} {nd} 0 {_val(tg[i])}")
    k = 0
    for i in range(cap.n):
        for j in range(i + 1, cap.n):
            if M[i, j] > floor:
                k += 1
                lines.append(f"Cm{k} {nodes[i]} {nodes[j]} {_val(M[i, j])}")
    lines.append(f".ends {name}")
    return "\n".join(lines) + "\n"


def _pwl(t, v, per_line=6):
    pts = [f"{a:.9g} {b:.7g}" for a, b in zip(t, v)]
    rows = ["+ " + " ".join(pts[i:i + per_line]) for i in range(0, len(pts), per_line)]
    return "PWL(\n" + "\n".join(rows) + " )"


def net_waveforms(dev, cap, t):
    """Source voltage of each net at times t (the first electrode of a net stands for it)."""
    from .device import electrode_waveforms
    V = electrode_waveforms(dev, t, with_load=False)
    idx = {e.name: k for k, e in enumerate(dev.electrodes)}
    return np.array([V[idx[m[0]]] for m in cap.members])


def drive_window(dev):
    """A time span that holds a whole period of the device's drive (s): the slowest repeating
    part (RF, tones, TWAVE cycle, pulse period, imported waveform), at most 1 ms."""
    return _timing(dev)[0]


def _timing(dev):
    """(window, shortest time feature) of a device's drive."""
    from .device import part_values
    _, meta = part_values(dev)
    spans, fine = [], []
    for m in meta.values():
        if m[0] in ("cos", "sin") and m[1] > 0:
            fine.append(1 / m[1] / 40)
        elif m[0] == "tw" and np.isfinite(m[1].step):
            fine.append(m[1].rise / 4 if m[1].rise > 0 else m[1].step / 100)
        elif m[0] == "pulse":
            fine += [r / 4 for r in (m[1].rise, m[1].fall) if r > 0]
        elif m[0] == "wave":
            fine.append(m[1].time_step())
    for m in meta.values():
        if m[0] in ("cos", "sin") and m[1] > 0:
            spans.append(1 / m[1])
        elif m[0] == "tw" and np.isfinite(m[1].step):
            spans.append(m[1].n_phases * m[1].step)
        elif m[0] == "pulse" and m[1].rate > 0:
            spans.append(1 / m[1].rate)
        elif m[0] == "wave":
            spans.append(m[1].cycle() + m[1].delay)
    slow = min(max(spans) if spans else 1e-6, 1e-3)
    # repeat over a window that also holds a whole number of the faster periods, as near as can be
    # (within 20 slow periods and 1 ms): then the steady state repeats without a phase jump
    best, span = np.inf, slow
    for k in range(1, 21):
        w = k * slow
        if k > 1 and w > 1e-3:
            break
        miss = max((abs(w / p - np.round(w / p)) for p in spans if p < slow), default=0.0)
        if miss < best - 1e-3:
            best, span = miss, w
        if miss < 0.01:
            break
    return span, min(fine) if fine else span / 1000


def drive_netlist(dev, cap, r_source=50.0, l_wire=0.0, window=None, periods=2, points=None, title="", r_dc=None):
    """A netlist that drives every net from its ION-SIM waveform through r_source (ohm) and
    l_wire (H), into the electrode load; nets held at DC go through r_dc (a supply's output
    resistor or filter; default r_source). Runs `periods` windows so the start-up settles; the
    last window is the steady state. points: samples per window (default: enough for the
    fastest feature of the drive, 2 000 to 100 000). Returns (netlist text, node names, window, points)."""
    win, fine = _timing(dev)
    window = window or win
    points = points or int(np.clip(window / max(fine, 1e-15), 2000, 100_000))
    t = np.linspace(0, periods * window, int(points * periods) + 1)
    V = net_waveforms(dev, cap, t)
    nodes = _unique_nodes(cap.names)
    out = [f"* {title or 'Electrode drive'} from ION-SIM: {cap.n} nets driven through "
           f"{r_source:g} ohm" + (f" and {l_wire * 1e9:g} nH" if l_wire else ""),
           spice_subckt(cap, title=title).rstrip()]
    for k, nd in enumerate(nodes):
        src = f"src_{k}"
        dc = np.ptp(V[k]) == 0
        r = r_source if not dc or r_dc is None else r_dc
        if dc:
            out.append(f"V{k} {src} 0 DC {_val(V[k][0])}")
        else:
            out.append(f"V{k} {src} 0 " + _pwl(t, V[k]))
        if l_wire > 0:
            out.append(f"R{k} {src} mid_{k} {_val(max(r, 1e-6))}")
            out.append(f"L{k} mid_{k} {nd} {_val(l_wire)}")
        else:
            out.append(f"R{k} {src} {nd} {_val(max(r, 1e-6))}")
    out.append("Xload " + " ".join(nodes) + " IONSIM_LOAD")
    step = window / points
    out.append(f".tran {_val(step)} {_val(periods * window)} 0 {_val(step)}")
    out.append(".options method=gear")
    out.append(".save " + " ".join([f"v({n})" for n in nodes] + [f"i(V{k})" for k in range(len(nodes))]))
    out.append(".end")
    return "\n".join(out) + "\n", nodes, window, points


def short_names(names):
    """Net labels for plots: the output names alone when every net is on the same driver."""
    drv = {n.split(" / ", 1)[0] for n in names if " / " in n}
    return [n.split(" / ", 1)[1] if " / " in n and len(drv) == 1 else n for n in names]


class LoadedDrive:
    """What each net sees with the load: t (one steady-state window from 0), source and
    electrode voltages (n_nets, len(t)), and the source current."""

    def __init__(self, cap, t, v_src, v_el, i_src, window):
        self.cap, self.t, self.v_src, self.v_el, self.i_src, self.window = cap, t, v_src, v_el, i_src, window

    def report(self):
        """[dict] per net: swing at the source and at the electrode, the largest difference,
        rms and peak current, reactive power."""
        out = []
        for k, nm in enumerate(self.cap.names):
            vs, ve, i = self.v_src[k], self.v_el[k], self.i_src[k]
            sw_s, sw_e = float(np.ptp(vs)), float(np.ptp(ve))
            out.append(dict(name=nm, swing_src=sw_s, swing_el=sw_e, max_err=float(np.max(np.abs(ve - vs))),
                            i_rms=float(np.sqrt(np.mean(i ** 2))), i_peak=float(np.max(np.abs(i))),
                            var=float(np.sqrt(np.mean(i ** 2)) * np.sqrt(np.mean((ve - ve.mean()) ** 2))),
                            driven=sw_s > 0))
        return out

    def deviations(self, max_points=20000):
        """{net name: waveforms.Trace} of electrode minus source voltage, one repeating window each."""
        from .waveforms import Trace
        return {nm: Trace(f"load on {nm}", self.t, self.v_el[k] - self.v_src[k],
                          f"SPICE with the electrode load: {nm}", repeat=True).decimated(max_points)
                for k, nm in enumerate(self.cap.names)}


def loaded_drive(dev, cap, r_source=50.0, l_wire=0.0, window=None, periods=2, points=None, exe=None, keep=None,
                 r_dc=None):
    """Run drive_netlist with ngspice and return a LoadedDrive (the last window). keep: a folder
    to leave the netlist and raw file in."""
    from .waveforms import run_spice, read_raw
    text, nodes, window, points = drive_netlist(dev, cap, r_source, l_wire, window, periods, points, r_dc=r_dc)
    folder = keep or tempfile.mkdtemp(prefix="ionsim_load_")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "electrode_drive.cir")
    with open(path, "w") as f:
        f.write(text)
    raw = run_spice(path, "ngspice", exe)
    data = {k.lower(): v for k, v in read_raw(raw).items()}
    t_all = data[f"v({nodes[0].lower()})"][0]
    t_new = np.linspace(0, window, points + 1)
    t0 = (periods - 1) * window
    pick = lambda key: np.interp(t0 + t_new, *data[key])
    v_el = np.array([pick(f"v({n.lower()})") for n in nodes])
    i_src = np.array([-pick(f"i(v{k})") for k in range(len(nodes))])     # SPICE source current is into its + pin
    v_src = net_waveforms(dev, cap, t0 + t_new)
    if len(t_all) < 2:
        raise RuntimeError("ngspice returned no time points")
    return LoadedDrive(cap, t_new, v_src, v_el, i_src, window)


def apply_load(dev, ld, min_rel=1e-3):
    """Give the device what the load does to each net (electrode minus source voltage), as one
    more field part per net on top of its drive; nets the load moves by less than min_rel of the
    largest drive voltage are left as they are. clear_load(dev) takes it off again."""
    floor = min_rel * max(float(np.max(np.ptp(ld.v_src, axis=1))), float(np.max(np.abs(ld.v_src))), 1e-12)
    dev.load = [(nm, list(mem), tr) for (nm, mem), tr in zip(zip(ld.cap.names, ld.cap.members), ld.deviations().values())
                if np.max(np.abs(tr.v)) > floor]
    return dev.load


def clear_load(dev):
    dev.load = []
