"""Quadrupole mass filter tab."""
import time
import numpy as np
from PySide6 import QtWidgets

from ..constants import AMU, E_CHARGE
from ..geometry import quadrupole_rods
from ..solver import solve_basis
from ..fields import GridField2D, IdealQuadField, rf_dc
from ..ions import make_beam
from ..integrator import Simulation
from .. import mathieu
from .common import ParamForm, SimTab

MODES = ["Trajectories at (a, q)", "Trajectories at (U, V)", "Mass scan"]


class QuadTab(SimTab):
    def __init__(self, parent=None):
        super().__init__(["Potential", "Trajectories", "Cross-section", "Stability", "Mass scan"], parent)
        self.geom = ParamForm("Rods", [
            ("r0", "Field radius r0", 4.0, 0.5, 50, 3, 0.1, "mm"),
            ("ratio", "Rod radius / r0", 1.1487, 0.5, 2.0, 4, 0.01, ""),
            ("length", "Rod length", 200, 1, 5000, 1, 10, "mm"),
            ("div", "Grid points per r0", 80, 20, 200, 0, 10, ""),
        ])
        self.rf = ParamForm("RF", [
            ("freq", "Frequency", 1.0, 0.01, 100, 4, 0.1, "MHz"),
            ("steps", "Steps per RF cycle", 40, 10, 400, 0, 5, ""),
        ])
        self.ion = ParamForm("Ions", [
            ("mass", "m/z", 500, 1, 1e6, 2, 1, ""),
            ("z", "Charge state", 1, 1, 100, 0, 1, ""),
            ("energy", "Axial energy", 5.0, 0.01, 1000, 2, 0.5, "eV"),
            ("radius", "Beam radius", 0.3, 0.0, 10, 3, 0.05, "mm"),
            ("angle", "Beam half-angle", 1.0, 0.0, 30, 2, 0.25, "deg"),
        ])
        self.mode = QtWidgets.QComboBox(); self.mode.addItems(MODES)
        self.op = ParamForm("Operating point", [
            ("a", "a", 0.20, -1, 1, 4, 0.005, ""),
            ("q", "q", 0.70, 0, 2, 4, 0.005, ""),
            ("U", "U (DC)", 82.6, -1e5, 1e5, 2, 1, "V"),
            ("V", "V (RF 0-peak)", 578.3, 0, 1e5, 2, 5, "V"),
            ("n", "Ions", 20, 1, 5000, 0, 5, ""),
        ])
        self.scan = ParamForm("Mass scan", [
            ("aq", "Scan line a/q", 0.332, 0.0, 0.3357, 4, 0.001, ""),
            ("width", "Range +/-", 1.5, 0.1, 50, 2, 0.5, "%"),
            ("npts", "Masses", 41, 3, 400, 0, 2, ""),
            ("per", "Ions per mass", 100, 5, 5000, 0, 10, ""),
        ])
        mode_box = QtWidgets.QGroupBox("Run mode")
        QtWidgets.QVBoxLayout(mode_box).addWidget(self.mode)
        for w in (self.geom, self.rf, self.ion, mode_box, self.op, self.scan):
            self.form_layout.addWidget(w)
        self.form_layout.addStretch(1)

        self.cache = {}            # geometry key -> (grid, basis, r0_eff)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        for f in (self.geom, self.rf, self.ion):
            f.changed.connect(self._sync_op)
        self.op.boxes["a"].valueChanged.connect(self._sync_op)
        self.op.boxes["q"].valueChanged.connect(self._sync_op)
        self.op.boxes["U"].valueChanged.connect(self._sync_aq)
        self.op.boxes["V"].valueChanged.connect(self._sync_aq)
        self.run_btn.clicked.connect(self.run)
        self._syncing = False
        self._mode_changed()
        self._draw_stability(None)

    # ---- keep (a, q) and (U, V) consistent ----
    def _r0_eff(self):
        g = self.geom.values()
        key = (g["r0"], g["ratio"], g["div"])
        return self.cache[key][2] if key in self.cache else g["r0"] * 1e-3

    def _sync_op(self):
        if self._syncing or self.mode.currentIndex() == 1:
            return
        self._syncing = True
        o, r, i = self.op.values(), self.rf.values(), self.ion.values()
        U, V = mathieu.voltages_for(o["a"], o["q"], i["mass"], self._r0_eff(), r["freq"] * 1e6, i["z"])
        self.op.set("U", U); self.op.set("V", V)
        self._syncing = False

    def _sync_aq(self):
        if self._syncing or self.mode.currentIndex() != 1:
            return
        self._syncing = True
        o, r, i = self.op.values(), self.rf.values(), self.ion.values()
        a, q = mathieu.aq(o["U"], o["V"], i["mass"], self._r0_eff(), r["freq"] * 1e6, i["z"])
        self.op.set("a", float(a)); self.op.set("q", float(q))
        self._syncing = False

    def _mode_changed(self):
        m = self.mode.currentIndex()
        self.op.setVisible(m in (0, 1)); self.scan.setVisible(m == 2)
        for k in ("a", "q"):
            self.op.boxes[k].setEnabled(m == 0)
        for k in ("U", "V"):
            self.op.boxes[k].setEnabled(m == 1)

    # ---- run ----
    def run(self):
        g, r, i, o, s = (f.values() for f in (self.geom, self.rf, self.ion, self.op, self.scan))
        mode = self.mode.currentIndex()
        key = (g["r0"], g["ratio"], g["div"])
        cache = self.cache

        def job(progress):
            out = {"mode": mode, "g": g, "r": r, "i": i}
            r0 = g["r0"] * 1e-3
            if key not in cache:
                t = time.time()
                progress(0.02)
                grid = quadrupole_rods(r0, rod_ratio=g["ratio"], dx=r0 / g["div"])
                basis = solve_basis(grid)
                static = GridField2D(grid, basis, {"A": 1.0, "B": -1.0})
                p = np.array([[0.1 * r0, 0, 0]])
                ratio = static.efield(0, p)[0, 0] / IdealQuadField(r0, 1.0).efield(0, p)[0, 0]
                cache[key] = (grid, basis, r0 / np.sqrt(ratio))
                out["solve_time"] = time.time() - t
            grid, basis, r0_eff = cache[key]
            out.update(grid=grid, basis=basis, r0_eff=r0_eff)
            freq = r["freq"] * 1e6
            dt = 1 / freq / r["steps"]
            mass, zc = i["mass"], i["z"]
            v_ax = np.sqrt(2 * i["energy"] * E_CHARGE * zc / (mass * AMU))
            L = g["length"] * 1e-3
            rng = np.random.default_rng()

            def field(U, V):
                phi0 = rf_dc(U, V, freq)
                return GridField2D(grid, basis, {"A": phi0, "B": lambda tt: -phi0(tt)})

            if mode in (0, 1):
                if mode == 0:
                    U, V = mathieu.voltages_for(o["a"], o["q"], mass, r0_eff, freq, zc)
                else:
                    U, V = o["U"], o["V"]
                a, q = mathieu.aq(U, V, mass, r0_eff, freq, zc)
                ions = make_beam(int(o["n"]), mass, zc, i["energy"], i["radius"] * 1e-3, i["angle"],
                                 birth_window=1 / freq, rng=rng)
                res = Simulation([field(U, V)], z_end=L).run(ions, dt, 3 * L / v_ax + 1 / freq,
                                                            record_every=2, rng=rng, progress=progress)
                out.update(U=U, V=V, a=float(a), q=float(q), res=res, stable=bool(mathieu.is_stable(a, q)))
            else:
                U, V = mathieu.voltages_for(s["aq"] * 0.706, 0.706, mass, r0_eff, freq, zc)
                masses = np.linspace(mass * (1 - s["width"] / 100), mass * (1 + s["width"] / 100), int(s["npts"]))
                per = int(s["per"])
                ions = make_beam(per * len(masses), np.repeat(masses, per), zc, i["energy"],
                                 i["radius"] * 1e-3, i["angle"], birth_window=1 / freq, rng=rng)
                v_min = np.sqrt(2 * i["energy"] * E_CHARGE * zc / (masses.max() * AMU))
                res = Simulation([field(U, V)], z_end=L).run(ions, dt, 1.05 * L / v_min + 1 / freq,
                                                            rng=rng, progress=progress)
                T = (res.status == 1).reshape(len(masses), per).mean(axis=1)
                a_m, q_m = mathieu.aq(U, V, masses, r0_eff, freq, zc)
                out.update(U=U, V=V, masses=masses, T=T, a_m=a_m, q_m=q_m,
                           pred=mathieu.is_stable(a_m, q_m))
            return out

        self.start(job, self._show, "Solving field and flying ions..." if key not in cache else "Flying ions...")

    # ---- display ----
    def _show(self, out):
        if "solve_time" in out:
            self.log(f"Laplace solved on {out['grid'].n}x{out['grid'].n} grid in {out['solve_time']:.1f} s; "
                     f"effective r0 = {out['r0_eff'] * 1e3:.4f} mm")
            self._sync_op()
        self._draw_potential(out)
        if out["mode"] in (0, 1):
            res = out["res"]
            self.log(f"U = {out['U']:.2f} V, V = {out['V']:.2f} V -> a = {out['a']:.4f}, q = {out['q']:.4f} "
                     f"({'Mathieu stable' if out['stable'] else 'Mathieu unstable'}); "
                     f"transmitted {np.sum(res.status == 1)}/{len(res.status)} ({res.transmission:.0%}), "
                     f"hit rods {np.sum(res.status == 2)}")
            self.last_table = (["ion", "status(1=through,2=hit)", "t_end_s", "x_m", "y_m", "z_m"],
                               [[k, int(res.status[k]), res.t_end[k], *res.final_pos[k]] for k in range(len(res.status))])
            self._draw_traj(out)
            self._draw_stability([(out["q"], out["a"])])
        else:
            T, m, pred = out["T"], out["masses"], out["pred"]
            msg = f"Mass scan U = {out['U']:.2f} V, V = {out['V']:.2f} V: peak T = {T.max():.0%}"
            if T.max() > 0:
                half = T >= T.max() / 2
                msg += f", FWHM {m[half].min():.2f}-{m[half].max():.2f}"
            if pred.any():
                msg += f"; Mathieu-stable {m[pred].min():.2f}-{m[pred].max():.2f}"
            self.log(msg)
            self.last_table = (["mz", "transmission", "a", "q", "mathieu_stable"],
                               [[m[k], T[k], out["a_m"][k], out["q_m"][k], int(pred[k])] for k in range(len(m))])
            self._draw_stability(list(zip(out["q_m"], out["a_m"])), T)
            self._draw_scan(out)

    def _draw_potential(self, out):
        grid = out["grid"]
        fig = self.plots.fig("Potential", show=False)
        ax = fig.add_subplot(111)
        pot = GridField2D(grid, out["basis"], {"A": 1.0, "B": -1.0}).potential(0)
        m = grid.electrode > 0
        cs = ax.contourf(grid.X * 1e3, grid.Y * 1e3, np.ma.masked_where(m, pot), 41, cmap="RdBu_r")
        ax.contourf(grid.X * 1e3, grid.Y * 1e3, m, levels=[0.5, 1.5], colors="0.45")
        ax.set_aspect("equal"); ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
        ax.set_title("Potential per volt (rods A = +1, B = -1)")
        fig.colorbar(cs, ax=ax, label="V")
        self.plots.draw("Potential")

    def _draw_traj(self, out):
        res, r0 = out["res"], out["g"]["r0"]
        fig = self.plots.fig("Trajectories")
        ax1, ax2 = fig.subplots(2, 1, sharex=True)
        for k in range(res.traj.shape[1]):
            ax1.plot(res.traj[:, k, 2] * 1e3, res.traj[:, k, 0] * 1e3, lw=0.7)
            ax2.plot(res.traj[:, k, 2] * 1e3, res.traj[:, k, 1] * 1e3, lw=0.7)
        for ax, lab in ((ax1, "x (mm)"), (ax2, "y (mm)")):
            ax.axhline(r0, color="k", lw=0.6); ax.axhline(-r0, color="k", lw=0.6)
            ax.set_ylim(-1.3 * r0, 1.3 * r0); ax.set_ylabel(lab)
        ax2.set_xlabel("z (mm)")
        ax1.set_title(f"a = {out['a']:.4f}, q = {out['q']:.4f}, transmission {res.transmission:.0%}")
        self.plots.draw("Trajectories")

        fig = self.plots.fig("Cross-section", show=False)
        ax = fig.add_subplot(111)
        grid = out["grid"]
        ax.contourf(grid.X * 1e3, grid.Y * 1e3, grid.electrode > 0, levels=[0.5, 1.5], colors="0.75")
        for k in range(res.traj.shape[1]):
            ax.plot(res.traj[:, k, 0] * 1e3, res.traj[:, k, 1] * 1e3, lw=0.5)
        lim = 1.6 * r0
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_title("Ion paths seen down the axis")
        self.plots.draw("Cross-section")

    def _draw_stability(self, pts, colour=None):
        if not hasattr(self, "_stab_map"):
            self._stab_map = mathieu.stability_map(1.0, 0.3, 240, 160)
        Q, A, S = self._stab_map
        fig = self.plots.fig("Stability", show=False)
        ax = fig.add_subplot(111)
        ax.contourf(Q, A, S, levels=[0.5, 1.5], colors=["#9ecae1"])
        if pts:
            q, a = np.array(pts).T
            if colour is None:
                ax.plot(q, a, "o", color="C3", ms=8, label="operating point")
            else:
                sc = ax.scatter(q, a, c=colour, s=10, cmap="viridis", label="scanned masses")
                fig.colorbar(sc, ax=ax, label="transmission")
            ax.legend(fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 0.3); ax.set_xlabel("q"); ax.set_ylabel("a")
        ax.set_title("Mathieu stability diagram (first region)")
        self.plots.draw("Stability")

    def _draw_scan(self, out):
        fig = self.plots.fig("Mass scan")
        ax = fig.add_subplot(111)
        ax.plot(out["masses"], out["T"] * 100, "o-", ms=3, label="simulated")
        ax.fill_between(out["masses"], 0, out["pred"] * 100, color="C1", alpha=0.2, step="mid",
                        label="Mathieu-stable (ideal)")
        ax.set_xlabel("m/z"); ax.set_ylabel("transmission (%)"); ax.legend()
        ax.set_title(f"U = {out['U']:.1f} V, V = {out['V']:.1f} V")
        self.plots.draw("Mass scan")
