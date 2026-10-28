"""Quadrupole mass filter tab."""
import time
import numpy as np
from PySide6 import QtCore, QtWidgets

from ..constants import AMU, E_CHARGE
from ..geometry import quadrupole_rods
from ..solver import solve_basis
from ..fields import GridField2D, IdealQuadField, rf_dc
from ..ions import make_beam, IonSpecies, expand
from ..integrator import Simulation
from .. import mathieu
from ..stages import ExitIons
from ..collisions import GASES, AxialField, HardSphereGas, ccs_from_k0
from .common import ParamForm, SimTab
from .stage_picker import SourcePicker
from .driver_store import get_store
from .ion_table import IonTable, path_style, fate_caption, species_legend, species_stats, species_color

MODES = ["Trajectories at (a, q)", "Trajectories at (U, V)", "Mass scan"]
DRIVER_MODES = ["Trajectories at the ion m/z", "Transmission vs m/z (excitation scan)"]
RODS = ["x+", "x-", "y+", "y-"]
BUILTIN = "Built-in RF/DC (set a, q or U, V below)"


class QuadTab(SimTab):
    def __init__(self, parent=None, store=None):
        super().__init__(["Potential", "Trajectories", "Cross-section", "Stability", "Mass scan", "Excitation"], parent)
        self.plots.hint("Mass scan", "Set Run mode to “Mass scan” and press Run.\n(With a driver picked under Drive: “Transmission vs m/z”.)")
        self.plots.hint("Excitation", "Pick a driver with excitation tones under Drive,\nset Run mode to “Transmission vs m/z (excitation scan)” and press Run.")
        self.store = store or get_store()
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
        self.ion_table = IonTable("Ions (one row per species; the first sets the operating point and scans)",
                                  [IonSpecies("m/z 500", 500, 1, 1.0, 20)], columns=("name", "mz", "z", "K0", "n"))
        self.ion_table.setToolTip("K0 (reduced mobility in the collision gas) sets the collision cross section; "
                                  "it only matters with gas in the quadrupole")
        self.ion = ParamForm("Beam", [
            ("energy", "Axial energy", 5.0, 0.01, 1000, 2, 0.5, "eV"),
            ("radius", "Beam radius", 0.3, 0.0, 10, 3, 0.05, "mm"),
            ("angle", "Beam half-angle", 1.0, 0.0, 30, 2, 0.25, "deg"),
        ])
        # ---- gas: an RF-only collision cell or cooling quadrupole ----
        self.gas = ParamForm("Collision gas", [
            ("p", "Pressure (0 = vacuum)", 0.0, 0.0, 1.0, 5, 0.001, "Torr"),
            ("T", "Gas temperature", 300, 4, 2000, 0, 10, "K"),
            ("ez", "Axial drag field", 0.0, -1e4, 1e4, 3, 0.1, "V/cm"),
            ("tmax", "Max flight time (0 = auto)", 0.0, 0.0, 1e3, 3, 0.5, "ms"),
        ])
        self.gas_kind = QtWidgets.QComboBox(); self.gas_kind.addItems(list(GASES)); self.gas_kind.setCurrentText("N₂")
        self.gas.layout().insertRow(1, "Gas", self.gas_kind)
        self.gas.setToolTip("With gas, every step each ion may hit one gas molecule (hard spheres, cross section "
                            "from its K0). The axial field pushes ions through when collisions have slowed them.")
        # ---- stages: start the ions that left a funnel or Device builder model ----
        from .funnel_tab import FunnelTab
        from .device_tab import DeviceTab
        self.picker = SourcePicker(self.ion.layout(), (FunnelTab, DeviceTab), parent=self,
                                   tip="Start the ions that left another model (an ion funnel or Device builder run "
                                       "with trajectories) at the rod entrance, ion by ion",
                                   step_tip="The DC step from that model's exit to the rod axis: it adds this much "
                                            "axial energy per charge")
        self.picker.step.setValue(5.0)
        self.picker.changed.connect(self._apply_src)
        self.mode = QtWidgets.QComboBox(); self.mode.addItems(MODES)
        self.op = ParamForm("Operating point", [
            ("a", "a", 0.20, -1, 1, 4, 0.005, ""),
            ("q", "q", 0.70, 0, 2, 4, 0.005, ""),
            ("U", "U (DC)", 82.6, -1e5, 1e5, 2, 1, "V"),
            ("V", "V (RF 0-peak)", 578.3, 0, 1e5, 2, 5, "V"),
        ])
        self.scan = ParamForm("Mass scan", [
            ("aq", "Scan line a/q", 0.332, 0.0, 0.3357, 4, 0.001, ""),
            ("width", "Range +/-", 1.5, 0.1, 50, 2, 0.5, "%"),
            ("npts", "Masses", 41, 3, 400, 0, 2, ""),
            ("per", "Ions per mass", 100, 5, 5000, 0, 10, ""),
        ])
        self.dscan = ParamForm("Excitation scan", [
            ("m1", "m/z from", 100, 1, 1e6, 1, 10, ""),
            ("m2", "m/z to", 2000, 1, 1e6, 1, 10, ""),
            ("npts", "Masses (log spaced)", 60, 3, 400, 0, 5, ""),
            ("per", "Ions per mass", 30, 5, 5000, 0, 5, ""),
        ])

        # ---- drive: built-in RF/DC or a driver from the Drivers tab ----
        drive = QtWidgets.QGroupBox("Drive")
        dl = QtWidgets.QFormLayout(drive)
        self.drive = QtWidgets.QComboBox()
        self.drive.setToolTip("Built-in RF/DC, or one of the drivers you built in the Drivers tab "
                              "(its RF, DC and excitation tones then drive the rods)")
        dl.addRow("Source", self.drive)
        self.rod_box = {}
        for rod, label in zip(RODS, ("Rod x+", "Rod x−", "Rod y+", "Rod y−")):
            cb = QtWidgets.QComboBox(); cb.currentIndexChanged.connect(self._driver_info)
            dl.addRow(label, cb); self.rod_box[rod] = cb
        self.drive_info = QtWidgets.QLabel(); self.drive_info.setWordWrap(True)
        self.drive_info.setTextFormat(QtCore.Qt.RichText)
        dl.addRow(self.drive_info)
        retune = QtWidgets.QHBoxLayout()
        self.tone_k = QtWidgets.QSpinBox(); self.tone_k.setRange(1, 64); self.tone_k.setPrefix("tone ")
        self.tone_mz = QtWidgets.QDoubleSpinBox(); self.tone_mz.setRange(1, 1e6); self.tone_mz.setValue(500)
        self.tone_mz.setPrefix("m/z ")
        b = QtWidgets.QPushButton("Aim"); b.setToolTip("Set this tone's frequency so it ejects that m/z")
        b.clicked.connect(self._retune)
        retune.addWidget(QtWidgets.QLabel("Aim")); retune.addWidget(self.tone_k)
        retune.addWidget(QtWidgets.QLabel("at")); retune.addWidget(self.tone_mz); retune.addWidget(b)
        self.retune_row = QtWidgets.QWidget(); self.retune_row.setLayout(retune)
        dl.addRow(self.retune_row)

        mode_box = QtWidgets.QGroupBox("Run mode")
        QtWidgets.QVBoxLayout(mode_box).addWidget(self.mode)
        for w in (self.geom, drive, self.rf, self.ion_table, self.ion, self.gas, mode_box, self.op, self.scan, self.dscan):
            self.form_layout.addWidget(w)
        self.form_layout.addStretch(1)

        self.cache = {}            # geometry key -> (grid, basis, r0_eff)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        for f in (self.geom, self.rf, self.ion, self.ion_table):
            f.changed.connect(self._sync_op)
        self.op.boxes["a"].valueChanged.connect(self._sync_op)
        self.op.boxes["q"].valueChanged.connect(self._sync_op)
        self.op.boxes["U"].valueChanged.connect(self._sync_aq)
        self.op.boxes["V"].valueChanged.connect(self._sync_aq)
        self.run_btn.clicked.connect(self.run)
        self._syncing = False
        self.drive.currentIndexChanged.connect(self._drive_changed)
        self.store.changed.connect(self._fill_drivers)
        self.ion_table.changed.connect(self._driver_info)
        self.geom.changed.connect(self._driver_info)
        self._fill_drivers()
        self._mode_changed()
        self._draw_stability(None)

    # ---- stages ----
    def set_sources(self, sources):
        self.picker.set_sources(sources)

    def source(self):
        return self.picker.source()

    def source_name(self):
        return self.picker.source_name()

    def _apply_src(self, *_):
        """Linked: the ion table lists the species that arrive (with how many), and the beam comes from them."""
        on = self.picker.linked()
        for k in ("energy", "radius", "angle"):
            self.ion.boxes[k].setEnabled(not on)
        self.ion_table.table.setEnabled(not on)
        for b in self.ion_table.findChildren(QtWidgets.QPushButton):
            b.setEnabled(not on)
        if not on:
            return
        name, ex = self.picker.source_name(), self.picker.data()
        if ex is None:
            self.picker.set_note(f"Run '{name}' with trajectories: the ions that leave it start here.")
            return
        cnt = np.bincount(ex.idx, minlength=len(ex.species))
        self.ion_table.set_species([IonSpecies(s.name, s.mz, abs(int(s.z)), s.K0, int(c))
                                    for s, c in zip(ex.species, cnt)])
        e = 0.5 * np.array([ex.species[k].mz * abs(ex.species[k].z) for k in ex.idx]) * AMU * ex.vel[:, 2] ** 2 \
            / (np.abs([ex.species[k].z for k in ex.idx]) * E_CHARGE) if ex.n else np.zeros(0)
        r = np.hypot(ex.pos[:, 0] - ex.center[0], ex.pos[:, 1] - ex.center[1]) * 1e3
        self.picker.set_note(f"From the {ex.source} of '{name}': {ex.n} ions"
                             + (f", {np.mean(e):.3g} eV axial per charge at its exit (plus the step), "
                                f"r_rms {np.sqrt(np.mean(r ** 2)):.3g} mm" if ex.n else "")
                             + ". They keep their position and side speed; their RF phase is new "
                               "(no timing from the model before).")

    def _ion_values(self):
        """Beam settings plus the reference ion (first row). a and q depend only on m/z, so the
        Mathieu maths uses "mass" = m/z with charge 1; the beam itself uses each ion's real
        mass and charge."""
        i = self.ion.values()
        try:
            ref = self.ion_table.first()
        except ValueError:
            ref = IonSpecies()
        i.update(mass=ref.mz, z=1, ref=ref)
        return i

    # ---- keep (a, q) and (U, V) consistent ----
    def _r0_eff(self):
        g = self.geom.values()
        key = (g["r0"], g["ratio"], g["div"])
        return self.cache[key][2] if key in self.cache else g["r0"] * 1e-3

    def _sync_op(self):
        if self._syncing or self.mode.currentIndex() == 1:
            return
        self._syncing = True
        o, r, i = self.op.values(), self.rf.values(), self._ion_values()
        U, V = mathieu.voltages_for(o["a"], o["q"], i["mass"], self._r0_eff(), r["freq"] * 1e6, i["z"])
        self.op.set("U", U); self.op.set("V", V)
        self._syncing = False

    def _sync_aq(self):
        if self._syncing or self.mode.currentIndex() != 1:
            return
        self._syncing = True
        o, r, i = self.op.values(), self.rf.values(), self._ion_values()
        a, q = mathieu.aq(o["U"], o["V"], i["mass"], self._r0_eff(), r["freq"] * 1e6, i["z"])
        self.op.set("a", float(a)); self.op.set("q", float(q))
        self._syncing = False

    # ---- driver ----
    def _driver(self):
        return self.store.get(self.drive.currentText()) if self.drive.currentIndex() > 0 else None

    def _fill_drivers(self):
        cur = self.drive.currentText()
        self.drive.blockSignals(True)
        self.drive.clear(); self.drive.addItem(BUILTIN); self.drive.addItems(self.store.names())
        i = self.drive.findText(cur)
        self.drive.setCurrentIndex(max(i, 0))
        self.drive.blockSignals(False)
        self._drive_changed(keep_wiring=True)

    def _drive_changed(self, *_, keep_wiring=False):
        d = self._driver()
        on = d is not None
        old = {r: cb.currentText() for r, cb in self.rod_box.items()}
        for rod, cb in self.rod_box.items():
            cb.blockSignals(True); cb.clear()
            if on:
                cb.addItems([o.name for o in d.outputs])
            cb.setEnabled(on)
            cb.blockSignals(False)
        if on:
            names = [o.name for o in d.outputs]
            guess = self._guess_wiring(names)
            for rod, cb in self.rod_box.items():
                pick = old[rod] if keep_wiring and old[rod] in names else guess[rod]
                cb.setCurrentIndex(names.index(pick))
        self.mode.blockSignals(True)
        m = self.mode.currentIndex()
        self.mode.clear(); self.mode.addItems(DRIVER_MODES if on else MODES)
        self.mode.setCurrentIndex(min(m, 1) if on else (2 if m == 1 and self._was_driver else min(m, 2)))
        self.mode.blockSignals(False)
        self._was_driver = on
        self.rf.boxes["freq"].setEnabled(not on)
        self.retune_row.setVisible(on and bool(d.tones))
        self._mode_changed()
        self._driver_info()

    _was_driver = False

    @staticmethod
    def _guess_wiring(names):
        """Which output goes on which rod, from the output names."""
        low = [n.lower() for n in names]
        def find(*keys, default=0):
            for k in keys:
                for i, n in enumerate(low):
                    if k in n:
                        return names[i]
            return names[min(default, len(names) - 1)]
        x = find("rods a", "(x)", "rods x", "rf+", default=0)
        y = find("rods b", "(y)", "rods y", "rf-", default=1)
        return {"x+": find("x+", "x1", default=names.index(x)), "x-": find("x-", "x2", default=names.index(x)),
                "y+": find("y+", "y1", default=names.index(y)), "y-": find("y-", "y2", default=names.index(y))}

    def _wiring(self):
        return {rod: cb.currentText() for rod, cb in self.rod_box.items()}

    def _driver_uv(self, d, wiring):
        """Effective U, V of the driver on the rods (for a, q): rods x at +(U + V cos), y at -(...)."""
        ph = {}
        for rod, out in wiring.items():
            o = d.output(out)
            ph[rod] = (o.dc, o.rf * np.exp(1j * np.radians(o.rf_phase)))
        dx = (ph["x+"][0] + ph["x-"][0]) / 2 - (ph["y+"][0] + ph["y-"][0]) / 2
        px = (ph["x+"][1] + ph["x-"][1]) / 2 - (ph["y+"][1] + ph["y-"][1]) / 2
        return dx / 2, abs(px) / 2

    def _excitation(self, d, wiring):
        """'dipolar x', 'dipolar y', 'quadrupolar' or '' from the tone signs on the rods."""
        s = {rod: d.output(out).tone_sign for rod, out in wiring.items()}
        if not d.active_tones() or not any(s.values()):
            return ""
        if s["x+"] != s["x-"]:
            return "dipolar x"
        if s["y+"] != s["y-"]:
            return "dipolar y"
        if s["x+"] != s["y+"]:
            return "quadrupolar"
        return ""

    def tone_targets(self, d, wiring, r0_eff, zc):
        """[(tone, [m/z ejected])]: dipolar at f = f_secular, quadrupolar at f = 2 f_secular."""
        kind = self._excitation(d, wiring)
        U, V = self._driver_uv(d, wiring)
        out = []
        for t in d.active_tones():
            if kind.startswith("dipolar"):
                m = mathieu.resonant_mz(t.freq, U, V, r0_eff, d.rf_freq, zc, kind[-1])
            elif kind == "quadrupolar":
                m = sorted(mathieu.resonant_mz(t.freq / 2, U, V, r0_eff, d.rf_freq, zc, "x")
                           + mathieu.resonant_mz(t.freq / 2, U, V, r0_eff, d.rf_freq, zc, "y"))
            else:
                m = []
            m = sorted({round(x, 1) for x in m})           # x and y coincide when U = 0
            out.append((t, m))
        return kind, out

    def _driver_info(self, *_):
        d = self._driver()
        if d is None:
            self.drive_info.setText("Rods x at +(U + V cos Ωt), rods y at −(U + V cos Ωt). "
                                    "Pick a driver to add excitation tones.")
            return
        w = self._wiring()
        i = self._ion_values()
        U, V = self._driver_uv(d, w)
        a, q = mathieu.aq(U, V, i["mass"], self._r0_eff(), d.rf_freq, i["z"])
        kind, tt = self.tone_targets(d, w, self._r0_eff(), i["z"])
        lines = [f"RF {d.rf_freq / 1e3:g} kHz, U = {U:.2f} V, V = {V:.2f} V → m/z {i['mass']:g}: a = {a:.4f}, q = {q:.4f}"]
        if kind:
            lines.append(f"<b>{kind.capitalize()} excitation</b>, each tone ejects:")
            for k, (t, ms) in enumerate(tt):
                lines.append(f"&nbsp;&nbsp;tone {d.tones.index(t) + 1}: {t.freq / 1e3:g} kHz, {t.amp:g} V → "
                             + (", ".join(f"m/z {m:.0f}" for m in ms) if ms else "no stable m/z"))
        elif d.active_tones():
            lines.append("Tones are not applied to the rods (Tones × is 0 or the same on every rod).")
        self.drive_info.setText("<br>".join(lines))

    def _retune(self):
        d = self._driver()
        if d is None or not d.tones:
            return
        k = self.tone_k.value() - 1
        if k >= len(d.tones):
            self.log(f"{d.name} has only {len(d.tones)} tones.")
            return
        w, i = self._wiring(), self._ion_values()
        kind = self._excitation(d, w) or "quadrupolar"
        U, V = self._driver_uv(d, w)
        fx, fy = mathieu.secular_freqs(U, V, self.tone_mz.value(), self._r0_eff(), d.rf_freq, i["z"])
        f = {"dipolar x": fx, "dipolar y": fy}.get(kind, 2 * fx)
        if not np.isfinite(f):
            self.log(f"m/z {self.tone_mz.value():g} is not stable at these voltages: no secular frequency.")
            return
        d.tones[k].freq = float(f)
        self.log(f"{d.name}: tone {k + 1} set to {f / 1e3:.3f} kHz ({kind}, aims at m/z {self.tone_mz.value():g})")
        self.store.touch()

    def _mode_changed(self):
        m = self.mode.currentIndex()
        drv = self._driver() is not None
        self.dscan.setVisible(drv and m == 1)
        if drv:
            self.op.setVisible(False); self.scan.setVisible(False)
            return
        self.op.setVisible(m in (0, 1)); self.scan.setVisible(m == 2)
        for k in ("a", "q"):
            self.op.boxes[k].setEnabled(m == 0)
        for k in ("U", "V"):
            self.op.boxes[k].setEnabled(m == 1)

    # ---- run ----
    def run(self):
        g, r, o, s = (f.values() for f in (self.geom, self.rf, self.op, self.scan))
        i = self._ion_values()
        try:
            species = [sp for sp in self.ion_table.species() if sp.n > 0]
        except ValueError as e:
            self.log(f"Input error: {e}")
            return
        mode = self.mode.currentIndex()
        key = (g["r0"], g["ratio"], g["div"])
        cache = self.cache
        drv = self._driver()
        drv = drv.copy() if drv is not None else None       # the run keeps its own copy
        wiring, ds = self._wiring(), self.dscan.values()
        gas = dict(self.gas.values(), M=GASES[self.gas_kind.currentText()], name=self.gas_kind.currentText())
        src = None
        if self.picker.linked():
            src = self.picker.data()
            if src is None or src.n == 0:
                self.log(f"Input error: no ions from '{self.picker.source_name()}' yet. Run it with trajectories first "
                         "(Run > Run with the stages before it, F6, runs both).")
                return
            src = (src, self.picker.step.value())
            species = [IonSpecies(s.name, s.mz, abs(int(s.z)), s.K0, 1) for s in src[0].species]
        self._run_gas, self._run_src = gas, src

        def job(progress):
            out = {"mode": mode, "g": g, "r": r, "i": i, "gas": gas}
            r0 = g["r0"] * 1e-3
            if key not in cache:
                t = time.time()
                progress(0.02)
                grid = quadrupole_rods(r0, rod_ratio=g["ratio"], dx=r0 / g["div"], split=True)
                basis = solve_basis(grid)
                static = GridField2D(grid, basis, {"x+": 1.0, "x-": 1.0, "y+": -1.0, "y-": -1.0})
                p = np.array([[0.1 * r0, 0, 0]])
                ratio = static.efield(0, p)[0, 0] / IdealQuadField(r0, 1.0).efield(0, p)[0, 0]
                cache[key] = (grid, basis, r0 / np.sqrt(ratio))
                out["solve_time"] = time.time() - t
            grid, basis, r0_eff = cache[key]
            out.update(grid=grid, basis=basis, r0_eff=r0_eff)
            freq = r["freq"] * 1e6
            dt = 1 / freq / r["steps"]
            mass, zc = i["mass"], i["z"]
            ref_z = i["ref"].z
            out["species"] = species
            out["src"] = src
            L = g["length"] * 1e-3
            rng = np.random.default_rng()

            def field(U, V):
                phi0 = rf_dc(U, V, freq)
                neg = lambda tt: -phi0(tt)
                return GridField2D(grid, basis, {"x+": phi0, "x-": phi0, "y+": neg, "y-": neg})

            if drv is not None:
                return self._driver_job(out, drv, wiring, grid, basis, r0_eff, i, o, ds, species, L, rng, progress)

            if mode in (0, 1):
                if mode == 0:
                    U, V = mathieu.voltages_for(o["a"], o["q"], mass, r0_eff, freq, zc)
                else:
                    U, V = o["U"], o["V"]
                a, q = mathieu.aq(U, V, mass, r0_eff, freq, zc)
                ions, res = self._fly_species(species, i, field(U, V), dt, L, 1 / freq, 2, rng, progress, gas, src)
                out.update(U=U, V=V, a=float(a), q=float(q), res=res, stable=bool(mathieu.is_stable(a, q)))
                out["aq"] = [tuple(float(x) for x in mathieu.aq(U, V, sp.mz, r0_eff, freq, 1)) for sp in species]
            else:
                U, V = mathieu.voltages_for(s["aq"] * 0.706, 0.706, mass, r0_eff, freq, zc)
                masses = np.linspace(mass * (1 - s["width"] / 100), mass * (1 + s["width"] / 100), int(s["npts"]))
                per = int(s["per"])
                ions = make_beam(per * len(masses), np.repeat(masses, per) * ref_z, ref_z, i["energy"],
                                 i["radius"] * 1e-3, i["angle"], birth_window=1 / freq, rng=rng)
                v_min = np.sqrt(2 * i["energy"] * E_CHARGE / (masses.max() * ref_z * AMU))
                res = Simulation([field(U, V)], z_end=L).run(ions, dt, 1.05 * L / v_min + 1 / freq,
                                                            rng=rng, progress=progress)
                T = (res.status == 1).reshape(len(masses), per).mean(axis=1)
                a_m, q_m = mathieu.aq(U, V, masses, r0_eff, freq, zc)
                out.update(U=U, V=V, masses=masses, T=T, a_m=a_m, q_m=q_m,
                           pred=mathieu.is_stable(a_m, q_m))
            return out

        self.start(job, self._show, "Solving field and flying ions..." if key not in cache else "Flying ions...")

    def _fly_species(self, species, i, fld, dt, L, birth, record_every, rng, progress, gas=None, src=None):
        """Fly every species of the ion table together (ions interleaved, so the recorded
        paths include each species). gas: the Collision gas settings (pressure 0 = vacuum).
        src: (ExitIons, DC step) to start the ions that left the model before instead of a beam."""
        if src is not None:
            idx, ions, notes = src[0].entry_ions(src[1], birth_window=birth, rng=rng)
            if not len(idx):
                raise ValueError("no ion gets into the rods: " + "; ".join(notes))
            zc = ions.charge / E_CHARGE
            mz = ions.mass / AMU / zc
            v_min = max(float(ions.vel[:, 2].min()), 10.0)
        else:
            idx, mz, zc, _ = expand(species)
            ions = make_beam(len(idx), mz * zc, zc, i["energy"], i["radius"] * 1e-3, i["angle"],
                             birth_window=birth, rng=rng)
            v_min = np.sqrt(2 * i["energy"] * E_CHARGE / ((mz * zc).max() * AMU))
            notes = []
        fields, hooks, cell = [fld], [], None
        t_max = 3 * L / v_min + birth
        if gas and gas["p"] > 0:
            k0 = np.array([s.K0 for s in species])[idx]
            cell = HardSphereGas(gas["p"], gas["T"], gas["M"], ccs_from_k0(k0, mz, zc, gas["M"], gas["T"]))
            hooks.append(cell)
            K = k0 * 1e-4 * (760 / gas["p"]) * (gas["T"] / 273.15)          # mobility at this pressure, m^2/Vs
            if gas["ez"] > 0:
                t_max = max(t_max, 3 * L / (K.min() * gas["ez"] * 100))
            else:
                t_max = max(t_max, 2e-3)
            t_max = min(t_max, 20e-3)
        if gas and gas["ez"]:
            fields.append(AxialField(gas["ez"] * 100))
        if gas and gas["tmax"] > 0:
            t_max = gas["tmax"] * 1e-3
        steps = t_max / dt
        res = Simulation(fields, z_end=L, hooks=hooks).run(ions, dt, t_max, record_every=record_every * max(1, int(steps // 20000)),
                                                           rng=rng, progress=progress)
        res.species = idx
        res.cell, res.notes = cell, notes
        return ions, res

    def _driver_job(self, out, d, wiring, grid, basis, r0_eff, i, o, ds, species, L, rng, progress):
        """Fly with a driver's waveforms on the rods."""
        f_rf = d.rf_freq
        steps = self.rf.values()["steps"]
        fmax = max(d.max_freq(), f_rf)
        dt = 1 / fmax / steps
        volts = {rod: (lambda tt, name=name: float(d.waveform(name, tt))) for rod, name in wiring.items()}
        fld = GridField2D(grid, basis, volts)
        U, V = self._driver_uv(d, wiring)
        zc = i["z"]
        kind, targets = self.tone_targets(d, wiring, r0_eff, zc)
        out.update(driver=d, U=U, V=V, kind=kind, targets=targets, f_rf=f_rf)
        if out["mode"] == 0:
            mass = i["mass"]
            a, q = mathieu.aq(U, V, mass, r0_eff, f_rf, zc)
            ions, res = self._fly_species(species, i, fld, dt, L, 1 / f_rf, max(1, int(2 * fmax / f_rf)), rng, progress,
                                          out["gas"], out["src"])
            out.update(a=float(a), q=float(q), res=res, stable=bool(mathieu.is_stable(a, q)), mode=0)
            out["aq"] = [tuple(float(x) for x in mathieu.aq(U, V, sp.mz, r0_eff, f_rf, 1)) for sp in species]
            return out
        m1, m2 = sorted((ds["m1"], ds["m2"]))
        masses = np.geomspace(m1, m2, int(ds["npts"]))
        hits = [m for _, ms in targets for m in ms if m1 <= m <= m2]
        masses = np.unique(np.concatenate([masses, hits]))       # make sure every notch is sampled
        per = int(ds["per"])
        rz = i["ref"].z
        ions = make_beam(per * len(masses), np.repeat(masses, per) * rz, rz, i["energy"],
                         i["radius"] * 1e-3, i["angle"], birth_window=1 / f_rf, rng=rng)
        v_min = np.sqrt(2 * i["energy"] * E_CHARGE / (masses.max() * rz * AMU))
        res = Simulation([fld], z_end=L).run(ions, dt, 1.05 * L / v_min + 1 / f_rf, rng=rng, progress=progress)
        T = (res.status == 1).reshape(len(masses), per).mean(axis=1)
        a_m, q_m = mathieu.aq(U, V, masses, r0_eff, f_rf, zc)
        out.update(masses=masses, T=T, a_m=a_m, q_m=q_m, pred=mathieu.is_stable(a_m, q_m), mode=2, r0_eff=r0_eff, zc=zc)
        return out

    # ---- display ----
    def _show(self, out):
        if "solve_time" in out:
            self.log(f"Laplace solved on {out['grid'].n}x{out['grid'].n} grid in {out['solve_time']:.1f} s; "
                     f"effective r0 = {out['r0_eff'] * 1e3:.4f} mm")
            self._sync_op()
        self._draw_potential(out)
        if "driver" in out:
            self.log(f"Driver '{out['driver'].name}'" + (f", {out['kind']} excitation with "
                     f"{len(out['targets'])} tones" if out["kind"] else ""))
        if out["mode"] in (0, 1):
            res, species = out["res"], out["species"]
            self.log(f"U = {out['U']:.2f} V, V = {out['V']:.2f} V")
            for d, (a, q) in zip(species_stats(res, species), out["aq"]):
                ok = bool(mathieu.is_stable(a, q))
                self.log(f"  {d['name']} (m/z {d['mz']:g}, {d['z']}+): a = {a:.4f}, q = {q:.4f} "
                         f"({'Mathieu stable' if ok else 'Mathieu unstable'}); transmitted {d['through']:.0%}, "
                         f"hit rods {d['lost']:.0%}")
            self.last_table = (["ion", "species", "mz", "z", "status(1=through,2=hit)", "t_end_s", "x_m", "y_m", "z_m"],
                               [[k, species[res.species[k]].name, species[res.species[k]].mz, species[res.species[k]].z,
                                 int(res.status[k]), res.t_end[k], *res.final_pos[k]] for k in range(len(res.status))])
            gas = out["gas"]
            for note in res.notes:
                self.log("  Entry: " + note)
            if res.cell is not None:
                self.log(f"  Collision gas {gas['name']} at {gas['p']:g} Torr, {gas['T']:g} K"
                         + (f", axial field {gas['ez']:g} V/cm" if gas["ez"] else ""))
                for d in species_stats(res, species):
                    self.log(f"    {d['name']}: " + res.cell.stats.summary(d["mask"]))
                if res.cell.p_max > 0.2:
                    self.log(f"  Note: up to {res.cell.p_max:.0%} chance of a collision in one step; raise Steps per "
                             "RF cycle so steps stay well under one collision")
                if (res.status <= 0).mean() > 0.2:
                    self.log(f"  Note: {(res.status <= 0).mean():.0%} still in the rods when the flight ended; "
                             "add an axial drag field or raise the max flight time")
            self.set_exit(ExitIons.from_result(f"quadrupole run at U = {out['U']:.2f} V, V = {out['V']:.2f} V"
                                               + (f" in {gas['p']:g} Torr {gas['name']}" if gas["p"] > 0 else ""),
                                               res, species, pressure_torr=gas["p"], temp_k=gas["T"]))
            self._draw_traj(out)
            self._draw_stability([(q, a) for a, q in out["aq"]], labels=[sp.name for sp in species])
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
            if "driver" in out:
                self._draw_excitation(out)

    def _draw_potential(self, out):
        grid = out["grid"]
        fig = self.plots.fig("Potential", show=False)
        ax = fig.add_subplot(111)
        pot = GridField2D(grid, out["basis"], {"x+": 1.0, "x-": 1.0, "y+": -1.0, "y-": -1.0}).potential(0)
        m = grid.electrode > 0
        cs = ax.contourf(grid.X * 1e3, grid.Y * 1e3, np.ma.masked_where(m, pot), 41, cmap="RdBu_r")
        ax.contourf(grid.X * 1e3, grid.Y * 1e3, m, levels=[0.5, 1.5], colors="0.45")
        ax.set_aspect("equal"); ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
        ax.set_title("Potential per volt (rods A = +1, B = -1)")
        fig.colorbar(cs, ax=ax, label="V")
        self.plots.draw("Potential")

    def _draw_traj(self, out):
        res, r0, species = out["res"], out["g"]["r0"], out["species"]
        fig = self.plots.fig("Trajectories")
        ax1, ax2 = fig.subplots(2, 1, sharex=True)
        multi = len(species) > 1
        style = lambda k: path_style(res.status[k], res.species[k], len(species)) if multi else (None, "-")
        for k in range(res.traj.shape[1]):
            c, ls = style(k)
            ax1.plot(res.traj[:, k, 2] * 1e3, res.traj[:, k, 0] * 1e3, lw=0.7, color=c, ls=ls)
            ax2.plot(res.traj[:, k, 2] * 1e3, res.traj[:, k, 1] * 1e3, lw=0.7, color=c, ls=ls)
        for ax, lab in ((ax1, "x (mm)"), (ax2, "y (mm)")):
            ax.axhline(r0, color="k", lw=0.6); ax.axhline(-r0, color="k", lw=0.6)
            ax.set_ylim(-1.3 * r0, 1.3 * r0); ax.set_ylabel(lab)
        ax2.set_xlabel("z (mm)")
        if multi:
            species_legend(ax1, species, loc="upper right")
            ax1.set_title(fate_caption(len(species)) + f"; overall transmission {res.transmission:.0%}", fontsize=9)
        else:
            ax1.set_title(f"a = {out['a']:.4f}, q = {out['q']:.4f}, transmission {res.transmission:.0%}")
        self.plots.draw("Trajectories")

        fig = self.plots.fig("Cross-section", show=False)
        ax = fig.add_subplot(111)
        grid = out["grid"]
        ax.contourf(grid.X * 1e3, grid.Y * 1e3, grid.electrode > 0, levels=[0.5, 1.5], colors="0.75")
        for k in range(res.traj.shape[1]):
            c, ls = style(k)
            ax.plot(res.traj[:, k, 0] * 1e3, res.traj[:, k, 1] * 1e3, lw=0.5, color=c, ls=ls)
        lim = 1.6 * r0
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_title("Ion paths seen down the axis")
        self.plots.draw("Cross-section")

    def _draw_stability(self, pts, colour=None, labels=None):
        if not hasattr(self, "_stab_map"):
            self._stab_map = mathieu.stability_map(1.0, 0.3, 240, 160)
        Q, A, S = self._stab_map
        fig = self.plots.fig("Stability", show=False)
        ax = fig.add_subplot(111)
        ax.contourf(Q, A, S, levels=[0.5, 1.5], colors=["#9ecae1"])
        if pts:
            q, a = np.array(pts).T
            if labels is not None and len(labels) > 1:
                for k, (qq, aa) in enumerate(pts):
                    ax.plot(qq, aa, "o", color=species_color(k), ms=8, label=labels[k])
            elif colour is None:
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
        if "driver" in out:
            ax.set_xscale("log")
            for k, (t, ms) in enumerate(out["targets"]):
                for m in ms:
                    if out["masses"].min() <= m <= out["masses"].max():
                        ax.axvline(m, color="C3", lw=0.8, ls="--")
                        ax.text(m, 103, f"{k + 1}", color="C3", ha="center", fontsize=7)
            ax.plot([], [], color="C3", ls="--", lw=0.8, label="tone resonances (numbered)")
        ax.set_xlabel("m/z"); ax.set_ylabel("transmission (%)"); ax.legend(fontsize=8)
        ax.set_title(f"U = {out['U']:.1f} V, V = {out['V']:.1f} V" +
                     (f", driver '{out['driver'].name}'" if "driver" in out else ""))
        self.plots.draw("Mass scan")

    def _draw_excitation(self, out):
        """Secular frequencies against m/z with the tones on top: where they cross, ions are ejected."""
        d, m = out["driver"], np.geomspace(out["masses"].min(), out["masses"].max(), 600)
        fx, fy = mathieu.secular_freqs(out["U"], out["V"], m, out["r0_eff"], out["f_rf"], out["zc"])
        fig = self.plots.fig("Excitation", show=False)
        ax = fig.add_subplot(111)
        ax.plot(m, fx / 1e3, "C0", label="secular f_x")
        ax.plot(m, fy / 1e3, "C1", label="secular f_y")
        if out["kind"] == "quadrupolar":
            ax.plot(m, 2 * fx / 1e3, "C0--", lw=0.8, label="2 f_x (quadrupolar resonance)")
            ax.plot(m, 2 * fy / 1e3, "C1--", lw=0.8, label="2 f_y")
        for k, (t, ms) in enumerate(out["targets"]):
            ax.axhline(t.freq / 1e3, color="C3", lw=0.8)
            ax.text(m[-1], t.freq / 1e3, f" tone {k + 1}", color="C3", va="center", fontsize=7)
        ax.set_xscale("log"); ax.set_xlabel("m/z"); ax.set_ylabel("frequency (kHz)")
        ax.set_title(f"Tones (red) against secular frequencies: crossings are ejected ({out['kind'] or 'no'} excitation)",
                     fontsize=9)
        ax.legend(fontsize=8)
        self.plots.draw("Excitation")
