"""Orthogonal-acceleration TOF tab: pusher, drift, reflectron, detector."""
import numpy as np
from dataclasses import replace
from matplotlib.patches import Rectangle
from PySide6 import QtWidgets

from .. import tof as T
from ..ions import IonSpecies
from .common import ParamForm, SimTab
from .stage_picker import SourcePicker
from .driver_store import get_store
from .ion_table import IonTable, species_color, species_legend

MODES = ["Single shot", "Scan a setting", "Tune reflectron for best resolution"]
MANUAL = "Manual (set below)"
FROM_STAGE = ("e", "de", "sx", "sy", "div", "temp")     # beam settings an earlier stage fills in
FLIGHT = {"Two-stage reflectron": "two-stage", "Single-stage reflectron": "single-stage",
          "Linear (no reflectron)": "none"}
SHAPES = {"Linear ramp": "linear", "RC (exponential)": "rc"}
VOLTS = [("pusher", ("rest", "rest_pull", "push", "pull", "drift")), ("tube", ("v1", "v2"))]


class TofTab(SimTab):
    def __init__(self, parent=None, store=None):
        super().__init__(["Layout", "Potentials", "Arrival times", "Mass spectrum", "Peak width", "Detector", "Scan"],
                         parent)
        self.plots.hint("Arrival times", "Set Run mode to “Single shot” and press Run.")
        self.plots.hint("Mass spectrum", "Set Run mode to “Single shot” and press Run.")
        self.plots.hint("Peak width", "Set Run mode to “Single shot” and press Run:\n"
                                      "shows what sets the peak width of the first ion.")
        self.plots.hint("Detector", "Set Run mode to “Single shot” and press Run.")
        self.plots.hint("Scan", "Set Run mode to “Scan a setting” or “Tune reflectron” and press Run.")
        self.store = store or get_store()

        self.ion_table = IonTable("Ions (one row per species; the first is the reference)",
                                  [IonSpecies("m/z 500", 500, 1, 1.0, 2000), IonSpecies("m/z 1000", 1000, 1, 1.0, 2000),
                                   IonSpecies("m/z 1001", 1001, 1, 1.0, 2000)], columns=("name", "mz", "z", "n"))
        self.polarity = QtWidgets.QComboBox(); self.polarity.addItems(["Positive ions", "Negative ions"])
        self.polarity.setToolTip("Switching flips the sign of every voltage on this tab")
        self.polarity.currentIndexChanged.connect(self._flip)
        self.ion_table.layout().addWidget(self.polarity)

        self.beam = ParamForm("Incoming beam (in the pusher)", [
            ("e", "Beam energy (along z)", 10.0, 0.01, 1e4, 2, 1, "eV"),
            ("de", "Energy spread (1σ)", 0.5, 0.0, 1e3, 2, 0.1, "eV"),
            ("x0", "Beam position from pusher plate", 2.5, 0.0, 1e3, 3, 0.1, "mm"),
            ("sx", "Beam size along push (1σ)", 0.5, 0.0, 100, 3, 0.05, "mm"),
            ("sy", "Beam size across (1σ)", 0.5, 0.0, 100, 3, 0.05, "mm"),
            ("div", "Divergence (1σ)", 0.5, 0.0, 30, 3, 0.1, "°"),
            ("temp", "Extra thermal spread", 0.0, 0.0, 1e5, 0, 50, "K"),
            ("win", "Pusher window length (z)", 20.0, 0.1, 1e3, 2, 1, "mm"),
        ])
        self.beam.setToolTip("The divergence and thermal spread set the turnaround time; the beam size sets "
                             "the spatial spread the source and reflectron must focus")
        # ---- stages: the beam can come from the exit of the model before this one ----
        from .quad_tab import QuadTab
        from .funnel_tab import FunnelTab
        from .device_tab import DeviceTab
        self.picker = SourcePicker(self.beam.layout(), (QuadTab, FunnelTab, DeviceTab), parent=self,
                                   tip="Take the beam and the ion mixture from the ions that left another model "
                                       "(a quadrupole, ion funnel or Device builder run with trajectories)",
                                   step_tip="The DC step from the last electrode of the model before to the pusher: "
                                            "it adds this much axial energy per charge")
        self._last_exit = None
        self.picker.changed.connect(self._apply_src)

        # ---- pusher ----
        self.pusher = ParamForm("Pusher and acceleration", [
            ("gp", "Pusher plate to puller grid", 5.0, 0.1, 1e3, 3, 0.5, "mm"),
            ("ga", "Puller grid to drift grid", 15.0, 0.1, 1e3, 3, 0.5, "mm"),
            ("rest", "Pusher plate DC (beam potential)", 0.0, -1e5, 1e5, 1, 10, "V"),
            ("rest_pull", "Puller grid DC", 0.0, -1e5, 1e5, 1, 10, "V"),
            ("push", "Pusher pulse", 600.0, -1e5, 1e5, 1, 10, "V"),
            ("pull", "Puller pulse", -600.0, -1e5, 1e5, 1, 10, "V"),
            ("drift", "Drift tube", -6000.0, -1e5, 1e5, 1, 100, "V"),
            ("rise", "Pulse rise (10-90 %)", 10.0, 0.0, 1e5, 2, 1, "ns"),
            ("fall", "Pulse fall (10-90 %)", 10.0, 0.0, 1e5, 2, 1, "ns"),
            ("width", "Pulse width", 5.0, 0.001, 1e5, 3, 0.5, "µs"),
            ("rate", "Pulse rate", 10.0, 0.001, 1e4, 3, 1, "kHz"),
        ])
        self.shape = QtWidgets.QComboBox(); self.shape.addItems(list(SHAPES))
        self.pusher.layout().addRow("Pulse edge shape", self.shape)
        self.src = QtWidgets.QComboBox()
        self.src.setToolTip("Take the pulse timing and pusher / puller voltages from a driver on the Drivers tab "
                            "(make one from the “OA-TOF pusher and puller” template)")
        self.out_push, self.out_pull = QtWidgets.QComboBox(), QtWidgets.QComboBox()
        self.pusher.layout().insertRow(0, "Pulse from", self.src)
        self.pusher.layout().insertRow(1, "  pusher plate output", self.out_push)
        self.pusher.layout().insertRow(2, "  puller grid output", self.out_pull)

        # ---- flight tube ----
        self.tube = ParamForm("Flight tube", [
            ("L", "Drift length (drift grid to reflectron)", 600.0, 1, 1e5, 1, 10, "mm"),
            ("d1", "Reflectron stage 1 depth", 20.0, 0.1, 1e4, 2, 1, "mm"),
            ("v1", "Reflectron stage 1 grid", -2000.0, -1e5, 1e5, 1, 10, "V"),
            ("d2", "Reflectron stage 2 depth", 200.0, 0.1, 1e4, 2, 5, "mm"),
            ("v2", "Reflectron back plate", 4975.0, -1e5, 1e5, 1, 10, "V"),
            ("dd", "Detector distance from reflectron", 590.0, 1, 1e5, 1, 10, "mm"),
            ("gt", "Grid transmission (each pass)", 90.0, 1, 100, 1, 1, "%"),
        ])
        self.flight = QtWidgets.QComboBox(); self.flight.addItems(list(FLIGHT))
        self.tube.layout().insertRow(0, "Type", self.flight)
        sug = QtWidgets.QPushButton("Suggest reflectron voltages")
        sug.setToolTip("First-order energy focusing for the reference ion, from the source's space focus")
        sug.clicked.connect(self._suggest)
        self.tube.layout().addRow(sug)
        self.focus = QtWidgets.QLabel(); self.focus.setWordWrap(True)
        self.tube.layout().addRow(self.focus)

        self.det = ParamForm("Detector", [
            ("wz", "Width along z", 40.0, 0.1, 1e3, 1, 1, "mm"),
            ("wy", "Width across", 40.0, 0.1, 1e3, 1, 1, "mm"),
            ("zc", "Centre along z", 0.0, -1e4, 1e4, 2, 1, "mm"),
            ("jit", "Timing jitter (1σ)", 0.2, 0.0, 1e3, 3, 0.05, "ns"),
            ("bin", "Digitizer bin", 0.25, 0.001, 1e3, 3, 0.05, "ns"),
        ])
        self.auto_z = QtWidgets.QCheckBox("Centre on where the reference ion lands"); self.auto_z.setChecked(True)
        self.auto_z.toggled.connect(lambda on: self.det.boxes["zc"].setEnabled(not on))
        self.det.boxes["zc"].setEnabled(False)
        self.det.layout().insertRow(2, self.auto_z)

        self.mode = QtWidgets.QComboBox(); self.mode.addItems(MODES)
        mode_box = QtWidgets.QGroupBox("Run mode")
        QtWidgets.QVBoxLayout(mode_box).addWidget(self.mode)
        self.scan = ParamForm("Scan", [
            ("v0", "From", 4500.0, -1e6, 1e6, 3, 10, ""),
            ("v1", "To", 5300.0, -1e6, 1e6, 3, 10, ""),
            ("np", "Points", 41, 2, 400, 0, 1, ""),
            ("n", "Ions per point", 1500, 100, 1e6, 0, 100, ""),
        ])
        self.scan_param = QtWidgets.QComboBox(); self.scan_param.addItems(list(T.SCAN_PARAMS))
        self.scan.layout().insertRow(0, "Setting", self.scan_param)
        self.scan_param.currentIndexChanged.connect(self._scan_range)
        for w in (self.ion_table, self.beam, self.pusher, self.tube, self.det, mode_box, self.scan):
            self.form_layout.addWidget(w)
        self.form_layout.addStretch(1)

        self.mode.currentIndexChanged.connect(lambda i: self.scan.setVisible(i == 1))
        self.scan.setVisible(False)
        self.run_btn.clicked.connect(self.run)
        for f in (self.beam, self.pusher, self.tube, self.det):
            f.changed.connect(self._draw_live)
        self.flight.currentIndexChanged.connect(self._flight_changed)
        self.shape.currentIndexChanged.connect(self._draw_live)
        self.ion_table.changed.connect(self._draw_live)
        self.src.currentIndexChanged.connect(self._src_changed)
        self.out_push.currentIndexChanged.connect(self._from_driver)
        self.out_pull.currentIndexChanged.connect(self._from_driver)
        self.store.changed.connect(self._fill_drivers)
        self._fill_drivers()
        self._flight_changed()
        self._scan_range()
        self._apply_src()

    # ---------- stages ----------
    def set_sources(self, sources):
        """sources: {model name: SimTab} whose exit ions can fill this beam (the models before this one)."""
        self.picker.set_sources(sources)

    def source(self):
        """The SimTab this beam comes from, or None when it is set by hand."""
        return self.picker.source()

    def source_name(self):
        return self.picker.source_name()

    def _apply_src(self, *_):
        tab = self.source()
        linked = tab is not None
        for k in FROM_STAGE:
            self.beam.boxes[k].setEnabled(not linked)
        self.ion_table.table.setEnabled(not linked)
        for b in self.ion_table.findChildren(QtWidgets.QPushButton):
            b.setEnabled(not linked)
        if not linked:
            return
        name = self.source_name()
        if tab.exit_ions is None:
            self.picker.set_note(f"Run '{name}' with trajectories: the ions that leave it fill this beam and the ion table.")
            return
        try:
            beam, species, notes = tab.exit_ions.tof_beam(self.picker.step.value())
        except ValueError as e:
            self.picker.set_note(f"From '{name}': {e}.")
            return
        for k, v in beam.items():
            self.beam.set(k, v)
        self.ion_table.set_species(species)
        ex = tab.exit_ions
        self.picker.set_note(f"From the {ex.source} of '{name}': {ex.n} ions. " + "; ".join(notes) + ".")
        if tab.exit_ions is not self._last_exit:
            self._last_exit = tab.exit_ions
            self.log(f"Beam updated from '{name}': {beam['e']:.3g} ± {beam['de']:.2g} eV, size {beam['sx']:.3g} × "
                     f"{beam['sy']:.3g} mm, divergence {beam['div']:.3g}°")

    # ---------- settings ----------
    def _species(self):
        sp = [s for s in self.ion_table.species() if s.n > 0]
        if self.polarity.currentIndex() == 1:
            sp = [replace(s, z=-abs(s.z)) for s in sp]
        return sp

    def _model(self):
        b, p, t, d = self.beam.values(), self.pusher.values(), self.tube.values(), self.det.values()
        beam = T.Beam(energy_ev=b["e"], energy_spread_ev=b["de"], x0=b["x0"] * 1e-3, sigma_x=b["sx"] * 1e-3,
                      sigma_y=b["sy"] * 1e-3, divergence_deg=b["div"], window=b["win"] * 1e-3, temperature_k=b["temp"])
        pulse = T.Pulse(rise=p["rise"] * 1e-9, fall=p["fall"] * 1e-9, width=p["width"] * 1e-6, rate=p["rate"] * 1e3,
                        shape=SHAPES[self.shape.currentText()])
        drv = self._driver()
        if drv is not None:
            pulse = T.Pulse.from_driver(drv)
        tof = T.OATof(push_gap=p["gp"] * 1e-3, accel_gap=p["ga"] * 1e-3, v_rest=p["rest"], v_rest_pull=p["rest_pull"],
                      v_push=p["push"], v_pull=p["pull"], v_drift=p["drift"], pulse=pulse,
                      drift_length=t["L"] * 1e-3, reflectron=FLIGHT[self.flight.currentText()],
                      r1_depth=t["d1"] * 1e-3, v_r1=t["v1"], r2_depth=t["d2"] * 1e-3, v_r2=t["v2"],
                      det_distance=t["dd"] * 1e-3, det_z=None if self.auto_z.isChecked() else d["zc"] * 1e-3,
                      det_width_z=d["wz"] * 1e-3, det_width_y=d["wy"] * 1e-3, grid_transmission=t["gt"] / 100,
                      det_jitter=d["jit"] * 1e-9, det_bin=d["bin"] * 1e-9)
        if not 0 < beam.x0 < tof.push_gap:
            raise ValueError("The beam must sit between the pusher plate and the puller grid")
        tof.planes()                                    # raises if the detector is outside the tube
        return tof, beam

    def _flight_changed(self, *_):
        kind = FLIGHT[self.flight.currentText()]
        for k in ("d1", "v1"):
            self.tube.boxes[k].setEnabled(kind == "two-stage")
        for k in ("d2", "v2", "dd"):
            self.tube.boxes[k].setEnabled(kind != "none")
        self._draw_live()

    def _flip(self, *_):
        """Positive <-> negative ions: every voltage changes sign (a driver's own voltages are
        changed on the Drivers tab)."""
        drv = self._driver()
        for form, keys in ((self.pusher, VOLTS[0][1]), (self.tube, VOLTS[1][1])):
            for k in keys:
                if drv is not None and form is self.pusher and k != "drift":
                    continue
                form.blockSignals(True)
                form.set(k, -form.values()[k])
                form.blockSignals(False)
        if drv is not None:
            self.log(f"Flip the pusher and puller pulses of '{drv.name}' on the Drivers tab too.")
        self._draw_live()

    def _scan_range(self, *_):
        key, unit = T.SCAN_PARAMS[self.scan_param.currentText()]
        try:
            tof, _ = self._model()
        except ValueError:
            return
        cur = getattr(tof.pulse, key.split(".")[1]) if key.startswith("pulse.") else getattr(tof, key)
        if cur is None:
            cur = tof.drift_length
        cur /= unit
        span = max(abs(cur) * (0.04 if unit == 1.0 else 0.15), 1.0)
        self.scan.set("v0", cur - span); self.scan.set("v1", cur + span)

    def _suggest(self):
        try:
            tof, beam = self._model()
        except ValueError as e:
            self.log(f"Input error: {e}")
            return
        new, note = T.suggest_reflectron(tof, beam)
        self.tube.blockSignals(True)
        self.tube.set("v1", new.v_r1); self.tube.set("v2", new.v_r2); self.tube.set("d2", new.r2_depth * 1e3)
        self.tube.blockSignals(False)
        self.log("Suggested reflectron: " + note + f"  Stage 1 {new.v_r1:.0f} V, back plate {new.v_r2:.0f} V.")
        self._draw_live()

    # ---------- driver ----------
    def _driver(self):
        return self.store.get(self.src.currentText()) if self.src.currentIndex() > 0 else None

    def _fill_drivers(self):
        cur = self.src.currentText()
        self.src.blockSignals(True)
        self.src.clear(); self.src.addItem(MANUAL)
        self.src.addItems([n for n in self.store.names() if self.store.get(n).has_pulse()])
        self.src.setCurrentIndex(max(self.src.findText(cur), 0))
        self.src.blockSignals(False)
        self._src_changed(keep=True)

    def _src_changed(self, *_, keep=False):
        d = self._driver()
        old = (self.out_push.currentText(), self.out_pull.currentText())
        for cb in (self.out_push, self.out_pull):
            cb.blockSignals(True); cb.clear()
            if d:
                cb.addItems([o.name for o in d.outputs])
            cb.setEnabled(d is not None)
            cb.blockSignals(False)
        if d:
            names = [o.name for o in d.outputs]
            low = [n.lower() for n in names]
            guess = lambda key, k: next((i for i, n in enumerate(low) if key in n), min(k, len(names) - 1))
            for cb, prev, key, k in ((self.out_push, old[0], "push", 0), (self.out_pull, old[1], "pull", 1)):
                cb.blockSignals(True)
                cb.setCurrentIndex(names.index(prev) if keep and prev in names else guess(key, k))
                cb.blockSignals(False)
        for k in ("rest", "rest_pull", "push", "pull", "rise", "fall", "width", "rate"):
            self.pusher.boxes[k].setEnabled(d is None)
        self.shape.setEnabled(d is None)
        self._from_driver()

    def _from_driver(self, *_):
        d = self._driver()
        if d is None:
            self._draw_live()
            return
        po, pl = d.output(self.out_push.currentText()), d.output(self.out_pull.currentText())
        self.pusher.blockSignals(True)
        for k, v in (("rest", po.dc), ("rest_pull", pl.dc), ("push", po.pulse), ("pull", pl.pulse),
                     ("rise", d.pulse_rise * 1e9), ("fall", d.pulse_fall * 1e9), ("width", d.pulse_width * 1e6),
                     ("rate", d.pulse_rate / 1e3)):
            self.pusher.set(k, v)
        self.pusher.blockSignals(False)
        self.shape.setCurrentIndex(list(SHAPES.values()).index(d.pulse_shape) if d.pulse_shape in SHAPES.values() else 0)
        self._draw_live()

    # ---------- live drawings ----------
    def _draw_live(self, *_):
        try:
            tof, beam = self._model()
        except ValueError as e:
            self.focus.setText(f"<span style='color:#b00'>{e}</span>")
            return
        except Exception:
            return
        D = T.wiley_mclaren_focus(tof, beam)
        K = T.nominal_energy(tof, beam)
        self.focus.setText(f"Ions gain {abs(K):.0f} eV per charge. Space focus of the source: "
                           + (f"{D * 1e3:.1f} mm past the drift grid." if np.isfinite(D) and D > 0 else "none (check the pulse voltages)."))
        self._draw_layout(tof, beam, None)
        self._draw_potentials(tof, beam)

    def _draw_layout(self, tof, beam, res):
        fig = self.plots.fig("Layout", show=res is not None)
        a_all, a_src = fig.subplots(1, 2, width_ratios=[1.3, 1])
        pl = tof.planes()
        paths = res.paths() if res is not None else []
        dz = res.det_z if res is not None else (tof.det_z if tof.det_z is not None else self._landing(tof, beam))
        zs = [-beam.window / 2, beam.window / 2, dz - tof.det_width_z / 2, dz + tof.det_width_z / 2]
        z0, z1 = (min(zs) - 5e-3) * 1e3, (max(zs) + 5e-3) * 1e3
        x_src = pl[2][0] + 0.25 * tof.accel_gap               # the source view ends a little past the drift grid
        w = beam.window / 2 * 1e3 + 3
        for ax, src in ((a_all, False), (a_src, True)):
            za, zb = (-w, w) if src else (z0, z1)
            for k, (t, x, z) in enumerate(paths):
                i = res.traj_ids[k]
                c = species_color(int(res.species[i])) if len(res.names) > 1 else \
                    {T.ST_DETECTED: "tab:blue"}.get(int(res.status[i]), "tab:red")
                ax.plot(z * 1e3, x * 1e3, lw=0.6, color=c, alpha=0.8)
            for x, kind, name in pl:
                if src and x > x_src or not src and x < pl[2][0] and kind != "wall":
                    continue
                if kind == "det":
                    ax.plot([(dz - tof.det_width_z / 2) * 1e3, (dz + tof.det_width_z / 2) * 1e3], [x * 1e3] * 2,
                            color="k", lw=4, solid_capstyle="butt")
                    ax.text((dz + tof.det_width_z / 2) * 1e3, x * 1e3, " detector", va="bottom", fontsize=7)
                    continue
                ax.plot([za, zb], [x * 1e3] * 2, color="0.3", lw=1.2 if kind == "wall" else 0.8,
                        ls="-" if kind == "wall" else (0, (2, 2)))
                ax.text(zb, x * 1e3, " " + name, va="center", fontsize=7, color="0.3", clip_on=True)
            ax.add_patch(Rectangle((-beam.window / 2 * 1e3, (beam.x0 - 2 * beam.sigma_x) * 1e3), beam.window * 1e3,
                                   max(4 * beam.sigma_x, 1e-5) * 1e3, color="tab:orange", alpha=0.3, lw=0))
            ax.set_xlim(za, zb + 0.5 * (zb - za))
            ax.set_xlabel("z, along the beam (mm)")
        a_src.set_ylim(-0.05 * x_src * 1e3, x_src * 1e3)
        a_all.set_ylabel("x, flight axis (mm)")
        a_all.set_title("Flight tube" + (f", {len(paths)} ion paths" if paths else ""), fontsize=9)
        a_src.set_title("Source (beam in orange)", fontsize=9)
        if res is not None:
            species_legend(a_all, [IonSpecies(n) for n in res.names], loc="center right")
        else:
            fig.suptitle("Press Run to fly ions", fontsize=9, color="0.4")
        self.plots.draw("Layout")

    def _landing(self, tof, beam):
        """Where the reference ion reaches the detector along z (one ion, no spreads)."""
        try:
            sp = replace(self._species()[0], n=1)
            b = replace(beam, energy_spread_ev=0, sigma_x=0, sigma_y=0, divergence_deg=0, temperature_k=0, window=0)
            r = T.fly(tof.copy(det_z=None), b, [sp], rng=np.random.default_rng(0), n_record=0)
            return float(r.det_z)
        except Exception:
            return 0.0

    def _draw_potentials(self, tof, beam):
        fig = self.plots.fig("Potentials", show=False)
        gs = fig.add_gridspec(2, 2)
        a_all, a_src, a_t = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
        p = tof.pulse
        t_on = p.delay + p._ramp_len(p.rise) + 1e-12
        x_src = tof.planes()[2][0] + 0.25 * tof.accel_gap
        for ax, src in ((a_all, False), (a_src, True)):
            for t, lab, c in ((0.0, "between pulses", "0.5"), (t_on, "pulse on", "tab:blue")):
                x = np.array([q[0] for q in tof.planes()]); v = tof.potentials(t)     # linear between grids
                ax.plot(x * 1e3, v, label=lab, color=c)
            for x, kind, name in tof.planes():
                ax.axvline(x * 1e3, color="0.85", lw=0.6, zorder=0)
            ax.set_xlabel("x along the flight axis (mm)")
        a_src.axvline(beam.x0 * 1e3, color="tab:orange", lw=1, ls=":", label="beam")
        a_src.set_xlim(-0.02 * x_src * 1e3, x_src * 1e3)
        a_all.set_ylabel("V"); a_src.legend(fontsize=7)
        a_all.set_title("Potential along the flight axis (ideal grids)", fontsize=9)
        a_src.set_title("Pusher and acceleration", fontsize=9)
        r = max(p._ramp_len(p.rise), 1e-9)
        tt = np.linspace(p.delay - 0.5 * r, p.delay + 3 * r, 600)
        pull = tof.v_rest if tof.v_rest_pull is None else tof.v_rest_pull
        g = p.level(tt)
        a_t.plot(tt * 1e9, tof.v_rest + tof.v_push * g, label="pusher plate")
        a_t.plot(tt * 1e9, pull + tof.v_pull * g, label="puller grid")
        a_t.set_xlabel("time (ns)"); a_t.set_ylabel("V"); a_t.legend(fontsize=7)
        a_t.set_title(f"Pulse edge: {p.rise * 1e9:g} ns 10-90 % ({p.shape}), width {p.width * 1e6:g} µs, "
                      f"{p.rate / 1e3:g} kHz", fontsize=9)
        self.plots.draw("Potentials")

    # ---------- run ----------
    def run(self):
        tab = self.source()
        if tab is not None and (tab.exit_ions is None or tab.exit_ions.n == 0):
            self.log(f"Input error: no ions from '{self.source_name()}' yet. Run it with trajectories first "
                     "(Run > Run with the stages before it, F6, runs both).")
            return
        try:
            species = self._species()
            tof, beam = self._model()
        except ValueError as e:
            self.log(f"Input error: {e}")
            return
        mode, sc = self.mode.currentIndex(), self.scan.values()
        label = self.scan_param.currentText()

        def job(progress):
            out = {"mode": mode, "tof": tof, "beam": beam, "species": species}
            if mode == 0:
                progress(0.05)
                out["res"] = T.fly(tof, beam, species, rng=np.random.default_rng())
                progress(0.5)
                out["broad"], _ = T.broadening(tof, beam, species)
            elif mode == 1:
                key, unit = T.SCAN_PARAMS[label]
                vals = np.linspace(sc["v0"], sc["v1"], int(sc["np"]))
                R, Tm = T.scan(tof, beam, species, key, vals * unit, n=int(sc["n"]), progress=progress)
                out.update(label=label, vals=vals[:len(R)], R=R, T=Tm)
            else:
                out["before"] = T.resolution(tof, beam, species)
                best, hist = T.tune(tof, beam, species, progress=progress)
                out.update(best=best, hist=hist, after=T.resolution(best, beam, species))
            return out

        self.start(job, self._done_run, f"{MODES[mode]}...")

    def _done_run(self, out):
        mode = out["mode"]
        if mode == 0:
            self._show_shot(out)
        elif mode == 1:
            self._show_scan(out)
        else:
            self._show_tune(out)

    def _show_shot(self, out):
        res, tof, beam, species = out["res"], out["tof"], out["beam"], out["species"]
        stats = res.stats()
        self._draw_layout(tof, beam, res)
        # arrival-time peaks, one panel per species
        fig = self.plots.fig("Arrival times", show=False)
        n = min(len(species), 6)
        axs = np.atleast_1d(fig.subplots(1, n, squeeze=False)[0])
        rng = np.random.default_rng(7)
        for k in range(n):
            ax, s = axs[k], stats[k]
            t, w = res.arrival(k)
            if len(t) == 0:
                ax.text(0.5, 0.5, "none detected", ha="center", transform=ax.transAxes); continue
            tj = t + tof.det_jitter * rng.standard_normal(len(t))
            fw = s["fwhm"] if np.isfinite(s["fwhm"]) and s["fwhm"] > 0 else np.ptp(tj) or 1e-9
            lo, hi = s["t_mean"] - 4 * fw, s["t_mean"] + 4 * fw
            ax.hist((tj - s["t_mean"]) * 1e9, bins=60, range=((lo - s["t_mean"]) * 1e9, (hi - s["t_mean"]) * 1e9),
                    weights=w, color=species_color(k), alpha=0.8)
            ax.set_title(f"{s['name']}\n{s['t_mean'] * 1e6:.4f} µs, R = {s['resolution']:.0f}", fontsize=8)
            ax.set_xlabel("t - mean (ns)", fontsize=8); ax.tick_params(labelsize=7)
        axs[0].set_ylabel("ions")
        self.plots.draw("Arrival times")

        tc, c, mz = res.spectrum()
        fig = self.plots.fig("Mass spectrum", show=False)
        if len(tc):
            t0, kcal = res.calibration()
            groups = _clusters([s["mz"] for s in stats if np.isfinite(s["t_mean"])])[:4]
            gs = fig.add_gridspec(2, max(len(groups), 1))
            ax = fig.add_subplot(gs[0, :])
            ax.plot(mz, c, lw=0.7, color="k")
            ax.set_xlabel("m/z (from t = t0 + k √(m/z))"); ax.set_ylabel(f"counts / {tof.det_bin * 1e9:g} ns")
            ax.set_title(f"Mass spectrum, one push: t0 = {t0 * 1e9:.2f} ns, k = {kcal * 1e6:.5f} µs per √(m/z)",
                         fontsize=9)
            for g, (lo, hi) in enumerate(groups):
                a = fig.add_subplot(gs[1, g])
                fw = max([s["mz"] / s["resolution"] for s in stats
                          if lo <= s["mz"] <= hi and np.isfinite(s["resolution"]) and s["resolution"] > 0] or [0.05])
                m = (mz > lo - 4 * fw) & (mz < hi + 4 * fw)
                a.plot(mz[m], c[m], lw=0.8, color="k")
                for k, s in enumerate(stats):
                    if lo <= s["mz"] <= hi:
                        a.axvline(s["mz"], color=species_color(k), lw=0.8, ls=":")
                a.set_title(f"m/z {lo:g}" + (f" to {hi:g}" if hi > lo else ""), fontsize=8)
                a.tick_params(labelsize=7); a.ticklabel_format(useOffset=False, axis="x")
        self.plots.draw("Mass spectrum")

        fig = self.plots.fig("Peak width", show=False)
        ax = fig.add_subplot(111)
        labels = [b[0] for b in out["broad"]]; vals = [b[1] * 1e9 for b in out["broad"]]
        y = np.arange(len(labels))[::-1]
        ax.barh(y, vals, color=["tab:blue"] * (len(vals) - 1) + ["tab:red"])
        ax.set_yticks(y, labels, fontsize=8)
        for yy, v in zip(y, vals):
            ax.text(v, yy, f" {v:.2f} ns", va="center", fontsize=8)
        ax.set_xlabel("peak FWHM (ns)")
        ax.set_title(f"What sets the peak width of {species[0].name} (each spread alone, then all together)", fontsize=9)
        self.plots.draw("Peak width")

        fig = self.plots.fig("Detector", show=False)
        ax = fig.add_subplot(111)
        for k in range(len(species)):
            m = (res.species == k) & np.isin(res.status, (T.ST_DETECTED, T.ST_MISSED))
            ax.scatter(res.z_hit[m] * 1e3, res.y_hit[m] * 1e3, s=2, color=species_color(k), label=species[k].name)
        ax.add_patch(Rectangle(((res.det_z - tof.det_width_z / 2) * 1e3, -tof.det_width_y / 2 * 1e3),
                               tof.det_width_z * 1e3, tof.det_width_y * 1e3, fill=False, color="k"))
        ax.set_xlabel("z (mm)"); ax.set_ylabel("y (mm)"); ax.set_aspect("equal", adjustable="datalim")
        ax.legend(fontsize=7, markerscale=4)
        ax.set_title("Where ions reach the detector plane (box = detector)", fontsize=9)
        self.plots.draw("Detector")
        self.plots.setCurrentIndex(0)

        t0, kcal = res.calibration()
        period = 1 / tof.pulse.rate
        mmax = ((period - t0) / kcal) ** 2 if np.isfinite(kcal) and kcal > 0 and period > t0 else np.nan
        self.log(f"--- Single shot: {_flight_name(tof)}, ions gain {abs(T.nominal_energy(tof, beam)):.0f} eV per charge ---")
        for s in stats:
            fates = ", ".join(f"{v} {k}" for k, v in s["fates"].items() if v and k != "detected")
            if not s["detected"]:
                self.log(f"{s['name']}: none reached the detector ({fates}). Check the voltage signs and the "
                         "reflectron back plate.")
                continue
            self.log(f"{s['name']}: t = {s['t_mean'] * 1e6:.4f} µs, FWHM {s['fwhm'] * 1e9:.2f} ns, "
                     f"R = {s['resolution']:.0f}; detected {s['detected']} of {s['n']}"
                     + (f" ({fates})" if fates else "")
                     + f"; with grids {s['transmission']:.1%}; duty cycle {s['duty']:.1%}")
        if np.isfinite(mmax):
            self.log(f"At {tof.pulse.rate / 1e3:g} kHz the next push comes after {period * 1e6:.1f} µs: "
                     f"m/z up to about {mmax:.0f} without overlap.")
        rows = [[res.names[int(res.species[i])], res.mz[int(res.species[i])], T.STATUS_NAMES.get(int(res.status[i]), ""),
                 res.t[i], res.weight[i], res.z_hit[i], res.y_hit[i], res.x0[i], res.vx0[i], res.energy[i]]
                for i in range(len(res.status))]
        self.last_table = (["species", "m/z", "fate", "t (s)", "grid weight", "z at detector (m)", "y at detector (m)",
                            "x start (m)", "vx start (m/s)", "drift energy (eV)"], rows)

    def _show_scan(self, out):
        fig = self.plots.fig("Scan")
        ax = fig.add_subplot(111)
        ax.plot(out["vals"], out["R"], "o-", color="tab:blue")
        ax.set_xlabel(out["label"]); ax.set_ylabel("resolution (t / 2 FWHM)", color="tab:blue")
        ax2 = ax.twinx()
        ax2.plot(out["vals"], out["T"] * 1e6, "--", color="0.5")
        ax2.set_ylabel("flight time (µs)", color="0.5")
        if np.any(np.isfinite(out["R"])):
            j = int(np.nanargmax(out["R"]))
            ax.axvline(out["vals"][j], color="tab:red", lw=0.8, ls=":")
            ax.set_title(f"{out['species'][0].name}: best R = {out['R'][j]:.0f} at {out['vals'][j]:.4g}", fontsize=9)
            self.log(f"--- Scan of {out['label']}: best resolution {out['R'][j]:.0f} at {out['vals'][j]:.4g} ---")
        self.plots.draw("Scan")
        self.last_table = ([out["label"], "resolution", "flight time (s)"],
                           [[v, r, t] for v, r, t in zip(out["vals"], out["R"], out["T"])])

    def _show_tune(self, out):
        best = out["best"]
        self.tube.blockSignals(True)
        self.tube.set("v1", best.v_r1); self.tube.set("v2", best.v_r2)
        self.tube.blockSignals(False)
        self._draw_potentials(best, out["beam"])
        fig = self.plots.fig("Scan")
        ax = fig.add_subplot(111)
        R = [out["before"]] + [h[2] for h in out["hist"]]
        ax.plot(range(len(R)), R, "o-")
        ax.set_xticks(range(len(R)), ["start"] + [f"{'back plate' if h[0] == 'v_r2' else 'stage 1'}\n{h[1]:.0f} V"
                                                   for h in out["hist"]], fontsize=7)
        ax.set_ylabel("resolution")
        ax.set_title(f"Reflectron tuning for {out['species'][0].name}: R {out['before']:.0f} → {out['after']:.0f}",
                     fontsize=9)
        self.plots.draw("Scan")
        self.log(f"--- Tuned reflectron: stage 1 {best.v_r1:.1f} V, back plate {best.v_r2:.1f} V; "
                 f"resolution {out['before']:.0f} → {out['after']:.0f} (values copied into Flight tube) ---")
        self.last_table = (["step", "setting", "value (V)", "resolution"],
                           [[k, h[0], h[1], h[2]] for k, h in enumerate(out["hist"])])


def _clusters(mzs, rel=0.02):
    """Group nearby m/z (isotopes, close masses) into (lo, hi) windows for zoomed views."""
    out = []
    for m in sorted(mzs):
        if out and m - out[-1][1] <= rel * m:
            out[-1][1] = m
        else:
            out.append([m, m])
    return [tuple(g) for g in out]


def _flight_name(tof):
    return {v: k for k, v in FLIGHT.items()}[tof.reflectron]
