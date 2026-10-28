"""FAIMS tab."""
import numpy as np
from PySide6 import QtWidgets

from ..faims import Faims, FaimsSpecies, cv_scan, cv_theory, dispersion_curves, DEFAULT_SPECIES
from .common import ParamForm, SimTab

MODES = ["CV scan", "Dispersion plot (theory)"]


class FaimsTab(SimTab):
    def __init__(self, parent=None):
        super().__init__(["Waveform", "CV spectrum", "Ion paths", "Alpha", "Dispersion"], parent)
        self.plots.hint("CV spectrum", "Set Run mode to “CV scan” and press Run.")
        self.plots.hint("Ion paths", "Set Run mode to “CV scan” and press Run.")
        self.plots.hint("Dispersion", "Set Run mode to “Dispersion plot (theory)” and press Run.")
        self.cell = ParamForm("Planar cell", [
            ("gap", "Gap", 1.88, 0.05, 20, 3, 0.05, "mm"),
            ("length", "Length", 40.0, 1, 500, 1, 5, "mm"),
            ("vgas", "Mean gas speed", 5.0, 0.1, 200, 2, 0.5, "m/s"),
        ])
        self.profile = QtWidgets.QComboBox(); self.profile.addItems(["parabolic", "plug"])
        self.cell.layout().addRow("Flow profile", self.profile)
        self.wave = ParamForm("Waveform", [
            ("dv", "Dispersion voltage (peak)", 4000, 0, 20000, 0, 100, "V"),
            ("freq", "Frequency", 750, 1, 20000, 0, 50, "kHz"),
            ("duty", "Duty (rectangular)", 0.333, 0.05, 0.5, 3, 0.01, ""),
            ("steps", "Steps per period", 30, 8, 400, 0, 2, ""),
        ])
        self.shape = QtWidgets.QComboBox(); self.shape.addItems(["bisinusoidal", "rectangular"])
        self.wave.layout().insertRow(0, "Shape", self.shape)
        self.gas = ParamForm("Gas", [
            ("p", "Pressure", 760, 1, 5000, 1, 10, "Torr"),
            ("T", "Temperature", 300, 50, 1000, 1, 5, "K"),
        ])
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Name", "K0 (cm²/Vs)", "a2 (Td⁻²)", "a4 (Td⁻⁴)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        for s in DEFAULT_SPECIES:
            self._add_row(s.name, s.K0, s.a2, s.a4)
        sp_box = QtWidgets.QGroupBox("Ion species  (α = a2·(E/N)² + a4·(E/N)⁴)")
        v = QtWidgets.QVBoxLayout(sp_box); v.addWidget(self.table)
        row = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("Add"); rem = QtWidgets.QPushButton("Remove")
        add.clicked.connect(lambda: self._add_row(f"ion {self.table.rowCount() + 1}", 1.0, 0.0, 0.0))
        rem.clicked.connect(lambda: self.table.removeRow(max(self.table.currentRow(), 0)) if self.table.rowCount() > 1 else None)
        row.addWidget(add); row.addWidget(rem); v.addLayout(row)
        self.mode = QtWidgets.QComboBox(); self.mode.addItems(MODES)
        mode_box = QtWidgets.QGroupBox("Run mode")
        QtWidgets.QVBoxLayout(mode_box).addWidget(self.mode)
        self.scan = ParamForm("CV scan", [
            ("cv0", "CV from", -45, -2000, 2000, 1, 1, "V"),
            ("cv1", "CV to", 25, -2000, 2000, 1, 1, "V"),
            ("ncv", "CV points", 36, 3, 400, 0, 1, ""),
            ("per", "Ions per CV point", 40, 5, 5000, 0, 10, ""),
        ])
        self.disp = ParamForm("Dispersion plot", [
            ("dvmax", "Max DV", 6000, 100, 20000, 0, 100, "V"),
        ])
        for w in (self.cell, self.wave, self.gas, sp_box, mode_box, self.scan, self.disp):
            self.form_layout.addWidget(w)
        self.form_layout.addStretch(1)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self._mode_changed()
        self.run_btn.clicked.connect(self.run)
        for f in (self.wave, self.cell, self.gas):
            f.changed.connect(self._preview)
        self.shape.currentIndexChanged.connect(self._preview)
        self.table.itemChanged.connect(self._preview)
        self._preview()

    def _add_row(self, name, K0, a2, a4):
        r = self.table.rowCount()
        self.table.insertRow(r)
        for c, val in enumerate([name, f"{K0:g}", f"{a2:g}", f"{a4:g}"]):
            self.table.setItem(r, c, QtWidgets.QTableWidgetItem(val))

    def _species(self):
        out = []
        for r in range(self.table.rowCount()):
            cell = lambda c: (self.table.item(r, c).text() if self.table.item(r, c) else "").strip()
            out.append(FaimsSpecies(cell(0) or f"ion {r + 1}", float(cell(1)), float(cell(2) or 0), float(cell(3) or 0)))
        if not out:
            raise ValueError("Add at least one species")
        return out

    def _faims(self):
        c, w, g = self.cell.values(), self.wave.values(), self.gas.values()
        return Faims(gap=c["gap"] * 1e-3, length=c["length"] * 1e-3, dv=w["dv"], freq=w["freq"] * 1e3,
                     waveform=self.shape.currentText(), duty=w["duty"], pressure_torr=g["p"], temp_k=g["T"],
                     gas_velocity=c["vgas"], flow_profile=self.profile.currentText())

    def _mode_changed(self):
        m = self.mode.currentIndex()
        self.scan.setVisible(m == 0); self.disp.setVisible(m == 1)

    def _preview(self, *_):
        try:
            f, sp = self._faims(), self._species()
        except Exception:
            return
        fig = self.plots.fig("Waveform", show=False)
        ax = fig.add_subplot(111)
        t = np.linspace(0, 2 / f.freq, 600)
        ax.plot(t * 1e6, f.dv * f.f(t), label="dispersion voltage")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("time (µs)"); ax.set_ylabel("V")
        ax.set_title(f"{f.waveform}: peak {f.dv:.0f} V → {f.dv / f.gap / 1e5:.1f} kV/cm, E/N = {f.ed_td():.1f} Td")
        self.plots.draw("Waveform")
        fig = self.plots.fig("Alpha", show=False)
        ax = fig.add_subplot(111)
        en = np.linspace(0, 1.3 * f.ed_td(), 300)
        for s in sp:
            ax.plot(en, 100 * s.alpha(en), label=f"{s.name}: CV ≈ {cv_theory(f, s):.1f} V")
        ax.axvline(f.ed_td(), color="k", ls=":", lw=1)
        ax.set_xlabel("E/N (Td)"); ax.set_ylabel("α (%)  = K(E)/K(0) − 1"); ax.legend(fontsize=8)
        ax.set_title("Field dependence of mobility (dotted: peak dispersion field)")
        self.plots.draw("Alpha")

    def run(self):
        try:
            f, sp = self._faims(), self._species()
        except Exception as e:
            self.log(f"Input error: {e}")
            return
        mode, s, d = self.mode.currentIndex(), self.scan.values(), self.disp.values()
        steps = int(self.wave.values()["steps"])

        def job(progress):
            if mode == 0:
                cvs = np.linspace(s["cv0"], s["cv1"], int(s["ncv"]))
                return {"mode": 0, "f": f, "sp": sp,
                        "res": cv_scan(f, sp, cvs, int(s["per"]), steps, record=12, progress=progress)}
            dvs = np.linspace(0, d["dvmax"], 41)
            return {"mode": 1, "f": f, "sp": sp, "dvs": dvs, "cv": dispersion_curves(f, sp, dvs)}

        self.start(job, self._show)

    def _show(self, out):
        f, sp = out["f"], out["sp"]
        if out["mode"] == 0:
            r = out["res"]
            fig = self.plots.fig("CV spectrum")
            ax = fig.add_subplot(111)
            for i, s in enumerate(sp):
                l, = ax.plot(r.cv, 100 * r.transmission[i], "o-", ms=3, label=s.name)
                if np.isfinite(r.cv_theory[i]):
                    ax.axvline(r.cv_theory[i], color=l.get_color(), ls=":", lw=1)
                T = r.transmission[i]
                pk = r.cv[np.argmax(T)] if T.max() > 0 else np.nan
                cen = np.sum(T * r.cv) / T.sum() if T.sum() > 0 else np.nan
                self.metrics.update({f"{s.name} CV peak (V)": float(pk), f"{s.name} max transmission (%)": 100 * float(T.max())})
                self.log(f"{s.name}: theory CV {r.cv_theory[i]:.2f} V, simulated peak {pk:.2f} V "
                         f"(centroid {cen:.2f} V), max transmission {T.max():.0%}")
            ax.set_xlabel("compensation voltage (V)"); ax.set_ylabel("transmission (%)")
            ax.set_title(f"DV {f.dv:.0f} V, gap {f.gap * 1e3:.2f} mm (dotted: theory)"); ax.legend(fontsize=8)
            self.plots.draw("CV spectrum")
            self.last_table = (["CV_V"] + [s.name for s in sp],
                               [[r.cv[k]] + list(r.transmission[:, k]) for k in range(len(r.cv))])
            if r.traj:
                fig = self.plots.fig("Ion paths", show=False)
                ax = fig.add_subplot(111)
                xy = r.traj["xy"]
                for k in range(xy.shape[1]):
                    ax.plot(xy[:, k, 0] * 1e3, xy[:, k, 1] * 1e3, lw=0.6)
                ax.axhline(f.gap / 2 * 1e3, color="k"); ax.axhline(-f.gap / 2 * 1e3, color="k")
                ax.set_xlabel("x along cell (mm)"); ax.set_ylabel("y across gap (mm)")
                ax.set_title(f"{sp[0].name} at its theoretical CV ({r.cv_theory[0]:.2f} V)")
                self.plots.draw("Ion paths")
        else:
            fig = self.plots.fig("Dispersion")
            ax = fig.add_subplot(111)
            for i, s in enumerate(sp):
                ax.plot(out["dvs"], out["cv"][i], label=s.name)
            ax.set_xlabel("dispersion voltage (V)"); ax.set_ylabel("compensation voltage (V)")
            ax.axhline(0, color="k", lw=0.5); ax.legend(fontsize=8); ax.set_title("Dispersion plot")
            self.plots.draw("Dispersion")
            self.last_table = (["DV_V"] + [s.name for s in sp],
                               [[out["dvs"][k]] + list(out["cv"][:, k]) for k in range(len(out["dvs"]))])
            self.log("Dispersion plot: CV vs DV from <K(E)E> = 0 for each species")
