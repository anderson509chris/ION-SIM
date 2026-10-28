"""ESI inlet capillary tab."""
import numpy as np
from PySide6 import QtCore, QtWidgets

from ..inlet import Inlet, InletIon, fly_inlet, DEFAULT_IONS, TORR
from ..ions import IonSpecies
from ..stages import ExitGas
from .common import ParamForm, SimTab
from .flow_movie import FlowMovie


class InletTab(SimTab):
    def __init__(self, parent=None):
        super().__init__(["Flow picture", "Gas flow", "Ion velocity", "Ion energy", "Effective temperature"], parent)
        self.movie = FlowMovie(self.plots.figs["Flow picture"])
        bar = QtWidgets.QWidget(); row = QtWidgets.QHBoxLayout(bar); row.setContentsMargins(0, 0, 0, 0)
        self.play = QtWidgets.QPushButton("Pause"); self.play.setCheckable(True)
        self.play.setToolTip("Stop or restart the moving gas and ions")
        self.play.toggled.connect(lambda on: self.play.setText("Play" if on else "Pause"))
        self.speed = QtWidgets.QComboBox(); self.speed.addItems(["0.25×", "0.5×", "1×", "2×", "4×"])
        self.speed.setCurrentText("1×"); self.speed.setProperty("ionsim_nosave", True)
        row.addWidget(self.play); row.addWidget(QtWidgets.QLabel("Speed")); row.addWidget(self.speed); row.addStretch(1)
        self.plots.add_controls("Flow picture", bar)
        self._timer = QtCore.QTimer(self); self._timer.setInterval(50)
        self._timer.timeout.connect(self._tick)
        self.plots.hint("Flow picture", "Press Run to see the gas and ions move from the spray into the vacuum chamber.")
        self.spray = ParamForm("ESI spray region", [
            ("gap", "Emitter to inlet", 5.0, 0.2, 50, 2, 0.5, "mm"),
            ("tip", "Emitter tip radius", 50, 1, 1000, 0, 5, "µm"),
            ("rel", "Ion release from tip", 1.0, 0.0, 50, 2, 0.1, "mm"),
            ("v", "Spray voltage (emitter − inlet)", 3500, -10000, 10000, 0, 100, "V"),
            ("tatm", "Room temperature", 298, 200, 400, 0, 1, "K"),
            ("patm", "Source pressure", 760, 100, 2000, 0, 10, "Torr"),
        ])
        self.cap = ParamForm("Inlet capillary", [
            ("d", "Inner diameter", 0.50, 0.05, 3, 3, 0.01, "mm"),
            ("L", "Length", 60, 1, 500, 1, 5, "mm"),
            ("T", "Capillary temperature", 275, 20, 500, 0, 5, "°C"),
        ])
        self.vac = ParamForm("First vacuum stage", [
            ("pb", "Pressure", 1.5, 0.01, 100, 3, 0.1, "Torr"),
            ("skim", "Exit to skimmer/lens", 3.0, 0.1, 50, 2, 0.5, "mm"),
            ("vx", "Capillary − skimmer voltage", 30, -500, 500, 1, 5, "V"),
        ])
        self.hf = QtWidgets.QCheckBox("High-field drag correction (recommended)"); self.hf.setChecked(True)
        self.vac.layout().addRow(self.hf)
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Name", "m/z", "z", "K0 (cm²/Vs)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        for i in DEFAULT_IONS:
            self._add_row(i.name, i.mz, i.charge_state, i.K0)
        box = QtWidgets.QGroupBox("Ions (desolvated, released downstream of the tip)")
        v = QtWidgets.QVBoxLayout(box); v.addWidget(self.table)
        row = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("Add"); rem = QtWidgets.QPushButton("Remove")
        add.clicked.connect(lambda: self._add_row(f"ion {self.table.rowCount() + 1}", 1000, 1, 1.0))
        rem.clicked.connect(lambda: self.table.removeRow(max(self.table.currentRow(), 0)) if self.table.rowCount() > 1 else None)
        row.addWidget(add); row.addWidget(rem); v.addLayout(row)
        for w in (self.spray, self.cap, self.vac, box):
            self.form_layout.addWidget(w)
        self.form_layout.addStretch(1)
        self.run_btn.clicked.connect(self.run)
        self.exit_gas = None        # stages.ExitGas of the last run, for an ion funnel after it

    def _add_row(self, name, mz, z, K0):
        r = self.table.rowCount()
        self.table.insertRow(r)
        for c, val in enumerate([name, f"{mz:g}", f"{z:d}", f"{K0:g}"]):
            self.table.setItem(r, c, QtWidgets.QTableWidgetItem(val))

    def _ions(self):
        out = []
        for r in range(self.table.rowCount()):
            cell = lambda c: (self.table.item(r, c).text() if self.table.item(r, c) else "").strip()
            out.append(InletIon(cell(0) or f"ion {r + 1}", float(cell(1)), int(cell(2) or 1), float(cell(3))))
        if not out:
            raise ValueError("Add at least one ion")
        return out

    def run(self):
        try:
            ions = self._ions()
        except Exception as e:
            self.log(f"Input error: {e}")
            return
        s, c, v, hf = self.spray.values(), self.cap.values(), self.vac.values(), self.hf.isChecked()

        def job(progress):
            inl = Inlet(spray_gap=s["gap"] * 1e-3, tip_radius=s["tip"] * 1e-6, release=s["rel"] * 1e-3, v_spray=s["v"],
                        t_atm=s["tatm"], p_atm=s["patm"], diameter=c["d"] * 1e-3, length=c["L"] * 1e-3,
                        t_cap=c["T"] + 273.15, p_back=v["pb"], skim_dist=v["skim"] * 1e-3, v_exit=v["vx"])
            return {"inlet": inl, "res": fly_inlet(inl, ions, high_field=hf, progress=progress)}

        self.start(job, self._show, "Solving gas flow and flying ions...")

    def _show(self, out):
        inl, res = out["inlet"], out["res"]
        L = inl.length * 1e3
        self.log(f"Gas: {inl.flow_slm:.2f} standard L/min, Re = {inl.reynolds:.0f}, "
                 f"entrance Mach {inl.cap['M'][0]:.2f}, exit {inl.cap['P'][-1] / TORR:.0f} Torr at "
                 f"{inl.cap['u'][-1]:.0f} m/s; Mach disk {inl.x_mach_disk * 1e3:.2f} mm after the exit; "
                 f"jet limit speed {inl.u_terminal:.0f} m/s")
        if inl.skim_dist >= inl.x_mach_disk:
            self.log("  Note: the skimmer is beyond the Mach disk, so ions pass through the shock.")
        rows = []
        for d in res:
            i = d["ion"]
            self.log(f"  {i.name}: at skimmer v = {d['v_final']:.0f} m/s, KE = {d['ke_final']:.2f} eV "
                     f"({d['ke_final'] / i.charge_state:.2f} eV per charge), transit {d['t_final'] * 1e6:.0f} µs, "
                     f"peak T_eff {d['teff'].max():.0f} K")
            rows.append([i.name, i.mz, i.charge_state, i.K0, d["v_final"], d["ke_final"], d["t_final"], d["teff"].max()])
        self.last_table = (["ion", "mz", "z", "K0", "v_skimmer_m_s", "KE_eV", "transit_s", "Teff_max_K"], rows)
        self.metrics = {"gas flow (slm)": float(inl.flow_slm)}
        for d in res:
            self.metrics.update({f"{d['ion'].name} energy at skimmer (eV)": float(d["ke_final"]),
                                 f"{d['ion'].name} speed at skimmer (m/s)": float(d["v_final"])})
        self.exit_gas = ExitGas(f"inlet run ({inl.flow_slm:.2f} slm)", inl.flow_slm, inl.p_back, inl.t_atm,
                                [IonSpecies(d["ion"].name, d["ion"].mz, d["ion"].charge_state, d["ion"].K0, 1) for d in res],
                                [float(d["v_final"]) for d in res])
        on = inl.axis["x"] <= inl.x_end
        self.speeds = {"span": (-inl.spray_gap, inl.x_end), "gas": (inl.axis["x"][on], inl.axis["u"][on]),
                       "ions": [(d["ion"].name, d["x"], d["v"]) for d in res]}
        self.exit_changed.emit()

        ax_ = inl.axis
        x = ax_["x"] * 1e3
        fig = self.plots.fig("Gas flow", show=False)
        a1, a2, a3 = fig.subplots(3, 1, sharex=True)
        a1.semilogy(x, ax_["P"] / TORR); a1.set_ylabel("P (Torr)")
        a2.plot(x, ax_["u"]); a2.set_ylabel("gas speed (m/s)")
        a3.plot(x, ax_["T"]); a3.set_ylabel("gas T (K)"); a3.set_xlabel("x (mm), 0 = capillary entrance")
        for a in (a1, a2, a3):
            self._marks(a, inl)
        a1.set_title(f"{inl.flow_slm:.2f} L/min, choked exit, jet to {inl.skim_dist * 1e3:.1f} mm")
        self.plots.draw("Gas flow")

        for name, key, lab in (("Ion velocity", "v", "ion speed (m/s)"),
                               ("Ion energy", "ke", "kinetic energy (eV)"),
                               ("Effective temperature", "teff", "T_eff (K)")):
            fig = self.plots.fig(name, show=False)
            ax = fig.add_subplot(111)
            for d in res:
                ax.plot(d["x"] * 1e3, d[key], label=d["ion"].name)
            if key == "v":
                ax.plot(x, ax_["u"], "k--", lw=1, label="gas")
            if key == "ke":
                ax.set_yscale("log")
            ax.set_xlabel("x (mm), 0 = capillary entrance"); ax.set_ylabel(lab)
            self._marks(ax, inl); ax.legend(fontsize=8)
            self.plots.draw(name)
        self.movie.show(inl, res)
        self.plots.setCurrentIndex(0)
        self._timer.start()

    def _tick(self):
        """Move the flow picture on while it is in sight and playing."""
        if self.play.isChecked() or not self.plots.figs["Flow picture"].canvas.isVisible():
            return
        self.movie.step(0.05 * float(self.speed.currentText().rstrip("×")))

    @staticmethod
    def _marks(ax, inl):
        for xm in (0, inl.length * 1e3):
            ax.axvline(xm, color="0.5", lw=0.8, ls=":")
