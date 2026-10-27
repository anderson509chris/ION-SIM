"""Ion heating controls and plots, shared by the tabs that fly ions through gas."""
import numpy as np
from PySide6 import QtWidgets

from ..heating import HeatingModel, GASES


class HeatingBox(QtWidgets.QGroupBox):
    """Settings for ion heating. model() -> HeatingModel, or None when switched off."""

    def __init__(self, title="Ion heating", parent=None):
        super().__init__(title, parent)
        self.setCheckable(True); self.setChecked(True)
        f = QtWidgets.QFormLayout(self)
        self.gas = QtWidgets.QComboBox(); self.gas.addItems(list(GASES))
        self.neq = QtWidgets.QDoubleSpinBox(); self.neq.setRange(1, 1e5); self.neq.setValue(10); self.neq.setDecimals(0)
        self.neq.setToolTip("How many collisions the ion's internal temperature needs to follow the effective "
                            "temperature: a few for small ions, tens or more for large ones")
        self.frag = QtWidgets.QCheckBox("Estimate fragmentation / unfolding")
        self.ea = QtWidgets.QDoubleSpinBox(); self.ea.setRange(0.01, 20); self.ea.setValue(1.2); self.ea.setDecimals(3)
        self.ea.setSuffix(" eV"); self.ea.setToolTip("Activation energy of the process (from a CIU or thermal fit)")
        self.loga = QtWidgets.QDoubleSpinBox(); self.loga.setRange(0, 30); self.loga.setValue(12); self.loga.setDecimals(1)
        self.loga.setPrefix("10^"); self.loga.setSuffix(" /s"); self.loga.setToolTip("Arrhenius pre-factor A")
        f.addRow("Gas", self.gas); f.addRow("Collisions to equilibrate", self.neq); f.addRow(self.frag)
        f.addRow("Activation energy", self.ea); f.addRow("Pre-factor A", self.loga)
        self.frag.toggled.connect(self._frag)
        self._frag(False)
        self.setToolTip("Tracks each ion's effective temperature (field-driven collisions with the gas) and "
                        "internal temperature along its path")

    def _frag(self, on):
        lay = self.layout()
        for w in (self.ea, self.loga):
            w.setVisible(on); lay.labelForField(w).setVisible(on)

    def model(self):
        if not self.isChecked():
            return None
        return HeatingModel(GASES[self.gas.currentText()], self.neq.value(),
                            self.ea.value() if self.frag.isChecked() else 0.0, self.loga.value())

    def values(self):
        return {"on": self.isChecked(), "gas": self.gas.currentText(), "neq": self.neq.value(),
                "frag": self.frag.isChecked(), "ea": self.ea.value(), "loga": self.loga.value()}

    def set_values(self, d):
        self.setChecked(d.get("on", True))
        i = self.gas.findText(d.get("gas", ""))
        if i >= 0:
            self.gas.setCurrentIndex(i)
        self.neq.setValue(d.get("neq", 10)); self.frag.setChecked(d.get("frag", False))
        self.ea.setValue(d.get("ea", 1.2)); self.loga.setValue(d.get("loga", 12))


def draw_heating(fig, heat, status=None, z_mm=True, title=""):
    """Effective and internal temperature along the ion axis, and per-ion peaks."""
    fig.clear()
    if heat is None:
        ax = fig.add_subplot(111); ax.axis("off")
        ax.text(0.5, 0.5, "Ion heating was off for this run", ha="center", va="center")
        return
    ax1, ax2 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [2, 1]})
    s = 1e3 if z_mm else 1.0
    z = heat.z * s
    ax1.plot(z, heat.teff_z, label="effective T (mean over ions)")
    ax1.plot(z, heat.tint_z, label="internal T (mean)")
    ax1.plot(z, heat.tint_max_z, "--", lw=0.8, label="internal T (hottest ion)")
    ax1.axhline(heat.temp_k, color="0.5", lw=0.6)
    ax1.set_xlabel("z (mm, ion axis)" if z_mm else "z (m)"); ax1.set_ylabel("temperature (K)")
    sec = ax1.secondary_yaxis("right", functions=(lambda t: 1.5 * 8.617333e-5 * t, lambda e: e / (1.5 * 8.617333e-5)))
    sec.set_ylabel("mean collision energy (eV)")
    if heat.model.ea_ev > 0 and np.nansum(heat.rate_z) > 0:
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.22))
        ax3.fill_between(z, 0, heat.rate_z / max(np.nansum(heat.rate_z), 1e-30), color="C3", alpha=0.15, step="mid")
        ax3.set_ylabel("where ions fragment (share)", color="C3"); ax3.set_ylim(bottom=0)
    ax1.legend(fontsize=7, loc="upper left")
    ax1.set_title(title or "Ion heating along the path", fontsize=9)
    m = np.ones(len(heat.tint_max), bool) if status is None else status >= 0
    ax2.hist(heat.tint_max[m], bins=30, color="C1", alpha=0.8, label="peak internal T")
    ax2.hist(heat.teff_mean[m], bins=30, color="C0", alpha=0.5, label="time-averaged T_eff")
    ax2.set_xlabel("K"); ax2.set_ylabel("ions"); ax2.legend(fontsize=7)
    t = "Per ion"
    if heat.model.ea_ev > 0:
        t += f": {np.mean(heat.survival[m]):.1%} intact"
    ax2.set_title(t, fontsize=9)
