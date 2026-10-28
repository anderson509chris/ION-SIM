"""Dialogs that bring a waveform into a driver: a SPICE result (file or a run), a scope
CSV file, or a live capture from a Rigol oscilloscope."""
import os
import numpy as np
from PySide6 import QtCore, QtWidgets

from .. import waveforms as W
from .. import scope as S

SETTINGS = QtCore.QSettings("IonSim", "IonSim")
SPICE_FILTER = "SPICE results (*.raw *.txt *.csv *.dat *.tsv);;All files (*)"
NETLIST_FILTER = "Netlists (*.cir *.net *.sp *.spi *.asc *.txt);;All files (*)"
SCOPE_FILTER = "Scope CSV (*.csv *.txt);;All files (*)"


def _busy(fn, *a, **kw):
    QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
    QtWidgets.QApplication.processEvents()
    try:
        return fn(*a, **kw)
    finally:
        QtWidgets.QApplication.restoreOverrideCursor()


def _fmt_time(s):
    for unit, k in (("s", 1), ("ms", 1e-3), ("µs", 1e-6), ("ns", 1e-9)):
        if abs(s) >= k or unit == "ns":
            return f"{s / k:.4g} {unit}"


class SignalPicker(QtWidgets.QDialog):
    """Pick the signals to keep from a file or capture, how many points, and optionally the
    driver output the first one should drive."""

    def __init__(self, traces, outputs, title, parent=None, trigger=False):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.traces = traces
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(QtWidgets.QLabel("Signals to add to the driver (each becomes a waveform an output can carry):"))
        self.list = QtWidgets.QListWidget()
        voltage_like = [n for n in traces if not n.lower().startswith(("i(", "ix(", "id(", "ib("))]
        for name, tr in traces.items():
            s = tr.stats()
            it = QtWidgets.QListWidgetItem(f"{name}    {s['points']} points, {_fmt_time(s['duration'])}, "
                                           f"{s['vmin']:.4g} to {s['vmax']:.4g}")
            it.setData(QtCore.Qt.UserRole, name)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            pick = name in voltage_like if len(voltage_like) <= 4 else name == voltage_like[-1]
            it.setCheckState(QtCore.Qt.Checked if len(traces) == 1 or pick else QtCore.Qt.Unchecked)
            self.list.addItem(it)
        lay.addWidget(self.list)
        form = QtWidgets.QFormLayout()
        self.points = QtWidgets.QSpinBox(); self.points.setRange(100, 2_000_000); self.points.setSingleStep(1000)
        self.points.setValue(W.MAX_POINTS)
        self.points.setToolTip("Longer records are averaged down to this many points (saved with the driver)")
        form.addRow("Keep at most", self.points)
        self.trigger = QtWidgets.QCheckBox("Start at the trigger (drop the pre-trigger part)")
        self.trigger.setChecked(trigger); self.trigger.setVisible(trigger)
        form.addRow(self.trigger)
        self.repeat = QtWidgets.QCheckBox("Repeat the record (otherwise hold its last value)")
        self.repeat.setChecked(True)
        form.addRow(self.repeat)
        self.output = QtWidgets.QComboBox(); self.output.addItem("(none, assign later)"); self.output.addItems(outputs)
        form.addRow("First checked signal drives output", self.output)
        lay.addLayout(form)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.resize(620, 380)

    def result_traces(self):
        out = []
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.checkState() == QtCore.Qt.Checked:
                tr = self.traces[it.data(QtCore.Qt.UserRole)]
                if self.trigger.isChecked():
                    tr = tr.from_trigger()
                tr = tr.decimated(self.points.value())
                tr.repeat = self.repeat.isChecked()
                out.append(tr)
        return out

    def output_name(self):
        return self.output.currentText() if self.output.currentIndex() > 0 else None


class SpiceRunDialog(QtWidgets.QDialog):
    """Run a netlist through ngspice or LTspice."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Run a SPICE netlist")
        form = QtWidgets.QFormLayout(self)
        self.path = QtWidgets.QLineEdit(SETTINGS.value("spice/netlist", ""))
        b = QtWidgets.QPushButton("Browse…"); b.clicked.connect(self._browse)
        h = QtWidgets.QHBoxLayout(); h.addWidget(self.path); h.addWidget(b)
        form.addRow("Netlist", h)
        self.sim = QtWidgets.QComboBox(); self.sim.addItems(["Automatic", "ngspice", "LTspice"])
        self.sim.setCurrentText(SETTINGS.value("spice/sim", "Automatic"))
        form.addRow("Simulator", self.sim)
        self.exe = QtWidgets.QLineEdit(SETTINGS.value("spice/exe", ""))
        self.exe.setPlaceholderText("found automatically; or the full path to ngspice / LTspice")
        b2 = QtWidgets.QPushButton("Browse…"); b2.clicked.connect(self._browse_exe)
        h2 = QtWidgets.QHBoxLayout(); h2.addWidget(self.exe); h2.addWidget(b2)
        form.addRow("Executable", h2)
        found = [f"{k}: {e}" for k, e in (W.find_simulator("ngspice"), W.find_simulator("ltspice")) if e]
        note = QtWidgets.QLabel("The netlist needs a transient analysis (.tran). "
                                + ("Found " + "; ".join(found) + "." if found else "No simulator found on this computer."))
        note.setWordWrap(True)
        form.addRow(note)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.button(QtWidgets.QDialogButtonBox.Ok).setText("Run")
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        form.addRow(bb)
        self.resize(640, 200)

    def _browse(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Netlist", os.path.dirname(self.path.text()), NETLIST_FILTER)
        if p:
            self.path.setText(p)

    def _browse_exe(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Simulator executable")
        if p:
            self.exe.setText(p)

    def run(self):
        SETTINGS.setValue("spice/netlist", self.path.text()); SETTINGS.setValue("spice/sim", self.sim.currentText())
        SETTINGS.setValue("spice/exe", self.exe.text())
        kind = {"Automatic": "auto", "ngspice": "ngspice", "LTspice": "ltspice"}[self.sim.currentText()]
        return _busy(W.run_spice, self.path.text().strip(), kind, self.exe.text().strip() or None)


class ScopeDialog(QtWidgets.QDialog):
    """Capture channels from a Rigol scope."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Capture from a Rigol oscilloscope")
        form = QtWidgets.QFormLayout(self)
        self.addr = QtWidgets.QComboBox(); self.addr.setEditable(True)
        last = SETTINGS.value("scope/address", "")
        if last:
            self.addr.addItem(last)
        self.addr.setToolTip("The scope's IP address (Utility > IO on the scope; LAN port 5555), or a VISA "
                             "resource such as USB0::0x1AB1::0x04CE::DS1ZA...::INSTR")
        self.addr.lineEdit().setPlaceholderText("192.168.1.50  or  USB0::0x1AB1::...::INSTR")
        b = QtWidgets.QPushButton("Find USB / VISA"); b.clicked.connect(self._find)
        h = QtWidgets.QHBoxLayout(); h.addWidget(self.addr, 1); h.addWidget(b)
        form.addRow("Address", h)
        self.ch = []
        hc = QtWidgets.QHBoxLayout()
        for k in range(4):
            c = QtWidgets.QCheckBox(f"CH{k + 1}"); c.setChecked(k == 0); self.ch.append(c); hc.addWidget(c)
        form.addRow("Channels", hc)
        self.memory = QtWidgets.QCheckBox("Whole acquisition memory (stops the scope), not just the screen")
        self.single = QtWidgets.QCheckBox("Take a single shot first (waits for a trigger)")
        self.trigger = QtWidgets.QCheckBox("Start the drive at the trigger"); self.trigger.setChecked(True)
        for w in (self.memory, self.single, self.trigger):
            form.addRow(w)
        self.status = QtWidgets.QLabel(""); self.status.setWordWrap(True)
        form.addRow(self.status)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.button(QtWidgets.QDialogButtonBox.Ok).setText("Capture")
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        form.addRow(bb)
        self.resize(600, 260)

    def _find(self):
        res = _busy(S.list_resources)
        self.status.setText(f"Found {len(res)} instrument(s)." if res else
                            "No VISA instruments found (install pyvisa and pyvisa-py, or NI-VISA, for USB).")
        for r in res:
            if self.addr.findText(r) < 0:
                self.addr.addItem(r)
        rig = [r for r in res if "0x1AB1" in r.upper() or "RIGOL" in r.upper()]
        if rig:
            self.addr.setCurrentText(rig[0])

    def capture(self):
        addr = self.addr.currentText().strip()
        if not addr:
            raise ValueError("enter the scope's IP address or VISA resource")
        SETTINGS.setValue("scope/address", addr)
        chans = [f"CHAN{k + 1}" for k, c in enumerate(self.ch) if c.isChecked()]
        if not chans:
            raise ValueError("tick at least one channel")

        def go():
            with S.RigolScope(S.connect(addr)) as sc:
                out = {}
                for k, ch in enumerate(chans):
                    tr = sc.capture(ch, memory=self.memory.isChecked(), single=self.single.isChecked() and k == 0)
                    out[tr.name] = tr
                return sc.idn, out
        return _busy(go)


def ask_spice_file(parent):
    p, _ = QtWidgets.QFileDialog.getOpenFileName(parent, "SPICE result", SETTINGS.value("spice/dir", ""), SPICE_FILTER)
    if p:
        SETTINGS.setValue("spice/dir", os.path.dirname(p))
    return p


def ask_scope_file(parent):
    p, _ = QtWidgets.QFileDialog.getOpenFileName(parent, "Scope CSV", SETTINGS.value("scope/dir", ""), SCOPE_FILTER)
    if p:
        SETTINGS.setValue("scope/dir", os.path.dirname(p))
    return p


def scope_file_traces(path):
    src = os.path.basename(path)
    return {nm: W.Trace(nm, t, v, f"Scope file: {src} {nm}") for nm, (t, v) in W.read_table(path).items()}
