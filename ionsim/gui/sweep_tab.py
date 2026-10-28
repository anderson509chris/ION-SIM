"""The Sweeps tab: change voltages and other settings anywhere in a chain of models and see what
comes out of the end.

Pick the last model of a chain (an OA-TOF fed by a quadrupole fed by a funnel, say) and add the
settings to vary: any number in any model's settings column, or any driver value (DC and RF on
an output, RF frequency, TWAVE amplitude and speed, pulse timing, tone amplitudes). Then either

  sweep each setting over a range, one at a time, the others at their nominal values, or
  run a tolerance study: every setting at random within its ± at once, many trials (Monte Carlo).

Each trial runs the models in order, starting at the first one whose settings changed, and keeps
the headline results of every model in the chain (transmission, resolution, arrival time, ...).
Supply ripple slower than the ions' transit is a "Ripple" spread on the DC; timing jitter is a
spread on a pulse delay. The settings go back to their nominal values at the end.
"""
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .. import sweep as SW
from . import settings_io
from .common import SimTab
from .driver_store import get_store

DRIVER_PARAMS = [("rf_freq", "RF frequency", "Hz"), ("tw_amp", "TWAVE amplitude", "V"),
                 ("tw_speed", "TWAVE speed", "m/s"), ("tw_rise", "TWAVE edge time", "s"),
                 ("pulse_rate", "pulse rate", "Hz"), ("pulse_delay", "pulse delay", "s"),
                 ("pulse_width", "pulse width", "s"), ("pulse_rise", "pulse rise", "s"),
                 ("pulse_fall", "pulse fall", "s")]
OUTPUT_PARAMS = [("dc", "DC", "V"), ("rf", "RF amplitude", "V"), ("rf_phase", "RF phase", "deg"),
                 ("pulse", "pulse height", "V"), ("wave_scale", "waveform scale", "")]
TONE_PARAMS = [("amp", "amplitude", "V"), ("freq", "frequency", "Hz"), ("phase", "phase", "deg")]
MODES = ["Sweep each setting over its range (the others at nominal)",
         "Tolerances: every setting at random within its ± (Monte Carlo)"]
COLS = ["Setting", "Nominal", "From", "To", "±", "Unit", "Spread"]


class Knob:
    """One setting to vary: an input in a model's settings column, or a value of a driver."""

    def __init__(self, kind, owner, key, label):
        self.kind, self.owner, self.key, self.label = kind, owner, key, label

    def to_dict(self):
        return {"kind": self.kind, "owner": self.owner, "key": list(self.key) if self.kind == "driver" else self.key,
                "label": self.label}

    @classmethod
    def from_dict(cls, d):
        return cls(d["kind"], d["owner"], tuple(d["key"]) if d["kind"] == "driver" else d["key"], d["label"])


def driver_knobs(d):
    """The values of a driver worth varying, as Knobs (only the parts it uses)."""
    used_tw = any(o.tw >= 0 for o in d.outputs)
    used_pulse = any(o.pulse for o in d.outputs)
    used_rf = any(o.rf for o in d.outputs)
    out = []
    for attr, lab, unit in DRIVER_PARAMS:
        if attr == "rf_freq" and not (used_rf or d.tones) or attr.startswith("tw_") and not used_tw \
                or attr.startswith("pulse_") and not used_pulse:
            continue
        out.append(Knob("driver", d.name, (attr,), f"{d.name}: {lab}" + (f" ({unit})" if unit else "")))
    for o in d.outputs:
        for attr, lab, unit in OUTPUT_PARAMS:
            if attr == "dc" or attr in ("rf", "rf_phase") and o.rf or attr == "pulse" and o.pulse \
                    or attr == "wave_scale" and o.wave:
                out.append(Knob("driver", d.name, ("out", o.name, attr),
                                f"{d.name} / {o.name}: {lab}" + (f" ({unit})" if unit else "")))
    for k, t in enumerate(d.tones):
        for attr, lab, unit in TONE_PARAMS:
            out.append(Knob("driver", d.name, ("tone", k, attr), f"{d.name}: tone {k + 1} {lab} ({unit})"))
    return out


def _driver_target(store, knob):
    d = store.get(knob.owner)
    if d is None:
        raise KeyError(f"driver '{knob.owner}' is not in the library")
    key = knob.key
    if key[0] == "out":
        o = next((o for o in d.outputs if o.name == key[1]), None)
        if o is None:
            raise KeyError(f"driver '{knob.owner}' has no output '{key[1]}'")
        return o, key[2]
    if key[0] == "tone":
        if int(key[1]) >= len(d.tones):
            raise KeyError(f"driver '{knob.owner}' has no tone {int(key[1]) + 1}")
        return d.tones[int(key[1])], key[2]
    return d, key[0]


class KnobPicker(QtWidgets.QDialog):
    """Pick settings from the chain's models and from the drivers."""

    def __init__(self, models, store, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add settings to vary")
        lay = QtWidgets.QVBoxLayout(self)
        self.filter = QtWidgets.QLineEdit(); self.filter.setPlaceholderText("Filter (e.g. DC, pressure, RF)")
        self.filter.textChanged.connect(self._filter)
        lay.addWidget(self.filter)
        self.tree = QtWidgets.QTreeWidget(); self.tree.setHeaderLabels(["Setting", "Now"])
        self.tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        lay.addWidget(self.tree, 1)
        for m in models:
            top = QtWidgets.QTreeWidgetItem([m.name, ""])
            for key, w in settings_io.inputs(m.ctl.scroll.widget()).items():
                if not isinstance(w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                    continue
                it = QtWidgets.QTreeWidgetItem([key.strip(" /"), f"{w.value():g}{w.suffix()}"])
                it.setData(0, QtCore.Qt.UserRole, Knob("input", m.name, key, f"{m.name}: {key.strip(' /')}"))
                if not w.isEnabled():
                    it.setForeground(0, QtGui.QBrush(QtGui.QColor("gray")))
                    it.setToolTip(0, "Not used in the model's present mode")
                top.addChild(it)
            self.tree.addTopLevelItem(top)
        drv = QtWidgets.QTreeWidgetItem(["Drivers", ""])
        for name in store.names():
            d = store.get(name)
            node = QtWidgets.QTreeWidgetItem([name, ""])
            for k in driver_knobs(d):
                obj, attr = _driver_target(store, k)
                it = QtWidgets.QTreeWidgetItem([k.label.split(": ", 1)[-1] if " / " not in k.label else
                                                k.label.split(f"{name} / ", 1)[-1], f"{getattr(obj, attr):g}"])
                it.setData(0, QtCore.Qt.UserRole, k)
                node.addChild(it)
            drv.addChild(node)
        self.tree.addTopLevelItem(drv)
        self.tree.expandAll()
        self.tree.resizeColumnToContents(0)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.button(QtWidgets.QDialogButtonBox.Ok).setText("Add")
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.resize(640, 560)

    def _filter(self, text):
        text = text.lower().strip()

        def walk(it):
            vis = False
            for i in range(it.childCount()):
                vis |= walk(it.child(i))
            knob = it.data(0, QtCore.Qt.UserRole)
            show = vis or (knob is not None and text in knob.label.lower())
            it.setHidden(not show and bool(text))
            return show
        for i in range(self.tree.topLevelItemCount()):
            walk(self.tree.topLevelItem(i))

    def knobs(self):
        return [it.data(0, QtCore.Qt.UserRole) for it in self.tree.selectedItems()
                if it.data(0, QtCore.Qt.UserRole) is not None]


class SweepTab(SimTab):
    PLOTS = ["Result vs setting", "Spread", "Sensitivity"]

    def __init__(self, ws, parent=None):
        super().__init__(self.PLOTS, parent)
        self.ws = ws
        self.store = get_store()
        self.knobs = []
        self.data = None                  # the last study: {"mode", "knobs", "rows": [(settings, results)], ...}
        self._busy = False

        box = QtWidgets.QGroupBox("1. Chain to run")
        f = QtWidgets.QFormLayout(box)
        self.target = QtWidgets.QComboBox()
        self.target.setToolTip("The last model of the chain: it runs with every model it takes its ions from")
        self.target.currentIndexChanged.connect(self._show_chain)
        self.target.activated.connect(lambda _: setattr(self, "_picked", True))
        self._picked = False               # the user chose the last model (else: the end of the longest chain)
        self.chain_label = QtWidgets.QLabel(""); self.chain_label.setWordWrap(True)
        f.addRow("Last model", self.target); f.addRow(self.chain_label)
        self.form_layout.addWidget(box)

        box = QtWidgets.QGroupBox("2. Settings to vary")
        v = QtWidgets.QVBoxLayout(box)
        self.table = QtWidgets.QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(180)
        self.table.setToolTip("From and To are the sweep's range. ± and Unit are the tolerance; Spread is how "
                              "trials fall inside it.")
        v.addWidget(self.table)
        h = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Add…"); b.clicked.connect(self.add_dialog)
        b2 = QtWidgets.QPushButton("Remove"); b2.clicked.connect(self.remove_selected)
        b3 = QtWidgets.QPushButton("Read nominal"); b3.clicked.connect(self.read_nominal)
        b3.setToolTip("Take each setting's value now as its nominal (and centre the ranges on it)")
        for w in (b, b2, b3):
            h.addWidget(w)
        v.addLayout(h)
        self.form_layout.addWidget(box)

        box = QtWidgets.QGroupBox("3. Study")
        f = QtWidgets.QFormLayout(box)
        self.mode = QtWidgets.QComboBox(); self.mode.addItems(MODES)
        self.points = QtWidgets.QSpinBox(); self.points.setRange(2, 500); self.points.setValue(11)
        self.trials = QtWidgets.QSpinBox(); self.trials.setRange(5, 20000); self.trials.setValue(50)
        self.seed = QtWidgets.QSpinBox(); self.seed.setRange(0, 1_000_000)
        self.seed.setToolTip("The same seed gives the same random trials")
        self.renominal = QtWidgets.QCheckBox("Run the nominal again at the end")
        self.renominal.setChecked(True)
        self.renominal.setToolTip("So each model's own tab shows its nominal results afterwards")
        f.addRow("Study", self.mode); f.addRow("Points per sweep", self.points)
        self.same = QtWidgets.QCheckBox("Same random ions in every run")
        self.same.setChecked(True)
        self.same.setToolTip("Every run starts its ions from the same random numbers, so the differences between "
                             "runs come from the settings and not from sampling noise")
        f.addRow("Trials", self.trials); f.addRow("Random seed", self.seed); f.addRow(self.same); f.addRow(self.renominal)
        note = QtWidgets.QLabel("Supply ripple slower than the ions' transit: a Ripple spread on the DC (± = the "
                                "ripple's peak). Timing jitter: a Gaussian spread on the pulse delay.")
        note.setWordWrap(True)
        f.addRow(note)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.form_layout.addWidget(box)
        self.form_layout.addStretch(1)

        self.metric = QtWidgets.QComboBox(); self.metric.setMinimumContentsLength(40)
        self.metric.currentIndexChanged.connect(lambda _: self._draw())
        for name in self.PLOTS:
            row = QtWidgets.QWidget(); hl = QtWidgets.QHBoxLayout(row); hl.setContentsMargins(4, 0, 4, 0)
            hl.addWidget(QtWidgets.QLabel("Result"))
            if name == self.PLOTS[0]:
                hl.addWidget(self.metric, 1)
            else:
                lab = QtWidgets.QLabel(""); lab.setObjectName("metric_echo"); hl.addWidget(lab, 1)
            self.plots.add_controls(name, row)
        self.plots.hint("Result vs setting", "Add settings, pick a study and press Run.\nEach point is a run of "
                                             "the whole chain.")
        self.plots.hint("Spread", "A tolerance study fills this: how each result spreads over the trials.")
        self.plots.hint("Sensitivity", "How much each setting moves the result: from the slope at nominal "
                                       "(sweep) or a fit over the trials (tolerances).")

        self.run_btn.clicked.connect(self.run)
        self.stop_btn.clicked.disconnect()
        self.stop_btn.clicked.connect(self.stop)
        self._mode_changed()

    # ---------- models and settings ----------
    def models_changed(self, renamed=None):
        """The workspace's models changed (opened, closed, linked or renamed)."""
        if renamed:
            for k in self.knobs:
                if k.kind == "input" and k.owner == renamed[0]:
                    k.owner = renamed[1]
                    k.label = f"{renamed[1]}: " + k.label.split(": ", 1)[-1]
            for r, k in enumerate(self.knobs):
                if self.table.item(r, 0):
                    self.table.item(r, 0).setText(k.label)
        cur = self.target.currentText()
        if renamed and cur == renamed[0]:
            cur = renamed[1]
        self.target.blockSignals(True)
        self.target.clear()
        names = [m.name for m in self.ws.models]
        self.target.addItems(names)
        if cur in names and self._picked:
            self.target.setCurrentText(cur)
        elif names:                        # the end of the longest chain
            self.target.setCurrentText(max(self.ws.models, key=lambda m: len(self.chain(m))).name)
        self.target.blockSignals(False)
        self._show_chain()

    def chain(self, m=None):
        m = m if m is not None else next((x for x in self.ws.models if x.name == self.target.currentText()), None)
        if m is None:
            return []
        c, x = [m], self.ws.source_of(m)
        while x is not None and x not in c:
            c.insert(0, x); x = self.ws.source_of(x)
        return c

    def _show_chain(self):
        c = self.chain()
        self.chain_label.setText(" → ".join(m.name for m in c) if c else "Open a model first (Home).")

    def _mode_changed(self):
        tol = self.mode.currentIndex() == 1
        self.points.setEnabled(not tol)
        self.trials.setEnabled(tol); self.seed.setEnabled(tol)

    def _model(self, name):
        return next((m for m in self.ws.models if m.name == name), None)

    def _widget(self, knob):
        m = self._model(knob.owner)
        if m is None:
            raise KeyError(f"model '{knob.owner}' is not open")
        w = settings_io.inputs(m.ctl.scroll.widget()).get(knob.key)
        if w is None:
            raise KeyError(f"'{knob.label}' is not in '{knob.owner}' any more")
        return w

    def get(self, knob):
        if knob.kind == "input":
            return float(self._widget(knob).value())
        obj, attr = _driver_target(self.store, knob)
        return float(getattr(obj, attr))

    def set(self, knob, v):
        """Set a value; returns the value the setting took (an input box clamps to its range and rounds)."""
        if knob.kind == "input":
            w = self._widget(knob)
            w.setValue(int(round(v)) if isinstance(w, QtWidgets.QSpinBox) else float(v))
            return float(w.value())
        obj, attr = _driver_target(self.store, knob)
        setattr(obj, attr, float(v))
        return float(v)

    def add_dialog(self):
        c = self.chain()
        if not c:
            QtWidgets.QMessageBox.information(self, "Sweeps", "Open a model first (Home).")
            return
        dlg = KnobPicker(c, self.store, self)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            for k in dlg.knobs():
                self.add_knob(k)

    def add_knob(self, knob, row=None):
        """Add a setting with a range of ±10 % and a tolerance of ±1 % (Gaussian), or the given row values."""
        if any(k.kind == knob.kind and k.owner == knob.owner and k.key == knob.key for k in self.knobs):
            return
        x0 = self.get(knob)
        span = abs(x0) * 0.1 or 1.0
        row = row or {"from": x0 - span, "to": x0 + span, "pm": 1.0, "unit": "%" if x0 else "abs",
                      "dist": SW.DISTRIBUTIONS[0]}
        self.knobs.append(knob)
        r = self.table.rowCount()
        self.table.insertRow(r)
        it = QtWidgets.QTableWidgetItem(knob.label); it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
        it.setToolTip(knob.label)
        self.table.setItem(r, 0, it)
        nom = QtWidgets.QTableWidgetItem(f"{x0:g}"); nom.setFlags(nom.flags() & ~QtCore.Qt.ItemIsEditable)
        self.table.setItem(r, 1, nom)
        for c, key in ((2, "from"), (3, "to"), (4, "pm")):
            self.table.setItem(r, c, QtWidgets.QTableWidgetItem(f"{row[key]:g}"))
        u = QtWidgets.QComboBox(); u.addItems(SW.UNITS); u.setCurrentText(row["unit"])
        d = QtWidgets.QComboBox(); d.addItems(SW.DISTRIBUTIONS); d.setCurrentText(row["dist"])
        self.table.setCellWidget(r, 5, u); self.table.setCellWidget(r, 6, d)
        self.table.resizeColumnsToContents()
        self.table.setColumnWidth(0, min(max(self.table.columnWidth(0), 160), 320))

    def remove_selected(self):
        for r in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(r)
            del self.knobs[r]

    def read_nominal(self):
        for r, k in enumerate(self.knobs):
            try:
                x0 = self.get(k)
            except KeyError as e:
                self.log(str(e)); continue
            old = self.rows()[r]
            half = (old["to"] - old["from"]) / 2
            self.table.item(r, 1).setText(f"{x0:g}")
            self.table.item(r, 2).setText(f"{x0 - half:g}"); self.table.item(r, 3).setText(f"{x0 + half:g}")

    def rows(self):
        out = []
        for r in range(self.table.rowCount()):
            def num(c):
                try:
                    return float(self.table.item(r, c).text())
                except (AttributeError, ValueError):
                    raise ValueError(f"row {r + 1} ({self.knobs[r].label}): '{COLS[c]}' is not a number")
            out.append({"from": num(2), "to": num(3), "pm": num(4), "unit": self.table.cellWidget(r, 5).currentText(),
                        "dist": self.table.cellWidget(r, 6).currentText()})
        return out

    def state(self):
        try:
            rows = self.rows()
        except ValueError:
            rows = []
        return {"target": self.target.currentText(), "mode": self.mode.currentIndex(), "points": self.points.value(),
                "trials": self.trials.value(), "seed": self.seed.value(), "same": self.same.isChecked(),
                "settings": [dict(k.to_dict(), **r) for k, r in zip(self.knobs, rows)]}

    def set_state(self, d):
        """Bring back a saved study. Settings whose model or driver is gone are skipped (returned)."""
        self.table.setRowCount(0); self.knobs = []
        self.models_changed()
        if d.get("target") in [m.name for m in self.ws.models]:
            self.target.setCurrentText(d["target"])
            self._picked = True
        self.mode.setCurrentIndex(int(d.get("mode", 0)))
        self.points.setValue(int(d.get("points", 11))); self.trials.setValue(int(d.get("trials", 50)))
        self.seed.setValue(int(d.get("seed", 0))); self.same.setChecked(bool(d.get("same", True)))
        missing = []
        for s in d.get("settings", []):
            try:
                self.add_knob(Knob.from_dict(s), row=s)
            except (KeyError, ValueError, IndexError):
                missing.append(s.get("label", "?"))
        return missing

    # ---------- running ----------
    def run(self):
        if self._busy:
            return
        try:
            self._start()
        except (KeyError, ValueError) as e:
            self.status.setText(f"Not started: {e}")
            self.log(f"Not started: {e}")

    def _start(self):
        chain = self.chain()
        if not chain:
            raise ValueError("open a model first (Home)")
        if not self.knobs:
            raise ValueError("add the settings to vary (2. Settings to vary)")
        busy = [m.name for m in chain if m.ctl._thread is not None]
        if busy:
            raise ValueError(f"'{busy[0]}' is running; wait for it or stop it")
        names = {m.name for m in chain}
        outside = [k.label for k in self.knobs if k.kind == "input" and k.owner not in names]
        if outside:
            raise ValueError(f"'{outside[0]}' is not in the chain " + " → ".join(m.name for m in chain))
        rows = self.rows()
        nominal = [self.get(k) for k in self.knobs]
        tol = [SW.tol_abs(x0, r["pm"], r["unit"]) for x0, r in zip(nominal, rows)]
        mode = self.mode.currentIndex()
        if mode == 0:
            trials = [{}]
            for i, r in enumerate(rows):
                trials += [{i: x} for x in SW.sweep_values(r["from"], r["to"], self.points.value())]
        else:
            X = SW.tolerance_trials(nominal, tol, [r["dist"] for r in rows], self.trials.value(), self.seed.value())
            trials = [{}] + [dict(enumerate(x)) for x in X]
        if self.renominal.isChecked():
            trials.append(None)               # the nominal again; not part of the results

        self.chain_now = chain
        self.nominal, self.tol, self.rows_now = nominal, tol, rows
        self.trials_left = trials
        self.n_trials = len(trials)
        self.applied = list(nominal)
        self.last_sig = {}
        self.data = {"mode": mode, "knobs": [k.label for k in self.knobs], "nominal": nominal, "tol": tol,
                     "dist": [r["dist"] for r in rows], "settings": [], "results": [], "which": []}
        self._decimals = {}
        for k in self.knobs:                  # fine steps must not be rounded away by a box's display
            if k.kind == "input":
                w = self._widget(k)
                if isinstance(w, QtWidgets.QDoubleSpinBox) and w.decimals() < 9:
                    self._decimals[k] = w.decimals()
                    w.setDecimals(9)
        for m in chain:
            m.ctl.ran.connect(self._model_ran)
        self._busy, self._stopping, self._failures, self._clamped = True, False, 0, set()
        self.same_random = SW.SameRandom(self.seed.value() + 20240) if self.same.isChecked() else None
        if self.same_random:
            self.same_random.on()
        self.run_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        self.bar.setValue(0)
        self.t_start = QtCore.QElapsedTimer(); self.t_start.start()
        self.log(f"{MODES[mode].split(':')[0].split(' (')[0]}: {len(trials)} runs of "
                 + " → ".join(m.name for m in chain))
        QtCore.QTimer.singleShot(0, self._next_trial)

    def stop(self):
        if not self._busy:
            return
        self._stopping = True
        self.status.setText("Stopping...")
        if self._running is not None:
            self._running.ctl.stop()

    _running = None
    same_random = None

    def abort(self):
        """Stop at once and put the settings back (the window is closing)."""
        if self._busy:
            self._stopping = True
            if self._running is not None:
                self._running.ctl.stop()
            self._finish()

    def _sig(self, m):
        """What a model's run depends on among the varied settings: its own and every driver value."""
        return tuple(x for k, x in zip(self.knobs, self.applied) if k.kind == "driver" or k.owner == m.name)

    def _next_trial(self):
        if self._stopping or not self.trials_left:
            return self._finish()
        trial = self.trials_left[0]
        want = list(self.nominal)
        for i, x in (trial or {}).items():
            want[i] = x
        drivers = False
        for i, (k, x) in enumerate(zip(self.knobs, want)):
            if x != self.applied[i] or not self.data["settings"]:
                got = self.set(k, x)
                if k.kind == "input" and abs(got - x) > 1e-9 * max(1.0, abs(x)):
                    self._clamped.add(k.label)
                self.applied[i] = got
                drivers |= k.kind == "driver"
        if drivers:
            self.store.changed.emit()         # the models take the new driver values (not saved to disk)
        start = next((j for j, m in enumerate(self.chain_now) if self.last_sig.get(m.name) != self._sig(m)),
                     len(self.chain_now))
        self.queue = list(self.chain_now[start:])
        self.trial_ok = True
        self._run_next_model()

    def _run_next_model(self):
        if self._stopping:
            return self._finish()
        if not self.queue:
            return self._trial_done()
        m = self._running = self.queue[0]
        if self.same_random:
            self.same_random.restart()
        m.ctl.run_btn.click()
        if m.ctl._thread is None:             # an input error: it says why in its results box
            self._model_failed(m)

    @QtCore.Slot(bool)
    def _model_ran(self, ok):
        m = self._running
        if m is None or self.sender() is not m.ctl:
            return
        if m.ctl._thread is not None:         # it started a further step of its own run
            return
        if not ok:
            return self._model_failed(m)
        self.last_sig[m.name] = self._sig(m)
        self.queue.pop(0)
        QtCore.QTimer.singleShot(0, self._run_next_model)

    def _model_failed(self, m):
        self.trial_ok = False
        self._failures += 1
        i = self.chain_now.index(m)
        for x in self.chain_now[i:]:          # these have to run again next time
            self.last_sig.pop(x.name, None)
        self.queue = []
        if not self.data["settings"]:         # the nominal must run
            self.log(f"The nominal run failed in '{m.name}' (see its results box); stopped.")
            self._stopping = True
            return self._finish()
        self.log(f"Run {self.n_trials - len(self.trials_left) + 1}: '{m.name}' failed (see its results box)")
        QtCore.QTimer.singleShot(0, self._trial_done)

    def _trial_done(self):
        self._running = None
        trial = self.trials_left.pop(0)
        if trial is not None:
            res = {}
            for m in self.chain_now:
                ran = m.name in self.last_sig
                for key, v in (m.ctl.metrics or {}).items():
                    res[f"{m.name}: {key}"] = float(v) if ran else np.nan
            self.data["settings"].append(list(self.applied))
            self.data["results"].append(res)
            self.data["which"].append(next(iter(trial)) if len(trial) == 1 and self.data["mode"] == 0 else -1)
        done = self.n_trials - len(self.trials_left)
        self.bar.setValue(int(100 * done / self.n_trials))
        el = self.t_start.elapsed() / 1000
        left = el / done * (self.n_trials - done) if done else 0
        self.status.setText(f"Run {done} of {self.n_trials}, about {_dur(left)} to go")
        if done % 5 == 0 or not self.trials_left:
            self._fill_metrics(); self._draw()
        QtCore.QTimer.singleShot(0, self._next_trial)

    def _finish(self):
        if self.same_random:
            self.same_random.off()
        for m in self.chain_now:
            try:
                m.ctl.ran.disconnect(self._model_ran)
            except (RuntimeError, TypeError):
                pass
        drivers = False
        for k, x in zip(self.knobs, self.nominal):
            try:
                self.set(k, x)
                drivers |= k.kind == "driver"
            except KeyError:
                pass
        for k, dec in self._decimals.items():
            try:
                w = self._widget(k)
                w.setDecimals(dec)
                w.setValue(self.nominal[self.knobs.index(k)])
            except KeyError:
                pass
        if drivers:
            self.store.changed.emit()
        self._busy, self._running = False, None
        self.run_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        n = len(self.data["results"])
        self.status.setText(("Stopped" if self._stopping else "Done") + f": {n} runs in {_dur(self.t_start.elapsed() / 1000)}"
                            + (f", {self._failures} failed" if self._failures else ""))
        if self._clamped:
            self.log("These settings were held at the end of their input box's range: " + ", ".join(sorted(self._clamped)))
        self.metrics = {}
        self._fill_metrics(); self._draw(); self._report()
        self.ran.emit(bool(n))

    # ---------- results ----------
    def metric_names(self):
        names = []
        for r in (self.data or {}).get("results", []):
            for k in r:
                if k not in names:
                    names.append(k)
        return names

    def _fill_metrics(self):
        cur = self.metric.currentText()
        names = self.metric_names()
        self.metric.blockSignals(True)
        self.metric.clear(); self.metric.addItems(names)
        if cur in names:
            self.metric.setCurrentText(cur)
        else:                                 # the last model's first result: what comes out of the end
            last = self.chain_now[-1].name + ": " if self.chain_now else ""
            self.metric.setCurrentText(next((n for n in names if n.startswith(last)), names[0] if names else ""))
        self.metric.blockSignals(False)
        header = ["run"] + self.data["knobs"] + names
        rows = []
        for i, (s, r) in enumerate(zip(self.data["settings"], self.data["results"])):
            rows.append([i] + [f"{x:.9g}" for x in s] + [("" if not np.isfinite(r.get(n, np.nan)) else f"{r[n]:.6g}")
                                                         for n in names])
        self.last_table = (header, rows)

    def column(self, name):
        return np.array([r.get(name, np.nan) for r in self.data["results"]], float)

    def sensitivity(self, name):
        """(change in the result for each setting's ±, label). Sweep: slope at nominal × ±. Tolerances:
        each setting's share of the spread from a straight-line fit over the trials."""
        d = self.data
        y = self.column(name)
        S = np.array(d["settings"], float)
        if d["mode"] == 0:
            which = np.array(d["which"])
            out = []
            for i in range(len(d["knobs"])):
                sel = (which == i) | (which == -1)
                out.append(SW.local_slope(S[sel, i], y[sel], d["nominal"][i]) * d["tol"][i])
            return np.array(out), "change for each setting's ± (slope at nominal × ±)"
        c, r2 = SW.contributions(S[1:], y[1:])
        return c, f"σ of the result from each setting (straight-line fit, R² = {r2:.2f})"

    def _draw(self):
        name = self.metric.currentText()
        for i in range(self.plots.count()):
            for lab in self.plots.widget(i).findChildren(QtWidgets.QLabel, "metric_echo"):
                lab.setText(name)
        if not self.data or not name or not self.data["results"]:
            return
        d = self.data
        y = self.column(name)
        S = np.array(d["settings"], float)
        n = len(d["knobs"])
        y0 = y[0]

        f = self.plots.fig("Result vs setting", show=False)
        cols = min(n, 3)
        rows_ = int(np.ceil(n / cols))
        for i in range(n):
            ax = f.add_subplot(rows_, cols, i + 1)
            if d["mode"] == 0:
                sel = np.array(d["which"]) == i
                o = np.argsort(S[sel, i])
                ax.plot(S[sel, i][o], y[sel][o], "o-", ms=3, lw=1.2, color="C0")
            else:
                ax.plot(S[1:, i], y[1:], ".", ms=4, alpha=0.6, color="C0")
            ax.plot([d["nominal"][i]], [y0], "s", color="C3", ms=6, label="nominal")
            ax.set_xlabel(_short(d["knobs"][i]), fontsize=8)
            ax.tick_params(labelsize=7)
            if i % cols == 0:
                ax.set_ylabel(_short(name, 40), fontsize=8)
            ax.grid(alpha=0.3)
        self.plots.draw("Result vs setting")

        f = self.plots.fig("Spread", show=False)
        ax = f.add_subplot(111)
        vals = y[1:] if d["mode"] == 1 else y
        ok = vals[np.isfinite(vals)]
        if d["mode"] == 1 and len(ok) > 1:
            s = SW.spread(ok)
            ax.hist(ok, bins=max(8, min(40, len(ok) // 4)), color="C0", alpha=0.8)
            ax.axvline(y0, color="C3", lw=1.5, label=f"nominal {y0:.4g}")
            ax.axvspan(s["p05"], s["p95"], color="0.5", alpha=0.15, label="5 to 95 %")
            ax.set_title(f"mean {s['mean']:.4g}, σ {s['std']:.3g} ({100 * s['std'] / abs(s['mean']) if s['mean'] else 0:.2g} %),"
                         f" {s['n']} trials", fontsize=10)
            ax.set_xlabel(name); ax.set_ylabel("trials"); ax.legend(fontsize=8)
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, "A tolerance study fills this.", ha="center", va="center", color="0.35",
                    transform=ax.transAxes)
        self.plots.draw("Spread")

        f = self.plots.fig("Sensitivity", show=False)
        ax = f.add_subplot(111)
        c, what = self.sensitivity(name)
        order = np.argsort(np.nan_to_num(np.abs(c), nan=-1))
        labels = [_short(d["knobs"][i], 50) for i in order]
        ax.barh(range(n), np.nan_to_num(c[order]), color=["C3" if v < 0 else "C0" for v in np.nan_to_num(c[order])])
        ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0, color="0.3", lw=0.8)
        ax.set_xlabel(what, fontsize=9)
        ax.set_title(_short(name, 60), fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        self.plots.draw("Sensitivity")
        if not self._busy:
            self.plots.setCurrentIndex(1 if d["mode"] == 1 else 0)

    def _report(self):
        d = self.data
        if not d or not d["results"]:
            return
        names = self.metric_names()
        if d["mode"] == 1:
            self.log(f"{'result':48s} {'nominal':>11s} {'mean':>11s} {'σ':>10s} {'5 %':>11s} {'95 %':>11s}")
            for n in names:
                y = self.column(n)
                s = SW.spread(y[1:])
                self.log(f"{_short(n, 48):48s} {y[0]:11.5g} {s['mean']:11.5g} {s['std']:10.3g} {s['p05']:11.5g} {s['p95']:11.5g}")
        else:
            self.log(f"{'result':48s} {'nominal':>11s} " + " ".join(f"{'±' + _short(k, 15):>16s}" for k in d["knobs"]))
            for n in names:
                c, _ = self.sensitivity(n)
                self.log(f"{_short(n, 48):48s} {self.column(n)[0]:11.5g} " + " ".join(f"{v:16.4g}" for v in c))
            self.log("(each column: how much the result moves for that setting's ±, from the slope at nominal)")


def _short(s, n=32):
    return s if len(s) <= n else s[:n - 1] + "…"


def _dur(s):
    return f"{s:.0f} s" if s < 120 else f"{s / 60:.1f} min" if s < 7200 else f"{s / 3600:.1f} h"
