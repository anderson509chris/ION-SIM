"""Drivers tab: build the electronics once (outputs, RF, TWAVE, excitation tones) and wire
them to electrodes in the other tabs."""
import json
import numpy as np
from PySide6 import QtCore, QtWidgets

from ..drivers import Driver, Output, Tone, TEMPLATES
from .common import ParamForm, PlotPanel
from .driver_store import get_store

OUT_COLS = ["Output", "DC (V)", "RF 0-pk (V)", "RF phase (°)", "TWAVE phase", "Tones ×"]
TONE_COLS = ["On", "Frequency (kHz)", "Amplitude 0-pk (V)", "Phase (°)"]


class DriversTab(QtWidgets.QWidget):
    def __init__(self, store=None, parent=None):
        super().__init__(parent)
        self.store = store or get_store()
        self.drv = None
        self._loading = False
        self.last_table = None
        self._thread = None

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        QtWidgets.QHBoxLayout(self).addWidget(split)
        left = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(left)
        scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(left)
        scroll.setMinimumWidth(500)
        split.addWidget(scroll)

        # ---- library ----
        box = QtWidgets.QGroupBox("1. Drivers  (saved automatically, used by the Quadrupole and Device builder tabs)")
        g = QtWidgets.QGridLayout(box)
        self.list = QtWidgets.QListWidget(); self.list.setMaximumHeight(130)
        self.list.currentTextChanged.connect(self._select)
        g.addWidget(self.list, 0, 0, 1, 4)
        self.template = QtWidgets.QComboBox(); self.template.addItems(list(TEMPLATES))
        add = QtWidgets.QPushButton("New from template"); add.clicked.connect(self._new)
        g.addWidget(self.template, 1, 0, 1, 3); g.addWidget(add, 1, 3)
        for col, (label, fn) in enumerate((("Duplicate", self._duplicate), ("Rename…", self._rename),
                                           ("Delete", self._delete), ("Import / export…", self._import_export))):
            b = QtWidgets.QPushButton(label); b.clicked.connect(fn); g.addWidget(b, 2, col)
        lay.addWidget(box)

        # ---- settings ----
        self.form = ParamForm("2. RF and TWAVE timing", [
            ("rf", "RF frequency", 1000.0, 1.0, 1e5, 3, 10, "kHz"),
            ("n", "TWAVE phases", 8, 2, 64, 0, 1, ""),
            ("on", "Phases high at once", 4, 1, 63, 0, 1, ""),
            ("amp", "TWAVE amplitude", 20.0, -1000, 1000, 2, 1, "V"),
            ("speed", "TWAVE speed", 200.0, 0.0, 1e5, 2, 10, "m/s"),
            ("pitch", "at electrode pitch", 0.508, 0.001, 100, 4, 0.01, "mm"),
            ("rise", "Edge rise time", 0.0, 0.0, 1e6, 1, 10, "ns"),
        ])
        self.form.changed.connect(self._settings_changed)
        self.timing = QtWidgets.QLabel(); self.timing.setWordWrap(True)
        self.form.layout().addRow(self.timing)
        lay.addWidget(self.form)

        # ---- outputs ----
        box = QtWidgets.QGroupBox("3. Outputs  (TWAVE phase: blank for none; Tones ×: +1, -1 or 0)")
        v = QtWidgets.QVBoxLayout(box)
        self.outs = QtWidgets.QTableWidget(0, len(OUT_COLS)); self.outs.setHorizontalHeaderLabels(OUT_COLS)
        self.outs.verticalHeader().setVisible(False); self.outs.setMinimumHeight(220)
        self.outs.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.outs.itemChanged.connect(self._outs_edited)
        self.outs.itemSelectionChanged.connect(self.plot)
        v.addWidget(self.outs)
        h = QtWidgets.QHBoxLayout()
        for label, fn in (("Add output", self._add_output), ("Remove selected", self._remove_outputs)):
            b = QtWidgets.QPushButton(label); b.clicked.connect(fn); h.addWidget(b)
        v.addLayout(h)
        lay.addWidget(box)

        # ---- tones ----
        box = QtWidgets.QGroupBox("4. Excitation tones  (added to every output with Tones × not 0)")
        v = QtWidgets.QVBoxLayout(box)
        self.tones = QtWidgets.QTableWidget(0, len(TONE_COLS)); self.tones.setHorizontalHeaderLabels(TONE_COLS)
        self.tones.verticalHeader().setVisible(False); self.tones.setMinimumHeight(200)
        self.tones.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.tones.itemChanged.connect(self._tones_edited)
        v.addWidget(self.tones)
        h = QtWidgets.QHBoxLayout()
        for label, fn in (("Add tone", self._add_tone), ("Remove selected", self._remove_tones)):
            b = QtWidgets.QPushButton(label); b.clicked.connect(fn); h.addWidget(b)
        v.addLayout(h)
        lay.addWidget(box)

        box = QtWidgets.QGroupBox("Notes")
        self.notes = QtWidgets.QPlainTextEdit(); self.notes.setMaximumHeight(70)
        self.notes.textChanged.connect(self._notes_changed)
        QtWidgets.QVBoxLayout(box).addWidget(self.notes)
        lay.addWidget(box)
        lay.addStretch(1)

        # ---- plots ----
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        h = QtWidgets.QHBoxLayout()
        h.addWidget(QtWidgets.QLabel("Time window"))
        self.window = QtWidgets.QDoubleSpinBox(); self.window.setRange(0.0, 1e6); self.window.setDecimals(2)
        self.window.setSuffix(" µs"); self.window.setSpecialValueText("auto"); self.window.setValue(0.0)
        self.window.valueChanged.connect(self.plot)
        h.addWidget(self.window); h.addWidget(QtWidgets.QLabel("Plots the selected outputs (all when none are selected)."))
        h.addStretch(1)
        rv.addLayout(h)
        self.plots = PlotPanel(["Waveforms", "TWAVE sequence"])
        rv.addWidget(self.plots)
        split.addWidget(right)
        split.setSizes([560, 800])

        self.store.changed.connect(self._library_changed)
        self._fill_list()

    # ---------- library ----------
    def _fill_list(self, select=None):
        cur = select or (self.drv.name if self.drv else None)
        self.list.blockSignals(True)
        self.list.clear(); self.list.addItems(self.store.names())
        self.list.blockSignals(False)
        names = self.store.names()
        if names:
            self.list.setCurrentRow(names.index(cur) if cur in names else 0)
            self._select(self.list.currentItem().text())
        else:
            self.drv = None

    def _library_changed(self):
        if self.drv is None or self.drv.name not in self.store.names():
            self._fill_list()

    def _new(self):
        make = TEMPLATES[self.template.currentText()]
        drv = make()
        name = self.store.lib.add(drv, replace=False)
        self.store.touch(); self._fill_list(name)

    def _duplicate(self):
        if self.drv:
            name = self.store.lib.add(self.drv.copy(self.drv.name + " copy"), replace=False)
            self.store.touch(); self._fill_list(name)

    def _rename(self):
        if not self.drv:
            return
        new, ok = QtWidgets.QInputDialog.getText(self, "Rename driver", "Name", text=self.drv.name)
        new = new.strip().replace(" / ", " - ")
        if ok and new and new != self.drv.name:
            if new in self.store.names():
                QtWidgets.QMessageBox.warning(self, "Rename", f"There is already a driver called '{new}'.")
                return
            self.store.lib.rename(self.drv.name, new)
            self.store.touch(); self._fill_list(new)

    def _delete(self):
        if not self.drv:
            return
        if QtWidgets.QMessageBox.question(self, "Delete driver", f"Delete '{self.drv.name}' from the library?") \
                == QtWidgets.QMessageBox.Yes:
            self.store.lib.remove(self.drv.name)
            self.drv = None
            self.store.touch(); self._fill_list()

    def _import_export(self):
        menu = QtWidgets.QMenu(self)
        a_imp = menu.addAction("Import drivers from a file…")
        a_exp = menu.addAction("Export this driver…")
        a_all = menu.addAction("Export all drivers…")
        act = menu.exec(self.cursor().pos())
        flt = "Ion Sim drivers (*.iondrv *.json)"
        if act is a_imp:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import drivers", "", flt)
            if path:
                with open(path) as f:
                    data = json.load(f)
                names = []
                for d in data.get("drivers", []):
                    names.append(self.store.lib.add(Driver.from_dict(d), replace=False))
                self.store.touch(); self._fill_list(names[0] if names else None)
        elif act in (a_exp, a_all) and (self.drv or act is a_all):
            default = (self.drv.name if act is a_exp else "drivers") + ".iondrv"
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export drivers", default, flt)
            if path:
                self.store.lib.save(path, [self.drv.name] if act is a_exp else None)

    # ---------- editing ----------
    def _select(self, name):
        self.drv = self.store.get(name)
        if not self.drv:
            return
        self._loading = True
        d = self.drv
        for k, val in (("rf", d.rf_freq / 1e3), ("n", d.tw_phases), ("on", d.tw_on), ("amp", d.tw_amp),
                       ("speed", d.tw_speed), ("pitch", d.tw_pitch * 1e3), ("rise", d.tw_rise * 1e9)):
            self.form.set(k, val)
        self._fill_outputs(); self._fill_tones()
        self.notes.setPlainText(d.notes)
        self._loading = False
        self._timing(); self.plot()

    def _fill_outputs(self):
        self.outs.blockSignals(True)
        self.outs.setRowCount(0)
        for o in self.drv.outputs:
            r = self.outs.rowCount(); self.outs.insertRow(r)
            vals = [o.name, f"{o.dc:g}", f"{o.rf:g}", f"{o.rf_phase:g}", str(o.tw + 1) if o.tw >= 0 else "",
                    f"{o.tone_sign:g}"]
            for c, val in enumerate(vals):
                self.outs.setItem(r, c, QtWidgets.QTableWidgetItem(val))
        self.outs.blockSignals(False)

    def _fill_tones(self):
        self.tones.blockSignals(True)
        self.tones.setRowCount(0)
        for t in self.drv.tones:
            r = self.tones.rowCount(); self.tones.insertRow(r)
            on = QtWidgets.QTableWidgetItem(); on.setFlags(on.flags() | QtCore.Qt.ItemIsUserCheckable)
            on.setCheckState(QtCore.Qt.Checked if t.on else QtCore.Qt.Unchecked)
            self.tones.setItem(r, 0, on)
            for c, val in enumerate([f"{t.freq / 1e3:g}", f"{t.amp:g}", f"{t.phase:g}"], start=1):
                self.tones.setItem(r, c, QtWidgets.QTableWidgetItem(val))
        self.tones.blockSignals(False)

    def _changed(self):
        self._timing()
        self.store.touch()
        self.plot()

    def _settings_changed(self):
        if self._loading or not self.drv:
            return
        v, d = self.form.values(), self.drv
        d.rf_freq, d.tw_phases, d.tw_on, d.tw_amp = v["rf"] * 1e3, int(v["n"]), int(v["on"]), v["amp"]
        d.tw_speed, d.tw_pitch, d.tw_rise = v["speed"], v["pitch"] * 1e-3, v["rise"] * 1e-9
        self._changed()

    def _notes_changed(self):
        if not self._loading and self.drv:
            self.drv.notes = self.notes.toPlainText()
            self.store.touch()

    @staticmethod
    def _num(item, default=0.0):
        try:
            return float((item.text() if item else "").strip() or default)
        except ValueError:
            return default

    def _outs_edited(self, item):
        if self._loading or not self.drv:
            return
        r = item.row()
        o = self.drv.outputs[r]
        cell = lambda c: self.outs.item(r, c)
        name = (cell(0).text() if cell(0) else "").strip().replace(" / ", " - ") or o.name
        if name != o.name and any(x.name == name for x in self.drv.outputs):
            name = o.name
        o.name = name
        o.dc, o.rf, o.rf_phase = self._num(cell(1)), self._num(cell(2)), self._num(cell(3))
        tw = (cell(4).text() if cell(4) else "").strip()
        o.tw = int(float(tw)) - 1 if tw else -1
        o.tone_sign = self._num(cell(5))
        self.outs.blockSignals(True)
        cell(0).setText(o.name)
        cell(4).setText(str(o.tw + 1) if o.tw >= 0 else "")
        self.outs.blockSignals(False)
        self._changed()

    def _tones_edited(self, item):
        if self._loading or not self.drv:
            return
        r = item.row()
        t = self.drv.tones[r]
        t.on = self.tones.item(r, 0).checkState() == QtCore.Qt.Checked
        t.freq = self._num(self.tones.item(r, 1), t.freq / 1e3) * 1e3
        t.amp = self._num(self.tones.item(r, 2), t.amp)
        t.phase = self._num(self.tones.item(r, 3), t.phase)
        self._changed()

    def _add_output(self):
        if self.drv:
            k = len(self.drv.outputs) + 1
            while any(o.name == f"Out {k}" for o in self.drv.outputs):
                k += 1
            self.drv.outputs.append(Output(f"Out {k}"))
            self._fill_outputs(); self._changed()

    def _remove_outputs(self):
        if self.drv:
            rows = sorted({i.row() for i in self.outs.selectedIndexes()}, reverse=True)
            for r in rows:
                del self.drv.outputs[r]
            self._fill_outputs(); self._changed()

    def _add_tone(self):
        if self.drv:
            f = (self.drv.tones[-1].freq * 1.2) if self.drv.tones else 100e3
            self.drv.tones.append(Tone(f, 1.0))
            self._fill_tones(); self._changed()

    def _remove_tones(self):
        if self.drv:
            for r in sorted({i.row() for i in self.tones.selectedIndexes()}, reverse=True):
                del self.drv.tones[r]
            self._fill_tones(); self._changed()

    # ---------- display ----------
    def _timing(self):
        d = self.drv
        if not d:
            return
        if d.has_twave():
            self.timing.setText(f"TWAVE step {d.step_time() * 1e6:.3f} µs, wave frequency {d.tw_frequency() / 1e3:.2f} kHz "
                                f"(speed / pitch). In a device the speed follows from its own pitch.")
        else:
            self.timing.setText("No output carries a TWAVE phase.")

    def _auto_window(self):
        d = self.drv
        spans = [5 / d.rf_freq] if any(o.rf for o in d.outputs) else []
        if d.has_twave() and np.isfinite(d.step_time()):
            spans.append(d.tw_phases * d.step_time())
        if any(o.tone_sign for o in d.outputs) and d.active_tones():
            spans.append(2 / min(t.freq for t in d.active_tones()))
        return max(spans) if spans else 1e-5

    def plot(self):
        d = self.drv
        if not d:
            return
        rows = sorted({i.row() for i in self.outs.selectedIndexes()})
        outs = [d.outputs[r] for r in rows] if rows else d.outputs
        span = self.window.value() * 1e-6 or self._auto_window()
        fmax = max(d.max_freq(), 1 / span)
        n = int(min(max(4000, span * fmax * 40), 200_000))
        t = np.linspace(0, span, n)
        fig = self.plots.fig("Waveforms", show=False)
        ax = fig.add_subplot(111)
        many = len(outs) > 6
        for k, o in enumerate(outs):
            ax.plot(t * 1e6, d.waveform(o, t) + (0 if not many else 0), lw=0.8, label=o.name)
        ax.set_xlabel("time (µs)"); ax.set_ylabel("V")
        ax.set_title(f"{d.name}: {len(outs)} of {len(d.outputs)} outputs", fontsize=10)
        if len(outs) <= 16:
            ax.legend(fontsize=7, ncol=2 if len(outs) > 8 else 1, loc="upper right")
        self.plots.draw("Waveforms")

        fig = self.plots.fig("TWAVE sequence", show=False)
        ax = fig.add_subplot(111)
        if d.has_twave() and np.isfinite(d.step_time()):
            st = d.step_time()
            tt = np.linspace(0, 2 * d.tw_phases * st, 2000)
            spec = d.tw_spec()
            for p in range(d.tw_phases):
                ax.fill_between(tt * 1e6, p, p + 0.8 * spec.level(p, tt), step=None, alpha=0.8)
            ax.set_yticks(np.arange(d.tw_phases) + 0.4, [f"TW{p + 1}" for p in range(d.tw_phases)])
            ax.set_xlabel("time (µs)")
            ax.set_title(f"TWAVE gates: {d.tw_on} of {d.tw_phases} high, step {st * 1e6:.3f} µs, "
                         f"rise {d.tw_rise * 1e9:.0f} ns", fontsize=10)
        else:
            ax.text(0.5, 0.5, "No TWAVE outputs", ha="center", va="center"); ax.set_axis_off()
        self.plots.draw("TWAVE sequence")

    def stop(self):
        pass
