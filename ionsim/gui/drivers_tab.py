"""Drivers tab: build the electronics once (outputs, RF, TWAVE, excitation tones, imported
SPICE or scope waveforms) and wire them to electrodes in the other tabs."""
import json
import os
import numpy as np
from PySide6 import QtCore, QtWidgets

from ..drivers import Driver, Output, Tone, TEMPLATES

PULSE_SHAPES = {"Linear ramp": "linear", "RC (exponential)": "rc"}
from .common import ParamForm, PlotPanel
from .driver_store import get_store
from . import waveform_import as WI
from .. import waveforms as W

OUT_COLS = ["Output", "DC (V)", "RF 0-pk (V)", "RF phase (°)", "TWAVE phase", "Tones ×", "Pulse (V)", "Waveform", "Waveform ×"]
WAVE_COLS = ["Waveform", "Points", "Length", "Min (V)", "Max (V)", "Repeat", "Period (µs)", "Delay (µs)", "Source"]
NO_WAVE = "(none)"
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
        box = QtWidgets.QGroupBox("1. Drivers  (saved automatically, used by the Quadrupole, OA-TOF and Device builder tabs)")
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
        self.form = ParamForm("2. RF, TWAVE and pulse timing", [
            ("rf", "RF frequency", 1000.0, 1.0, 1e5, 3, 10, "kHz"),
            ("n", "TWAVE phases", 8, 2, 64, 0, 1, ""),
            ("on", "Phases high at once", 4, 1, 63, 0, 1, ""),
            ("amp", "TWAVE amplitude", 20.0, -1000, 1000, 2, 1, "V"),
            ("speed", "TWAVE speed", 200.0, 0.0, 1e5, 2, 10, "m/s"),
            ("pitch", "at electrode pitch", 0.508, 0.001, 100, 4, 0.01, "mm"),
            ("rise", "Edge rise time", 0.0, 0.0, 1e6, 1, 10, "ns"),
            ("prate", "Pulse rate", 10.0, 0.001, 1e5, 3, 1, "kHz"),
            ("pdelay", "Pulse delay", 0.0, 0.0, 1e6, 3, 0.1, "µs"),
            ("pwidth", "Pulse width", 5.0, 0.0, 1e6, 3, 0.5, "µs"),
            ("prise", "Pulse rise (10-90 %)", 10.0, 0.0, 1e6, 1, 1, "ns"),
            ("pfall", "Pulse fall (10-90 %)", 10.0, 0.0, 1e6, 1, 1, "ns"),
        ])
        self.pshape = QtWidgets.QComboBox(); self.pshape.addItems(list(PULSE_SHAPES))
        self.pshape.setToolTip("Linear ramp, or the exponential edge of an RC-limited pulser")
        self.pshape.currentIndexChanged.connect(self._settings_changed)
        self.form.layout().addRow("Pulse edge shape", self.pshape)
        self.form.changed.connect(self._settings_changed)
        self.timing = QtWidgets.QLabel(); self.timing.setWordWrap(True)
        self.form.layout().addRow(self.timing)
        lay.addWidget(self.form)

        # ---- outputs ----
        box = QtWidgets.QGroupBox("3. Outputs  (TWAVE phase: blank for none; Tones ×: +1, -1 or 0; Pulse: V added while the pulse is high; "
                                  "Waveform: an imported waveform from section 5, times Waveform ×)")
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

        # ---- imported waveforms ----
        box = QtWidgets.QGroupBox("5. Imported waveforms  (a SPICE simulation or a scope capture; pick one per output in section 3)")
        v = QtWidgets.QVBoxLayout(box)
        self.waves = QtWidgets.QTableWidget(0, len(WAVE_COLS)); self.waves.setHorizontalHeaderLabels(WAVE_COLS)
        self.waves.verticalHeader().setVisible(False); self.waves.setMinimumHeight(150)
        self.waves.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.waves.setToolTip("Repeat: the record starts again after its period (blank or 0 = the record length); "
                              "otherwise it holds its last value. Delay shifts it later in time.")
        self.waves.itemChanged.connect(self._waves_edited)
        self.waves.itemSelectionChanged.connect(self._plot_waves)
        v.addWidget(self.waves)
        g = QtWidgets.QGridLayout()
        for k, (label, fn, tip) in enumerate((
                ("SPICE result…", self._import_spice, "An ngspice or LTspice .raw file, or a text export (LTspice "
                                                      "File > Export data as text, ngspice wrdata)"),
                ("Run SPICE netlist…", self._run_spice, "Run a netlist with ngspice or LTspice and take its node voltages"),
                ("Scope CSV file…", self._import_scope_file, "A CSV saved on the scope (Rigol and most others) or any "
                                                             "time / voltage table"),
                ("Capture from Rigol scope…", self._capture_scope, "Read channels straight from a Rigol scope over "
                                                                   "LAN or USB"),
                ("Rename…", self._rename_wave, ""), ("Remove", self._remove_waves, ""),
                ("Start at trigger", self._wave_from_trigger, "Drop the part of a scope record before its trigger"),
                ("Save as CSV…", self._save_wave, "Write the waveform as time, voltage"))):
            b = QtWidgets.QPushButton(label); b.clicked.connect(fn)
            if tip:
                b.setToolTip(tip)
            g.addWidget(b, k // 2, k % 2)
        v.addLayout(g)
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
        self.plots = PlotPanel(["Waveforms", "TWAVE sequence", "Imported waveforms"])
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
                       ("speed", d.tw_speed), ("pitch", d.tw_pitch * 1e3), ("rise", d.tw_rise * 1e9),
                       ("prate", d.pulse_rate / 1e3), ("pdelay", d.pulse_delay * 1e6), ("pwidth", d.pulse_width * 1e6),
                       ("prise", d.pulse_rise * 1e9), ("pfall", d.pulse_fall * 1e9)):
            self.form.set(k, val)
        self.pshape.setCurrentIndex(list(PULSE_SHAPES.values()).index(d.pulse_shape) if d.pulse_shape in PULSE_SHAPES.values() else 0)
        self._fill_outputs(); self._fill_tones(); self._fill_waves()
        self.notes.setPlainText(d.notes)
        self._loading = False
        self._timing(); self.plot()

    def _fill_outputs(self):
        self.outs.blockSignals(True)
        self.outs.setRowCount(0)
        for o in self.drv.outputs:
            r = self.outs.rowCount(); self.outs.insertRow(r)
            vals = [o.name, f"{o.dc:g}", f"{o.rf:g}", f"{o.rf_phase:g}", str(o.tw + 1) if o.tw >= 0 else "",
                    f"{o.tone_sign:g}", f"{o.pulse:g}", "", f"{o.wave_scale:g}"]
            for c, val in enumerate(vals):
                self.outs.setItem(r, c, QtWidgets.QTableWidgetItem(val))
            cb = QtWidgets.QComboBox()
            cb.addItems([NO_WAVE] + [w.name for w in self.drv.waves])
            cb.setCurrentText(o.wave if self.drv.wave(o.wave) is not None else NO_WAVE)
            cb.setEnabled(bool(self.drv.waves))
            cb.currentTextChanged.connect(lambda txt, row=r: self._wave_picked(row, txt))
            self.outs.setCellWidget(r, 7, cb)
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
        d.pulse_rate, d.pulse_delay, d.pulse_width = v["prate"] * 1e3, v["pdelay"] * 1e-6, v["pwidth"] * 1e-6
        d.pulse_rise, d.pulse_fall = v["prise"] * 1e-9, v["pfall"] * 1e-9
        d.pulse_shape = PULSE_SHAPES[self.pshape.currentText()]
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
        o.pulse = self._num(cell(6))
        o.wave_scale = self._num(cell(8), 1.0)
        self.outs.blockSignals(True)
        cell(0).setText(o.name)
        cell(4).setText(str(o.tw + 1) if o.tw >= 0 else "")
        self.outs.blockSignals(False)
        self._changed()

    def _wave_picked(self, row, txt):
        if self._loading or not self.drv or row >= len(self.drv.outputs):
            return
        self.drv.outputs[row].wave = "" if txt == NO_WAVE else txt
        self._changed()

    # ---------- imported waveforms ----------
    def _fill_waves(self):
        self.waves.blockSignals(True)
        self.waves.setRowCount(0)
        for w in self.drv.waves:
            r = self.waves.rowCount(); self.waves.insertRow(r)
            s = w.stats()
            vals = [w.name, str(s["points"]), WI._fmt_time(s["duration"]), f"{s['vmin']:.4g}", f"{s['vmax']:.4g}", "",
                    f"{w.period * 1e6:g}" if w.period else "", f"{w.delay * 1e6:g}", w.source]
            for c, val in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(val)
                if c in (1, 2, 3, 4, 8):
                    it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
                if c == 5:
                    it.setFlags((it.flags() | QtCore.Qt.ItemIsUserCheckable) & ~QtCore.Qt.ItemIsEditable)
                    it.setCheckState(QtCore.Qt.Checked if w.repeat else QtCore.Qt.Unchecked)
                self.waves.setItem(r, c, it)
        self.waves.blockSignals(False)
        self._plot_waves()

    def _waves_edited(self, item):
        if self._loading or not self.drv:
            return
        r = item.row()
        w = self.drv.waves[r]
        if item.column() == 0:
            new = item.text().strip().replace(" / ", " - ")
            if new and new != w.name and self.drv.wave(new) is None:
                for o in self.drv.outputs:
                    if o.wave == w.name:
                        o.wave = new
                w.name = new
        w.repeat = self.waves.item(r, 5).checkState() == QtCore.Qt.Checked
        w.period = max(self._num(self.waves.item(r, 6), 0.0), 0.0) * 1e-6
        w.delay = self._num(self.waves.item(r, 7), 0.0) * 1e-6
        self._fill_waves(); self._fill_outputs(); self._changed()

    def _selected_waves(self):
        rows = sorted({i.row() for i in self.waves.selectedIndexes()})
        return [self.drv.waves[r] for r in rows] if self.drv else []

    def _add_traces(self, traces, title, trigger=False):
        """Let the user pick from {name: Trace}, then add them to the driver."""
        if not self.drv or not traces:
            return
        dlg = WI.SignalPicker(traces, [o.name for o in self.drv.outputs], title, self, trigger=trigger)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        added = [self.drv.add_wave(tr) for tr in dlg.result_traces()]
        out = dlg.output_name()
        if added and out:
            self.drv.output(out).wave = added[0]
        self._fill_waves(); self._fill_outputs(); self._changed()
        if added:
            self.plots.setCurrentIndex(2)

    def _error(self, what, e):
        QtWidgets.QMessageBox.warning(self, what, str(e))

    def _spice_traces(self, path):
        steps = W.raw_steps(path) if path.lower().endswith(".raw") else 1
        step = None
        if steps > 1:
            k, ok = QtWidgets.QInputDialog.getInt(self, "Stepped run", f"This run has {steps} steps (.step). Which one?",
                                                  1, 1, steps)
            if not ok:
                return {}
            step = k - 1
        return W.spice_traces(path, step)

    def _import_spice(self):
        path = WI.ask_spice_file(self)
        if path:
            try:
                self._add_traces(self._spice_traces(path), f"Signals in {os.path.basename(path)}")
            except Exception as e:
                self._error("SPICE result", e)

    def _run_spice(self):
        if not self.drv:
            return
        dlg = WI.SpiceRunDialog(self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        try:
            raw = dlg.run()
            self._add_traces(self._spice_traces(raw), f"Signals from {os.path.basename(dlg.path.text())}")
        except Exception as e:
            self._error("Run SPICE", e)

    def _import_scope_file(self):
        path = WI.ask_scope_file(self)
        if path:
            try:
                tr = WI.scope_file_traces(path)
                self._add_traces(tr, f"Channels in {os.path.basename(path)}",
                                 trigger=any(t.t0 < 0 for t in tr.values()))
            except Exception as e:
                self._error("Scope file", e)

    def _capture_scope(self):
        if not self.drv:
            return
        dlg = WI.ScopeDialog(self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        try:
            idn, traces = dlg.capture()
        except Exception as e:
            self._error("Scope capture", f"{e}\n\nCheck the address (LAN: the scope's IP, port 5555) and that "
                                         "nothing else (another program or the scope's web page) holds the connection.")
            return
        if dlg.trigger.isChecked():
            traces = {k: t.from_trigger() for k, t in traces.items()}
        self._add_traces(traces, f"Captured from {idn.split(',')[1] if ',' in idn else idn}")

    def _rename_wave(self):
        ws = self._selected_waves()
        if ws:
            self.waves.editItem(self.waves.item(self.drv.waves.index(ws[0]), 0))

    def _remove_waves(self):
        for w in self._selected_waves():
            self.drv.remove_wave(w.name)
        self._fill_waves(); self._fill_outputs(); self._changed()

    def _wave_from_trigger(self):
        for w in self._selected_waves():
            k = self.drv.waves.index(w)
            self.drv.waves[k] = w.from_trigger()
        self._fill_waves(); self._changed()

    def _save_wave(self):
        ws = self._selected_waves()
        if not ws:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save waveform", ws[0].name + ".csv", "CSV (*.csv)")
        if path:
            np.savetxt(path, np.column_stack([ws[0].t, ws[0].v]), delimiter=",", header="time (s),voltage (V)",
                       comments="")

    def _plot_waves(self):
        d = self.drv
        if not d:
            return
        fig = self.plots.fig("Imported waveforms", show=False)
        ax = fig.add_subplot(111)
        ws = self._selected_waves() or d.waves
        if not ws:
            ax.text(0.5, 0.5, "No imported waveforms.\nAdd a SPICE result or a scope capture in section 5.",
                    ha="center", va="center"); ax.set_axis_off()
        else:
            span = max(w.duration() for w in ws)
            unit, k = next(((u, f) for u, f in (("ms", 1e-3), ("µs", 1e-6), ("ns", 1e-9)) if span >= 2 * f), ("ns", 1e-9))
            for w in ws:
                ax.plot((w.t + w.delay) / k, w.v, lw=0.8, label=w.name)
            ax.set_xlabel(f"time ({unit})"); ax.set_ylabel("V")
            users = {w.name: [o.name for o in d.outputs if o.wave == w.name] for w in ws}
            ax.set_title("As imported (before Waveform ×). " + "; ".join(
                f"{n} → {', '.join(u) if u else 'no output'}" for n, u in users.items())[:120], fontsize=9)
            ax.legend(fontsize=7, loc="upper right")
        self.plots.draw("Imported waveforms")

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
        txt = []
        if d.has_twave():
            txt.append(f"TWAVE step {d.step_time() * 1e6:.3f} µs, wave frequency {d.tw_frequency() / 1e3:.2f} kHz "
                       f"(speed / pitch). In a device the speed follows from its own pitch.")
        else:
            txt.append("No output carries a TWAVE phase.")
        if d.has_pulse():
            period = 1 / d.pulse_rate if d.pulse_rate > 0 else np.inf
            txt.append(f"Pulse every {period * 1e6:.1f} µs, {d.pulse_width / period:.1%} duty."
                       + ("  The pulse is longer than its period." if d.pulse_delay + d.pulse_width > period else ""))
        else:
            txt.append("No output is pulsed.")
        used = [f"{o.name} ← {o.wave}" for o in d.outputs if d.wave_of(o)]
        if used:
            txt.append("Imported waveforms: " + ", ".join(used) + ".")
        self.timing.setText(" ".join(txt))

    def _auto_window(self):
        d = self.drv
        spans = [5 / d.rf_freq] if any(o.rf for o in d.outputs) else []
        if d.has_twave() and np.isfinite(d.step_time()):
            spans.append(d.tw_phases * d.step_time())
        if any(o.tone_sign for o in d.outputs) and d.active_tones():
            spans.append(2 / min(t.freq for t in d.active_tones()))
        if d.has_pulse():
            spans.append(min(d.pulse_delay + 1.5 * d.pulse_width + 4 * d.pulse_fall,
                             1 / d.pulse_rate if d.pulse_rate > 0 else np.inf))
        for o in d.outputs:
            w = d.wave_of(o)
            if w:
                spans.append(w[0].delay + (w[0].cycle() if w[0].repeat else w[0].duration()))
        return max(spans) if spans else 1e-5

    def plot(self):
        d = self.drv
        if not d:
            return
        rows = sorted({i.row() for i in self.outs.selectedIndexes()})
        outs = [d.outputs[r] for r in rows] if rows else d.outputs
        span = self.window.value() * 1e-6 or self._auto_window()
        fmax = max(d.max_freq(), 1 / span)
        wdt = [d.wave_of(o)[0].sample_dt() for o in outs if d.wave_of(o)]
        if wdt:
            fmax = max(fmax, 0.25 / min(wdt))              # about two plot points per sample
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
