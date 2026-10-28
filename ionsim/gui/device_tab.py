"""Device builder: load a DXF or 3D CAD file, pick electrodes, apply voltage patterns,
solve (quick or detailed) and fly ions."""
import json
import os
import time
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.collections import LineCollection
from PySide6 import QtCore, QtGui, QtWidgets

from ..device import KINDS, QUALITY, Device, DeviceField, from_drawing, from_bodies, fly_device, device_gas_flow
from ..cad import CAD_EXT, MESH_EXT, UNITS, load_bodies
from ..stages import ExitIons
from .common import ParamForm, SimTab
from .electrode_view import ElectrodeView, INSULATOR
from .funnel_tab import FLOWS
from .driver_store import get_store
from .heating_view import HeatingBox, draw_heating
from .ion_table import IonTable, species_color, path_style, fate_caption, species_legend, species_stats, flight_metrics
from .traj3d import draw_traj3d
from .speeds import ion_speeds, gas_line
from ..ions import IonSpecies

MM = 1e-3
DXF_KINDS = ["planar2d", "axisym", "boards"]
PATTERNS = ["Fixed voltage", "DC gradient along the ion axis", "TWAVE phases", "Ground (0 V, no RF)",
            "Remove from TWAVE", "Wire to a driver (Drivers tab)", "Disconnect from driver"]
COLS = ["Electrode", "Layer", "DC (V)", "RF 0-pk (V)", "RF phase (°)", "TWAVE phase", "Set by", "Insulator εr",
        "Driver output"]
WIRINGS = ["All selected to one output", "TWAVE sequence along the ion axis"]
RF_ALT = {"RF+ / RF- alternate across the path": "across", "RF+ / RF- alternate along the path": "along",
          "RF+ outputs only": "none"}
FILE_FILTER = "Geometry (*.dxf *.step *.stp *.iges *.igs *.brep *.stl *.obj *.ply *.off *.glb *.gltf)"
MAX_NODES = 40_000_000
NETS = ["One per driver output (electrodes wired together)", "One per electrode"]


def examples_dir():
    """examples/geometry next to the source tree, or inside the standalone app."""
    import sys
    base = getattr(sys, "_MEIPASS", os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.abspath(os.path.join(base, "examples", "geometry"))


class DeviceTab(SimTab):
    def __init__(self, parent=None, store=None):
        super().__init__(["Geometry", "Potential", "Trajectories", "3D", "Results", "Gas flow", "Heating",
                          "Capacitance", "Loaded drive"], parent)
        self.plots.hint("Gas flow", "Set Gas flow (section 7) to “Flow field from gas throughput” and press Solve and fly.")
        self.plots.hint("Heating", "Tick Ion heating (section 8) and press Solve and fly.")
        self.plots.hint("Capacitance", "Press “Capacitance and crosstalk” in section 9.")
        self.plots.hint("Loaded drive", "Press “Check the drive with the load” in section 9 (needs ngspice).")
        self.cap = None             # circuit.Capacitance of the last load solve
        self.loaded = None          # circuit.LoadedDrive of the last check
        self.dev = None
        self.grid = None            # (key, DeviceGrid)
        self.field = None           # last DeviceField (warm start)
        self.selected = set()
        self.store = store or get_store()
        self.store.changed.connect(self._library_changed)
        self._building_table = False
        self._project_dir = None
        self._flow = None           # (key, GasFlow)

        # ---- 1. geometry ----
        box = QtWidgets.QGroupBox("1. Geometry")
        v = QtWidgets.QVBoxLayout(box)
        row = QtWidgets.QHBoxLayout()
        load = QtWidgets.QPushButton("Open DXF / STEP / STL…"); load.clicked.connect(self._open_file)
        row.addWidget(load)
        v.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        sv = QtWidgets.QPushButton("Save setup…"); sv.clicked.connect(self.save_setup)
        op = QtWidgets.QPushButton("Open setup…"); op.clicked.connect(self.open_setup)
        sv.setToolTip("Save the file, how it is read, every electrode's voltages and the ion settings (.ionsim)")
        row.addWidget(sv); row.addWidget(op)
        v.addLayout(row)
        self.file_label = QtWidgets.QLabel("No file loaded. Try examples/geometry/*.dxf"); self.file_label.setWordWrap(True)
        v.addWidget(self.file_label)
        form = QtWidgets.QFormLayout()
        self.kind = QtWidgets.QComboBox()
        for k in KINDS:
            self.kind.addItem(KINDS[k], k)
        self.kind.setToolTip("How the drawing becomes space. DXF: 2D cross-section, axisymmetric half profile, or pads on two boards.")
        self.units = QtWidgets.QComboBox(); self.units.addItems(list(UNITS))
        self.units.setToolTip("Units of the drawing numbers when the file does not say (STL/OBJ, unitless DXF)")
        self.axis = QtWidgets.QComboBox(); self.axis.addItems(["z", "x", "y"])
        self.axis.setToolTip("3D files: the CAD axis the ions travel along")
        form.addRow("Read as", self.kind); form.addRow("Units if not in file", self.units); form.addRow("Ion axis (3D files)", self.axis)
        v.addLayout(form)
        self.geo = ParamForm("", [
            ("gap", "Board gap (boards)", 2.75, 0.05, 100, 3, 0.05, "mm"),
            ("length", "Length along ion axis (2D)", 100.0, 0.1, 10000, 1, 10, "mm"),
            ("bt", "PCB substrate behind pads (0 = none)", 1.6, 0, 50, 3, 0.1, "mm"),
            ("beps", "Substrate εr (FR4 4.4)", 4.4, 1, 100, 2, 0.1, ""),
        ])
        self.geo.boxes["bt"].setToolTip("Boards: an insulating board of this thickness behind the pads, "
                                        "with a ground plane on its back face")
        self.geo.setFlat(True)
        self.both = QtWidgets.QCheckBox("Same pads on both boards"); self.both.setChecked(True)
        self.open_in = QtWidgets.QCheckBox("Open entrance (ions arrive from upstream)"); self.open_in.setChecked(True)
        v.addWidget(self.geo); v.addWidget(self.both); v.addWidget(self.open_in)
        self.form_layout.addWidget(box)

        # ---- 2. electrodes ----
        box = QtWidgets.QGroupBox("2. Electrodes  (click them in the drawing, Ctrl-click to add)")
        v = QtWidgets.QVBoxLayout(box)
        row = QtWidgets.QHBoxLayout()
        self.layer = QtWidgets.QComboBox()
        sel_layer = QtWidgets.QPushButton("Select layer"); sel_layer.clicked.connect(self._select_layer)
        sel_all = QtWidgets.QPushButton("All"); sel_all.clicked.connect(lambda: self._set_selection(set(self.dev.names()) if self.dev else set()))
        row.addWidget(self.layer, 1); row.addWidget(sel_layer); row.addWidget(sel_all)
        v.addLayout(row)
        self.table = QtWidgets.QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.table.setMinimumHeight(220)
        self.table.horizontalHeaderItem(7).setToolTip("Blank or 0: an electrode. A number makes the part an insulator "
                                                     "(PCB, spacer) with that relative permittivity; it carries no voltage.")
        self.table.itemSelectionChanged.connect(self._table_selection)
        self.table.itemChanged.connect(self._table_edit)
        v.addWidget(self.table)
        self.hover = QtWidgets.QLabel(" "); self.hover.setWordWrap(True)
        v.addWidget(self.hover)
        self.form_layout.addWidget(box)

        # ---- 3. patterns ----
        box = QtWidgets.QGroupBox("3. Apply to selected electrodes")
        v = QtWidgets.QVBoxLayout(box)
        self.sel_label = QtWidgets.QLabel("Nothing selected"); v.addWidget(self.sel_label)
        self.pattern = QtWidgets.QComboBox(); self.pattern.addItems(PATTERNS); v.addWidget(self.pattern)
        self.p_fixed = ParamForm("", [("dc", "DC", 0.0, -1e5, 1e5, 3, 1, "V"), ("rf", "RF amplitude (0-pk)", 0.0, 0, 1e5, 2, 5, "V"),
                                      ("ph", "RF phase", 0.0, -360, 360, 1, 180, "°")])
        self.p_grad = ParamForm("", [("v0", "DC at first (upstream)", 80.0, -1e5, 1e5, 3, 1, "V"),
                                     ("v1", "DC at last (downstream)", 20.0, -1e5, 1e5, 3, 1, "V"),
                                     ("rf", "RF amplitude (0-pk)", 0.0, 0, 1e5, 2, 5, "V")])
        self.alt = QtWidgets.QCheckBox("Alternate RF 0° / 180° along the axis"); self.alt.setChecked(True)
        self.p_grad.layout().addRow(self.alt)
        self.p_tw = ParamForm("", [("start", "First electrode is phase", 1, 1, 64, 0, 1, ""),
                                   ("dc", "DC on these pads (wave adds to it)", 0.0, -1e5, 1e5, 3, 1, "V")])
        self.p_tw.layout().addRow(QtWidgets.QLabel("Phases are numbered along the ion axis; pads side by side share a phase."))
        self.p_drv = QtWidgets.QGroupBox(); self.p_drv.setFlat(True)
        fl = QtWidgets.QFormLayout(self.p_drv)
        self.drv_name = QtWidgets.QComboBox(); self.drv_wiring = QtWidgets.QComboBox(); self.drv_wiring.addItems(WIRINGS)
        self.drv_out = QtWidgets.QComboBox()
        self.drv_start = QtWidgets.QSpinBox(); self.drv_start.setRange(1, 64)
        self.drv_alt = QtWidgets.QComboBox(); self.drv_alt.addItems(list(RF_ALT))
        fl.addRow("Driver", self.drv_name); fl.addRow("Wiring", self.drv_wiring); fl.addRow("Output", self.drv_out)
        fl.addRow("First electrode is phase", self.drv_start); fl.addRow("RF polarity", self.drv_alt)
        fl.addRow(QtWidgets.QLabel("A wired electrode takes its DC, RF, tones and TWAVE from the driver. "
                                   "Build and edit drivers in the Drivers tab."))
        self.drv_name.currentIndexChanged.connect(self._drv_pattern_changed)
        self.drv_wiring.currentIndexChanged.connect(self._drv_pattern_changed)
        for f in (self.p_fixed, self.p_grad, self.p_tw, self.p_drv):
            f.setFlat(True); v.addWidget(f)
        apply = QtWidgets.QPushButton("Apply"); apply.clicked.connect(self._apply_pattern)
        v.addWidget(apply)
        self.pattern.currentIndexChanged.connect(self._pattern_changed)
        self.form_layout.addWidget(box)

        # ---- 4. drive ----
        self.drive = ParamForm("4. RF and traveling wave", [
            ("freq", "RF frequency", 1000.0, 1, 1e5, 1, 50, "kHz"),
            ("n", "TWAVE phases", 8, 2, 64, 0, 1, ""),
            ("on", "TWAVE phases on", 4, 1, 63, 0, 1, ""),
            ("amp", "TWAVE amplitude", 20.0, 0, 1e4, 2, 1, "V"),
            ("speed", "TWAVE speed", 200.0, 0.1, 1e5, 1, 10, "m/s"),
            ("pitch", "Electrode pitch (set by TWAVE pattern)", 0.508, 0.001, 1000, 4, 0.01, "mm"),
        ])
        self.step_label = QtWidgets.QLabel(""); self.drive.layout().addRow(self.step_label)
        self.drive.changed.connect(self._drive_changed)
        self.form_layout.addWidget(self.drive)

        # ---- 5. grid ----
        box = QtWidgets.QGroupBox("5. Grid")
        form = QtWidgets.QFormLayout(box)
        self.quality = QtWidgets.QComboBox(); self.quality.addItems(["Quick", "Detailed", "Custom"])
        self.quality.setToolTip("Quick: coarse grid, seconds. Detailed: fine grid for final numbers. Custom: set the spacing.")
        self.h = QtWidgets.QDoubleSpinBox(); self.h.setDecimals(4); self.h.setRange(0.0005, 100); self.h.setSuffix(" mm")
        self.h.setEnabled(False)
        self.grid_label = QtWidgets.QLabel(""); self.grid_label.setWordWrap(True)
        chk = QtWidgets.QPushButton("Check geometry on this grid"); chk.clicked.connect(self.check_geometry)
        form.addRow("Quality", self.quality); form.addRow("Grid spacing", self.h); form.addRow(self.grid_label); form.addRow(chk)
        self.quality.currentIndexChanged.connect(self._quality_changed)
        self.h.valueChanged.connect(self._grid_info)
        self.form_layout.addWidget(box)

        # ---- 6. ions ----
        self.ion_table = IonTable("6. Ions (one row per species, flown together)", [IonSpecies("m/z 500", 500, 1, 1.0, 200)])
        self.form_layout.addWidget(self.ion_table)
        self.ions = ParamForm("Gas, start and timing", [
            ("p", "Gas pressure (≈0 for vacuum)", 4.0, 0.0, 1e4, 6, 0.5, "Torr"),
            ("T", "Gas temperature", 300.0, 1, 2000, 1, 5, "K"),
            ("u", "Gas speed along the ion axis", 0.0, -1e4, 1e4, 2, 1, "m/s"),
            ("E", "Axial ion energy at start (per charge)", 0.0, 0, 1e5, 3, 1, "eV"),
            ("x0", "Start centre x", 0.0, -1e5, 1e5, 3, 0.5, "mm"),
            ("y0", "Start centre y", 0.0, -1e5, 1e5, 3, 0.5, "mm"),
            ("r", "Start radius", 0.3, 0, 1e4, 3, 0.1, "mm"),
            ("z0", "Start z", 0.0, -1e5, 1e5, 3, 0.5, "mm"),
            ("z1", "Detector z", 10.0, -1e5, 1e5, 3, 0.5, "mm"),
            ("dt", "Time step (0 = auto)", 0.0, 0, 1e6, 4, 0.01, "µs"),
            ("tmax", "Max time (0 = auto)", 0.0, 0, 1e7, 3, 1, "ms"),
        ])
        self.form_layout.addWidget(self.ions)

        # ---- 7. gas flow and space charge ----
        self.extra = ParamForm("7. Gas flow and space charge", [
            ("slm", "Gas throughput", 1.0, 0.001, 1000, 3, 0.1, "slm"),
            ("r_in", "Gas inlet radius (disc at the entrance, about the ion start)", 2.0, 0.01, 1000, 3, 0.5, "mm"),
            ("I", "Ion current (0 = no space charge)", 0.0, 0, 1e6, 3, 1, "nA"),
            ("it", "Space-charge iterations", 4, 1, 30, 0, 1, ""),
        ])
        self.flow = QtWidgets.QComboBox(); self.flow.addItems(FLOWS)
        self.flow.setToolTip("Uniform uses the gas speed in 6. Flow field: viscous flow from the inlet disc, "
                             "pumped out through every other open side (axisymmetric and 3D devices)")
        self.extra.layout().insertRow(0, "Gas flow", self.flow)
        self.flow.currentIndexChanged.connect(self._flow_changed)
        self.form_layout.addWidget(self.extra)
        self.heat = HeatingBox("8. Ion heating")
        self.form_layout.addWidget(self.heat)

        # ---- 9. electrode load ----
        self.load = ParamForm("9. Electrode load and drive (SPICE)", [
            ("r", "Driver source resistance", 50.0, 0.001, 1e9, 3, 10, "Ω"),
            ("rdc", "Source resistance of nets held at DC", 1000.0, 0.001, 1e12, 3, 100, "Ω"),
            ("l", "Wiring inductance per net", 0.0, 0.0, 1e6, 2, 10, "nH"),
            ("per", "Drive periods to settle", 2, 2, 50, 0, 1, ""),
        ])
        self.nets = QtWidgets.QComboBox(); self.nets.addItems(NETS)
        self.nets.setToolTip("Electrodes on one driver output share a wire, so they are one net; or treat every electrode "
                             "as its own net (one field solve each)")
        self.load.layout().insertRow(0, "Nets", self.nets)
        lb = QtWidgets.QGridLayout()
        for k, (label, fn, tip) in enumerate((
                ("Capacitance and crosstalk", self.solve_load, "One field solve per net: total, to-ground and mutual "
                                                               "capacitance, and how much of one net's swing reaches another"),
                ("Export SPICE load…", self.export_load, "The electrodes as a SPICE subcircuit (.lib) to connect to your "
                                                         "own driver model"),
                ("Check the drive with the load", self.check_drive, "Run ngspice: every net driven by its waveform through "
                                                                    "the source resistance into the electrode load"),
                ("Fly with the load", self.use_loaded, "Add what the load does to each net (electrode minus source "
                                                       "voltage) to the drive, so Solve and fly uses the voltages the "
                                                       "electrodes really get"),
                ("Take the load off", self.clear_load, "Fly with the drive as set again"))):
            b = QtWidgets.QPushButton(label); b.setToolTip(tip); b.clicked.connect(fn)
            lb.addWidget(b, k // 2, k % 2)
        self.load.layout().addRow(lb)
        self.load_label = QtWidgets.QLabel(""); self.load_label.setWordWrap(True)
        self.load.layout().addRow(self.load_label)
        self.form_layout.addWidget(self.load)
        self.form_layout.addStretch(1)

        self.scroll.setMinimumWidth(470)
        self.run_btn.setText("Solve and fly")
        self.run_btn.clicked.connect(self.run)
        for w in (self.kind, self.axis, self.units):
            w.currentIndexChanged.connect(self._reload_needed)
        self.geo.changed.connect(self._geometry_params_changed)
        self.both.toggled.connect(self._geometry_params_changed)
        self.open_in.toggled.connect(self._geometry_params_changed)
        self.view = ElectrodeView(self.plots.figs["Geometry"].canvas, self._picked, self._hovered)
        self._pattern_changed()
        self._drive_changed()
        self._flow_changed()
        self._show_rows()
        self._draw_geometry()

    # ================= loading =================
    def _open_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open geometry", examples_dir(), FILE_FILTER)
        if path:
            self.load_file(path)

    def load_file(self, path, kind=None):
        ext = os.path.splitext(path)[1].lower()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            if ext == ".dxf":
                from ..dxf import load_dxf
                drawing = load_dxf(path, self.units.currentText())
                kind = kind or (self.kind.currentData() if self.kind.currentData() in DXF_KINDS else "planar2d")
                dev = from_drawing(drawing, kind)
                warnings = drawing.warnings
                if drawing.units in UNITS:
                    self._set_combo(self.units, drawing.units)
            elif ext in CAD_EXT + MESH_EXT:
                bodies = load_bodies(path, self.units.currentText(), self.axis.currentText())
                dev = from_bodies(bodies, os.path.abspath(path), self.units.currentText())
                warnings = []
            else:
                raise ValueError(f"Unsupported file type '{ext}'. Use DXF, STEP/IGES/BREP, or STL/OBJ.")
            if not dev.electrodes:
                raise ValueError("No electrodes found in the file. " + " ".join(warnings))
        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            self._error("Could not open the file", f"{os.path.basename(path)}:\n{e}")
            return False
        QtWidgets.QApplication.restoreOverrideCursor()
        self._install(dev, warnings)
        return True

    def _install(self, dev, warnings=()):
        self.dev = dev
        self.grid = self.field = None
        self.selected = set()
        self.kind.blockSignals(True)
        for i in range(self.kind.count()):
            item = self.kind.model().item(i)
            item.setEnabled((self.kind.itemData(i) == "cad3d") == (dev.kind == "cad3d"))
        self._set_combo(self.kind, dev.kind, data=True)
        self.kind.blockSignals(False)
        self._geo_to_device(dev)
        self._drive_to_device()
        layers = sorted({e.layer for e in dev.electrodes if e.layer})
        self.layer.clear(); self.layer.addItems(layers or ["(no layers)"])
        self._fill_table()
        self.table.clearSelection()
        self.sel_label.setText("Nothing selected: click electrodes in the drawing or rows in the table")
        self._show_rows()
        self._default_ion_path()
        self._quality_changed()
        self.file_label.setText(f"{os.path.basename(dev.source)}: {len(dev.electrodes)} electrodes"
                                + (f" on {len(layers)} layers" if layers else "") + f", units {dev.units}")
        self.log(f"Loaded {os.path.basename(dev.source)} as {KINDS[dev.kind]}: {len(dev.electrodes)} electrodes.")
        for w in warnings:
            self.log("  Note: " + w)
        self._draw_geometry()
        self.plots.setCurrentIndex(0)

    def _reload_needed(self, *_):
        if self.dev is None:
            return
        kind = self.kind.currentData()
        if self.dev.source.lower().endswith(".dxf") and kind in DXF_KINDS and kind != self.dev.kind:
            keep = self.dev.to_dict()
            if self.load_file(self.dev.source, kind):
                self.dev.apply_dict({k: v for k, v in keep.items() if k != "kind"})
                self._fill_table(); self._draw_geometry()
        elif not self.dev.source.lower().endswith(".dxf") and self.sender() in (self.axis, self.units):
            keep = self.dev.to_dict()
            if self.load_file(self.dev.source):
                self.dev.apply_dict(keep)
                self._fill_table(); self._draw_geometry()

    def _show_rows(self):
        k = self.dev.kind if self.dev else self.kind.currentData()
        lay = self.geo.layout()
        lay.setRowVisible(0, k == "boards"); lay.setRowVisible(1, k == "planar2d")
        lay.setRowVisible(2, k == "boards"); lay.setRowVisible(3, k == "boards")
        self.both.setVisible(k == "boards")
        self.axis.setEnabled(k == "cad3d")

    def _geometry_params_changed(self, *_):
        if self.dev is None:
            return
        g = self.geo.values()
        self._geo_to_device(self.dev)
        self.grid = self.field = None
        if self.dev.kind == "planar2d":
            self.ions.set("z1", g["length"])
        self._quality_changed()

    def _geo_to_device(self, dev):
        g = self.geo.values()
        dev.board_gap, dev.length = g["gap"] * MM, g["length"] * MM
        dev.board_thickness, dev.board_eps = g["bt"] * MM, g["beps"]
        dev.both_boards, dev.open_entrance = self.both.isChecked(), self.open_in.isChecked()

    def _flow_changed(self, *_):
        k = self.flow.currentIndex()
        self.ions.boxes["u"].setEnabled(k == 1)
        self.extra.boxes["slm"].setEnabled(k == 2); self.extra.boxes["r_in"].setEnabled(k == 2)

    def _default_ion_path(self):
        lo, hi = self.dev.extent()
        mid = (lo + hi) / 2
        if self.dev.kind == "axisym":
            mid[:2] = 0.0
        if self.dev.kind == "boards":
            mid[1] = 0.0
        L = hi[2] - lo[2]
        self.ions.set("x0", mid[0] / MM); self.ions.set("y0", mid[1] / MM)
        self.ions.set("z0", (lo[2] + (0.0 if self.dev.kind == "planar2d" else 0.02 * L)) / MM)
        self.ions.set("z1", hi[2] / MM)
        size = (hi - lo)[:2]
        self.ions.set("r", max(0.05 * float(np.min(size[size > 0])) if np.any(size > 0) else 0.3e-3, 1e-5) / MM)

    # ================= table and selection =================
    def _fill_table(self):
        self._building_table = True
        t = self.table
        t.setRowCount(0)
        for e in self.dev.electrodes:
            r = t.rowCount(); t.insertRow(r)
            for c in range(len(COLS)):
                it = QtWidgets.QTableWidgetItem("")
                if c in (0, 1, 6, 8):
                    it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
                t.setItem(r, c, it)
        self._building_table = False
        self._refresh_rows()
        self._sync_table_selection()

    def _refresh_rows(self):
        self._building_table = True
        grey = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        for r, e in enumerate(self.dev.electrodes):
            dc, rf, ph, tw, wired = self._eff(e)
            self.table.item(r, 0).setText(e.name); self.table.item(r, 1).setText(e.layer)
            for c, val in ((2, f"{dc:.4g}"), (3, f"{rf:.4g}"), (4, f"{ph:.4g}"),
                           (5, "" if tw < 0 else str(tw + 1)), (6, e.role), (7, f"{e.eps:g}" if e.eps > 0 else ""),
                           (8, wired)):
                it = self.table.item(r, c)
                it.setText(val)
                if 2 <= c <= 5:
                    it.setForeground(grey if e.drive else QtGui.QBrush())
                    it.setToolTip("From the driver: edit it in the Drivers tab, or type a value here to disconnect"
                                  if e.drive else "")
        self._building_table = False

    def _table_edit(self, item):
        if self._building_table or self.dev is None:
            return
        e = self.dev.electrodes[item.row()]
        try:
            val = float(item.text() or 0)
        except ValueError:
            self._refresh_rows()
            return
        c = item.column()
        if e.drive and 2 <= c <= 5:                        # typing over a driver value disconnects it
            dc, rf, ph, tw, _ = self._eff(e)
            e.dc, e.rf, e.phase, e.tw = dc, rf, ph, tw
            self.log(f"{e.name} disconnected from {e.drive}")
            e.drive = ""
        if c == 2:
            e.dc = val
        elif c == 3:
            e.rf = abs(val)
        elif c == 4:
            e.phase = val
        elif c == 5:
            e.tw = int(val) - 1 if val >= 1 else -1
        elif c == 7:
            e.eps = max(val, 0.0)
            self.grid = self.field = None
        e.role = "insulator" if e.eps > 0 else "edited"
        self._refresh_rows()
        self._repaint()

    def _table_selection(self):
        if self._building_table or self.dev is None:
            return
        rows = {i.row() for i in self.table.selectedIndexes()}
        self.selected = {self.dev.electrodes[r].name for r in rows}
        self._after_selection()

    def _sync_table_selection(self):
        self._building_table = True
        self.table.clearSelection()
        mode = self.table.selectionMode()
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        first = None
        for r, e in enumerate(self.dev.electrodes if self.dev else []):
            if e.name in self.selected:
                self.table.selectRow(r)
                first = r if first is None else first
        self.table.setSelectionMode(mode)
        if first is not None:
            self.table.scrollToItem(self.table.item(first, 0))
        self._building_table = False

    def _set_selection(self, names):
        self.selected = set(names)
        self._sync_table_selection()
        self._after_selection()

    def _picked(self, name, additive):
        if additive:
            self._set_selection(self.selected ^ {name})
        else:
            self._set_selection({name})

    def _hovered(self, name):
        if name is None or self.dev is None:
            self.hover.setText(" ")
            return
        e = self.dev.get(name)
        if e.eps > 0:
            self.hover.setText(f"<b>{name}</b> ({e.layer or 'no layer'}): insulator, εr {e.eps:g}")
            return
        dc, rf, ph, tw, wired = self._eff(e)
        txt = f"<b>{name}</b> ({e.layer or 'no layer'}): DC {dc:.4g} V"
        if rf:
            txt += f", RF {rf:.4g} V at {ph:g}°"
        if tw >= 0:
            txt += f", TWAVE phase {tw + 1}"
        if wired:
            w = self.dev.driver_of(e)
            if w and w[1].tone_sign and w[0].active_tones():
                txt += f", {len(w[0].active_tones())} tones"
            txt += f"  ← {wired}"
        self.hover.setText(txt)

    def _select_layer(self):
        if self.dev:
            lay = self.layer.currentText()
            self._set_selection({e.name for e in self.dev.electrodes if e.layer == lay})

    def _after_selection(self):
        n = len(self.selected)
        if n == 0:
            self.sel_label.setText("Nothing selected: click electrodes in the drawing or rows in the table")
        elif n == 1:
            e = self.dev.get(next(iter(self.selected)))
            self.sel_label.setText(f"Selected: <b>{e.name}</b>" + (f" (wired to {e.drive})" if e.drive else ""))
            dc, rf, ph, _, _ = self._eff(e)
            self.p_fixed.set("dc", dc); self.p_fixed.set("rf", rf); self.p_fixed.set("ph", ph)
        else:
            self.sel_label.setText(f"Selected: <b>{n} electrodes</b>")
        self._repaint()

    # ================= patterns and drive =================
    def _pattern_changed(self, *_):
        k = self.pattern.currentIndex()
        self.p_fixed.setVisible(k == 0); self.p_grad.setVisible(k == 1); self.p_tw.setVisible(k == 2)
        self.p_drv.setVisible(k == 5)
        if k == 5:
            self._fill_drivers()

    # ================= drivers =================
    def _fill_drivers(self):
        cur = self.drv_name.currentText()
        self.drv_name.blockSignals(True)
        self.drv_name.clear(); self.drv_name.addItems(self.store.names())
        i = self.drv_name.findText(cur)
        self.drv_name.setCurrentIndex(max(i, 0))
        self.drv_name.blockSignals(False)
        self._drv_pattern_changed()

    def _drv_pattern_changed(self, *_):
        d = self.store.get(self.drv_name.currentText())
        cur = self.drv_out.currentText()
        self.drv_out.clear()
        if d:
            self.drv_out.addItems([o.name for o in d.outputs])
            i = self.drv_out.findText(cur)
            if i >= 0:
                self.drv_out.setCurrentIndex(i)
        seq = self.drv_wiring.currentIndex() == 1
        lay = self.p_drv.layout()
        for w, show in ((self.drv_out, not seq), (self.drv_start, seq), (self.drv_alt, seq)):
            w.setVisible(show); lay.labelForField(w).setVisible(show)

    def _library_changed(self):
        """A driver was edited in the Drivers tab: refresh the wiring and the colours."""
        if self.p_drv.isVisible():
            self._fill_drivers()
        if self.dev is not None:
            self._sync_drivers()
            self._refresh_rows(); self._repaint()

    def _sync_drivers(self):
        """Give the device the library's current copy of every driver it is wired to."""
        for name in {e.drive.split(" / ", 1)[0] for e in self.dev.electrodes if e.drive}:
            d = self.store.get(name)
            if d is not None:
                self.dev.drivers[name] = d.copy()

    def _eff(self, e):
        """What an electrode carries now: (dc, rf, phase, tw, driver label)."""
        w = self.dev.driver_of(e)
        if w:
            o = w[1]
            return o.dc, o.rf, o.rf_phase, o.tw, e.drive
        return e.dc, e.rf, e.phase, e.tw, (e.drive + " (missing)" if e.drive else "")

    def _apply_pattern(self):
        if not self.dev or not self.selected:
            self._error("Nothing selected", "Select electrodes first: click them in the drawing, Ctrl-click to add more, "
                        "or pick a layer and press Select layer.")
            return
        names = [e.name for e in self.dev.electrodes if e.name in self.selected]
        k = self.pattern.currentIndex()
        if k in (0, 1, 2, 3):
            self.dev.unwire(names)
        if k == 0:
            f = self.p_fixed.values(); self.dev.apply_fixed(names, f["dc"], f["rf"], f["ph"])
        elif k == 1:
            f = self.p_grad.values(); self.dev.apply_gradient(names, f["v0"], f["v1"], f["rf"], self.alt.isChecked())
        elif k == 2:
            self._drive_to_device()
            f = self.p_tw.values(); self.dev.apply_twave(names, int(f["start"]) - 1, f["dc"])
            self.drive.blockSignals(True); self.drive.set("pitch", self.dev.twave.pitch / MM); self.drive.blockSignals(False)
            self._drive_changed()
        elif k == 3:
            self.dev.apply_fixed(names, 0.0, 0.0, 0.0, label="ground"); self.dev.clear_twave(names)
        elif k == 4:
            self.dev.clear_twave(names)
        elif k == 5:
            d = self.store.get(self.drv_name.currentText())
            if d is None:
                self._error("No driver", "Build a driver in the Drivers tab first.")
                return
            if self.drv_wiring.currentIndex() == 0:
                self.dev.wire_driver(names, d.copy(), "one", self.drv_out.currentText())
            else:
                missing = self.dev.wire_driver(names, d.copy(), "twave", start_phase=self.drv_start.value() - 1,
                                               rf_alt=RF_ALT[self.drv_alt.currentText()])
                if missing:
                    self.log(f"  Note: {d.name} has no output for TWAVE phase(s) {missing}; those electrodes were left as they were.")
                self.drive.blockSignals(True); self.drive.set("pitch", self.dev.twave.pitch / MM); self.drive.blockSignals(False)
                self._drive_changed()
                st = d.step_time()
                if np.isfinite(st) and self.dev.twave.pitch > 0:
                    self.log(f"  Wave speed on this board: {self.dev.twave.pitch / st:.4g} m/s "
                             f"(pitch {self.dev.twave.pitch / MM:.4g} mm, step {st * 1e6:.4g} µs from the driver)")
        else:
            self.dev.unwire(names)
        self._refresh_rows()
        self._repaint()
        self.log(f"{PATTERNS[k]} → {len(names)} electrode(s)")

    def _drive_to_device(self):
        if self.dev is None:
            return
        d = self.drive.values()
        self.dev.rf_freq = d["freq"] * 1e3
        tw = self.dev.twave
        tw.n_phases, tw.n_on, tw.amplitude, tw.speed = int(d["n"]), min(int(d["on"]), int(d["n"]) - 1), d["amp"], d["speed"]
        tw.pitch = d["pitch"] * MM

    def _drive_changed(self, *_):
        self._drive_to_device()
        d = self.drive.values()
        st = d["pitch"] * MM / max(d["speed"], 1e-9)
        self.step_label.setText(f"Wave steps one electrode every {st * 1e6:.3g} µs "
                                f"(wave frequency {1 / (st * d['n']) / 1e3:.3g} kHz)")

    # ================= grid =================
    def _quality_changed(self, *_):
        q = self.quality.currentText()
        self.h.setEnabled(q == "Custom")
        if self.dev is not None and q != "Custom":
            self.h.blockSignals(True); self.h.setValue(self.dev.default_h(q) / MM); self.h.blockSignals(False)
        self._grid_info()

    def _grid_info(self, *_):
        if self.dev is None:
            self.grid_label.setText("")
            return
        h = self.h.value() * MM
        lo, hi = self.dev.extent()
        size = hi - lo
        pad = 2 * max(8 * h, 0.1 * float(np.max(size[:2] if self.dev.kind == "planar2d" else size)))
        if self.dev.kind == "planar2d":
            n = ((max(abs(lo[0]), abs(hi[0]), abs(lo[1]), abs(hi[1])) * 2 + pad) / h) ** 2
        elif self.dev.kind == "axisym":
            n = ((hi[0] + pad / 2) / h) * ((size[2] + pad) / h)
        elif self.dev.kind == "boards":
            n = ((size[0] + pad) / h) * ((self.dev.board_gap + 2 * self.dev.board_thickness + 8 * h) / h) * ((size[2] + pad) / h)
        else:
            n = np.prod((size + pad) / h)
        dim = "2D" if self.dev.kind in ("planar2d", "axisym") else "3D"
        txt = f"{dim} grid ≈ {n / 1e6:.2f} M nodes"
        if n > MAX_NODES:
            txt += "  (too many: use a larger spacing)"
        self.grid_label.setText(txt)

    def _grid_key(self):
        return (self.dev.source, self.dev.kind, self.units.currentText(), self.axis.currentText(), self.dev.board_gap,
                self.dev.both_boards, self.dev.length, self.dev.open_entrance, round(self.h.value(), 6),
                self.dev.board_thickness, self.dev.board_eps, tuple(e.eps for e in self.dev.electrodes))

    def check_geometry(self):
        if self.dev is None:
            self._error("No geometry", "Open a DXF or 3D file first.")
            return
        t = time.time()
        dg = self.dev.build(self.h.value() * MM)
        self.grid = (self._grid_key(), dg)
        self.field = None
        self.log(f"Grid {dg.nodes / 1e6:.2f} M nodes, spacing {dg.h / MM:.4g} mm, painted in {time.time() - t:.1f} s.")
        if dg.warnings:
            for w in dg.warnings:
                self.log("  Check: " + w)
            QtWidgets.QMessageBox.warning(self, "Geometry check", "\n\n".join(dg.warnings[:8]))
        else:
            self.log("  Check: no problems found.")

    # ================= run =================
    def run(self):
        if self.dev is None:
            self._error("No geometry", "Open a DXF or 3D file first (examples are in examples/geometry).")
            return
        self._sync_drivers()
        if not any(any(self._eff(e)[:3]) or self._eff(e)[3] >= 0 or self.dev.wave_of(e) or self.dev.pulse_of(e)
                   for e in self.dev.electrodes):
            self._error("No voltages", "Every electrode is at 0 V. Select electrodes and apply a pattern first.")
            return
        self._drive_to_device()
        if self.dev.has_twave() and self.dev.twave.pitch <= 0:
            self._error("TWAVE pitch", "Set the electrode pitch for the traveling wave.")
            return
        key = self._grid_key()
        dg = self.grid[1] if self.grid and self.grid[0] == key else None
        prev = self.field if dg is not None else None
        io = self.ions.values()
        try:
            species = [sp for sp in self.ion_table.species() if sp.n > 0]
        except ValueError as e:
            self._error("Ions", str(e))
            return
        n_tot = sum(sp.n for sp in species)
        ex, flow_mode = self.extra.values(), self.flow.currentIndex()
        if flow_mode == 2 and self.dev.kind == "planar2d":
            self._error("Gas flow", "A 2D cross-section has no flow pattern along the axis: use a uniform gas speed.")
            return
        flow_key = (key, ex["slm"], io["p"], io["T"], ex["r_in"], io["x0"], io["y0"])
        flow_cached = self._flow[1] if flow_mode == 2 and dg is not None and self._flow and self._flow[0] == flow_key else None
        h = self.h.value() * MM
        dev = self.dev
        heat_model = self.heat.model()

        def job(progress):
            t0 = time.time()
            g = dg
            if g is None:
                progress(0.01)
                g = dev.build(h)
            t1 = time.time()
            fld = DeviceField(g, prev, progress=lambda f: progress(0.3 * f))
            gas_u, flow = (io["u"] if flow_mode == 1 else 0.0), None
            if flow_mode == 2:
                flow = flow_cached or device_gas_flow(fld, ex["slm"], max(io["p"], 1e-6), io["T"], ex["r_in"] * MM,
                                                      (io["x0"] * MM, io["y0"] * MM))
                gas_u = flow
            t2 = time.time()
            res, info = fly_device(fld, [sp.mz for sp in species], [sp.K0 for sp in species], [sp.n for sp in species],
                                   io["p"], io["T"], [sp.z for sp in species],
                                   (io["x0"] * MM, io["y0"] * MM), io["r"] * MM, io["z0"] * MM, io["z1"] * MM,
                                   io["E"], gas_u, io["dt"] * 1e-6 or None, io["tmax"] * 1e-3 or None,
                                   record=min(n_tot, max(40, 10 * len(species))),
                                   progress=lambda f: progress(0.3 + 0.7 * f),
                                   current=ex["I"] * 1e-9, sc_iterations=int(ex["it"]), heating=heat_model)
            return dict(grid=g, field=fld, res=res, info=info, built=dg is None, t_build=t1 - t0, t_solve=t2 - t1,
                        t_fly=time.time() - t2, io=io, key=key, flow=flow, flow_key=flow_key, ex=ex, species=species, gas_u=gas_u)

        self.start(job, self._show, "Solving fields and flying ions…" if dg is None else "Re-solving voltages and flying ions…")

    def _show(self, out):
        g, fld, res, info = out["grid"], out["field"], out["res"], out["info"]
        self.grid, self.field = (out["key"], g), fld
        if out["built"]:
            self.log(f"Grid {g.nodes / 1e6:.2f} M nodes (spacing {g.h / MM:.4g} mm) built in {out['t_build']:.1f} s")
            for w in g.warnings:
                self.log("  Check: " + w)
        self.log(f"Fields: {len(fld.parts)} parts ({', '.join(fld.parts)}) solved in {out['t_solve']:.1f} s"
                 + ("" if fld.insulator is None else f"; insulators (εr > 1) on {fld.insulator.mean():.0%} of the grid"))
        ex = out["ex"]
        if out["flow"] is not None:
            f = out["flow"]
            self._flow = (out["flow_key"], f)
            self.log(f"Gas flow: {ex['slm']:g} slm enters a {2 * ex['r_in']:g} mm disc at {f.inlet_speed:.3g} m/s mean; "
                     f"{f.exit_fraction:.1%} leaves through the far end, the rest through the sides")
            if f.warning(out["io"]["T"]):
                self.log("  Note: " + f.warning(out["io"]["T"]))
            self._draw_flow(f, out)
        if res.sc_history:
            hst = res.sc_history
            self.log(f"Space charge at {ex['I']:g} nA: peak potential " + " → ".join(f"{p:.3g}" for _, p, _ in hst)
                     + f" V over {len(hst)} solves, {hst[-1][2] / 1.602e-19:.3g} charges in the device"
                     + ("" if res.sc_converged else "  (NOT settled: raise the iterations, or the current is more "
                                                    "than this device carries and ions pile up)"))
        st = res.status
        species = out["species"]
        self.log(f"Flown {len(st)} ions of {len(species)} species in {out['t_fly']:.1f} s "
                 f"(dt {info['dt'] * 1e9:.3g} ns, max {info['t_max'] * 1e3:.3g} ms)")
        L = info["z_end"] - info["z_start"]
        for d in species_stats(res, species, L):
            msg = (f"{d['name']} (m/z {d['mz']:g}, {d['z']}+): transmitted {d['through']:.0%}, lost on electrodes "
                   f"{d['lost']:.0%}, still flying {d['flying']:.0%}")
            if np.isfinite(d["transit"]):
                msg += f"; mean transit {d['transit'] * 1e3:.4g} ms, mean axial speed {d['speed']:.4g} m/s"
            self.log(msg)
            if res.heat is not None:
                self.log(f"  Heating {d['name']}: " + res.heat.summary(d["mask"] & (st >= 0)))
        self.last_table = (["species", "mz", "z", "status(1=out,2=lost,0=flying)", "x_mm", "y_mm", "z_mm", "t_ms"],
                           [[species[k].name, species[k].mz, species[k].z, int(s), *(p / MM), t * 1e3]
                            for k, s, p, t in zip(res.species, st, res.final_pos, res.t_end)])
        self.metrics = flight_metrics(res, species, L)
        io = out["io"]
        z0, z1 = info["z_start"], info["z_end"]
        self.speeds = {"span": (z0, z1), "gas": gas_line(out["gas_u"], (io["x0"] * MM, io["y0"] * MM), z0, z1),
                       "ions": ion_speeds(res, species, z0, z1)}
        self.set_exit(ExitIons.from_result(f"device run at {io['p']:g} Torr", res, species,
                                           (io["x0"] * MM, io["y0"] * MM), io["p"], io["T"]))
        self._draw_potential(fld, res.space_charge)
        self._draw_traj(fld, res, info, species)
        self._draw_results(res, info, species)
        draw_heating(self.plots.figs["Heating"], res.heat, st,
                     title=f"{', '.join(sp.name for sp in species)} in {out['io']['p']:g} Torr: ion heating along the ion axis")
        self.plots.draw("Heating")

    # ================= drawing =================
    def _repaint(self):
        if self.dev is None:
            return
        sm = self.view.paint({e.name: self._eff(e)[0] for e in self.dev.electrodes}, self.selected,
                             rf={e.name for e in self.dev.electrodes if self._eff(e)[1] and not e.eps},
                             insulators={e.name for e in self.dev.electrodes if e.eps > 0})
        if getattr(self, "_cbar", None) is not None:
            self._cbar.update_normal(sm)
        self.plots.draw("Geometry")

    def _draw_geometry(self):
        fig = self.plots.fig("Geometry", show=False)
        self.view.reset()
        self._cbar = None
        if self.dev is None:
            ax = fig.add_subplot(111); ax.axis("off")
            ax.text(0.5, 0.55, "Open a DXF, STEP or STL file", ha="center", va="center", fontsize=13)
            ax.text(0.5, 0.42, "Examples: examples/geometry/quad_rods.dxf, ring_funnel_profile.dxf, twave_board.dxf",
                    ha="center", va="center", fontsize=9, color="0.4")
            self.plots.draw("Geometry")
            return
        dev = self.dev
        if dev.kind == "cad3d":
            axes = fig.subplots(1, 2, sharey=True)
            from .funnel_tab import _mesh_section
            lo, hi = dev.extent()
            mid = (lo + hi) / 2
            for ax, cut, keep, at, lab in ((axes[0], 1, 0, mid[1], "x"), (axes[1], 0, 1, mid[0], "y")):
                for e in dev.electrodes:
                    seg = _mesh_section(e.tris / MM, cut, keep, at / MM)
                    if len(seg):
                        lc = LineCollection(seg, lw=1.3)
                        ax.add_collection(lc)
                        self.view.add(e.name, lc)
                ax.set_xlim(lo[keep] / MM - 1, hi[keep] / MM + 1); ax.set_ylim(lo[2] / MM - 1, hi[2] / MM + 1)
                ax.set_xlabel(f"{lab} (mm)")
                ax.set_title(f"section at {'y' if cut == 1 else 'x'} = {at / MM:.3g} mm", fontsize=9)
            axes[0].set_ylabel("z (mm, ion axis)")
            ax = axes[1]
        else:
            ax = fig.add_subplot(111)
            for e in dev.electrodes:
                s = e.shape
                verts = [s.outer / MM] + [h / MM for h in s.holes]
                codes = []
                for v in verts:
                    codes += [Path.MOVETO] + [Path.LINETO] * (len(v) - 1)
                p = PathPatch(Path(np.vstack(verts), codes))
                ax.add_patch(p)
                self.view.add(e.name, p)
                tw = self._eff(e)[3]
                if tw >= 0:
                    c = s.centroid / MM
                    ax.text(c[0], c[1], str(tw + 1), ha="center", va="center", fontsize=6, zorder=5)
            b = np.array([e.shape.bbox for e in dev.electrodes]) / MM
            lo, hi = b[:, :2].min(0), b[:, 2:].max(0)
            m = 0.05 * float(np.max(hi - lo))
            ax.set_xlim(lo[0] - m, hi[0] + m); ax.set_ylim(lo[1] - m, hi[1] + m)
            ax.set_aspect("equal", adjustable="datalim")
            labels = {"planar2d": ("x (mm)", "y (mm)", "cross-section; ions travel out of the page (z)"),
                      "axisym": ("z (mm, ion axis)", "r (mm)", "half profile, spun about the axis (r = 0)"),
                      "boards": ("z (mm, along the ion path)", "x (mm, across)", f"board face; boards {dev.board_gap / MM:g} mm apart")}
            xl, yl, tt = labels[dev.kind]
            ax.set_xlabel(xl); ax.set_ylabel(yl)
            ax.set_title(tt + ".  Colour: DC, hatched: RF, numbers: TWAVE phase, tan: insulator", fontsize=9)
            if dev.kind == "axisym":
                ax.axhline(0, color="k", lw=0.6, ls="-.")
        sm = self.view.paint({e.name: self._eff(e)[0] for e in dev.electrodes}, self.selected,
                             rf={e.name for e in dev.electrodes if self._eff(e)[1] and not e.eps},
                             insulators={e.name for e in dev.electrodes if e.eps > 0})
        self._cbar = fig.colorbar(sm, ax=ax, label="DC voltage (V)")
        self.plots.draw("Geometry")

    def _draw_potential(self, fld, sc=None):
        secs = fld.sections(0.0)
        if sc is not None:                                 # space charge on the first section's plane
            s0 = secs[0]
            phi = _device_section(sc.phi, fld)
            secs.append(dict(s0, phi=phi, title=s0["title"] + ": space-charge potential"))
        fig = self.plots.fig("Potential", show=False)
        axes = np.atleast_1d(fig.subplots(len(secs), 1))
        for ax, s in zip(axes, secs):
            phi = np.ma.masked_where(s["mask"], s["phi"])
            im = ax.imshow(phi, origin="lower", aspect="auto" if s["h"][-1] - s["h"][0] > 3 * (s["v"][-1] - s["v"][0]) else "equal",
                           extent=[s["h"][0], s["h"][-1], s["v"][0], s["v"][-1]], cmap="viridis")
            _outline(ax, s)
            ax.set_xlabel(s["hl"]); ax.set_ylabel(s["vl"]); ax.set_title(f"Potential at t = 0, {s['title']}", fontsize=9)
            fig.colorbar(im, ax=ax, label="V")
        self.plots.draw("Potential")

    def _draw_traj(self, fld, res, info, species):
        secs = fld.sections(0.0)
        fig = self.plots.fig("Trajectories")
        if res.traj is None:
            self.plots.draw("Trajectories")
            return
        styles = [path_style(res.status[i], res.species[i], len(species)) for i in range(res.traj.shape[1])]
        if fld.kind == "planar2d":
            ax1, ax2 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 2]})
            s = secs[0]
            _outline(ax1, s)
            for i, (c, ls) in enumerate(styles):
                ax1.plot(res.traj[:, i, 0] / MM, res.traj[:, i, 1] / MM, lw=0.5, color=c, ls=ls)
                ax2.plot(res.traj[:, i, 2] / MM, res.traj[:, i, 0] / MM, lw=0.5, color=c, ls=ls)
            species_legend(ax2, species, loc="upper right")
            ax1.set_aspect("equal"); ax1.set_xlabel("x (mm)"); ax1.set_ylabel("y (mm)"); ax1.set_title("end view", fontsize=9)
            ax2.set_xlabel("z (mm)"); ax2.set_ylabel("x (mm)"); ax2.set_title("side view", fontsize=9)
        else:
            axes = np.atleast_1d(fig.subplots(len(secs), 1, sharex=True))
            for ax, s, comp in zip(axes, secs, (0, 1)):
                _outline(ax, s)
                for i, (c, ls) in enumerate(styles):
                    tv = np.hypot(res.traj[:, i, 0], res.traj[:, i, 1]) if fld.kind == "axisym" else res.traj[:, i, comp]
                    ax.plot(res.traj[:, i, 2] / MM, tv / MM, lw=0.5, color=c, ls=ls)
                ax.set_ylabel(s["vl"]); ax.axvline(info["z_end"] / MM, color="k", ls=":", lw=1)
            axes[-1].set_xlabel("z (mm, ion axis)")
            species_legend(axes[0], species, loc="upper right")
        fig.suptitle(fate_caption(len(species)), fontsize=9)
        self.plots.draw("Trajectories")
        draw_traj3d(self.plots.fig("3D", show=False), res.traj, styles, getattr(fld.dg, "g", fld.dg), fate_caption(len(species)),
                    z_end=info["z_end"])
        self.plots.draw("3D")

    def _draw_results(self, res, info, species):
        fig = self.plots.fig("Results", show=False)
        ax1, ax2 = fig.subplots(1, 2)
        tr = res.status == 1
        if tr.any():
            bins = np.linspace(np.nanmin(res.t_end[tr]), np.nanmax(res.t_end[tr]), 41) * 1e3
            for k, sp in enumerate(species):
                m = tr & (res.species == k)
                if m.any():
                    ax1.hist(res.t_end[m] * 1e3, bins=bins, alpha=0.6, color=species_color(k), label=sp.name)
                    ax2.plot(res.final_pos[m, 0] / MM, res.final_pos[m, 1] / MM, ".", ms=3, color=species_color(k))
            if len(species) > 1:
                ax1.legend(fontsize=7)
        ax1.set_xlabel("arrival time (ms)"); ax1.set_ylabel("ions"); ax1.set_title("Arrival times", fontsize=9)
        ax2.set_xlabel("x (mm)"); ax2.set_ylabel("y (mm)"); ax2.set_title("Where ions cross the detector", fontsize=9)
        ax2.set_aspect("equal", adjustable="datalim")
        self.plots.draw("Results")

    def _draw_flow(self, flow, out):
        sec = flow.section((out["io"]["x0"] * MM, out["io"]["y0"] * MM))
        fig = self.plots.fig("Gas flow", show=False)
        ax = fig.add_subplot(111)
        Z, A = sec["z"] / MM, sec["coords"] / MM
        speed = np.ma.masked_where(sec["mask"], np.hypot(sec["ut"], sec["uz"]))
        vmax = float(np.percentile(speed.compressed(), 99)) if speed.count() else 1.0
        im = ax.imshow(speed, origin="lower", aspect="auto", cmap="viridis", extent=[Z[0], Z[-1], A[0], A[-1]], vmax=vmax)
        fig.colorbar(im, ax=ax, label="gas speed (m/s)", extend="max")
        ax.streamplot(Z, A, np.where(sec["mask"], 0, sec["uz"]), np.where(sec["mask"], 0, sec["ut"]),
                      color="w", density=1.2, linewidth=0.6, arrowsize=0.7)
        ax.contourf(Z, A, sec["mask"].astype(float), levels=[0.5, 1.5], colors="0.55")
        ax.set_xlim(Z[0], Z[-1]); ax.set_ylim(A[0], A[-1])
        ax.set_xlabel("z (mm, ion axis)"); ax.set_ylabel("r (mm)" if flow.gmap.kind == "axisym" else "x (mm)")
        ax.set_title(f"Gas flow, {out['ex']['slm']:g} slm (viscous, pumped through every open side); "
                     f"{flow.exit_fraction:.1%} leaves through the far end", fontsize=9)
        self.plots.draw("Gas flow")

    # ================= project files =================
    def save_setup(self):
        if self.dev is None:
            self._error("Nothing to save", "Open a geometry file first.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save device setup", os.path.splitext(self.dev.source)[0] + ".ionsim",
                                                        "Ion Sim setup (*.ionsim)")
        if path:
            self.save_setup_to(path)

    def save_setup_to(self, path):
        d = self.setup_dict(os.path.dirname(os.path.abspath(path)))
        with open(path, "w") as f:
            json.dump(d, f, indent=1)
        self.log(f"Saved setup to {path}")

    def setup_dict(self, base_dir=None):
        """Everything this tab needs to come back: the device (electrodes, wiring, drivers), the geometry
        file (relative to base_dir when given) and every setting. None when no geometry is open."""
        if self.dev is None:
            return None
        d = self.dev.to_dict()
        d["source"] = os.path.relpath(self.dev.source, base_dir) if base_dir else os.path.abspath(self.dev.source)
        d["gui"] = {"units": self.units.currentText(), "axis": self.axis.currentText(), "quality": self.quality.currentText(),
                    "h_mm": self.h.value(), "ions": self.ions.values(), "species": self._species_list(),
                    "drive": self.drive.values(), "geo": self.geo.values(),
                    "extra": self.extra.values(), "flow": self.flow.currentIndex(), "heat": self.heat.values()}
        return d

    def open_setup(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open device setup", examples_dir(), "Ion Sim setup (*.ionsim)")
        if path:
            self.open_setup_from(path)

    def open_setup_from(self, path):
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception as e:
            self._error("Could not read the setup", str(e))
            return
        if self.apply_setup(d, os.path.dirname(os.path.abspath(path))):
            self.log(f"Opened setup {os.path.basename(path)}")

    def apply_setup(self, d, base_dir=""):
        """Bring back a setup_dict. Returns False (and says why) when its geometry file is missing."""
        src = d.get("source", "")
        src = src if os.path.isabs(src) else os.path.normpath(os.path.join(base_dir, src))
        if not os.path.exists(src):
            self._error("Geometry file missing", f"The setup refers to\n{src}\nwhich was not found.")
            return False
        gui = d.get("gui", {})
        self._set_combo(self.units, gui.get("units", "mm")); self._set_combo(self.axis, gui.get("axis", "z"))
        self.flow.setCurrentIndex(int(gui.get("flow", 0)))
        if "heat" in gui:
            self.heat.set_values(gui["heat"])
        for form, key in ((self.geo, "geo"), (self.drive, "drive"), (self.ions, "ions"), (self.extra, "extra")):
            form.blockSignals(True)
            for k, v in gui.get(key, {}).items():
                if k in form.boxes:
                    form.set(k, v)
            form.blockSignals(False)
        self.both.setChecked(d.get("both_boards", True)); self.open_in.setChecked(d.get("open_entrance", True))
        if not self.load_file(src, d.get("kind")):
            return False
        missing = self.dev.apply_dict(d)
        self._adopt_drivers()
        for form, key in ((self.ions, "ions"),):
            for k, v in gui.get(key, {}).items():
                if k in form.boxes:
                    form.set(k, v)
        old = gui.get("ions", {})
        if gui.get("species"):
            self.ion_table.from_list(gui["species"])
        elif "mz" in old:                          # setups from before ion mixtures: one species
            self.ion_table.set_species([IonSpecies(f"m/z {old['mz']:g}", old["mz"], int(old.get("z", 1)),
                                                   old.get("K0", 1.0), int(old.get("n", 200)))])
        self.drive.blockSignals(True); self.drive.set("pitch", self.dev.twave.pitch / MM); self.drive.blockSignals(False)
        self._drive_changed()
        self._set_combo(self.quality, gui.get("quality", "Quick"))
        if gui.get("quality") == "Custom":
            self.h.setValue(gui.get("h_mm", self.h.value()))
        self._fill_table(); self._draw_geometry()
        if missing:
            self.log("  Note: these electrodes are no longer in the file: " + ", ".join(missing))
        return True

    def _species_list(self):
        try:
            return self.ion_table.to_list()
        except ValueError:
            return []

    def _adopt_drivers(self):
        """Drivers saved in a setup go into the library. One that differs from the library's
        driver of the same name comes in as '<name> (from setup)' so neither is lost."""
        changed = False
        for name, drv in list(self.dev.drivers.items()):
            have = self.store.get(name)
            if have is not None and have.to_dict() == drv.to_dict():
                continue
            new = name
            if have is not None:
                new = self.store.lib.add(drv.copy(f"{name} (from setup)"), replace=False)
                for e in self.dev.electrodes:
                    if e.drive.startswith(name + " / "):
                        e.drive = new + e.drive[len(name):]
                del self.dev.drivers[name]
                self.dev.drivers[new] = self.store.get(new).copy()
                self.log(f"  The setup's driver '{name}' differs from yours; added it as '{new}'.")
            else:
                self.store.lib.add(drv.copy())
                self.log(f"  Added the setup's driver '{name}' to your drivers.")
            changed = True
        if changed:
            self.store.touch()

    # ================= helpers =================
    # ================= electrode load =================
    def _load_grid(self):
        """The device's grid for a load solve (the last one if the geometry is unchanged)."""
        self._sync_drivers(); self._drive_to_device()
        key = self._grid_key()
        return self.grid[1] if self.grid and self.grid[0] == key else None, key

    def solve_load(self):
        if self.dev is None:
            self._error("No geometry", "Open a DXF or 3D file first.")
            return
        dg, key = self._load_grid()
        h, dev = self.h.value() * MM, self.dev
        by = "output" if self.nets.currentIndex() == 0 else "electrode"
        from ..circuit import device_capacitance, nets

        def job(progress):
            g = dg or dev.build(h)
            t0 = time.time()
            cap = device_capacitance(g, by, progress=lambda f: progress(0.05 + 0.95 * f))
            return dict(grid=g, key=key, cap=cap, t=time.time() - t0)

        self.log(f"Capacitance: {len(nets(dev, by))} nets, one field solve each…")
        self.start(job, self._show_load, "Solving the capacitance of each net…")

    def _show_load(self, out):
        self.grid = (out["key"], out["grid"])
        cap = self.cap = out["cap"]
        self.loaded = None
        self.log(f"Capacitance of {cap.n} nets in {out['t']:.1f} s" + (f" ({cap.note})" if cap.note else "")
                 + (f"; matrix symmetric to {cap.asym:.1%}" if cap.n > 1 else ""))
        for row in cap.table()[1:][:40]:
            self.log("  " + row.replace(", ", " | "))
        worst = cap.worst_pairs(5)
        if worst:
            self.log("  Strongest coupling (share of a swing on the second net that reaches the first if it floats): "
                     + "; ".join(f"{a} ← {b} {c:.1%} ({m * 1e12:.3g} pF)" for a, b, c, m in worst))
        self.load_label.setText(f"{cap.n} nets, {cap.total().sum() * 1e12:.4g} pF in all. Export it, or check the drive.")
        self.last_table = (["net", "electrodes", "total_F", "to_ground_F"] + [f"C_{n}_F" for n in cap.names],
                           [[n, " ".join(m), cap.C[i, i], cap.to_ground()[i], *cap.C[i]] for i, (n, m) in
                            enumerate(zip(cap.names, cap.members))])
        self._draw_cap()

    def _draw_cap(self):
        from ..circuit import short_names
        cap = self.cap
        fig = self.plots.fig("Capacitance")
        a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(width_ratios=[1.2, 1]))
        y = np.arange(cap.n)
        a1.barh(y, cap.to_ground() * 1e12, label="to ground")
        a1.barh(y, (cap.total() - cap.to_ground()) * 1e12, left=cap.to_ground() * 1e12, label="to other nets")
        a1.set_yticks(y, [f"{k + 1}  {n if len(n) < 26 else n[:24] + '…'}" for k, n in enumerate(short_names(cap.names))],
                      fontsize=7 if cap.n < 20 else 5)
        a1.invert_yaxis(); a1.set_xlabel("pF"); a1.legend(fontsize=7)
        a1.set_title("Capacitance of each net", fontsize=9)
        K = cap.coupling() * 100
        im = a2.imshow(K, cmap="magma_r", vmin=0, vmax=max(K.max(), 1e-3))
        a2.set_title("Crosstalk: % of net j's swing on net i if i floats", fontsize=9)
        a2.set_xlabel("net j"); a2.set_ylabel("net i")
        if cap.n <= 16:
            a2.set_xticks(y, [str(k + 1) for k in y], fontsize=7); a2.set_yticks(y, [str(k + 1) for k in y], fontsize=7)
        fig.colorbar(im, ax=a2, shrink=0.8, label="%")
        self.plots.draw("Capacitance")

    def export_load(self):
        if self.cap is None:
            self._error("Electrode load", "Press “Capacitance and crosstalk” first.")
            return
        from ..circuit import spice_subckt, drive_netlist
        name = os.path.splitext(os.path.basename(self.dev.source or "device"))[0]
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export SPICE load", f"{name}_load.lib",
                                                        "SPICE library (*.lib *.sub *.inc);;Netlist with the drive (*.cir)")
        if not path:
            return
        if path.lower().endswith(".cir"):
            v = self.load.values()
            text = drive_netlist(self.dev, self.cap, v["r"], v["l"] * 1e-9, periods=int(v["per"]), title=name,
                                 r_dc=v["rdc"])[0]
        else:
            text = spice_subckt(self.cap, title=name)
        with open(path, "w") as f:
            f.write(text)
        self.log(f"Wrote {path}: " + ("the load with every net driven by its waveform" if path.lower().endswith(".cir")
                                     else "subcircuit IONSIM_LOAD, one pin per net; .include it and connect your driver"))

    def check_drive(self):
        if self.cap is None:
            self._error("Electrode load", "Press “Capacitance and crosstalk” first.")
            return
        self._sync_drivers(); self._drive_to_device()
        from ..circuit import loaded_drive
        from ..waveforms import find_simulator
        if not find_simulator("ngspice")[1]:
            self._error("ngspice", "ngspice was not found. Install it (ngspice.sourceforge.io), or use Export SPICE load "
                                   "and run the .cir netlist in LTspice, then import the electrode voltages on the Drivers tab.")
            return
        v, dev, cap = self.load.values(), self.dev, self.cap

        def job(progress):
            progress(0.1)
            return loaded_drive(dev, cap, v["r"], v["l"] * 1e-9, periods=int(v["per"]), r_dc=v["rdc"])
        self.start(job, self._show_loaded, "Running ngspice with the electrode load…")

    def _show_loaded(self, ld):
        self.loaded = ld
        rep = ld.report()
        self.log(f"Drive with the load ({ld.window * 1e6:.4g} µs window, steady state):")
        for r in rep:
            if r["driven"]:
                self.log(f"  {r['name']}: swing {r['swing_src']:.4g} V at the source → {r['swing_el']:.4g} V on the "
                         f"electrodes ({r['swing_el'] / r['swing_src'] - 1:+.2%}), largest difference {r['max_err']:.3g} V, "
                         f"current {r['i_rms'] * 1e3:.3g} mA rms ({r['i_peak'] * 1e3:.3g} mA peak), {r['var']:.3g} VA reactive")
            else:
                self.log(f"  {r['name']} (held at {ld.v_src[rep.index(r)][0]:.4g} V): crosstalk {r['swing_el']:.3g} V pk-pk, "
                         f"{r['max_err']:.3g} V at worst")
        self.last_table = (["t_s"] + [f"src {n}" for n in ld.cap.names] + [f"el {n}" for n in ld.cap.names]
                           + [f"I {n}" for n in ld.cap.names],
                           [[t, *ld.v_src[:, k], *ld.v_el[:, k], *ld.i_src[:, k]] for k, t in enumerate(ld.t)])
        self.load_label.setText("Checked. “Use the loaded drive” flies ions with these electrode voltages.")
        self._draw_loaded()

    def _draw_loaded(self):
        from ..circuit import short_names
        ld = self.loaded
        names = short_names(ld.cap.names)
        fig = self.plots.fig("Loaded drive")
        a1, a2 = fig.subplots(2, 1, sharex=True)
        rep = ld.report()
        driven = [k for k, r in enumerate(rep) if r["driven"]]
        quiet = [k for k, r in enumerate(rep) if not r["driven"]]
        span = ld.t * 1e6
        for c, k in enumerate(driven[:6]):
            col = f"C{c}"
            a1.plot(span, ld.v_src[k], color=col, lw=0.8, ls="--")
            a1.plot(span, ld.v_el[k], color=col, lw=1.0, label=names[k])
        a1.set_ylabel("V"); a1.set_title("Driven nets: source (dashed) and electrodes (solid)", fontsize=9)
        if driven:
            a1.legend(fontsize=6, loc="upper right")
        if quiet:
            for c, k in enumerate(sorted(quiet, key=lambda k: -rep[k]["max_err"])[:6]):
                a2.plot(span, ld.v_el[k] - ld.v_src[k], lw=0.9, label=names[k])
            a2.set_title("Crosstalk onto nets held at DC (electrode minus supply)", fontsize=9)
            a2.legend(fontsize=6, loc="upper right")
        else:
            for c, k in enumerate(driven[:6]):
                a2.plot(span, ld.v_el[k] - ld.v_src[k], lw=0.9, label=names[k])
            a2.set_title("Electrode minus source", fontsize=9)
        a2.set_xlabel("time (µs)"); a2.set_ylabel("V")
        self.plots.draw("Loaded drive")

    def use_loaded(self):
        if self.loaded is None:
            self._error("Electrode load", "Press “Check the drive with the load” first.")
            return
        from ..circuit import apply_load
        on = apply_load(self.dev, self.loaded)
        if not on:
            self.load_label.setText("The load moves no net by more than 0.1 % of the drive: nothing to add.")
            self.log("The load changes every net by under 0.1 % of the drive, so the flight would not change; nothing added.")
            return
        worst = max((float(np.max(np.abs(tr.v))) for _, _, tr in on), default=0.0)
        self.load_label.setText(f"Flying with the load on {len(on)} nets (up to {worst:.3g} V off the set drive). "
                                "Solve and fly to see the effect.")
        self.log(f"The load is on: {len(on)} nets get their electrode-minus-source voltage added (largest {worst:.3g} V). "
                 "It is saved with the setup; “Take the load off” removes it.")

    def clear_load(self):
        if self.dev is not None and self.dev.load:
            from ..circuit import clear_load
            clear_load(self.dev)
            self.load_label.setText("The load is off: the electrodes carry the drive as set.")
            self.log("The load is off.")

    def _error(self, title, text):
        self.log(f"{title}: {text}")
        if QtWidgets.QApplication.platformName() != "offscreen":
            QtWidgets.QMessageBox.warning(self, title, text)

    @staticmethod
    def _set_combo(combo, value, data=False):
        i = combo.findData(value) if data else combo.findText(value)
        if i >= 0:
            combo.setCurrentIndex(i)


def _outline(ax, s):
    """Electrodes grey, insulators in board colour, on a DeviceField section dict."""
    ax.contourf(s["h"], s["v"], s["mask"].astype(float), levels=[0.5, 1.5], colors="0.55")
    if s.get("ins") is not None and s["ins"].any():
        ax.contourf(s["h"], s["v"], (s["ins"] & ~s["mask"]).astype(float), levels=[0.5, 1.5], colors=[INSULATOR])


def _device_section(arr, fld):
    """A node array of the field's grid on the plane of fld.sections()[0]."""
    if fld.kind == "planar2d":
        return arr.T
    if fld.kind == "axisym":
        return np.concatenate([arr[:0:-1], arr], axis=0)
    g = fld.dg.g
    return arr[:, int(np.argmin(np.abs(g.y))), :]
