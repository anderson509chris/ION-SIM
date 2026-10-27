"""Ion funnel tab: stacked-ring, planar two-board, or imported CAD geometry."""
import os
import matplotlib
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from PySide6 import QtCore, QtWidgets

from ..funnel import RingFunnel, PlanarFunnel, CadFunnel, FunnelElectrics, run_funnel, pseudopotential, gas_flow
from ..laplace import Drive
from ..cad import load_bodies, FILE_FILTER, UNITS
from ..twave import mobility
from .common import ParamForm, SimTab
from .electrode_view import ElectrodeView, INSULATOR
from .heating_view import HeatingBox, draw_heating
from .ion_table import IonTable, species_color, path_style, fate_caption, species_legend, species_stats
from ..ions import IonSpecies

TYPES = ["Ring funnel (stacked PCBs)", "Planar funnel (two boards)", "Imported CAD (STEP / STL / OBJ)"]
MODES = ["Trajectories", "Transmission vs m/z"]
FLOWS = ["Still gas", "Uniform axial speed", "Flow field from gas throughput"]
MAX_NODES = 40_000_000
MM = 1e-3



def _entrance_bore(bodies, center):
    """Inner radius (m) of the body nearest the entrance, or None if it blocks the axis."""
    first = min(bodies, key=lambda b: b[1][:, :, 2].mean())[1].reshape(-1, 3)
    r = np.hypot(first[:, 0] - center[0], first[:, 1] - center[1])
    span = np.ptp(first[:, :2], 0).max()
    return float(r.min()) if r.min() > 0.02 * span else None

class FunnelTab(SimTab):
    def __init__(self, parent=None):
        super().__init__(["Geometry", "Potential", "Trajectories", "Transmission", "Exit beam", "Gas flow", "Heating"], parent)
        self.kind = QtWidgets.QComboBox(); self.kind.addItems(TYPES)
        kind_box = QtWidgets.QGroupBox("Funnel type")
        QtWidgets.QVBoxLayout(kind_box).addWidget(self.kind)
        r, p = RingFunnel(), PlanarFunnel()
        self.ring = ParamForm("Ring funnel", [
            ("n", "Rings", r.n_rings, 3, 1000, 0, 1, ""),
            ("pitch", "Ring pitch", r.pitch / MM, 0.1, 20, 3, 0.1, "mm"),
            ("thick", "Ring thickness", r.thickness / MM, 0.02, 10, 3, 0.05, "mm"),
            ("id_in", "Entrance ID", r.id_in / MM, 0.5, 200, 2, 0.5, "mm"),
            ("id_out", "Last ring ID", r.id_out / MM, 0.2, 200, 2, 0.1, "mm"),
            ("straight", "Constant-ID rings first", r.n_straight, 0, 1000, 0, 1, ""),
            ("od", "Plate outer diameter", r.od / MM, 1, 400, 1, 1, "mm"),
            ("exit_id", "Exit orifice diameter", r.exit_id / MM, 0.1, 50, 2, 0.1, "mm"),
            ("exit_gap", "Last ring to exit plate", r.exit_gap / MM, 0.1, 20, 2, 0.1, "mm"),
            ("exit_t", "Exit plate thickness", r.exit_thickness / MM, 0.02, 10, 2, 0.1, "mm"),
            ("cu_w", "Copper ring width (0 = metal plate)", r.copper_width / MM, 0, 200, 2, 0.5, "mm"),
            ("eps", "Board εr (FR4 4.4)", r.board_eps, 1, 100, 2, 0.1, ""),
            ("h", "Grid spacing", r.h / MM, 0.005, 1, 3, 0.01, "mm"),
        ])
        self.ring.boxes["cu_w"].setToolTip("PCB rings: copper around the bore (plated through, both faces), "
                                           "FR4 board beyond it out to the plate diameter")
        self.planar = ParamForm("Planar funnel (boards at ±gap/2)", [
            ("n", "Strips per board", p.n_strips, 3, 1000, 0, 1, ""),
            ("pitch", "Strip pitch", p.pitch / MM, 0.1, 20, 3, 0.1, "mm"),
            ("w", "Strip width", p.strip_width / MM, 0.02, 10, 3, 0.05, "mm"),
            ("gap", "Board gap", p.gap / MM, 0.5, 50, 2, 0.5, "mm"),
            ("w_in", "Strip length at entrance", p.width_in / MM, 0.5, 200, 2, 0.5, "mm"),
            ("w_out", "Strip length at exit", p.width_out / MM, 0.2, 200, 2, 0.1, "mm"),
            ("straight", "Constant-length strips first", p.n_straight, 0, 1000, 0, 1, ""),
            ("g_gap", "Strip end to guard", p.guard_gap / MM, 0.05, 10, 2, 0.05, "mm"),
            ("g_band", "Guard width", p.guard_band / MM, 0.2, 50, 2, 0.5, "mm"),
            ("ex_w", "Exit slit width (x)", p.exit_width / MM, 0.1, 50, 2, 0.1, "mm"),
            ("ex_h", "Exit slit height (y)", p.exit_height / MM, 0.1, 50, 2, 0.1, "mm"),
            ("exit_gap", "Last strip to exit plate", p.exit_gap / MM, 0.1, 20, 2, 0.1, "mm"),
            ("exit_t", "Exit plate thickness", p.exit_thickness / MM, 0.02, 10, 2, 0.1, "mm"),
            ("board_t", "Board thickness (0 = none)", p.board_thickness / MM, 0, 20, 2, 0.1, "mm"),
            ("eps", "Board εr (FR4 4.4)", p.board_eps, 1, 100, 2, 0.1, ""),
            ("h", "Grid spacing", p.h / MM, 0.02, 2, 3, 0.02, "mm"),
        ])
        self.cad_box = self._build_cad_box()
        e = FunnelElectrics()
        self.elec = ParamForm("Electrics", [
            ("rf", "RF amplitude (0-peak)", e.rf, 0, 5000, 1, 5, "V"),
            ("freq", "RF frequency", e.freq / 1e3, 1, 100000, 0, 50, "kHz"),
            ("grad", "DC gradient", e.gradient, -1000, 1000, 2, 1, "V/cm"),
            ("dc_exit", "DC on last electrode", e.dc_exit, -10000, 10000, 1, 1, "V"),
            ("exit_off", "Exit plate below last", e.exit_offset, -1000, 1000, 1, 1, "V"),
            ("guard", "Guard above strips", e.guard_offset, -1000, 1000, 1, 1, "V"),
        ])
        self.gas = ParamForm("Gas", [
            ("p", "Pressure", 2.0, 0.01, 100, 3, 0.1, "Torr"),
            ("T", "Temperature", 300, 50, 1000, 0, 5, "K"),
            ("u", "Axial gas speed", 0.0, -2000, 2000, 1, 5, "m/s"),
            ("slm", "Gas throughput", 1.0, 0.001, 100, 3, 0.1, "slm"),
            ("r_in", "Gas inlet radius (0 = entrance bore)", 0.0, 0, 500, 2, 0.5, "mm"),
        ])
        self.flow = QtWidgets.QComboBox(); self.flow.addItems(FLOWS)
        self.flow.setToolTip("Flow field: viscous flow from the inlet disc on the entrance face, pumped out through every "
                             "other open side (between electrodes, through the exit), scaled to the throughput")
        self.gas.layout().insertRow(2, "Gas flow", self.flow)
        self.flow.currentIndexChanged.connect(self._flow_changed)
        self.space = ParamForm("Space charge", [
            ("I", "Ion current (0 = off)", 0.0, 0, 1e6, 3, 1, "nA"),
            ("it", "Iterations", 4, 1, 30, 0, 1, ""),
        ])
        self.space.setToolTip("The ions are flown several times; each flight's charge (current x time spent) "
                              "sets the space-charge field for the next. Each iteration costs one more flight.")
        self.ion_table = IonTable("Ions (one row per species, flown together)", [IonSpecies("m/z 500", 500, 1, 1.0, 200)])
        self.ion_table.setToolTip("Every species is flown in the same run. The m/z scan uses the first row as its "
                                  "reference (its charge, and K0 scaled from its m/z).")
        self.ions = ParamForm("Start and time step", [
            ("r0", "Start radius", 8.0, 0.0, 100, 2, 0.5, "mm"),
            ("steps", "Steps per RF cycle", 30, 8, 400, 0, 2, ""),
            ("tmax", "Max flight time (0 = auto)", 0.0, 0, 1000, 2, 1, "ms"),
        ])
        self.mode = QtWidgets.QComboBox(); self.mode.addItems(MODES)
        mode_box = QtWidgets.QGroupBox("Run mode")
        QtWidgets.QVBoxLayout(mode_box).addWidget(self.mode)
        self.scan = ParamForm("m/z scan", [
            ("m0", "From m/z", 20, 1, 1e6, 0, 10, ""),
            ("m1", "To m/z", 3000, 1, 1e6, 0, 100, ""),
            ("nm", "Points (log spaced)", 12, 2, 200, 0, 1, ""),
            ("per", "Ions per point", 60, 5, 10000, 0, 10, ""),
            ("expo", "K0 ∝ (m/z)^-n, n =", 0.667, 0, 2, 3, 0.05, ""),
        ])
        self.heat = HeatingBox()
        for w in (kind_box, self.ring, self.planar, self.cad_box, self.elec, self.gas, self.space, self.ion_table, self.ions, self.heat,
                  mode_box, self.scan):
            self.form_layout.addWidget(w)
        self.form_layout.addStretch(1)

        self.bodies, self.cad_path = None, None
        self._geo_cache = None        # (key, grid, solver, last phi parts for warm starts)
        self._flow_cache = None       # (key, GasFlow)
        self.kind.currentIndexChanged.connect(self._kind_changed)
        self.mode.currentIndexChanged.connect(lambda i: self.scan.setVisible(i == 1))
        self.ring.changed.connect(self._preview); self.planar.changed.connect(self._preview)
        self.run_btn.clicked.connect(self.run)
        self.scan.setVisible(False)
        self._flow_changed()
        self._kind_changed()

    # ---------- imported geometry ----------
    def _build_cad_box(self):
        box = QtWidgets.QGroupBox("Imported geometry (one electrode per solid body)")
        v = QtWidgets.QVBoxLayout(box)
        row = QtWidgets.QHBoxLayout()
        load = QtWidgets.QPushButton("Load STEP / STL / OBJ...")
        load.clicked.connect(self._load_cad)
        self.cad_label = QtWidgets.QLabel("No file loaded"); self.cad_label.setWordWrap(True)
        row.addWidget(load); v.addLayout(row); v.addWidget(self.cad_label)
        form = QtWidgets.QFormLayout()
        self.units = QtWidgets.QComboBox(); self.units.addItems(list(UNITS))
        self.units.setToolTip("Units of the numbers in STL / OBJ files (STEP files carry their own units)")
        self.axis = QtWidgets.QComboBox(); self.axis.addItems(["z", "x", "y"])
        self.axis.setToolTip("CAD axis the ions travel along (it becomes z here); ions enter at the low end")
        form.addRow("Mesh file units", self.units); form.addRow("Ion axis in the CAD file", self.axis)
        v.addLayout(form)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Grid quality"))
        self.quality = QtWidgets.QComboBox(); self.quality.addItems(["Quick", "Detailed", "Custom"])
        self.quality.setCurrentText("Custom")
        self.quality.setToolTip("Quick: ~0.4 M nodes (seconds). Detailed: ~4 M nodes (final numbers). Custom: your spacing.")
        self.quality.currentIndexChanged.connect(self._quality_changed)
        row.addWidget(self.quality, 1); v.addLayout(row)
        self.cad_grid = ParamForm("Grid and ion path (mm, after the axis swap)", [
            ("h", "Grid spacing", 0.25, 0.02, 5, 3, 0.05, "mm"),
            ("pad", "Space around parts", 3.0, 0.5, 100, 1, 0.5, "mm"),
            ("x0", "Start centre x", 0.0, -1e4, 1e4, 3, 0.5, "mm"),
            ("y0", "Start centre y", 0.0, -1e4, 1e4, 3, 0.5, "mm"),
            ("z0", "Start z", 0.0, -1e4, 1e4, 3, 0.5, "mm"),
            ("z1", "Detector z (transmitted past here)", 10.0, -1e4, 1e4, 3, 0.5, "mm"),
        ])
        self.cad_grid.changed.connect(self._cad_info)
        self.cad_grid.changed.connect(self._preview)
        v.addWidget(self.cad_grid)
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Body", "DC (V)", "RF 0-pk (V)", "RF phase (°)", "Insulator εr"])
        self.table.horizontalHeaderItem(4).setToolTip("Leave blank for an electrode. A number makes the body an "
                                                     "insulator (PCB, spacer) with that relative permittivity.")
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(200)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.table.itemSelectionChanged.connect(self._repaint_cad)
        self.table.itemChanged.connect(self._repaint_cad)
        v.addWidget(self.table)
        v.addWidget(QtWidgets.QLabel("Click a part in the Geometry plot to find its row (Ctrl-click adds)."))
        row = QtWidgets.QHBoxLayout()
        self.last_exit = QtWidgets.QCheckBox("Last body is a DC exit plate"); self.last_exit.setChecked(True)
        auto = QtWidgets.QPushButton("Auto-assign funnel pattern")
        auto.setToolTip("Sort bodies along the ion axis; alternate RF 0/180°; DC ladder from the Electrics box")
        auto.clicked.connect(self._auto_assign)
        row.addWidget(self.last_exit); row.addWidget(auto)
        v.addLayout(row)
        return box

    def _quality_changed(self, *_):
        q = self.quality.currentText()
        self.cad_grid.boxes["h"].setEnabled(q == "Custom")
        if q != "Custom" and self.bodies:
            target = {"Quick": 400_000, "Detailed": 4_000_000}[q]
            lo, hi = CadFunnel(self.bodies).bounds()
            pad = self.cad_grid.values()["pad"] * MM
            size = (hi - lo) - 2 * CadFunnel(self.bodies).pad + 2 * pad
            self.cad_grid.set("h", float((np.prod(size) / target) ** (1 / 3)) / MM)

    def _cad_pick(self, name, additive):
        rows = [r for r in range(self.table.rowCount()) if self.table.item(r, 0).text() == name]
        if not rows:
            return
        if not additive:
            self.table.clearSelection()
        mode = self.table.selectionMode()
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.table.selectRow(rows[0])
        self.table.setSelectionMode(mode)
        self.table.scrollToItem(self.table.item(rows[0], 0))

    def _cad_hover(self, name):
        if name is None:
            self.status.setText("Ready")
            return
        if name in self._cad_dielectrics():
            self.status.setText(f"{name}: insulator, εr {self._cad_dielectrics()[name]:g}")
            return
        d = self._cad_drives().get(name)
        if d is not None:
            self.status.setText(f"{name}: DC {d.dc:g} V, RF {d.rf:g} V at {d.phase:g}°")

    def _repaint_cad(self, *_):
        if self.kind.currentIndex() != 2 or not self.bodies or not getattr(self, "cad_view", None):
            return
        try:
            drives = self._cad_drives()
        except Exception:
            return
        sel = {self.table.item(i.row(), 0).text() for i in self.table.selectedIndexes()}
        self.cad_view.paint({n: d.dc for n, d in drives.items()}, sel, insulators=self._cad_dielectrics())
        self.plots.draw("Geometry")

    def _load_cad(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load electrode geometry", "", FILE_FILTER)
        if path:
            self.load_geometry(path)

    def load_geometry(self, path):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            bodies = load_bodies(path, self.units.currentText(), self.axis.currentText())
        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.log(f"Could not load {path}: {e}")
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        if not bodies:
            self.log(f"No bodies found in {path}")
            return
        self.bodies, self.cad_path = bodies, path
        pts = np.concatenate([t.reshape(-1, 3) for _, t in bodies])
        lo, hi = pts.min(0) / MM, pts.max(0) / MM
        self.cad_grid.set("x0", (lo[0] + hi[0]) / 2); self.cad_grid.set("y0", (lo[1] + hi[1]) / 2)
        self.cad_grid.set("z0", lo[2] + 0.5); self.cad_grid.set("z1", hi[2] + 0.5)
        bore = _entrance_bore(bodies, (lo[:2] + hi[:2]) / 2 * MM)
        if bore:                                     # start ions well inside the first opening
            self.ions.set("r0", round(0.6 * bore / MM, 2))
        self.table.setRowCount(0)
        for name, _ in bodies:
            r = self.table.rowCount(); self.table.insertRow(r)
            it = QtWidgets.QTableWidgetItem(name); it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(r, 0, it)
        self._auto_assign()
        if self.quality.currentText() != "Custom":
            self._quality_changed()
        self.log(f"Loaded {os.path.basename(path)}: {len(bodies)} bodies, extent "
                 f"x {lo[0]:.2f}..{hi[0]:.2f}, y {lo[1]:.2f}..{hi[1]:.2f}, z {lo[2]:.2f}..{hi[2]:.2f} mm")
        self._cad_info(); self._preview()

    def _auto_assign(self):
        if not self.bodies:
            return
        d = CadFunnel(self.bodies, dielectrics=self._cad_dielectrics()).funnel_pattern(self._electrics(), self.last_exit.isChecked())
        self.table.blockSignals(True)
        for r in range(self.table.rowCount()):
            dr = d.get(self.table.item(r, 0).text())
            for c, val in enumerate([dr.dc, dr.rf, dr.phase] if dr else ["", "", ""], start=1):
                self.table.setItem(r, c, QtWidgets.QTableWidgetItem(val if isinstance(val, str) else f"{round(val, 3):g}"))
        self.table.blockSignals(False)
        self._repaint_cad()

    def _cad_drives(self):
        out, diel = {}, self._cad_dielectrics()
        for r in range(self.table.rowCount()):
            cell = lambda c: (self.table.item(r, c).text() if self.table.item(r, c) else "").strip() or "0"
            name = self.table.item(r, 0).text()
            if name not in diel:
                out[name] = Drive(float(cell(1)), float(cell(2)), float(cell(3)))
        return out

    def _cad_dielectrics(self):
        """Bodies with a permittivity in the Insulator column: {name: eps_r}."""
        out = {}
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 4)
            txt = it.text().strip() if it else ""
            try:
                if txt and float(txt) > 0:
                    out[self.table.item(r, 0).text()] = float(txt)
            except ValueError:
                pass
        return out

    def _cad(self):
        c = self.cad_grid.values()
        return CadFunnel(self.bodies, h=c["h"] * MM, pad=c["pad"] * MM, dielectrics=self._cad_dielectrics())

    def _cad_info(self, *_):
        if self.bodies:
            n = self._cad().nodes()
            self.cad_label.setText(f"{os.path.basename(self.cad_path)}: {len(self.bodies)} bodies, "
                                   f"3D grid ≈ {n / 1e6:.1f} M nodes" + ("  (too many: raise the grid spacing)" if n > MAX_NODES else ""))

    # ---------- parameters ----------
    def _electrics(self):
        e = self.elec.values()
        return FunnelElectrics(rf=e["rf"], freq=e["freq"] * 1e3, gradient=e["grad"], dc_exit=e["dc_exit"],
                               exit_offset=e["exit_off"], guard_offset=e["guard"])

    def _geometry(self):
        k = self.kind.currentIndex()
        if k == 0:
            v = self.ring.values()
            return RingFunnel(int(v["n"]), v["pitch"] * MM, v["thick"] * MM, v["id_in"] * MM, v["id_out"] * MM,
                              int(v["straight"]), v["od"] * MM, v["exit_id"] * MM, v["exit_gap"] * MM, v["exit_t"] * MM, v["h"] * MM,
                              v["cu_w"] * MM, v["eps"])
        if k == 1:
            v = self.planar.values()
            return PlanarFunnel(int(v["n"]), v["pitch"] * MM, v["w"] * MM, v["gap"] * MM, v["w_in"] * MM, v["w_out"] * MM,
                                int(v["straight"]), v["g_gap"] * MM, v["g_band"] * MM, v["ex_w"] * MM, v["ex_h"] * MM,
                                v["exit_gap"] * MM, v["exit_t"] * MM, v["h"] * MM, v["board_t"] * MM, v["eps"])
        if not self.bodies:
            raise ValueError("Load a geometry file first")
        return self._cad()

    def _kind_changed(self, *_):
        k = self.kind.currentIndex()
        self.ring.setVisible(k == 0); self.planar.setVisible(k == 1); self.cad_box.setVisible(k == 2)
        self.elec.boxes["guard"].setEnabled(k == 1)
        self.ions.set("r0", {0: 8.0, 1: 1.5, 2: self.ions.values()["r0"]}[k])
        self._preview()

    def _flow_changed(self, *_):
        k = self.flow.currentIndex()
        self.gas.boxes["u"].setEnabled(k == 1)
        self.gas.boxes["slm"].setEnabled(k == 2); self.gas.boxes["r_in"].setEnabled(k == 2)

    def _inlet_radius(self, geo, start):
        r = self.gas.values()["r_in"] * MM
        if r > 0:
            return r
        if isinstance(geo, RingFunnel):
            return geo.id_in / 2
        if isinstance(geo, PlanarFunnel):
            return geo.gap / 2
        bore = _entrance_bore(self.bodies, start)
        return bore or 3 * MM

    # ---------- plots ----------
    def _preview(self, *_):
        fig = self.plots.fig("Geometry", show=False)
        k = self.kind.currentIndex()
        try:
            geo = self._geometry()
        except Exception:
            ax = fig.add_subplot(111); ax.axis("off")
            ax.text(0.5, 0.5, "Load a STEP, STL or OBJ file", ha="center", va="center")
            self.plots.draw("Geometry")
            return
        if k == 0:
            ax = fig.add_subplot(111)
            for r0, r1, z0, z1 in geo.boards():
                for s in (1, -1):
                    ax.add_patch(_rect(z0, s * r0, z1 - z0, s * (r1 - r0), INSULATOR))
            for name, r0, r1, z0, z1 in geo.rings():
                c = "0.35" if name == "exit plate" else "tab:orange"
                for s in (1, -1):
                    ax.add_patch(_rect(z0, s * r0, z1 - z0, s * (r1 - r0), c))
            ax.set_xlim(-2, geo.z_end / MM + 2); ax.set_ylim(-geo.od / 2 / MM - 1, geo.od / 2 / MM + 1)
            ax.set_xlabel("z (mm)"); ax.set_ylabel("r (mm)")
            ax.set_title(f"{geo.n_rings} rings, ID {geo.id_in / MM:.1f} → {geo.id_out / MM:.1f} mm, exit orifice {geo.exit_id / MM:.1f} mm"
                         + (f"\ncopper (orange) {geo.copper_width / MM:g} mm wide on boards εr {geo.board_eps:g} (tan)" if geo.boards() else ""),
                         fontsize=9)
        elif k == 1:
            ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
            w = geo.strip_width / 2
            for L, z in zip(geo.strip_lengths(), geo.strip_z()):
                ax1.add_patch(_rect(z - w, -L / 2, 2 * w, L, "tab:orange"))
                for s in (1, -1):
                    ax1.add_patch(_rect(z - w, s * (L / 2 + geo.guard_gap), 2 * w, s * (geo.x_edge - L / 2 - geo.guard_gap), "tab:blue", 0.5))
            ex = geo.exit_width / 2
            for s in (1, -1):
                ax1.add_patch(_rect(geo.z_plate, s * ex, geo.exit_thickness, s * (geo.x_edge - ex), "0.35"))
                ax2.add_patch(_rect(geo.z_plate, s * geo.exit_height / 2, geo.exit_thickness, s * (geo.gap / 2 - geo.exit_height / 2), "0.35"))
                ax2.plot(geo.strip_z() / MM, np.full(geo.n_strips, s * geo.gap / 2 / MM), "|", color="tab:orange", ms=6)
                if geo.board_thickness > 0:
                    zb = geo.strip_z()[0] - geo.pitch
                    ax2.add_patch(_rect(zb, s * geo.gap / 2, geo.strip_z()[-1] + geo.pitch / 2 - zb, s * geo.board_thickness, INSULATOR))
            ax1.set_xlim(-2, geo.z_end / MM + 2); ax1.set_ylim(-geo.x_edge / MM - 1, geo.x_edge / MM + 1)
            ax2.set_ylim(-(geo.gap / 2 + geo.board_thickness) / MM - 0.5, (geo.gap / 2 + geo.board_thickness) / MM + 0.5)
            ax1.set_ylabel("x (mm)"); ax2.set_ylabel("y (mm)"); ax2.set_xlabel("z (mm)")
            ax1.set_title("Board face: RF strips (orange), DC guards (blue), exit plate (grey)")
        else:
            ax1, ax2 = fig.subplots(1, 2, sharey=True)
            c = self.cad_grid.values()
            if getattr(self, "cad_view", None) is None:
                self.cad_view = ElectrodeView(fig.canvas, self._cad_pick, self._cad_hover)
            self.cad_view.reset()
            for i, (name, t) in enumerate(geo.bodies):
                for ax, cut, keep, at in ((ax1, 1, 0, c["y0"]), (ax2, 0, 1, c["x0"])):
                    seg = _mesh_section(t / MM, cut, keep, at)
                    if len(seg):
                        lc = LineCollection(seg, lw=1.3)
                        ax.add_collection(lc)
                        self.cad_view.add(name, lc)
            try:
                drives = self._cad_drives()
                sm = self.cad_view.paint({n: d.dc for n, d in drives.items()}, insulators=self._cad_dielectrics())
                fig.colorbar(sm, ax=ax2, label="DC voltage (V)")
            except Exception:
                pass
            lo = np.min([t.reshape(-1, 3).min(0) for _, t in geo.bodies], axis=0) / MM
            hi = np.max([t.reshape(-1, 3).max(0) for _, t in geo.bodies], axis=0) / MM
            ax1.set_xlim(lo[0] - 1, hi[0] + 1); ax2.set_xlim(lo[1] - 1, hi[1] + 1)
            ax1.set_ylim(min(lo[2], c["z0"]) - 1, max(hi[2], c["z1"]) + 1)
            for ax, a0 in ((ax1, c["x0"]), (ax2, c["y0"])):
                ax.axhline(c["z0"], color="tab:green", lw=1); ax.axhline(c["z1"], color="tab:red", lw=1)
                ax.plot([a0], [c["z0"]], "^", color="tab:green")
            ax1.set_xlabel("x (mm)"); ax2.set_xlabel("y (mm)"); ax1.set_ylabel("z (mm, ion axis)")
            ax1.set_title(f"section at y = {c['y0']:g} mm", fontsize=9)
            ax2.set_title(f"section at x = {c['x0']:g} mm  (green: start, red: detector)", fontsize=9)
        self.plots.draw("Geometry")

    # ---------- run ----------
    def run(self):
        try:
            geo, el = self._geometry(), self._electrics()
            k = self.kind.currentIndex()
            drives = self._cad_drives() if k == 2 else geo.drives(el)
        except Exception as e:
            self.log(f"Input error: {e}")
            return
        io, gs, sc, mode = self.ions.values(), self.gas.values(), self.scan.values(), self.mode.currentIndex()
        if k == 2:
            c = self.cad_grid.values()
            start, z0, z1 = (c["x0"] * MM, c["y0"] * MM), c["z0"] * MM, c["z1"] * MM
            nodes, key = geo.nodes(), ("cad", self.cad_path, self.units.currentText(), self.axis.currentText(), c["h"], c["pad"],
                                       tuple(sorted(geo.dielectrics.items())))
        else:
            start, z0, z1 = (0.0, 0.0), geo.z_start, geo.z_end
            nodes, key = None, (k, repr(geo))
        if nodes and nodes > MAX_NODES:
            self.log(f"Grid would have {nodes / 1e6:.0f} M nodes; raise the grid spacing (limit {MAX_NODES / 1e6:.0f} M).")
            return
        cache = self._geo_cache if self._geo_cache and self._geo_cache[0] == key else None
        flow_mode, current, sc_it = self.flow.currentIndex(), self.space.values()["I"] * 1e-9, int(self.space.values()["it"])
        heat_model = self.heat.model()
        r_in = self._inlet_radius(geo, start) if flow_mode == 2 else 0.0
        flow_key = (key, gs["slm"], gs["p"], gs["T"], r_in, start)
        flow_cache = self._flow_cache[1] if flow_mode == 2 and cache and self._flow_cache and self._flow_cache[0] == flow_key else None
        try:
            species = self.ion_table.species()
        except ValueError as e:
            self.log(f"Input error: {e}")
            return
        ref = species[0]
        if mode == 0:
            species = [s for s in species if s.n > 0]
            mz, K0 = np.array([s.mz for s in species]), np.array([s.K0 for s in species])
            zc, n = np.array([s.z for s in species]), np.array([s.n for s in species])
            rec = min(int(n.sum()), max(40, 10 * len(species)))
        else:
            mz = np.geomspace(sc["m0"], sc["m1"], int(sc["nm"]))
            K0 = ref.K0 * (mz / ref.mz) ** (-sc["expo"]); n, rec, zc = int(sc["per"]), 0, ref.z

        def job(progress):
            import time
            t0 = time.time()
            guess = None
            if cache:
                grid, solver, guess = cache[1], cache[2], cache[3]
            else:
                progress(0.0)
                grid = geo.build()
                solver = grid.solver()
            fld = geo.field(drives, el.freq, grid, solver, guess) if k == 2 else geo.field(el, grid, solver, guess)
            t_solve = time.time() - t0
            flow, gas_u = None, (gs["u"] if flow_mode == 1 else 0.0)
            if flow_mode == 2:
                flow = flow_cache or gas_flow(fld, gs["slm"], gs["p"], gs["T"], r_in, start)
                gas_u = flow
            t1 = time.time()
            res, mzi = run_funnel(fld, z0, z1, mz, K0, n, gs["p"], gs["T"], zc, io["r0"] * MM, start,
                                  int(io["steps"]), io["tmax"] * 1e-3 or None, el.gradient, gas_u, rec,
                                  progress=progress, current=current, sc_iterations=sc_it, heating=heat_model)
            return {"key": key, "grid": grid, "solver": solver, "field": fld, "res": res, "mzi": mzi, "mz": mz,
                    "K0": K0, "mode": mode, "t_solve": t_solve, "cached": bool(cache), "start": start, "z": (z0, z1),
                    "gas": gs, "io": io, "el": el, "species": species, "ref": ref, "flow": flow, "flow_key": flow_key, "r_in": r_in,
                    "current": current, "t_fly": time.time() - t1}

        # while the geometry is unchanged the grid and solver are reused: new voltages only re-solve
        self.start(job, self._show, "Flying ions..." if cache else "Solving fields...")

    def _show(self, out):
        self._geo_cache = (out["key"], out["grid"], out["solver"], out["field"].phi)
        res, mzi, el = out["res"], out["mzi"], out["el"]
        g = out["grid"]
        what = "reused solver" if out["cached"] else f"{g.nodes / 1e6:.2f} M nodes"
        ins = "" if g.insulator is None else f", insulators on {g.insulator.mean():.0%} of the grid"
        self.log(f"Fields: {len(g.names)} electrodes{ins}, {what}, ready in {out['t_solve']:.1f} s")
        if out["flow"] is not None:
            f = out["flow"]
            self._flow_cache = (out["flow_key"], f)
            self.log(f"Gas flow: {out['gas']['slm']:g} slm at {out['gas']['p']:g} Torr enters a {2 * out['r_in'] / MM:.1f} mm "
                     f"disc at {f.inlet_speed:.1f} m/s mean; {f.exit_fraction:.1%} leaves through the exit end, the rest "
                     "is pumped out between the electrodes")
            if f.warning(out["gas"]["T"]):
                self.log("  Note: " + f.warning(out["gas"]["T"]))
            self._draw_flow(out)
        if res.sc_history:
            h = res.sc_history
            self.log(f"Space charge at {out['current'] * 1e9:g} nA: peak potential "
                     + " → ".join(f"{p:.3g}" for _, p, _ in h) + f" V over {len(h)} solves, "
                     f"{h[-1][2] / 1.602e-19:.3g} charges in the device"
                     + ("" if res.sc_converged else "  (NOT settled: raise the iterations, or the current is more "
                                                    "than this funnel carries and ions pile up)"))
        self.log(f"Flown in {out['t_fly']:.1f} s")
        self._draw_potential(out)
        if out["mode"] == 0:
            st = res.status
            species = out["species"]
            for d in species_stats(res, species):
                tr = d["mask"] & (st == 1)
                self.log(f"{d['name']} (m/z {d['mz']:g}, {d['z']}+): transmitted {d['through']:.0%}, lost {d['lost']:.0%}, "
                         f"still flying {d['flying']:.0%}"
                         + (f"; transit {d['transit'] * 1e3:.2f} ms, exit r_rms "
                            f"{np.sqrt(np.mean(np.sum((res.final_pos[tr, :2] - out['start']) ** 2, 1))) / MM:.3f} mm" if tr.any() else ""))
                if res.heat is not None:
                    self.log(f"  Heating {d['name']}: " + res.heat.summary(d["mask"] & (st >= 0)))
            self._draw_traj(out)
            self._draw_exit(out)
            names = ", ".join(s.name for s in species)
            draw_heating(self.plots.figs["Heating"], res.heat, st,
                         title=f"{names} in {out['gas']['p']:g} Torr: ion heating along the funnel")
            self.plots.draw("Heating")
            self.last_table = (["species", "mz", "z", "status(1=out,2=lost,0=flying)", "x_mm", "y_mm", "z_mm", "t_ms"],
                               [[species[k].name, species[k].mz, species[k].z, int(s), *(p / MM), t * 1e3]
                                for k, s, p, t in zip(res.species, st, res.final_pos, res.t_end)])
        else:
            mz = out["mz"]
            T = np.array([np.mean(res.status[mzi == m] == 1) for m in mz])
            L = np.array([np.mean(res.status[mzi == m] == 2) for m in mz])
            F = 1 - T - L
            fig = self.plots.fig("Transmission")
            ax = fig.add_subplot(111)
            ax.semilogx(mz, 100 * T, "o-", label="transmitted")
            ax.semilogx(mz, 100 * L, "s--", label="lost on electrodes")
            if F.max() > 0:
                ax.semilogx(mz, 100 * F, "^:", label="trapped / still flying")
            ax.set_xlabel("m/z"); ax.set_ylabel("%"); ax.set_ylim(-3, 103); ax.legend(fontsize=8)
            ax.set_title(f"{el.rf:.0f} V RF at {el.freq / 1e3:.0f} kHz, {el.gradient:g} V/cm, {out['gas']['p']:g} Torr")
            self.plots.draw("Transmission")
            h = res.heat
            for m, t, l in zip(mz, T, L):
                extra = ""
                if h is not None:
                    sel = mzi == m
                    extra = f", internal T peak {np.median(h.tint_max[sel]):.0f} K"
                    if h.model.ea_ev > 0:
                        extra += f", intact {np.mean(h.survival[sel]):.0%}"
                self.log(f"  m/z {m:8.1f}: transmitted {t:5.0%}, lost {l:5.0%}{extra}")
            self.last_table = (["mz", "K0", "transmitted", "lost", "trapped"], [list(r) for r in zip(mz, out["K0"], T, L, F)])

    def _draw_potential(self, out):
        fld, el = out["field"], out["el"]
        sec = fld.sections(out["start"])
        sc = out["res"].space_charge
        fig = self.plots.fig("Potential", show=False)
        axes = fig.subplots(3 if sc is not None else 2, 1, sharex=True)
        s = sec["x"]
        Z, A = s["z"] / MM, s["coords"] / MM
        solid = s["mask"] if s.get("insulator") is None else s["mask"] | s["insulator"]
        im = axes[0].imshow(np.ma.masked_where(solid, s["dc"]), origin="lower", aspect="auto", cmap="viridis",
                            extent=[Z[0], Z[-1], A[0], A[-1]])
        fig.colorbar(im, ax=axes[0], label="V")
        e2 = sum(sum(gg ** 2 for gg in np.gradient(p, s["h"])) for p in s["rf"]) * 0.5
        ref = out["ref"]
        K = mobility(ref.K0, out["gas"]["p"], out["gas"]["T"])
        U = pseudopotential(e2, ref.mz, int(ref.z), K, el.freq)
        im = axes[1].imshow(np.ma.masked_where(solid, np.clip(U, 1e-3, 10)), origin="lower", aspect="auto", cmap="magma",
                            extent=[Z[0], Z[-1], A[0], A[-1]], norm=LogNorm(1e-3, 10))
        fig.colorbar(im, ax=axes[1], label="V")
        if sc is not None:
            phi = _section_of(sc.phi, fld, out["start"])
            im = axes[2].imshow(np.ma.masked_where(solid, phi), origin="lower", aspect="auto", cmap="inferno",
                                extent=[Z[0], Z[-1], A[0], A[-1]])
            fig.colorbar(im, ax=axes[2], label="V")
            axes[2].set_title(f"Space-charge potential of the {out['current'] * 1e9:g} nA beam", fontsize=9)
        for ax in axes:
            _outline(ax, s)
            ax.set_ylabel("x (mm)")
        axes[0].set_title("DC potential (x-z section)", fontsize=9)
        axes[1].set_title(f"RF effective potential for {ref.name} (m/z {ref.mz:g}, {ref.z}+; log scale)", fontsize=9)
        axes[-1].set_xlabel("z (mm)")
        self.plots.draw("Potential")

    def _draw_traj(self, out):
        res, fld = out["res"], out["field"]
        sec = fld.sections(out["start"])
        fig = self.plots.fig("Trajectories")
        axes = fig.subplots(2, 1, sharex=True)
        for ax, key, c in ((axes[0], "x", 0), (axes[1], "y", 1)):
            s = sec[key]
            _outline(ax, s)
            if res.traj is not None:
                for i in range(res.traj.shape[1]):
                    col, ls = path_style(res.status[i], res.species[i], len(out["species"]))
                    ax.plot(res.traj[:, i, 2] / MM, res.traj[:, i, c] / MM, lw=0.6, color=col, ls=ls)
            ax.set_ylabel(f"{key} (mm)")
            ax.axvline(out["z"][1] / MM, color="k", ls=":", lw=1)
        axes[1].set_xlabel("z (mm)")
        species_legend(axes[0], out["species"], loc="upper right")
        axes[0].set_title(fate_caption(len(out["species"])) + "; dotted black: detector", fontsize=9)
        self.plots.draw("Trajectories")

    def _draw_exit(self, out):
        res = out["res"]
        tr = res.status == 1
        fig = self.plots.fig("Exit beam", show=False)
        ax1, ax2 = fig.subplots(1, 2)
        species = out["species"]
        bins = np.linspace(np.nanmin(res.t_end[tr]), np.nanmax(res.t_end[tr]), 41) * 1e3 if tr.any() else 40
        for k, s in enumerate(species):
            m = tr & (res.species == k)
            p = (res.final_pos[m, :2] - out["start"]) / MM
            ax1.plot(p[:, 0], p[:, 1], ".", ms=3, color=species_color(k), label=s.name)
            if m.any():
                ax2.hist(res.t_end[m] * 1e3, bins=bins, alpha=0.6, color=species_color(k), label=s.name)
        ax1.set_aspect("equal", adjustable="datalim")
        ax1.set_xlabel("x (mm)"); ax1.set_ylabel("y (mm)"); ax1.set_title("Transmitted ions at the detector", fontsize=9)
        ax2.set_xlabel("transit time (ms)"); ax2.set_ylabel("ions"); ax2.set_title("Transit time", fontsize=9)
        if len(species) > 1:
            ax2.legend(fontsize=7)
        self.plots.draw("Exit beam")


    def _draw_flow(self, out):
        sec = out["flow"].section(out["start"])
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
        ax.set_xlabel("z (mm)"); ax.set_ylabel("x (mm)")
        ax.set_title(f"Gas flow, {out['gas']['slm']:g} slm at {out['gas']['p']:g} Torr (viscous, pumped through every open side); "
                     f"{out['flow'].exit_fraction:.1%} through the exit", fontsize=9)
        self.plots.draw("Gas flow")


def _outline(ax, s):
    """Electrodes grey, insulators in board colour, on a section dict."""
    ax.contourf(s["z"] / MM, s["coords"] / MM, s["mask"].astype(float), levels=[0.5, 1.5], colors="0.55")
    if s.get("insulator") is not None and s["insulator"].any():
        ax.contourf(s["z"] / MM, s["coords"] / MM, (s["insulator"] & ~s["mask"]).astype(float), levels=[0.5, 1.5],
                    colors=[INSULATOR])


def _section_of(arr, fld, center):
    """The x-z section of a node array of the field's grid (same plane as fld.sections)."""
    g = fld.grid
    if hasattr(g, "r"):
        return np.concatenate([arr[:0:-1], arr], axis=0)
    jy = int(np.clip(round((center[1] - g.y[0]) / g.h), 0, g.shape[1] - 1))
    return arr[:, jy, :]


def _mesh_section(tris, cut, keep, at):
    """Segments where triangles cross the plane coordinate[cut] = at, as (keep-axis, z) pairs."""
    d = tris[:, :, cut] - at
    pts, crosses = [], []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        x = d[:, a] * d[:, b] < 0
        f = (d[:, a] / np.where(x, d[:, a] - d[:, b], 1))[:, None]
        pts.append(tris[:, a] + f * (tris[:, b] - tris[:, a])); crosses.append(x)
    crosses = np.array(crosses).T
    ok = crosses.sum(1) == 2
    first_two = np.argsort(~crosses[ok], axis=1, kind="stable")[:, :2]
    seg = np.take_along_axis(np.stack(pts, axis=1)[ok], first_two[:, :, None], axis=1)
    return seg[:, :, [keep, 2]]


def _rect(x, y, w, h, color, alpha=1.0):
    from matplotlib.patches import Rectangle
    return Rectangle((x / MM, y / MM), w / MM, h / MM, color=color, alpha=alpha, lw=0)
