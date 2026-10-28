"""The Instrument tab: one CAD model of the whole system, and which simulation models each part is.

Load the STEP of the instrument (chambers, inlet, funnel, rods, ...). Its parts are drawn cut along
the ion axis, from the side (x) and from above (y), and in 3D. Click a part's outline (Ctrl or Shift
click for more), pick the model it is and press Tag: parts tagged with the same stage name form one
stage. Make models opens one linked tab per stage, in the order the stages sit along the ion path;
each tab then gets its own detailed geometry (its DXF or STEP) and runs on its own or as part of the
chain (F6).
"""
import os
import numpy as np
import matplotlib
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from PySide6 import QtCore, QtWidgets

from ..cad import load_bodies, mesh_section, FILE_FILTER, UNITS
from .common import PlotPanel
from .speeds import draw_speeds
from .electrode_view import ElectrodeView

MM = 1e-3
NONE = "—"
WALL = "Chamber or wall (no model)"
SHORT = {"Quadrupole mass filter": "Quadrupole", "TWAVE traveling wave": "TWAVE", "ESI inlet capillary": "Inlet",
         "Device builder (DXF / CAD)": "Device", "Ion funnel": "Funnel", "OA-TOF": "OA-TOF", "FAIMS": "FAIMS"}


class InstrumentTab(QtWidgets.QWidget):
    make_models = QtCore.Signal(list)       # [(template, stage name, (z0, z1) in mm)] in order along the ion path
    changed = QtCore.Signal()               # tags or the file changed (the workspace saves them)

    def set_templates(self, templates):
        """The models a part can be (the built-in ones and any loaded from your folders)."""
        self.templates = list(templates)
        cur = self.kind.currentText()
        self.kind.blockSignals(True)
        self.kind.clear(); self.kind.addItems([NONE, WALL] + self.templates)
        self.kind.setCurrentText(cur if cur in [NONE, WALL] + self.templates else NONE)
        self.kind.blockSignals(False)

    def __init__(self, templates, parent=None):
        super().__init__(parent)
        self.templates = list(templates)
        self.path, self.bodies = None, []
        self.tags = {}                      # part name -> (template or WALL, stage name)
        self.selected = []

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        QtWidgets.QHBoxLayout(self).addWidget(split)
        left = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(left)

        box = QtWidgets.QGroupBox("Instrument CAD model")
        f = QtWidgets.QFormLayout(box)
        load = QtWidgets.QPushButton("Load STEP…"); load.clicked.connect(self._browse)
        load.setToolTip("The whole instrument in one file: chambers, inlet, funnel boards, rods, TOF plates. "
                        "Each solid becomes a part you can tag.")
        self.file_label = QtWidgets.QLabel("No file loaded"); self.file_label.setWordWrap(True)
        self.units = QtWidgets.QComboBox(); self.units.addItems(list(UNITS))
        self.units.setToolTip("Units of mesh files (STL, OBJ, ...); STEP files carry their own")
        self.axis = QtWidgets.QComboBox(); self.axis.addItems(["z", "x", "y"])
        self.axis.setToolTip("The CAD axis the ions travel along")
        for w in (self.units, self.axis):
            w.currentIndexChanged.connect(self._reload)
        f.addRow(load); f.addRow(self.file_label)
        f.addRow("Mesh units", self.units); f.addRow("Ion axis", self.axis)
        v.addWidget(box)

        parts = QtWidgets.QGroupBox("Parts (click them in the drawing; Ctrl or Shift click adds)")
        pv = QtWidgets.QVBoxLayout(parts)
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Part", "z (mm)", "Model", "Stage"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.itemSelectionChanged.connect(self._table_selected)
        self.table.itemChanged.connect(self._stage_renamed)
        pv.addWidget(self.table)
        row = QtWidgets.QHBoxLayout()
        self.kind = QtWidgets.QComboBox(); self.kind.addItems([NONE, WALL] + self.templates)
        self.kind.setToolTip("What the selected parts are")
        self.stage = QtWidgets.QLineEdit(); self.stage.setPlaceholderText("stage name (blank: automatic)")
        self.stage.setToolTip("Parts with the same stage name are one model (e.g. both funnel boards)")
        tag = QtWidgets.QPushButton("Tag"); tag.clicked.connect(self._tag)
        row.addWidget(QtWidgets.QLabel("Selected are")); row.addWidget(self.kind, 1); row.addWidget(self.stage, 1); row.addWidget(tag)
        pv.addLayout(row)
        v.addWidget(parts, 1)

        out = QtWidgets.QGroupBox("Stages")
        ov = QtWidgets.QVBoxLayout(out)
        self.summary = QtWidgets.QLabel("Tag parts to make stages."); self.summary.setWordWrap(True)
        self.make_btn = QtWidgets.QPushButton("Make linked models")
        self.make_btn.setToolTip("Open a tab for each stage that has none yet (named after the stage), and link "
                                 "each one to the stage before it")
        self.make_btn.clicked.connect(self._make)
        ov.addWidget(self.summary); ov.addWidget(self.make_btn)
        v.addWidget(out)

        self.plots = PlotPanel(["Layout", "3D", "Speeds"])
        self.plots.hint("Speeds", "Run the models (F6 on the last one runs the chain) to see\n"
                                  "the gas and ion speeds along the whole instrument here.")
        self.plots.hint("Layout", "Load a STEP of the whole instrument (Load STEP… on the left).")
        self.plots.hint("3D", "Load a STEP of the whole instrument (Load STEP… on the left).")
        self.view = ElectrodeView(self.plots.figs["Layout"].canvas, self._picked, self._hover)
        self.hover = QtWidgets.QLabel(" ")
        right = QtWidgets.QWidget(); rv = QtWidgets.QVBoxLayout(right); rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(self.plots, 1); rv.addWidget(self.hover)
        split.addWidget(left); split.addWidget(right)
        split.setSizes([460, 940])

    # ---------- file ----------
    def _browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load the instrument", os.path.dirname(self.path or ""),
                                                        FILE_FILTER)
        if path:
            self.load(path)

    def load(self, path, keep_tags=False):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            bodies = load_bodies(path, self.units.currentText(), self.axis.currentText())
        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.file_label.setText(f"Could not load {os.path.basename(path)}: {e}")
            return False
        QtWidgets.QApplication.restoreOverrideCursor()
        self.path, self.bodies = path, bodies
        names = {n for n, _ in bodies}
        self.tags = {k: v for k, v in self.tags.items() if k in names} if keep_tags else {}
        self.selected = []
        self.file_label.setText(f"{os.path.basename(path)}: {len(bodies)} parts")
        self._fill(); self._draw(); self._draw_3d(); self._summarise()
        self.changed.emit()
        return True

    def _reload(self, *_):
        if self.path:
            self.load(self.path, keep_tags=True)

    # ---------- parts ----------
    def _zrange(self, tris):
        z = tris[:, :, 2]
        return z.min() / MM, z.max() / MM

    def _fill(self):
        self.table.blockSignals(True)
        order = sorted(self.bodies, key=lambda b: self._zrange(b[1])[0])
        self.table.setRowCount(len(order))
        for r, (name, tris) in enumerate(order):
            z0, z1 = self._zrange(tris)
            kind, stage = self.tags.get(name, (NONE, ""))
            for c, text in enumerate([name, f"{z0:.0f} … {z1:.0f}", SHORT.get(kind, kind), stage]):
                it = QtWidgets.QTableWidgetItem(text)
                if c != 3:
                    it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
                it.setData(QtCore.Qt.UserRole, name)
                self.table.setItem(r, c, it)
        self.table.blockSignals(False)

    def _row_of(self, name):
        for r in range(self.table.rowCount()):
            if self.table.item(r, 0).text() == name:
                return r
        return -1

    def _picked(self, name, additive):
        if additive:
            self.selected = [n for n in self.selected if n != name] if name in self.selected else self.selected + [name]
        else:
            self.selected = [name]
        self.table.blockSignals(True)
        self.table.clearSelection()
        mode = self.table.selectionMode()
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        for n in self.selected:
            r = self._row_of(n)
            if r >= 0:
                self.table.selectRow(r)
        self.table.setSelectionMode(mode)
        self.table.blockSignals(False)
        if self.selected:
            self.table.scrollToItem(self.table.item(self._row_of(self.selected[-1]), 0))
            kind, stage = self.tags.get(self.selected[-1], (NONE, ""))
            self.kind.setCurrentText(kind); self.stage.setText(stage)
        self._paint()

    def _table_selected(self):
        rows = sorted({i.row() for i in self.table.selectedItems()})
        self.selected = [self.table.item(r, 0).text() for r in rows]
        if self.selected:
            kind, stage = self.tags.get(self.selected[-1], (NONE, ""))
            self.kind.setCurrentText(kind); self.stage.setText(stage)
        self._paint()

    def _hover(self, name):
        if name is None:
            self.hover.setText(" ")
            return
        kind, stage = self.tags.get(name, (NONE, ""))
        self.hover.setText(f"{name}" + (f": {stage} ({SHORT.get(kind, kind)})" if kind not in (NONE, WALL)
                                         else (": chamber or wall" if kind == WALL else ": not tagged")))

    def _tag(self):
        if not self.selected:
            return
        kind = self.kind.currentText()
        for n in self.selected:
            if kind == NONE:
                self.tags.pop(n, None)
            elif kind == WALL:
                self.tags[n] = (WALL, "")
        if kind not in (NONE, WALL):
            stage = self.stage.text().strip() or self._auto_name(kind)
            for n in self.selected:
                self.tags[n] = (kind, stage)
            self.stage.setText(stage)
        self._fill(); self._paint(); self._draw_3d(); self._summarise()
        self.changed.emit()

    def _auto_name(self, kind):
        base = SHORT.get(kind, kind)
        used = {s for _, s in self.tags.values()}
        if base not in used:
            return base
        k = 2
        while f"{base} {k}" in used:
            k += 1
        return f"{base} {k}"

    def _stage_renamed(self, item):
        if item.column() != 3:
            return
        name = item.data(QtCore.Qt.UserRole)
        kind, old = self.tags.get(name, (NONE, ""))
        new = item.text().strip()
        if kind in (NONE, WALL):
            self._fill()
            return
        self.tags[name] = (kind, new or old)
        self._fill(); self._paint(); self._draw_3d(); self._summarise()
        self.changed.emit()

    def stages(self):
        """[(template, stage name, (z0, z1) mm, [part names])] in order along the ion path."""
        tris = dict(self.bodies)
        groups = {}
        for name, (kind, stage) in self.tags.items():
            if kind in (NONE, WALL) or name not in tris:
                continue
            g = groups.setdefault(stage, [kind, [], []])
            g[1].append(name)
            g[2].append(self._zrange(tris[name]))
        out = [(kind, stage, (min(a for a, _ in zs), max(b for _, b in zs)), names)
               for stage, (kind, names, zs) in groups.items()]
        return sorted(out, key=lambda s: (s[2][0] + s[2][1]) / 2)

    def _summarise(self):
        st = self.stages()
        self.summary.setText(" → ".join(f"{s}  ({SHORT.get(k, k)}, z {z0:.0f}…{z1:.0f} mm)" for k, s, (z0, z1), _ in st)
                             if st else "Tag parts to make stages.")
        self.make_btn.setEnabled(bool(st))

    def _make(self):
        self.make_models.emit([(k, s, z) for k, s, z, _ in self.stages()])

    # ---------- drawing ----------
    def _colors(self):
        """part name -> colour: one colour per stage, grey for walls, dark grey untagged."""
        cm = matplotlib.colormaps["tab10"]
        stage_ix = {s: i for i, (_, s, _, _) in enumerate(self.stages())}
        out = {}
        for name, _ in self.bodies:
            kind, stage = self.tags.get(name, (NONE, ""))
            out[name] = (0.72, 0.72, 0.72, 1.0) if kind == WALL else (cm(stage_ix[stage] % 10) if stage in stage_ix
                                                                       else (0.25, 0.25, 0.25, 1.0))
        return out

    def _draw(self):
        fig = self.plots.fig("Layout", show=False)
        self.view.reset()
        if not self.bodies:
            self.plots.hint("Layout", "No parts in the file.")
            return
        ax1, ax2 = fig.subplots(2, 1, sharex=True)
        self.axes = (ax1, ax2)
        for name, t in self.bodies:
            t = t / MM
            for ax, cut, keep in ((ax1, 1, 0), (ax2, 0, 1)):
                seg = mesh_section(t, cut, keep, 0.0)
                if len(seg):
                    art = LineCollection(seg[:, :, ::-1], lw=1.4)        # (z, across) so z runs left to right
                else:                                                # misses this cut: its outline from the side
                    lo, hi = t.reshape(-1, 3).min(0), t.reshape(-1, 3).max(0)
                    art = Rectangle((lo[2], lo[keep]), hi[2] - lo[2], hi[keep] - lo[keep], fill=False, ls="--", lw=1.0)
                ax.add_artist(art) if isinstance(art, Rectangle) else ax.add_collection(art)
                self.view.add(name, art)
        pts = np.concatenate([t.reshape(-1, 3) for _, t in self.bodies]) / MM
        lo, hi = pts.min(0), pts.max(0)
        pad = 0.03 * (hi[2] - lo[2])
        ax1.set_xlim(lo[2] - pad, hi[2] + pad)
        ax1.set_ylim(lo[0] - pad, hi[0] + pad); ax2.set_ylim(lo[1] - pad, hi[1] + pad)
        for ax in (ax1, ax2):
            ax.set_aspect("equal", adjustable="datalim")
            ax.axhline(0, color="0.6", lw=0.6, ls=":")
        ax1.set_ylabel("x (mm)"); ax2.set_ylabel("y (mm)"); ax2.set_xlabel("z (mm, ion axis)")
        ax1.set_title("Side: cut at y = 0 (dashed: parts off the cut, drawn by their outline)", fontsize=9)
        ax2.set_title("Top: cut at x = 0", fontsize=9)
        self._paint()

    def _paint(self):
        if not self.bodies:
            return
        col = self._colors()
        for name, arts in self.view.artists.items():
            sel = name in self.selected
            for a in arts:
                if isinstance(a, LineCollection):
                    a.set_color(col[name]); a.set_linewidth(3.0 if sel else 1.4)
                else:
                    a.set_edgecolor(col[name]); a.set_linewidth(2.6 if sel else 1.0)
        self.plots.draw("Layout")

    def _draw_3d(self):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        fig = self.plots.fig("3D", show=False)
        if not self.bodies:
            return
        ax = fig.add_subplot(111, projection="3d")
        col = self._colors()
        rng = np.random.default_rng(0)
        for name, t in self.bodies:
            t = t / MM
            kind, _ = self.tags.get(name, (NONE, ""))
            c = col[name]
            if kind == WALL:                         # walls as their outline only, so the parts inside show
                lo, hi = t.reshape(-1, 3).min(0), t.reshape(-1, 3).max(0)
                for a in ((0, 1), (0, 2), (1, 2)):   # the 12 edges of the box, as (z, x, y)
                    for u in (lo, hi):
                        for w in (lo, hi):
                            p, q = u.copy(), u.copy()
                            k = 3 - sum(a)
                            p[a[0]], p[a[1]] = u[a[0]], w[a[1]]
                            q[a[0]], q[a[1]] = u[a[0]], w[a[1]]
                            p[k], q[k] = lo[k], hi[k]
                            ax.plot([p[2], q[2]], [p[0], q[0]], [p[1], q[1]], color="0.6", lw=0.8)
                continue
            if len(t) > 6000:
                t = t[rng.choice(len(t), 6000, replace=False)]
            alpha = 0.7 if kind != NONE else 0.15
            pc = Poly3DCollection(t[:, :, [2, 0, 1]], facecolor=(*c[:3], alpha), edgecolor=(*c[:3], 0.0), linewidth=0)
            ax.add_collection3d(pc)
        pts = np.concatenate([t.reshape(-1, 3) for _, t in self.bodies]) / MM
        lo, hi = pts.min(0), pts.max(0)
        ax.set_xlim(lo[2], hi[2]); ax.set_ylim(lo[0], hi[0]); ax.set_zlim(lo[1], hi[1])
        ax.set_box_aspect((hi[2] - lo[2], hi[0] - lo[0], hi[1] - lo[1]))
        ax.set_xlabel("z (mm)"); ax.set_ylabel("x (mm)"); ax.set_zlabel("y (mm)")
        ax.set_title("Drag to turn; walls faint, tagged stages in their colour", fontsize=9)
        self.plots.draw("3D")

    def show_speeds(self, rows):
        """rows: [(stage name, speeds, offset in m)] from the workspace, in order along the ion path."""
        if not rows:
            return
        draw_speeds(self.plots.fig("Speeds", show=False), rows)
        self.plots.draw("Speeds")

    def redraw(self):
        self._draw(); self._draw_3d()

    # ---------- setups ----------
    def state(self, base_dir=None):
        if not self.path:
            return None
        path = os.path.relpath(self.path, base_dir) if base_dir else os.path.abspath(self.path)
        return {"path": path, "units": self.units.currentText(), "axis": self.axis.currentText(),
                "tags": {k: list(v) for k, v in self.tags.items()}}

    def set_state(self, d, base_dir=""):
        if not d:
            return True
        path = d["path"] if os.path.isabs(d["path"]) else os.path.normpath(os.path.join(base_dir, d["path"]))
        for w, key in ((self.units, "units"), (self.axis, "axis")):
            w.blockSignals(True); w.setCurrentText(d.get(key, w.currentText())); w.blockSignals(False)
        self.tags = {k: tuple(v) for k, v in d.get("tags", {}).items()}
        if not os.path.exists(path):
            self.file_label.setText(f"{path} not found")
            return False
        return self.load(path, keep_tags=True)
