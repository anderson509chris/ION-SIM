"""The Ion Sim workspace: one window with a model tree beside the device it is set to.

Each model is one of the device simulations (quadrupole, funnel, OA-TOF, ...) and keeps the
look of its tab: the settings in one column with Run, Stop and progress under it, the plots
beside it and the results text under the plots. The tree on the left lists every model:

    Global definitions > Drivers     the electronics, shared by every model
    each model                       Geometry, Electrodes and drive, Gas, Ions, Study (picking one
                                     scrolls the settings to it), Results (picking one shows that plot)

Models are stages: an OA-TOF can take its beam and ion mixture from the ions that left a
quadrupole, ion funnel or Device builder model (Ions > Incoming beam > Ions from).
"""
import re
from PySide6 import QtCore, QtGui, QtWidgets

from .quad_tab import QuadTab
from .twave_tab import TwaveTab
from .faims_tab import FaimsTab
from .inlet_tab import InletTab
from .funnel_tab import FunnelTab
from .device_tab import DeviceTab
from .tof_tab import TofTab
from .drivers_tab import DriversTab

APP_NAME = "Ion Sim"

# template name -> (controller class, one-line description)
TEMPLATES = {
    "Quadrupole mass filter": (QuadTab, "Round rods with RF + DC or a driver with excitation tones; Mathieu "
                                        "stability, mass scans, tone ejection."),
    "TWAVE traveling wave": (TwaveTab, "Stepped electrode boards in gas: mobility separation, wave-speed sweeps."),
    "FAIMS": (FaimsTab, "Asymmetric waveform across a planar gap at 1 atm: CV spectra and dispersion."),
    "ESI inlet capillary": (InletTab, "Spray region, choked capillary flow and free jet: ion speed and energy."),
    "Ion funnel": (FunnelTab, "Ring, planar or imported CAD funnels in 1-10 Torr: transmission, gas flow, "
                              "space charge, heating."),
    "Device builder (DXF / CAD)": (DeviceTab, "Your own drawing or 3D model: click electrodes, wire drivers, "
                                              "solve and fly."),
    "OA-TOF": (TofTab, "Pusher, drift, reflectron and detector: resolution, mass spectra, tuning."),
}

CATEGORIES = ["Geometry", "Electrodes and drive", "Gas", "Ions", "Study"]
_RULES = [
    ("Study", r"run mode|scan|sweep|dispersion plot"),
    ("Gas", r"\bgas\b|space charge|heating|flow"),
    ("Ions", r"\bions?\b|beam|species|start and|path and time"),
    ("Geometry", r"geometry|rods|boards|\bcell\b|funnel type|ring funnel|planar funnel|imported|\bgrid\b|"
                 r"capillary|spray|vacuum stage"),
]
# the plot to bring forward when a settings category is picked (the first one the model has)
VIEW_FOR = {"Geometry": ("Geometry", "Layout", "Cross-section", "TW field", "Gas flow"),
            "Electrodes and drive": ("Geometry", "Layout", "Potential", "Potentials", "Waveform", "TW field")}
SOURCES = (QuadTab, FunnelTab, DeviceTab)
ROLE_KIND, ROLE_MODEL, ROLE_KEY = QtCore.Qt.UserRole, QtCore.Qt.UserRole + 1, QtCore.Qt.UserRole + 2


def category_of(title):
    t = short_title(title).lower()
    for cat, pat in _RULES:
        if re.search(pat, t):
            return cat
    return "Electrodes and drive"


def short_title(title):
    """'2. Electrodes  (click them ...)' -> 'Electrodes'."""
    t = re.sub(r"^\s*\d+\.\s*", "", title)
    t = re.sub(r"\s*\(.*$", "", t).strip()
    return t or title


class Model:
    """One device simulation in the workspace: its controller (the device's tab, shown whole)
    and its setting groups, sorted into the tree's categories."""

    def __init__(self, name, template):
        self.name, self.template = name, template
        self.ctl = TEMPLATES[template][0]()
        self.groups = []            # (category, title, widget)
        lay = self.ctl.form_layout
        for i in range(lay.count()):
            w = lay.itemAt(i).widget()
            if w is None:
                continue
            title = w.title() if isinstance(w, QtWidgets.QGroupBox) else ""
            self.groups.append((category_of(title) if title else "Electrodes and drive", title or "Settings", w))
        self.plots = self.ctl.plots
        self._fitted = False

    def fit(self):
        """Give the settings column the width its groups need, so nothing is clipped (once, when first shown)."""
        split = self.ctl.layout().itemAt(0).widget()
        if self._fitted or not split.isVisible():
            return
        self._fitted = True
        sa = self.ctl.scroll
        need = sa.widget().minimumSizeHint().width() + sa.verticalScrollBar().sizeHint().width() + 2 * sa.frameWidth() + 6
        sizes = split.sizes()
        col = max(380, min(need, 640))
        split.setSizes([col, max(sum(sizes) - col, 300)])

    @property
    def is_source(self):
        """Its runs leave exit ions that a later stage can take."""
        return isinstance(self.ctl, SOURCES)

    @property
    def is_stage(self):
        """It can take its ions from another model."""
        return isinstance(self.ctl, TofTab)

    def scroll_to(self, w):
        sa = self.ctl.scroll
        QtCore.QTimer.singleShot(0, lambda: sa.verticalScrollBar().setValue(max(w.y() - 6, 0)))


class _VisWatch(QtCore.QObject):
    """Shows or hides a tree item when its setting group is shown or hidden by its model."""

    def __init__(self, item, widget):
        super().__init__(widget)
        self.item = item
        widget.installEventFilter(self)
        item.setHidden(widget.isHidden())

    def eventFilter(self, obj, ev):
        if ev.type() in (QtCore.QEvent.ShowToParent, QtCore.QEvent.HideToParent):
            self.item.setHidden(ev.type() == QtCore.QEvent.HideToParent)
        return False


class Workspace(QtWidgets.QMainWindow):
    def __init__(self, models=None, persist=True):
        """models: [(template, name or None)] or [(template, name, ions from)] to open; None reopens the
        last session's models. persist: remember the open models for next time (off for tests and screenshots)."""
        super().__init__()
        self.persist = persist
        self.setWindowTitle(APP_NAME)
        self.resize(1500, 920)
        self.models = []
        self.settings = QtCore.QSettings("IonSim", "IonSim")
        self._syncing = False
        self._chain = self._chain_now = None        # Run with the stages before it: models still to run, the one running

        # ---- tree ----
        self.tree = QtWidgets.QTreeWidget(); self.tree.setHeaderHidden(True)
        self.tree.setMinimumWidth(200)
        self.tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._tree_menu)
        self.tree.currentItemChanged.connect(self._select)
        self.tree.itemDoubleClicked.connect(self._double)
        self.glob = QtWidgets.QTreeWidgetItem(self.tree, ["Global definitions"])
        self.glob.setData(0, ROLE_KIND, "global")
        self.drv_item = QtWidgets.QTreeWidgetItem(self.glob, ["Drivers"])
        self.drv_item.setData(0, ROLE_KIND, "drivers")
        self.drv_item.setToolTip(0, "The electronics: RF, TWAVE, excitation tones and pulses, wired to electrodes in the models")
        self.add_item = QtWidgets.QTreeWidgetItem(self.tree, ["+ Add model…"])
        self.add_item.setData(0, ROLE_KIND, "add")
        f = self.add_item.font(0); f.setItalic(True); self.add_item.setFont(0, f)

        # ---- the selected model, laid out as its tab ----
        self.view = QtWidgets.QStackedWidget()
        self.start = self._start_page()
        self.view.addWidget(self.start)
        self.drivers = DriversTab()
        self.view.addWidget(self.drivers)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(self.tree); split.addWidget(self.view)
        split.setSizes([230, 1270])
        split.setStretchFactor(1, 1)
        self.setCentralWidget(split)
        self._menus()
        self.statusBar().showMessage("Add a model, set it up and press Run")

        rows = models if models is not None else (self._saved_models() if persist else [])
        rows = [tuple(r) + (None,) * (3 - len(r)) for r in rows]
        for tpl, name, _ in rows:
            if tpl in TEMPLATES:
                self.add_model(tpl, name, select=False)
        for m, (_, _, src) in zip(self.models, [r for r in rows if r[0] in TEMPLATES]):
            if src and m.is_stage:
                self.link(m, src)
        self.tree.expandItem(self.glob)
        if self.models:
            self.tree.setCurrentItem(self.models[0].root)
        else:
            self.tree.setCurrentItem(self.add_item)

    # ---------- menus ----------
    def _menus(self):
        m = self.menuBar().addMenu("&File")
        add = m.addMenu("Add model")
        for tpl in TEMPLATES:
            add.addAction(tpl).triggered.connect(lambda _=False, t=tpl: self.add_model(t))
        act = m.addAction("Rename model…"); act.triggered.connect(lambda: self._rename(self._current_model()))
        act = m.addAction("Remove model"); act.triggered.connect(lambda: self._remove(self._current_model()))
        m.addSeparator()
        act = m.addAction("Save current plot…"); act.setShortcut("Ctrl+S"); act.triggered.connect(self.save_plot)
        act = m.addAction("Export results (CSV)…"); act.setShortcut("Ctrl+E"); act.triggered.connect(self.export_csv)
        m.addSeparator()
        act = m.addAction("Quit"); act.setShortcut("Ctrl+Q"); act.triggered.connect(self.close)
        r = self.menuBar().addMenu("&Run")
        act = r.addAction("Run selected model"); act.setShortcut("F5"); act.triggered.connect(self.run_current)
        act = r.addAction("Run with the stages before it"); act.setShortcut("F6")
        act.triggered.connect(lambda: self.run_chain(self._current_model()))
        act = r.addAction("Stop"); act.setShortcut("Shift+F5"); act.triggered.connect(self.stop_current)
        v = self.menuBar().addMenu("&View")
        v.addAction("Classic tabs window…").triggered.connect(self._classic)
        h = self.menuBar().addMenu("&Help")
        h.addAction("About").triggered.connect(self._about)

    # ---------- models ----------
    def add_model(self, template, name=None, select=True):
        base = name or template
        name, k = base, 2
        while any(m.name == name for m in self.models):
            name = f"{base} {k}"; k += 1
        m = Model(name, template)
        self.models.append(m)
        root = QtWidgets.QTreeWidgetItem([name])
        self.tree.insertTopLevelItem(self.tree.indexOfTopLevelItem(self.add_item), root)
        root.setData(0, ROLE_KIND, "model"); root.setData(0, ROLE_MODEL, m)
        root.setToolTip(0, TEMPLATES[template][1])
        f = root.font(0); f.setBold(True); root.setFont(0, f)
        m.root = root
        m.items = {}
        for cat in CATEGORIES:
            groups = [(t, w) for c, t, w in m.groups if c == cat]
            if not groups:
                continue
            ci = QtWidgets.QTreeWidgetItem(root, [cat])
            ci.setData(0, ROLE_KIND, "cat"); ci.setData(0, ROLE_MODEL, m); ci.setData(0, ROLE_KEY, cat)
            for t, w in groups:
                gi = QtWidgets.QTreeWidgetItem(ci, [short_title(t)])
                gi.setToolTip(0, t)
                gi.setData(0, ROLE_KIND, "group"); gi.setData(0, ROLE_MODEL, m); gi.setData(0, ROLE_KEY, cat)
                gi.setData(0, QtCore.Qt.UserRole + 3, id(w))
                _VisWatch(gi, w)
            m.items[cat] = ci
        ri = QtWidgets.QTreeWidgetItem(root, ["Results"])
        ri.setData(0, ROLE_KIND, "results"); ri.setData(0, ROLE_MODEL, m)
        m.result_items = []
        for k, pname in enumerate(m.plots.figs):
            pi = QtWidgets.QTreeWidgetItem(ri, [pname])
            pi.setData(0, ROLE_KIND, "plot"); pi.setData(0, ROLE_MODEL, m); pi.setData(0, ROLE_KEY, k)
            m.result_items.append(pi)
        m.results_item = ri
        self.view.addWidget(m.ctl)
        m.plots.currentChanged.connect(lambda k, m=m: self._plot_shown(m, k))
        m.ctl.ran.connect(lambda ok, m=m: self._ran(m, ok))
        if m.is_stage:
            m.ctl.beam_src.currentIndexChanged.connect(lambda _=0, m=m: self._stage_changed(m))
        self.tree.expandItem(root)
        for ci in m.items.values():
            self.tree.expandItem(ci)
        self._relink()
        if select:
            self.tree.setCurrentItem(root)
        return m

    def _remove(self, m):
        if m is None:
            return
        if QtWidgets.QMessageBox.question(self, APP_NAME, f"Remove '{m.name}' from the workspace?") \
                != QtWidgets.QMessageBox.Yes:
            return
        m.ctl.stop()
        if m.ctl._thread is not None:
            m.ctl._thread.quit(); m.ctl._thread.wait(3000)
        self.tree.takeTopLevelItem(self.tree.indexOfTopLevelItem(m.root))
        self.view.removeWidget(m.ctl); m.ctl.setParent(None)
        self.models.remove(m)
        self._relink()
        self.tree.setCurrentItem(self.models[-1].root if self.models else self.add_item)

    def _rename(self, m):
        if m is None:
            return
        new, ok = QtWidgets.QInputDialog.getText(self, "Rename model", "Name", text=m.name)
        new = new.strip()
        if ok and new and not any(x.name == new for x in self.models if x is not m):
            old = m.name
            m.name = new
            self._relink(renamed=(old, new))

    def _saved_models(self):
        raw = self.settings.value("workspace/models", "")
        out = []
        for line in str(raw).split("\n"):
            if "\t" in line:
                parts = line.split("\t")
                out.append((parts[0], parts[1], parts[2] if len(parts) > 2 and parts[2] else None))
        return out

    def _save_models(self):
        if self.persist:
            self.settings.setValue("workspace/models", "\n".join(
                f"{m.template}\t{m.name}\t{(m.ctl.source_name() or '') if m.is_stage else ''}" for m in self.models))

    # ---------- stages ----------
    def link(self, stage, source):
        """Make `stage` (an OA-TOF model) take its ions from the model named `source`; None sets them by hand."""
        cb = stage.ctl.beam_src
        cb.setCurrentIndex(max(cb.findText(f"{source} exit"), 0) if source else 0)

    def source_of(self, m):
        name = m.ctl.source_name() if m is not None and m.is_stage else None
        return next((x for x in self.models if x.name == name), None)

    def _relink(self, renamed=None):
        """Offer every source model to every stage, keeping their links across renames and removals."""
        for m in self.models:
            if not m.is_stage:
                continue
            keep = m.ctl.source_name()
            if renamed and keep == renamed[0]:
                keep = renamed[1]
            m.ctl.beam_src.blockSignals(True)
            m.ctl.set_sources({x.name: x.ctl for x in self.models if x.is_source})
            m.ctl.beam_src.blockSignals(False)
            self.link(m, keep)
        for m in self.models:
            self._label(m)
        self._save_models()

    def _stage_changed(self, m):
        self._label(m)
        self._save_models()

    def _label(self, m):
        src = self.source_of(m)
        m.root.setText(0, f"{m.name}  ← {src.name}" if src else m.name)
        m.root.setToolTip(0, TEMPLATES[m.template][1] + (f"\nIons from the exit of '{src.name}'" if src else ""))

    def run_chain(self, m):
        """Run the models this one takes its ions from, in order, then this one."""
        if m is None:
            return
        chain, x = [m], self.source_of(m)
        while x is not None and x not in chain:
            chain.insert(0, x); x = self.source_of(x)
        if len(chain) > 1:
            self.statusBar().showMessage("Running " + " → ".join(c.name for c in chain), 8000)
        self._chain = chain
        self._next_in_chain()

    def _next_in_chain(self):
        if not self._chain:
            self._chain = self._chain_now = None
            return
        m = self._chain_now = self._chain.pop(0)
        if m.ctl._thread is not None:              # already running: the next one starts when it ends
            return
        m.ctl.run_btn.click()
        if m.ctl._thread is None:                  # the run did not start (an input error): stop the chain
            self._chain = self._chain_now = None
            self.statusBar().showMessage(f"'{m.name}' did not start (see its results box); the stages after it were not run", 8000)

    def _ran(self, m, ok):
        if self._chain is None or m is not self._chain_now:
            return
        if not ok:
            self._chain = self._chain_now = None
            self.statusBar().showMessage(f"'{m.name}' failed; the stages after it were not run", 8000)
            return
        self._next_in_chain()

    # ---------- selection ----------
    def _current_model(self):
        it = self.tree.currentItem()
        return it.data(0, ROLE_MODEL) if it is not None else None

    def _select(self, item, _prev=None):
        if item is None or self._syncing:
            return
        kind = item.data(0, ROLE_KIND)
        m = item.data(0, ROLE_MODEL)
        if kind in ("add", "global", "drivers"):
            self.view.setCurrentWidget(self.start if kind == "add" else self.drivers)
            return
        self.view.setCurrentWidget(m.ctl)
        for other in self.models:                   # keep the tree short: only this model open
            if other is not m:
                other.root.setExpanded(False)
        m.root.setExpanded(True)
        QtCore.QTimer.singleShot(0, m.fit)        # once it is laid out
        if kind == "plot":
            self._syncing = True
            m.plots.setCurrentIndex(item.data(0, ROLE_KEY))
            self._syncing = False
        elif kind in ("cat", "group"):
            cat = item.data(0, ROLE_KEY)
            if kind == "group":
                wid = item.data(0, QtCore.Qt.UserRole + 3)
                w = next(w for _, _, w in m.groups if id(w) == wid)
            else:
                w = next((w for c, _, w in m.groups if c == cat and not w.isHidden()), None)
            if w is not None:
                m.scroll_to(w)
            names = list(m.plots.figs)
            pick = next((n for n in VIEW_FOR.get(cat, ()) if n in names), None)
            if pick is not None:
                self._syncing = True
                m.plots.setCurrentIndex(names.index(pick))
                self._syncing = False

    def _plot_shown(self, m, k):
        """A plot tab was picked, or a run brought a plot to the front: select its Results node."""
        if self._syncing or not 0 <= k < len(m.result_items):
            return
        if self._current_model() is m:
            self._syncing = True
            self.tree.setCurrentItem(m.result_items[k])
            self._syncing = False

    def _double(self, item, _col):
        if item.data(0, ROLE_KIND) == "model":
            self._rename(item.data(0, ROLE_MODEL))

    def _tree_menu(self, pos):
        item = self.tree.itemAt(pos)
        menu = QtWidgets.QMenu(self)
        m = item.data(0, ROLE_MODEL) if item is not None else None
        if m is not None:
            menu.addAction(f"Run '{m.name}'").triggered.connect(m.ctl.run_btn.click)
            if self.source_of(m) is not None:
                menu.addAction("Run with the stages before it").triggered.connect(lambda: self.run_chain(m))
            if m.is_stage:
                take = menu.addMenu("Take ions from")
                for x in [None] + [x for x in self.models if x.is_source]:
                    a = take.addAction("Set by hand" if x is None else x.name)
                    a.setCheckable(True); a.setChecked(self.source_of(m) is x)
                    a.triggered.connect(lambda _=False, x=x: self.link(m, x.name if x else None))
            menu.addAction("Rename…").triggered.connect(lambda: self._rename(m))
            menu.addAction("Remove").triggered.connect(lambda: self._remove(m))
            menu.addSeparator()
        add = menu.addMenu("Add model")
        for tpl in TEMPLATES:
            add.addAction(tpl).triggered.connect(lambda _=False, t=tpl: self.add_model(t))
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    # ---------- start page ----------
    def _start_page(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(40, 30, 40, 30)
        title = QtWidgets.QLabel("Add a model")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        v.addWidget(title)
        sub = QtWidgets.QLabel("Each model is one device. Several models can sit in the workspace side by side; "
                               "the Drivers under Global definitions are shared by all of them. An OA-TOF can take "
                               "its ions from a quadrupole, ion funnel or Device builder model, so models can be "
                               "chained into stages of one instrument.")
        sub.setWordWrap(True); v.addWidget(sub); v.addSpacing(12)
        grid = QtWidgets.QGridLayout()
        for i, (tpl, (_, desc)) in enumerate(TEMPLATES.items()):
            b = QtWidgets.QPushButton(tpl); b.setMinimumHeight(44)
            b.setStyleSheet("text-align: left; padding: 6px 12px; font-weight: bold;")
            b.clicked.connect(lambda _=False, t=tpl: self.add_model(t))
            lab = QtWidgets.QLabel(desc); lab.setWordWrap(True)
            grid.addWidget(b, i, 0); grid.addWidget(lab, i, 1)
        grid.setColumnStretch(1, 1)
        v.addLayout(grid); v.addStretch(1)
        return w

    # ---------- actions ----------
    def run_current(self):
        m = self._current_model()
        if m is not None:
            m.ctl.run_btn.click()

    def stop_current(self):
        self._chain = self._chain_now = None
        m = self._current_model()
        if m is not None:
            m.ctl.stop()

    def save_plot(self):
        m = self._current_model()
        fig = m.plots.current_figure() if m else (self.drivers.plots.current_figure()
                                                  if self.view.currentWidget() is self.drivers else None)
        if fig is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save plot", "plot.png", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            fig.savefig(path, dpi=200)
            self.statusBar().showMessage(f"Saved {path}", 5000)

    def export_csv(self):
        import csv
        m = self._current_model()
        if m is None or not m.ctl.last_table:
            QtWidgets.QMessageBox.information(self, APP_NAME, "Run a model first.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export results", "results.csv", "CSV (*.csv)")
        if path:
            header, rows = m.ctl.last_table
            with open(path, "w", newline="") as f:
                w = csv.writer(f); w.writerow(header); w.writerows(rows)
            self.statusBar().showMessage(f"Saved {path}", 5000)

    def _classic(self):
        from .app import MainWindow
        self._classic_win = MainWindow()
        self._classic_win.show()

    def _about(self):
        from .app import VERSION
        QtWidgets.QMessageBox.about(self, APP_NAME,
            f"<b>{APP_NAME} {VERSION}</b><p>Ion optics and ion mobility simulator.</p>"
            "<p>Add a model for each device, pick it in the model tree on the left, set it up and press Run (F5). "
            "Drivers under Global definitions are shared by every model.</p>"
            "<p>An OA-TOF can take its ions from another model (Incoming beam &gt; Ions from); "
            "Run &gt; Run with the stages before it (F6) runs the chain in order.</p>"
            "<p>The previous one-tab-per-device window is under View &gt; Classic tabs window.</p>")

    def closeEvent(self, e):
        self._save_models()
        for m in self.models:
            m.ctl.stop()
            if m.ctl._thread is not None:
                m.ctl._thread.quit(); m.ctl._thread.wait(3000)
        e.accept()
