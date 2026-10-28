"""The Ion Sim workspace: one window with a model tree, a view and a settings panel.

Each model is one of the device simulations (quadrupole, funnel, OA-TOF, ...). The model
keeps its own controller (the device's SimTab), whose setting groups, plots, run controls
and results text are laid out here instead of in a tab:

    tree (left)        Global definitions > Drivers; then per model: Geometry,
                       Electrodes and drive, Gas, Ions, Study, Results
    view (centre)      the selected model's plots (one per Results node), or the Drivers editor
    settings (right)   the setting groups of the selected tree node
    run bar (bottom)   Run / Stop / progress and the results text of the selected model
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
    """One device simulation in the workspace: its controller and where its pieces went."""

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
        self.pages = {}             # category -> (scroll area, page widget)
        self.plots = self.ctl.plots
        self.plots.tabBar().hide()
        self.plots.setDocumentMode(True)
        # run controls and results text move to the run bar
        self.bar = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(self.bar); h.setContentsMargins(0, 0, 0, 0)
        c = self.ctl
        c.run_btn.setMinimumWidth(110); c.stop_btn.setMinimumWidth(80)
        h.addWidget(c.run_btn); h.addWidget(c.stop_btn); h.addWidget(c.bar, 1); h.addWidget(c.status, 2)
        self.log = c.results

    def page(self, cat):
        if cat not in self.pages:
            page = QtWidgets.QWidget()
            v = QtWidgets.QVBoxLayout(page)
            for gc, _, w in self.groups:
                if gc == cat:
                    v.addWidget(w)
            v.addStretch(1)
            sa = QtWidgets.QScrollArea(); sa.setWidgetResizable(True); sa.setWidget(page)
            self.pages[cat] = (sa, page)
        return self.pages[cat]


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
        """models: [(template, name or None)] to open; None reopens the last session's models.
        persist: remember the open models for next time (off for tests and screenshots)."""
        super().__init__()
        self.persist = persist
        self.setWindowTitle(APP_NAME)
        self.resize(1500, 920)
        self.models = []
        self.settings = QtCore.QSettings("IonSim", "IonSim")
        self._syncing = False

        # ---- tree ----
        self.tree = QtWidgets.QTreeWidget(); self.tree.setHeaderHidden(True)
        self.tree.setMinimumWidth(230)
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

        # ---- centre and right ----
        self.view = QtWidgets.QStackedWidget()
        self.start = self._start_page()
        self.view.addWidget(self.start)
        self.drivers = DriversTab()
        self.view.addWidget(self.drivers)
        self.props = QtWidgets.QStackedWidget(); self.props.setMinimumWidth(400)
        self.blank = QtWidgets.QLabel("Pick a node in the model tree to see its settings.")
        self.blank.setAlignment(QtCore.Qt.AlignCenter); self.blank.setWordWrap(True)
        self.props.addWidget(self.blank)
        self.props_title = QtWidgets.QLabel(); self.props_title.setStyleSheet("font-weight: bold; padding: 4px;")
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right); rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(self.props_title); rv.addWidget(self.props, 1)
        self.right = right

        # ---- bottom: run bar + log ----
        self.bars = QtWidgets.QStackedWidget(); self.bars.addWidget(QtWidgets.QWidget())
        self.logs = QtWidgets.QStackedWidget()
        empty = QtWidgets.QPlainTextEdit(); empty.setReadOnly(True)
        empty.setPlaceholderText("Results of the selected model appear here.")
        self.logs.addWidget(empty)
        bottom = self.bottom = QtWidgets.QWidget()
        bv = QtWidgets.QVBoxLayout(bottom); bv.setContentsMargins(4, 2, 4, 2)
        self.bars.setMaximumHeight(40)
        bv.addWidget(self.bars); bv.addWidget(self.logs, 1)

        for wdg in (self.view, self.bars, self.logs):    # let the splitter shrink the centre below its pages' hints
            wdg.setMinimumWidth(360)
        mid = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        mid.addWidget(self.view); mid.addWidget(bottom); mid.setSizes([700, 170])
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(self.tree); split.addWidget(mid); split.addWidget(right)
        split.setSizes([240, 820, 440])
        split.setStretchFactor(1, 1)
        self.setCentralWidget(split)
        self._menus()
        self.statusBar().showMessage("Add a model, set it up from the tree, press Run")

        names = models if models is not None else (self._saved_models() if persist else [])
        for tpl, name in names:
            if tpl in TEMPLATES:
                self.add_model(tpl, name, select=False)
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
        self.view.addWidget(m.plots)
        self.bars.addWidget(m.bar)
        self.logs.addWidget(m.log)
        m.plots.currentChanged.connect(lambda k, m=m: self._plot_shown(m, k))
        self.tree.expandItem(root); self.tree.expandItem(ri)
        for ci in m.items.values():
            self.tree.expandItem(ci)
        self._save_models()
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
        for w in (m.plots, m.bar, m.log):
            w.setParent(None)
        for sa, _ in m.pages.values():
            self.props.removeWidget(sa); sa.setParent(None)
        self.models.remove(m)
        self._save_models()
        self.tree.setCurrentItem(self.models[-1].root if self.models else self.add_item)

    def _rename(self, m):
        if m is None:
            return
        new, ok = QtWidgets.QInputDialog.getText(self, "Rename model", "Name", text=m.name)
        new = new.strip()
        if ok and new and not any(x.name == new for x in self.models if x is not m):
            m.name = new; m.root.setText(0, new)
            self._save_models()

    def _saved_models(self):
        raw = self.settings.value("workspace/models", "")
        out = []
        for line in str(raw).split("\n"):
            if "\t" in line:
                tpl, name = line.split("\t", 1)
                out.append((tpl, name))
        return out

    def _save_models(self):
        if self.persist:
            self.settings.setValue("workspace/models", "\n".join(f"{m.template}\t{m.name}" for m in self.models))

    # ---------- selection ----------
    def _current_model(self):
        it = self.tree.currentItem()
        return it.data(0, ROLE_MODEL) if it is not None else None

    def _select(self, item, _prev=None):
        if item is None or self._syncing:
            return
        kind = item.data(0, ROLE_KIND)
        m = item.data(0, ROLE_MODEL)
        if kind in ("add", "global", "drivers"):            # full-width pages: no settings panel or run bar
            self.view.setCurrentWidget(self.start if kind == "add" else self.drivers)
            self.right.hide(); self.bottom.hide()
            return
        self.right.show(); self.bottom.show()
        self.view.setCurrentWidget(m.plots)
        self.bars.setCurrentWidget(m.bar); self.logs.setCurrentWidget(m.log)
        for other in self.models:                   # keep the tree short: only this model open
            if other is not m:
                other.root.setExpanded(False)
        m.root.setExpanded(True)
        if kind == "plot":
            self._syncing = True
            m.plots.setCurrentIndex(item.data(0, ROLE_KEY))
            self._syncing = False
            self._show_props(m, self._last_cat(m))
        elif kind in ("cat", "group"):
            cat = item.data(0, ROLE_KEY)
            m.last_cat = cat
            self._show_props(m, cat)
            names = list(m.plots.figs)
            pick = next((n for n in VIEW_FOR.get(cat, ()) if n in names), None)
            if pick is not None:
                self._syncing = True
                m.plots.setCurrentIndex(names.index(pick))
                self._syncing = False
            if kind == "group":
                wid = item.data(0, QtCore.Qt.UserRole + 3)
                for _, _, w in m.groups:
                    if id(w) == wid:
                        QtCore.QTimer.singleShot(0, lambda w=w, sa=m.pages[cat][0]: sa.verticalScrollBar().setValue(w.y()))
        else:                               # model root or Results
            self._show_props(m, self._last_cat(m))

    def _last_cat(self, m):
        return getattr(m, "last_cat", None) or next(iter(m.items), None)

    def _show_props(self, m, cat, text=None):
        if m is None or cat is None:
            self.blank.setText(text or "Pick a node in the model tree to see its settings.")
            self.props.setCurrentWidget(self.blank); self.props_title.setText("")
            return
        sa, page = m.page(cat)
        if self.props.indexOf(sa) < 0:
            self.props.addWidget(sa)
        need = page.minimumSizeHint().width() + sa.verticalScrollBar().sizeHint().width() + 4
        sp = self.centralWidget()                     # fit the settings panel to the page: never clip it
        sizes = sp.sizes()
        grow = max(440, need) - sizes[2]
        if grow and sizes[1] - grow >= 360:
            sp.setSizes([sizes[0], sizes[1] - grow, sizes[2] + grow])
        self.props.setCurrentWidget(sa)
        self.props_title.setText(f"{m.name}  ›  {cat}")

    def _plot_shown(self, m, k):
        """A run brought a plot to the front: select its Results node."""
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
                               "the Drivers under Global definitions are shared by all of them.")
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
            "<p>Add a model for each device, set it up from the model tree on the left, and press Run (F5). "
            "Drivers under Global definitions are shared by every model.</p>"
            "<p>The previous one-tab-per-device window is under View &gt; Classic tabs window.</p>")

    def closeEvent(self, e):
        self._save_models()
        for m in self.models:
            m.ctl.stop()
            if m.ctl._thread is not None:
                m.ctl._thread.quit(); m.ctl._thread.wait(3000)
        e.accept()
