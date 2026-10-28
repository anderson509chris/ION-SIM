"""The Ion Sim workspace: a start page, then one tab per model.

The window opens on Home, the start page, where you pick the devices to set up. Each pick
opens a model as its own tab, laid out like the device's classic tab: settings in one column
with Run, Stop and progress under it, plot tabs beside it, results text under the plots.
Several models can be open at once (two funnels, a funnel and a TOF, ...), and the Drivers
tab at the end holds the electronics they share.

Models are stages: an OA-TOF can take its beam and ion mixture from the ions that left a
quadrupole, ion funnel or Device builder model (Incoming beam > Ions from).
"""
from PySide6 import QtCore, QtWidgets

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
    "OA-TOF": (TofTab, "Pusher, drift, reflectron and detector: resolution, mass spectra, tuning. Can take its "
                       "ions from a quadrupole, funnel or Device builder model."),
}
SOURCES = (QuadTab, FunnelTab, DeviceTab)


class Model:
    """One device simulation in the workspace: a name, its template and its controller (the device's tab)."""

    def __init__(self, name, template):
        self.name, self.template = name, template
        self.ctl = TEMPLATES[template][0]()
        self.plots = self.ctl.plots
        self._fitted = False

    @property
    def is_source(self):
        """Its runs leave exit ions that a later stage can take."""
        return isinstance(self.ctl, SOURCES)

    @property
    def is_stage(self):
        """It can take its ions from another model."""
        return isinstance(self.ctl, TofTab)

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


class Workspace(QtWidgets.QMainWindow):
    def __init__(self, models=None, persist=True):
        """models: [(template, name or None)] or [(template, name, ions from)] to open; None reopens the
        last session's models. persist: remember the open models for next time (off for tests and screenshots).
        The window always opens on Home."""
        super().__init__()
        self.persist = persist
        self.setWindowTitle(APP_NAME)
        self.resize(1500, 920)
        self.models = []
        self.settings = QtCore.QSettings("IonSim", "IonSim")
        self._chain = self._chain_now = None        # Run with the stages before it: models still to run, the one running

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.setMovable(True)
        self.tabs.tabCloseRequested.connect(lambda i: self._remove(self._model_at(i)))
        self.tabs.currentChanged.connect(self._tab_changed)
        bar = self.tabs.tabBar()
        bar.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        bar.customContextMenuRequested.connect(self._tab_menu)
        bar.tabBarDoubleClicked.connect(lambda i: self._rename(self._model_at(i)))
        self.home = self._start_page()
        self.drivers = DriversTab()
        for w, name in ((self.home, "Home"), (self.drivers, "Drivers")):
            self.tabs.addTab(w, name)
            i = self.tabs.indexOf(w)
            bar.setTabButton(i, QtWidgets.QTabBar.RightSide, None)
            bar.setTabButton(i, QtWidgets.QTabBar.LeftSide, None)
        self.tabs.setTabToolTip(self.tabs.indexOf(self.drivers),
                                "The electronics: RF, TWAVE, excitation tones and pulses, wired to electrodes in the models")
        self.setCentralWidget(self.tabs)
        self._menus()
        self.statusBar().showMessage("Pick a device to set up, or a tab to carry on")

        rows = models if models is not None else (self._saved_models() if persist else [])
        rows = [tuple(r) + (None,) * (3 - len(r)) for r in rows if r[0] in TEMPLATES]
        for tpl, name, _ in rows:
            self.add_model(tpl, name, select=False)
        for m, (_, _, src) in zip(self.models, rows):
            if src and m.is_stage:
                self.link(m, src)
        self.tabs.setCurrentWidget(self.home)

    # ---------- menus ----------
    def _menus(self):
        m = self.menuBar().addMenu("&File")
        act = m.addAction("Home"); act.setShortcut("Ctrl+H"); act.triggered.connect(lambda: self.tabs.setCurrentWidget(self.home))
        add = m.addMenu("Add model")
        for tpl in TEMPLATES:
            add.addAction(tpl).triggered.connect(lambda _=False, t=tpl: self.add_model(t))
        act = m.addAction("Rename model…"); act.triggered.connect(lambda: self._rename(self._current_model()))
        act = m.addAction("Close model"); act.setShortcut("Ctrl+W"); act.triggered.connect(lambda: self._remove(self._current_model()))
        m.addSeparator()
        act = m.addAction("Save current plot…"); act.setShortcut("Ctrl+S"); act.triggered.connect(self.save_plot)
        act = m.addAction("Export results (CSV)…"); act.setShortcut("Ctrl+E"); act.triggered.connect(self.export_csv)
        m.addSeparator()
        act = m.addAction("Quit"); act.setShortcut("Ctrl+Q"); act.triggered.connect(self.close)
        r = self.menuBar().addMenu("&Run")
        act = r.addAction("Run this model"); act.setShortcut("F5"); act.triggered.connect(self.run_current)
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
        self.tabs.insertTab(self.tabs.indexOf(self.drivers), m.ctl, name)
        m.ctl.ran.connect(lambda ok, m=m: self._ran(m, ok))
        if m.is_stage:
            m.ctl.beam_src.currentIndexChanged.connect(lambda _=0, m=m: self._stage_changed(m))
        self._relink()
        if select:
            self.tabs.setCurrentWidget(m.ctl)
        return m

    def _model_at(self, i):
        w = self.tabs.widget(i)
        return next((m for m in self.models if m.ctl is w), None)

    def _remove(self, m):
        if m is None:
            return
        if QtWidgets.QMessageBox.question(self, APP_NAME, f"Close '{m.name}'? Its settings and results are dropped.") \
                != QtWidgets.QMessageBox.Yes:
            return
        m.ctl.stop()
        if m.ctl._thread is not None:
            m.ctl._thread.quit(); m.ctl._thread.wait(3000)
        self.tabs.removeTab(self.tabs.indexOf(m.ctl)); m.ctl.setParent(None)
        self.models.remove(m)
        self._relink()
        self.tabs.setCurrentWidget(self.models[-1].ctl if self.models else self.home)

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
            order = sorted(self.models, key=lambda m: self.tabs.indexOf(m.ctl))
            self.settings.setValue("workspace/models", "\n".join(
                f"{m.template}\t{m.name}\t{(m.ctl.source_name() or '') if m.is_stage else ''}" for m in order))
        self._fill_open()

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
        i = self.tabs.indexOf(m.ctl)
        self.tabs.setTabText(i, f"{m.name}  ← {src.name}" if src else m.name)
        self.tabs.setTabToolTip(i, TEMPLATES[m.template][1] + (f"\nIons from the exit of '{src.name}'" if src else ""))

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

    # ---------- tabs ----------
    def _current_model(self):
        return self._model_at(self.tabs.currentIndex())

    def _tab_changed(self, i):
        m = self._model_at(i)
        if m is not None:
            QtCore.QTimer.singleShot(0, m.fit)        # once it is laid out

    def _tab_menu(self, pos):
        bar = self.tabs.tabBar()
        m = self._model_at(bar.tabAt(pos))
        menu = QtWidgets.QMenu(self)
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
            menu.addAction("Close").triggered.connect(lambda: self._remove(m))
            menu.addSeparator()
        add = menu.addMenu("Add model")
        for tpl in TEMPLATES:
            add.addAction(tpl).triggered.connect(lambda _=False, t=tpl: self.add_model(t))
        menu.exec(bar.mapToGlobal(pos))

    # ---------- Home ----------
    def _start_page(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(40, 30, 40, 30)
        title = QtWidgets.QLabel("What do you want to simulate?")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        v.addWidget(title)
        sub = QtWidgets.QLabel("Each pick opens a model in its own tab. Several can be open at once, and the Drivers "
                               "tab holds the electronics they share. An OA-TOF can take its ions from a quadrupole, "
                               "ion funnel or Device builder model, so models can be chained into the stages of one "
                               "instrument.")
        sub.setWordWrap(True); v.addWidget(sub); v.addSpacing(12)
        grid = QtWidgets.QGridLayout()
        for i, (tpl, (_, desc)) in enumerate(TEMPLATES.items()):
            b = QtWidgets.QPushButton(tpl); b.setMinimumHeight(44)
            b.setStyleSheet("text-align: left; padding: 6px 12px; font-weight: bold;")
            b.clicked.connect(lambda _=False, t=tpl: self.add_model(t))
            lab = QtWidgets.QLabel(desc); lab.setWordWrap(True)
            grid.addWidget(b, i, 0); grid.addWidget(lab, i, 1)
        grid.setColumnStretch(1, 1)
        v.addLayout(grid)
        v.addSpacing(16)
        self.open_title = QtWidgets.QLabel("Open models")
        self.open_title.setStyleSheet("font-size: 15px; font-weight: bold;")
        v.addWidget(self.open_title)
        self.open_row = QtWidgets.QHBoxLayout()
        v.addLayout(self.open_row)
        v.addStretch(1)
        return w

    def _fill_open(self):
        """Home lists the open models, so the last session's work is one click away."""
        while self.open_row.count():
            it = self.open_row.takeAt(0)
            if it.widget():
                it.widget().hide(); it.widget().setParent(None)
        order = sorted(self.models, key=lambda m: self.tabs.indexOf(m.ctl))
        self.open_title.setVisible(bool(order))
        for m in order:
            src = self.source_of(m)
            b = QtWidgets.QPushButton(f"{m.name}  ← {src.name}" if src else m.name)
            b.setMinimumHeight(36)
            b.clicked.connect(lambda _=False, m=m: self.tabs.setCurrentWidget(m.ctl))
            self.open_row.addWidget(b)
        self.open_row.addStretch(1)

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
                                                  if self.tabs.currentWidget() is self.drivers else None)
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
            "<p>Pick a device on Home to open it as a tab, set it up and press Run (F5). "
            "The Drivers tab is shared by every model.</p>"
            "<p>An OA-TOF can take its ions from another model (Incoming beam &gt; Ions from); "
            "Run &gt; Run with the stages before it (F6) runs the chain in order.</p>"
            "<p>The window with one fixed tab per device is under View &gt; Classic tabs window.</p>")

    def closeEvent(self, e):
        self._save_models()
        for m in self.models:
            m.ctl.stop()
            if m.ctl._thread is not None:
                m.ctl._thread.quit(); m.ctl._thread.wait(3000)
        e.accept()
