"""The Ion Sim workspace: a start page, then one tab per model.

The window opens on Home, the start page, where you pick the devices to set up. Each pick
opens a model as its own tab, laid out like the device's classic tab: settings in one column
with Run, Stop and progress under it, plot tabs beside it, results text under the plots.
Several models can be open at once (two funnels, a funnel and a TOF, ...), and the Drivers
tab at the end holds the electronics they share.

Models are stages: an ion funnel can take its gas and ion species from an ESI inlet, a
quadrupole its ions from a funnel or Device builder, and an OA-TOF its beam from any of those
(the "... from" row at the top of the model's gas or beam settings). Chained, inlet → funnel →
quadrupole (gas-filled as a collision cell, or not) → OA-TOF runs in order with F6.
"""
import json
import os
from PySide6 import QtCore, QtWidgets

from ..drivers import LIBRARY_PATH, Driver
from . import settings_io

from .quad_tab import QuadTab
from .twave_tab import TwaveTab
from .faims_tab import FaimsTab
from .inlet_tab import InletTab
from .funnel_tab import FunnelTab
from .device_tab import DeviceTab
from .tof_tab import TofTab
from .drivers_tab import DriversTab
from .instrument_tab import InstrumentTab

APP_NAME = "Ion Sim"
SETUP_FILTER = "Ion Sim setup (*.ionsetup)"
LAST_SETUP = os.path.join(os.path.dirname(LIBRARY_PATH), "last_setup.ionsetup")   # brought back on the next start

# template name -> (controller class, one-line description)
TEMPLATES = {
    "Quadrupole mass filter": (QuadTab, "Round rods with RF + DC or a driver with excitation tones; Mathieu "
                                        "stability, mass scans, tone ejection. With gas: a collision cell. Can take "
                                        "its ions from a funnel."),
    "TWAVE traveling wave": (TwaveTab, "Stepped electrode boards in gas: mobility separation, wave-speed sweeps."),
    "FAIMS": (FaimsTab, "Asymmetric waveform across a planar gap at 1 atm: CV spectra and dispersion."),
    "ESI inlet capillary": (InletTab, "Spray region, choked capillary flow and free jet: ion speed and energy."),
    "Ion funnel": (FunnelTab, "Ring, planar or imported CAD funnels in 1-10 Torr: transmission, gas flow, "
                              "space charge, heating. Can take its gas and ions from an ESI inlet."),
    "Device builder (DXF / CAD)": (DeviceTab, "Your own drawing or 3D model: click electrodes, wire drivers, "
                                              "solve and fly."),
    "OA-TOF": (TofTab, "Pusher, drift, reflectron and detector: resolution, mass spectra, tuning. Can take its "
                       "ions from a quadrupole, funnel or Device builder model."),
}


class Model:
    """One device simulation in the workspace: a name, its template and its controller (the device's tab)."""

    def __init__(self, name, template):
        self.name, self.template = name, template
        self.ctl = TEMPLATES[template][0]()
        self.plots = self.ctl.plots
        self._fitted = False

    @property
    def is_stage(self):
        """It can take its ions (or gas) from another model."""
        return hasattr(self.ctl, "picker")

    def can_take(self, other):
        return self.is_stage and other is not self and isinstance(other.ctl, self.ctl.picker.accepts)

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
        self.instrument = InstrumentTab(TEMPLATES)
        self.instrument.make_models.connect(self.make_stages)
        self.drivers = DriversTab()
        for w, name in ((self.home, "Home"), (self.instrument, "Instrument"), (self.drivers, "Drivers")):
            self.tabs.addTab(w, name)
            i = self.tabs.indexOf(w)
            bar.setTabButton(i, QtWidgets.QTabBar.RightSide, None)
            bar.setTabButton(i, QtWidgets.QTabBar.LeftSide, None)
        self.tabs.setTabToolTip(self.tabs.indexOf(self.instrument),
                                "Load a CAD model of the whole instrument and say which model each part is")
        self.tabs.setTabToolTip(self.tabs.indexOf(self.drivers),
                                "The electronics: RF, TWAVE, excitation tones and pulses, wired to electrodes in the models")
        self.setCentralWidget(self.tabs)
        self._menus()
        self.statusBar().showMessage("Pick a device to set up, or a tab to carry on")

        self.path = None                            # the setup file open now
        if models is None and persist and os.path.exists(LAST_SETUP):
            try:
                self.apply_setup(self._read(LAST_SETUP), os.path.dirname(LAST_SETUP), quiet=True)
            except Exception as e:                  # a damaged last setup must never stop the app from opening
                self.statusBar().showMessage(f"Could not bring back the last setup: {e}", 10000)
        else:
            rows = models if models is not None else (self._saved_models() if persist else [])
            self._open_rows(rows)
        self.tabs.setCurrentWidget(self.home)
        self._title()

    def _open_rows(self, rows):
        rows = [tuple(r) + (None,) * (3 - len(r)) for r in rows if r[0] in TEMPLATES]
        for tpl, name, _ in rows:
            self.add_model(tpl, name, select=False)
        for m, (_, _, src) in zip(self.models[-len(rows):] if rows else [], rows):
            if src and m.is_stage:
                self.link(m, src)

    # ---------- menus ----------
    def _menus(self):
        m = self.menuBar().addMenu("&File")
        act = m.addAction("Home"); act.setShortcut("Ctrl+H"); act.triggered.connect(lambda: self.tabs.setCurrentWidget(self.home))
        act = m.addAction("New setup"); act.triggered.connect(self.new_setup)
        act = m.addAction("Open setup…"); act.setShortcut("Ctrl+O"); act.triggered.connect(self.open_setup)
        act = m.addAction("Save setup"); act.setShortcut("Ctrl+Shift+S"); act.triggered.connect(self.save_setup)
        act = m.addAction("Save setup as…"); act.triggered.connect(lambda: self.save_setup(ask=True))
        m.addSeparator()
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
            m.ctl.picker.combo.currentIndexChanged.connect(lambda _=0, m=m: self._stage_changed(m))
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
        self._drop(m)
        self.tabs.setCurrentWidget(self.models[-1].ctl if self.models else self.home)

    def _drop(self, m):
        m.ctl.stop()
        if m.ctl._thread is not None:
            m.ctl._thread.quit(); m.ctl._thread.wait(3000)
        self.tabs.removeTab(self.tabs.indexOf(m.ctl)); m.ctl.setParent(None)
        self.models.remove(m)
        self._relink()

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

    # ---------- setup files ----------
    def setup_state(self, base_dir=None):
        """The whole workspace as plain data: every model's settings and link, and the drivers.
        Geometry files are named relative to base_dir when given. Results are not kept."""
        from .device_tab import DeviceTab
        from .funnel_tab import FunnelTab
        rows = []
        for m in sorted(self.models, key=lambda m: self.tabs.indexOf(m.ctl)):
            row = {"template": m.template, "name": m.name,
                   "ions_from": m.ctl.source_name() if m.is_stage else None}
            if isinstance(m.ctl, DeviceTab):
                row["device"] = m.ctl.setup_dict(base_dir)
            else:
                row["settings"] = settings_io.save(m.ctl.scroll.widget())
            if isinstance(m.ctl, FunnelTab) and m.ctl.cad_path:
                row["cad"] = _rel(m.ctl.cad_path, base_dir)
            rows.append(row)
        return {"ionsim_setup": 1, "models": rows, "drivers": self.drivers.store.lib.to_list(),
                "instrument": self.instrument.state(base_dir)}

    def apply_setup(self, d, base_dir="", quiet=False):
        """Replace the open models with the ones in a setup_state."""
        from .device_tab import DeviceTab
        from .funnel_tab import FunnelTab
        if "ionsim_setup" not in d:
            raise ValueError("not an Ion Sim setup file")
        rename = self._adopt_drivers(d.get("drivers", []), quiet)
        for m in list(self.models):
            self._drop(m)
        rows = [r for r in d.get("models", []) if r.get("template") in TEMPLATES]
        self._open_rows([(r["template"], r["name"], r.get("ions_from")) for r in rows])
        notes = []
        for m, r in zip(self.models, rows):
            if isinstance(m.ctl, DeviceTab):
                if r.get("device"):
                    m.ctl.apply_setup(_renamed(r["device"], rename), base_dir)
                continue
            if isinstance(m.ctl, FunnelTab) and r.get("cad"):
                path = _abs(r["cad"], base_dir)
                if os.path.exists(path):
                    m.ctl.load_geometry(path)
                else:
                    notes.append(f"'{m.name}': geometry file {path} not found")
            bad = settings_io.restore(m.ctl.scroll.widget(), _renamed(r.get("settings", {}), rename))
            if bad:
                notes.append(f"'{m.name}': " + ", ".join(b.split(" / ")[-1] for b in bad) + " no longer available")
            m._fitted = False
        for m, r in zip(self.models, rows):                 # links again, now every model's settings are in
            if m.is_stage:
                self.link(m, r.get("ions_from"))
        if d.get("instrument") and not self.instrument.set_state(d["instrument"], base_dir):
            notes.append("the instrument CAD file was not found")
        self._save_models()
        if notes and not quiet and QtWidgets.QApplication.platformName() != "offscreen":
            box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Information, APP_NAME,
                                        "Opened, with notes:\n" + "\n".join(notes), parent=self)
            box.setModal(False); box.show()
        return notes

    def _adopt_drivers(self, rows, quiet):
        """The setup's drivers go into the library. One that differs from the library's driver of the same
        name comes in as '<name> (from setup)', and the models use that one. Returns {old name: new name}."""
        store, rename = self.drivers.store, {}
        for row in rows:
            drv = Driver.from_dict(row)
            have = store.get(drv.name)
            if have is not None and have.to_dict() == drv.to_dict():
                continue
            if have is not None:
                new = store.lib.add(drv.copy(f"{drv.name} (from setup)"), replace=False)
                rename[drv.name] = new
            else:
                store.lib.add(drv.copy())
        if rows:
            store.touch()
        return rename

    def _read(self, path):
        with open(path) as f:
            return json.load(f)

    def _write(self, path, base_dir=None):
        d = self.setup_state(base_dir)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(d, f, indent=1)
        os.replace(tmp, path)

    def new_setup(self):
        if self.models and QtWidgets.QMessageBox.question(
                self, APP_NAME, "Close every model and start an empty setup?") != QtWidgets.QMessageBox.Yes:
            return
        for m in list(self.models):
            self._drop(m)
        self.path = None
        self._save_models(); self._title()
        self.tabs.setCurrentWidget(self.home)

    def open_setup(self, path=None):
        if not path:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open setup", os.path.dirname(self.path or ""), SETUP_FILTER)
        if not path:
            return
        try:
            notes = self.apply_setup(self._read(path), os.path.dirname(os.path.abspath(path)))
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, APP_NAME, f"Could not open {path}:\n{e}")
            return
        self.path = path
        self._title()
        self.statusBar().showMessage(f"Opened {os.path.basename(path)}" + (" (with notes)" if notes else ""), 8000)
        self.tabs.setCurrentWidget(self.models[-1].ctl if self.models else self.home)

    def save_setup(self, ask=False):
        path = self.path
        if ask or not path:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save setup", self.path or "instrument.ionsetup", SETUP_FILTER)
        if not path:
            return
        if not path.endswith(".ionsetup"):
            path += ".ionsetup"
        try:
            self._write(path, os.path.dirname(os.path.abspath(path)))
        except OSError as e:
            QtWidgets.QMessageBox.warning(self, APP_NAME, f"Could not save {path}:\n{e}")
            return
        self.path = path
        self._title()
        self.statusBar().showMessage(f"Saved {path}", 6000)

    def _title(self):
        self.setWindowTitle(f"{os.path.basename(self.path)} – {APP_NAME}" if self.path else APP_NAME)

    # ---------- stages ----------
    def make_stages(self, stages):
        """From the Instrument tab: [(template, stage name, z range)] in order along the ion path. Opens a tab
        for each stage that has none, and links each to the nearest stage before it that can feed it."""
        made, chain = [], []
        for tpl, name, z in stages:
            m = next((x for x in self.models if x.name == name), None)
            if m is None:
                m = self.add_model(tpl, name, select=False)
                made.append(m)
            m.region = z
            chain.append(m)
        for k, m in enumerate(chain):
            src = next((p for p in reversed(chain[:k]) if m.can_take(p)), None)
            if m.is_stage and src is not None:
                self.link(m, src.name)
        self.statusBar().showMessage(
            (f"Opened {', '.join(x.name for x in made)}. " if made else "Every stage already has a tab. ")
            + "Load each one's geometry in its tab; F6 on the last runs the chain.", 12000)
        if made:
            self.tabs.setCurrentWidget(made[0].ctl)

    def link(self, stage, source):
        """Make `stage` take its ions (or gas) from the model named `source`; None sets them by hand."""
        stage.ctl.picker.pick(source)

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
            m.ctl.picker.combo.blockSignals(True)
            m.ctl.picker.set_sources({x.name: x.ctl for x in self.models if m.can_take(x)})
            m.ctl.picker.combo.blockSignals(False)
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

    def speed_rows(self):
        """The models that have run, in order along the instrument, each with where its axis starts: at
        its tagged place in the Instrument tab's CAD model, or else after the stage before it."""
        order, seen = [], set()

        def chain_of(m):
            c, x = [m], self.source_of(m)
            while x is not None and x not in c:
                c.insert(0, x); x = self.source_of(x)
            return c
        for m in sorted(self.models, key=lambda m: -len(chain_of(m))):      # the longest chains first
            for x in chain_of(m):
                if x not in seen:
                    seen.add(x); order.append(x)
        tagged = [m for m in order if getattr(m, "region", None)]
        if tagged:
            order = sorted(tagged, key=lambda m: m.region[0]) + [m for m in order if m not in tagged]
        rows, end = [], None
        for m in order:
            sp = m.ctl.speeds
            if sp is None:
                continue
            z0, z1 = sp["span"]
            if getattr(m, "region", None):
                off = m.region[0] * 1e-3 - z0
            else:
                off = (0.0 if end is None else end + 5e-3) - z0
            rows.append((m.name, sp, off))
            end = z1 + off
        return rows

    def _ran(self, m, ok):
        if ok:
            self.instrument.show_speeds(self.speed_rows())
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
                for x in [None] + [x for x in self.models if m.can_take(x)]:
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
                               "tab holds the electronics they share. Models chain into the stages of one instrument: "
                               "ESI inlet → ion funnel → quadrupole (or collision cell) → OA-TOF. To start from a CAD "
                               "model of the whole instrument, use the Instrument tab: tag its parts and it opens the "
                               "linked models for you.")
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
            "<p>A funnel can take its gas from an inlet, a quadrupole its ions from a funnel, and an OA-TOF its beam "
            "from either (the “… from” row in their settings); "
            "Run &gt; Run with the stages before it (F6) runs the chain in order.</p>"
            "<p>The window with one fixed tab per device is under View &gt; Classic tabs window.</p>")

    def closeEvent(self, e):
        self._save_models()
        if self.persist:
            try:
                self._write(LAST_SETUP)
            except Exception as err:            # never block quitting
                print("Could not keep the setup for next time:", err)
        for m in self.models:
            m.ctl.stop()
            if m.ctl._thread is not None:
                m.ctl._thread.quit(); m.ctl._thread.wait(3000)
        e.accept()


def _rel(path, base_dir):
    return os.path.relpath(path, base_dir) if base_dir else os.path.abspath(path)


def _abs(path, base_dir):
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base_dir or "", path))


def _renamed(state, rename):
    """A saved state with driver names swapped for the names they came in under."""
    if not rename:
        return state
    def fix(v):
        if isinstance(v, str):
            for old, new in rename.items():
                if v == old or v.startswith(old + " / "):
                    return new + v[len(old):]
            return v
        if isinstance(v, dict):
            return {k: fix(x) for k, x in v.items()}
        if isinstance(v, list):
            return [fix(x) for x in v]
        return v
    return fix(state)
