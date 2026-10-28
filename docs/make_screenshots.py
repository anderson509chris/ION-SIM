"""Regenerate the app screenshots used in README.md.

Run:  python docs/make_screenshots.py      (opens the window, runs every tab, ~2 min)
"""
import os, sys, tempfile
os.environ.setdefault("IONSIM_DRIVERS", os.path.join(tempfile.mkdtemp(), "drivers.json"))   # the default drivers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from PySide6 import QtCore, QtWidgets
from ionsim.gui.app import MainWindow

OUT = os.path.join(os.path.dirname(__file__), "screenshots")
os.makedirs(OUT, exist_ok=True)
app = QtWidgets.QApplication(sys.argv)
app.setStyle("Fusion")
w = MainWindow(); w.resize(1400, 860); w.show()


def settle(n=30):
    for _ in range(n):
        app.processEvents(); QtCore.QThread.msleep(20)


def run(tab):
    w.tabs.setCurrentWidget(tab)
    tab.run(); settle(5)
    while tab._thread is not None:
        settle(1)
    settle(50)


def shot(tab, plot, name):
    tab.plots.setCurrentIndex(list(tab.plots.figs).index(plot))
    settle()
    w.grab().save(os.path.join(OUT, name + ".png"))
    print("saved", name)


def go():
    run(w.quad)
    shot(w.quad, "Trajectories", "quad_trajectories")
    shot(w.quad, "Stability", "quad_stability")
    # quadrupole driven by the 8-tone driver: excitation scan
    q = w.quad
    q.geom.set("div", 50)
    q.drive.setCurrentText("Quad RF + 8 tones"); settle()
    q.mode.setCurrentIndex(1)
    q.dscan.set("m1", 150); q.dscan.set("m2", 2500); q.dscan.set("npts", 60); q.dscan.set("per", 20)
    run(q)
    shot(q, "Mass scan", "quad_excitation_scan")
    shot(q, "Excitation", "quad_excitation_tones")
    q.drive.setCurrentIndex(0); q.geom.set("div", 80)
    # Drivers tab: the TWAVE/RF combiner
    dr = w.drivers
    w.tabs.setCurrentWidget(dr)
    dr.list.setCurrentRow(dr.store.names().index("TWAVE/RF combiner")); settle()
    dr.form.set("rise", 100.0); settle()
    dr.outs.selectRow(0); settle()
    for r in (1, 2, 3):
        dr.outs.selectionModel().select(dr.outs.model().index(r, 0),
                                        QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)
    settle()
    shot(dr, "Waveforms", "drivers_waveforms")
    shot(dr, "TWAVE sequence", "drivers_twave_sequence")
    run(w.twave)
    shot(w.twave, "Arrival times", "twave_arrival_times")
    shot(w.twave, "TW field", "twave_field")
    run(w.faims)
    shot(w.faims, "CV spectrum", "faims_cv_spectrum")
    shot(w.faims, "Alpha", "faims_alpha")
    run(w.inlet)
    shot(w.inlet, "Ion energy", "inlet_ion_energy")
    shot(w.inlet, "Gas flow", "inlet_gas_flow")
    w.funnel.flow.setCurrentIndex(2); w.funnel.space.set("I", 1.0)     # PCB rings, gas flow field, 1 nA beam
    run(w.funnel)
    shot(w.funnel, "Trajectories", "funnel_trajectories")
    shot(w.funnel, "Potential", "funnel_potential")
    shot(w.funnel, "Gas flow", "funnel_gas_flow")
    shot(w.funnel, "Heating", "funnel_heating")
    w.funnel.flow.setCurrentIndex(0); w.funnel.space.set("I", 0.0)
    w.funnel.kind.setCurrentIndex(2)
    w.funnel.load_geometry(os.path.join(os.path.dirname(__file__), "..", "examples", "geometry", "ring_funnel_30.step"))
    w.funnel.ions.set("r0", 3.5)
    run(w.funnel)
    shot(w.funnel, "Geometry", "funnel_cad_geometry")
    # Device builder: TWAVE board DXF with patterns, then a funnel half profile
    d = w.device
    geo = os.path.join(os.path.dirname(__file__), "..", "examples", "geometry")
    w.tabs.setCurrentWidget(d)
    d.open_setup_from(os.path.join(geo, "twave_board.ionsim"))
    d._picked("TW 9", False)
    shot(d, "Geometry", "device_twave_board")
    run(d)
    shot(d, "Trajectories", "device_twave_trajectories")
    d.open_setup_from(os.path.join(geo, "twave_board_combiner.ionsim"))
    run(d)
    shot(d, "Heating", "device_combiner_heating")
    d.open_setup_from(os.path.join(geo, "ring_funnel_profile.ionsim"))
    run(d)
    shot(d, "Trajectories", "device_funnel_profile")
    # OA-TOF: single shot, then a back-plate scan
    t = w.tof
    t.mode.setCurrentIndex(0)
    run(t)
    shot(t, "Layout", "tof_layout")
    shot(t, "Mass spectrum", "tof_mass_spectrum")
    shot(t, "Peak width", "tof_peak_width")
    t.mode.setCurrentIndex(1); t.scan_param.setCurrentIndex(0); settle()
    run(t)
    shot(t, "Scan", "tof_scan")
    workspace()
    app.quit()


def workspace():
    """The workspace window: a funnel and an OA-TOF model, the TOF run."""
    from ionsim.gui.workspace import Workspace
    ws = Workspace(models=[("Ion funnel", None), ("OA-TOF", None)], persist=False)
    ws.resize(1500, 900); ws.show(); settle()
    m = ws.models[1]
    ws.tree.setCurrentItem(m.items["Electrodes and drive"]); settle()
    m.ctl.run_btn.click(); settle(5)
    while m.ctl._thread is not None:
        settle(1)
    settle(50)
    ws.tree.setCurrentItem(m.items["Electrodes and drive"]); settle()
    m.plots.setCurrentIndex(list(m.plots.figs).index("Mass spectrum")); settle()
    ws.grab().save(os.path.join(OUT, "workspace.png")); print("saved workspace")
    ws.tree.setCurrentItem(ws.add_item); settle()
    ws.grab().save(os.path.join(OUT, "workspace_add_model.png")); print("saved workspace_add_model")
    ws.close()


QtCore.QTimer.singleShot(500, go)
app.exec()
