"""Regenerate the app screenshots used in README.md.

Run:  python docs/make_screenshots.py      (opens the window, runs every tab, ~2 min)
"""
import os, sys
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
    run(w.twave)
    shot(w.twave, "Arrival times", "twave_arrival_times")
    shot(w.twave, "TW field", "twave_field")
    run(w.faims)
    shot(w.faims, "CV spectrum", "faims_cv_spectrum")
    shot(w.faims, "Alpha", "faims_alpha")
    run(w.inlet)
    shot(w.inlet, "Ion energy", "inlet_ion_energy")
    shot(w.inlet, "Gas flow", "inlet_gas_flow")
    run(w.funnel)
    shot(w.funnel, "Trajectories", "funnel_trajectories")
    shot(w.funnel, "Potential", "funnel_potential")
    w.funnel.kind.setCurrentIndex(2)
    w.funnel.load_geometry(os.path.join(os.path.dirname(__file__), "..", "examples", "geometry", "ring_funnel_30.step"))
    w.funnel.ions.set("r0", 3.5)
    run(w.funnel)
    shot(w.funnel, "Geometry", "funnel_cad_geometry")
    app.quit()


QtCore.QTimer.singleShot(500, go)
app.exec()
