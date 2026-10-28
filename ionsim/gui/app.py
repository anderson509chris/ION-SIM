"""Ion Sim desktop application."""
import csv
import sys
from PySide6 import QtGui, QtWidgets

from .quad_tab import QuadTab
from .twave_tab import TwaveTab
from .faims_tab import FaimsTab
from .inlet_tab import InletTab
from .funnel_tab import FunnelTab
from .device_tab import DeviceTab
from .drivers_tab import DriversTab
from .tof_tab import TofTab

APP_NAME = "Ion Sim"
VERSION = "0.1"


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {VERSION}")
        self.resize(1400, 900)
        self.tabs = QtWidgets.QTabWidget()
        self.quad = QuadTab(); self.twave = TwaveTab(); self.faims = FaimsTab(); self.inlet = InletTab(); self.funnel = FunnelTab(); self.device = DeviceTab()
        self.tof = TofTab()
        self.drivers = DriversTab()
        self.tabs.addTab(self.quad, "Quadrupole mass filter")
        self.tabs.addTab(self.twave, "TWAVE traveling wave")
        self.tabs.addTab(self.faims, "FAIMS")
        self.tabs.addTab(self.inlet, "ESI inlet capillary")
        self.tabs.addTab(self.funnel, "Ion funnel")
        self.tabs.addTab(self.device, "Device builder (DXF / CAD)")
        self.tabs.addTab(self.tof, "OA-TOF")
        self.tabs.addTab(self.drivers, "Drivers")
        self.tof.set_sources({"Quadrupole": self.quad, "Ion funnel": self.funnel, "Device builder": self.device})
        self.setCentralWidget(self.tabs)

        m = self.menuBar().addMenu("&File")
        act = m.addAction("Save current plot..."); act.setShortcut("Ctrl+S"); act.triggered.connect(self.save_plot)
        act = m.addAction("Export results (CSV)..."); act.setShortcut("Ctrl+E"); act.triggered.connect(self.export_csv)
        m.addSeparator()
        act = m.addAction("Quit"); act.setShortcut("Ctrl+Q"); act.triggered.connect(self.close)
        h = self.menuBar().addMenu("&Help")
        h.addAction("About").triggered.connect(self.about)
        self.statusBar().showMessage("Set parameters on the left and press Run")

    def save_plot(self):
        tab = self.tabs.currentWidget()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save plot", "plot.png",
                                                        "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            tab.plots.current_figure().savefig(path, dpi=200)
            self.statusBar().showMessage(f"Saved {path}", 5000)

    def export_csv(self):
        tab = self.tabs.currentWidget()
        if not tab.last_table:
            QtWidgets.QMessageBox.information(self, APP_NAME, "Run a simulation first.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export results", "results.csv", "CSV (*.csv)")
        if path:
            header, rows = tab.last_table
            with open(path, "w", newline="") as f:
                w = csv.writer(f); w.writerow(header); w.writerows(rows)
            self.statusBar().showMessage(f"Saved {path}", 5000)

    def about(self):
        QtWidgets.QMessageBox.about(self, APP_NAME,
            f"<b>{APP_NAME} {VERSION}</b><p>Ion optics and ion mobility simulator.</p>"
            "<p>Quadrupole: Laplace field on a grid, RK4 flight, Mathieu stability.<br>"
            "TWAVE: traveling-wave field from the electrode boards, mobility drift + diffusion.<br>"
            "FAIMS: asymmetric waveform in a planar gap, field-dependent mobility, CV scans.<br>"
            "Inlet: ESI spray region, choked capillary flow and free jet, ion velocity and energy.<br>"
            "Funnel: ring, planar or imported CAD electrodes, RF + DC fields, Langevin ion flight in gas.<br>"
            "OA-TOF: pusher pulse, acceleration, drift, reflectron and detector; resolution and mass spectra.<br>"
            "Drivers: the electronics (RF, TWAVE, excitation tones) wired to electrodes in the other tabs.</p>")

    def closeEvent(self, e):
        for tab in (self.quad, self.twave, self.faims, self.inlet, self.funnel, self.device, self.tof):
            tab.stop()
            if tab._thread is not None:
                tab._thread.quit(); tab._thread.wait(3000)
        e.accept()


def main():
    """Opens the workspace; `--classic` opens the one-tab-per-device window instead."""
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    if "--classic" in sys.argv:
        w = MainWindow()
    else:
        from .workspace import Workspace
        w = Workspace()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
