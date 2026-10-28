"""An example of a model loaded from your own folder: a field-free flight tube.

Copy this file into a folder of your own, rename it (keep the _plugin.py ending), and load the folder
with Models > Load models folder… in the app. The model then shows on Home under "Your models".
Everything in that folder stays on your computer; nothing about it needs to be in this repository.
"""
import numpy as np

from ionsim.constants import AMU, E_CHARGE
from ionsim.gui.common import ParamForm, SimTab


class FlightTimeTab(SimTab):
    """Settings on the left, Run, plots on the right: the same layout as every model tab."""

    def __init__(self, parent=None):
        super().__init__(["Flight time"], parent)
        self.form = ParamForm("Flight tube", [
            ("L", "Length", 1000.0, 1, 1e5, 1, 10, "mm"),
            ("V", "Acceleration", 5000.0, 1, 1e5, 1, 100, "V"),
            ("m1", "From m/z", 10.0, 1, 1e6, 1, 10, ""),
            ("m2", "To m/z", 2000.0, 1, 1e6, 1, 100, ""),
        ])
        self.form_layout.addWidget(self.form)
        self.form_layout.addStretch(1)
        self.run_btn.clicked.connect(self.run)

    def run(self):
        p = self.form.values()

        def job(progress):                       # runs in the background; progress(f) returns True on Stop
            m = np.geomspace(p["m1"], p["m2"], 200)
            v = np.sqrt(2 * E_CHARGE * p["V"] / (m * AMU))
            return m, p["L"] * 1e-3 / v
        self.start(job, self._show)

    def _show(self, out):                        # back in the window: draw, log, and give the results
        m, t = out
        ax = self.plots.fig("Flight time").add_subplot(111)
        ax.plot(m, t * 1e6)
        ax.set_xlabel("m/z"); ax.set_ylabel("flight time (µs)"); ax.grid(alpha=0.3)
        self.plots.draw("Flight time")
        self.log(f"m/z {m[0]:g}: {t[0] * 1e6:.3f} µs, m/z {m[-1]:g}: {t[-1] * 1e6:.3f} µs")
        self.last_table = (["mz", "flight_time_s"], [[float(a), float(b)] for a, b in zip(m, t)])   # File > Export CSV
        self.metrics = {"flight time at the last m/z (µs)": float(t[-1] * 1e6)}                   # read by the Sweeps tab


MODELS = {"Flight tube (example plugin)": (FlightTimeTab, "Field-free flight time against m/z: an example to copy "
                                                          "for your own models.")}
