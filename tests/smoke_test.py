"""Quick end-to-end check that an install works (about a minute). Run:
    python tests/smoke_test.py          (set QT_QPA_PLATFORM=offscreen on a headless machine)
Exits non-zero with a message if anything is broken."""
import os, sys, time
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
GEO = os.path.join(ROOT, "examples", "geometry")
t0 = time.time()


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        sys.exit(1)


from ionsim.dxf import load_dxf
from ionsim.device import from_drawing, from_bodies, DeviceField, fly_device
from ionsim.cad import load_bodies
from ionsim import mathieu

# 2D cross-section: quadrupole at a stable and an unstable Mathieu point
dev = from_drawing(load_dxf(os.path.join(GEO, "quad_rods.dxf")), "planar2d")
dev.length = 0.1
A = [e.name for e in dev.electrodes if e.layer == "ROD_A"]
B = [e.name for e in dev.electrodes if e.layer == "ROD_B"]
dg = dev.build(dev.default_h("Quick"))
for (a, q), want in (((0.2, 0.7), 1.0), ((0.26, 0.7), 0.0)):
    U, V = mathieu.voltages_for(a, q, 500, 4e-3, 1e6)
    dev.apply_fixed(A, U, V, 0); dev.apply_fixed(B, -U, V, 180)
    res, _ = fly_device(DeviceField(dg), 500, 1.0, 30, 1e-7, energy_ev=5, radius=0.3e-3, z_start=0, z_end=0.1,
                        rng=np.random.default_rng(0))
    check(f"quad DXF a={a} q={q}", abs(res.transmission - want) < 0.15, f"transmission {res.transmission:.0%}")

# axisymmetric profile: ring funnel
dev = from_drawing(load_dxf(os.path.join(GEO, "ring_funnel_profile.dxf")), "axisym")
dev.apply_gradient([e.name for e in dev.electrodes if e.layer == "RINGS"], 78, 20, rf=50)
dev.apply_fixed([e.name for e in dev.electrodes if e.layer == "EXIT"], 15)
res, _ = fly_device(DeviceField(dev.build(dev.default_h("Quick"))), 500, 1.0, 40, 4.0, radius=3e-3,
                    z_start=0.5e-3, z_end=31e-3, rng=np.random.default_rng(0))
check("funnel profile DXF", res.transmission > 0.8, f"transmission {res.transmission:.0%}")

# boards: TWAVE pads build and solve
dev = from_drawing(load_dxf(os.path.join(GEO, "twave_board.dxf")), "boards")
dev.apply_twave([e.name for e in dev.electrodes if e.layer == "TW"])
check("TWAVE pitch measured", abs(dev.twave.pitch - 0.508e-3) < 1e-6, f"{dev.twave.pitch * 1e3:.3f} mm")

# STEP import
bodies = load_bodies(os.path.join(GEO, "ring_funnel_30.step"))
check("STEP import", len(bodies) == 31 and bodies[0][0] == "ring 1", f"{len(bodies)} bodies, first '{bodies[0][0]}'")

# the app window builds
from PySide6 import QtWidgets
from ionsim.gui.app import MainWindow
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
w = MainWindow()
check("app window", w.tabs.count() >= 6, f"{w.tabs.count()} tabs")
print(f"all good in {time.time() - t0:.0f} s")
