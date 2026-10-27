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

# dielectric: a slab between plates must match the series-capacitor result exactly
from ionsim.grid3d import Grid3D
from ionsim.axisym import AxisymGrid
from ionsim.gridtools import GridMap, EPS0
from ionsim.gasflow import GasFlow
h = 0.25e-3
g = Grid3D((-1e-3, 1e-3, -1e-3, 1e-3, 0, 10e-3), h, open_faces=("x", "y"))
g.add_box("top", -1e-3, 1e-3, -1e-3, 1e-3, 10e-3, 10e-3)
g.add_dielectric_box(-1e-3, 1e-3, -1e-3, 1e-3, 0.25e-3, 4.75e-3, 4.0)
phi = g.solver().solve(np.where(g.electrode > 0, 100.0, 0.0))
want = 100 * (19 / 4) / (19 / 4 + 1 / 1.6 + 20)        # 19 links in eps 4, one interface link, 20 in vacuum
check("dielectric slab", abs(phi[4, 4, 19] - want) < 1e-3, f"{phi[4, 4, 19]:.4f} V vs {want:.4f} V")

# space charge: uniformly charged beam in a grounded tube
a_, R, L, rho = 1e-3, 4e-3, 30e-3, 1e-6
rng = np.random.default_rng(0)
r = a_ * np.sqrt(rng.random(200_000)); th = 2 * np.pi * rng.random(200_000)
pts = np.column_stack([r * np.cos(th), r * np.sin(th), L * rng.random(200_000)])
ga = AxisymGrid(R, 0, L, 0.05e-3); ga.add_annulus("tube", R, R, 0, L)
gm = GridMap.of(ga)
phi = ga.solver().solve(np.zeros(gm.shape), source=gm.charge_source(gm.deposit(pts, rho * np.pi * a_ ** 2 * L / len(pts))))
want = rho * a_ ** 2 / (4 * EPS0) * (1 + 2 * np.log(R / a_))
check("space charge (Poisson)", abs(phi[0, phi.shape[1] // 2] / want - 1) < 0.02, f"{phi[0, phi.shape[1] // 2]:.4f} V vs {want:.4f} V")

# gas flow: Poiseuille profile in a pipe
gp = AxisymGrid(5e-3, 0, 40e-3, 0.1e-3); gp.add_annulus("wall", 3e-3, 5e-3, 0, 40e-3)
fl = GasFlow(gp, 1e-3, 3e-3)
u = fl.velocity(np.array([[0, 0, 25e-3], [1.5e-3, 0, 25e-3]]))[:, 2] / (1e-3 / (np.pi * 2.95e-3 ** 2))
check("gas flow (Poiseuille)", abs(u[0] - 2) < 0.12 and abs(u[1] - 2 * (1 - (1.5 / 2.95) ** 2)) < 0.12, f"u/u_mean {u[0]:.2f}, {u[1]:.2f}")

# device: a part marked as an insulator leaves the electrodes and enters the permittivity map
from ionsim.device import device_gas_flow
dev = from_drawing(load_dxf(os.path.join(GEO, "ring_funnel_profile.dxf")), "axisym")
dev.apply_gradient([e.name for e in dev.electrodes if e.layer == "RINGS"], 78, 20, rf=50)
dev.apply_fixed([e.name for e in dev.electrodes if e.layer == "EXIT"], 15)
dev.get("RINGS 10").eps = 4.4
dg = dev.build(dev.default_h("Quick"))
k = dev.names().index("RINGS 10") + 1
check("insulator part", dg.media.insulator.sum() > 0 and not (dg.electrode == k).any(),
      f"{int(dg.media.insulator.sum())} insulator nodes")
dev.get("RINGS 10").eps = 0.0

# device: gas flow field and space charge together
fld = DeviceField(dev.build(dev.default_h("Quick")))
res, _ = fly_device(fld, 500, 1.0, 30, 4.0, radius=3e-3, z_start=0.5e-3, z_end=31e-3, rng=np.random.default_rng(0),
                    gas_velocity=device_gas_flow(fld, 0.05, 4.0, 300, 4e-3), current=0.2e-9, sc_iterations=1)
check("gas flow + space charge", res.transmission > 0.8 and res.sc_history[0][1] > 0,
      f"transmission {res.transmission:.0%}, space-charge peak {res.sc_history[0][1]:.2f} V")

# the app window builds
from PySide6 import QtWidgets
from ionsim.gui.app import MainWindow
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
w = MainWindow()
check("app window", w.tabs.count() >= 6, f"{w.tabs.count()} tabs")
print(f"all good in {time.time() - t0:.0f} s")
