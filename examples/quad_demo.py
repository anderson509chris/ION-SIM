"""Prototype 1: quadrupole mass filter.

Run:  python examples/quad_demo.py        (writes PNGs to ./out)
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ionsim import quadrupole_rods, solve_basis, GridField2D, IdealQuadField, rf_dc, make_beam, Simulation, mathieu

# ---- default instrument parameters (change freely) ----
R0 = 4.0e-3          # field radius, m
FREQ = 1.0e6         # RF frequency, Hz
MASS = 500.0         # ion m/z (singly charged), amu
LENGTH = 0.20        # rod length, m
ENERGY = 5.0         # axial ion energy, eV
STEPS_PER_RF = 40

OUT = os.path.join(os.path.dirname(__file__), "..", "out")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(1)

# ---- 1. geometry + Laplace solve ----
t = time.time()
grid = quadrupole_rods(R0, rod_ratio=1.1487, dx=R0 / 80)
basis = solve_basis(grid)
print(f"grid {grid.n}x{grid.n}, Laplace solved in {time.time() - t:.1f}s")


def quad_field(U, V):
    """Rods A at +(U + V cos wt), rods B at -(U + V cos wt)."""
    phi0 = rf_dc(U, V, FREQ)
    return GridField2D(grid, basis, {"A": phi0, "B": lambda tt: -phi0(tt)})


# field check: grid solution vs ideal hyperbolic field near the axis
static = GridField2D(grid, basis, {"A": 1.0, "B": -1.0})
probe = np.column_stack([np.linspace(0, 0.8 * R0, 9), np.zeros(9), np.zeros(9)])
Eg = static.efield(0, probe)[:, 0]
Ei = IdealQuadField(R0, 1.0).efield(0, probe)[:, 0]
print("Ex(grid)/Ex(ideal) along x at r/r0 = 0.1..0.8:", np.round(Eg[1:] / Ei[1:], 4))
# staircased rods give a slightly weaker quadrupole term; express it as an effective r0
R0_EFF = R0 / np.sqrt(Eg[1] / Ei[1])
print(f"effective r0 from grid field: {R0_EFF * 1e3:.4f} mm (nominal {R0 * 1e3} mm)")

fig, ax = plt.subplots(figsize=(6, 5))
pot = static.potential(0)
m = grid.electrode > 0
cs = ax.contourf(grid.X * 1e3, grid.Y * 1e3, np.ma.masked_where(m, pot), 41, cmap="RdBu_r")
ax.contourf(grid.X * 1e3, grid.Y * 1e3, m, levels=[0.5, 1.5], colors="0.4")
ax.set_aspect("equal"); ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.set_title("Quadrupole potential, rods A=+1 V, B=-1 V")
fig.colorbar(cs, label="V"); fig.tight_layout(); fig.savefig(f"{OUT}/1_potential.png", dpi=130)

# ---- 2. trajectories at three operating points ----
dt = 1 / FREQ / STEPS_PER_RF
v_axial = np.sqrt(2 * ENERGY * 1.602e-19 / (MASS * 1.6605e-27))
t_max = 3 * LENGTH / v_axial
points = {"stable (a=0.20, q=0.70)": (0.20, 0.70),
          "unstable in y (a=0.26, q=0.70)": (0.26, 0.70),
          "unstable in x (a=0, q=0.95)": (0.0, 0.95)}
fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
for ax, (label, (a, q)) in zip(axes, points.items()):
    U, V = mathieu.voltages_for(a, q, MASS, R0_EFF, FREQ)
    ions = make_beam(12, MASS, energy_ev=ENERGY, radius=0.3e-3, birth_window=1 / FREQ, rng=rng)
    res = Simulation([quad_field(U, V)], z_end=LENGTH).run(ions, dt, t_max, record_every=2, rng=rng)
    for k in range(ions.n):
        ax.plot(res.traj[:, k, 2] * 1e3, res.traj[:, k, 0] * 1e3, lw=0.7, color="C0")
        ax.plot(res.traj[:, k, 2] * 1e3, res.traj[:, k, 1] * 1e3, lw=0.7, color="C3")
    ax.axhline(R0 * 1e3, color="k", lw=0.5); ax.axhline(-R0 * 1e3, color="k", lw=0.5)
    ax.set_ylim(-1.3 * R0 * 1e3, 1.3 * R0 * 1e3)
    ax.set_title(f"{label}: U={U:.1f} V, V={V:.1f} V, transmission {res.transmission:.0%}", fontsize=9)
    ax.set_ylabel("x (blue), y (red) mm")
    print(f"{label:34s} U={U:7.2f} V  V={V:7.2f} V  stable={bool(mathieu.is_stable(a, q))}  T={res.transmission:.0%}")
axes[-1].set_xlabel("z (mm)")
fig.tight_layout(); fig.savefig(f"{OUT}/2_trajectories.png", dpi=130)

# ---- 3. mass scan along a fixed U/V line ----
A_OVER_Q = 0.332                       # scan-line slope (apex is 0.2370/0.7060 = 0.3357)
U, V = mathieu.voltages_for(A_OVER_Q * 0.706, 0.706, MASS, R0_EFF, FREQ)
masses = np.linspace(MASS * 0.985, MASS * 1.015, 61)
per = 150
ions = make_beam(per * len(masses), np.repeat(masses, per), energy_ev=ENERGY, radius=0.3e-3,
                 half_angle_deg=0.5, birth_window=1 / FREQ, rng=rng)
t = time.time()
res = Simulation([quad_field(U, V)], z_end=LENGTH).run(ions, dt, t_max * 1.05, rng=rng)
T = (res.status == 1).reshape(len(masses), per).mean(axis=1)
a_m, q_m = mathieu.aq(U, V, masses, R0_EFF, FREQ)
pred = mathieu.is_stable(a_m, q_m)
print(f"mass scan: {ions.n} ions in {time.time() - t:.1f}s; "
      f"Mathieu-stable band {masses[pred].min():.1f}-{masses[pred].max():.1f} amu" if pred.any() else "no stable mass")
half = T >= T.max() / 2
print(f"simulated peak: max T {T.max():.0%}, FWHM {masses[half].min():.1f}-{masses[half].max():.1f} amu")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(masses, T * 100, "o-", ms=3, label="simulated (grid field, round rods)")
ax.fill_between(masses, 0, pred * 100, color="C1", alpha=0.2, step="mid", label="Mathieu-stable (ideal)")
ax.set_xlabel("m/z"); ax.set_ylabel("transmission (%)")
ax.set_title(f"Mass scan: U={U:.1f} V, V={V:.1f} V, a/q={A_OVER_Q}")
ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/3_mass_scan.png", dpi=130)

# ---- 4. stability diagram ----
Q, A, S = mathieu.stability_map(1.0, 0.3)
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.contourf(Q, A, S, levels=[0.5, 1.5], colors=["#9ecae1"])
ax.plot([0, 1], [0, A_OVER_Q], "k--", lw=1, label=f"scan line a/q={A_OVER_Q}")
ax.scatter(q_m, a_m, s=4, c=T, cmap="viridis", label="scanned masses (colour = T)")
for label, (a, q) in points.items():
    ax.plot(q, a, "o", label=label)
ax.set_xlabel("q"); ax.set_ylabel("a"); ax.set_xlim(0, 1); ax.set_ylim(0, 0.3)
ax.set_title("Mathieu stability diagram (first region)")
ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{OUT}/4_stability.png", dpi=130)
print("plots written to", os.path.abspath(OUT))
