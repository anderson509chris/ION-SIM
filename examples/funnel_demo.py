"""Prototype 5: ion funnels - stacked rings, planar boards, and imported CAD geometry.

Run:  python examples/funnel_demo.py        (writes PNGs 15-20 to ./out, ~6 min)
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ionsim.funnel import RingFunnel, PlanarFunnel, CadFunnel, FunnelElectrics, run_funnel, gas_flow
from ionsim.cad import write_ring_stack_step, load_bodies

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "out")
GEO = os.path.join(HERE, "geometry")
os.makedirs(OUT, exist_ok=True); os.makedirs(GEO, exist_ok=True)
MM = 1e-3
P_TORR = 2.0
rng = np.random.default_rng(7)


def outline(ax, sec):
    ax.contourf(sec["z"] / MM, sec["coords"] / MM, sec["mask"].astype(float), levels=[0.5, 1.5], colors="0.6")


def paths(ax, res, c):
    for i in range(res.traj.shape[1]):
        ok = res.status[i] == 1
        ax.plot(res.traj[:, i, 2] / MM, res.traj[:, i, c] / MM, lw=0.5, color="tab:blue" if ok else "tab:red")


# ---- 1. the standard ring funnel ----
ring, el = RingFunnel(), FunnelElectrics()
t = time.time(); grid = ring.build(); solver = grid.solver(); fld = ring.field(el, grid, solver)
print(f"ring funnel: {ring.n_rings} rings, {grid.nodes / 1e6:.2f} M nodes, solved in {time.time() - t:.1f} s")
res, _ = run_funnel(fld, ring.z_start, ring.z_end, 500, 1.0, 300, P_TORR, start_radius=8 * MM, record=40, rng=rng)
tr = res.status == 1
print(f"  m/z 500: transmitted {res.transmission:.0%}, transit {np.nanmean(res.t_end[tr]) * 1e3:.2f} ms, "
      f"exit r_rms {np.sqrt(np.mean(np.sum(res.final_pos[tr, :2] ** 2, 1))) / MM:.3f} mm")
fig, ax = plt.subplots(figsize=(9, 3.6))
outline(ax, fld.sections()["x"]); paths(ax, res, 0)
ax.set_xlabel("z (mm)"); ax.set_ylabel("x (mm)")
ax.set_title(f"Ring funnel: 100 rings, ID 25.4 → 2.5 mm, {2 * el.rf:.0f} Vpp at {el.freq / 1e6:.0f} MHz, "
             f"{el.gradient:.0f} V/cm, {P_TORR} Torr, m/z 500: {res.transmission:.0%} transmitted", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/15_funnel_trajectories.png", dpi=130)

# ---- 2. transmission vs m/z for three RF amplitudes ----
mz = np.geomspace(10, 5000, 16)
K0 = 1.0 * (mz / 500) ** (-2 / 3)          # compact ions: cross section ~ m^(2/3)
fig, ax = plt.subplots(figsize=(7, 4))
rows = []
for rf in (15, 50, 150):
    e = FunnelElectrics(rf=rf)
    r, mzi = run_funnel(ring.field(e, grid, solver), ring.z_start, ring.z_end, mz, K0, 60, P_TORR,
                        start_radius=8 * MM, rng=rng)
    T = np.array([np.mean(r.status[mzi == m] == 1) for m in mz])
    ax.semilogx(mz, 100 * T, "o-", ms=4, label=f"{2 * rf} Vpp")
    rows.append(T)
    print(f"  {2 * rf:4d} Vpp: " + " ".join(f"{m:.0f}:{t:.0%}" for m, t in zip(mz, T)))
ax.set_xlabel("m/z"); ax.set_ylabel("transmission (%)"); ax.set_ylim(-3, 103); ax.legend()
ax.set_title(f"Ring funnel transmission, {el.gradient:.0f} V/cm, {P_TORR} Torr", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/16_funnel_transmission.png", dpi=130)

# ---- 3. planar two-board funnel ----
pf = PlanarFunnel()
t = time.time(); pfld = pf.field(el)
print(f"planar funnel: {pf.n_strips} strips per board, {pfld.grid.nodes / 1e6:.2f} M nodes, solved in {time.time() - t:.1f} s")
res, _ = run_funnel(pfld, pf.z_start, pf.z_end, 500, 1.0, 300, P_TORR, start_radius=1.5 * MM, record=40, rng=rng)
tr = res.status == 1
print(f"  m/z 500: transmitted {res.transmission:.0%}, exit beam x rms {res.final_pos[tr, 0].std() / MM:.2f} mm, "
      f"y rms {res.final_pos[tr, 1].std() / MM:.2f} mm")
fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
sec = pfld.sections()
for ax, key, c in ((axes[0], "x", 0), (axes[1], "y", 1)):
    outline(ax, sec[key]); paths(ax, res, c); ax.set_ylabel(f"{key} (mm)")
axes[0].set_title(f"Planar funnel: boards {pf.gap / MM:.0f} mm apart, strips {pf.width_in / MM:.0f} → "
                  f"{pf.width_out / MM:.0f} mm, guards +{el.guard_offset:g} V, m/z 500: {res.transmission:.0%} transmitted", fontsize=9)
axes[1].set_xlabel("z (mm)")
fig.tight_layout(); fig.savefig(f"{OUT}/17_funnel_planar.png", dpi=130)

# ---- 4. CAD import check: a ring stack written as STEP, imported, solved in 3D ----
short = RingFunnel(n_rings=30, id_in=12 * MM, id_out=2.5 * MM, n_straight=4, od=20 * MM, copper_width=0.0)   # metal rings, as in the STEP file
step = os.path.join(GEO, "ring_funnel_30.step")
if not os.path.exists(step):                 # shipped with the repo; only rebuilt if deleted
    write_ring_stack_step(step, short.rings())
bodies = load_bodies(step)
cad = CadFunnel(bodies, h=0.2 * MM)
t = time.time(); cfld = cad.field(cad.funnel_pattern(el, last_is_exit=True), el.freq)
print(f"STEP import: {len(bodies)} solids, 3D grid {cfld.grid.nodes / 1e6:.2f} M nodes, solved in {time.time() - t:.1f} s")
afld = short.field(el)
z = np.linspace(1 * MM, short.z_plate - 0.3 * MM, 600)
axis = np.column_stack([0 * z, 0 * z, z])
off = np.column_stack([0 * z + 0.6 * np.interp(z, short.ring_z(), short.ring_ids() / 2), 0 * z, z])
tq = 0.25 / el.freq
fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
for f_, lab, ls in ((afld, "axisymmetric (exact geometry)", "-"), (cfld, "3D from STEP file", "--")):
    a1.plot(z / MM, f_.efield(tq, axis)[:, 2] / 100, ls, label=lab)
    rf = f_.efield(0, off) - f_.efield(tq, off)
    a2.plot(z / MM, np.hypot(rf[:, 0], rf[:, 1]) / 100, ls, label=lab)
a1.set_ylabel("axial DC field (V/cm)"); a1.legend(fontsize=8)
a2.set_ylabel("radial RF field (V/cm)\nat 60% of the bore"); a2.set_xlabel("z (mm)")
mzs = np.array([50, 200, 500, 2000.0]); K = (mzs / 500) ** (-2 / 3)
txt = []
for f_, lab in ((afld, "axisym"), (cfld, "STEP 3D")):
    r, mzi = run_funnel(f_, short.z_start, short.z_end, mzs, K, 150, P_TORR, start_radius=3.5 * MM, rng=np.random.default_rng(5))
    trn = r.status == 1
    txt.append(f"{lab} " + "/".join(f"{100 * np.mean(r.status[mzi == m] == 1):.0f}" for m in mzs)
               + f"%, {np.nanmean(r.t_end[trn]) * 1e3:.3f} ms")
    print("  " + txt[-1])
a1.set_title("30-ring funnel: imported STEP (3D, 0.2 mm grid) vs exact axisymmetric solve\n"
             + "transmission at m/z 50/200/500/2000 and mean transit:  " + ";  ".join(txt), fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/18_funnel_cad_check.png", dpi=130)

# ---- 5. gas flow field through the ring funnel ----
SLM = 2.0
t = time.time(); flow = gas_flow(fld, SLM, P_TORR, inlet_radius=ring.id_in / 2)
print(f"gas flow: {SLM} slm, solved in {time.time() - t:.1f} s, {flow.exit_fraction:.2%} through the exit")
res_f, _ = run_funnel(fld, ring.z_start, ring.z_end, 500, 1.0, 300, P_TORR, start_radius=8 * MM, gas_velocity=flow,
                      record=30, rng=np.random.default_rng(3))
res_0, _ = run_funnel(fld, ring.z_start, ring.z_end, 500, 1.0, 300, P_TORR, start_radius=8 * MM, rng=np.random.default_rng(3))
tt = [np.nanmean(r.t_end[r.status == 1]) * 1e3 for r in (res_0, res_f)]
print(f"  m/z 500: still gas {res_0.transmission:.0%} in {tt[0]:.3f} ms, with flow {res_f.transmission:.0%} in {tt[1]:.3f} ms")
sec = flow.section()
fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
Z, A = sec["z"] / MM, sec["coords"] / MM
speed = np.ma.masked_where(sec["mask"], np.hypot(sec["ut"], sec["uz"]))
im = a1.imshow(speed, origin="lower", aspect="auto", extent=[Z[0], Z[-1], A[0], A[-1]], vmax=np.percentile(speed.compressed(), 99))
fig.colorbar(im, ax=a1, label="gas speed (m/s)", extend="max")
a1.streamplot(Z, A, np.where(sec["mask"], 0, sec["uz"]), np.where(sec["mask"], 0, sec["ut"]), color="w", density=1.3, linewidth=0.5, arrowsize=0.6)
s0 = fld.sections()["x"]
a1.contourf(Z, A, s0["mask"].astype(float), levels=[0.5, 1.5], colors="tab:orange")
a1.contourf(Z, A, (s0["insulator"] & ~s0["mask"]).astype(float), levels=[0.5, 1.5], colors=[(0.80, 0.68, 0.40)])
a1.set_ylabel("x (mm)")
a1.set_title(f"Gas flow: {SLM} slm into the 25.4 mm entrance at {P_TORR} Torr, viscous, pumped out between the PCB rings; "
             f"{flow.exit_fraction:.1%} reaches the exit", fontsize=9)
zz = np.linspace(0, ring.z_end, 400)
a2.plot(zz / MM, flow.velocity(np.column_stack([0 * zz, 0 * zz, zz]))[:, 2])
a2.set_ylabel("axial gas speed\non axis (m/s)"); a2.set_xlabel("z (mm)")
a2.set_title(f"m/z 500 transit: {tt[0]:.2f} ms in still gas, {tt[1]:.2f} ms with the flow", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/19_funnel_gas_flow.png", dpi=130)

# ---- 6. space charge: transmission against beam current ----
currents = [0.3, 1, 3, 10, 30]
rows = []
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
for I in currents:
    t = time.time()
    r, _ = run_funnel(fld, ring.z_start, ring.z_end, 500, 1.0, 150, P_TORR, start_radius=8 * MM, rng=np.random.default_rng(4),
                      current=I * 1e-9)
    trn = r.status == 1
    rr = np.sqrt(np.mean(np.sum(r.final_pos[trn, :2] ** 2, 1))) / MM if trn.any() else np.nan
    rows.append((r.transmission, rr, r.sc_history[-1][1], r.sc_converged))
    a2.plot(r.space_charge.gmap.origin[1] / MM + r.space_charge.gmap.h / MM * np.arange(r.space_charge.phi.shape[1]),
            r.space_charge.phi[0], "-" if r.sc_converged else ":", label=f"{I:g} nA" + ("" if r.sc_converged else " (piles up)"))
    print(f"  {I:5g} nA: transmitted {r.transmission:.0%}, exit r_rms {rr:.3f} mm, peak {r.sc_history[-1][1]:.2f} V"
          f"{'' if r.sc_converged else ' (not settled)'} ({time.time() - t:.0f} s)")
T = np.array([row[0] for row in rows])
a1.semilogx(currents, 100 * T, "o-", label="transmitted (%)")
a1.set_xlabel("ion current (nA)"); a1.set_ylabel("m/z 500 transmitted (%)"); a1.set_ylim(-3, 103)
ax_r = a1.twinx(); ax_r.semilogx(currents, [row[1] for row in rows], "s--", color="tab:red")
ax_r.set_ylabel("exit beam r_rms (mm)", color="tab:red")
a1.set_title("Space charge: the beam's own field against the funnel", fontsize=10)
settled = [row[2] for row in rows if row[3]]
a2.set_ylim(0, 1.4 * max(settled))
a2.set_xlabel("z (mm)"); a2.set_ylabel("space-charge potential on axis (V)"); a2.legend(fontsize=8)
a2.set_title(f"100 Vpp, {el.gradient:.0f} V/cm, {P_TORR} Torr", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/20_funnel_space_charge.png", dpi=130)
print("plots written to", os.path.abspath(OUT))
