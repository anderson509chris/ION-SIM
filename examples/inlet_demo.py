"""Prototype 4: ESI ions through the inlet capillary and free jet.

Run:  python examples/inlet_demo.py        (writes PNGs 12-14 to ./out, a few seconds)
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ionsim.inlet import Inlet, InletIon, fly_inlet, DEFAULT_IONS, TORR

OUT = os.path.join(os.path.dirname(__file__), "..", "out")
os.makedirs(OUT, exist_ok=True)
inl = Inlet()        # 0.5 mm x 60 mm capillary at 275 C, 1.5 Torr, skimmer 3 mm away at -30 V
print(f"gas: {inl.flow_slm:.2f} standard L/min, Re {inl.reynolds:.0f}, entrance Mach {inl.cap['M'][0]:.2f}, "
      f"exit {inl.cap['P'][-1] / TORR:.0f} Torr / {inl.cap['u'][-1]:.0f} m/s, Mach disk {inl.x_mach_disk * 1e3:.2f} mm")

ax_ = inl.axis
x = ax_["x"] * 1e3
fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
a1.semilogy(x, ax_["P"] / TORR); a1.set_ylabel("P (Torr)")
a2.plot(x, ax_["u"]); a2.set_ylabel("gas speed (m/s)")
a3.plot(x, ax_["T"]); a3.set_ylabel("gas T (K)"); a3.set_xlabel("x (mm), 0 = capillary entrance")
a1.set_title(f"Gas: spray region → capillary (choked) → free jet, {inl.flow_slm:.2f} L/min")
fig.tight_layout(); fig.savefig(f"{OUT}/12_inlet_gas.png", dpi=130)

res = {lab: fly_inlet(Inlet(v_exit=vx), DEFAULT_IONS) for lab, vx in (("0 V", 0.0), ("30 V", 30.0))}
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, (lab, rr) in zip(axes, res.items()):
    for d in rr:
        ax.plot(d["x"] * 1e3, d["v"], label=d["ion"].name)
    ax.plot(x, ax_["u"], "k--", lw=1, label="gas")
    ax.set_xlabel("x (mm)"); ax.set_title(f"capillary − skimmer = {lab}")
axes[0].set_ylabel("speed (m/s)"); axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/13_inlet_velocity.png", dpi=130)

fig, ax = plt.subplots(figsize=(7, 4))
for (lab, rr), ls in zip(res.items(), ("--", "-")):
    for k, d in enumerate(rr):
        ax.plot(d["x"] * 1e3, d["ke"], ls, color=f"C{k}", label=f"{d['ion'].name}, {lab}")
        print(f"  {lab:>4}  {d['ion'].name:14s} v = {d['v_final']:6.0f} m/s  KE = {d['ke_final']:6.2f} eV")
ax.set_yscale("log"); ax.set_xlim(55, None)
ax.set_xlabel("x (mm)"); ax.set_ylabel("kinetic energy (eV)"); ax.legend(fontsize=7)
ax.set_title("Ion energy through the exit and free jet")
fig.tight_layout(); fig.savefig(f"{OUT}/14_inlet_energy.png", dpi=130)

# heavy-ion check: a slow-relaxing ion with no field should end near (1/2) m u_gas^2
u = Inlet(v_exit=0).jet["u"][-1]
for ion in (InletIon("m/z 100", 100, 1, 2.0), InletIon("m/z 5000", 5000, 1, 0.5)):
    d = fly_inlet(Inlet(v_exit=0), [ion])[0]
    print(f"  check {ion.name}: v/u_gas = {d['v_final'] / u:.3f}, KE {d['ke_final']:.2f} eV vs "
          f"(1/2) m u_gas^2 = {0.5 * ion.mass * u ** 2 / 1.602176634e-19:.2f} eV")
print("plots written to", os.path.abspath(OUT))
