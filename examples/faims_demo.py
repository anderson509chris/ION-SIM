"""Prototype 3: FAIMS CV spectrum and dispersion plot.

Run:  python examples/faims_demo.py        (writes PNGs 9-11 to ./out, ~30 s)
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ionsim.faims import Faims, cv_scan, cv_theory, dispersion_curves, DEFAULT_SPECIES

OUT = os.path.join(os.path.dirname(__file__), "..", "out")
os.makedirs(OUT, exist_ok=True)
f = Faims()          # 1.88 mm gap, 40 mm long, 4000 V bisinusoidal at 750 kHz, 1 atm N2, 5 m/s
sp = DEFAULT_SPECIES
print(f"peak dispersion field {f.dv / f.gap / 1e5:.1f} kV/cm = {f.ed_td():.1f} Td")

# ---- 1. alpha functions ----
fig, ax = plt.subplots(figsize=(6, 4))
en = np.linspace(0, 1.3 * f.ed_td(), 300)
for s in sp:
    ax.plot(en, 100 * s.alpha(en), label=s.name)
ax.axvline(f.ed_td(), color="k", ls=":", lw=1)
ax.set_xlabel("E/N (Td)"); ax.set_ylabel("α (%)"); ax.legend()
ax.set_title("Field dependence of mobility, K(E) = K(0)(1 + α)")
fig.tight_layout(); fig.savefig(f"{OUT}/9_faims_alpha.png", dpi=130)

# ---- 2. simulated CV spectrum vs theory ----
cvs = np.linspace(-45, 25, 71)
t = time.time()
r = cv_scan(f, sp, cvs, ions_per_cv=60, rng=np.random.default_rng(3))
print(f"CV scan: {len(sp) * len(cvs) * 60} ions in {time.time() - t:.1f} s")
fig, ax = plt.subplots(figsize=(7, 4))
for i, s in enumerate(sp):
    T = r.transmission[i]
    l, = ax.plot(cvs, 100 * T, "o-", ms=3, label=s.name)
    ax.axvline(r.cv_theory[i], color=l.get_color(), ls=":", lw=1)
    print(f"  {s.name}: theory CV {r.cv_theory[i]:.2f} V, simulated centroid {np.sum(T * cvs) / T.sum():.2f} V, "
          f"FWHM {np.ptp(cvs[T >= T.max() / 2]) + (cvs[1] - cvs[0]):.1f} V, max T {T.max():.0%}")
ax.set_xlabel("compensation voltage (V)"); ax.set_ylabel("transmission (%)"); ax.legend()
ax.set_title("FAIMS CV spectrum, DV 4000 V (dotted: theory)")
fig.tight_layout(); fig.savefig(f"{OUT}/10_faims_cv.png", dpi=130)

# ---- 3. dispersion plot ----
dvs = np.linspace(0, 6000, 61)
cv = dispersion_curves(f, sp, dvs)
fig, ax = plt.subplots(figsize=(6, 4))
for i, s in enumerate(sp):
    ax.plot(dvs, cv[i], label=s.name)
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("dispersion voltage (V)"); ax.set_ylabel("compensation voltage (V)"); ax.legend()
ax.set_title("Dispersion plot")
fig.tight_layout(); fig.savefig(f"{OUT}/11_faims_dispersion.png", dpi=130)
print("plots written to", os.path.abspath(OUT))
