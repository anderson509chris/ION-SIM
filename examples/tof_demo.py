"""Orthogonal-acceleration TOF: resolution of a two-stage reflectron and what limits it.

Run:  python examples/tof_demo.py        (seconds, writes PNGs 21-23 to ./out)
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ionsim import tof as T
from ionsim.ions import IonSpecies

OUT = os.path.join(os.path.dirname(__file__), "..", "out")
os.makedirs(OUT, exist_ok=True)

beam = T.Beam()                                   # 10 eV beam, 0.5 mm, 0.5 deg, 20 mm pusher window
tof, note = T.suggest_reflectron(T.OATof(), beam)  # 6 kV, 5 + 15 mm source, 600 mm drift, two-stage reflectron
print(note)
ions = [IonSpecies("m/z 500", 500, 1, 1, 3000), IonSpecies("m/z 1000", 1000, 1, 1, 3000),
        IonSpecies("m/z 1001", 1001, 1, 1, 3000)]
res = T.fly(tof, beam, ions, rng=np.random.default_rng(1))
for s in res.stats():
    print(f"{s['name']}: {s['t_mean'] * 1e6:.4f} us, FWHM {s['fwhm'] * 1e9:.2f} ns, R = {s['resolution']:.0f}")

# 21: spectrum around m/z 1000 / 1001
tc, c, mz = res.spectrum()
m = (mz > 999.5) & (mz < 1001.5)
plt.figure(figsize=(7, 4))
plt.plot(mz[m], c[m], "k", lw=0.8)
plt.xlabel("m/z"); plt.ylabel("counts / 0.25 ns"); plt.title("OA-TOF, two-stage reflectron: m/z 1000 and 1001")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "21_tof_spectrum.png"), dpi=130)

# 22: resolution vs reflectron back plate
vals = tof.v_r2 + np.linspace(-150, 150, 31)
R, _ = T.scan(tof, beam, ions, "v_r2", vals)
plt.figure(figsize=(7, 4))
plt.plot(vals, R, "o-")
plt.xlabel("reflectron back plate (V)"); plt.ylabel("resolution (m/z 500)")
plt.title("Resolution vs reflectron voltage")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "22_tof_reflectron_scan.png"), dpi=130)

# 23: linear TOF: best detector distance = Wiley-McLaren space focus
lin = T.OATof(reflectron="none", pulse=T.Pulse(rise=0))
b = T.Beam(divergence_deg=0, energy_spread_ev=0)
D = T.wiley_mclaren_focus(lin, b)
Ls = np.linspace(0.5 * D, 1.5 * D, 31)
R, _ = T.scan(lin, b, ions, "drift_length", Ls)
plt.figure(figsize=(7, 4))
plt.plot(Ls * 1e3, R, "o-"); plt.axvline(D * 1e3, color="r", ls=":", label=f"Wiley-McLaren {D * 1e3:.1f} mm")
plt.xlabel("detector distance past the drift grid (mm)"); plt.ylabel("resolution (beam width only)")
plt.legend(); plt.title("Linear TOF: space focus")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "23_tof_space_focus.png"), dpi=130)
print("wrote 21-23 to", OUT)
