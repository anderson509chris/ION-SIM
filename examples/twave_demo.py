"""Prototype 2: TWAVE traveling-wave ion mobility separation.

Run:  python examples/twave_demo.py        (writes PNGs to ./out, ~2-3 min)
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ionsim.twave import TWBoards, mobility, fly_mobility, mean_velocity_sine, fwhm

# ---- default TWAVE parameters (change freely) ----
PRESSURE = 4.0           # Torr N2
TEMP = 300.0             # K
AMPLITUDE = 30.0         # TW amplitude, V
SPEED = 200.0            # TW speed, m/s
PATH = 1.0               # separation length, m (real TWAVE serpentines are ~13 m)
K0_PAIR = (1.00, 1.03)   # reduced mobilities of the two ions, cm^2/Vs (3% apart)
N_IONS = 1000            # per species
DT = 1e-7                # s

OUT = os.path.join(os.path.dirname(__file__), "..", "out")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(2)
tw = TWBoards(amplitude=AMPLITUDE, speed=SPEED)
print(f"wavelength {tw.wavelength * 1e3:.3f} mm, peak midplane Ez {tw.e_max:.0f} V/m, "
      f"electrode step every {tw.step_time * 1e6:.2f} us")

# ---- 1. the wave itself ----
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
z = np.linspace(0, 2 * tw.wavelength, 2000)
a1.plot(z * 1e3, tw.board_potential(z) * AMPLITUDE)
a1.set_ylabel("board potential (V)")
for k, c in zip(range(3), ["C0", "C1", "C2"]):
    a2.plot(z * 1e3, tw.ez(k * tw.step_time, z), color=c, label=f"step {k}")
a2.set_ylabel("midplane Ez (V/m)"); a2.set_xlabel("z (mm)"); a2.legend(fontsize=8)
a1.set_title("TW electrodes: 4 on / 4 off, stepping one electrode at a time")
fig.tight_layout(); fig.savefig(f"{OUT}/5_tw_field.png", dpi=130)

# ---- 2. validation: mean ion velocity vs analytic TW theory ----
L_check = 0.1
c_over_s, v_sine, v_elec = [], [], []
for K0 in np.linspace(0.4, 1.6, 9):
    K = mobility(K0, PRESSURE, TEMP)
    z0 = rng.random(200) * tw.wavelength
    for wf, store in [("sine", v_sine), ("electrodes", v_elec)]:
        w = TWBoards(amplitude=AMPLITUDE, speed=SPEED, waveform=wf)
        r = fly_mobility(w, K, z0, L_check, DT, 3 * L_check / mean_velocity_sine(K * tw.e_max, SPEED), TEMP, rng=rng)
        store.append(np.nanmean((L_check - z0) / r.arrival))
    c_over_s.append(K * tw.e_max / SPEED)
c_over_s = np.array(c_over_s)
cs = np.linspace(0, 1.7, 200)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(cs, mean_velocity_sine(cs * SPEED, SPEED) / SPEED, "k-", label="theory (sine wave)")
ax.plot(c_over_s, np.array(v_sine) / SPEED, "o", label="sim, sine wave")
ax.plot(c_over_s, np.array(v_elec) / SPEED, "s", label="sim, TWAVE electrodes")
ax.set_xlabel("c / s   (K·Emax / wave speed)"); ax.set_ylabel("mean ion velocity / wave speed")
ax.set_title("Surfing vs rolling over"); ax.legend(); fig.tight_layout()
fig.savefig(f"{OUT}/6_tw_velocity.png", dpi=130)
err = np.max(np.abs(np.array(v_sine) - mean_velocity_sine(c_over_s * SPEED, SPEED)) / SPEED)
print(f"sine-wave check: max |sim - theory| = {err:.3f} of wave speed")


def separate(speed, n, path, record=False):
    w = TWBoards(amplitude=AMPLITUDE, speed=speed)
    K = np.repeat([mobility(k, PRESSURE, TEMP) for k in K0_PAIR], n)
    species = np.repeat([0, 1], n)
    z0 = rng.random(2 * n) * w.wavelength          # inject one wavelength's worth of ions
    v_slow = mean_velocity_sine(K.min() * w.e_max, speed)
    r = fly_mobility(w, K, z0, path, DT, 3 * path / max(v_slow, 1.0), TEMP, species=species, rng=rng)
    t = [r.arrival[species == s] for s in (0, 1)]
    w0, w1 = fwhm(t[0]), fwhm(t[1])
    m0, m1 = np.nanmedian(t[0]), np.nanmedian(t[1])
    rpp = 1.18 * abs(m0 - m1) / (w0 + w1)
    return t, (m0, m1), (w0, w1), rpp


# ---- 3. arrival-time distributions for the pair ----
t0 = time.time()
t, m, w, rpp = separate(SPEED, N_IONS, PATH)
print(f"separation over {PATH} m in {time.time() - t0:.1f}s: "
      f"t = {m[0] * 1e3:.2f} / {m[1] * 1e3:.2f} ms, "
      f"resolving power {m[0] / w[0]:.0f} / {m[1] / w[1]:.0f}, two-peak resolution Rpp = {rpp:.2f}")
fig, ax = plt.subplots(figsize=(7, 4))
bins = np.linspace(min(np.nanmin(x) for x in t), max(np.nanmax(x) for x in t), 120) * 1e3
for x, k0 in zip(t, K0_PAIR):
    ax.hist(x * 1e3, bins=bins, alpha=0.6, label=f"K0 = {k0} cm²/Vs")
ax.set_xlabel("arrival time (ms)"); ax.set_ylabel("ions")
ax.set_title(f"TWAVE TW separation, {PATH} m, {AMPLITUDE:.0f} V, {SPEED:.0f} m/s, {PRESSURE} Torr N2: Rpp = {rpp:.2f}",
             fontsize=9)
ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/7_twave_atd.png", dpi=130)

# ---- 4. resolution vs wave speed (the usual TWAVE tuning curve) ----
speeds = np.array([130, 150, 170, 190, 210, 240, 270, 300])
rpps = []
for s in speeds:
    rpps.append(separate(s, 300, 0.5)[3])
    print(f"  speed {s:4d} m/s  Rpp(0.5 m) = {rpps[-1]:.2f}")
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(speeds, rpps, "o-")
ax.set_xlabel("TW speed (m/s)"); ax.set_ylabel("two-peak resolution Rpp (0.5 m path)")
ax.set_title(f"Resolution vs wave speed, {AMPLITUDE:.0f} V amplitude"); fig.tight_layout()
fig.savefig(f"{OUT}/8_rpp_vs_speed.png", dpi=130)
print("plots written to", os.path.abspath(OUT))
