# ionsim — a SIMION-style ion simulator in Python

* Prototype 1: quadrupole mass filter (four round rods, RF + DC).
* Prototype 2: TWAVE traveling-wave (TW) ion mobility separation.

## Desktop app

```
pip install -r requirements.txt
python ion_sim_app.py             # launches the Ion Sim window
```

Two tabs, Quadrupole mass filter and TWAVE traveling wave. Set the parameters on the left and press Run. Runs happen in the background with a progress bar and a Stop button. Plots can be zoomed and saved, File > Export writes the results as CSV.

**Standalone app (no Python needed to run it):**
* Windows: double-click `build_windows.bat`, then run `dist\IonSim\IonSim.exe`.
* macOS / Linux: `./build.sh`, then open `dist/`.
* GitHub: once pushed, `.github/workflows/build.yml` builds Windows, macOS and Linux versions on every push, which you download from the Actions tab.

PyInstaller can only build for the operating system it runs on, which is why there is one build per OS.

## Scripts

```
python examples/quad_demo.py      # ~1 min, writes PNGs 1-4 to out/
python examples/twave_demo.py      # a few min, writes PNGs 5-8 to out/
```

## How it works (same idea as SIMION)

| Step | Module | What it does |
|---|---|---|
| Geometry | `ionsim/geometry.py` | Paints electrodes onto a uniform 2D grid (x,y cross-section). `quadrupole_rods(r0, rod_ratio=1.1487)` builds the four rods; outer edge is a grounded can. |
| Field solve | `ionsim/solver.py` | Sparse direct Laplace solve, once per electrode at 1 V ("basis potentials"). |
| Fields | `ionsim/fields.py` | Superposes basis potentials with time-dependent voltages (SIMION's "fast adjust"), bilinear field interpolation. `IdealQuadField` is the analytic hyperbolic field for checks. |
| Flight | `ionsim/integrator.py` | Vectorised RK4 over thousands of ions, electrode hit detection, staggered birth times to sample RF phase. |
| Theory | `ionsim/mathieu.py` | a/q from U, V, m, r0, f, and Floquet-based stability test (no lookup tables). |

Rods A (on x) sit at +(U + V cos Ωt), rods B (on y) at −(U + V cos Ωt), V zero-to-peak, so
a = 8eU/(m r0² Ω²), q = 4eV/(m r0² Ω²).

## Default parameters (edit the top of `examples/quad_demo.py`)

r0 = 4.0 mm, rod radius = 1.1487 r0, f = 1.0 MHz, m/z 500, rod length 200 mm, 5 eV axial energy, grid spacing r0/80, 40 RK4 steps per RF cycle.

## Validation results (current run)

* Grid field vs ideal hyperbolic: 0.99 near axis (rod staircasing); expressed as effective r0 = 4.019 mm and used for a/q.
* Operating points: (a=0.20, q=0.70) transmits 100%; (0.26, 0.70) and (0, 0.95) transmit 0%, matching Mathieu.
* Mass scan at a/q = 0.332: Mathieu-stable band 499–506; simulated FWHM 498–505 with soft edges from the finite number of RF cycles (~145), as expected for a real filter.

## Limits of prototype 1

* 2D field: no fringe fields at the rod entrance/exit (3D grid is the obvious next field upgrade).
* No space charge, no gas.
* Pure NumPy; Numba on `efield`/RK4 would give a large speed-up when ion counts grow.

## Next: drag-field quadrupole (collision cell)

The integrator already has the three hooks needed:

1. **Axial drag field** → a new `FieldSource` that adds E_z(z) (from segmented rods, tilted/tapered auxiliary electrodes, or a resistive coating). For segmented rods, solve an axial 1D/3D problem and add a DC ramp per segment.
2. **Gas collisions** → a `hooks` entry: per step, collision probability P = 1 − exp(−n σ v_rel dt) with a hard-sphere (or Langevin) cross section, and a centre-of-mass random scatter against a thermal gas atom (He/N₂/Ar at a set pressure and temperature).
3. **Optional viscous drag shortcut** → a `forces` entry with mobility K: a = (q/m)E − v/τ, fast for high pressure.

Outputs to add then: transit time distributions, energy thermalisation, CID energy, and transmission vs pressure.

## TWAVE traveling-wave separation (`ionsim/twave.py`)

* **Boards and electrodes:** two boards `gap` apart, TW electrodes at `pitch` with `width`, 8 per wavelength, 4 on / 4 off, stepped one electrode every pitch/speed. The field between boards comes from an exact Fourier (Laplace) solution of the periodic electrode pattern, with a linear ramp across the insulating gaps.
* **Ion motion:** 4 Torr is the mobility regime, so ions follow dz = K·E dt + √(2D dt)·N(0,1), with K from K0, pressure and temperature, and D from the Einstein relation.
* **Validation:** for an ideal sine wave the mean velocity must follow theory, s for surfing (c ≥ s) and s − √(s² − c²) for rolling over, where c = K·Emax.
* **Demo:** two ions with K0 = 1.00 and 1.03 cm²/Vs over 1 m at 30 V and 200 m/s in N2, plus resolution against wave speed.

Defaults: pitch 0.508 mm, width 0.432 mm, gap 2.75 mm, 30 V, 200 m/s, 4 Torr N2, 300 K.

Not modelled yet: the RF guard confinement across the device (ions ride the midplane), turns, space charge, high-field mobility (E/N dependence), and real ion injection or trapping. A 13 m serpentine is just a longer `PATH`, and the run time grows with it.
