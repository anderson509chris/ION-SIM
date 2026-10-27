# Ion Sim

**A SIMION-style ion optics and ion mobility simulator in Python, with a desktop app.**

[![build-app](https://github.com/anderson509chris/ION-SIM/actions/workflows/build.yml/badge.svg)](https://github.com/anderson509chris/ION-SIM/actions/workflows/build.yml)
![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)
![Platforms](https://img.shields.io/badge/platforms-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)

![Ion Sim, quadrupole tab](docs/screenshots/quad_trajectories.png)

Ion Sim solves electrode fields on a grid, flies thousands of ions through them, and checks the results against theory. It currently models five devices:

* **Quadrupole mass filter:** four round rods with RF + DC, a Laplace field solve, RK4 ion flight and a Mathieu stability check.
* **TWAVE traveling-wave ion mobility separation:** stepped electrode boards in 4 Torr gas, with mobility drift and diffusion.
* **FAIMS:** an asymmetric waveform across a planar gap at 1 atm, field-dependent mobility α(E/N), CV spectra and dispersion plots.
* **ESI inlet capillary:** spray region, choked capillary flow and free-jet expansion, with ion velocity and kinetic energy from the emitter to the skimmer.
* **Ion funnel:** the PNNL stacked-ring funnel, a planar two-board funnel, or **your own electrodes imported from STEP, STL or OBJ**, with RF + DC fields and ion flight through 1–10 Torr gas.

---

## Contents

* [Quick start](#quick-start)
* [The desktop app](#the-desktop-app)
* [Example results](#example-results)
* [How it works](#how-it-works)
* [Validation](#validation)
* [Limits and roadmap](#limits-and-roadmap)
* [Project layout](#project-layout)

---

## Quick start

```bash
git clone https://github.com/anderson509chris/ION-SIM.git
cd ION-SIM
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python ion_sim_app.py              # launches the Ion Sim window
```

Requires Python 3.12 or newer. Dependencies: NumPy, SciPy, Matplotlib, PySide6, and for the ion funnel PyAMG (3D field solver), trimesh (STL/OBJ) and gmsh (STEP/IGES).

### Standalone app (no Python needed)

| Platform | How |
|---|---|
| **Download** | Every push builds Windows, macOS and Linux versions. Grab the zip from the latest run on the [Actions tab](https://github.com/anderson509chris/ION-SIM/actions). |
| Windows | Double-click `build_windows.bat`, then run `dist\IonSim\IonSim.exe`. |
| macOS / Linux | `./build.sh`, then open `dist/`. |

PyInstaller can only build for the operating system it runs on, which is why there is one build per OS.

---

## The desktop app

Set the parameters on the left and press **Run**. Runs happen in the background with a progress bar and a **Stop** button. Every plot can be zoomed, panned and saved, and **File > Export results (CSV)** writes the numbers out.

### Quadrupole mass filter

Rod geometry, RF frequency, ion beam and operating point (a, q or U, V, kept in sync). It runs either a single operating point or a full mass scan. Plot tabs: Potential, Trajectories, Cross-section, Stability and Mass scan.

| Ion trajectories | Stability diagram |
|---|---|
| ![Quadrupole trajectories](docs/screenshots/quad_trajectories.png) | ![Quadrupole stability](docs/screenshots/quad_stability.png) |

### TWAVE traveling wave

Board geometry, wave amplitude, speed and shape, gas pressure and temperature, and a list of ion mobilities (K0) to separate. It runs either a single separation or a sweep over wave speed. Plot tabs: TW field, Arrival times, Ion velocity and Speed sweep.

| Arrival-time distribution | Traveling-wave field |
|---|---|
| ![TWAVE arrival times](docs/screenshots/twave_arrival_times.png) | ![TWAVE field](docs/screenshots/twave_field.png) |

### FAIMS

Planar cell (gap, length, gas speed and flow profile), waveform (bisinusoidal or rectangular, DV, frequency), gas, and an editable table of species with K0, a2 and a4. It runs either a simulated CV scan or a theoretical dispersion plot. Plot tabs: Waveform, CV spectrum, Ion paths, Alpha and Dispersion.

| CV spectrum | Alpha functions |
|---|---|
| ![FAIMS CV spectrum](docs/screenshots/faims_cv_spectrum.png) | ![FAIMS alpha](docs/screenshots/faims_alpha.png) |

### ESI inlet capillary

Spray region (emitter distance, tip radius, spray voltage, where ions leave the droplets), capillary (diameter, length, temperature), first vacuum stage (pressure, skimmer distance and voltage), and an editable ion table (m/z, z, K0). Plot tabs: Gas flow, Ion velocity, Ion energy and Effective temperature.

| Ion energy | Gas flow |
|---|---|
| ![Inlet ion energy](docs/screenshots/inlet_ion_energy.png) | ![Inlet gas flow](docs/screenshots/inlet_gas_flow.png) |

### Ion funnel

Pick the funnel type at the top: **Ring funnel** (stacked PCBs), **Planar funnel** (two boards), or **Imported CAD**. Then set the electrics (RF amplitude and frequency, DC gradient, exit plate), gas (pressure, temperature, axial gas speed) and ions (m/z, charge, K0, start radius). It runs either trajectories for one m/z or transmission against m/z. Plot tabs: Geometry, Potential (DC and RF effective potential), Trajectories, Transmission and Exit beam. While the geometry stays the same, the solver is kept, so changing only voltages re-runs in about a second.

| Ring funnel trajectories | DC and RF effective potential |
|---|---|
| ![Funnel trajectories](docs/screenshots/funnel_trajectories.png) | ![Funnel potential](docs/screenshots/funnel_potential.png) |

**Importing your own funnel (STEP, STL or OBJ):**

1. Choose **Imported CAD** and press **Load STEP / STL / OBJ...**. Every solid body in the file becomes one electrode, named from the file.
2. Say which CAD axis the ions travel along (they enter at its low end). For STL/OBJ, also give the units, because those formats don't store them.
3. The table fills with a standard funnel pattern: bodies sorted along the axis, RF alternating 0°/180°, and a DC ladder from the Electrics box. Tick **Last body is a DC exit plate** if it is. Then edit any DC, RF amplitude or phase by hand. Guards, lenses and grounded parts are just rows with RF set to 0.
4. Check the **Geometry** tab (true cross-sections through the start point), set the start point and detector plane, and pick a grid spacing about a third of your smallest electrode gap. The label shows the grid size, and the limit is 40 M nodes.

`examples/geometry/ring_funnel_30.step` is a ready-made 30-ring funnel to try.

| Imported STEP file |
|---|
| ![Imported STEP funnel](docs/screenshots/funnel_cad_geometry.png) |

<sub>To refresh these screenshots after changing the GUI, run `python docs/make_screenshots.py`.</sub>

---

## Example results

The scripts in `examples/` reproduce these plots without the GUI:

```bash
python examples/quad_demo.py       # ~1 min, writes PNGs 1-4 to out/
python examples/twave_demo.py      # a few min, writes PNGs 5-8 to out/
python examples/faims_demo.py      # ~40 s, writes PNGs 9-11 to out/
python examples/inlet_demo.py      # seconds, writes PNGs 12-14 to out/
python examples/funnel_demo.py     # ~2 min, writes PNGs 15-18 to out/ and the example STEP file
```

### Quadrupole mass filter

| | |
|---|---|
| ![Quadrupole potential](out/1_potential.png) | ![Trajectories](out/2_trajectories.png) |
| **Field solve.** Potential of the four round rods (A = +1 V, B = −1 V) inside a grounded can. | **Ion flight.** A stable point transmits 100%. Points just outside the stability region in y or x crash into the rods within a few mm. |
| ![Mass scan](out/3_mass_scan.png) | ![Stability diagram](out/4_stability.png) |
| **Mass scan.** Simulated transmission at a/q = 0.332 against the ideal Mathieu-stable band (m/z 499–506). | **Mathieu stability diagram.** The first stability region, the scan line, and the three operating points from the trajectory plot. |

### TWAVE traveling-wave separation

| | |
|---|---|
| ![TW field](out/5_tw_field.png) | ![TW velocity](out/6_tw_velocity.png) |
| **Traveling wave.** 4 electrodes on / 4 off, stepping one electrode at a time, and the resulting midplane field. | **Surfing vs rolling over.** Mean ion velocity against c/s, with simulation matching the sine-wave theory. |
| ![Arrival times](out/7_twave_atd.png) | ![Resolution vs speed](out/8_rpp_vs_speed.png) |
| **Separation.** K0 = 1.00 and 1.03 cm²/Vs over 1 m at 30 V, 200 m/s, 4 Torr N₂: two-peak resolution Rpp = 2.65. | **Tuning curve.** Two-peak resolution against wave speed over a 0.5 m path. |

### FAIMS

| | |
|---|---|
| ![Alpha](out/9_faims_alpha.png) | ![CV spectrum](out/10_faims_cv.png) |
| **Alpha functions.** Type A (mobility rises with field), type B (rises then falls) and type C (falls). Dotted line: peak dispersion field, 87 Td. | **CV spectrum.** 4000 V bisinusoidal at 750 kHz across 1.88 mm. Each simulated peak sits on its theoretical CV (dotted). |
| ![Dispersion](out/11_faims_dispersion.png) | |
| **Dispersion plot.** CV against DV for the three types. | |

### ESI inlet capillary

| | |
|---|---|
| ![Gas](out/12_inlet_gas.png) | ![Velocity](out/13_inlet_velocity.png) |
| **Gas.** Room air drawn in, Fanno flow down a 0.5 mm × 60 mm capillary at 275 °C (choked at the exit, 203 Torr), then the free jet. | **Ion velocity.** Ions ride the gas through the capillary, then reach the jet speed (left) or overshoot it with 30 V to the skimmer (right). |
| ![Energy](out/14_inlet_energy.png) | |
| **Ion energy.** Kinetic energy grows with mass in the jet, and the skimmer voltage adds to it. | |

### Ion funnel

| | |
|---|---|
| ![Ring funnel](out/15_funnel_trajectories.png) | ![Funnel transmission](out/16_funnel_transmission.png) |
| **Ring funnel.** 100 rings tapering from 25.4 to 2.5 mm, 100 Vpp at 1 MHz, 20 V/cm, 2 Torr. Ions entering across 16 mm leave in a 0.23 mm (rms) beam, 1.3 ms later. | **Transmission against m/z.** Low RF (30 Vpp) loses heavy ions onto the tapering rings (weak confinement). High RF (300 Vpp) traps mid-mass ions on axis at the small last rings, where the RF makes an axial barrier. At 100 Vpp everything from m/z 35 to 5000 gets through. |
| ![Planar funnel](out/17_funnel_planar.png) | ![CAD check](out/18_funnel_cad_check.png) |
| **Planar funnel.** Two boards 4 mm apart, RF strips shortening from 20 to 2 mm, +3 V DC guards beyond the strip ends. Top: the board face (x); bottom: across the gap (y). | **CAD import check.** A 30-ring funnel written to STEP, imported and solved in 3D, against the exact axisymmetric solve of the same rings: same fields, same transmission. |

---

## How it works

The quadrupole uses the same approach as SIMION:

| Step | Module | What it does |
|---|---|---|
| Geometry | `ionsim/geometry.py` | Paints electrodes onto a uniform 2D grid (x, y cross-section). `quadrupole_rods(r0, rod_ratio=1.1487)` builds the four rods. The outer edge is a grounded can. |
| Field solve | `ionsim/solver.py` | Sparse direct Laplace solve, once per electrode at 1 V ("basis potentials"). |
| Fields | `ionsim/fields.py` | Superposes basis potentials with time-dependent voltages (SIMION's "fast adjust") and interpolates the field bilinearly. `IdealQuadField` is the analytic hyperbolic field used for checks. |
| Flight | `ionsim/integrator.py` | Vectorised RK4 over thousands of ions, with electrode hit detection and staggered birth times to sample the RF phase. |
| Theory | `ionsim/mathieu.py` | a/q from U, V, m, r0 and f, plus a Floquet-based stability test (no lookup tables). |

Rods A (on x) sit at +(U + V cos Ωt) and rods B (on y) at −(U + V cos Ωt), with V zero-to-peak, so

a = 8eU / (m r0² Ω²),  q = 4eV / (m r0² Ω²)

### TWAVE traveling wave (`ionsim/twave.py`)

* **Boards and electrodes:** two boards `gap` apart, with TW electrodes at `pitch` and `width`, 8 per wavelength, 4 on / 4 off, stepped one electrode every pitch/speed. The field between the boards comes from an exact Fourier (Laplace) solution of the periodic electrode pattern, with a linear ramp across the insulating gaps.
* **Ion motion:** 4 Torr is the mobility regime, so ions follow dz = K·E dt + √(2D dt)·N(0,1). K comes from K0, pressure and temperature, and D from the Einstein relation.

### FAIMS (`ionsim/faims.py`)

* **Field:** uniform across the gap, E(t) = (DV·f(t) + CV)/gap. f(t) is the bisinusoidal waveform (2 sin ωt + sin(2ωt − π/2))/3, or a rectangular one with a set duty cycle, with peak +1 and zero mean.
* **Mobility:** K(E) = K(0)·(1 + a2 (E/N)² + a4 (E/N)⁴) with E/N in Td, scaled to the gas density.
* **Ion motion:** across the gap dy = K(E)·E dt + √(2D dt)·N(0,1), and along it dx = u_gas(y) dt with a parabolic or plug flow profile. Ions that touch a plate are lost.
* **Theory:** the compensation voltage is where the waveform-averaged drift ⟨K(E)E⟩ is zero, found by root search (`cv_theory`).

### ESI inlet capillary (`ionsim/inlet.py`)

Everything is 1D along the axis, from the emitter to the skimmer.

* **Spray region:** a hyperbolic-tip field from the emitter, normalised to the spray voltage, and room air drawn into the capillary as a sink flow. Ions start already desolvated, a set distance downstream of the tip.
* **Capillary:** compressible flow with wall friction (Fanno flow), choked at the exit, with the friction factor from the Reynolds number. The gas is taken to be at the capillary temperature from the entrance on.
* **Free jet:** Ashkenas–Sherman centreline Mach number for an under-expanded sonic jet, up to the Mach disk at 0.67·D·√(P0/Pb).
* **Ions:** dv/dt = (u_gas − v)/τ + (q/m)E with τ = mK/q and K scaled with gas density. An exponential step keeps this stable from about 1 ns at 1 atm to the slow relaxation in the jet. A hard-sphere correction stops the drift growing linearly at high E/N. The outputs are speed, kinetic energy and the Wannier effective temperature, T_eff = T + M_gas (v − u)²/3k.

### Ion funnel (`ionsim/funnel.py`)

* **Geometry:** `RingFunnel` (stacked rings, constant ID first and then a linear taper, with an exit plate), `PlanarFunnel` (two boards of strips across the ion path, DC guards beyond the strip ends, an exit slit), or `CadFunnel` (any bodies from `ionsim/cad.py`).
* **Electrics:** each electrode gets a DC voltage, an RF amplitude and an RF phase. Laplace is linear, so three solves (DC, cos and sin parts, `ionsim/laplace.py`) give the field at any time for any number of electrodes. Alternating 0°/180° RF needs only two.
* **Fields:** rings are solved on an axisymmetric r–z grid (`ionsim/axisym.py`, finite volumes with face-radius weights, exact symmetry). Everything else uses a 3D voxel grid (`ionsim/grid3d.py`). Both use an algebraic multigrid solver (PyAMG): 1 M nodes take about 1–2 s, 8 M about 10 s. The entrance face is open (zero normal field), as if an inlet sat in front, rather than a grounded wall.
* **CAD import (`ionsim/cad.py`):** STEP/IGES/BREP are read with gmsh (OpenCASCADE) and meshed. STL/OBJ/PLY are read with trimesh, plus a small OBJ reader that keeps named objects. Meshes are voxelised by ray parity along z (fills closed solids, holes included). A surface shell sampled at h/2 is added only where a part is too thin to fill, so 35 µm PCB copper still becomes electrode.
* **Ion motion (`ionsim/gasflight.py`):** Langevin dynamics. The field, a drag that relaxes the velocity to the gas velocity with τ = mK/q, and thermal kicks. Over each step the Ornstein–Uhlenbeck equations are integrated exactly, with correlated position and velocity noise. That stays stable and correct both when τ is longer than the RF period (RF inertia, the effective potential) and when it is far shorter (pure mobility drift).
* **Effective potential (plots):** U = q⟨E²⟩ / (2m(ω² + 1/τ²)), the RF pseudopotential reduced by gas damping.

### Default parameters

| Quadrupole (`examples/quad_demo.py`) | TWAVE (`examples/twave_demo.py`) |
|---|---|
| r0 = 4.0 mm, rod radius = 1.1487 r0 | pitch 0.508 mm, width 0.432 mm, gap 2.75 mm |
| f = 1.0 MHz, m/z 500, rod length 200 mm | 30 V, 200 m/s |
| 5 eV axial energy | 4 Torr N₂, 300 K |
| grid spacing r0/80, 40 RK4 steps per RF cycle | K0 = 1.00 and 1.03 cm²/Vs over 1 m |

| FAIMS (`examples/faims_demo.py`) | ESI inlet (`examples/inlet_demo.py`) |
|---|---|
| planar, gap 1.88 mm, 40 mm long, 5 m/s parabolic flow | emitter 5 mm from the inlet at 3500 V, ions released 1 mm from the tip |
| 4000 V bisinusoidal at 750 kHz (87 Td) | capillary 0.5 mm × 60 mm at 275 °C |
| N₂ at 760 Torr, 300 K | 1.5 Torr first stage, skimmer 3 mm away at 30 V |
| example type A / B / C species | m/z 200 (1+), 500 (1+), 1500 (3+) |

| Ring funnel (`examples/funnel_demo.py`) | Planar funnel |
|---|---|
| 100 rings, 1 mm pitch, 0.5 mm thick, 45 mm plates | 60 strips per board, 1 mm pitch, 0.5 mm wide |
| ID 25.4 mm (10 rings) tapering to 2.5 mm, 2.5 mm exit orifice | boards 4 mm apart, strips 20 → 2 mm, 2 × 3 mm exit slit |
| 100 Vpp at 1 MHz, 20 V/cm, last ring 20 V, exit plate 15 V | same electrics, guards +3 V above the local strip DC |
| 2 Torr N₂, 300 K, m/z 500 with K0 = 1.0 cm²/Vs, grid 0.05 mm | grid 0.2 mm (1.2 M nodes) |

---

## Validation

**Quadrupole**
* Grid field vs ideal hyperbolic: 0.99 near the axis (rod staircasing). This is expressed as an effective r0 = 4.019 mm and used for a/q.
* Operating points: (a = 0.20, q = 0.70) transmits 100%. (0.26, 0.70) and (0, 0.95) transmit 0%, matching Mathieu.
* Mass scan at a/q = 0.332: the Mathieu-stable band is 499–506. The simulated FWHM is 498–505, with soft edges from the finite number of RF cycles (~145), as expected for a real filter.

**TWAVE**
* For an ideal sine wave the mean velocity must follow theory: s for surfing (c ≥ s) and s − √(s² − c²) for rolling over, where c = K·Emax. The simulation matches to within 0.017 of the wave speed.
* K0 = 1.00 vs 1.03 cm²/Vs over 1 m gives resolving powers of 60 / 63 and a two-peak resolution of Rpp = 2.65.

**FAIMS**
* Simulated CV peaks land on the ⟨K(E)E⟩ = 0 theory: type A −31.59 vs −31.63 V, type B −6.99 vs −7.01 V, type C 13.94 vs 13.80 V.

**ESI inlet**
* Gas: 0.82 standard L/min at Re ≈ 1600 (laminar), with an entrance Mach number of 0.31 and a choked exit at 203 Torr.
* Heavy-ion limit: with no field in the jet, a light ion reaches 0.996 of the gas speed, and its energy matches ½·m·u_gas² (0.53 eV for m/z 100). Heavy ions lag the gas (0.95 for m/z 5000).
* Ion energy at the skimmer grows with mass: 1.05, 2.61 and 22.5 eV for m/z 200, 500 and 1500 (3+) with no field, or 2.7, 5.8 and 47.7 eV with 30 V.

**Ion funnel**
* Langevin flight in a uniform field: drift speed = K·E within 0.6%, and spread = √(2Dt) within the 2% statistical error. This holds both with time steps 20× shorter than τ and 20× longer.
* Axisymmetric solver vs the exact coaxial-cylinder potential: 5×10⁻⁶ V error (of 1 V), 0.02% field error.
* 3D solver and voxeliser vs concentric spheres: potential error 1.1% at a 0.25 mm grid and 0.5% at 0.125 mm (first order, as expected for voxels), and the voxelised sphere volume is within 1%.
* STEP import end to end: a 30-ring funnel written to STEP, imported, voxelised and solved in 3D matches the exact axisymmetric solve. The DC field agrees within 0.4% mid-funnel and within 2% at the ends (where the 3D box and the cylinder enclosure differ), and the RF field converges with grid spacing (16–17% → 7–9% rms going from 0.2 to 0.1 mm). Transmission at m/z 50–2000 is identical (100%), and transit times agree within 7%.
* Physics: with 140 Vpp and a 2.0 mm final ID, m/z 500 ions stall on axis between the last two rings. The pseudopotential calculation shows a 1.1 V axial barrier there against a 1.5 V/mm push. That's the known small-ID trapping at funnel exits, and the reason the defaults use 2.5 mm and 100 Vpp.

---

## Limits and roadmap

**Quadrupole, not modelled yet**
* 2D field only: no fringe fields at the rod entrance and exit (a 3D grid is the obvious next field upgrade).
* No space charge and no gas.
* Pure NumPy. Numba on `efield`/RK4 would give a large speed-up as ion counts grow.

**TWAVE, not modelled yet**
* The RF guard confinement across the device (ions ride the midplane), turns, space charge, high-field mobility (E/N dependence), and real ion injection or trapping.
* A 13 m serpentine is just a longer `PATH`, and the run time grows with it.

**FAIMS, not modelled yet**
* Cylindrical or curved electrodes (ion focusing), space charge, gas-mixture and clustering effects, and fringe fields at the cell ends. Diffusion uses the low-field D.

**ESI inlet, not modelled yet**
* Droplets and desolvation (ions start as bare ions), radial motion and wall losses in the capillary, gas heating along the capillary (the gas is set to the wall temperature at the entrance), and the 3D jet structure off axis.
* The mobility model is extrapolated into the very low-density jet. Treat jet energies as estimates, best for comparing ions and settings.

**Ion funnel, not modelled yet**
* Gas flow is a uniform axial speed, not a computed flow field (no jet from the inlet, no jet disrupter).
* Electrodes sit in vacuum: no PCB dielectric and no surface charging. No space charge.
* One RF frequency for the whole device (any phases). The exit region is DC only.
* 3D voxel fields are first order in the grid spacing, and the grid is capped at 40 M nodes, so very fine features in a large part need a coarser compromise or a cut-down model.

**Next: drag-field quadrupole (collision cell).** The integrator already has the three hooks this needs:

1. **Axial drag field:** a new `FieldSource` that adds E_z(z) (from segmented rods, tilted or tapered auxiliary electrodes, or a resistive coating). For segmented rods, solve an axial 1D/3D problem and add a DC ramp per segment.
2. **Gas collisions:** a `hooks` entry that applies, per step, a collision probability P = 1 − exp(−n σ v_rel dt) with a hard-sphere (or Langevin) cross section, and a centre-of-mass random scatter against a thermal gas atom (He, N₂ or Ar at a set pressure and temperature).
3. **Optional viscous drag shortcut:** a `forces` entry with mobility K, a = (q/m)E − v/τ, which is fast at high pressure.

Outputs to add then: transit-time distributions, energy thermalisation, CID energy, and transmission vs pressure.

---

## Project layout

```
ION-SIM/
├── ion_sim_app.py          # desktop app entry point
├── ionsim/
│   ├── geometry.py         # electrodes on a grid
│   ├── solver.py           # Laplace basis solves
│   ├── fields.py           # fast-adjust superposition, interpolation
│   ├── integrator.py       # vectorised RK4 ion flight
│   ├── mathieu.py          # a/q and stability
│   ├── twave.py            # traveling-wave boards and mobility flight
│   ├── faims.py            # FAIMS waveform, alpha(E/N), CV scans, dispersion
│   ├── inlet.py            # ESI spray region, capillary flow, free jet, ion energy
│   ├── funnel.py           # ring, planar and CAD ion funnels, electrics, flight helper
│   ├── laplace.py          # multigrid Laplace solve and DC/RF electrode drives
│   ├── axisym.py           # axisymmetric r-z grid and field (ring electrodes)
│   ├── grid3d.py           # 3D voxel grid, mesh voxeliser and field
│   ├── cad.py              # STEP / IGES / STL / OBJ import
│   ├── gasflight.py        # Langevin ion flight through gas
│   └── gui/                # PySide6 app: one tab per device, shared widgets
├── examples/               # one demo per device, geometry/ with an example STEP funnel
├── out/                    # example plots
├── docs/                   # screenshots and the script that makes them
├── build.sh, build_windows.bat, ion_sim.spec   # standalone builds
└── .github/workflows/build.yml                 # CI builds for all three OSes
```
