# Ion Sim

**A SIMION-style ion optics and ion mobility simulator in Python, with a desktop app.**

[![build-app](https://github.com/anderson509chris/ION-SIM/actions/workflows/build.yml/badge.svg)](https://github.com/anderson509chris/ION-SIM/actions/workflows/build.yml)
![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)
![Platforms](https://img.shields.io/badge/platforms-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)

![Ion Sim, quadrupole tab](docs/screenshots/quad_trajectories.png)

Ion Sim solves electrode fields on a grid, flies thousands of ions through them, and checks the results against theory. It currently models five devices, plus a device builder for your own drawings:

* **Quadrupole mass filter:** four round rods with RF + DC, a Laplace field solve, RK4 ion flight and a Mathieu stability check.
* **TWAVE traveling-wave ion mobility separation:** stepped electrode boards in 4 Torr gas, with mobility drift and diffusion.
* **FAIMS:** an asymmetric waveform across a planar gap at 1 atm, field-dependent mobility α(E/N), CV spectra and dispersion plots.
* **ESI inlet capillary:** spray region, choked capillary flow and free-jet expansion, with ion velocity and kinetic energy from the emitter to the skimmer.
* **Ion funnel:** the PNNL stacked-ring funnel (copper rings on FR4 PCBs), a planar two-board funnel, or **your own electrodes imported from STEP, STL or OBJ**, with RF + DC fields and ion flight through 1–10 Torr gas. PCB dielectric, a computed **gas flow field** and the beam's **space charge** are included.
* **Device builder:** load a **DXF** drawing or a **STEP / STL / OBJ** model, click each electrode to see and set its DC, RF and TWAVE phase (colored by voltage), apply a DC gradient or an 8-phase TWAVE pattern across many pads at once, then solve and fly ions. Works for flat boards (TWAVE), ring stacks drawn as a profile, rod sets, and full 3D parts.

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

Requires Python 3.12 or newer. Dependencies: NumPy, SciPy, Matplotlib, PySide6, and for imported geometry PyAMG (3D field solver), ezdxf (DXF), trimesh (STL/OBJ) and gmsh (STEP/IGES).

On Linux, gmsh and Qt need a few system libraries: `sudo apt install libglu1-mesa libxcursor1 libxinerama1 libxft2 libegl1 libxkbcommon0`.

To check an install, run `python tests/smoke_test.py` (about 20 s). It loads the example DXF and STEP files, solves them and flies ions.

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

Pick the funnel type at the top: **Ring funnel** (stacked PCBs), **Planar funnel** (two boards), or **Imported CAD**. Then set the electrics (RF amplitude and frequency, DC gradient, exit plate), gas and ions (m/z, charge, K0, start radius). It runs either trajectories for one m/z or transmission against m/z. Plot tabs: Geometry, Potential (DC, RF effective potential and space charge), Trajectories, Transmission, Exit beam and Gas flow. While the geometry stays the same, the solver is kept, so changing only voltages re-runs in about a second.

* **PCB dielectric.** Ring funnel: each ring is a copper ring around the bore (**Copper ring width**, plated through both faces) on an FR4 board out to the plate diameter (**Board εr**, 4.4 for FR4). Set the width to 0 for all-metal plates. Planar funnel: the strips sit on **Board thickness** of FR4 (0 for strips in free space). Imported CAD: type a permittivity in a body's **Insulator εr** column to make it a board or spacer instead of an electrode. Insulators are drawn in tan, shape the field and stop ions, but carry no voltage.
* **Gas flow.** *Still gas*, *Uniform axial speed*, or *Flow field from gas throughput*. The flow field takes the throughput in slm and an inlet disc on the entrance face (radius 0 = the entrance bore). It solves viscous flow through the open space, pumped out through every open side, and shows it on the **Gas flow** tab. In the ring funnel almost all the gas leaves between the rings, and about 0.2% reaches the exit.
* **Space charge.** Give the **Ion current** in nA (0 = off). The ions are flown several times. Each flight's charge (the current times the time spent near each grid node) sets the space-charge field for the next flight. The log shows the space-charge potential at each step and says if it hasn't settled, which usually means more current than the funnel can carry.

| Ring funnel trajectories (PCB rings, gas flow, 1 nA) | DC, RF effective and space-charge potential |
|---|---|
| ![Funnel trajectories](docs/screenshots/funnel_trajectories.png) | ![Funnel potential](docs/screenshots/funnel_potential.png) |

| Gas flow through the ring funnel |
|---|
| ![Funnel gas flow](docs/screenshots/funnel_gas_flow.png) |

**Importing your own funnel (STEP, STL or OBJ):**

1. Choose **Imported CAD** and press **Load STEP / STL / OBJ...**. Every solid body in the file becomes one electrode, named from the file.
2. Say which CAD axis the ions travel along (they enter at its low end). For STL/OBJ, also give the units, because those formats don't store them.
3. The table fills with a standard funnel pattern: bodies sorted along the axis, RF alternating 0°/180°, and a DC ladder from the Electrics box. Tick **Last body is a DC exit plate** if it is. Then edit any DC, RF amplitude or phase by hand. Guards, lenses and grounded parts are just rows with RF set to 0. Boards and spacers get a number in **Insulator εr** (press Auto-assign again afterwards so the pattern skips them).
4. Check the **Geometry** tab (true cross-sections through the start point), set the start point and detector plane, and pick a grid spacing about a third of your smallest electrode gap. The label shows the grid size, and the limit is 40 M nodes.

`examples/geometry/ring_funnel_30.step` is a ready-made 30-ring funnel to try.

| Imported STEP file |
|---|
| ![Imported STEP funnel](docs/screenshots/funnel_cad_geometry.png) |

### Device builder (DXF / CAD)

The general tab for your own electrodes. Work down the left panel:

1. **Geometry.** Press **Open DXF / STEP / STL…** and pick the file. Under **Read as**, choose *PCB boards* (pads drawn as seen from above, painted on two facing boards a set gap apart, ions travel along the drawing's x), *Axisymmetric half profile* (a half cross-section of rings or lenses: drawing x is the axis, y the radius), *2D cross-section* (rods or plates extruded a set length), or *3D solids* for STEP/STL/OBJ. Units come from the DXF header when it has them.
2. **Electrodes.** Every closed shape becomes an electrode, named by its layer and position (or by a TEXT label inside it). Click a shape on the **Geometry** plot, or a row in the table, to see and edit its DC, RF and TWAVE phase. Ctrl- or Shift-click selects several. Colors show the DC voltage, and RF electrodes are hatched. You can pick a whole layer at once. A number in **Insulator εr** makes a part an insulator (PCB, spacer). For PCB boards, **PCB substrate behind pads** adds an FR4 board of that thickness behind the pads, with a ground plane on its back.
3. **Patterns** for the selected electrodes: **Fixed** (same voltages), **DC gradient** (start and end voltage along the ion axis, with optional RF alternating 0°/180° pad to pad), **TWAVE phases** (pads numbered 1..N along the axis, pads side by side share a phase, pitch measured from the drawing), **Ground**, **Remove from TWAVE**.
4. **Electrics:** RF frequency, and the TWAVE amplitude, speed, number of phases (8 by default) and how many are on. The step time is shown.
5. **Grid:** **Quick** (seconds, for setting up) or **Detailed** (finer, for final numbers), or a custom spacing. **Check geometry on this grid** reports overlapping electrodes, electrodes closer than the grid spacing, parts too thin for the grid, and parts touching the grid edge.
6. **Ions and gas**, then **Solve and fly**. Changing only voltages re-uses the grid and warm-starts the solve.
7. **Gas flow and space charge:** the same flow field (axisymmetric and 3D devices) and space-charge iteration as the funnel tab.

**Save setup…** writes a `.ionsim` file (the drawing path plus every voltage and setting). `examples/geometry/` has three ready setups: `twave_board.ionsim` (48 pads, 8-phase TWAVE with RF guards), `ring_funnel_profile.ionsim` (a 20-ring funnel drawn as a DXF profile) and `quad_rods.ionsim` (quadrupole rods). `examples/geometry/make_dxf_examples.py` shows how the DXFs are drawn.

| TWAVE board from DXF | Ions surfing the wave |
|---|---|
| ![Device builder TWAVE board](docs/screenshots/device_twave_board.png) | ![Device builder TWAVE trajectories](docs/screenshots/device_twave_trajectories.png) |

| Ring funnel drawn as a DXF profile |
|---|
| ![Device builder funnel profile](docs/screenshots/device_funnel_profile.png) |

In the **Ion funnel** tab, imported CAD bodies are also clickable and colored by voltage, and the same Quick / Detailed grid choice is there.

<sub>To refresh these screenshots after changing the GUI, run `python docs/make_screenshots.py`.</sub>

---

## Example results

The scripts in `examples/` reproduce these plots without the GUI:

```bash
python examples/quad_demo.py       # ~1 min, writes PNGs 1-4 to out/
python examples/twave_demo.py      # a few min, writes PNGs 5-8 to out/
python examples/faims_demo.py      # ~40 s, writes PNGs 9-11 to out/
python examples/inlet_demo.py      # seconds, writes PNGs 12-14 to out/
python examples/funnel_demo.py     # ~6 min, writes PNGs 15-20 to out/
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
| **Ring funnel.** 100 PCB rings (2 mm copper on FR4) tapering from 25.4 to 2.5 mm, 100 Vpp at 1 MHz, 20 V/cm, 2 Torr. Ions entering across 16 mm leave in a 0.24 mm (rms) beam, 1.3 ms later. | **Transmission against m/z.** Low RF (30 Vpp) loses heavy ions onto the tapering rings (weak confinement). High RF (300 Vpp) traps mid-mass ions on axis at the small last rings, where the RF makes an axial barrier. At 100 Vpp everything from m/z 35 to 5000 gets through. |
| ![Planar funnel](out/17_funnel_planar.png) | ![CAD check](out/18_funnel_cad_check.png) |
| **Planar funnel.** Two FR4 boards 4 mm apart, RF strips shortening from 20 to 2 mm, +3 V DC guards beyond the strip ends. Top: the board face (x); bottom: across the gap (y). | **CAD import check.** A 30-ring funnel written to STEP, imported and solved in 3D, against the exact axisymmetric solve of the same rings: same fields, same transmission. |
| ![Gas flow](out/19_funnel_gas_flow.png) | ![Space charge](out/20_funnel_space_charge.png) |
| **Gas flow.** 2 slm into the 25.4 mm entrance at 2 Torr. The gas spreads out of the bore and is pumped away between the PCB rings. About 4–5 m/s remains on axis, rising to ~10 m/s near the exit, and 0.2% of the gas gets through. m/z 500 transit drops from 1.33 to 1.25 ms. | **Space charge.** m/z 500 against beam current. Up to 3 nA everything gets through and the beam just gets wider (0.27 → 0.45 mm rms). At 10 nA the 4 V space-charge potential costs 12% of the ions. At 30 nA the charge piles up at the exit, where the RF makes a small axial barrier, and the beam chokes. |

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

* **Geometry:** `RingFunnel` (stacked PCB rings, constant ID first and then a linear taper, with an exit plate), `PlanarFunnel` (two FR4 boards of strips across the ion path, DC guards beyond the strip ends, an exit slit), or `CadFunnel` (any bodies from `ionsim/cad.py`, some of them insulators).
* **Dielectrics (`ionsim/laplace.py`, `ionsim/gridtools.py`):** each grid node carries a relative permittivity, and every link weight is scaled by the harmonic mean of its two ends (the series capacitance of two half cells). The same matrix then solves ∇·(ε∇φ) = −ρ. Links into an electrode take the permittivity of the free side. Insulator nodes aren't fixed in the solve, but ions that reach them stop.
* **Gas flow (`ionsim/gasflow.py`):** viscous, incompressible flow in the lubrication (local Poiseuille) approximation: u = −(k/μ)∇p with ∇·(k∇p) = 0 and k = c·δ(2L − δ). Here δ is the distance to the nearest wall and L the local channel half-width, found by following δ uphill to the channel's centre ridge. With c = ½ in slits and ¼ in round bores, that is the exact Poiseuille profile and conductance for straight channels, so a funnel's narrow ring gaps pass far less gas than their area suggests. Gas enters through a disc on the entrance face, every other open side is pumped, and the result is scaled to the throughput (μ drops out). Ions read u at their mid-step position.
* **Space charge (`ionsim/spacecharge.py`):** a steady beam of current I is represented by the N flown ions, each standing for I/N. The charge near each node is I/N times the total time ions spend there, deposited cloud-in-cell during the flight. Poisson is solved for it with every electrode at 0 V (same matrix and factorization as the applied field), its field is added to the applied field, and the ions are flown again. Four passes with 0.4 under-relaxation settle well-behaved beams. The run reports when the potential hasn't settled.
* **Electrics:** each electrode gets a DC voltage, an RF amplitude and an RF phase. Laplace is linear, so three solves (DC, cos and sin parts, `ionsim/laplace.py`) give the field at any time for any number of electrodes. Alternating 0°/180° RF needs only two.
* **Fields:** rings are solved on an axisymmetric r–z grid (`ionsim/axisym.py`, finite volumes with face-radius weights, exact symmetry). Everything else uses a 3D voxel grid (`ionsim/grid3d.py`). 2D grids are solved by direct sparse LU: the factorization is kept, so a re-solve takes well under a second. 3D grids use algebraic multigrid (PyAMG) at a relative tolerance of 1e-7, warm-started from the previous solution when only voltages change: 1 M nodes take about 1–2 s, 8 M about 10 s. The entrance face is open (zero normal field), as if an inlet sat in front, rather than a grounded wall.
* **CAD import (`ionsim/cad.py`):** STEP/IGES/BREP are read with gmsh (OpenCASCADE) and meshed. STL/OBJ/PLY are read with trimesh, plus a small OBJ reader that keeps named objects. Meshes are voxelised by ray parity along z (fills closed solids, holes included). A surface shell sampled at h/2 is added only where a part is too thin to fill, so 35 µm PCB copper still becomes electrode.
* **Ion motion (`ionsim/gasflight.py`):** Langevin dynamics. The field (sampled at the middle of each step, which keeps RF motion accurate at 30 steps per cycle even in vacuum), a drag that relaxes the velocity to the gas velocity with τ = mK/q, and thermal kicks. Over each step the Ornstein–Uhlenbeck equations are integrated exactly, with correlated position and velocity noise. That stays stable and correct both when τ is longer than the RF period (RF inertia, the effective potential) and when it is far shorter (pure mobility drift).
* **Effective potential (plots):** U = q⟨E²⟩ / (2m(ω² + 1/τ²)), the RF pseudopotential reduced by gas damping.

### Device builder (`ionsim/dxf.py`, `ionsim/device.py`)

* **DXF reading:** ezdxf reads polylines, circles, arcs, splines, ellipses, solids and hatches (blocks are exploded). Arcs and curves are flattened to polygons. Loose LINE and ARC pieces are joined end to end into closed loops, and a loop inside another on the same layer becomes a hole. Open lines that never close are reported, not used.
* **Electrode model:** each electrode has a DC voltage, an RF amplitude and phase, and an optional TWAVE phase number. Laplace is linear, so the field is the sum of one solve for DC, two for RF (cos and sin) and one per TWAVE phase. TWAVE phase k is on while ((k − step) mod N) < n_on, and the step advances every pitch/speed.
* **Grids:** flat boards and 3D CAD use the 3D voxel grid; profiles use the axisymmetric grid; 2D cross-sections use a planar 2D grid. Interiors are painted first, then outlines, so neighbors never steal each other's nodes.
* **Flight:** the same Langevin integrator as the funnel. The time step is picked from the RF period, the TWAVE step time and the device length, and flight time from the expected drift.

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
| 2 mm copper ring around the bore on FR4 (εr 4.4) | strips on 1.6 mm FR4 (εr 4.4) |
| ID 25.4 mm (10 rings) tapering to 2.5 mm, 2.5 mm exit orifice | boards 4 mm apart, strips 20 → 2 mm, 2 × 3 mm exit slit |
| 100 Vpp at 1 MHz, 20 V/cm, last ring 20 V, exit plate 15 V | same electrics, guards +3 V above the local strip DC |
| 2 Torr N₂, 300 K, m/z 500 with K0 = 1.0 cm²/Vs, grid 0.05 mm | grid 0.2 mm (1.95 M nodes) |

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
* Solver speed: the 30-ring STEP funnel at a 0.2 mm grid went from 19.5 s to 2.9 s with the 1e-7 tolerance, and a warm-started re-solve after a voltage change takes 1.5 s, with no visible change in the fields.
* Dielectric: a slab of εr 4 between parallel plates gives exactly the series-capacitor potential (to 1e-4 V of 100 V).
* Space charge: a uniformly charged beam in a grounded tube gives the analytic axis potential ρa²/4ε₀·(1 + 2 ln R/a) within 0.6%, on both the axisymmetric and the 3D grid.
* Gas flow: a pipe gives the Poiseuille profile 2(1 − r²/R²)·ū within ~3%. Earlier, an inviscid (potential flow) version sent 99.99999% of the gas out between the rings, because it ignores how much viscosity holds back flow through narrow gaps. That's why the lubrication model is used.
* PCB rings vs all-metal plates (100 Vpp, m/z 500): both transmit 100%, and transit is 1.33 vs 1.29 ms. The FR4 between the copper rings slightly weakens the axial DC field.
* Physics: with 140 Vpp and a 2.0 mm final ID, m/z 500 ions stall on axis between the last two rings. The pseudopotential calculation shows a 1.1 V axial barrier there against a 1.5 V/mm push. That's the known small-ID trapping at funnel exits, and the reason the defaults use 2.5 mm and 100 Vpp.

**Device builder**
* Quadrupole rods from a DXF (2D cross-section): 100% transmitted at (a = 0.20, q = 0.70) and 0% at (0.26, 0.70) and (0, 0.95), as the Mathieu diagram says. Before the midpoint field sampling, the gas-flight integrator lost about 60% at the stable point in vacuum at 30 steps per cycle.
* Ring funnel drawn as a DXF profile (axisymmetric): 100% of m/z 500 transmitted at 2 Torr.
* TWAVE board from DXF (48 pads, 0.508 mm pitch, 8 phases, 4 on, 20 V, 150 m/s, RF guard strips): 97–98% transmitted, mean ion speed 14.3–14.4 m/s against 13.4 m/s from the sine-wave roll-over theory. The real wave is square rather than sine, so it pushes slightly harder. Quick grid: 0.53 M nodes, 10 field parts solved in 23 s, re-solve after a voltage change 4 s.

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

**Ion funnel and device builder, not modelled yet**
* Gas flow is viscous and incompressible with no inertia. The jet from an inlet capillary therefore doesn't carry its momentum downstream (there's no jet disrupter effect), and inlet speeds near sound speed (flagged in the log) are rough. A CFD import of the velocity field would cover both.
* Insulators carry no surface charge (no charging of exposed FR4 by stray ions).
* Space charge is the steady state of a continuous beam (DC current). Trapped or accumulated ion packets, and very high currents where the beam chokes, are only flagged, not followed in time. Charge is deposited on the field grid, so very thin beams need a fine grid.
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
│   ├── gasflow.py          # viscous gas flow field on the grid (lubrication model)
│   ├── spacecharge.py      # beam space charge: Poisson iteration with the flight
│   ├── gridtools.py        # insulators, grid interpolation, charge deposition
│   ├── dxf.py              # DXF reader: closed shapes, holes, layers, units
│   ├── device.py           # device builder: electrodes, patterns, grids, solve, flight
│   └── gui/                # PySide6 app: one tab per device, shared widgets,
│                           #   device_tab.py and electrode_view.py (clickable electrodes)
├── examples/               # one demo per device; geometry/ has example STEP, DXF and .ionsim setups
├── tests/smoke_test.py     # quick end-to-end check of imports, solves and the app window
├── out/                    # example plots
├── docs/                   # screenshots and the script that makes them
├── build.sh, build_windows.bat, ion_sim.spec   # standalone builds
└── .github/workflows/build.yml                 # CI builds for all three OSes
```
