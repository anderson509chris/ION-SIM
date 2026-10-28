"""Quick end-to-end check that an install works (about a minute). Run:
    python tests/smoke_test.py          (set QT_QPA_PLATFORM=offscreen on a headless machine)
Exits non-zero with a message if anything is broken."""
import os, sys, time, tempfile, json
import numpy as np

os.environ.setdefault("IONSIM_DRIVERS", os.path.join(tempfile.mkdtemp(), "drivers.json"))   # keep the user's driver library untouched

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

# drivers: TWAVE gates (4 of 8 high, stepping one phase per step) and wiring a board to a combiner
from ionsim.drivers import twave_rf_combiner, quad_excitation, Tone
drv = twave_rf_combiner()
st = drv.step_time()
gates = np.array([drv.tw_spec().level(p, (np.arange(8) + 0.5) * st) for p in range(8)])
check("driver TWAVE gates", np.all(gates.sum(0) == 4) and gates[0, 0] == 1 and gates[0, 1] == 0 and gates[1, 1] == 1,
      f"{int(gates.sum(0)[0])} of 8 high")
dev = from_drawing(load_dxf(os.path.join(GEO, "twave_board.dxf")), "boards")
tw_pads = [e.name for e in dev.electrodes if e.layer == "TW"]
missing = dev.wire_driver(tw_pads, drv, "twave")
from ionsim.device import part_values
parts, _ = part_values(dev)
check("driver wiring", not missing and sum(p.startswith("tw@") for p in parts) == 8 and "cos" in parts,
      f"{len(parts)} field parts for {len(drv.outputs)} outputs")

# quadrupole: a tone at twice the secular frequency, on the RF, ejects that m/z (and only with the tone)
from ionsim import quadrupole_rods, solve_basis, GridField2D, make_beam, Simulation
g = quadrupole_rods(4e-3, dx=4e-3 / 40, split=True); basis = solve_basis(g)
qd = quad_excitation()
fx, _ = mathieu.secular_freqs(0, qd.output("Rods A (x)").rf, 28, 4.03e-3, 1e6)
T = []
for amp in (0.0, 1.0):
    qd.tones = [Tone(2 * fx, amp)] if amp else []
    volts = {r: (lambda tt, o=o: float(qd.waveform(o, tt))) for r, o in
             (("x+", "Rods A (x)"), ("x-", "Rods A (x)"), ("y+", "Rods B (y)"), ("y-", "Rods B (y)"))}
    ions = make_beam(20, 28, 1, 1.0, 0.3e-3, 1.0, birth_window=1e-6, rng=np.random.default_rng(0))
    T.append(Simulation([GridField2D(g, basis, volts)], z_end=0.2).run(ions, 1 / 30e6, 3e-4).transmission)
check("quad tone ejects its m/z", T[0] > 0.9 and T[1] < 0.2, f"no tone {T[0]:.0%}, tone at 2 f_sec {T[1]:.0%}")

# ion heating: effective temperature in a DC drift field = T + M (K E)^2 / 3k
from ionsim.gasflight import disc_ions, fly
from ionsim.heating import HeatingModel
from ionsim.twave import mobility
from ionsim.constants import AMU, K_B, E_CHARGE


class Uniform:
    def efield(self, t, p):
        e = np.zeros_like(p); e[:, 2] = 2e4; return e

    def hit(self, p):
        return np.zeros(len(p), bool)


Kh = mobility(1.0, 4.0, 300)
res = fly(Uniform(), disc_ions(200, 500, Kh, 1, radius=0, rng=np.random.default_rng(0)), 2e-8, 2e-5, 1.0,
          heating=HeatingModel(), rng=np.random.default_rng(0))
want = 300 + 28.013 * AMU * (Kh * 2e4) ** 2 / (3 * K_B)
check("ion heating (Wannier)", abs(res.heat.teff_mean.mean() - want) < 0.01 * want,
      f"{res.heat.teff_mean.mean():.1f} K vs {want:.1f} K")

# ion mixtures: two species flown together, interleaved, each with its own m/z, charge and mobility
res, info = fly_device(fld, [200, 1200], [1.5, 0.8], [20, 20], 4.0, charge_state=[1, 2], radius=2e-3,
                       z_start=0.5e-3, z_end=31e-3, rng=np.random.default_rng(1))
sp = res.species
t_light, t_heavy = (np.nanmean(res.t_end[(sp == k) & (res.status == 1)]) for k in (0, 1))
check("ion mixture", list(sp[:4]) == [0, 1, 0, 1] and (sp == 1).sum() == 20 and t_light < t_heavy,
      f"transit {t_light * 1e3:.2f} vs {t_heavy * 1e3:.2f} ms, transmission {res.transmission:.0%}")

# OA-TOF: exact flight time of an ion at rest (instant pulse, ideal grids, linear) ...
from ionsim import tof as TOF
from ionsim.ions import IonSpecies
lin = TOF.OATof(reflectron="none", pulse=TOF.Pulse(rise=0, fall=0), det_jitter=0)
still = TOF.Beam(sigma_x=0, divergence_deg=0, energy_spread_ev=0)
r = TOF.fly(lin, still, [IonSpecies("a", 500, 1, 1, 3)], rng=np.random.default_rng(0))
a1 = E_CHARGE * (lin.v_push - lin.v_pull) / lin.push_gap / (500 * AMU)
t1 = np.sqrt(lin.push_gap / a1); v1 = a1 * t1                     # starts mid-gap
a2 = E_CHARGE * (lin.v_pull - lin.v_drift) / lin.accel_gap / (500 * AMU)
v2 = np.sqrt(v1 ** 2 + 2 * a2 * lin.accel_gap)
want = t1 + (v2 - v1) / a2 + lin.drift_length / v2
check("TOF flight time (exact)", abs(r.t[0] / want - 1) < 1e-9, f"{r.t[0] * 1e6:.6f} µs vs {want * 1e6:.6f} µs")
# ... the best detector distance of a linear TOF is the Wiley-McLaren space focus ...
spread = TOF.Beam(sigma_x=0.3e-3, divergence_deg=0, energy_spread_ev=0)
D = TOF.wiley_mclaren_focus(lin, spread)
Ls = np.linspace(0.6 * D, 1.4 * D, 17)
R, _ = TOF.scan(lin, spread, [IonSpecies("a", 500, 1, 1, 1500)], "drift_length", Ls)
check("TOF space focus (Wiley-McLaren)", abs(Ls[np.argmax(R)] / D - 1) < 0.06, f"best {Ls[np.argmax(R)] * 1e3:.1f} mm vs {D * 1e3:.1f} mm")
# ... and the suggested two-stage reflectron reaches high resolution
refl, _ = TOF.suggest_reflectron(TOF.OATof(), TOF.Beam())
Rr = TOF.resolution(refl, TOF.Beam(), [IonSpecies("a", 500, 1, 1, 2000)])
check("TOF reflectron", Rr > 10000, f"resolution {Rr:.0f}")
# a pulsed driver output becomes its own field part in the Device builder
from ionsim.drivers import tof_pulser
pz = tof_pulser()
dev = from_drawing(load_dxf(os.path.join(GEO, "ring_funnel_profile.dxf")), "axisym")
dev.wire_driver([dev.electrodes[0].name], pz, "one", "Pusher")
parts, meta = part_values(dev)
lv = pz.pulse_spec().level(np.array([0.0, 1e-6, 1 / pz.pulse_rate + 1e-6]))
check("driver pulse", any(m[0] == "pulse" for m in meta.values()) and list(lv) == [0.0, 1.0, 1.0],
      f"parts {list(parts)}")

# imported waveforms: a SPICE result (ngspice binary raw) ...
from ionsim import waveforms as WF
from ionsim.drivers import Driver, Output
SP = os.path.join(ROOT, "examples", "spice")
spice = WF.spice_traces(os.path.join(SP, "oa_tof_pusher.raw"))
push = spice["v(push)"]
check("SPICE raw (ngspice)", set(spice) == {"v(push)", "v(pull)"} and abs(push.level(4e-6) - 600) < 5,
      f"{push}, {push.level(4e-6):.1f} V on the flat top")
# ... an LTspice raw file: UTF-16 header, float32 values, compressed (negative) times, two .step runs ...
def ltspice_raw(path, t, cols, fast=False):
    names = ["time"] + list(cols)
    hdr = ("Title: * test.asc\nDate: today\nPlotname: Transient Analysis\n"
           f"Flags: real forward stepped{' fastaccess' if fast else ''}\nNo. Variables: {len(names)}\n"
           f"No. Points: {len(t)}\nOffset:   0.0000000000000000e+000\nCommand: Linear Technology Corporation LTspice XVII\n"
           "Variables:\n" + "".join(f"\t{k}\t{n}\t{'time' if k == 0 else 'voltage'}\n" for k, n in enumerate(names)) + "Binary:\n")
    tt = np.array(t, "<f8"); tt[3::4] *= -1                      # compressed points carry a negative time
    vals = [np.asarray(c, "<f4") for c in cols.values()]
    if fast:
        body = tt.tobytes() + b"".join(v.tobytes() for v in vals)
    else:
        rec = np.zeros(len(t), [("t", "<f8")] + [(f"v{k}", "<f4") for k in range(len(vals))])
        rec["t"] = tt
        for k, v in enumerate(vals):
            rec[f"v{k}"] = v
        body = rec.tobytes()
    with open(path, "wb") as f:
        f.write(hdr.encode("utf-16-le") + body)
tl = np.linspace(0, 1e-6, 201)
t2 = np.r_[tl, tl]                                             # two stepped runs back to back
lt_ok = []
for fast in (False, True):
    fn = os.path.join(tempfile.mkdtemp(), "lt.raw")
    ltspice_raw(fn, t2, {"V(out)": np.r_[np.sin(2e7 * tl), 2 * np.sin(2e7 * tl)], "V(n001)": np.r_[tl * 0 + 1, tl * 0 + 2]}, fast)
    r1, r2 = WF.read_raw(fn), WF.read_raw(fn, step=1)
    lt_ok.append(WF.raw_steps(fn) == 2 and np.allclose(r1["V(out)"][1], np.sin(2e7 * tl), atol=1e-6)
                 and np.allclose(r2["V(out)"][1], 2 * np.sin(2e7 * tl), atol=1e-6) and np.all(np.diff(r1["V(out)"][0]) > 0))
check("SPICE raw (LTspice)", all(lt_ok), "float32, compressed, fastaccess, 2 steps")
# ... running ngspice itself, when it is installed
if WF.find_simulator("ngspice")[1]:
    run = WF.spice_traces(WF.run_spice(os.path.join(SP, "oa_tof_pusher.cir"), "ngspice"))
    check("SPICE run (ngspice)", np.allclose(run["v(push)"].level([1e-6, 4e-6]), push.level([1e-6, 4e-6]), atol=1e-3))
# ... and a Rigol CSV saved on the scope (t = 0 at the trigger)
sc = WF.read_table(os.path.join(ROOT, "examples", "scope", "rigol_pusher_ch1.csv"))
tr = WF.Trace("CH1", *sc["CH1"])
check("scope CSV (Rigol)", abs(tr.t0 + 1.5e-6) < 1e-12 and abs(tr.from_trigger().level(3e-6) - 600) < 20,
      f"{tr}, trigger {-tr.t0 * 1e6:.2f} µs in")

# live capture from a Rigol scope (a pretend one on a local socket): screen, whole memory in chunks, 16-bit DHO
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fake_rigol import FakeRigol
from ionsim.scope import RigolScope, connect
caps = []
for model in ("DS1104Z", "DHO1204"):
    fk = FakeRigol(model, screen=1200, memory=600_000, xinc=1e-10)
    with RigolScope(connect(f"127.0.0.1:{fk.port}")) as scope:
        on = scope.channels()
        a = scope.capture("CHAN1")
        b = scope.capture("CH1", memory=True)
    want = fk.signal("CHAN1", b.t + b.t0)
    caps.append((on == ["CHAN1", "CHAN2"] and len(a.t) == 1200 and len(b.t) == 600_000 and
                 np.max(np.abs(b.v - want)) <= 0.51 * (1600 / (65535 if fk.word else 255)), len(b.t)))
check("scope capture (Rigol SCPI)", all(c[0] for c in caps), f"{caps[0][1]} points read in chunks, BYTE and WORD")

# a driver output carrying a waveform is one field part, and flies like the same signal made by the driver
qa = Driver("RF from a trace", rf_freq=1e6, outputs=[Output("A", rf=300.0), Output("B", rf=300.0, rf_phase=180.0)])
tq = np.linspace(0, 1e-6, 2001)
qb = Driver("RF as a trace", rf_freq=1e6, outputs=[Output("A", wave="rf", wave_scale=300.0),
                                                     Output("B", wave="rf", wave_scale=-300.0)])
qb.add_wave(WF.Trace("rf", tq, np.cos(2 * np.pi * 1e6 * tq)))
qb = Driver.from_dict(json.loads(json.dumps(qb.to_dict())))   # and it survives the driver library's JSON
dev = from_drawing(load_dxf(os.path.join(GEO, "quad_rods.dxf")), "planar2d")
dev.length = 0.1
A = [e.name for e in dev.electrodes if e.layer == "ROD_A"]
B = [e.name for e in dev.electrodes if e.layer == "ROD_B"]
dg = dev.build(dev.default_h("Quick"))
fins = []
for drv in (qa, qb):
    dev.drivers = {}
    dev.wire_driver(A, drv, "one", "A"); dev.wire_driver(B, drv, "one", "B")
    fld = DeviceField(dg)
    r, _ = fly_device(fld, 500, 1.0, 20, 1e-7, energy_ev=5, radius=0.3e-3, z_start=0, z_end=0.1, dt=1 / 1e6 / 40,
                      rng=np.random.default_rng(0))
    fins.append((r, list(fld.parts)))
dpos = np.max(np.abs(fins[0][0].final_pos - fins[1][0].final_pos))
check("driver waveform part", fins[1][1] == ["dc", "wave@RF as a trace#rf"] and dpos < 2e-6,
      f"parts {fins[1][1]}, paths agree to {dpos * 1e6:.2f} µm, transmission {fins[1][0].transmission:.0%}")
# electrodes as a circuit load: a coax's capacitance per length from the field solve
from ionsim.circuit import capacitance_matrix, device_capacitance, spice_subckt, loaded_drive, apply_load, clear_load
from ionsim.waveforms import find_simulator
from ionsim.axisym import AxisymGrid
a_c, b_c, h_c = 1e-3, 3e-3, 0.05e-3
c_in = []
for L in (20e-3, 30e-3):                    # the difference of two lengths takes the ends out
    gc = AxisymGrid(b_c + 2 * h_c, 0, L + 10e-3, h_c)
    gc.add_annulus("in", 0, a_c, 5e-3, 5e-3 + L); gc.add_annulus("out", b_c, b_c + h_c, 1e-3, L + 9e-3)
    c_in.append(capacitance_matrix(gc.solver(), gc.electrode, 2 * np.pi * h_c, [[1], [2]]))
per_m = (c_in[1][0, 0] - c_in[0][0, 0]) / 10e-3
theory = 2 * np.pi * 8.8541878128e-12 / np.log(b_c / a_c)
check("electrode capacitance", abs(per_m / theory - 1) < 0.02 and abs(c_in[1][0, 1] - c_in[1][1, 0]) < 0.02 * abs(c_in[1][0, 1]),
      f"coax {per_m * 1e12:.2f} pF/m vs 2πε0/ln(b/a) {theory * 1e12:.2f} pF/m")
# the quadrupole's two rod pairs as a load: SPICE subcircuit, and the drive through 10 kΩ against RC theory
cap = device_capacitance(dg)
sub = spice_subckt(cap)
c_eff = cap.C[0, 0] - cap.C[0, 1]           # rod pairs driven + and -: each charges C_AA - C_AB
ok_c = cap.n == 2 and cap.asym < 0.02 and cap.mutual()[0, 1] > 0 and ".subckt IONSIM_LOAD" in sub and ".ends" in sub
detail = f"{cap.C[0, 0] * 1e12:.2f} pF per rod pair, {cap.mutual()[0, 1] * 1e12:.2f} pF between pairs"
if find_simulator("ngspice")[1]:
    ld = loaded_drive(dev, cap, r_source=1e4, periods=3)
    rep = ld.report()
    want = 1 / np.sqrt(1 + (2 * np.pi * 1e6 * 1e4 * c_eff) ** 2)
    got = rep[0]["swing_el"] / rep[0]["swing_src"]
    apply_load(dev, ld)
    with_load = list(part_values(dev)[0])
    clear_load(dev)
    ok_c = ok_c and abs(got / want - 1) < 0.02 and any(k.startswith("load#") for k in with_load) and not dev.load
    detail += f"; through 10 kΩ at 1 MHz the rods get {got:.3f} of the swing, RC theory {want:.3f}"
check("electrode load (SPICE)", ok_c, detail)
# sweeps and tolerances: spreads, sampling, and a sensitivity fit that finds the setting that matters
from ionsim import sweep as SWP
rng_s = np.random.default_rng(3)
sd_g = np.std(SWP.sample(10.0, 1.0, SWP.DISTRIBUTIONS[0], rng_s, 40000))
sd_u = np.std(SWP.sample(10.0, 1.0, SWP.DISTRIBUTIONS[1], rng_s, 40000))
sd_r = np.std(SWP.sample(10.0, 1.0, SWP.DISTRIBUTIONS[2], rng_s, 40000))
Xs = SWP.tolerance_trials([100.0, 5.0], [1.0, 0.1], [SWP.DISTRIBUTIONS[0]] * 2, 400, seed=1)
ys = 3.0 * (Xs[:, 0] - 100) + 0.1 * (Xs[:, 1] - 5) + 0.001 * rng_s.standard_normal(400)
con, r2 = SWP.contributions(Xs, ys)
slope = SWP.local_slope(np.linspace(0, 2, 11), np.linspace(0, 2, 11) ** 2, 1.0)
same = SWP.SameRandom(5); same.on()
draw1 = np.random.default_rng().random(3); same.restart(); draw2 = np.random.default_rng().random(3)
seeded = np.random.default_rng(1).random()
same.off()
ok_same = np.array_equal(draw1, draw2) and seeded == np.random.default_rng(1).random() and np.random.default_rng.__name__ == "default_rng"
check("sweep statistics", ok_same and all(abs(a / b - 1) < 0.03 for a, b in ((sd_g, SWP.sigma(1, SWP.DISTRIBUTIONS[0])),
      (sd_u, SWP.sigma(1, SWP.DISTRIBUTIONS[1])), (sd_r, SWP.sigma(1, SWP.DISTRIBUTIONS[2]))))
      and abs(con[0] / (3.0 * Xs[:, 0].std()) - 1) < 0.01 and abs(con[1]) < 0.01 and r2 > 0.99 and abs(slope - 2) < 0.05,
      f"σ {sd_g:.3f} / {sd_u:.3f} / {sd_r:.3f}; fit share {con[0]:.2f} and {con[1]:.4f}, R² {r2:.3f}")

# the OA-TOF pushed by imported pusher / puller waveforms: the ideal pulse as a trace gives the same flight times,
# and the SPICE pulse (with its ringing) still flies
tofi = TOF.OATof()
tp = np.linspace(0, 20e-6, 20001)
g = tofi.pulse.level(tp)
tofw = tofi.copy(v_push=0, v_pull=0, push_wave=(WF.Trace("p", tp, g), tofi.v_push), pull_wave=(WF.Trace("q", tp, g), tofi.v_pull))
sp1 = [IonSpecies("a", 500, 1, 1, 300)]
ra, rb = (TOF.fly(x, TOF.Beam(), sp1, rng=np.random.default_rng(1)) for x in (tofi, tofw))
ok = np.isfinite(ra.t) & np.isfinite(rb.t)
dt_rel = np.max(np.abs(rb.t[ok] / ra.t[ok] - 1))
tofs = tofi.copy(v_push=0, v_pull=0, push_wave=(push, 1.0), pull_wave=(spice["v(pull)"], 1.0))
Rs = TOF.resolution(tofs, TOF.Beam(), [IonSpecies("a", 500, 1, 1, 1000)])
check("TOF pushed by a waveform", ok.sum() > 250 and dt_rel < 1e-5 and Rs > 1000,
      f"flight times agree to {dt_rel:.1e}; SPICE pusher resolution {Rs:.0f}")

# the app window builds
from PySide6 import QtWidgets
from ionsim.gui.app import MainWindow
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
w = MainWindow()
check("app window", w.tabs.count() >= 8, f"{w.tabs.count()} tabs")

# Drivers tab: a scope capture through the dialog lands on an output, and the OA-TOF tab picks it up
from ionsim.gui import waveform_import as WI
from ionsim.drivers import tof_pulser
WI.SignalPicker.exec = lambda self: QtWidgets.QDialog.Accepted          # answer the dialogs without showing them
WI.ScopeDialog.exec = lambda self: QtWidgets.QDialog.Accepted
dr = w.drivers
name = dr.store.lib.add(tof_pulser("Scope pusher"), replace=False); dr.store.touch(); dr._fill_list(name)
fk = FakeRigol("DS1104Z", screen=1200, memory=1200, xinc=5e-9)
_dlg = WI.ScopeDialog.__init__
WI.ScopeDialog.__init__ = lambda self, parent=None: (_dlg(self, parent), self.addr.setCurrentText(f"127.0.0.1:{fk.port}"))[0]
_pick = WI.SignalPicker.__init__
WI.SignalPicker.__init__ = lambda self, *a, **k: (_pick(self, *a, **k), self.output.setCurrentText("Pusher"))[0]
dr._capture_scope()
drv = dr.store.get(name)
drv.output("Pusher").pulse = 0.0
w.tof.src.setCurrentText(name)
tof_m, _ = w.tof._model()
check("drive from a scope capture", drv.output("Pusher").wave == "DS1104Z CHAN1" and tof_m.push_wave is not None
      and abs(tof_m.on_levels()[0] - 600) < 15, f"pusher on at {tof_m.on_levels()[0]:.0f} V from {drv.waves[0]}")
from ionsim.gui.workspace import Workspace, TEMPLATES
ws = Workspace(models=[(t, None) for t in TEMPLATES], persist=False)
check("workspace", len(ws.models) == len(TEMPLATES) and ws.tabs.count() == len(TEMPLATES) + 4
      and ws.tabs.currentWidget() is ws.home, f"{len(ws.models)} model tabs, opens on Home")
# stages: an OA-TOF model takes its beam and ions from the ions that left a quadrupole model
from ionsim.stages import ExitIons
from ionsim.integrator import Result
from ionsim.ions import IonSpecies
rng = np.random.default_rng(1)
n = 400
vz = np.sqrt(2 * 5.0 * 1.602176634e-19 / (500 * 1.66053906660e-27))       # 5 eV, m/z 500
fake = Result(status=np.r_[np.ones(n // 2, int), 2 * np.ones(n // 2, int)], t_end=np.ones(n),
              final_pos=np.c_[1e-4 * rng.standard_normal((n, 2)), np.zeros(n)],
              final_vel=np.c_[10 * rng.standard_normal((n, 2)), np.full(n, vz)])
fake.species = np.r_[np.zeros(n // 4, int), np.ones(n // 4, int), np.zeros(n // 4, int), np.ones(n // 4, int)]
ex = ExitIons.from_result("test", fake, [IonSpecies("a", 500, 1, 1, n // 2), IonSpecies("b", 500, 1, 1, n // 2)])
beam, sp, _ = ex.tof_beam(10.0)
check("stage handoff", ex.n == n // 2 and abs(beam["e"] - 15.0) < 0.01 and abs(beam["sx"] - 0.1) < 0.02
      and 0 < beam["div"] < 1 and [s.n for s in sp] == [2000, 2000],
      f"{ex.n} exit ions -> {beam['e']:.2f} eV, {beam['sx']:.3f} mm, {beam['div']:.2f}°")
ws2 = Workspace(models=[("Quadrupole mass filter", "Q1"), ("OA-TOF", "TOF", "Q1")], persist=False)
q1, tof = ws2.models
q1.ctl.set_exit(ex)
check("stage link", ws2.source_of(tof) is q1 and abs(tof.ctl.beam.values()["e"] - 15.0) < 0.01
      and "Q1" in ws2.tabs.tabText(ws2.tabs.indexOf(tof.ctl)), ws2.tabs.tabText(ws2.tabs.indexOf(tof.ctl)))
# collision cell: hard-sphere gas thermalises ions and gives the K0 drift it was built from
from ionsim.collisions import HardSphereGas, AxialField, ccs_from_k0, GASES
from ionsim.integrator import Simulation
from ionsim.ions import Ion
from ionsim.constants import AMU, E_CHARGE, K_B
n_hs, m_hs, T_hs = 200, 500 * AMU, 300.0
sig = ccs_from_k0(np.full(n_hs, 1.0), 500, 1, GASES["N₂"], T_hs)
gas = HardSphereGas(0.5, T_hs, GASES["N₂"], sig)
rate = gas.n * sig[0] * 400
ions = Ion(np.zeros((n_hs, 3)), np.sqrt(K_B * T_hs / m_hs) * rng.standard_normal((n_hs, 3)), np.full(n_hs, m_hs),
           np.full(n_hs, E_CHARGE))
vz_hs, ke_hs = [], []
def _rec(t, dt, ions_, s, rng_):
    if t > 500 / rate:
        vz_hs.append(ions_.vel[:, 2].mean()); ke_hs.append((0.5 * m_hs * (ions_.vel[:, :2] ** 2).sum(1)).mean())
Simulation([AxialField(150.0)], z_end=1e9, hooks=[gas, _rec]).run(ions, 0.05 / rate, 2000 / rate, rng=rng)
K_hs = 1e-4 * 760 / 0.5 * T_hs / 273.15
check("collision cell gas", abs(np.mean(ke_hs) / K_B / T_hs - 1) < 0.05 and abs(np.mean(vz_hs) / (K_hs * 150) - 1) < 0.15,
      f"T {np.mean(ke_hs) / K_B:.0f} K vs {T_hs:.0f} K, drift {np.mean(vz_hs):.2f} vs K E {K_hs * 150:.2f} m/s")

# the full chain: inlet gas into a funnel, funnel ions into a quadrupole, quadrupole ions into the TOF
from ionsim.stages import ExitGas
ws3 = Workspace(models=[("ESI inlet capillary", "In"), ("Ion funnel", "F", "In"), ("Quadrupole mass filter", "Q", "F"),
                        ("OA-TOF", "T", "Q")], persist=False)
inl3, f3, q3, t3 = ws3.models
inl3.ctl.exit_gas = ExitGas("test", 0.8, 1.5, 298, [IonSpecies("a", 200, 1, 1.4, 1), IonSpecies("b", 500, 1, 1.0, 1)],
                            [1600.0, 1500.0])
inl3.ctl.exit_changed.emit()
f3.ctl.set_exit(ex)
idx_e, ions_e, _ = ex.entry_ions(5.0, rng=rng)
check("stage chain", [ws3.source_of(m).name if ws3.source_of(m) else None for m in ws3.models] == [None, "In", "F", "Q"]
      and f3.ctl.gas.values()["p"] == 1.5 and len(f3.ctl.ion_table.species()) == 2
      and len(q3.ctl.ion_table.species()) == 2 and len(idx_e) == ex.n and abs(ions_e.pos[:, 2]).max() == 0,
      " → ".join(m.name for m in ws3.models))
# setup files: every setting and link comes back
from ionsim.gui import settings_io
q3.ctl.gas.set("p", 0.02); q3.ctl.gas_kind.setCurrentText("Ar"); t3.ctl.tube.set("L", 777.0)
f3.ctl.ion_table.set_species([IonSpecies("x", 321, 2, 0.9, 50)])
setup = os.path.join(tempfile.mkdtemp(), "test.ionsetup")
ws3._write(setup, os.path.dirname(setup))
ws4 = Workspace(models=[], persist=False)
notes = ws4.apply_setup(ws4._read(setup), os.path.dirname(setup))
same = all(settings_io.save(a.ctl.scroll.widget()) == settings_io.save(b.ctl.scroll.widget())
           for a, b in zip(ws3.models, ws4.models))
check("setup file", same and not notes and [ws4.source_of(m).name if ws4.source_of(m) else None for m in ws4.models]
      == [None, "In", "F", "Q"], f"{len(ws4.models)} models, {os.path.getsize(setup) // 1024} kB")
# sweeps tab: vary a voltage of a model and a driver value, run, and put them back
from ionsim.gui.sweep_tab import Knob, driver_knobs
ws6 = Workspace(models=[("ESI inlet capillary", "In")], persist=False)
sw6 = ws6.sweeps
key6 = "First vacuum stage / Capillary − skimmer voltage"
w6 = settings_io.inputs(ws6.models[0].ctl.scroll.widget())[key6]
v6 = w6.value()
sw6.add_knob(Knob("input", "In", key6, "In: capillary − skimmer"))
dk = driver_knobs(dr.store.get(name))[0]
d_before = sw6.get(dk)
sw6.set(dk, d_before + 1.0); d_moved = sw6.get(dk); sw6.set(dk, d_before)
sw6.table.item(0, 2).setText(f"{v6 / 2:g}"); sw6.table.item(0, 3).setText(f"{v6 * 2:g}")
sw6.points.setValue(2); sw6.renominal.setChecked(False)
sw6.run()
t_sw = time.time()
while sw6._busy and time.time() - t_sw < 300:
    app.processEvents(); time.sleep(0.01)
e6 = sw6.column("In: m/z 500 energy at skimmer (eV)")
check("sweeps tab", len(sw6.data["results"]) == 3 and e6[1] < e6[0] < e6[2] and w6.value() == v6
      and d_moved == d_before + 1.0 and sw6.get(dk) == d_before and len(sw6.last_table[1]) == 3,
      f"energy at the skimmer {e6[1]:.2f} / {e6[0]:.2f} / {e6[2]:.2f} eV for {v6 / 2:g} / {v6:g} / {v6 * 2:g} V")
# instrument tab: tag the parts of one STEP, get linked models in order along the ion path
ws5 = Workspace(models=[], persist=False)
it = ws5.instrument
it.load(os.path.join(os.path.dirname(__file__), "..", "examples", "geometry", "instrument.step"))
for parts, kind, stage in ((["solid 1"], "ESI inlet capillary", ""), (["solid 3", "solid 4"], "Ion funnel", "Planar funnel"),
                           (["solid 6", "solid 7", "solid 8", "solid 9"], "Quadrupole mass filter", "Cell"),
                           (["solid 10", "solid 11"], "OA-TOF", "")):
    it.selected = parts; it.kind.setCurrentText(kind); it.stage.setText(stage); it._tag()
it._make()
check("instrument tab", len(it.bodies) == 11 and [m.name for m in ws5.models] == ["Inlet", "Planar funnel", "Cell", "OA-TOF"]
      and [ws5.source_of(m).name if ws5.source_of(m) else None for m in ws5.models] == [None, "Inlet", "Planar funnel", "Cell"],
      " → ".join(f"{m.name} {m.region[0]:.0f}-{m.region[1]:.0f} mm" for m in ws5.models))
# 3D path view: a ring (axisymmetric) and a box (3D grid) come out as faces in the right place
from matplotlib.figure import Figure
from ionsim.axisym import AxisymGrid
from ionsim.grid3d import Grid3D
from ionsim.gui.traj3d import voxels, faces, draw_traj3d
ga = AxisymGrid(5e-3, 0, 10e-3, 0.1e-3); ga.add_annulus("ring", 2e-3, 3e-3, 4e-3, 6e-3)
va = faces(*voxels(ga))[0]
rr = np.hypot(*va.mean(1)[:, 1:].T)                        # face centres
gb = Grid3D((-3e-3, 3e-3, -3e-3, 3e-3, 0, 10e-3), 0.1e-3); gb.add_box("box", 1e-3, 2e-3, -1e-3, 1e-3, 2e-3, 8e-3)
vb = faces(*voxels(gb))[0]
fig3 = Figure()
path3 = np.zeros((5, 2, 3)); path3[:, :, 2] = np.linspace(0, 9e-3, 5)[:, None]
ax3 = draw_traj3d(fig3, path3, [("b", "-")] * 2, gb)
check("3D path view", 1.6 < rr.min() < 2.2 and 2.8 < rr.max() < 3.5 and vb[..., 1].min() > 0.85 and vb[..., 1].max() < 2.2
      and vb[..., 0].min() > 1.85 and vb[..., 0].max() < 8.15 and len(ax3.collections) == 1,
      f"ring r {rr.min():.2f}-{rr.max():.2f} mm, box x {vb[..., 1].min():.2f}-{vb[..., 1].max():.2f} mm")
# inlet flow picture: the capillary profile carries the 1D mean flow, the jet is zero outside its barrel,
# and the moving picture moves gas and ions on
from ionsim.inlet import Inlet, fly_inlet, DEFAULT_IONS
from ionsim.gui.flow_movie import FlowMovie
inl_f = Inlet()
a_f, L_f = inl_f.diameter / 2, inl_f.length
rr_f = np.linspace(0, a_f, 2001)
ux_f, _ = inl_f.velocity_2d(np.full_like(rr_f, L_f / 2), rr_f)
mean_f = np.trapezoid(2 * rr_f * ux_f, rr_f) / a_f ** 2
u1d_f = np.interp(L_f / 2, inl_f.cap["x"], inl_f.cap["u"])
xi_f = 1e-3
on_f = inl_f.velocity_2d(L_f + xi_f, 0.0)[0]
off_f = np.hypot(*inl_f.velocity_2d(L_f + xi_f, 2 * inl_f.jet_radius(xi_f)))
movie = FlowMovie(Figure())
movie.show(inl_f, fly_inlet(inl_f, DEFAULT_IONS[:2]))
g0 = movie.panels[1].gx.copy()
movie.step(0.5)
ion_x = movie.ion_dots[1].get_offsets()[:, 0]
check("inlet flow picture", abs(mean_f / u1d_f - 1) < 0.01 and abs(on_f / inl_f.jet_centre(np.array([xi_f]))[2][0] - 1) < 1e-6
      and off_f == 0 and np.mean(movie.panels[1].gx > g0) > 0.5 and np.all(np.isfinite(ion_x)) and np.all(ion_x > 0),
      f"capillary mean {mean_f:.1f} vs 1D {u1d_f:.1f} m/s; jet axis {float(on_f):.0f} m/s at 1 mm")
# speeds along the instrument: ion speed from the recorded paths, stages laid end to end
from ionsim.gui.speeds import ion_speeds
from types import SimpleNamespace
tt = np.linspace(0, 1e-3, 101)
fake = SimpleNamespace(traj=np.zeros((101, 4, 3)), traj_t=tt, species=np.array([0, 0, 1, 1]))
fake.traj[:, :2, 2] = 50.0 * tt[:, None]; fake.traj[:, 2:, 2] = 20.0 * tt[:, None]
spd = ion_speeds(fake, [IonSpecies("a", 100, 1, 1, 2), IonSpecies("b", 200, 1, 1, 2)], 0, 0.05, nbins=10)
for m_, span in zip(ws3.models, ((-5e-3, 63e-3), (0.0, 0.1), (0.0, 0.2), None)):
    m_.ctl.speeds = None if span is None else {"span": span, "gas": None, "ions": []}
rows_s = ws3.speed_rows()
check("speeds along the instrument", np.allclose(spd[0][2], 50) and np.allclose(spd[1][2], 20)
      and [r[0] for r in rows_s] == ["In", "F", "Q"] and abs(rows_s[1][2] - 0.073) < 1e-9 and abs(rows_s[2][2] - 0.178) < 1e-9,
      ", ".join(f"{r[0]} at {r[2] * 1e3:.0f} mm" for r in rows_s))
print(f"all good in {time.time() - t0:.0f} s")
