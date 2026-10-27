"""Ion funnels at ~1-10 Torr: RF confinement + DC drive through a narrowing electrode stack.

Three geometries share the same electrics and flight:
  * RingFunnel   - the standard PNNL-style funnel: a stack of PCB rings (a copper ring
                   around the bore on an FR4 board, or all-metal plates) whose inner
                   diameter shrinks toward a conductance-limit exit plate. Solved on an
                   axisymmetric r-z grid (fast, exact symmetry).
  * PlanarFunnel - two parallel boards carrying electrode strips across the ion path, on
                   an FR4 substrate. The strips shorten toward the exit, and DC guard
                   electrodes beyond the strip ends squeeze the ions sideways. Solved in 3D.
  * CadFunnel    - any electrodes imported from STEP / STL / OBJ (cad.py); bodies can be
                   marked as insulators with a permittivity. Solved in 3D.

Electrics: adjacent electrodes carry RF 180 deg apart (amplitude rf, zero-to-peak) and a
DC ladder that falls by `gradient` V/cm toward the exit, ending at `dc_exit` on the last
electrode. The exit plate sits `exit_offset` V below the last electrode.

Ions are flown with Langevin dynamics (gasflight.py): RF inertia, gas drag and diffusion
together, which is what produces the funnel's radial focusing and its low-m/z cutoff.
Optional: a gas flow field from the gas throughput (gasflow.py) instead of a uniform
speed, and the beam's own space charge for a given ion current (spacecharge.py).
"""
import numpy as np
from dataclasses import dataclass, field as dc_field
from .axisym import AxisymGrid, AxisymField
from .grid3d import Grid3D, GridField3D
from .laplace import Drive
from .gasflight import disc_ions, fly
from .gasflow import GasFlow, volumetric_flow
from .gridtools import GridMap
from .spacecharge import fly_with_space_charge
from .twave import mobility
from .constants import AMU, E_CHARGE


@dataclass
class FunnelElectrics:
    rf: float = 50.0            # V zero-to-peak (100 Vpp)
    freq: float = 1.0e6         # Hz
    gradient: float = 20.0      # V/cm, DC falling toward the exit
    dc_exit: float = 20.0       # V on the last funnel electrode
    exit_offset: float = 5.0    # exit plate is this many V below the last electrode
    guard_offset: float = 3.0   # planar funnel: guards this many V above the local strip DC
                                # (too high and the DC minimum moves toward the boards: two lanes)

    def ladder(self, z, z_last):
        return self.dc_exit + self.gradient * 100 * (z_last - z)


@dataclass
class RingFunnel:
    n_rings: int = 100
    pitch: float = 1.0e-3           # ring-to-ring spacing
    thickness: float = 0.5e-3       # ring (PCB copper + board) thickness
    id_in: float = 25.4e-3          # entrance inner diameter
    id_out: float = 2.5e-3          # last ring inner diameter
    n_straight: int = 10            # constant-ID rings at the entrance before the taper
    od: float = 45.0e-3             # plate outer diameter
    exit_id: float = 2.5e-3         # conductance-limit orifice diameter
    exit_gap: float = 1.0e-3        # last ring to exit plate (centre to front face)
    exit_thickness: float = 0.5e-3
    h: float = 0.05e-3              # grid spacing
    copper_width: float = 2.0e-3    # copper ring around the bore (plated through the bore,
                                    # both faces); FR4 beyond it. 0 = all-metal plates
    board_eps: float = 4.4          # board relative permittivity (FR4 ~4.4, PTFE 2.1, alumina 9.8)

    def ring_ids(self):
        n_taper = max(self.n_rings - self.n_straight, 1)
        k = np.arange(self.n_rings) - self.n_straight
        frac = np.clip((k + 1) / n_taper, 0, 1)
        return self.id_in + (self.id_out - self.id_in) * frac

    def ring_z(self):
        return self.pitch * np.arange(self.n_rings)

    @property
    def z_plate(self):
        return self.ring_z()[-1] + self.exit_gap

    @property
    def z_start(self):
        return 0.5 * self.pitch

    @property
    def z_end(self):
        return self.z_plate + self.exit_thickness + 0.5e-3

    def rings(self):
        """[(name, r_in, r_out, z0, z1)] for every electrode, exit plate last."""
        t = self.thickness / 2
        out = [(f"ring {k + 1}", d / 2, self._copper_edge(d), z - t, z + t) for k, (d, z) in enumerate(zip(self.ring_ids(), self.ring_z()))]
        out.append(("exit plate", self.exit_id / 2, self.od / 2, self.z_plate, self.z_plate + self.exit_thickness))
        return out

    def _copper_edge(self, d):
        return min(d / 2 + self.copper_width, self.od / 2) if self.copper_width > 0 else self.od / 2

    def boards(self):
        """[(r_in, r_out, z0, z1)] insulating board beyond each copper ring."""
        t = self.thickness / 2
        return [(self._copper_edge(d), self.od / 2, z - t, z + t) for d, z in zip(self.ring_ids(), self.ring_z())
                if self._copper_edge(d) < self.od / 2]

    def build(self):
        g = AxisymGrid(self.od / 2 + 2e-3, -3e-3, self.z_end + 3e-3, self.h, open_entrance=True)
        for r0, r1, z0, z1 in self.boards():
            g.add_dielectric_annulus(r0, r1, z0, z1, self.board_eps)
        for name, r0, r1, z0, z1 in self.rings():
            g.add_annulus(name, r0, r1, z0, z1)
        return g

    def drives(self, el):
        z = self.ring_z()
        d = {f"ring {k + 1}": Drive(el.ladder(zk, z[-1]), el.rf, 180.0 * (k % 2)) for k, zk in enumerate(z)}
        d["exit plate"] = Drive(el.dc_exit - el.exit_offset)
        return d

    def field(self, el, grid=None, solver=None, guess=None):
        """Solve the fields. Pass the grid and solver from an earlier call to skip the
        matrix setup when only voltages changed (and its phi as guess to warm-start)."""
        g = grid or self.build()
        return AxisymField(g, self.drives(el), el.freq, solver, guess)


@dataclass
class PlanarFunnel:
    n_strips: int = 60
    pitch: float = 1.0e-3
    strip_width: float = 0.5e-3     # along the ion path
    gap: float = 4.0e-3             # board-to-board spacing (y)
    width_in: float = 20.0e-3       # strip length at the entrance (x)
    width_out: float = 2.0e-3       # strip length at the exit
    n_straight: int = 5
    guard_gap: float = 0.5e-3       # strip end to guard
    guard_band: float = 3.0e-3      # guard electrode width beyond the widest strip
    exit_width: float = 2.0e-3      # exit slit (x)
    exit_height: float = 3.0e-3     # exit slit (y): ions fill most of the gap, so keep it tall
    exit_gap: float = 1.0e-3
    exit_thickness: float = 0.5e-3
    h: float = 0.2e-3
    board_thickness: float = 1.6e-3 # FR4 substrate behind the strips (0 = strips in free space)
    board_eps: float = 4.4

    def strip_lengths(self):
        n_taper = max(self.n_strips - self.n_straight, 1)
        k = np.arange(self.n_strips) - self.n_straight
        return self.width_in + (self.width_out - self.width_in) * np.clip((k + 1) / n_taper, 0, 1)

    def strip_z(self):
        return self.pitch * np.arange(self.n_strips)

    @property
    def z_plate(self):
        return self.strip_z()[-1] + self.exit_gap

    @property
    def z_start(self):
        return 0.5 * self.pitch

    @property
    def z_end(self):
        return self.z_plate + self.exit_thickness + 0.5e-3

    @property
    def x_edge(self):
        return self.width_in / 2 + self.guard_gap + self.guard_band

    def build(self):
        xe, yb, h = self.x_edge + 1e-3, self.gap / 2, self.h
        ye = yb + self.board_thickness + 2 * h
        g = Grid3D((-xe, xe, -ye, ye, -2e-3, self.z_end + 2e-3), h, open_faces=("y", "z-"))
        if self.board_thickness > 0:
            zb0, zb1 = self.strip_z()[0] - self.pitch, self.strip_z()[-1] + self.pitch / 2
            for s in (-1, 1):
                y0, y1 = sorted((s * (yb + h), s * (yb + self.board_thickness)))
                g.add_dielectric_box(-xe, xe, y0, y1, zb0, zb1, self.board_eps)
        w = self.strip_width / 2
        for k, (L, z) in enumerate(zip(self.strip_lengths(), self.strip_z())):
            for y in (-yb, yb):
                g.add_box(f"strip {k + 1}", -L / 2, L / 2, y, y, z - w, z + w)
                g.add_box(f"guard {k + 1}", L / 2 + self.guard_gap, self.x_edge, y, y, z - w, z + w)
                g.add_box(f"guard {k + 1}", -self.x_edge, -L / 2 - self.guard_gap, y, y, z - w, z + w)
        z0, z1 = self.z_plate, self.z_plate + self.exit_thickness
        ex, ey = self.exit_width / 2, self.exit_height / 2
        g.add_box("exit plate", -xe, -ex - h, -ye, ye, z0, z1)
        g.add_box("exit plate", ex + h, xe, -ye, ye, z0, z1)
        g.add_box("exit plate", -ex, ex, ey + h, ye, z0, z1)
        g.add_box("exit plate", -ex, ex, -ye, -ey - h, z0, z1)
        return g

    def drives(self, el):
        z = self.strip_z()
        d = {}
        for k, zk in enumerate(z):
            dc = el.ladder(zk, z[-1])
            d[f"strip {k + 1}"] = Drive(dc, el.rf, 180.0 * (k % 2))
            d[f"guard {k + 1}"] = Drive(dc + el.guard_offset)
        d["exit plate"] = Drive(el.dc_exit - el.exit_offset)
        return d

    def field(self, el, grid=None, solver=None, guess=None):
        g = grid or self.build()
        return GridField3D(g, self.drives(el), el.freq, solver, guess)


@dataclass
class CadFunnel:
    bodies: list                    # [(name, triangles (M,3,3) m)], ion axis along z
    h: float = 0.25e-3
    pad: float = 3.0e-3             # free space around the parts (grounded box beyond)
    open_entrance: bool = True      # the z-min face is open (ions come from an inlet), not grounded
    dielectrics: dict = dc_field(default_factory=dict)   # body name -> relative permittivity (insulators)

    def bounds(self):
        pts = np.concatenate([t.reshape(-1, 3) for _, t in self.bodies])
        lo, hi = pts.min(0) - self.pad, pts.max(0) + self.pad
        return lo, hi

    def nodes(self):
        lo, hi = self.bounds()
        return int(np.prod(np.round((hi - lo) / self.h) + 1))

    def build(self):
        lo, hi = self.bounds()
        g = Grid3D((lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]), self.h, open_faces=("z-",) if self.open_entrance else ())
        for name, tris in self.bodies:
            if name in self.dielectrics:
                g.add_dielectric_mesh(tris, self.dielectrics[name])
        for name, tris in self.bodies:
            if name not in self.dielectrics:
                g.add_mesh(name, tris)
        return g

    def body_z(self):
        return {name: float(t[:, :, 2].mean()) for name, t in self.bodies}

    def funnel_pattern(self, el, last_is_exit=False):
        """Drives for a stack of electrodes along z: alternating RF phase and the DC ladder,
        in order of position. With last_is_exit the last body is a DC-only exit plate.
        A useful starting point to edit."""
        bz = {n: z for n, z in self.body_z().items() if n not in self.dielectrics}
        order = sorted(bz, key=bz.get)
        stack = order[:-1] if last_is_exit and len(order) > 1 else order
        z_last = bz[stack[-1]]
        d = {name: Drive(el.ladder(bz[name], z_last), el.rf, 180.0 * (k % 2)) for k, name in enumerate(stack)}
        if len(stack) < len(order):
            d[order[-1]] = Drive(el.dc_exit - el.exit_offset)
        return d

    def field(self, drives, freq, grid=None, solver=None, guess=None):
        g = grid or self.build()
        return GridField3D(g, drives, freq, solver, guess)


def pseudopotential(e2, mz, charge_state, K, freq):
    """Effective RF potential (V) for mean square RF field e2 = <E^2>, including gas damping:
    U = q <E^2> / (2 m (w^2 + 1/tau^2))."""
    m, q = mz * charge_state * AMU, charge_state * E_CHARGE
    w, tau = 2 * np.pi * freq, m * K / q
    return q * e2 / (2 * m * (w ** 2 + tau ** -2))


def gas_flow(field, slm, pressure_torr, temp_k=300.0, inlet_radius=5e-3, center=(0.0, 0.0)):
    """Flow field for a gas throughput in standard litres per minute entering through a disc
    of inlet_radius at the entrance of the field's grid."""
    return GasFlow(field.grid, volumetric_flow(slm, pressure_torr, temp_k), inlet_radius, center)


def run_funnel(field, z_start, z_end, mz, K0, n, pressure_torr, temp_k=300.0, charge_state=1,
               start_radius=5e-3, center=(0.0, 0.0), steps_per_cycle=30, t_max=None, gradient=20.0,
               gas_velocity=0.0, record=0, rng=None, progress=None, current=0.0, sc_iterations=4):
    """Fly n ions per m/z value. mz and K0 may be scalars or equal-length lists (mass scan).
    gas_velocity: uniform axial speed (m/s) or a GasFlow. current: ion beam current (A); when
    non-zero the beam's space charge is iterated (sc_iterations passes, see spacecharge.py)
    and the result carries .space_charge and .sc_history.
    Returns (FlightResult, per-ion m/z)."""
    mz = np.atleast_1d(np.asarray(mz, float))
    K0 = np.broadcast_to(np.atleast_1d(np.asarray(K0, float)), mz.shape)
    mz_ion = np.repeat(mz, n)
    K = np.repeat(mobility(K0, pressure_torr, temp_k), n)
    freq = field.freq
    rng = rng or np.random.default_rng()
    if t_max is None:
        u = 0.0 if hasattr(gas_velocity, "velocity") else gas_velocity   # a flow field mostly leaves sideways
        v = K.min() * max(gradient, 1.0) * 100 + abs(u)
        t_max = float(np.clip(3 * (z_end - z_start) / v, 50 / freq, 0.05))

    def fly_once(fld, dwell=None, dwell_every=5, prog=progress):
        ions = disc_ions(len(mz_ion), mz_ion * charge_state, K, charge_state, center, start_radius, z_start,
                         temp_k, birth_window=1 / freq, rng=rng)
        return fly(fld, ions, 1 / (freq * steps_per_cycle), t_max, z_end, temp_k, gas_velocity,
                   record=record, record_every=max(1, steps_per_cycle // 3), rng=rng, progress=prog,
                   dwell=dwell, dwell_every=dwell_every)

    if current:
        res, _, _ = fly_with_space_charge(fly_once, field, GridMap.of(field.grid), field.solver, current,
                                          sc_iterations, progress=progress)
    else:
        res = fly_once(field)
    return res, mz_ion
