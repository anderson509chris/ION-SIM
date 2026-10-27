"""Gas flow through a device, solved on the electrostatic grid.

Viscous, incompressible flow in the lubrication (local Poiseuille) approximation:

    u = -(k / mu) grad p,   div(k grad p) = 0,   k = c delta (2 L - delta)

where delta is the distance to the nearest wall (electrode or insulator) and L the local
channel half-width (delta on the channel's centre ridge, found by walking uphill in
delta). That is the Poiseuille profile: c = 1/2 in a slit (between boards or plates),
c = 1/4 in a round bore (axisymmetric grids, where the ridge is the axis), so both the
profile and the conductance (w^3/12mu per width, pi R^4/8mu) are exact for straight
channels. The flow therefore splits correctly between the funnel bore and the narrow
gaps between electrodes, which pass far less gas than their area suggests, and the speed
falls to zero at the walls. 3D grids use c = 1/2 everywhere (round bores there carry up
to 2x too much conductance). p = 1 on the inlet, 0 on every outlet,
no flow through solids; the result is scaled so the inlet carries the given flow and mu
drops out. Gas enters through a disc on the entrance (z-min) face; every other open
boundary face is pumped.

Limits: no inertia, so a jet does not carry its momentum downstream (funnels run at
Reynolds numbers ~10-100), and no compressibility (fine well below the speed of sound;
the supersonic expansion from an inlet capillary is in inlet.py).
"""
import numpy as np
from scipy import ndimage
from .laplace import LaplaceSolver
from .gridtools import GridMap


def volumetric_flow(slm, pressure_torr, temp_k):
    """Standard litres per minute (0 °C, 1 atm) to m^3/s at the device pressure and temperature."""
    return slm * 1e-3 / 60 * (760.0 / pressure_torr) * (temp_k / 273.15)


def _links(ndim, ax):
    lo = tuple(slice(0, -1) if a == ax else slice(None) for a in range(ndim))
    hi = tuple(slice(1, None) if a == ax else slice(None) for a in range(ndim))
    return lo, hi


def _half_width(delta, gas, axis_ridge):
    """Local channel half-width: each node takes the delta of the ridge it reaches by
    stepping to its largest uphill neighbour. Also returns which nodes drain to the axis
    (round bore) when axis_ridge (axisymmetric grids: r index 0)."""
    L = delta.copy()
    tube = np.zeros(delta.shape, bool)
    if axis_ridge:
        tube[0] = gas[0]
    for _ in range(int(delta.max() / max(delta[delta > 0].min(), 1e-30)) + 2):
        best, bdelta, btube = L.copy(), delta.copy(), tube.copy()
        for ax in range(delta.ndim):
            lo, hi = _links(delta.ndim, ax)
            for a, b in ((lo, hi), (hi, lo)):          # node a looks at neighbour b
                up = gas[a] & (delta[b] > bdelta[a])
                bdelta[a] = np.where(up, delta[b], bdelta[a])
                best[a] = np.where(up, L[b], best[a])
                btube[a] = np.where(up, tube[b], btube[a])
        if np.array_equal(best, L) and np.array_equal(btube, tube):
            break
        L, tube = best, btube
    return L, tube


def _extend(psi, gas, sweeps=2):
    """Give solid nodes next to the gas the mean of their gas neighbours, so gradients taken
    at the wall have no normal component (no flow into the wall)."""
    psi = np.where(gas, psi, 0.0)
    known = gas.copy()
    for _ in range(sweeps):
        s, c = np.zeros_like(psi), np.zeros_like(psi)
        for ax in range(psi.ndim):
            lo, hi = _links(psi.ndim, ax)
            v = np.where(known, psi, 0.0)
            s[lo] += v[hi]; c[lo] += known[hi]
            s[hi] += v[lo]; c[hi] += known[lo]
        new = ~known & (c > 0)
        psi[new] = s[new] / c[new]
        known |= new
    return psi


class GasFlow:
    """Steady gas velocity field. velocity(pos) -> (N,3) m/s, for gasflight.fly."""

    def __init__(self, grid, flow, inlet_radius, center=(0.0, 0.0)):
        """grid: the field's AxisymGrid or Grid3D. flow: volumetric flow (m^3/s) entering
        through a disc of inlet_radius about center (x, y) on the z-min face."""
        gm = GridMap.of(grid)
        self.gmap, self.flow, self.center = gm, flow, center
        solid = grid.solid
        gas = ~solid
        inlet, outlet = np.zeros(solid.shape, bool), np.zeros(solid.shape, bool)
        if gm.kind == "axisym":
            inlet[:, 0] = grid.r <= inlet_radius
            outlet[-1, :] = outlet[:, -1] = True
            area = np.pi * inlet_radius ** 2
        else:
            X, Y = np.meshgrid(grid.x - center[0], grid.y - center[1], indexing="ij")
            inlet[:, :, 0] = np.hypot(X, Y) <= inlet_radius
            for ax in range(3):
                for end in (0, -1):
                    if (ax, end) != (2, 0):
                        outlet[tuple(end if a == ax else slice(None) for a in range(3))] = True
            area = inlet[:, :, 0].sum() * gm.h ** 2
        inlet &= gas
        outlet &= gas & ~inlet
        if not inlet.any():
            raise ValueError("Gas inlet is blocked: no open space within the inlet radius on the entrance face")
        delta = ndimage.distance_transform_edt(gas) * gm.h
        L, tube = _half_width(delta, gas, gm.kind == "axisym")
        k = np.where(tube, 0.25, 0.5) * delta * (2 * L - delta)       # local Poiseuille permeability
        lab, _ = ndimage.label(gas)
        dead = gas & ~np.isin(lab, np.unique(lab[inlet | outlet]))      # pockets the gas cannot reach
        fixed = solid | inlet | outlet | dead
        weights = []
        for ax, W in enumerate(grid.link_weights()):
            lo, hi = _links(gas.ndim, ax)
            both = gas[lo] & gas[hi]
            weights.append(np.broadcast_to(W, both.shape) * both * (0.5 * (k[lo] + k[hi])))
        solver = LaplaceSolver(fixed, weights)
        psi = solver.solve(inlet.astype(float))

        # fluxes through the inlet and the outlets (exit face = z max)
        free = psi.ravel()[~fixed.ravel()]
        inl, outl = inlet.ravel(), outlet.ravel()
        exit_face = np.zeros(solid.shape, bool); exit_face[..., -1] = True
        ef = exit_face.ravel()
        f_in = f_out = f_exit = 0.0
        for fi, fx, w in solver.couplings:
            m = inl[fx]
            f_in += np.sum(w[m] * (1.0 - free[fi[m]]))
            m = outl[fx]
            q = w[m] * free[fi[m]]
            f_out += q.sum(); f_exit += q[ef[fx[m]]].sum()
        self.exit_fraction = float(f_exit / f_out) if f_out > 0 else 0.0
        scale = flow / (f_in * gm.flux_scale())
        self.psi = _extend(psi, gas)
        self.table = gm.gradient_table(self.psi, -scale) * k[..., None].astype(np.float32)
        self.table[solid] = 0.0
        self.solid = solid
        self.inlet_speed = flow / area       # mean speed over the inlet disc

    def warning(self, temp_k=300.0, gas_mass_amu=28.0):
        """A note when the inlet speed is too high for an incompressible model, else ''."""
        c = np.sqrt(1.4 * 1.380649e-23 * temp_k / (gas_mass_amu * 1.66053907e-27))
        if self.inlet_speed > 0.3 * c:
            return (f"inlet speed is {self.inlet_speed / c:.1f}x the speed of sound: the incompressible model is rough here; "
                    "use a larger inlet radius or less throughput")
        return ""

    def velocity(self, pos):
        return self.gmap.vector(self.table, pos)

    def section(self, center=(0.0, 0.0)):
        """x-z plane through the axis (y = center[1]) for plots: coords, z, transverse and
        axial speed, solid mask."""
        gm, T = self.gmap, self.table
        if gm.kind == "axisym":
            n = T.shape[0]
            r = gm.h * np.arange(n)
            mir = lambda a, s=1: np.concatenate([s * a[:0:-1], a], axis=0)
            return {"coords": mir(r, -1), "z": gm.origin[1] + gm.h * np.arange(T.shape[1]),
                    "ut": mir(T[..., 0], -1), "uz": mir(T[..., 1]), "mask": mir(self.solid)}
        j = int(np.clip(round((center[1] - gm.origin[1]) / gm.h), 0, gm.shape[1] - 1))
        return {"coords": gm.origin[0] + gm.h * np.arange(gm.shape[0]), "z": gm.origin[2] + gm.h * np.arange(gm.shape[2]),
                "ut": T[:, j, :, 0], "uz": T[:, j, :, 2], "mask": self.solid[:, j, :]}
