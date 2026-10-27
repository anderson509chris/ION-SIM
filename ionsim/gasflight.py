"""Ion flight through a buffer gas at ~0.1-50 Torr (Langevin dynamics).

Each ion feels the electric field, a drag force that relaxes its velocity to the gas
velocity u with time constant tau = m K / q (K = mobility at the gas density), and
random thermal kicks. Over one step the field is held constant and the
Ornstein-Uhlenbeck equations are integrated exactly:

    a = qE/m,  w = v - u - a tau,  c = exp(-dt/tau)
    v' = u + a tau + w c                  + noise_v
    x' = x + (u + a tau) dt + w tau (1-c) + noise_x

with the exact correlated Gaussian noise. E is sampled at the middle of the step (time
and predicted position), which keeps RF motion accurate when damping is weak. The step
is stable for any dt/tau, so the
same code covers RF inertia (tau ~ RF period: funnels at a few Torr) and the
overdamped mobility limit (tau << dt), where it reduces to dx = K E dt + sqrt(2 D dt) N.

A field object needs efield(t, pos) -> (N,3) V/m and hit(pos) -> (N,) bool.
"""
import numpy as np
from dataclasses import dataclass
from .constants import AMU, E_CHARGE, K_B


@dataclass
class GasIons:
    pos: np.ndarray       # (N,3) m
    vel: np.ndarray       # (N,3) m/s
    mass: np.ndarray      # (N,) kg
    charge: np.ndarray    # (N,) C
    K: np.ndarray         # (N,) mobility at the gas conditions, m^2/Vs
    t0: np.ndarray        # (N,) birth time, s

    @property
    def mz(self):
        return self.mass / AMU / (self.charge / E_CHARGE)


def disc_ions(n, mass_amu, K, charge_state=1, center=(0.0, 0.0), radius=1e-3, z0=0.0,
              temp_k=300.0, birth_window=0.0, rng=None):
    """Ions uniform in a disc of `radius` at z = z0, thermal velocities, birth times
    uniform in [0, birth_window). mass_amu and K may be scalars or (n,) arrays."""
    rng = rng or np.random.default_rng()
    mass = np.broadcast_to(np.asarray(mass_amu, float), (n,)) * AMU
    K = np.broadcast_to(np.asarray(K, float), (n,)).copy()
    r = radius * np.sqrt(rng.random(n))
    th = 2 * np.pi * rng.random(n)
    pos = np.column_stack([center[0] + r * np.cos(th), center[1] + r * np.sin(th), np.full(n, z0)])
    vel = np.sqrt(K_B * temp_k / mass)[:, None] * rng.standard_normal((n, 3))
    return GasIons(pos, vel, mass.copy(), np.full(n, charge_state * E_CHARGE), K, birth_window * rng.random(n))


@dataclass
class FlightResult:
    status: np.ndarray        # -1 not born, 0 still flying at t_max, 1 transmitted (passed z_end), 2 lost
    t_end: np.ndarray
    final_pos: np.ndarray
    traj_t: np.ndarray = None
    traj: np.ndarray = None   # (n_samples, n_record, 3), NaN after an ion stops

    @property
    def transmission(self):
        return float(np.mean(self.status == 1))


def fly(field, ions, dt, t_max, z_end, temp_k=300.0, gas_velocity=0.0, record=0, record_every=10,
        rng=None, progress=None):
    """Fly GasIons until they pass z = z_end, hit something, or t_max.
    gas_velocity: uniform gas speed along +z (m/s). record: number of ions whose paths are kept.
    progress: optional callable(fraction) -> True to stop early."""
    rng = rng or np.random.default_rng()
    pos, vel = ions.pos.copy(), ions.vel.copy()
    n = len(pos)
    tau = ions.mass * ions.K / ions.charge
    qm = ions.charge / ions.mass
    c = np.exp(-dt / tau)
    one_c = -np.expm1(-dt / tau)
    kt_m = K_B * temp_k / ions.mass
    g = 1 / tau
    var_v = kt_m * (1 - c * c)
    var_x = np.maximum(kt_m / g ** 2 * (2 * g * dt - 3 + 4 * c - c * c), 0.0)
    cov = kt_m / g * one_c ** 2
    sv = np.sqrt(var_v)
    kx1 = np.divide(cov, sv, out=np.zeros(n), where=sv > 0)
    kx2 = np.sqrt(np.maximum(var_x - kx1 ** 2, 0.0))
    u = np.array([0.0, 0.0, gas_velocity])

    status = np.where(ions.t0 > 0, -1, 0).astype(np.int8)
    t_end = np.full(n, np.nan)
    rec = np.arange(min(record, n))
    traj, traj_t = [], []
    t, step = 0.0, 0
    while t < t_max and np.any(status <= 0):
        status[(status == -1) & (ions.t0 <= t)] = 0
        s = np.nonzero(status == 0)[0]
        if len(s):
            p, v = pos[s], vel[s]
            # field sampled at the step midpoint (time and predicted position): second order
            # when gas damping is weak (vacuum-like RF devices), unchanged when overdamped
            atau = (qm[s] * tau[s])[:, None] * field.efield(t + 0.5 * dt, p + (0.5 * dt) * v)
            w = v - u - atau
            x1, x2 = rng.standard_normal((len(s), 3)), rng.standard_normal((len(s), 3))
            pos[s] = p + (u + atau) * dt + w * (tau[s] * one_c[s])[:, None] + kx1[s, None] * x1 + kx2[s, None] * x2
            vel[s] = u + atau + w * c[s, None] + sv[s, None] * x1
            hit = field.hit(pos[s])
            done = pos[s, 2] >= z_end
            status[s[hit & ~done]] = 2
            status[s[done]] = 1
            t_end[s[hit | done]] = t + dt
        t += dt; step += 1
        if len(rec) and step % record_every == 0:
            snap = pos[rec].copy()
            snap[status[rec] != 0] = np.nan
            just = np.isclose(t_end[rec], t)
            snap[just] = pos[rec][just]
            traj.append(snap); traj_t.append(t)
        if progress and step % 200 == 0 and progress(max(t / t_max, np.mean(status > 0))):
            break
    res = FlightResult(status, t_end, pos)
    if len(rec):
        res.traj_t, res.traj = np.array(traj_t), np.array(traj)
    return res
