"""Vectorised RK4 ion flight.

Extension points for the drag-field / collision-cell stage:
  fields : list of FieldSource      (e.g. add an axial drag-field source)
  forces : list of f(t, pos, vel, ions) -> acceleration (N,3)   (e.g. viscous drag)
  hooks  : list of h(t, dt, ions, active, rng) called after each step, may change
           ions.vel in place                                   (e.g. Monte Carlo gas collisions)
"""
import numpy as np
from dataclasses import dataclass, field


@dataclass
class Result:
    status: np.ndarray        # -1 = not born, 0 = still flying, 1 = transmitted (reached z_end), 2 = hit electrode
    t_end: np.ndarray
    final_pos: np.ndarray
    final_vel: np.ndarray
    traj_t: np.ndarray = None
    traj: np.ndarray = None   # (n_samples, N, 3), NaN after an ion stops

    @property
    def transmission(self):
        return float(np.mean(self.status == 1))


@dataclass
class Simulation:
    fields: list
    z_end: float
    forces: list = field(default_factory=list)
    hooks: list = field(default_factory=list)

    def accel(self, t, pos, vel, ions, sel):
        E = sum(f.efield(t, pos) for f in self.fields)
        a = (ions.charge[sel] / ions.mass[sel])[:, None] * E
        for f in self.forces:
            a = a + f(t, pos, vel, ions, sel)
        return a

    def run(self, ions, dt, t_max, record_every=0, rng=None, progress=None):
        """progress: optional callable(fraction) -> True to stop early (used by the GUI)."""
        rng = rng or np.random.default_rng()
        pos, vel = ions.pos.copy(), ions.vel.copy()
        n = len(pos)
        t0 = ions.t0 if ions.t0 is not None else np.zeros(n)
        status = np.where(t0 > 0, -1, 0).astype(np.int8)
        t_end = np.full(n, np.nan)
        traj, traj_t = [], []
        t, step = 0.0, 0
        while t < t_max and np.any(status <= 0):
            status[(status == -1) & (t0 <= t)] = 0
            s = np.nonzero(status == 0)[0]
            if len(s) == 0:
                t += dt; step += 1
                continue
            p, v = pos[s], vel[s]
            k1v = self.accel(t, p, v, ions, s);                    k1x = v
            k2v = self.accel(t + dt/2, p + dt/2*k1x, v + dt/2*k1v, ions, s); k2x = v + dt/2*k1v
            k3v = self.accel(t + dt/2, p + dt/2*k2x, v + dt/2*k2v, ions, s); k3x = v + dt/2*k2v
            k4v = self.accel(t + dt, p + dt*k3x, v + dt*k3v, ions, s);       k4x = v + dt*k3v
            pos[s] = p + dt/6 * (k1x + 2*k2x + 2*k3x + k4x)
            vel[s] = v + dt/6 * (k1v + 2*k2v + 2*k3v + k4v)
            t += dt; step += 1
            if progress and step % 200 == 0 and progress(max(t / t_max, np.mean(status > 0))):
                break
            for h in self.hooks:
                ions.pos, ions.vel = pos, vel
                h(t, dt, ions, s, rng)
            hit = np.zeros(len(s), dtype=bool)
            for f in self.fields:
                hit |= f.hit(pos[s])
            done = pos[s, 2] >= self.z_end
            status[s[hit]] = 2
            status[s[done & ~hit]] = 1
            t_end[s[hit | done]] = t
            if record_every and step % record_every == 0:
                snap = pos.copy()
                snap[status != 0] = np.nan
                snap[s[hit | done]] = pos[s[hit | done]]   # keep the final point
                traj.append(snap); traj_t.append(t)
        res = Result(status, t_end, pos, vel)
        if record_every:
            res.traj_t, res.traj = np.array(traj_t), np.array(traj)
        return res
