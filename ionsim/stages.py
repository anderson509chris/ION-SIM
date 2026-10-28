"""Stages: one model's exit ions become the next model's starting ions.

A model that flies ions (quadrupole, ion funnel, Device builder) keeps the ions that left
through its exit as an ExitIons. The next stage reads it: a quadrupole starts those same
ions at its entrance (entry_ions), and the OA-TOF turns them into the beam in its pusher
(energy, energy spread, size, divergence) and its ion mixture (tof_beam).

The ESI inlet hands on gas instead (ExitGas): the gas flow and pressure it delivers to the
first vacuum stage, and its ion species, which an ion funnel takes as its gas and ions.
"""
import numpy as np
from dataclasses import dataclass

from .constants import AMU, E_CHARGE
from .ions import IonSpecies, Ion


@dataclass
class ExitIons:
    """The ions that left a model through its exit (status 1), one row per ion."""
    source: str                 # what made them, e.g. "Ion funnel, Trajectories run"
    species: list               # IonSpecies of the run (n = ions flown)
    idx: np.ndarray             # species index of each exit ion
    pos: np.ndarray             # (N,3) m, where each ion left
    vel: np.ndarray             # (N,3) m/s
    t: np.ndarray               # (N,) s, when it left
    flown: np.ndarray           # ions flown per species
    center: tuple = (0.0, 0.0)  # the model's ion axis (x, y), m
    survival: np.ndarray = None # (N,) fraction intact after heating, None when heating was off
    tint_max: np.ndarray = None # (N,) hottest internal temperature on the way, K
    pressure_torr: float = 0.0  # gas at the exit (0 = vacuum)
    temp_k: float = 300.0

    @classmethod
    def from_result(cls, source, res, species, center=(0.0, 0.0), pressure_torr=0.0, temp_k=300.0):
        """res: an integrator.Result or gasflight.FlightResult with .species set."""
        idx_all = res.species if res.species is not None else np.zeros(len(res.status), int)
        out = res.status == 1
        vel = getattr(res, "final_vel", None)
        if vel is None:
            raise ValueError("this run did not keep the ions' velocities")
        heat = getattr(res, "heat", None)
        return cls(source, list(species), idx_all[out].astype(int), res.final_pos[out].copy(), vel[out].copy(),
                   res.t_end[out].copy(), np.bincount(idx_all, minlength=len(species)), tuple(center),
                   None if heat is None else heat.survival[out].copy(),
                   None if heat is None else heat.tint_max[out].copy(), pressure_torr, temp_k)

    @property
    def n(self):
        return len(self.idx)

    def weights(self):
        """Per ion: 1, or the fraction intact when heating was tracked."""
        return np.ones(self.n) if self.survival is None else np.clip(self.survival, 0, 1)

    def per_species(self):
        """Per species: (exit ions, intact ions, fraction of those flown that arrive intact)."""
        w = self.weights()
        cnt = np.bincount(self.idx, minlength=len(self.species))
        intact = np.bincount(self.idx, weights=w, minlength=len(self.species))
        return [(int(c), float(i), float(i / f) if f else 0.0) for c, i, f in zip(cnt, intact, self.flown)]

    def entry_ions(self, accel_v, birth_window=0.0, rng=None):
        """These ions at the entrance (z = 0, on the axis the next model uses) after a DC step of
        accel_v volts that pushes them along z. Ions that did not survive the heating are dropped
        (each with its chance of breaking up), and so are ions the step turns back.
        Returns (species index per ion, Ion, notes)."""
        rng = rng or np.random.default_rng()
        keep = rng.random(self.n) < self.weights()
        mz = np.array([s.mz for s in self.species])[self.idx]
        zc = np.abs(np.array([s.z for s in self.species], float))[self.idx]
        m, q = mz * zc * AMU, zc * E_CHARGE
        vel = self.vel.copy()
        ez = 0.5 * m * np.maximum(vel[:, 2], 0) ** 2 + q * accel_v
        through = ez > 0
        vel[:, 2] = np.sqrt(np.maximum(ez, 0) * 2 / m)
        k = keep & through
        pos = np.c_[self.pos[:, 0] - self.center[0], self.pos[:, 1] - self.center[1], np.zeros(self.n)]
        ions = Ion(pos[k], vel[k], m[k], q[k], birth_window * rng.random(int(k.sum())))
        notes = []
        if (~keep).any():
            notes.append(f"{int((~keep).sum())} broke up on the way (ion heating)")
        if (keep & ~through).any():
            notes.append(f"{int((keep & ~through).sum())} turned back by the {accel_v:g} V step")
        return self.idx[k], ions, notes

    def tof_beam(self, accel_v, n_ref=2000):
        """The OA-TOF pusher beam these ions make after a DC step of accel_v volts (per charge)
        between the exit and the pusher. Returns (beam settings dict in the TOF tab's units,
        species list with ion counts scaled so the most abundant species has n_ref, notes)."""
        if self.n == 0:
            raise ValueError(f"no ions left {self.source.split(',')[0]}: nothing to hand on")
        mz = np.array([s.mz for s in self.species])[self.idx]
        zc = np.abs(np.array([s.z for s in self.species], float))[self.idx]
        m = mz * zc * AMU
        q = zc * E_CHARGE
        w = self.weights()
        vx, vy, vz = self.vel.T
        e_exit = 0.5 * m * vz ** 2 / q                       # axial energy per charge at the exit, eV
        e = np.maximum(e_exit + accel_v, 1e-3)
        vz_p = np.sqrt(2 * q * e / m)                        # axial speed in the pusher (the side speed is kept)
        ang = np.degrees(np.concatenate([np.arctan2(vx, vz_p), np.arctan2(vy, vz_p)]))
        x, y = self.pos[:, 0] - self.center[0], self.pos[:, 1] - self.center[1]

        def wstd(a, ww):
            mu = np.average(a, weights=ww)
            return float(np.sqrt(np.average((a - mu) ** 2, weights=ww)))

        w2 = np.concatenate([w, w])
        beam = {"e": float(np.average(e, weights=w)), "de": wstd(e, w) if self.n > 1 else 0.0,
                "sx": wstd(x, w) * 1e3 if self.n > 1 else 0.0, "sy": wstd(y, w) * 1e3 if self.n > 1 else 0.0,
                "div": wstd(ang, w2) if self.n > 1 else 0.0, "temp": 0.0}
        rows = self.per_species()
        top = max(i for _, i, _ in rows)
        species = [IonSpecies(s.name, s.mz, abs(int(s.z)), s.K0, int(round(n_ref * i / top)) if top > 0 else 0)
                   for s, (_, i, _) in zip(self.species, rows)]
        notes = []
        for s, (c, i, f), fl in zip(self.species, rows, self.flown):
            txt = f"{s.name}: {c} of {fl} left"
            if self.survival is not None:
                txt += f", {i:.0f} intact"
            notes.append(txt + f" ({f:.0%})")
        if self.pressure_torr > 0:
            notes.append(f"left at {self.pressure_torr:g} Torr with {np.average(e_exit, weights=w):.3g} eV axial energy "
                         "per charge; collisions on the way to the pusher are not modelled")
        return beam, species, notes


@dataclass
class ExitGas:
    """What an ESI inlet delivers to the first vacuum stage: its gas and its ion species."""
    source: str
    slm: float                  # gas throughput, standard L/min
    pressure_torr: float        # first vacuum stage pressure
    temp_k: float               # room temperature (the throughput is counted at standard conditions)
    species: list               # IonSpecies (name, m/z, z, K0)
    speed: list                 # each species' speed at the skimmer, m/s
