import numpy as np
from dataclasses import dataclass, asdict
from .constants import AMU, E_CHARGE


@dataclass
class IonSpecies:
    """One kind of ion in a mixture."""
    name: str = "ion"
    mz: float = 500.0
    z: int = 1               # charge state
    K0: float = 1.0          # reduced mobility, cm^2/Vs (gas-phase devices)
    n: int = 100             # ions of this kind to fly

    def to_dict(self):
        return asdict(self)


def interleave(counts):
    """Species index of every ion, round-robin (0, 1, 2, 0, 1, 2, ...), so the first few
    ions (the ones whose paths are recorded) include every species."""
    counts = np.maximum(np.asarray(counts, int), 0)
    idx = np.repeat(np.arange(len(counts)), counts)
    rank = np.concatenate([np.arange(c) for c in counts]) if len(counts) else np.zeros(0, int)
    return idx[np.argsort(rank, kind="stable")]


def expand(species):
    """Per-ion arrays for a list of IonSpecies: (species index, m/z, charge state, K0 in cm^2/Vs)."""
    idx = interleave([s.n for s in species])
    pick = lambda k: np.array([getattr(s, k) for s in species], float)[idx]
    return idx, pick("mz"), pick("z"), pick("K0")


@dataclass
class Ion:
    """A population of N ions. Arrays are (N,) or (N,3)."""
    pos: np.ndarray
    vel: np.ndarray
    mass: np.ndarray      # kg
    charge: np.ndarray    # C
    t0: np.ndarray = None # birth time (s); spread over an RF period to sample all phases

    @property
    def n(self):
        return len(self.pos)

    @property
    def mz(self):
        return self.mass / AMU / (self.charge / E_CHARGE)


def make_beam(n, mass_amu, charge_state=1, energy_ev=5.0, radius=0.2e-3,
              half_angle_deg=1.0, energy_spread_ev=0.0, birth_window=0.0, rng=None):
    """Ions starting at z=0, uniform in a disc of `radius`, travelling +z with a cone of directions.
    mass_amu and charge_state may be scalars or (n,) arrays (mass scans, ion mixtures).
    Birth times are uniform in [0, birth_window); use one RF period to randomise RF phase."""
    rng = rng or np.random.default_rng()
    mass = np.broadcast_to(np.asarray(mass_amu, float), (n,)) * AMU
    charge = np.broadcast_to(np.asarray(charge_state, float), (n,)) * E_CHARGE
    r = radius * np.sqrt(rng.random(n))
    th = 2 * np.pi * rng.random(n)
    pos = np.column_stack([r * np.cos(th), r * np.sin(th), np.zeros(n)])
    ke = np.maximum(energy_ev + energy_spread_ev * rng.standard_normal(n), 1e-3) * E_CHARGE
    speed = np.sqrt(2 * ke / mass)
    polar = np.radians(half_angle_deg) * np.sqrt(rng.random(n))
    az = 2 * np.pi * rng.random(n)
    vel = speed[:, None] * np.column_stack([np.sin(polar) * np.cos(az),
                                            np.sin(polar) * np.sin(az),
                                            np.cos(polar)])
    return Ion(pos, vel, mass.copy(), charge, birth_window * rng.random(n))
