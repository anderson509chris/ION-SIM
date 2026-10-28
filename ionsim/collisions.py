"""Gas in a vacuum-style device: one collision at a time (a collision cell or cooling quadrupole).

At 1e-4 to 1e-1 Torr an ion meets a gas molecule every few mm, so the smooth drag of
gasflight.py (many collisions per step) no longer fits. Here each ion flies on the RK4
integrator and, after every step, may hit one gas molecule:

    P = 1 - exp(-n sigma g dt),   g = |v_ion - v_gas|, v_gas drawn from the thermal gas

A hit is an elastic hard-sphere collision: the relative velocity keeps its size and turns to
a random direction in the centre-of-mass frame. sigma is the ion's collision cross section,
from its reduced mobility K0 in that gas (Mason-Schamp), so the same K0 that sets the drift
in a funnel sets the collision rate here. Many collisions in a row cool the ions (collisional
focusing); fast ions entering the gas heat on the first ones (CID energy).
"""
import numpy as np
from dataclasses import dataclass

from .constants import AMU, E_CHARGE, K_B
from .fields import FieldSource

N_LOSCHMIDT = 2.6867811e25     # m^-3 at 273.15 K, 760 Torr (the reference for K0)
TORR = 133.322
GASES = {"He": 4.0026, "N₂": 28.0134, "Ar": 39.948}


def ccs_from_k0(k0_cm2, mz, charge_state, gas_mass_amu, temp_k):
    """Collision cross section (m^2) from the reduced mobility, Mason-Schamp:
    Omega = 3 z e / (16 N0 K0) * sqrt(2 pi / (mu k T))."""
    m = np.asarray(mz, float) * np.asarray(charge_state, float) * AMU
    M = gas_mass_amu * AMU
    mu = m * M / (m + M)
    return (3 * np.asarray(charge_state, float) * E_CHARGE / (16 * N_LOSCHMIDT * np.asarray(k0_cm2, float) * 1e-4)
            * np.sqrt(2 * np.pi / (mu * K_B * temp_k)))


class AxialField(FieldSource):
    """A uniform push along z (V/m): the drag field of a collision cell (segmented rods,
    auxiliary electrodes or a resistive coating, here taken as uniform)."""

    def __init__(self, ez):
        self.ez = ez

    def efield(self, t, pos):
        E = np.zeros_like(pos)
        E[:, 2] = self.ez
        return E


@dataclass
class CollisionStats:
    count: np.ndarray      # collisions per ion
    ecom_first: np.ndarray # centre-of-mass energy of the first collision, eV (NaN if none)
    ecom_max: np.ndarray   # highest centre-of-mass energy, eV
    ecom_sum: np.ndarray   # summed centre-of-mass energy, eV

    def summary(self, mask):
        if not mask.any():
            return "no ions"
        c = self.count[mask]
        first = self.ecom_first[mask]
        first = first[np.isfinite(first)]
        txt = f"{np.mean(c):.0f} collisions per ion (median {np.median(c):.0f})"
        if len(first):
            txt += (f"; first collision {np.median(first):.3g} eV centre-of-mass (median), "
                    f"highest {np.max(self.ecom_max[mask]):.3g} eV")
        return txt


class HardSphereGas:
    """Integrator hook: elastic hard-sphere collisions with a thermal gas.
    sigma: per-ion cross section (m^2), the same order as the ions."""

    def __init__(self, pressure_torr, temp_k, gas_mass_amu, sigma):
        self.n = pressure_torr * TORR / (K_B * temp_k)
        self.temp_k = temp_k
        self.M = gas_mass_amu * AMU
        self.sigma = np.asarray(sigma, float)
        k = len(self.sigma)
        self.stats = CollisionStats(np.zeros(k, int), np.full(k, np.nan), np.zeros(k), np.zeros(k))
        self.p_max = 0.0

    def __call__(self, t, dt, ions, s, rng):
        v = ions.vel[s]
        m = ions.mass[s]
        u = np.sqrt(K_B * self.temp_k / self.M) * rng.standard_normal((len(s), 3))
        g = v - u
        gn = np.linalg.norm(g, axis=1)
        p = 1 - np.exp(-self.n * self.sigma[s] * gn * dt)
        if len(p):
            self.p_max = max(self.p_max, float(p.max()))
        hit = rng.random(len(s)) < p
        if not hit.any():
            return
        k = s[hit]
        mh, uh, gh, gnh = m[hit], u[hit], g[hit], gn[hit]
        M = self.M
        vcm = (mh[:, None] * v[hit] + M * uh) / (mh + M)[:, None]
        d = rng.standard_normal((len(k), 3))
        d /= np.linalg.norm(d, axis=1)[:, None]
        ions.vel[k] = vcm + (M / (mh + M))[:, None] * gnh[:, None] * d
        ecom = 0.5 * (mh * M / (mh + M)) * gnh ** 2 / E_CHARGE
        st = self.stats
        first = st.count[k] == 0
        st.ecom_first[k[first]] = ecom[first]
        st.count[k] += 1
        st.ecom_max[k] = np.maximum(st.ecom_max[k], ecom)
        st.ecom_sum[k] += ecom
