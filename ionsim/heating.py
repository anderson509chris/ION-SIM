"""Ion heating: how hot ions get from field-driven collisions with the gas.

Effective temperature (two-temperature / Wannier theory). An ion driven through gas at a
velocity w relative to the gas has a collision-averaged energy as if it were at

    T_eff = T + M_gas w^2 / (3 k)

where w is the field-driven (noise-free) part of the ion's velocity relative to the gas.
In a DC field w = K E (the classic drift-tube result); in RF the ion's oscillation counts
too, so RF confinement heats ions even with no net drift. Here w is integrated along every
ion's actual path with the same exact drag step as the flight, so it covers DC, RF,
TWAVE, inlet jets and anything in between. The mean centre-of-mass collision energy is
(3/2) k T_eff.

Internal temperature. Collisions move the ion's internal temperature toward T_eff:

    dT_int / dt = (T_eff - T_int) * nu / n_eq

nu is the collision rate (from the mobility, via the Mason-Schamp cross section) and n_eq
the number of collisions it takes the internal modes to follow (a few to a few tens:
small ions follow fast, big ones lag). T_int is what decides fragmentation or unfolding.

Fragmentation / unfolding estimate (optional). A first-order Arrhenius rate
k = A exp(-Ea / k T_int) is integrated along the path; the surviving fraction is
exp(-integral k dt). Ea and A come from the ion (activation energy and pre-factor, e.g.
from a CIU or thermal-dissociation fit); 1e12-1e13 /s is typical for A.

This is the standard effective-temperature estimate. It is not a molecular-dynamics
model of energy deposition, so use it to compare devices, settings and regions.
"""
import numpy as np
from dataclasses import dataclass
from .constants import AMU, K_B, E_CHARGE

GASES = {"N₂ (28)": 28.013, "He (4)": 4.0026, "Ar (40)": 39.948, "Air (29)": 28.96, "SF₆ (146)": 146.06, "CO₂ (44)": 44.01}


@dataclass
class HeatingModel:
    gas_mass_amu: float = 28.013
    n_equil: float = 10.0          # collisions for T_int to follow T_eff
    ea_ev: float = 0.0             # activation energy for the fragmentation / unfolding estimate (0 = off)
    log10_a: float = 12.0          # Arrhenius pre-factor, log10(A / s^-1)
    bins: int = 120                # profile along the ion axis


class HeatTracker:
    """Per-ion heating along a flight. gasflight.fly calls step() every time step."""

    def __init__(self, model, ions, temp_k, z0, z1, tau):
        self.m, self.T = model, float(temp_k)
        n = len(ions.mass)
        M = model.gas_mass_amu * AMU
        self.M = M
        mu = ions.mass * M / (ions.mass + M)
        # Mason-Schamp: N*Omega = 3 q / (16 K) * sqrt(2 pi / (mu k T)); rate = N Omega <v_rel>
        self.n_omega = 3 * ions.charge / (16 * ions.K) * np.sqrt(2 * np.pi / (mu * K_B * temp_k))
        self.vrel_k = np.sqrt(8 * K_B / (np.pi * mu))
        v0 = getattr(ions, "v0", None)                # deterministic launch velocity (e.g. an injected beam)
        self.wd = np.zeros_like(ions.vel) if v0 is None else np.array(v0, float)   # driven velocity rel. to the gas
        self.t_int = np.full(n, float(temp_k))
        self.teff_max = np.full(n, float(temp_k))
        self.tint_max = np.full(n, float(temp_k))
        self.teff_int = np.zeros(n)                 # integral of T_eff dt
        self.time = np.zeros(n)
        self.ncoll = np.zeros(n)
        self.log_survive = np.zeros(n)
        self.where_max = np.full(n, np.nan)         # z of each ion's hottest internal temperature
        self.edges = np.linspace(z0, z1, model.bins + 1)
        nb = model.bins
        self.p_teff, self.p_tint, self.p_w = np.zeros(nb), np.zeros(nb), np.zeros(nb)
        self.p_max = np.full(nb, float(temp_k))
        self.p_rate = np.zeros(nb)                  # integral of the Arrhenius rate dt per bin (fragmentation map)
        self.k_a = 10.0 ** model.log10_a
        self.ea = model.ea_ev * E_CHARGE

    def step(self, s, u, atau, c, dt, pos):
        """s: flying ions; u: gas velocity there; atau = a * tau (drift velocity in the local
        field); c = exp(-dt / tau). Same exact step as the flight, without the noise."""
        w = (atau + (self.wd[s] - atau) * c[:, None])  # relative to the gas: steady state is K E
        self.wd[s] = w
        teff = self.T + self.M * np.sum(w * w, axis=1) / (3 * K_B)
        nu = self.n_omega[s] * self.vrel_k[s] * np.sqrt(teff)
        ti = self.t_int[s]
        ti = teff + (ti - teff) * np.exp(-nu * dt / max(self.m.n_equil, 1e-9))
        self.t_int[s] = ti
        hot = ti > self.tint_max[s]
        self.tint_max[s] = np.maximum(self.tint_max[s], ti)
        self.where_max[s[hot]] = pos[hot, 2]
        self.teff_max[s] = np.maximum(self.teff_max[s], teff)
        self.teff_int[s] += teff * dt
        self.time[s] += dt
        self.ncoll[s] += nu * dt
        rate = None
        if self.ea > 0:
            rate = self.k_a * np.exp(-self.ea / (K_B * ti))
            self.log_survive[s] -= rate * dt
        b = np.clip(np.searchsorted(self.edges, pos[:, 2]) - 1, 0, len(self.p_w) - 1)
        self.p_w += np.bincount(b, minlength=len(self.p_w)) * dt
        self.p_teff += np.bincount(b, weights=teff * dt, minlength=len(self.p_w))
        self.p_tint += np.bincount(b, weights=ti * dt, minlength=len(self.p_w))
        np.maximum.at(self.p_max, b, ti)
        if rate is not None:
            self.p_rate += np.bincount(b, weights=rate * dt, minlength=len(self.p_w))

    def result(self):
        z = 0.5 * (self.edges[1:] + self.edges[:-1])
        ok = self.p_w > 0
        nan = np.full(len(z), np.nan)
        return HeatResult(
            teff_max=self.teff_max, tint_max=self.tint_max,
            teff_mean=np.divide(self.teff_int, self.time, out=np.full(len(self.time), self.T), where=self.time > 0),
            n_coll=self.ncoll, survival=np.exp(self.log_survive), where_max=self.where_max,
            z=z, teff_z=np.where(ok, self.p_teff / np.where(ok, self.p_w, 1), nan),
            tint_z=np.where(ok, self.p_tint / np.where(ok, self.p_w, 1), nan),
            tint_max_z=np.where(ok, self.p_max, nan), rate_z=self.p_rate,
            temp_k=self.T, model=self.m)


@dataclass
class HeatResult:
    teff_max: np.ndarray       # per ion, K
    tint_max: np.ndarray       # per ion, K
    teff_mean: np.ndarray      # per ion, time-averaged, K
    n_coll: np.ndarray         # per ion
    survival: np.ndarray       # per ion, fraction intact (1 when the estimate is off)
    where_max: np.ndarray      # per ion, z of the hottest internal temperature
    z: np.ndarray              # profile along the ion axis
    teff_z: np.ndarray         # time-weighted mean T_eff in each z bin
    tint_z: np.ndarray         # time-weighted mean T_int
    tint_max_z: np.ndarray     # hottest T_int seen in each bin
    rate_z: np.ndarray         # summed fragmentation probability in each bin
    temp_k: float
    model: HeatingModel

    def ecom_ev(self, t):
        """Mean centre-of-mass collision energy (eV) at effective temperature t."""
        return 1.5 * K_B * np.asarray(t) / E_CHARGE

    def summary(self, mask=None):
        m = np.ones(len(self.teff_max), bool) if mask is None else mask
        if not m.any():
            return "no ions"
        txt = (f"T_eff peak {np.median(self.teff_max[m]):.0f} K (median; max {self.teff_max[m].max():.0f} K), "
               f"time-averaged {np.median(self.teff_mean[m]):.0f} K; internal T peak {np.median(self.tint_max[m]):.0f} K "
               f"(max {self.tint_max[m].max():.0f} K); mean collision energy at the peak "
               f"{self.ecom_ev(np.median(self.teff_max[m])):.3f} eV; {np.median(self.n_coll[m]):.3g} collisions per ion")
        if self.model.ea_ev > 0:
            txt += (f"; intact {np.mean(self.survival[m]):.1%} "
                    f"(Ea {self.model.ea_ev:g} eV, A 1e{self.model.log10_a:g}/s)")
        return txt
