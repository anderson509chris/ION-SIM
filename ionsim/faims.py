"""FAIMS (differential mobility) in a planar gap.

Model
-----
* Two parallel plates `gap` apart, length `length`. Carrier gas flows along x with a
  laminar (parabolic) or plug profile. Ions enter spread across the gap.
* Across the gap the field is uniform: E(t) = (DV * f(t) + CV) / gap, where f(t) is
  the asymmetric waveform with peak +1 (bisinusoidal or rectangular, zero mean).
* Field-dependent mobility:  K(E) = K(0) * (1 + alpha(E/N)),
      alpha = a2 (E/N)^2 + a4 (E/N)^4     (E/N in Townsend)
  a2 > 0 is "type A" (mobility rises with field), a2 < 0 "type C", a2 > 0 with a4 < 0 "type B".
* Ions move in the mobility regime: dy = K(E) E dt + sqrt(2 D dt) N(0,1) across the gap,
  dx = u_gas(y) dt along it. An ion that touches a plate is lost.

The theoretical compensation voltage is where the waveform-averaged drift vanishes,
<K(E) E>_t = 0, which `cv_theory` finds by root search.
"""
import numpy as np
from dataclasses import dataclass, field
from .constants import E_CHARGE, K_B

TD = 1e-21          # 1 Townsend in V m^2


@dataclass
class FaimsSpecies:
    name: str
    K0: float               # reduced mobility, cm^2/Vs
    a2: float = 0.0         # Td^-2
    a4: float = 0.0         # Td^-4
    charge_state: int = 1

    def alpha(self, en_td):
        en2 = en_td * en_td
        return self.a2 * en2 + self.a4 * en2 * en2


@dataclass
class Faims:
    gap: float = 1.88e-3          # m
    length: float = 40e-3         # m
    dv: float = 4000.0            # peak dispersion voltage, V
    freq: float = 750e3           # Hz
    waveform: str = "bisinusoidal"   # or "rectangular"
    duty: float = 1 / 3           # rectangular: fraction of the period at high field
    pressure_torr: float = 760.0
    temp_k: float = 300.0
    gas_velocity: float = 5.0     # mean carrier gas speed, m/s
    flow_profile: str = "parabolic"  # or "plug"

    @property
    def n_gas(self):
        return self.pressure_torr * 133.322 / (K_B * self.temp_k)

    def f(self, t):
        """Waveform with max +1 and zero mean."""
        ph = np.mod(t * self.freq, 1.0)
        if self.waveform == "rectangular":
            d = self.duty
            return np.where(ph < d, 1.0, -d / (1 - d))
        w = 2 * np.pi * ph
        return (2 * np.sin(w) + np.sin(2 * w - np.pi / 2)) / 3

    def k_low(self, sp):
        """Low-field mobility (m^2/Vs) at the gas density."""
        return sp.K0 * 1e-4 * (2.6867811e25 / self.n_gas)

    def drift(self, sp, E):
        en = np.abs(E) / self.n_gas / TD
        return self.k_low(sp) * (1 + sp.alpha(en)) * E

    def u_gas(self, y):
        if self.flow_profile == "plug":
            return np.full_like(y, self.gas_velocity)
        return 1.5 * self.gas_velocity * (1 - (2 * y / self.gap) ** 2)

    def ed_td(self):
        return self.dv / self.gap / self.n_gas / TD


def cv_theory(faims, sp, n=2000):
    """CV (V) at which the period-averaged transverse drift is zero."""
    t = np.arange(n) / n / faims.freq
    fd = faims.f(t)

    def mean_drift(cv):
        return np.mean(faims.drift(sp, (faims.dv * fd + cv) / faims.gap))

    lo, hi = -0.3 * faims.dv, 0.3 * faims.dv
    flo, fhi = mean_drift(lo), mean_drift(hi)
    if flo * fhi > 0:
        return np.nan
    for _ in range(80):                       # bisection: robust for any alpha shape
        mid = 0.5 * (lo + hi)
        fm = mean_drift(mid)
        if fm * flo > 0:
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass
class FaimsResult:
    cv: np.ndarray            # (n_cv,)
    transmission: np.ndarray  # (n_species, n_cv)
    cv_theory: np.ndarray     # (n_species,)
    traj: dict = field(default_factory=dict)


def cv_scan(faims, species, cvs, ions_per_cv=60, steps_per_period=30, rng=None,
            record=0, progress=None):
    """Simulated transmission vs CV for each species. All CVs and species fly together.
    record: number of ions (at the theoretical CV of species 0) whose y(t) is stored."""
    rng = rng or np.random.default_rng()
    cvs = np.asarray(cvs, float)
    ns, nc, per = len(species), len(cvs), int(ions_per_cv)
    sp_idx = np.repeat(np.arange(ns), nc * per)
    cv = np.tile(np.repeat(cvs, per), ns)
    if record:
        cv0 = cv_theory(faims, species[0])
        sp_idx = np.concatenate([sp_idx, np.zeros(record, int)])
        cv = np.concatenate([cv, np.full(record, cv0 if np.isfinite(cv0) else 0.0)])
    n = len(cv)
    K = np.array([faims.k_low(s) for s in species])[sp_idx]
    a2 = np.array([s.a2 for s in species])[sp_idx]
    a4 = np.array([s.a4 for s in species])[sp_idx]
    zc = np.array([s.charge_state for s in species])[sp_idx]
    D = K * K_B * faims.temp_k / (zc * E_CHARGE)
    half = faims.gap / 2
    y = (rng.random(n) * 2 - 1) * half * 0.9
    x = np.zeros(n)
    alive = np.ones(n, bool)
    through = np.zeros(n, bool)
    dt = 1 / faims.freq / steps_per_period
    sig = np.sqrt(2 * D * dt)
    t_max = 4 * faims.length / faims.gas_velocity
    t, step = rng.random() / faims.freq, 0
    rec_t, rec_y = [], []
    N = faims.n_gas
    while t < t_max and alive.any():
        a = np.nonzero(alive)[0]
        E = (faims.dv * faims.f(t) + cv[a]) / faims.gap
        en2 = (E / N / TD) ** 2
        vy = K[a] * (1 + a2[a] * en2 + a4[a] * en2 * en2) * E
        y[a] += vy * dt + sig[a] * rng.standard_normal(len(a))
        x[a] += faims.u_gas(y[a]) * dt
        t += dt; step += 1
        hit = np.abs(y[a]) >= half
        out = x[a] >= faims.length
        alive[a[hit | out]] = False
        through[a[out & ~hit]] = True
        if record and step % 5 == 0:
            yy = np.where(alive[-record:], y[-record:], np.nan)
            rec_t.append(t); rec_y.append(np.column_stack([x[-record:], yy]))
        if progress and step % 500 == 0 and progress(max(t / t_max, 1 - alive.mean())):
            break
    m = through[: ns * nc * per].reshape(ns, nc, per).mean(axis=2)
    res = FaimsResult(cvs, m, np.array([cv_theory(faims, s) for s in species]))
    if record:
        res.traj = {"t": np.array(rec_t), "xy": np.array(rec_y)}
    return res


def dispersion_curves(faims, species, dvs):
    """Theoretical CV vs DV for each species (the FAIMS 'dispersion plot')."""
    out = np.zeros((len(species), len(dvs)))
    for j, dv in enumerate(dvs):
        f = Faims(**{**faims.__dict__, "dv": dv})
        for i, s in enumerate(species):
            out[i, j] = cv_theory(f, s)
    return out


DEFAULT_SPECIES = [
    FaimsSpecies("type A", 1.50, a2=1.0e-5, a4=0.0),
    FaimsSpecies("type B", 1.40, a2=1.6e-5, a4=-1.8e-9),
    FaimsSpecies("type C", 1.20, a2=-4.0e-6, a4=0.0),
]
