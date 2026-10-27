"""TWAVE traveling-wave (TW) ion mobility separation.

Model
-----
* Two parallel boards a distance `gap` apart, each carrying the same row of TW
  electrodes (pitch `pitch`, width `width`). Every `n_per_wave` electrodes form one
  wavelength; `n_on` adjacent electrodes sit at the TW amplitude, the rest at 0 V.
  The pattern steps one electrode every pitch/speed seconds.
* The field between the boards is solved exactly with a Fourier series (the
  geometry is periodic along z), with a linear potential ramp across the insulating
  gap between electrodes. As in the quad module, a basis field is solved once and
  re-used: the stepped wave is just the same table shifted by whole electrodes.
* Ions are at ~4 Torr, so they move in the mobility regime (overdamped):
      dz = K E_z dt + sqrt(2 D dt) N(0,1),   D = K kT / (ze)  (Einstein)
  Ions travel along the midplane of the boards (x = 0); the RF guard confinement
  across the device is not modelled in this first version.
"""
import dataclasses
import numpy as np
from dataclasses import dataclass
from .constants import E_CHARGE, K_B


def mobility(K0_cm2, pressure_torr, temp_k=300.0):
    """Reduced mobility K0 (cm^2/Vs) -> mobility K (m^2/Vs) at pressure and temperature."""
    return K0_cm2 * 1e-4 * (760.0 / pressure_torr) * (temp_k / 273.15)


@dataclass
class TWBoards:
    pitch: float = 0.508e-3       # electrode centre-to-centre spacing, m
    width: float = 0.432e-3       # electrode width, m
    gap: float = 2.75e-3          # board-to-board spacing, m
    n_per_wave: int = 8
    n_on: int = 4
    amplitude: float = 30.0       # V on the "on" electrodes
    speed: float = 200.0          # wave speed, m/s
    height: float = 0.0           # distance from the midplane where ions fly, m
    waveform: str = "electrodes"  # "electrodes" (solved) or "sine" (ideal, for checks)
    n_table: int = 4096
    n_modes: int = 400

    def __post_init__(self):
        self.wavelength = self.pitch * self.n_per_wave
        self.step_time = self.pitch / self.speed
        self.z_table = np.arange(self.n_table) * self.wavelength / self.n_table
        if self.waveform == "sine":
            # sine with the same peak midplane field as the electrode pattern
            ref = dataclasses.replace(self, waveform="electrodes")
            self.e_max = np.abs(ref.ez_table).max()
            self.ez_table = -self.e_max * np.sin(2 * np.pi * self.z_table / self.wavelength)
        else:
            self.ez_table = self.amplitude * self._solve_pattern()
            self.e_max = np.abs(self.ez_table).max()

    def board_potential(self, z):
        """Potential on the board surface (per volt) for the wave at step 0."""
        zz = np.mod(z, self.wavelength)
        v = np.zeros_like(zz)
        half_w = self.width / 2
        g = self.pitch - self.width
        on = lambda j: 1.0 if (j % self.n_per_wave) < self.n_on else 0.0
        for j in range(-1, self.n_per_wave + 1):
            c = (j + 0.5) * self.pitch
            v = np.where(np.abs(zz - c) <= half_w, on(j), v)
            # linear ramp across the insulating gap to the next electrode
            lo, hi = c + half_w, c + half_w + g
            ramp = (zz > lo) & (zz < hi)
            v = np.where(ramp, on(j) + (on(j + 1) - on(j)) * (zz - lo) / g, v)
        return v

    def _solve_pattern(self):
        """E_z at `height` from Laplace between two identical boards (Fourier series)."""
        zs = np.arange(self.n_table) * self.wavelength / self.n_table
        c = np.fft.rfft(self.board_potential(zs)) / self.n_table
        n = np.arange(len(c))[: self.n_modes]
        k = 2 * np.pi * n / self.wavelength
        # phi(x, z) = sum c_n cosh(k x) / cosh(k gap/2) e^{ikz}, symmetric boards
        h, g2 = abs(self.height), self.gap / 2
        att = np.exp(k * (h - g2)) * (1 + np.exp(-2 * k * h)) / (1 + np.exp(-2 * k * g2))  # cosh(kh)/cosh(kg/2)
        coef = c[: self.n_modes] * att
        ez_hat = -1j * k * coef
        spec = np.zeros(self.n_table // 2 + 1, dtype=complex)
        spec[: self.n_modes] = ez_hat
        return np.fft.irfft(spec * self.n_table, n=self.n_table)

    def ez(self, t, z):
        """Axial field at time t for ion positions z (m)."""
        if self.waveform == "sine":
            shift = self.speed * t                         # continuous ideal wave
        else:
            shift = np.floor(t / self.step_time) * self.pitch   # stepped electrode wave
        u = np.mod(z - shift, self.wavelength) / self.wavelength * self.n_table
        i = np.floor(u).astype(int) % self.n_table
        f = u - np.floor(u)
        return self.ez_table[i] * (1 - f) + self.ez_table[(i + 1) % self.n_table] * f


@dataclass
class MobilityResult:
    arrival: np.ndarray       # arrival time at z_end (s), NaN if not arrived
    species: np.ndarray
    traj_t: np.ndarray = None
    traj_z: np.ndarray = None


def fly_mobility(tw, K, z0, z_end, dt, t_max, temp_k=300.0, charge_state=1,
                 species=None, record_every=0, rng=None, progress=None):
    """Overdamped Langevin flight of N ions through a TW. K may be per-ion (N,).
    progress: optional callable(fraction) -> True to stop early (used by the GUI)."""
    rng = rng or np.random.default_rng()
    z = np.array(z0, float)
    n = len(z)
    K = np.broadcast_to(np.asarray(K, float), (n,))
    D = K * K_B * temp_k / (charge_state * E_CHARGE)
    sig = np.sqrt(2 * D * dt)
    arrival = np.full(n, np.nan)
    alive = np.ones(n, dtype=bool)
    tz, tt = [], []
    t, step = 0.0, 0
    while t < t_max and alive.any():
        a = np.nonzero(alive)[0]
        z[a] += K[a] * tw.ez(t, z[a]) * dt + sig[a] * rng.standard_normal(len(a))
        t += dt; step += 1
        if progress and step % 500 == 0 and progress(max(t / t_max, 1 - alive.mean())):
            break
        done = z[a] >= z_end
        arrival[a[done]] = t
        alive[a[done]] = False
        if record_every and step % record_every == 0:
            tz.append(np.where(alive, z, np.nan)); tt.append(t)
    res = MobilityResult(arrival, np.zeros(n, int) if species is None else species)
    if record_every:
        res.traj_t, res.traj_z = np.array(tt), np.array(tz)
    return res


def mean_velocity_sine(c, s):
    """Analytic mean drift for a continuous sinusoidal TW (c = K E_max, s = wave speed).
    Surfing (c >= s): v = s.  Rolling over (c < s): v = s - sqrt(s^2 - c^2)."""
    c = np.asarray(c, float)
    return np.where(c >= s, s, s - np.sqrt(np.maximum(s * s - c * c, 0)))


def fwhm(x):
    """Peak FWHM estimated as 2.355 sigma, with sigma from the interquartile range
    (robust to the few stragglers that diffusion leaves a wave behind)."""
    x = x[np.isfinite(x)]
    q1, q3 = np.percentile(x, [25, 75])
    return 2.355 * (q3 - q1) / 1.349
