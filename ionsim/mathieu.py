"""Mathieu stability for a quadrupole with rods at +-(U + V cos(Omega t)).

a = 8 z e U / (m r0^2 Omega^2),  q = 4 z e V / (m r0^2 Omega^2)
x obeys Mathieu(a, q), y obeys Mathieu(-a, -q). Stability is decided numerically
from the Floquet monodromy matrix (|trace| < 2), so no lookup tables are needed.
"""
import numpy as np
from .constants import AMU, E_CHARGE


def aq(U, V, mass_amu, r0, freq, charge_state=1):
    m = np.asarray(mass_amu, float) * AMU
    k = charge_state * E_CHARGE / (m * r0 ** 2 * (2 * np.pi * freq) ** 2)
    return 8 * k * U, 4 * k * V


def voltages_for(a, q, mass_amu, r0, freq, charge_state=1):
    """Inverse of aq(): the U, V that place this ion at (a, q)."""
    m = mass_amu * AMU
    k = charge_state * E_CHARGE / (m * r0 ** 2 * (2 * np.pi * freq) ** 2)
    return a / (8 * k), q / (4 * k)


def _trace(a, q, steps=400):
    """Trace of the monodromy matrix over one period xi in [0, pi]."""
    a, q = np.broadcast_arrays(np.asarray(a, float), np.asarray(q, float))
    h = np.pi / steps
    # two fundamental solutions: (u, u') = (1,0) and (0,1)
    y = np.zeros((4,) + a.shape)
    y[0] = 1.0; y[3] = 1.0

    def f(xi, y):
        w = a - 2 * q * np.cos(2 * xi)
        return np.stack([y[1], -w * y[0], y[3], -w * y[2]])

    xi = 0.0
    for _ in range(steps):
        k1 = f(xi, y); k2 = f(xi + h/2, y + h/2*k1); k3 = f(xi + h/2, y + h/2*k2); k4 = f(xi + h, y + h*k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4); xi += h
    return y[0] + y[3]


def is_stable(a, q):
    return (np.abs(_trace(a, q)) < 2) & (np.abs(_trace(-a, -q)) < 2)


def beta(a, q):
    """Secular frequency parameter beta (0..1) in the first stability region, NaN outside.
    From the Floquet trace: cos(pi beta) = trace / 2."""
    tr = _trace(a, q) / 2
    return np.where(np.abs(tr) < 1, np.arccos(np.clip(tr, -1, 1)) / np.pi, np.nan)


def secular_freqs(U, V, mass_amu, r0, freq, charge_state=1):
    """Fundamental secular frequencies (f_x, f_y) in Hz: beta * f_rf / 2. NaN where unstable."""
    a, q = aq(U, V, mass_amu, r0, freq, charge_state)
    return beta(a, q) * freq / 2, beta(-a, -q) * freq / 2


def resonant_mz(f_target, U, V, r0, freq, charge_state=1, axis="x", mz_range=(5, 50000), n=4000):
    """m/z values whose secular frequency along `axis` equals f_target (Hz). Usually one."""
    m = np.geomspace(*mz_range, n)
    fx, fy = secular_freqs(U, V, m, r0, freq, charge_state)
    f = (fx if axis == "x" else fy) - f_target
    out = []
    for i in np.nonzero(np.isfinite(f[:-1]) & np.isfinite(f[1:]) & (np.sign(f[:-1]) != np.sign(f[1:])))[0]:
        out.append(float(m[i] - f[i] * (m[i + 1] - m[i]) / (f[i + 1] - f[i])))
    return out


def stability_map(q_max=1.0, a_max=0.3, nq=300, na=200):
    q = np.linspace(0, q_max, nq)
    a = np.linspace(0, a_max, na)
    Q, A = np.meshgrid(q, a)
    return Q, A, is_stable(A, Q)
