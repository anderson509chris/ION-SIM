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


def stability_map(q_max=1.0, a_max=0.3, nq=300, na=200):
    q = np.linspace(0, q_max, nq)
    a = np.linspace(0, a_max, na)
    Q, A = np.meshgrid(q, a)
    return Q, A, is_stable(A, Q)
