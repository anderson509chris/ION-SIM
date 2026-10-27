"""Space charge of a steady ion beam (gun-code iteration).

A continuous beam of current I is represented by the N flown ions: each stands for a
stream I/N (A), so the charge it leaves near a node is I/N times the time it spends
there. One flight therefore gives the steady charge density:

    Q_node = (I / N) * sum over ions and steps of dt * (cloud-in-cell weight).

Poisson is solved for that charge with every electrode at 0 V (same matrix as the
applied field, insulators included), and its field is added to the applied field for the
next flight. A few passes with under-relaxation settle the beam and its own field
together. Ions that are still flying at t_max (trapped) are counted only for the time
flown, so a trap's charge is a lower bound.
"""
import numpy as np
from .gridtools import Dwell


class SpaceCharge:
    def __init__(self, gmap, solver):
        self.gmap, self.solver = gmap, solver
        self.charge = None
        self.phi = np.zeros(gmap.shape)
        self.table = None

    def update(self, charge, relax=1.0):
        """New node charges (C); relax < 1 mixes them with the previous ones."""
        self.charge = charge if self.charge is None else (1 - relax) * self.charge + relax * charge
        self.phi = self.solver.solve(np.zeros(self.gmap.shape), source=self.gmap.charge_source(self.charge))
        self.table = self.gmap.gradient_table(self.phi)

    def efield(self, pos):
        if self.table is None:
            return np.zeros((len(pos), 3))
        return self.gmap.vector(self.table, pos)

    @property
    def total_charge(self):
        return 0.0 if self.charge is None else float(self.charge.sum())


class WithSpaceCharge:
    """Applied field plus a space-charge field; everything else comes from the applied field."""

    def __init__(self, base, sc):
        self.base, self.sc = base, sc

    def efield(self, t, pos):
        return self.base.efield(t, pos) + self.sc.efield(pos)

    def __getattr__(self, k):
        return getattr(self.base, k)


def fly_with_space_charge(fly_once, field, gmap, solver, current, iterations=4, relax=0.4, progress=None):
    """fly_once(field, dwell=..., prog=...) -> FlightResult (all ions flown; the dwell accumulator
    gets the ions' positions).
    current: beam current (A, signed like the ions' charge). Flies iterations + 1 times; the
    last flight is the answer. Returns (result, SpaceCharge, history) where history lists
    (pass, peak space-charge potential V, charge in the device C). If the peak still moves by
    more than 15% between the last two solves the beam has not settled (often: more current
    than the device can carry, so ions pile up); res.sc_converged says which."""
    sc = SpaceCharge(gmap, solver)
    history = []
    n_pass = iterations + 1
    res = None
    for k in range(n_pass):
        sub = None
        if progress:
            sub = lambda f, k=k: progress((k + f) / n_pass)
        dwell = Dwell(gmap)
        res = fly_once(field if k == 0 else WithSpaceCharge(field, sc), dwell=dwell, prog=sub)
        if k == n_pass - 1 or (progress and progress((k + 1) / n_pass)):
            break
        n = len(res.status)
        sc.update(current / n * dwell.result(), 1.0 if k == 0 else relax)
        peak = float(sc.phi.max() if current >= 0 else sc.phi.min())
        history.append((k + 1, peak, sc.total_charge))
    res.space_charge, res.sc_history = sc, history
    res.sc_converged = len(history) < 2 or abs(history[-1][1] - history[-2][1]) <= 0.15 * abs(history[-1][1]) + 1e-3
    return res, sc, history
