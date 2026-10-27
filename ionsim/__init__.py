"""ionsim: a small SIMION-style ion optics simulator.

Pipeline (same idea as SIMION):
  1. geometry  - paint electrodes onto a grid
  2. solver    - solve Laplace once per electrode at unit volts (basis potentials)
  3. fields    - superpose basis potentials with time-dependent voltages
  4. integrator- fly ions through the fields, with pluggable extra forces
"""
from .constants import AMU, E_CHARGE
from .geometry import Grid2D, quadrupole_rods
from .solver import solve_basis
from .fields import GridField2D, IdealQuadField, rf_dc
from .ions import Ion, make_beam
from .integrator import Simulation, Result
from . import mathieu
