"""Electrode geometry on a uniform 2D grid (x, y cross-section; z is the ion axis)."""
import numpy as np


class Grid2D:
    def __init__(self, half_width, dx):
        """Square grid spanning [-half_width, half_width] in x and y, spacing dx (metres)."""
        self.dx = dx
        m = int(np.ceil(half_width / dx))      # nodes each side of the axis, so x=0 is a node
        self.n = 2 * m + 1
        self.x = dx * np.arange(-m, m + 1)
        self.y = self.x.copy()
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")
        # electrode id per node, 0 = free space, 1..N = electrode index
        self.electrode = np.zeros((self.n, self.n), dtype=np.int16)
        self.names = []

    def add_electrode(self, name, mask):
        if name in self.names:
            idx = self.names.index(name) + 1
        else:
            self.names.append(name)
            idx = len(self.names)
        self.electrode[mask] = idx
        return idx

    def add_rod(self, name, cx, cy, radius):
        return self.add_electrode(name, (self.X - cx) ** 2 + (self.Y - cy) ** 2 <= radius ** 2)


def quadrupole_rods(r0, rod_ratio=1.1487, dx=None, margin=1.5, split=False):
    """Four round rods at (+-(r0+R),0), (0,+-(r0+R)).

    Rods on the x axis form electrode "A", rods on the y axis form "B". With split, each
    rod is its own electrode: "x+", "x-", "y+", "y-" (for dipolar excitation).
    rod_ratio = R/r0; 1.1487 is the classic best round-rod approximation to hyperbolic.
    The outer grid boundary is a grounded can.
    """
    R = rod_ratio * r0
    dx = dx or r0 / 80
    half = (r0 + 2 * R) * margin
    g = Grid2D(half, dx)
    c = r0 + R
    names = ["x+", "x-", "y+", "y-"] if split else ["A", "A", "B", "B"]
    for (cx, cy), name in zip([(c, 0), (-c, 0), (0, c), (0, -c)], names):
        g.add_rod(name, cx, cy, R)
    g.r0 = r0
    g.rod_radius = R
    return g
