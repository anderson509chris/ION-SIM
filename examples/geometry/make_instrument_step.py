"""Write instrument.step, a simple whole-instrument model for the Instrument tab.

Run:  python examples/geometry/make_instrument_step.py
Ions travel along +z (mm):
  inlet capillary   z -60 .. 2     through the source wall of the first chamber
  planar funnel     z 10 .. 70     two boards, 6 mm gap, in the first chamber (about 2 Torr)
  interface wall    z 77 .. 83     the walls between the chambers, 2 mm orifice
  quadrupole        z 95 .. 295    four rods, r0 4 mm, in the second chamber
  OA-TOF pusher     z 310 .. 340   pusher plate and puller grid, beside the beam
  vacuum chamber    two boxes around all of it
"""
import os
import gmsh

HERE = os.path.dirname(os.path.abspath(__file__))


def main(path=os.path.join(HERE, "instrument.step")):
    gmsh.initialize(interruptible=False)
    gmsh.option.setNumber("General.Terminal", 0)
    occ = gmsh.model.occ
    gmsh.model.add("instrument")

    def shell(x0, y0, z0, dx, dy, dz, t):
        outer = occ.addBox(x0, y0, z0, dx, dy, dz)
        inner = occ.addBox(x0 + t, y0 + t, z0 + t, dx - 2 * t, dy - 2 * t, dz - 2 * t)
        return occ.cut([(3, outer)], [(3, inner)])[0]

    def tube(z0, length, r_out, r_in):
        o = occ.addCylinder(0, 0, z0, 0, 0, length, r_out)
        i = occ.addCylinder(0, 0, z0 - 1, 0, 0, length + 2, r_in)
        return occ.cut([(3, o)], [(3, i)])[0]

    # chambers: the first (funnel) and the second (quadrupole and TOF); the first's back wall is the interface
    def holed(box, holes):
        for z0, h in holes:
            box = occ.cut(box, [(3, occ.addCylinder(0, 0, z0, 0, 0, h, 1.0))])[0]
        return box
    holed(shell(-40, -40, -5, 80, 80, 85, 3), [(-10, 10), (75, 10)])         # first chamber, z -5 .. 80
    holed(shell(-40, -40, 80, 80, 80, 280, 3), [(75, 15)])                   # second chamber, z 80 .. 360
    tube(-60, 62, 0.9, 0.25)                                 # inlet capillary
    occ.addBox(-15, 3, 10, 30, 1.6, 60)                      # planar funnel, upper board
    occ.addBox(-15, -4.6, 10, 30, 1.6, 60)                   # lower board
    for cx, cy in ((8.6, 0), (-8.6, 0), (0, 8.6), (0, -8.6)):
        occ.addCylinder(cx, cy, 95, 0, 0, 200, 4.6)          # quadrupole rods
    occ.addBox(-20, -12, 310, 3, 24, 30)                     # TOF pusher plate
    occ.addBox(-2, -12, 310, 0.5, 24, 30)                    # puller grid (as a thin plate)
    occ.synchronize()
    gmsh.write(path)
    gmsh.finalize()
    print("wrote", path)


if __name__ == "__main__":
    main()
