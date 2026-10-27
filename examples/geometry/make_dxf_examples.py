"""Write the example DXF drawings used by the Device builder tab.

Run:  python examples/geometry/make_dxf_examples.py
  quad_rods.dxf            4 round rods, cross-section (2D)           layers ROD_A, ROD_B
  ring_funnel_profile.dxf  30-ring funnel, half profile (axisymmetric) layers RINGS, EXIT
  twave_board.dxf          TWAVE board top view (boards)              layers TW, RF, GUARD
All in mm. Some outlines are drawn as separate LINEs on purpose, to show joining.
"""
import os
import ezdxf

HERE = os.path.dirname(os.path.abspath(__file__))


def new():
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 4          # mm
    return doc, doc.modelspace()


def rect(msp, u0, u1, v0, v1, layer, as_lines=False):
    pts = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
    if as_lines:
        for a, b in zip(pts, pts[1:] + pts[:1]):
            msp.add_line(a, b, dxfattribs={"layer": layer})
    else:
        msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": layer})


def quad():
    doc, msp = new()
    r0, R = 4.0, 4.0 * 1.1487
    c = r0 + R
    for x, y, layer in [(c, 0, "ROD_A"), (-c, 0, "ROD_A"), (0, c, "ROD_B"), (0, -c, "ROD_B")]:
        msp.add_circle((x, y), R, dxfattribs={"layer": layer})
    doc.saveas(os.path.join(HERE, "quad_rods.dxf"))


def funnel():
    doc, msp = new()
    n, pitch, t, od = 30, 1.0, 0.5, 20.0
    for k in range(n):
        d = 12.0 + (2.5 - 12.0) * min(max((k - 3) / (n - 4), 0), 1)
        z = pitch * k
        rect(msp, z - t / 2, z + t / 2, d / 2, od / 2, "RINGS", as_lines=(k % 2 == 1))
    z = pitch * (n - 1) + 1.0
    rect(msp, z, z + 0.5, 1.25, od / 2, "EXIT")
    msp.add_line((0, -3), (10, -3), dxfattribs={"layer": "NOTES"})        # stray open line
    doc.saveas(os.path.join(HERE, "ring_funnel_profile.dxf"))


def twave():
    doc, msp = new()
    pitch, w, n = 0.508, 0.432, 48
    for k in range(n):
        u0 = k * pitch
        for v0, v1 in ((0.35, 1.35), (-1.35, -0.35)):
            rect(msp, u0, u0 + w, v0, v1, "TW")
    L = n * pitch
    for v0, v1 in ((-0.25, 0.25), (1.45, 1.85), (-1.85, -1.45)):
        rect(msp, -0.2, L + 0.2, v0, v1, "RF")
    for v0, v1 in ((2.0, 2.6), (-2.6, -2.0)):
        rect(msp, -0.2, L + 0.2, v0, v1, "GUARD")
    doc.saveas(os.path.join(HERE, "twave_board.dxf"))


if __name__ == "__main__":
    quad(); funnel(); twave()
    print("wrote quad_rods.dxf, ring_funnel_profile.dxf, twave_board.dxf")
