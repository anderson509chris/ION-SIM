# PyInstaller spec for the Ion Sim desktop app.  Build with:  pyinstaller ion_sim.spec
import glob, os, sys
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

# gmsh (STEP import) loads its native library with ctypes, so PyInstaller can't see it.
# pip puts it in <prefix>/lib (macOS, Linux) or <prefix>/Lib (Windows); gmsh.py looks next to itself.
gmsh_libs = [(p, ".") for d in ("lib", "Lib", "bin") for p in glob.glob(os.path.join(sys.prefix, d, "*gmsh*"))
             if os.path.isfile(p) and not p.endswith((".py", ".jl"))]

a = Analysis(
    ["ion_sim_app.py"],
    pathex=["."],
    binaries=gmsh_libs,
    hiddenimports=["scipy.sparse.linalg", "matplotlib.backends.backend_qtagg", "gmsh", "ezdxf"]
                  + collect_submodules("pyamg") + collect_submodules("trimesh.exchange")
                  + collect_submodules("ezdxf.path") + collect_submodules("ezdxf.entities"),
    datas=[(p, "examples/geometry") for p in glob.glob("examples/geometry/*")
           if p.endswith((".dxf", ".step", ".ionsim"))],                        # sample files for the Device builder
    excludes=["tkinter", "PyQt5", "PyQt6", "IPython", "pytest"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="IonSim", console=False)
coll = COLLECT(exe, a.binaries, a.datas, name="IonSim")
app = BUNDLE(coll, name="IonSim.app", bundle_identifier="com.ionsim.app")  # used on macOS only
