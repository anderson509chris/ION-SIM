# PyInstaller spec for the Ion Sim desktop app.  Build with:  pyinstaller ion_sim.spec
block_cipher = None

a = Analysis(
    ["ion_sim_app.py"],
    pathex=["."],
    hiddenimports=["scipy.sparse.linalg", "matplotlib.backends.backend_qtagg"],
    excludes=["tkinter", "PyQt5", "PyQt6", "IPython", "pytest"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="IonSim", console=False)
coll = COLLECT(exe, a.binaries, a.datas, name="IonSim")
app = BUNDLE(coll, name="IonSim.app", bundle_identifier="com.ionsim.app")  # used on macOS only
