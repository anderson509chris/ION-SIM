@echo off
REM Builds dist\IonSim\IonSim.exe  (run from this folder in a terminal with Python 3.10+)
python -m venv .venv
call .venv\Scripts\activate
pip install -r requirements.txt pyinstaller
pyinstaller --noconfirm ion_sim.spec
echo.
echo Done. Run dist\IonSim\IonSim.exe  (zip the dist\IonSim folder to share it)
