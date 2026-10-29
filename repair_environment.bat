@echo off
setlocal
cd /d "%~dp0"
echo Refreshing Python launchers for the current project folder...
py -3.11 -m venv .venv
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m pip install --timeout 120 --force-reinstall pip
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m pip install --timeout 120 --force-reinstall -r requirements-lock.txt
if errorlevel 1 goto failed
.venv\Scripts\python.exe main.py doctor
if errorlevel 1 goto failed
echo Environment repaired. Run run_demo.bat or run_webcam.bat.
pause
exit /b 0
:failed
echo Repair failed. Check Python 3.11 installation and Internet access.
pause
exit /b 1
