@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto install
py -3.11 -m venv .venv
if errorlevel 1 (
  echo Install Python 3.11 from python.org, then run this file again.
  pause
  exit /b 1
)
:install
.venv\Scripts\python.exe -m pip install --timeout 120 -r requirements.txt
if errorlevel 1 goto failed
.venv\Scripts\python.exe main.py download-model
if errorlevel 1 goto failed
echo Setup complete. Double-click run_demo.bat or run_webcam.bat.
pause
exit /b 0
:failed
echo Setup failed. Check the message above and your Internet connection.
pause
exit /b 1
