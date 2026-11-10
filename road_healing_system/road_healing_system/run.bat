@echo off
REM Context-Aware Road Network Reconstruction - Windows start script.
REM First run: creates a virtual environment and installs the requirements.
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (set "PY=py -3") else (set "PY=python")

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  %PY% -m venv .venv
  if errorlevel 1 (
    echo Could not create a virtual environment. Install Python 3.10-3.12 from python.org and tick "Add python.exe to PATH".
    pause
    exit /b 1
  )
  call ".venv\Scripts\activate.bat"
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  if errorlevel 1 (
    echo Installing the requirements failed. Check your internet connection and try again.
    pause
    exit /b 1
  )
) else (
  call ".venv\Scripts\activate.bat"
)

if not exist ".env" copy ".env.example" ".env" >nul

echo Starting the app at http://127.0.0.1:5000 (or the APP_PORT set in .env). Press Ctrl+C to stop.
start "" /b cmd /c "timeout /t 4 /nobreak >nul & start "" http://127.0.0.1:5000"
python app.py
pause
