@echo off
REM Runs the automated tests.
cd /d "%~dp0"
call ".venv\Scripts\activate.bat"
python -m pytest -q
pause
