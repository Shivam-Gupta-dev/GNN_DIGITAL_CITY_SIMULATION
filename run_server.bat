@echo off
cd /d "%~dp0"
call twin-city-env\Scripts\activate.bat
python backend\app.py
pause
