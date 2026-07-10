@echo off
echo Clearing old environments...
wsl bash -c "cd \"$(wslpath '%~dp0')\" && rm -rf .venv"
echo Setting up virtual environment in WSL...
wsl bash -c "cd \"$(wslpath '%~dp0')\" && python3 -m venv .venv"
echo Installing required packages...
wsl bash -c "cd \"$(wslpath '%~dp0')/src\" && ../.venv/bin/pip install -r requirements.txt"
echo Generating launcher...
echo @echo off > "%~dp0scattera.bat"
echo echo Launching Scattera... >> "%~dp0scattera.bat"
echo wsl bash -c "cd \"$(wslpath '%%~dp0')/src\" && ../.venv/bin/python3 analysis_main.py" >> "%~dp0scattera.bat"
echo pause >> "%~dp0scattera.bat"
echo Setup complete. Double-click scattera.bat to run the main application.
pause