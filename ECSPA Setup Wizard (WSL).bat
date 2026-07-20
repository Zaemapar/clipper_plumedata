@echo off
:: First we ensure that any previous virtual environment is removed
echo Clearing old environments...
wsl bash -c "cd \"$(wslpath '%~dp0')\" && rm -rf .venv"
:: Make a new virtual environment
echo Setting up virtual environment in WSL...
wsl bash -c "cd \"$(wslpath '%~dp0')\" && python3 -m venv .venv"
:: Pip install requirements.txt, which contains all required packages
echo Installing required packages...
wsl bash -c "cd \"$(wslpath '%~dp0')/src\" && ../.venv/bin/pip install -r requirements.txt"
:: Write to a new .bat file a script to run analysis_main.py inside the src folder
echo Generating launcher...
echo @echo off > "%~dp0ecspa.bat"
:: Put in an extra echo call to signify startup
echo echo Launching ECSPA... >> "%~dp0ecspa.bat"
echo wsl bash -c "cd \"$(wslpath '%%~dp0')/src\" && ../.venv/bin/python3 analysis_main.py" >> "%~dp0scattera.bat"
echo pause >> "%~dp0ecspa.bat"

echo Setup complete. Double-click ecspa.bat to run the main application.
pause