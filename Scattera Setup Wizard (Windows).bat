@echo off
echo Clearing old environments...
if exist "%~dp0.venv" rmdir /s /q "%~dp0.venv"

echo Setting up virtual environment in Windows...
:: cd /d enters the correct folder
cd /d "%~dp0"
:: Make a new virtual environment
python -m venv .venv

echo Installing required packages...
:: Point directly to the file requirements.txt to install collected packages; this mirrors pip install
"%~dp0.venv\Scripts\pip.exe" install -r "%~dp0src\requirements.txt"

echo Generating launcher...
echo @echo off > "%~dp0scattera.bat"
echo echo Launching Scattera... >> "%~dp0scattera.bat"
:: Tell the launcher file to call analysis_main inside the src folder
echo cd /d "%%~dp0src" >> "%~dp0scattera.bat"
:: Mirrors python3 analysis_main.py
echo "..\.venv\Scripts\python.exe" analysis_main.py >> "%~dp0scattera.bat"
echo pause >> "%~dp0scattera.bat"

echo Setup complete. Double-click scattera.bat to run the main application.
pause