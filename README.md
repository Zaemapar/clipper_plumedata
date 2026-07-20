# Europa Clipper Scattering Prediction Application (ECSPA)
The purpose of this tool is to allow users to utilize numerous scattering theories to plot reflectance or a related quantity over a given angle range or spectrum given a set of initial conditions such as particle size and composition, ultimately serving as a rudimentary visualization from which predictions can be made about the hypothetical cryovolcanic plumes on Europa, which NASA's Clipper probe aims to observe in 2030. It offers a few select plotting capabilities designed to model what might be observed by Clipper's instruments during its visit to Europa, meaning observations consistent with the predictions of this program will likely indicate cryovolcanic activity on the moon. Other features of the program are useful for the study of scattering theories in general, as users can plot multiple different initial conditions on the same graph for direct comparison between the effects of different parameters on how materials scatter light.

## Installation
To install EPSCA, you must have a Python installation of no later than 3.11 (since PyMieScatt is incompatible with newer versions of Python). You must also have pip installed.
To install the application, download the zip file or clone the repository into your files:
```
git clone https://github.com/Zaemapar/clipper_plumedata.git
```
In the resulting folder, double-click the setup wizard that best suits your operating system (`ECSPA Setup Wizard (Windows).bat` for Windows, `ECSPA Setup Wizard (WSL).bat` for WSL, `ECSPA Setup Wizard (MacOS).command` for Mac) and wait for the program to finish running. All of the packages in src/requirements.txt will be installed in the .venv virtual environment, and a main application launcher will be created (either `epsca.bat` or `epsca.command` depending on your operating system). Once this file has been created, double-click it to run the application.

If you would prefer to launch the program from the terminal, run the following commands from outside the folder if on Windows:
```
cd clipper_plumedata
python -m venv .venv
.venv\Scripts\activate
cd src
pip install -r requirements.txt
python analysis_main.py
```
Or, if you are on Linux/WSL:
```
cd clipper_plumedata
python3 -m venv .venv
source .venv/bin/activate
cd src
pip install -r requirements.txt
python3 analysis_main.py
```
After the first installation, only `python3 analysis_main.py` needs to be run from the `src` directory.

## Navigating the Home Screen
Once the main program has been launched, a PyQt5 window will appear as shown below, and a sample graph will quickly load onto the displayed graph:

