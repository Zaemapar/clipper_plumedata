"""
A script to run the gui.py application, which contains a graphical user interface in which users can change various
parameters related to particle scattering and watch the graph update in real time. This script should be run in the
terminal to launch the GUI.

Authors: Parker A. Zaemann, Matthew M. Hedman
Date: 17 Jul 2026
Source: https://github.com/Zaemapar/clipper_plumedata
Contact: mhedman@uidaho.edu
"""

import numpy as np
import scipy as sp
import utils
import pred_vars as vars
import gui
import os
import sys
from PyQt5.QtWidgets import QApplication

if __name__ == "__main__":
    # Every PyQt app needs exactly one QApplication instance
    # sys.argv allows you to pass command line arguments to your app
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Instantiate window and display it
    window = gui.MainWindow()
    window.show()

    # Start the application's event loop and safely exit when closed
    sys.exit(app.exec())