import numpy as np
import scipy as sp
import utils
import pred_vars as vars
import gui
import os
import sys
from PyQt5.QtWidgets import QApplication

"""def power_output():
    # Make directory to hold the plot
    dir_path = utils.create_dirpath("analysis_main")

    # Define a function to use in curve-fitting HG function to match Mie via scale factor
    def hg_iterator(angles, scale_factor):
        hg_ifs = utils.henyey_greenstein(angles, [[init_hg_params[1], init_hg_params[0] * scale_factor], [init_hg_params[3], init_hg_params[2] * scale_factor]])
        return np.log10(hg_ifs)
    
    # Loop through each sensor and output graphs based on wavelength ranges
    for sensor in vars.INSTRUMENTS.keys():
        wavel_range = vars.INSTRUMENTS[sensor]
        wavels = np.arange(wavel_range[0], wavel_range[1] + 0.01, 0.01) # Get a list of wavelengths for each instrument (assuming 10 nm resolution)
        midrange = wavels[int(np.floor(len(wavels) / 2))] # Get the wavelength around the median
        
        # Compute the reflectances as they vary with scattering angle based on calculated particle size distribution at the given wavelength
        reflectances, _ = utils.angle_mie_reflectances(disttype[0], disttype[1], disttype[2], theta_min=vars.ANGLE_LOWERBOUND, theta_max=vars.ANGLE_UPPERBOUND, wavels=wavels, nsize=vars.NSIZE, tau=vars.TAU)
        midrange_idx = np.where(wavels == midrange)[0][0]

        # Try to best-fit the Henyey-Greenstein function to the Mie function via a scale factor
        try:
            # Fit in log space so the tail isn't ignored
            popt, _ = sp.optimize.curve_fit(
                hg_iterator, 
                angles, 
                np.log10(reflectances[midrange_idx]), # Log of original dataset is being fit with log of best-fit dataset
                p0=[1e-6], 
                maxfev=10000,
            )
            factor = popt.tolist()[0]
        except Exception as e:
            print(f"Curve fitting failed: {e}")
            factor = 1e-6

        hg_reflectances = utils.henyey_greenstein(np.radians(angles), [[init_hg_params[1], init_hg_params[0] * factor], [init_hg_params[3], init_hg_params[2] * factor]])
    
        params = [disttype[0], disttype[1], disttype[2], vars.TAU, sensor, midrange] # Array to contain all parameters, for graphing
        hg_params = [init_hg_params[0], init_hg_params[1], init_hg_params[2], init_hg_params[3], factor] # Array to contain all Henyey-Greenstein parameters, for graphing

        # Make angle plot
        utils.plt_angle(os.path.join(dir_path, f"{sensor}_ifs_angles_gringlike.png"), angles, reflectances[midrange_idx], hg_reflectances, params, hg_params) # Convert angles to degrees because PyMieScatt outputs radians
        
        # Make wavelength plots
        utils.plt_wavel(os.path.join(dir_path, f"{sensor}_ifs_wavels_gringlike.png"), wavels, angles, vars.PLOT_ANGLES, reflectances, params)

        # Make csv
        utils.angle_wavel_csv(os.path.join(dir_path, f"{sensor}_reflectances_gringlike.csv"), wavels, angles, reflectances)
"""

if __name__ == "__main__":
    # Every PyQt app needs exactly one QApplication instance
    # sys.argv allows you to pass command line arguments to your app
    app = QApplication(sys.argv)

    # Instantiate window and display it
    window = gui.MainWindow()
    window.show()

    # Start the application's event loop and safely exit when closed
    sys.exit(app.exec())