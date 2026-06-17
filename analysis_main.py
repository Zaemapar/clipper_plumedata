import numpy as np
import scipy as sp
import scipy.optimize
import csv
import matplotlib.pyplot as plt
import PyMieScatt as PyMie
import utils
import pred_vars as vars
import os
import sys

disttype_arr = [[4.4, 8.8, 2.16],
                [1.0, 7.7, 4.00]]

if __name__ == "__main__":
    # Figure out which type of particle size distribution is being input
    if vars.TYPE == "gring-like":
        type_idx = 0
    elif vars.TYPE == "ering-like":
        type_idx = 1
    elif vars.TYPE == "encleadus-like":
        type_idx = 2
    else:
        raise RuntimeError("TYPE should be either gring-like, ering-like, or encleadus-like.")

    disttype = disttype_arr[type_idx] # Get information about the size distribution
    angles = np.arange(vars.ANGLE_LOWERBOUND, vars.ANGLE_UPPERBOUND + 1, 1) # Array of scattering angles for graphing purposes

    # Make directory to hold the plot
    dir_path = utils.create_dirpath("analysis_main")

    # Loop through each sensor and output graphs based on wavelength ranges
    for sensor in vars.INSTRUMENTS.keys():
        wavel_range = vars.INSTRUMENTS[sensor]
        wavels = np.arange(wavel_range[0], wavel_range[1] + 0.01, 0.01) # Get a list of wavelengths for each instrument (assuming 10 nm resolution)
        midrange = wavels[int(np.floor(len(wavels) / 2))] # Get the wavelength around the median

        # Obtain the optical constants at the desired wavelength
        n,k=utils.get_nk(midrange, vars.COMP)

        # Compute the reflectance (I/F) of the particle population.
        # Note this uses the same normalization process as described in Appendix B of
        # de Pater et al. 2026 Characterization of the Outer Uranian Rings.... in JGR Planets 131 e2025JE009404.
        # doi.org/10.1029/2025JE00404
        m = complex(n, k) # Compute complex refractive index
        
        # Compute the reflectances as they vary with scattering angle based on calculated particle size distribution at the given wavelength
        theta_reflectances, _ = utils.angle_mie_reflectances(disttype[0], disttype[1], disttype[2], m, None, vars.ANGLE_LOWERBOUND, vars.ANGLE_UPPERBOUND, theta_min=vars.ANGLE_LOWERBOUND, theta_max=vars.ANGLE_UPPERBOUND, wavel=midrange, nsize=vars.NSIZE, tau=vars.TAU)

        params = [disttype[0], disttype[1], disttype[2], vars.TAU, sensor, midrange] # Array to contain all parameters, for graphing

        # Make angle plot
        utils.plt_angle(os.path.join(dir_path, f"{sensor}_ifs_angles.png"), angles, theta_reflectances, params, True) # Convert angles to degrees because PyMieScatt outputs radians
        # Make wavelength plot
        utils.plt_wavel(os.path.join(dir_path, f"{sensor}_ifs_wavels.png"), wavels, params, angle=vars.WAVEL_SNAPSHOT_ANGLE, composition=vars.COMP, sizes=vars.NSIZE)