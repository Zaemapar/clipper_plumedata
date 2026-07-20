import numpy as np
import pred_vars as pvars
import utils
import sys
from multiprocessing import shared_memory
import pickle
import base64

if __name__ == "__main__":
    """
    A program that takes input scattering data as might be observed by one of Europa Clipper's instruments,
    as well as additional data about the fit and the plot type desired, and outputs the corresponding x and y
    arrays for a plot.

    :param graphmode: Integer, indicates whether to output reflectance vs. wavelength (0), surface reflectance
                     vs. wavelength (1), or reflectance vs. scattering angle (2)
    :param sensor: String sensor from which to read wavelength range
    :param comps: Dictionary with keys as materials and values as volume fractions of materials
    :param mixmodel: String mixture model to use, either 'Areal' or 'Molecular'
    :param datamodel: Array of size distribution strings [min particle size, max particle size, power law,
                      r, G, x0]
    :param fitmodel: String for scattering theory to use, either 'Semi-Empirical Mie', 'Pure Mie',
                     'Mie Scattering', 'Mie External Reflection', or 'Mie Transmission'.
    :param param: Fixed parameter to base reflectances on. Int scattering angle in degrees if graphmode = 0, 
                  float effective grain size in microns if graphmode = 1, float wavelength in microns if 
                  graphmode = 2
    :param tau: Float optical depth (unitless)
    :param nsizes: List [size mode, integer number of sizes to use in size distribution]
    :param wavelbounds: Tuple (float, float) wavelength bounds in microns as established by the composition, 
                        representing data available for index of refraction interpolation
    :param output: List of strings indicating the type of output to return. First element is x axis, either
                   'Phase Angle', 'Scattering Angle', or 'Wavelength'. Second element is y axis, either 
                   'Reflectance' or 'Phase Function'.
    :returns: List containing x data values (wavelengths in microns, angles in degrees)
              List containing float reflectance/albedo/phase function data values (unitless)
    """
    prog, graphmode, sensor, comps, mixmodel, datamodel, fitmodel, param, tau, nsizes, wavelbounds, output = sys.argv # Unpack argument values
    graphmode = int(graphmode)


    # Try to extract r, G, and tau from the datamodel array; if they are NaNs, set them to their default values
    if datamodel[3] == 'NaN':
        r = 1 # R is a ratio and should be 1 by default
    else:
        r = float(datamodel[3])

    if datamodel[4] == 'NaN':
        G = 1
    else:
        G = float(datamodel[4])

    if tau == 'NaN':
        tau = 0
    else:
        tau = float(tau)
    
    if fitmodel == 'Semi-Empirical Mie':
        x0 = float(datamodel[5])
    elif fitmodel == 'Pure Mie':
        x0 = np.inf # Pure Mie treats all particles as small, using Mie theory for perfect spheres
    else:
        x0 = 0 # All other components of Mie treat all particles as large to use only the selected components

    # nsizes also needs to be checked for nans as it only applies to semi-empirical Mie, pure Mie, and Mie scattering
    if nsizes == ['NaN', 'NaN']:
        nsizes = pvars.NSIZE # It won't be used anyway

    # If surface reflectance mode, create an array of wavelengths and calculate reflectances using surface formulas
    if graphmode == 1:
        wavel_range = pvars.INSTRUMENTS[sensor] # Get range of all available wavelengths in the sensor
        xarr = np.arange(round(max(wavelbounds[0], wavel_range[0]), 5), round(min(wavelbounds[1], wavel_range[1]), 5) + 1e-20, 0.01) # Assume a 10 nm resolution. Get the most limiting bounds
        reflectances = utils.fresn_surface_reflectances(xarr, param, comps=comps, mixmodel=mixmodel)
    # If reflectance vs. angle mode, create an array of all colatitude angles and call the semi-empirical Mie theory
    elif graphmode == 2:
        xarr = np.arange(0, 181, 1) if output[0] == 'Scattering Angle' else np.arange(180, -1, -1) # Declare array of angles 0 to 180 degrees (or 180 to 0 degrees if phase)
        # Get reflectances. Programmed so that angles start, stop, and step the same way as xarr
        reflectances = utils.angle_mie_reflectances(float(datamodel[0]), float(datamodel[1]), float(datamodel[2]), r=r, G=G, x0=x0, theta_min=0, theta_max=180, wavels=[param], comps=comps, mixmodel=mixmodel, nsize=nsizes, tau=tau, section=fitmodel, output=output[1])[0][0]

        # For Henyey-Greenstein, curve fitting is needed to determine global scale factor
        if fitmodel == 'Henyey-Greenstein':
            # Iterate through given size distributions and see if input size distribution matches.
            # If so, get the HG parameters for that size distribution
            init_hg_params = []
            for model in pvars.SIZEDISTS.values():
                if datamodel == model[0]:
                    init_hg_params = model[1]
                    break
            
            # Define a function to use in curve-fitting HG function to match Mie via scale factor
            def hg_iterator(angles, scale_factor):
                hg_ifs = utils.henyey_greenstein(xarr, [[init_hg_params[1], init_hg_params[0] * scale_factor], [init_hg_params[3], init_hg_params[2] * scale_factor]])
                return np.log10(hg_ifs) # Will be fit on a log scale

            # Try to best-fit the Henyey-Greenstein function to the Mie function via a scale factor
            try:
                popt, _ = sp.optimize.curve_fit(
                    hg_iterator, 
                    xarr, 
                    np.log10(reflectances), # Log of original dataset is being fit with log of best-fit dataset
                    p0=[1e-6], # Standard scale for HG weights
                    maxfev=10000,
                )
                factor = popt.tolist()[0]
            except Exception as e:
                print(f"Curve fitting failed: {e}")
                factor = 1e-6

            # Final Henyey-Greenstein reflectances
            reflectances = utils.henyey_greenstein(np.radians(xarr), [[init_hg_params[1], init_hg_params[0] * factor], [init_hg_params[3], init_hg_params[2] * factor]])
    # If reflectances vs. wavelengths mode, get the wavelength range from the sensor and use that in the semi-empirical Mie methods
    elif graphmode == 0:
        wavel_range = pvars.INSTRUMENTS[sensor] # Plot over range of all available wavelengths in the sensor
        xarr = np.arange(round(max(wavelbounds[0], wavel_range[0]), 5), round(min(wavelbounds[1], wavel_range[1]), 5) + 1e-20, 0.01) # Assume a 10 nm resolution. Get the most limiting bounds
        reflectances = utils.angle_mie_reflectances(float(datamodel[0]), float(datamodel[1]), float(datamodel[2]), r=r, G=G, x0=x0, theta_min=param, theta_max=param, wavels=xarr, comps=comps, mixmodel=mixmodel, nsize=nsizes, tau=tau, section=fitmodel, output=output[1])[0]
        # angle_mie_reflectances returns a weird column array here, with shape [[element1], [element2], [element3]]
        # Need to make array not as deep
        reflectances = [element[0] for element in reflectances]

    # Create a shared memory to transmit data to gui.py
    shm = shared_memory.SharedMemory(create=True, size=xarr.nbytes + reflectances.nbytes, name="sim_data")
    # Create x and y arrays within memory
    dat_list = np.ndarray((2, xarr.shape[0],), dtype=np.float64, buffer=shm.buf)
    # Write to memory
    dat_list[0, :] = xarr[:]
    dat_list[1, :] = reflectances[:]