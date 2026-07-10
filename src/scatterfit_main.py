"""
A program for computing the best-fit semi-empirical Mie distribution or Henyey-
Greenstein function for a given dataset of reflectance vs. scattering angle, outputting
all optimized fit parameters and a graph of the best-fit curve superimposed on the original
dataset.

Author: Parker A. Zaemann
Date: 06 Jul 2026
Source: https://github.com/Zaemapar/clipper_plumedata
Contact: mhedman@uidaho.edu
"""
# Import all the required libraries
import numpy as np
import scipy as sp
import csv
import matplotlib.pyplot as plt
import PyMieScatt as PyMie
import utils
import fit_vars as vars
import os
import sys

if __name__ == "__main__":
    # Read which model to use
    if vars.MODEL == "mie":
        use_mie = True
    elif vars.MODEL == "hg":
        use_mie = False
    else:
        raise RuntimeError("MODEL should be either mie (for Mie scattering theory) or hg (for Henyey-Greenstein function)")

    # Make a new graph directory to hold output graphs, if not created already
    graph_dir = utils.create_dirpath("scatterfit_main")

    # Extract original angles and reflectances from data path
    theta1_orig, reflectances_orig = utils.get_data(vars.DATA_PATH)

    # Sort and zip in case of scrambled values
    sorted_pairs = sorted(zip(theta1_orig, reflectances_orig))
    theta1_orig, reflectances_orig = zip(*sorted_pairs)

    # Cast as array for numpy manipulation
    theta1_orig = np.asarray(theta1_orig)
    reflectances_orig = np.asarray(reflectances_orig)

    # Round off thetas to get an appropriate range
    min_theta = np.ceil(theta1_orig[0])
    max_theta = np.floor(theta1_orig[-1])
    theta1 = np.arange(min_theta, max_theta + 1, 1) # Array of evenly spaced thetas, same as used in Mie SF_SD function

    # Interpolate reflectances to evenly spaced theta range
    orig_interp = sp.interpolate.interp1d(theta1_orig, reflectances_orig)
    reflectances = orig_interp(theta1)

    # Get a list of valid angles to test, within angle bounds
    valid_tests = np.where((theta1 <= vars.ANGLE_UPPERBOUND) & (theta1 >= vars.ANGLE_LOWERBOUND))[0]

    # Convert theta to radians for Henyey_Greenstein
    theta1_rad = theta1 * np.pi / 180

    # Slice angles and reflectances along valid indices
    valid_rads = theta1_rad[valid_tests]
    valid_degs = theta1[valid_tests]
    valid_reflectances = reflectances[valid_tests]

    # --- MIE ANGLE MODEL ---
    if use_mie and vars.ALTITUDE is None:
        # Get reference reflectance, for use in calculating tau. It must be at the lowest angle bound.
        ref_if_idx = np.where(theta1 == vars.ANGLE_LOWERBOUND)[0]
        reflectance_ref = reflectances[ref_if_idx]

        # Mie function handling arguments, used in curve fitting
        def mie_iterator(thetas, smin, smax, powlaw, r, G, x0):
            """
            A function to package Mie reflectance computation in a format a scipy curve fit function can handle.

            :param thetas: Integer list of scattering angles in degrees
            :param smin: Float minimum particle radius in microns
            :param smax: Float maximum particle radius in microns
            :param powlaw: Power law to use in distribution
            :param r: Float ratio of the surface area of an irregular particle to the surface area of a sphere of the same 
                      volume
            :param G: Float constant to use in Mie transmission equation
            :param x0: Float size cutoff above which to use semi-empirical methods and below which to use pure Mie
            :returns: Float list of log base 10 of I/F reflectances as computed by Mie theory. Shape len(thetas).
            """
            # Reject invalid size combinations
            if smin > smax:
                return np.full_like(thetas, np.nan)
            # Compute reflectances based on input parameters. R, G, and x0 are used in semi-empirical parts
            reflectances = utils.angle_mie_reflectances(smin, smax, powlaw, r=r, G=G, x0=x0, ref_if=reflectance_ref)[0][0]
            return np.log10(reflectances) # Fit will be done to log scale

        # Obtain the optical constants at the desired wavelength
        ns,ks=utils.get_nk([vars.WAVEL], {vars.COMP: 1})
        # There will only be one wavelength in each array
        n, k = ns[0], ks[0]
        m = complex(n, k) # Compute complex refractive index

        # Provide initial guesses and bounds (with slight asymmetry to avoid singular jacobian)
        # Trying to hit somewhere in the range of each parameter
        p0 = [vars.S_MIN + (vars.S_MAX-vars.S_MIN)/4, vars.S_MIN + (vars.S_MAX-vars.S_MIN)/3, vars.POWLAW_MIN + (vars.POWLAW_MAX-vars.POWLAW_MIN)/2, vars.R_MIN + (vars.R_MAX - vars.R_MIN)/3, vars.G_MIN + (vars.G_MAX - vars.G_MIN)/4, vars.x0_MIN]
        bounds = ([vars.S_MIN, vars.S_MIN, vars.POWLAW_MIN, vars.R_MIN, vars.G_MIN, vars.x0_MIN], 
                [vars.S_MAX, vars.S_MAX, vars.POWLAW_MAX, vars.R_MAX, vars.G_MAX, vars.x0_MAX])
        de_bounds = list(zip(bounds[0], bounds[1])) # Bounds for differential evolution

        # Difference of squares computation between initial y data and parameters. This will be minimized
        def sum_of_squares_mie(params, x_data, y_data_log):
            """
            A function to compare the results of the mie_iterator with the initial data y_data_log and output the
            sum of the square residuals. This will be minimized to find the right Mie fit for the data.

            :params: Tuple of float size distribution parameters (smin, smax, powlaw, r, G, x0)
            :param x_data: List of integer scattering angles in degrees
            :param y_data_log: List of float reflectances as read from a data file, sliced at valid angles. Shape
                               len(x_data)
            :returns: List of float sum of square residuals between log of Mie output and y_data_log
            """
            smin, smax, powlaw, r, G, x0 = params
    
            # Get the semi-empirical Mie output
            y_model_log = mie_iterator(x_data, smin, smax, powlaw, r, G, x0)
            
            # If the model kicked back NaNs (like if smin > smax), return a massive error penalty
            if np.any(np.isnan(y_model_log)):
                return np.inf
                
            # Return the sum of squared residuals
            return np.sum((y_data_log - y_model_log) ** 2)

        try:
            # The optimization curve fit function, which will iterate through all independent parameters and find the best combination
            # We use differential evolution to avoid the problem where changing x0 does not change the graph in certain regions
            popt = sp.optimize.differential_evolution(
                sum_of_squares_mie,
                de_bounds,
                args=(valid_degs, np.log10(valid_reflectances)),
                strategy='best1bin',
                maxiter=1000,
                popsize=15, # Increase if it struggles to find the global minimum
                tol=1e-3,
                mutation=(0.5, 1.0),
                recombination=0.7
            )
            params = popt.x.tolist() # Extract the list of fit parameters
        except Exception as e:
            print(f"Curve fitting failed: {e}")
            params = p0 # Fall back on the original point if failed

        # Retrieve the tau corresponding to the fitted parameters (matching the dataset's first reflectance and the fit's first reflectance)
        optimal_tau = utils.angle_mie_reflectances(params[0], params[1], params[2], r=params[3], G=params[4], x0=params[5], ref_if=reflectance_ref)[1][0][0]
        params.append(optimal_tau) # append to parameter list

        # Then get the full line across all angles
        plt_reflectances = utils.angle_mie_reflectances(params[0], params[1], params[2], r=params[3], G=params[4], x0=params[5], theta_min=theta1[0], theta_max=theta1[-1], tau=optimal_tau)[0][0]

        print(f"Done\nOptimized distribution:\nsmin={params[0]}\nsmax={params[1]}\npowlaw={params[2]}\nr={params[3]}\nG={params[4]}\nx0={params[5]}\ntau={params[6]}")

    # --- HENYEY-GREENSTEIN ANGLE MODEL ---
    elif not use_mie and vars.ALTITUDE is None:
        # Henyey-Greenstein function handling arguments, used in curve fitting
        def hg_model(angle, w1, g1, w2, g2):
            """
            A function to package Henyey-Greenstein reflectance computation in a format a scipy curve fit function 
            can handle.

            :param angle: List of integer scattering angles in degrees
            :param w1: The weight of the first term
            :param g1: The asymmetry parameter of the first term
            :param w2: The weight of the second term
            :param g2: The asymmetry parameter of the second term
            :returns: Float list of reflectances as computed by a Henyey-Greenstein function. Shape len(angle).
            """
            # Mainly just reformatting input parameters
            return np.log10(utils.henyey_greenstein(angle, [[g1, w1], [g2, w2]])) # Fit will be done to log scale

        # Provide initial guesses and bounds (with slight asymmetry to avoid singular jacobian)
        p0 = [vars.WMAX/2, 0.9, vars.WMAX/3, 0.5]
        bounds = ([vars.WMIN+1e-10, vars.GMIN, vars.WMIN+1e-10, vars.GMIN], 
                    [vars.WMAX, vars.GMAX, vars.WMAX, vars.GMAX])
            
        try:
            # Here we can use curve_fit because all of the HG parameters change the graph in all regions
            popt, _ = sp.optimize.curve_fit(
                hg_model, 
                valid_rads, 
                np.log10(valid_reflectances), # Log of original dataset is being fit with log of best-fit dataset
                p0=p0, 
                bounds=bounds,
                maxfev=10000,
                x_scale=[vars.WMAX, vars.GMAX, vars.WMAX, vars.GMAX] # help the optimizer understand the scale
            )
            params = popt.tolist()
        except Exception as e:
            print(f"Curve fitting failed: {e}")
            params = p0
        
        # Compute full array of fitted values
        plt_reflectances = utils.henyey_greenstein(theta1_rad, [[params[1], params[0]], [params[3], params[2]]])

        # Normalize weights
        w1 = params[0]
        w2 = params[2]
        scale_factor = w1 + w2

        params[0] = w1 / scale_factor
        params[2] = w2 / scale_factor
        # Add on the normalization factor, since that is important
        # It might have something to do with optical depth and extinction coefficient
        params.append(scale_factor)

        print(f"Done\nOptimized Henyey-Greenstein parameters:\nW1={params[0]}\nG1={params[1]}\nW2={params[2]}\nG2={params[3]}\nscale={params[4]}")

    # Make plot regardless of method used
    utils.mkplt(graph_dir, theta1, reflectances, plt_reflectances, params, use_mie)