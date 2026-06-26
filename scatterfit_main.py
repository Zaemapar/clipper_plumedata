# Import all the required libraries
# If PyMieScatt not installed add following line:
# !pip install PyMieScatt

import numpy as np
import scipy as sp
import scipy.optimize
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

    # Interpolate to evenly spaced theta range
    orig_interp = sp.interpolate.interp1d(theta1_orig, reflectances_orig)
    reflectances = orig_interp(theta1)

    # Get a list of valid angles to test, within angle bounds
    valid_tests = np.where((theta1 <= vars.ANGLE_UPPERBOUND) & (theta1 >= vars.ANGLE_LOWERBOUND))[0]

    # Convert theta to radians for Henyey_Greenstein
    theta1_rad = theta1 * np.pi / 180

    # Slice along valid indices
    valid_rads = theta1_rad[valid_tests]
    valid_degs = theta1[valid_tests]
    valid_reflectances = reflectances[valid_tests]

    # --- MIE ANGLE MODEL ---
    if use_mie and vars.ALTITUDE is None:
        # Get reference reflectance, for use in calculating tau
        ref_if_idx = np.where(theta1 == vars.ANGLE_LOWERBOUND)[0]
        reflectance_ref = reflectances[ref_if_idx]

        # Mie function handling arguments, used in curve fitting
        def mie_iterator(thetas, smin, smax, powlaw):
            # Reject invalid size combinations
            if smin > smax:
                return np.full_like(thetas, np.nan)
            # Compute reflectances based on constant min_theta, max_theta, m, and valid reflectances
            reflectances = utils.angle_mie_reflectances(smin, smax, powlaw, ref_if=reflectance_ref)[0][0]
            return np.log10(reflectances) # Fit will be done to log scale

        # Obtain the optical constants at the desired wavelength
        n,k=utils.get_nk([vars.WAVEL], {vars.COMP: 1}, returnmode=['single', vars.WAVEL])

        # Compute the reflectance (I/F) of the particle population.
        # Note this uses the same normalization process as described in Appendix B of
        # de Pater et al. 2026 Characterization of the Outer Uranian Rings.... in JGR Planets 131 e2025JE009404.
        # doi.org/10.1029/2025JE00404
        m = complex(n, k) # Compute complex refractive index

        # Provide initial guesses and bounds (with slight asymmetry to avoid singular jacobian)
        p0 = [vars.S_MIN + (vars.S_MAX-vars.S_MIN)/4, vars.S_MIN + (vars.S_MAX-vars.S_MIN)/3, vars.POWLAW_MIN + (vars.POWLAW_MAX-vars.POWLAW_MIN)/2]
        bounds = ([vars.S_MIN, vars.S_MIN, vars.POWLAW_MIN], 
                [vars.S_MAX, vars.S_MAX, vars.POWLAW_MAX])

        try:
            # Fit in log space so the tail isn't ignored
            popt, _ = sp.optimize.curve_fit(
                mie_iterator, 
                valid_degs, 
                np.log10(valid_reflectances), # Log of original dataset is being fit with log of best-fit dataset
                p0=p0, 
                bounds=bounds,
                maxfev=10000,
                diff_step=0.1, # If too small, optimization fails to progress due to noise in Mie
                x_scale=[vars.S_MAX, vars.S_MAX, vars.POWLAW_MAX] # Help the optimizer understand the scale
            )
            params = popt.tolist()
        except Exception as e:
            print(f"Curve fitting failed: {e}")
            params = p0

        # Retrieve the tau corresponding to the fitted parameters
        optimal_tau = utils.angle_mie_reflectances(params[0], params[1], params[2], ref_if=reflectance_ref)[1][0][0]
        params.append(optimal_tau) # append to parameter list

        # Then get the full line across all angles
        plt_reflectances = utils.angle_mie_reflectances(params[0], params[1], params[2], theta_min=theta1[0], theta_max=theta1[-1], tau=optimal_tau)[0][0]

        print(f"Done\nOptimized distribution:\nsmin={params[0]}\nsmax={params[1]}\npowlaw={params[2]}\ntau={params[3]}")

        # Generate a plot of reflectances vs wavelengths for these parameters
        wavels = np.arange(vars.MIN_WAVEL_TEST, vars.MAX_WAVEL_TEST + vars.WAVEL_STEP, vars.WAVEL_STEP)
        utils.wavel_plot(graph_dir, wavels, params)

    # --- HENYEY-GREENSTEIN ANGLE MODEL ---
    elif not use_mie and vars.ALTITUDE is None:
        # Henyey-Greenstein function handling arguments, used in curve fitting
        def hg_model(angle, w1, g1, w2, g2):
            # Mainly just reformatting input parameters
            return np.log10(utils.henyey_greenstein(angle, [[g1, w1], [g2, w2]])) # Fit will be done to log scale

        # Provide initial guesses and bounds (with slight asymmetry to avoid singular jacobian)
        p0 = [vars.WMAX/2, 0.9, vars.WMAX/3, 0.5]
        bounds = ([vars.WMIN+1e-10, vars.GMIN, vars.WMIN+1e-10, vars.GMIN], 
                    [vars.WMAX, vars.GMAX, vars.WMAX, vars.GMAX])
            
        try:
            # Fit in log space so the tail isn't ignored
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