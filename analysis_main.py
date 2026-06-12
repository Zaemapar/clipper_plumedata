# Import all the required libraries
# If PyMieScatt not installed add following line:
# !pip install PyMieScatt

import numpy as np
import scipy as sp
import scipy.optimize
import csv
import matplotlib.pyplot as plt
import PyMieScatt as PyMie
import os
import sys

# UNIVERSAL SIMULATION PARAMETERS
DATA_PATH = "ering.csv"
# Angle scan bounds. For use in fitting only, final plots will plot over all available theta
ANGLE_LOWERBOUND = 1
ANGLE_UPPERBOUND = 10

# MIE DEFAULT PARAMETERS
# Min and max particle sizes to scan for in Mie distribution
S_MIN = 1
S_MAX = 10
# Min and max power law to scan for in Mie distribution
POWLAW_MIN = 2
POWLAW_MAX = 4
NSIZE=51 # Number of sizes to include in distribution (resolution)
COMP = 'Water Ice' # Composition of material. Currently only this is supported.
WAVEL = 0.647 # Given reading wavelength of instrument
WAVEL_SNAPSHOT_ANGLE = 20 # Once size distribution is determined at WAVEL, angle will be fixed to this and wavelength varied to examine effects

# WAVELENGTH PLOT PARAMETERS. FOR MIE TESTS ONLY, AFTER APPROPRIATE PARAMETERS HAVE BEEN COMPUTED.
MIN_WAVEL_TEST = 0.05
MAX_WAVEL_TEST = 5
WAVEL_STEP = 0.05

# HENYEY-GREENSTEIN PARAMETER BOUNDS
# Gs are the asymmetry parameters
GMIN = 0
GMAX = 1
# Ws are the term weights
WMIN = 0
WMAX = 10e-5

def wb08read():
    """
    Extracts information on complex index of refraction from provided datasheets for a given material.

    :returns: Numpy array for given wavelength,
              Numpy array for n values (refractive indices) at each wavelength,
              Numpy array for k values (absorbtion coefficients) at each wavelength
    """
    wavew, nw, kw, temp = [], [], [], []

    with open('WB08_Iceconstants.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader) # Skip header row
        #line=line.strip(',')
        #print(line)
        for line in reader:
            # Append wavelength, real part n, and imaginary part k
            wavew.append(float(line[0])) 
            nw.append(float(line[1]))
            kw.append(float(line[2]))

    # Cast as numpy arrays for easier interpolation later
    wavew=np.array(wavew)
    nw=np.array(nw)
    kw=np.array(kw)
    return wavew,nw,kw

def get_nk(wavel,comp):
    """
    Determines complex index of refraction n + ki of a material given its composition and the incident wavelength.

    :param wavel: Float wavelength in microns
    :param comp: String composition of material (currently only 'Water Ice' is supported.)
    :returns: Float n (refractive index) at the given wavelength
              Float k (absorption coefficient) at the given wavelength
    """
    if comp == 'Water Ice':
        # Calling function to read appropriate tables of optical constants for water ice
        wavex,nx,kx=wb08read()
    
    # Interpolate to given wavelength
    nxfunc=sp.interpolate.interp1d(wavex,nx)
    kxfunc=sp.interpolate.interp1d(wavex,kx)
    n=nxfunc(wavel)
    k=kxfunc(wavel)
    return n,k

def get_plt_data(src):
    """
    Reads data from a given sample and extracts scattering angle and corresponding I/F reflectance.

    :param src: String path to data file
    :returns: Array for theta values,
              Array for corresponding reflectance values
    """
    # Empty arrays for data
    avg_thet = []
    reflectance = []

    with open(DATA_PATH, "r") as f:
        reader = csv.reader(f)
        next(reader) # Skip header row

        for line in reader:
            # Append scattering angle and reflectance
            avg_thet.append(float(line[0]))
            reflectance.append(float(line[1]))

    return [180 - i for i in avg_thet], reflectance

def angle_mie_reflectances(smin, smax, powlaw, m, given_ifs, minangle, maxangle, theta_min=ANGLE_LOWERBOUND, theta_max=ANGLE_UPPERBOUND, wavel=WAVEL, nsize=NSIZE, tau=None):
    """
    Computes curve-fit I/F reflectance using Mie scattering theory, given a power law and a range of sizes and angles.
    First guesses tau based on initial reflectance data and scales best-fit line accordingly. Due to the tendency of
    Mie theory to produce unusual dips at large scattering angles, data is clamped to be greater than or equal to the
    last reflectance.

    :param smin: Float minimum size used in size distribution
    :param smax: Float maximum size used in size distribution
    :param powlaw: Float power law used in size distribution
    :param m: Complex index of refraction for material
    :param given_ifs: Initial reflectances from which to compute optical depth tau
    :param minangle: Integer minimum angle provided in original data
    :param maxangle: Integer maximum angle provided in original data
    :param theta_min: Integer minimum angle with which to curve fit to original data
    :param theta_max: Integer maximum angle with which to curve fit to original data
    :param wavel: Float wavelength at which to evaluate Mie
    :param nsize: Integer number of particle sizes to use (resolution)
    :param tau: Float allowing for tau to be input, rather than calculated
    :returns: Array of reflectance values sliced between theta_min and theta_max,
              Float for calculated optical depth tau
    """
    # If a small enough angle, don't care about clamping or calculating Mie for large scattering angles
    if theta_max <= 45:
        maxangle = theta_max

    # Compute the nominal (unnormalized) particle size distribution
    radii=np.linspace(smin, smax, nsize)
    #dr=radii[1]-radii[0]
    diameters=radii*2
    sizedists=radii**(-1*powlaw)

    # Use PyMieScatt to compute scattering angles and intensities
    # Takes wavelength & diameters in nanometers, so need to convert from microns
    theta1, SL, SR, SU = PyMie.SF_SD(m, wavel*1000, diameters*1000, sizedists,
                                     minAngle=minangle, maxAngle=maxangle,
                                     angularResolution=1.0, space='theta')

    # Calculate coefficients given the data, also applying conversions
    qdict = PyMie.Mie_SD(m, wavel*1000., diameters*1000, sizedists, asDict=True)
 
    # Extract extinction coefficient bext
    bsca = qdict['Bsca'] * 1.e6

    # Extract extinction coefficient. PyMieScatt applies 10^-6 scale factor assuming we
    # input size distribution in inverse cubic centimeters, and so outputs inverse
    # megameters. But we input inverse cubic microns, and we want our output in inverse
    # meters. This will work perfectly, but we do need to undo the 10^-6 scale factor.
    bext = qdict['Bext'] * 1.e6

    # Find indexes of angles to return for fit
    thetamin_idx = np.argmin(np.abs(theta1 - (theta_min * np.pi / 180)))
    thetamax_idx = np.argmin(np.abs(theta1 - (theta_max * np.pi / 180)))

    # Calculate tau from first available reflectance if not provided
    # Note that given_ifs will be sliced such that the first element is at theta_min, so SU is indexed accordingly
    if tau is None:
        tau = given_ifs[0] * bext * 4 * np.pi / (SU[thetamin_idx] * (wavel*1000)**2)

    # Calculate reflectance using scattering data, converting wavelength to nanometers. SU
    # is in the same units as sizedists, meaning we have square nanometers over cubic microns,
    # which is the same as inverse meters. Our units thus cancel assuming tau and reflectances
    # should be unitless.
    # If done correctly, the reflectances at index thetamin_idx should equal the given reflectance at that angle
    reflectances=np.asarray(SU)*(wavel*1000)**2/(4*np.pi)*tau/bext

    if theta_max > 45:
        # Clamp all reflectances below the last value
        last_reflectance = reflectances[-1]
        dip_idxs = np.where(reflectances < last_reflectance)[0]
        reflectances[dip_idxs] = last_reflectance

    # Return a slice of reflectances at the appropriate angles, as well as tau
    return reflectances[thetamin_idx:thetamax_idx + 1], tau

def henyey_greenstein(angle, gweights):
    """
    Computes probability density of light scattering at a given angle using a two-term Henyey-Greenstein function
    with provided weights g (asymmetry parameter) and w (unnormalized term weight)

    :param angle: Integer angle at which to calculate probability density
    :param gweights: Array of [g, w] parameters for each term. Shape (2, 2)
    :returns: Float unnormalized probability density of light scattering at the given angle
    """
    hg_term = 0   
    
    # Loop over gweights (each contains info about a term of the function)
    for term in gweights:
        g = term[0]
        w = term[1]
        hg_term += w/(4*np.pi)*(1-g**2)/(1+g**2-2*g*np.cos(angle))**1.5
    return hg_term

def RMSE(y_act, y_pred):
    """
    Computes the Root Mean Square Error between a given dataset and its best-fit curve

    :param y_act: Array of original data points
    :param y_pred: Array of predicted data points from best-fit curve
    :returns: Float for RMSE of dataset and best-fit curve
    """
    # Root mean square error
    return np.sqrt(np.mean((y_act-y_pred)**2))

import numpy as np

def diff_analysis(y_act, y_pred, angles):
    """
    Computes various statistical parameters to determine if the optimized fit is accurate.
    Uses R-squared to determine overall goodness-of-fit and notifies user if fit is terrible.
    Uses MAPE to determine if fit significantly and potentially erroneously deviates from actual values.
    Uses Z-scores to determine if and where significant spikes in error occur.

    :param y_act: Array of original data points
    :param y_pred: Array of predicted data points from best-fit curve
    :param angles: Array of x-coordinate angles, used to identify where spikes occur
    """
    # Casting as numpy arrays for easy manipulation
    y_act = np.asarray(y_act)
    y_pred = np.asarray(y_pred)
    angles = np.asarray(angles)
    
    # 1. R-Squared (Coefficient of Determination)
    ss_res = np.sum((y_act - y_pred)**2) # Sum of squares of residuals
    ss_tot = np.sum((y_act - np.mean(y_act))**2) # Total sum of squares
    r_squared = 1 - (ss_res / ss_tot)
    
    # 2. MAPE (Mean Absolute Percentage Error)
    # Adding a tiny epsilon (1e-10) to y_act to prevent division by zero
    mape = np.mean(np.abs((y_act - y_pred) / (y_act + 1e-10))) * 100
    
    # Evaluate Global Fit
    print("\n--- Global Fit Analysis ---")
    if r_squared < 0.5:
        print(f"CRITICAL WARNING: Terrible global fit! R^2 is incredibly low ({r_squared:.2f}).")
    else:
        print(f"Fit Quality: R^2 = {r_squared:.4f}")
        
    if mape > 25.0:  # Adjust this threshold based on your needs
        print(f"CRITICAL WARNING: Fit deviates by an average of {mape:.1f}%. Massive inaccuracy detected.")
    else:
        print(f"Average Error: {mape:.1f}%")

    # Determine if there are spikes in error/spikes in fit
    print("--- Local Spike Analysis ---")
    raw_diffs = y_act - y_pred
    mean_raw = np.mean(raw_diffs)
    std_raw = np.std(raw_diffs)

    if std_raw == 0:
        print("No variance in error. Perfect relative smoothness.")
        return

    # Calculate Z-scores. If Z-score is greater than 3, that usually indicates a spike
    zscores = np.abs((raw_diffs - mean_raw) / std_raw)
    points = np.where(zscores >= 3)[0]
    
    if len(points) > 0:
        spike_angles = np.round(angles[points], 2)
        print(f"WARNING: Fit model suddenly fails to track data at theta={spike_angles}")
    else:
        print("Curve behaves consistently (no sudden relative spikes detected).")

def mkplt(folder, theta, if_orig, if_fit, params, use_mie):
    """
    Plots and saves a graph of original and best-fit Mie or Henyey-Greenstein reflectances over various angles.
    Displays parameters (particle size distribution for Mie, weights and asymmetry parameters for Henyey-Greenstein)
    necessary to produce the displayed fit, as well as overall RMSE.

    :param folder: String folder in which to save graph
    :param theta: Array of thetas in degrees, to be plotted along the x-axis
    :param if_orig: Array of original reflectances, interpolated to thetas
    :param if_fit: Array of reflectances from line of best fit
    :param params: Array of parameters required to produce given fit. Shape varies based on model used.
    :param use_mie: Boolean for determining whether Mie or Henyey-Greenstein methods were used.
    """
    # Plot setup - using a log scale plot, plus a space for text to display parameters
    fig, (iflogplt, txt) = plt.subplots(1, 2, figsize=(12, 8), layout='constrained')

    # Calculate overall error between datasets
    plt_RMSE = RMSE(if_orig, if_fit)
    print(f"\nRMSE ({ANGLE_LOWERBOUND}-{ANGLE_UPPERBOUND} degrees): {plt_RMSE}")

    # Run error analysis to determine accuracy of fit.
    # For analysis purposes, we use linear versions of the datasets to make sure the tests run accurately (i.e. percent error)
    diff_analysis(if_orig, if_fit, theta)

    # Plotting the given datasets
    iflogplt.plot(theta, if_orig, color='black', label=f'Reflectance from data')
    iflogplt.plot(theta, if_fit, color='blue', label=f"Computed {'Mie' if use_mie else 'Henyey-Greenstein'} (RMSE={plt_RMSE:.2e})")

    # Cosmetic setup
    iflogplt.set_xlabel('Scattering Angle (degrees)')
    iflogplt.set_ylabel('Reflectance (I/F)')
    iflogplt.set_title(f'{DATA_PATH[:-4]} Reflectance vs Scattering Angle')
    iflogplt.grid()
    iflogplt.legend()
    iflogplt.set_yscale('log')
    iflogplt.set_xscale('log')

    # Fill in textbox with necessary parameters
    if use_mie:
        txt.text(0.25, 0.5, f"Mie Params:\nsmin={params[0]:.4f}\nsmax={params[1]:.4f}\npowlaw={params[2]:.4f}\ntau={params[3]:.2e}", bbox=dict(facecolor='lightblue', alpha=0.5, edgecolor='black', pad=10), fontsize=15)
    else:
        txt.text(0.25, 0.5, f"HG Params:\nw1={params[0]:.2f}\ng1={params[1]:.2f}\nw2={params[2]:.2f}\ng2={params[3]:.2f}\nwscale={params[4]:.2e}", bbox=dict(facecolor='lightblue', alpha=0.5, edgecolor='black', pad=10), fontsize=15)
    
    # Save to file
    fig_path = os.path.join(folder, f"{DATA_PATH[:-4]}_{'miefit' if use_mie else 'hgfit'}.png")
    plt.savefig(fig_path)
    print(f"\nGraph fit saved to {fig_path}")

def wavel_plot(folder, wavels, params, angles, given_ifs, angle=WAVEL_SNAPSHOT_ANGLE):
    """
    Plots and saves a graph of reflectances varied over different wavelenghts at a fixed scattering angle, 
    using the size distribution calculated with Mie theory. As only Mie theory is dependent on wavelength, 
    this graph will only be generated if Mie theory is used.

    :param folder: String folder in which to save graph
    :param wavels: Array of wavelengths at which to extract reflectance data
    :param params: Array of particle distribution parameters calculated from Mie theory.
    :param given_ifs: Array of original reflectances. Not used, but required by angle_mie_reflectances
    :param angle: Integer scattering angle at which to loop over wavelengths
    """
    reflectances = []
    # Find the index of the given scattering angle
    angle_idx = np.where(angles == angle)[0][0]
    for w in wavels:
        # Compute new complex index of refraction at each wavelength
        n, k = get_nk(w, COMP)
        m = complex(n, k)
        # Get reflectance at singular angle using given Mie parameters, passing existing tau
        reflectance_wavel, _ = angle_mie_reflectances(params[0], params[1], params[2], m, [given_ifs[angle_idx]], angle, angle, theta_min=angle, theta_max=angle, wavel=w, tau=params[3])
        reflectances.append(reflectance_wavel[0]) # Get first element of single-element array returned

    # Plot setup, plus a space for text to display parameters
    fig, (ifplt, txt) = plt.subplots(1, 2, figsize=(12, 8), layout='constrained')

    # Plot the computed reflectances
    ifplt.plot(wavels, reflectances, color='black', label=f'Computed Mie')

    # Cosmetic setup
    ifplt.set_xlabel('Wavelength (microns)')
    ifplt.set_ylabel('Reflectance (I/F)')
    ifplt.set_title(f'{DATA_PATH[:-4]} Reflectance vs Wavelength')
    ifplt.grid()
    ifplt.legend()

    # Fill in textbox with necessary parameters
    txt.text(0.25, 0.5, f"Mie Params:\nsmin={params[0]:.4f}\nsmax={params[1]:.4f}\npowlaw={params[2]:.4f}\ntau={params[3]:.2e}", bbox=dict(facecolor='lightblue', alpha=0.5, edgecolor='black', pad=10), fontsize=15)

    # Save to file
    fig_path = os.path.join(folder, f"{DATA_PATH[:-4]}_wavels_miefit.png")
    plt.savefig(fig_path)
    print(f"\nWavelength I/F plot saved to {fig_path}")

if __name__ == "__main__":
    # Program takes one extra argument: which model to use
    args = sys.argv
    if len(args) != 2:
        raise RuntimeError("""Usage: python3 {os.path.basename(__file__)} <theory>
                           Example: python3 {os.path.basename(__file__)} mie
                           Arguments:
                           \t<theory> Which theory to use to estimate data. Takes either mie for Mie theory or hg for a Henyey-Greenstein function.""")
    if args[1] == "mie":
        use_mie = True
    elif args[1] == "hg":
        use_mie = False

    # Define the path for the output directory
    graph_dir = "fit_tests"

    # Create the directory to contain the plots
    try:
        os.makedirs(graph_dir, exist_ok=True)
    except OSError as error:
        print(f"Error creating directory: {error}") # Error handling

    # Extract original angles and reflectances from data path
    theta1_orig, reflectances_orig = get_plt_data(DATA_PATH)

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
    valid_tests = np.where((theta1 <= ANGLE_UPPERBOUND) & (theta1 >= ANGLE_LOWERBOUND))[0]

    # Convert theta to radians for Henyey_Greenstein
    theta1_rad = theta1 * np.pi / 180

    # Slice along valid indices
    valid_angles = theta1_rad[valid_tests]
    valid_reflectances = reflectances[valid_tests]

    # --- MIE MODEL ---
    if use_mie:
        # Mie function handling arguments, used in curve fitting
        def mie_iterator(thetas, smin, smax, powlaw):
            # Reject invalid size combinations
            if smin > smax:
                return np.full_like(thetas, 1e10)
            # Compute reflectances based on constant min_theta, max_theta, m, and valid reflectances
            # thetas is not used because angle_mie_reflectances recreates it through SF_SD.
            # This cannot be avoided.
            reflectances, _ = angle_mie_reflectances(smin, smax, powlaw, m, valid_reflectances, min_theta, max_theta)
            return np.log10(reflectances) # Fit will be done to log scale

        # Obtain the optical constants at the desired wavelength
        n,k=get_nk(WAVEL, COMP)

        # Compute the reflectance (I/F) of the particle population.
        # Note this uses the same normalization process as described in Appendix B of
        # de Pater et al. 2026 Characterization of the Outer Uranian Rings.... in JGR Planets 131 e2025JE009404.
        # doi.org/10.1029/2025JE00404
        m = complex(n, k) # Compute complex refractive index

        # Provide initial guesses and bounds (with slight asymmetry to avoid singular jacobian)
        p0 = [S_MIN + (S_MAX-S_MIN)/3, S_MIN + (S_MAX-S_MIN)/2, POWLAW_MIN + (POWLAW_MAX-POWLAW_MIN)/4]
        bounds = ([S_MIN, S_MIN, POWLAW_MIN], 
                [S_MAX, S_MAX, POWLAW_MAX])

        try:
            # Fit in log space so the tail isn't ignored
            popt, _ = sp.optimize.curve_fit(
                mie_iterator, 
                valid_angles, 
                np.log10(valid_reflectances), # Log of original dataset is being fit with log of best-fit dataset
                p0=p0, 
                bounds=bounds,
                maxfev=10000,
                diff_step=0.1, # If too small, optimization fails to progress due to noise in Mie
                x_scale=[S_MAX, S_MAX, POWLAW_MAX] # Help the optimizer understand the scale
            )
            params = popt.tolist()
        except Exception as e:
            print(f"Curve fitting failed: {e}")
            params = p0

        # Retrieve the tau corresponding to the fitted parameters
        _, optimal_tau = angle_mie_reflectances(params[0], params[1], params[2], m, valid_reflectances, min_theta, max_theta)
        params.append(optimal_tau) # append to parameter list

        # Then get the full line across all angles
        plt_reflectances, _ = angle_mie_reflectances(params[0], params[1], params[2], m, reflectances, min_theta, max_theta, theta_min=theta1[0], theta_max=theta1[-1], tau=optimal_tau)

        print(f"Done\nOptimized distribution:\nsmin={params[0]}\nsmax={params[1]}\npowlaw={params[2]}\ntau={params[3]}")

        # Generate a plot of reflectances vs wavelengths for these parameters
        wavels = np.arange(MIN_WAVEL_TEST, MAX_WAVEL_TEST + WAVEL_STEP, WAVEL_STEP)
        wavel_plot(graph_dir, wavels, params, theta1, reflectances)

    # --- HENYEY-GREENSTEIN MODEL ---
    else:
        # Henyey-Greenstein function handling arguments, used in curve fitting
        def hg_model(angle, w1, g1, w2, g2):
            # Mainly just reformatting input parameters
            return np.log10(henyey_greenstein(angle, [[g1, w1], [g2, w2]])) # Fit will be done to log scale

        # Provide initial guesses and bounds (with slight asymmetry to avoid singular jacobian)
        p0 = [WMAX/2, 0.9, WMAX/3, 0.5]
        bounds = ([WMIN+1e-10, GMIN, WMIN+1e-10, GMIN], 
                    [WMAX, GMAX, WMAX, GMAX])
            
        try:
            # Fit in log space so the tail isn't ignored
            popt, _ = sp.optimize.curve_fit(
                hg_model, 
                valid_angles, 
                np.log10(valid_reflectances), # Log of original dataset is being fit with log of best-fit dataset
                p0=p0, 
                bounds=bounds,
                maxfev=10000,
                x_scale=[WMAX, GMAX, WMAX, GMAX] # help the optimizer understand the scale
            )
            params = popt.tolist()
        except Exception as e:
            print(f"Curve fitting failed: {e}")
            params = p0
        
        # Compute full array of fitted values
        plt_reflectances = henyey_greenstein(theta1_rad, [[params[1], params[0]], [params[3], params[2]]])

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
    mkplt(graph_dir, theta1, reflectances, plt_reflectances, params, use_mie)