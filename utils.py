import numpy as np
import scipy as sp
import scipy.optimize
import csv
import matplotlib.pyplot as plt
import PyMieScatt as PyMie
import fit_vars as vars
import pred_vars as pvars
import os
import sys

G = 5
r = 2

def wb08read(data_path):
    """
    Extracts information on complex index of refraction from provided datasheets for a given material.

    :param data_path: String file path for index of refraction data
    :returns: Numpy array for given wavelength,
              Numpy array for n values (refractive indices) at each wavelength,
              Numpy array for k values (absorbtion coefficients) at each wavelength
    """
    wavew, nw, kw, temp = [], [], [], []

    with open(data_path, 'r') as f:
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
    :param comp: String composition of material
    :returns: Float n (refractive index) at the given wavelength
              Float k (absorption coefficient) at the given wavelength
    """
    idx_refraction_path = os.path.join("data", comp + "_constants.csv")
    
    # Calling function to read appropriate tables of optical constants for water ice
    try:
        wavex,nx,kx=wb08read(idx_refraction_path)
    except FileNotFoundErrors:
        raise FileNotFoundError("Material type not supported.")
    
    # Interpolate to given wavelength
    nxfunc=sp.interpolate.interp1d(wavex,nx)
    kxfunc=sp.interpolate.interp1d(wavex,kx)
    n=nxfunc(wavel)
    k=kxfunc(wavel)
    return n,k

def get_minmax_wavel(comp):
    """
    Reads the same file as get_nk, but searches for min and max wavelength supported.

    :param comp: String composition of material
    :returns: Float minimum wavelength supported
              Float maximum wavelength supported
    """
    idx_refraction_path = os.path.join("data", comp + "_constants.csv")
    
    # Calling function to read appropriate tables of optical constants for water ice
    try:
        wavex,_,_=wb08read(idx_refraction_path)
        wavex.sort() # Sort just in case
        return wavex[0], wavex[-1] # Return min and max
    except FileNotFoundErrors:
        raise FileNotFoundError("Material type not supported.")

def get_data(src, altitude=None):
    """
    Reads data from a given sample and extracts scattering angle (or wavelength) and corresponding I/F reflectance (or optical depth tau).
    Parses header using string search to determine how to format the data.

    :param src: String path to data file
    :param altitude: Float altitude if reading wavelength data
    :returns: Array for theta/wavelength values,
              Array for corresponding reflectance/tau values
    """
    # Empty arrays for data
    thetas = []
    reflectances = []

    with open(src, "r") as f:
        reader = csv.reader(f)
        header = next(reader) # Grab header row

        # Parse through header to determine where relevant data is and how to scale it
        theta_idx = 0
        reflectance_idx = 0
        phase_theta = False
        if_scale_factor = 0
        if altitude is not None:
            theta_idx = 0 # Wavelengths are always listed in the left column
            reflectance_idx = np.where(np.asarray(header) == str(altitude))[0]
            if_scale_factor = -2 # Equivalent width is usually listed in those tables, to convert to tau divide by 100
        else:
            for idx, headstring in enumerate(header):
                lc_headstring = headstring.lower()
                # Check to see if it's an angle column
                if ("theta" in lc_headstring or "angle" in lc_headstring) and "range" not in lc_headstring:
                    theta_idx = idx
                    if "phase" in lc_headstring:
                        phase_theta = True
                # Check to see if it's a reflectance column
                elif "reflect" in lc_headstring or "contr" in lc_headstring:
                    reflectance_idx = idx
                    modifier_idx = lc_headstring.find("10^")
                    if modifier_idx > 0:
                        if_scale_factor = int(lc_headstring[(modifier_idx + 3):])

        # Iterate through each line in the rest of the document
        for line in reader:
            # Append scattering angle and reflectance
            thetas.append(float(line[theta_idx]) if not phase_theta else 180 - float(line[theta_idx]))
            reflectances.append(float(line[reflectance_idx]) * 10**if_scale_factor)

    return thetas, reflectances

def angle_mie_reflectances(smin, smax, powlaw, theta_min=vars.ANGLE_LOWERBOUND, theta_max=vars.ANGLE_UPPERBOUND, wavels=[vars.WAVEL], nsize=vars.NSIZE, comp='Water Ice', tau=None, ref_if=0):
    """
    Computes curve-fit I/F reflectance using Mie scattering theory, given a power law and a range of sizes, angles, and wavelengths.
    First guesses tau based on initial reflectance data and scales best-fit line accordingly. Due to the tendency of
    Mie theory to produce unusual dips at large scattering angles, data is clamped to be greater than or equal to the
    last reflectance.

    :param smin: Float minimum size used in size distribution
    :param smax: Float maximum size used in size distribution
    :param powlaw: Float power law used in size distribution
    :param given_ifs: Initial reflectances from which to compute optical depth tau
    :param theta_min: Integer minimum angle with which to curve fit to original data
    :param theta_max: Integer maximum angle with which to curve fit to original data
    :param wavel: Array of wavelengths at which to evaluate Mie
    :param nsize: Integer number of particle sizes to use (resolution)
    :param comp: Composition of material. Currently only Water Ice and Tholins are accepted.
    :param tau: Float allowing for optical detph tau to be input, rather than calculated
    :param ref_if: Float reflectance from original data from which to calculate optical depth tau. ref_if MUST correspond to theta_min.
    :returns: 2D Array of reflectance values between theta_min, theta_max, and the min and max of wavels. Shape (len(wavels), len(angles))
              Array for calculated optical depths at each wavelength
    """

    #Compute the nominal (unnormalized) particle size distribution
    radii=np.linspace(smin, smax, nsize)
    #dr=radii[1]-radii[0]
    diameters=radii*2
    sizedists=radii**(-1*powlaw)

    # Empty array to hold reflectances and taus
    reflectances = []
    taus = []

    x0 = 5000 # Cutoff size parameter to define small and large particles
    angle_range = np.radians(np.arange(theta_min, theta_max + 1, 1)) # Range of angles in radians
    solid_angles_degs = np.arange(0, 181, 1) # All solid angles, for integrals
    solid_angles = np.radians(solid_angles_degs)
    solid_min_idx = np.where(solid_angles_degs == theta_min)[0][0] # Locate angle bounds within solid angles
    solid_max_idx = np.where(solid_angles_degs == theta_max)[0][0]

    for wavel in wavels:
        sizeparams = 2 * np.pi * radii / wavel # This is the size parameter by which we gague small and large particles
        small_idxs = np.where(sizeparams <= x0)[0]
        large_idxs = np.where(sizeparams > x0)[0]
        small_sizes = sizeparams[small_idxs]
        large_sizes = sizeparams[large_idxs]
        small_diameters = diameters[small_idxs]
        large_diameters = diameters[large_idxs]
        small_dist = sizedists[small_idxs]
        large_dist = sizedists[large_idxs]

        # --- SMALL REGIME PARTICLES ---
        # Compute index of refraction at given wavelength
        n, k = get_nk(wavel, comp)
        m = complex(n, k)

        SU = np.zeros_like(angle_range)

        if len(small_idxs > 0):
            # Use PyMieScatt to compute scattering angles and intensities in the small regime
            # Takes wavelength & diameters in nanometers, so need to convert from microns
            theta1, SL, SR, SU_small = PyMie.SF_SD(m, wavel*1000, small_diameters*1000, small_dist,
                                        minAngle=theta_min, maxAngle=theta_max,
                                        angularResolution=1.0, space='theta')

            # Calculate coefficients given the data, also applying conversions
            qdict_small = PyMie.Mie_SD(m, wavel*1000., small_diameters*1000, small_dist, asDict=True)
            bsca_small = qdict_small['Bsca'] * 1.e6 # Extract extinction coefficient bext

            # Extract extinction coefficient. PyMieScatt applies 10^-6 scale factor assuming we
            # input size distribution in inverse cubic centimeters, and so outputs inverse
            # megameters. But we input inverse cubic microns, and we want our output in inverse
            # meters. This will work perfectly, but we do need to undo the 10^-6 scale factor.
            bext_small = qdict_small['Bext'] * 1.e6

            SU += SU_small

        if len(large_idxs > 0):
            # --- LARGE REGIME: DIFFRACTION ---
            P_large_diff = []

            for theta in solid_angles:
                # For z = x * sin(theta)
                z = large_sizes * np.sin(theta)

                # Handle the theta = 0 limit where J1(z)/z -> 1/2
                with np.errstate(divide='ignore', invalid='ignore'):
                    j1_term = sp.special.j1(z) / z
                    j1_term[z == 0] = 0.5  # L'Hopital's rule limit

                # Exact physical optics diffraction
                d_x_unnormalized = (large_sizes**2 / 4 / np.pi) * (2 * j1_term)**2 * 0.5 * (1 + np.cos(theta)**2)
                integrand = d_x_unnormalized * np.pi * large_sizes**2 * large_dist
                P_large_diff.append(np.trapz(integrand, large_sizes))
            
            # Normalizing so that integral over all solid angles equals 1
            diff_int_solidangles = 0.5 * np.trapz(P_large_diff * np.sin(solid_angles), solid_angles)
            P_large_diff = (np.asarray(P_large_diff) / diff_int_solidangles)[solid_min_idx:solid_max_idx+1]

            # --- LARGE REGIME: EXTERNAL REFLECTION ---
            P_large_extref = 0.5*((np.sin(solid_angles / 2) - (np.abs(m)**2 - 1 + np.sin(solid_angles / 2)**2)**0.5)/(np.sin(solid_angles / 2) + (np.abs(m)**2 - 1 + np.sin(solid_angles / 2)**2)**0.5))**2 + 0.5 * ((np.abs(m)**2 * np.sin(solid_angles / 2) - (np.abs(m)**2 - 1 + np.sin(solid_angles / 2)**2)**0.5)/(np.abs(m)**2 * np.sin(solid_angles / 2) + (np.abs(m)**2 - 1 + np.sin(solid_angles / 2)**2)**0.5))**2
            
            # Normalizing so that integral over all solid angles equals 1
            extref_int_solidangles = 0.5 * np.trapz(P_large_extref * np.sin(solid_angles), solid_angles)
            P_large_extref = (P_large_extref / extref_int_solidangles)[solid_min_idx:solid_max_idx+1] # Normalize so integral over all solid angles equals 1

            # --- LARGE REGIME: TRANSMISSION ---

            # I have no idea how b relates to G or whatever so I just guessed G
            P_large_trans = np.e * G**(-2*solid_angles/np.pi)

            # Normalizing so that integral over all solid angles equals 1
            trans_int_solidangles = 0.5 * np.trapz(P_large_trans * np.sin(solid_angles), solid_angles)
            P_large_trans = (P_large_trans / trans_int_solidangles)[solid_min_idx:solid_max_idx+1]

            # I have no idea how these combine but AI said so so I guess it's fine?
            # Something about how we can just add these intensities but they have some weird Q factor attached
            # Like for diffraction it's 0.5, external reflection something small, and transmission makes up the rest
            # Is this physics accurate?
            fresn_ref = extref_int_solidangles
            qdict_large = PyMie.Mie_SD(m, wavel*1000., diameters[large_idxs]*1000, sizedists[large_idxs], asDict=True)
            bsca_large = qdict_large['Bsca'] * 1.e6 # Extract extinction coefficient bext
            SU_large_diff = P_large_diff * 0.5 * bsca_large * np.pi / ((wavel*1000)**2)
            SU_large_extref = P_large_extref * 0.5 * fresn_ref * bsca_large * np.pi / ((wavel*1000)**2)
            SU_large_trans = P_large_trans * 0.5 * (1 - fresn_ref) * bsca_large * np.pi / ((wavel*1000)**2)

            SU += (SU_large_diff + SU_large_extref + SU_large_trans) * r # Apparently we have to scale it up by some factor that prioritizes larger particles since they have more "dents"


        # Calculate coefficients given the data, also applying conversions
        qdict_tot = PyMie.Mie_SD(m, wavel*1000., diameters*1000, sizedists, asDict=True)
        bsca_tot = qdict_tot['Bsca'] * 1.e6 # Extract extinction coefficient bext

        # Extract extinction coefficient. PyMieScatt applies 10^-6 scale factor assuming we
        # input size distribution in inverse cubic centimeters, and so outputs inverse
        # megameters. But we input inverse cubic microns, and we want our output in inverse
        # meters. This will work perfectly, but we do need to undo the 10^-6 scale factor.
        bext_tot = qdict_tot['Bext'] * 1.e6

        # Calculate tau from first available reflectance if not provided
        # Note that given_ifs will be sliced such that the first element is at theta_min, so SU is indexed accordingly
        if tau is None:
            tau = ref_if * bext_tot * 4 * np.pi / (SU[0] * (wavel*1000)**2)

        # Calculate reflectance using scattering data, converting wavelength to nanometers. SU
        # is in the same units as sizedists, meaning we have square nanometers over cubic microns,
        # which is the same as inverse meters. Our units thus cancel assuming tau and reflectances
        # should be unitless.
        # If done correctly, the reflectances at index thetamin_idx should equal the given reflectance at that angle
        reflectances.append(np.asarray(SU)*(wavel*1000)**2/(4*np.pi)*tau/bext_tot)
        taus.append(tau)

    # Return a slice of reflectances at the appropriate angles, as well as tau
    return reflectances, taus

def henyey_greenstein(angle, gweights):
    """
    Computes probability density of light scattering at a given angle using a two-term Henyey-Greenstein function
    with provided weights g (asymmetry parameter) and w (unnormalized term weight)

    :param angle: Integer angle IN RADIANS at which to calculate probability density
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

def output_graph(ifvangle, sensor, comp, datamodel, fitmodel, param, tau):
    """
    A function that takes input scattering data as might be observed by one of Europa Clipper's instruments,
    as well as additional data about the fit and the plot type desired, and outputs the corresponding x and y
    arrays.

    :param ifvangle: Boolean. True indicates output x array will be scattering angles, false means it will be
                     wavelengths.
    :param sensor: String sensor from which to read wavelength range
    :param comp: String composition of material
    :param datamodel: String data model to use, either 'G-ring-like' or 'E-ring-like'
    :param fitmodel: String for fit model to use, either 'Mie' or 'Henyey-Greenstein'
    :param param: Fixed parameter to base reflectances on. Float wavelength if ifvangle = True, int scattering
                  angle if False
    :param tau: Float optical depth

    :returns: List containing x data values (integer if scattering angles, float if wavelengths)
              List containing float reflectance data values
    """
    if datamodel == "G-ring-like":
        type_idx = 0
    elif datamodel == "E-ring-like":
        type_idx = 1
    disttype = pvars.DISTTYPE_ARR[type_idx] # Get information about the size distribution from the array above

    # Angle mode
    if ifvangle:
        xarr = np.arange(0, 181, 1) # Declare array of angles 0 to 180 degrees
        # Get reflectances. Programmed so that angles start, stop, and step the same way as xarr
        reflectances = angle_mie_reflectances(disttype[0], disttype[1], disttype[2], theta_min=0, theta_max=180, wavels=[param], comp=comp, nsize=pvars.NSIZE, tau=tau)[0][0]

        # For Henyey-Greenstein, curve fitting is needed to determine global scale factor
        if fitmodel == 'Henyey-Greenstein':
            init_hg_params = pvars.HG_PARAMS_ARR[type_idx] # Get information on parameters for HG function
            
            # Define a function to use in curve-fitting HG function to match Mie via scale factor
            def hg_iterator(angles, scale_factor):
                hg_ifs = henyey_greenstein(xarr, [[init_hg_params[1], init_hg_params[0] * scale_factor], [init_hg_params[3], init_hg_params[2] * scale_factor]])
                return np.log10(hg_ifs)

            # Try to best-fit the Henyey-Greenstein function to the Mie function via a scale factor
            try:
                # Fit in log space so the tail isn't ignored
                popt, _ = sp.optimize.curve_fit(
                    hg_iterator, 
                    xarr, 
                    np.log10(reflectances), # Log of original dataset is being fit with log of best-fit dataset
                    p0=[1e-6], 
                    maxfev=10000,
                )
                factor = popt.tolist()[0]
            except Exception as e:
                print(f"Curve fitting failed: {e}")
                factor = 1e-6

            # Final Henyey-Greenstein reflectances
            reflectances = henyey_greenstein(np.radians(xarr), [[init_hg_params[1], init_hg_params[0] * factor], [init_hg_params[3], init_hg_params[2] * factor]])
    # Wavelength mode
    else:
        wavel_range = pvars.INSTRUMENTS[sensor] # Plot over range of all available wavelengths in the sensor
        xarr = np.arange(wavel_range[0], wavel_range[1] + 0.01, 0.01) # Assume a 10 nm resolution
        reflectances = angle_mie_reflectances(disttype[0], disttype[1], disttype[2], theta_min=param, theta_max=param, wavels=xarr, nsize=pvars.NSIZE, tau=tau)[0]
        # angle_mie_reflectances returns a weird column array shape [[element1], [element2], [element3]]
        # Need to make array not as deep
        reflectances = [element[0] for element in reflectances]

    return xarr.tolist(), reflectances # xarr should not be a numpy array because PyQt6 doesn't know what numpy is for some reason


def RMSE(y_act, y_pred):
    """
    Computes the Root Mean Square Error between a given dataset and its best-fit curve

    :param y_act: Array of original data points
    :param y_pred: Array of predicted data points from best-fit curve
    :returns: Float for RMSE of dataset and best-fit curve
    """
    # Root mean square error
    return np.sqrt(np.mean((y_act-y_pred)**2))

def create_dirpath(dirname, dirs_removed=0):
    """Creates a directory within the graphs directory to hold a program's output graphs.

    :param dirname: String name of desired directory within graphs folder
    :returns: String path of resultant directory
    """
    # Define the path for the output directory
    graph_dir = os.path.join("graphs", dirname)
    for i in range(dirs_removed):
        graph_dir = os.path.join("..", graph_dir)
    # Create the directory to contain the plots
    try:
        os.makedirs(graph_dir, exist_ok=True)
    except OSError as error:
        print(f"Error creating directory: {error}") # Error handling

    return graph_dir

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

    # Getting logarithmic versions for appropriate relative magnitude comparison in R^2 and Z score
    log_act = np.log10(y_act)
    log_pred = np.log10(y_pred)

    # 1. R-Squared (Coefficient of Determination)
    ss_res = np.sum((log_act - log_pred)**2) # Sum of squares of residuals
    ss_tot = np.sum((log_act - np.mean(log_pred))**2) # Total sum of squares
    r_squared = 1 - (ss_res / ss_tot)
    
    # 2. MAPE (Mean Absolute Percentage Error)
    # Adding a tiny epsilon (1e-10) to y_act to prevent division by zero
    mape = np.mean(np.abs((y_act - y_pred) / (y_act + 1e-10))) * 100
    
    # Evaluate Global Fit
    print("\n--- Global Fit Analysis ---")
    if r_squared < 0.5:
        print(f"CRITICAL WARNING: Terrible global fit! Log R^2 is incredibly low ({r_squared:.2f}).")
    else:
        print(f"Fit Quality: Log R^2 = {r_squared:.4f}")
        
    if mape > 25.0:  # Adjust this threshold based on your needs
        print(f"CRITICAL WARNING: Fit deviates by an average of {mape:.1f}%. Massive inaccuracy detected.")
    else:
        print(f"Average Error: {mape:.1f}%")

    # Determine if there are spikes in error/spikes in fit
    print("--- Local Spike Analysis ---")
    raw_diffs = log_act - log_pred
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
    print(f"\nRMSE ({vars.ANGLE_LOWERBOUND}-{vars.ANGLE_UPPERBOUND} degrees): {plt_RMSE}")

    # Run error analysis to determine accuracy of fit.
    # For analysis purposes, we use linear versions of the datasets to make sure the tests run accurately (i.e. percent error)
    diff_analysis(if_orig, if_fit, theta)

    # Plotting the given datasets
    iflogplt.plot(theta, if_orig, color='black', label=f'Reflectance from data')
    iflogplt.plot(theta, if_fit, color='blue', label=f"Computed {'Mie' if use_mie else 'Henyey-Greenstein'} (RMSE={plt_RMSE:.2e})")

    # Cosmetic setup
    iflogplt.set_xlabel('Scattering Angle (degrees)')
    iflogplt.set_ylabel('Reflectance (I/F)')
    iflogplt.set_title(f'{vars.DATA_FILE[:-4]} Reflectance vs Scattering Angle')
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
    fig_path = os.path.join(folder, f"{vars.DATA_FILE[:-4]}_{'miefit' if use_mie else 'hgfit'}.png")
    plt.savefig(fig_path)
    print(f"\nGraph fit saved to {fig_path}")

def wavel_plot(folder, wavels, params, angle=vars.WAVEL_SNAPSHOT_ANGLE):
    """
    Plots and saves a graph of reflectances varied over different wavelenghts at a fixed scattering angle, 
    using the size distribution calculated with Mie theory. As only Mie theory is dependent on wavelength, 
    this graph will only be generated if Mie theory is used.

    :param folder: String folder in which to save graph
    :param wavels: Array of wavelengths at which to extract reflectance data
    :param params: Array of particle distribution parameters calculated from Mie theory.
    :param angle: Integer scattering angle at which to loop over wavelengths
    """
    reflectances = []
    for w in wavels:
        # Get reflectance at singular angle using given Mie parameters, passing existing tau
        reflectance_wavel = angle_mie_reflectances(params[0], params[1], params[2], theta_min=angle, theta_max=angle, wavels=[w], tau=params[3])[0][0][0]
        reflectances.append(reflectance_wavel) # Get first element of single-element array returned

    # Plot setup, plus a space for text to display parameters
    fig, (ifplt, txt) = plt.subplots(1, 2, figsize=(12, 8), layout='constrained')

    # Plot the computed reflectances
    ifplt.plot(wavels, reflectances, color='black', label=f'Computed Mie')

    # Cosmetic setup
    ifplt.set_xlabel('Wavelength (microns)')
    ifplt.set_ylabel('Reflectance (I/F)')
    ifplt.set_title(f'{vars.DATA_FILE[:-4]} Reflectance vs Wavelength')
    ifplt.grid()
    ifplt.legend()

    # Fill in textbox with necessary parameters
    txt.text(0.25, 0.5, f"Mie Params:\nsmin={params[0]:.4f}\nsmax={params[1]:.4f}\npowlaw={params[2]:.4f}\ntau={params[3]:.2e}", bbox=dict(facecolor='lightblue', alpha=0.5, edgecolor='black', pad=10), fontsize=15)

    # Save to file
    fig_path = os.path.join(folder, f"{vars.DATA_FILE[:-4]}_wavels_miefit.png")
    plt.savefig(fig_path)
    print(f"\nWavelength I/F plot saved to {fig_path}")

def plt_angle(fig_path, theta, mie_ifs, hg_ifs, params, hg_params):
    """
    Plots and saves a graph of reflectances over various angles.
    Displays parameters (particle size distribution for Mie, weights and asymmetry parameters for Henyey-Greenstein)
    necessary to produce the displayed graph.

    :param fig_path: String path to which to save graph, including image name
    :param theta: Array of thetas in degrees, to be plotted along the x-axis
    :param mie_ifs: Array of reflectances computed from Mie theory
    :param hg_ifs: Array of reflectances computed from Henyey-Greenstein function
    :param params: Array of Mie and miscellaneous parameters used to produce given plot
    :param hg_params: Array of Henyey-Greenstein parameters used to produce given plot
    """
    # Plot setup - using a log scale plot, plus a space for text to display parameters
    fig, (iflogplt, txt) = plt.subplots(1, 2, figsize=(12, 8), layout='constrained')

    # Plotting the given datasets
    iflogplt.plot(theta, mie_ifs, color='blue', label=f'Mie Reflectance')
    iflogplt.plot(theta, hg_ifs, color='red', label=f'Henyey-Greenstein Reflectance')

    # Cosmetic setup
    iflogplt.set_xlabel('Scattering Angle (degrees)')
    iflogplt.set_ylabel('Reflectance (I/F)')
    iflogplt.set_title(f'{params[4]} Reflectance vs Scattering Angle at {params[5]:.3f} μm')
    iflogplt.grid()
    iflogplt.legend()
    iflogplt.set_yscale('log')
    iflogplt.set_xscale('log')

    # Fill in textbox with necessary parameters
    txt.text(0.3, 0.6, f"Mie Params:\nsmin={params[0]:.4f}\nsmax={params[1]:.4f}\npowlaw={params[2]:.4f}\ntau={params[3]:.2e}", bbox=dict(facecolor='lightblue', alpha=0.5, edgecolor='black', pad=10), fontsize=15)
    txt.text(0.3, 0.3, f"Henyey-Greenstein Params:\nw1={hg_params[0]:.2f}\ng1={hg_params[1]:.2f}\nw2={hg_params[2]:.2f}\ng2={hg_params[3]:.2f}\nwscale={hg_params[4]:.2e}", bbox=dict(facecolor='pink', alpha=0.5, edgecolor='black', pad=10), fontsize=15)
    
    # Save to file
    plt.savefig(fig_path)
    print(f"Angle I/F plot saved to {fig_path}")

def plt_wavel(fig_path, wavels, angles, plt_angles, reflectances, params):
    """
    Plots and saves a graph of reflectances varied over different wavelenghts at a fixed scattering angle, 
    using the size distributions given. As only Mie theory is dependent on wavelength, 
    this graph will only be generated if Mie theory is used.

    :param fig_path: String path to which to save graph, including image name
    :param wavels: Array of wavelengths comprising row-axis of reflectances
    :param angles: Array of angles comprising column-axis of reflectances
    :param plt_angles: Array of angles (degrees) for which to produce graphs
    :param reflectances: 2D Numpy array of reflectances at different angles and wavelengths. Shape (len(wavels), len(angles))
    :param params: Array of particle distribution parameters calculated from Mie theory.
    """

    # Plot setup, plus a space for text to display parameters
    fig, (ifplt, txt) = plt.subplots(1, 2, figsize=(12, 8), layout='constrained')

    for angle in plt_angles:
        angle_idx = np.where(angles == angle)[0][0]
        ifplt.plot(wavels, np.asarray(reflectances)[:, angle_idx], label=f'theta={angle}')

    # Cosmetic setup
    ifplt.set_xlabel('Wavelength (microns)')
    ifplt.set_ylabel('Log Reflectance (I/F)')
    ifplt.set_title(f'{params[4]} Reflectance vs Wavelength')
    ifplt.grid()
    ifplt.set_yscale('log')
    ifplt.legend()

    # Fill in textbox with necessary parameters
    txt.text(0.25, 0.5, f"Mie Params:\nsmin={params[0]:.4f}\nsmax={params[1]:.4f}\npowlaw={params[2]:.4f}\ntau={params[3]:.2e}", bbox=dict(facecolor='lightblue', alpha=0.5, edgecolor='black', pad=10), fontsize=15)

    # Save to file
    plt.savefig(fig_path)
    print(f"Wavelength I/F plot saved to {fig_path}")

def angle_wavel_csv(file_path, wavels, angles, reflectances):
    """
    A function that takes input reflectance data and outputs it to a csv file, with rows being wavelengths and 
    columns being reflectances.

    :param file_path: String path to which to save csv
    :param wavels: Row-axis array for data
    :param angles: Column-axis array for data, in degrees
    :param reflectances: 2D Numpy array containing reflectance data. Shape (len(wavels), len(angles))
    """

    with open(file_path, 'w') as f:
        writer = csv.writer(f)
        header = ["Wavelength (μm) | Scattering Angle (°)"]
        header.extend(angles)
        writer.writerow(header)
        for idx, wavel_slice in enumerate(reflectances):
            row = [wavels[idx]]
            row.extend(wavel_slice)
            writer.writerow(row)
    print(f"Wavelength-angle reflectance table saved to {file_path}")

    