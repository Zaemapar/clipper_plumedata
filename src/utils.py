"""
A file containing various helper methods to organize and read files, perform major computations relating to
scattering theories, or output graphs given an input dataset.

Authors: Matthew M. Hedman, Parker A. Zaemann
Date: 14 August 2026
Source: https://github.com/Zaemapar/clipper_plumedata
Contact: mhedman@uidaho.edu
"""

import numpy as np
import scipy as sp
import scipy.optimize
import csv
import matplotlib.pyplot as plt
import PyMieScatt as PyMie
import fit_vars as vars
import os
import sys
import functools
import multiprocessing as mp

@functools.lru_cache() # Set up a cache so the same material doesn't get need to get read twice in a row
def wb08read(data_path):
    """
    Extracts information on complex index of refraction from provided datasheets for a given material.

    :param data_path: String file path for index of refraction data
    :returns: Numpy array for given wavelength,
              Numpy array for n values (refractive indices) at each wavelength,
              Numpy array for k values (absorbtion coefficients) at each wavelength
    """
    wavew, nw, kw, temp = [], [], [], []

    # Load the entire file containing index of refraction data into memory at once
    txt = np.loadtxt(data_path, delimiter=',', skiprows=2)

    if txt.ndim == 1:
        txt = np.asarray([txt]) # Numpy does this weird thing where if there's only one row, it returns a 1d array rather than a 2d array.

    for line in txt:
        # Append wavelength, real part n, and imaginary part k
        wavew.append(float(line[0])) 
        nw.append(float(line[1]))
        kw.append(float(line[2]))

    # Cast as numpy arrays for easier interpolation later
    wavew=np.array(wavew)
    nw=np.array(nw)
    kw=np.array(kw)
    
    return wavew,nw,kw

def get_nk(wavels, comps, mixmodel='Molecular'):
    """
    Determines complex index of refraction n + ki of a material given its composition and the incident wavelength.

    :param wavel: Float wavelength in microns
    :param comps: Dictionary of string compositions of material and given volume fractions
    :param mixmodel: String mixture model to use, only 'Molecular' is supported
    :returns: Float n (refractive index) at the given wavelength
              Float k (absorption coefficient) at the given wavelength
    """

    # Create empty arrays to hold all the data
    dielectrics, fs, temp_ns, temp_ks = [], [], [], []

    for comp in comps.keys():
        # Material files MUST be formatted with "_constants.csv" at the end!
        idx_refraction_path = os.path.join("../data", comp + "_constants.csv")
        try:
            # Reads the appropriate csv file
            wavex,nx,kx=wb08read(idx_refraction_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Material {comp} not supported.")
    
        # Interpolate to given wavelength
        nxfunc=sp.interpolate.interp1d(wavex,nx)
        kxfunc=sp.interpolate.interp1d(wavex,kx)
        temp_ns.append(nxfunc(wavels))
        temp_ks.append(kxfunc(wavels))

        # Get the ns and ks of the most recent materials
        last_ns = np.asarray(temp_ns[-1])
        last_ks = np.asarray(temp_ks[-1])

        dielectrics.append(((last_ns**2 - last_ks**2) + 1j*2*last_ns*last_ks).tolist()) # Compute dielectrics across all wavelengths
        fs.append(float(comps[comp])) # Append v/v fraction for material

    if len(dielectrics) == 1 or mixmodel == 'Areal':
        ns = temp_ns[0]
        ks = temp_ks[0] # Return single material n, k if only one material or areal mixing (which combines reflectances as a whole rather than refractive indices)
    else:
        if mixmodel == 'Molecular':
            # Molecular mixture requires that we consider particles of evenly mixed materials
            # This requires considering the dielectric constants of each
            first_mat = np.asarray(dielectrics[0])
            second_mat = np.asarray(dielectrics[1])
            eps_x = first_mat*(1 + 3*fs[1]*(second_mat-first_mat)/(second_mat+2*first_mat)/(1 - fs[1]*(second_mat-first_mat)/(second_mat+2*first_mat)))
            # Here we synthesize a combined index of refraction for the mixture at each wavelength
            ns=(np.sqrt(.5)*np.sqrt(np.sqrt(np.real(eps_x)**2+np.imag(eps_x)**2)+np.real(eps_x))).tolist()
            ks=(np.sqrt(.5)*np.sqrt(np.sqrt(np.real(eps_x)**2+np.imag(eps_x)**2)-np.real(eps_x))).tolist()
        else:
            # Handle the case of an improper argument
            raise ValueError(f"Mixture model {mixmodel} not supported.")
    
    # Full returns full array across all wavelengths
    return ns, ks

def get_minmax_wavel(comp):
    """
    Reads the same file as get_nk, but searches for min and max wavelength supported.

    :param comp: String composition of material
    :returns: Float minimum wavelength supported in microns
              Float maximum wavelength supported in microns
    """
    idx_refraction_path = os.path.join("../data", comp + "_constants.csv")
    
    # Calling function to read appropriate tables of optical constants for the given material
    try:
        wavex,_,_=wb08read(idx_refraction_path)
        wavex.sort() # Sort just in case
        return [comp, wavex[0]], [comp, wavex[-1]] # Return min and max
    except FileNotFoundError:
        raise FileNotFoundError("Material type not supported.")

def get_available_materials():
    """
    Searches through the 'data' subfolder and finds all properly-formatted material data files. If Water Ice is
    found, it will be put on top.

    :returns: List of string material names before the '_constants.csv', with 'Water Ice' at the top if found
    """
    files = os.listdir("../data") # List all files
    data_files = []
    for file in files:
        if file.endswith("_constants.csv"): # Ensure files end with _constants.csv
            data_files.append(file[:-14]) # Append the material name part of the filename
    if "Water Ice" in data_files: # Move Water Ice to the top if found
        data_files.remove("Water Ice")
        temp_files = ["Water Ice"]
        temp_files.extend(data_files)
        data_files = temp_files
    return data_files

def get_data(src, target_x="Scattering Angle", target_y="Reflectance", altitude=None):
    """
    Reads data from a given sample and extracts scattering angle (or wavelength) and corresponding I/F reflectance.
    Parses header using string search to determine how to format the data.

    :param src: String path to data file
    :param target_x: String x output type. "Scattering Angle" for scattering angle array, "Phase Angle" for phase 
                     angle array, "Wavelength" for wavelength array
    :param target_y: String y output type. "Reflectance" for reflectance array, "Phase Function" for phase function, 
                     "Intensity" for scattering intensity, "Albedo" for surface albedo
    :param altitude: Float altitude in kilometers if reading data that depends on it
    :returns: Sorted array for theta/wavelength values in degrees/microns,
              Correspondingly sorted array for reflectance/tau values (unitless)
              Error code (0 is successful, 1 means the desired x data couldn't be found, 2 means the desired y data
                          couldn't be found, 3 means the x data had an invalid value, 4 means the y data had an
                          invalid value, 5 means the arrays weren't the same length)
    """
    # Empty arrays for data
    xarr = []
    yarr = []

    with open(src, "r") as f:
        reader = csv.reader(f)
        next(reader) # Skip title row
        header = next(reader) # Grab header row

        # Parse through header to determine where relevant data is and how to scale it
        x_idx = None
        y_idx = None
        phase_theta = False
        if_scale_factor = 0
        if target_x == "Wavelength" and altitude is not None:
            x_idx = 0 # Wavelengths are always listed in the left column
            y_idx = np.where(np.asarray(header) == str(altitude))[0] # Find the column corresponding to the input altitude
            if_scale_factor = -2 # Equivalent width is usually listed in those tables, to convert to tau divide by 100
        else:
            for idx, headstring in enumerate(header):
                lc_headstring = headstring.lower()
                # Check to see if it's an angle column
                if (target_x == "Phase Angle" or target_x == "Scattering Angle") and ("theta" in lc_headstring or "angle" in lc_headstring) and "range" not in lc_headstring:
                    x_idx = idx
                    # Phase vs scattering angle needs to be tracked so operations can be performed in the end
                    # Ultimately if both are provided the code will use whichever one comes last
                    if "phase" in lc_headstring:
                        phase_theta = True
                    else:
                        phase_theta = False
                # Check to see if it's a wavelength column
                elif target_x == "Wavelength" and "wave" in lc_headstring and "range" not in lc_headstring:
                    x_idx = idx
                # Check to see if it's a reflectance/phase function/intensity column
                elif (target_y == "Reflectance" and "reflect" in lc_headstring or "contr" in lc_headstring) or (target_y == "Phase Function" and "phase" in lc_headstring and "angle" not in lc_headstring) or (target_y == "Intensity" and "intens" in lc_headstring) or (target_y == "Albedo" and "alb" in lc_headstring):
                    y_idx = idx
                    # Find where it is multiplied by a scale factor, if any. Reflectance often is.
                    modifier_idx = lc_headstring.find("10^")
                    if modifier_idx > 0:
                        try:
                            if_scale_factor = int(lc_headstring[(modifier_idx + 3):])
                        except ValueError:
                            pass # That was not the right reflectance column
                
        # If data failed to find, return empty arrays and error code
        if x_idx is None:
            return [], [], 1
        elif y_idx is None:
            return [], [], 2
        # Iterate through each line in the rest of the document
        for line in reader:
            try:
                # Append scattering angle (convert to scattering if phase requested, convert to phase if scattering requested
                try:
                    x_append = float(line[x_idx])
                except ValueError:
                    return [], [], 3
                if (phase_theta and target_x == "Scattering Angle") or (not phase_theta and target_x == "Phase Angle"):
                    x_append = 180 - x_append # Convert to phase/scattering
                xarr.append(x_append)
                # Append reflectance & scale by scale factor
                try:
                    yarr.append(float(line[y_idx]) * 10**if_scale_factor)
                except ValueError:
                    return [], [], 4
            except IndexError:
                return [], [], 5

    # On a success, zip xarr and yarr together and sort them
    sorted_pairs = sorted(zip(xarr, yarr))
    xarr, yarr = zip(*sorted_pairs)
    return xarr, yarr, 0 # Success

def log_dist(smin, smax, base):
    """
    Creates a logarithmically-spaced array of values between smin and smax, with an exponential base

    :param smin: Float minimum particle size in microns, the start size of the array
    :param smax: Float maximum particle size in microns, the ending max size of the array
    :param base: Float base to use in exponential spacing
    :returns: Float array of logarithmically spaced values between smin and smax in microns
    """

    # First element should be smin
    sizes = [smin]
    i = 0
    while(sizes[-1] < smax):
        sizes.append(sizes[0] + base**i) # Function will look like smin + base**x
        i += 1

    # Just in case the last element overshot too much
    if sizes[-1] > smax:
        sizes[-1] = smax # We clamp the end to match smax
    elif sizes[-1] < smax:
        sizes.append(smax) # Adding smax if the array falls short

    return np.asarray(sizes)

def angle_mie_reflectances(smin, smax, powlaw, r=1, G=1, x0=np.inf, theta_min=vars.ANGLE_LOWERBOUND, theta_max=vars.ANGLE_UPPERBOUND, wavels=[vars.WAVEL], nsize=vars.NSIZE, comps={vars.COMP: 1}, mixmodel='Molecular', tau=None, ref_if=0, section='Semi-Empirical Mie', output='Reflectance', updates=[False, None]):
    """
    Computes a 2D array of reflectances given arrays of angles and wavelengths. By default, a semi-empirical Mie
    theory is used based on methods from Pollack & Cuzzi's 'Scattering of Nonspherical Particles of Size Comparable 
    to a Wavelength: A New Semi-Empirical Theory and Its Application to Troposhperic Aerosols' (1979), which uses
    pure Mie theory for small particles and a combination of diffraction, external reflection, and transmission
    theories for large particles. Some modifications to the theory have been made by Prof. Matthew Hedman to
    preserve energy conservation.

    :param smin: Float minimum size in microns used in size distribution
    :param smax: Float maximum size in microns used in size distribution
    :param powlaw: Float power law used in size distribution
    :param G: Float constant to use in Mie transmission equation
    :param r: Float ratio of the surface area of an irregular particle to the surface area of a sphere of the same 
              volume (unitless)
    :param x0: Float size cutoff above which to use semi-empirical methods and below which to use pure Mie 
               (unitless)
    :param theta_min: Integer lower angle bound in degrees
    :param theta_max: Integer upper angle bound in degrees
    :param wavel: Array of wavelengths in microns at which to evaluate Mie
    :param nsize: List [size mode, integer number of sizes to use in size distribution]
    :param comps: Dictionary with keys as mixture materials and values as volume fractions of materials
    :param mixmodel: String mixture model to use, either 'Areal' or 'Molecular'  
    :param tau: Float allowing for optical detph tau (unitless) to be input, rather than calculated
    :param ref_if: Float reflectance from original data (unitless) from which to calculate optical depth tau. 
                   ref_if MUST correspond to theta_min.
    :param section: String indicating which term of semi-empirical Mie methods to use
    :param output: String indicating what to return ('Reflectance' for reflectances or 'Phase Function' for phase 
                   function)
    :param updates: List containing boolean of whether or not to print progress update fraction after each 
                    wavelength iteration as well as multiprocessing queue to pass it to
    :returns: 2D Array of reflectance values (unitless) between theta_min, theta_max, and the min and max of 
                 wavels. Shape (len(wavels), len(angles))
              Array for calculated optical depths (unitless) at each wavelength
    """
    # Compute the nominal (unnormalized) particle size distribution
    if nsize[0] == 'Linear':
        radii=np.linspace(smin, smax, nsize[1])
    elif nsize[0] == 'Logarithmic':
        radii=log_dist(smin, smax, nsize[1])

    angle_range = np.radians(np.arange(theta_min, theta_max + 1, 1)) # Range of given angles in radians
    solid_angles_degs = np.arange(0, 181, 1) # All colatitude angles in the range of solid angles, for integrals
    solid_angles = np.radians(solid_angles_degs) # All colatitude angles in radians
    solid_min_idx = np.where(solid_angles_degs == theta_min)[0][0] # Locate given angle bounds within solid angles
    solid_max_idx = np.where(solid_angles_degs == theta_max)[0][0]

    # Figure out whether to iterate over all materials and sum weighted reflectances (areal model)
    # or all at once and get the averaged index of refraction (molecular model)
    comp_names = comps.keys() if mixmodel == 'Areal' else [0]

    # Empty array to hold reflectances/phase function and taus
    output_arr = np.zeros((len(wavels), len(angle_range)))
    taus = np.zeros((len(wavels)))

    # Diffraction term actually doesn't depend on material or index of refraction
    # We create a placeholder here for phase
    P_large_diff_2d = np.zeros((len(wavels), len(solid_angles)))

    # Get the number of compositions/iterations (for progress updates)
    n_comps = len(comp_names)
    n_wavels = len(wavels)
    n_iterations = n_comps * n_wavels

    # Iterate through each material
    for i, mat_idx in enumerate(comp_names):
        # Compute index of refraction across all wavelengths
        ns, ks = get_nk(wavels, ({mat_idx: comps[mat_idx]} if mixmodel == 'Areal' else comps), mixmodel=mixmodel)
        # Iterate through all wavelengths
        for j, wavel in enumerate(wavels):
            # Get the smallest and largest available size parameters
            min_size = 2 * np.pi * smin / wavel
            max_size = 2 * np.pi * smax / wavel

            # Ideally, we will divide our small and large sizes such that small sizes are 1/4 the distribution, large sizes are 3/4
            # But if x0 is placed such that there are no small or large sizes, this will change
            n_small_size = round(nsize[1] / 4) # Linspace only takes integers
            n_large_size = nsize[1] - n_small_size # So we keep total sizes equal to nsize[1] to stay true to user specification

            # Create placeholder variables
            small_radii = np.asarray([])
            large_radii = np.asarray([])
            small_sizes = np.asarray([])
            large_sizes = np.asarray([])

            # Now we figure out whether or not there are small parameters/large parameters based on the position of x0
            if x0 < min_size:
                n_small_size = 0
                if x0 < max_size:
                    n_large_size = nsize[1] # If there are no small sizes, we will let large sizes take up all of them
            if x0 >= max_size:
                n_large_size = 0
                if x0 >= min_size:
                    n_small_size = nsize[1] # If there are no large sizes, we will let small sizes take up all of them

            if nsize[0] == 'Linear':
                # Linearly spaced arrays for both small and large sizes, assuming there's more than 0
                if n_small_size > 0:
                    small_radii = np.linspace(smin, min(x0, max_size) * wavel / (2*np.pi), n_small_size)
                if n_large_size > 0:
                    large_radii = np.linspace(max(x0 + 0.01, min_size) * wavel / (2*np.pi), smax, n_large_size)
            elif nsize[0] == 'Logarithmic':
                # Logarithmically spaced distributions will have linear spacing for small sizes and log spacing for large sizes
                # Whatever base is used for the large sizes will be the number of sizes in the small size array.
                if n_small_size > 0:
                    small_radii = np.linspace(smin, min(x0, max_size) * wavel / (2*np.pi), nsize[1])
                if n_large_size > 0:
                    large_radii = log_dist(max(x0 + 0.01, min_size) * wavel / (2*np.pi), smax, nsize[1])

            # Regardless of what was computed before, we calculate diameters and sizes normally and size distributions using a power law
            small_sizes = 2 * np.pi * small_radii / wavel
            large_sizes = 2 * np.pi * large_radii / wavel
            small_diameters = 2*small_radii
            large_diameters = 2*large_radii
            small_dist = small_radii**(-1*powlaw)
            large_dist = large_radii**(-1*powlaw)
            # Then we concatenate to get the full distributions, important for determining F and other things
            diameters = np.concatenate(((small_diameters if n_small_size > 0 else []), (large_diameters if n_large_size > 0 else [])))
            sizedist = np.concatenate(((small_dist if n_small_size > 0 else []), (large_dist if n_large_size > 0 else [])))
            sizeparams = np.concatenate(((small_sizes if n_small_size > 0 else []), (large_sizes if n_large_size > 0 else [])))

            # Grab specific index of refraction for this wavelength
            n, k = ns[j], ks[j]
            m = complex(n, k)

            # Calculating a ratio of the size of the large particle regime vs the size of the small particle regime
            # See Pollack & Cuzzi, 1980, eq. 7d
            F = np.trapz(large_dist * np.pi * large_sizes**2, large_sizes) / np.trapz(sizedist * np.pi * sizeparams**2, sizeparams)

            # First get Qs across all sizes for constant calculations later
            Q_full = [PyMie.MieQ(m, wavel*1000, diameter, asDict=True) for diameter in diameters*1000]
            # Then split them up into small and large sizes
            Q_small = Q_full[:n_small_size]
            Q_large = Q_full[n_small_size:]

            # --- SMALL REGIME PARTICLES ---
            # Default parameters in case there are no small particles
            SU_small = np.zeros_like(solid_angles)
            P_small = np.zeros_like(angle_range)
            solid_int_Ps = 0
            Ps0 = 0
            Q_SS = 0
            Q_SA = 0

            # The Mie term will only be calculated if there are small particles present.
            if n_small_size > 0:
                # There is a specific case for which this needs to be optimized, otherwise the code will take eons to run
                # If only the mie theory is needed (no large sizes) and the output is requested in reflectances, there is a shortcut we can take.
                if n_large_size == 0 and (output == 'Reflectance' or output == 'Intensity'):
                    # Use PyMieScatt to compute scattering angles and intensities in the small regime
                    # Takes wavelength & diameters in nanometers, so need to convert from microns
                    # Note how this shortcut allows us to calculate the scattering intensity only within the angle range
                    # This is because reflectance doesn't have to be normalized by integrating over all solid angles
                    theta1, SL, SR, SU_small = PyMie.SF_SD(m, wavel*1000, small_diameters*1000, small_dist,
                                            minAngle=theta_min, maxAngle=theta_max,
                                            angularResolution=1.0, space='theta')
                        
                    # Calculate Q coefficients given the data, also applying conversions
                    qdict = PyMie.Mie_SD(m, wavel*1000., small_diameters*1000, small_dist, asDict=True)

                    # Extract extinction coefficient. PyMieScatt applies 10^-6 scale factor assuming
                    # we input size distribution in inverse cubic centimeters, and so outputs inverse
                    # megameters. But we input inverse cubic microns, and we want our output in
                    # inverse meters. This will work perfectly, but we do need to undo the 10^-6 
                    # scale factor.
                    bext = qdict['Bext'] * 1.e6

                    # Calculate tau for this wavelength from first available reflectance if tau is not provided
                    if tau is None:
                        tau_wavel = ref_if * bext * 4 * np.pi / (SU_small[0] * (wavel*1000)**2)
                    else:
                        # Otherwise, we use a fixed tau for all wavelengths
                        tau_wavel = tau

                    if output == 'Intensity':
                        # Append to relevant arrays
                        output_arr[j] += (float(comps[mat_idx]) if mixmodel == 'Areal' else 1) * SU_small
                        taus[j] += (float(comps[mat_idx]) if mixmodel == 'Areal' else 1) * tau_wavel

                    else:
                        # Calculate reflectance, converting wavelength to nanometers. SU_small is in the
                        # same units as sizedists, meaning we have square nanometers over cubic microns,
                        # which is the same as inverse meters. Our units thus cancel assuming tau and 
                        # reflectances are unitless.
                        # If this is being curve-fit to data and tau is input as None, this should make
                        # the output reflectance and input reflectance match at theta=angle_range[0]
                        refl=np.asarray(SU_small)*(wavel*1000)**2/(4*np.pi)*tau_wavel/bext

                        # Append to relevant arrays
                        output_arr[j] += (float(comps[mat_idx]) if mixmodel == 'Areal' else 1) * refl
                        taus[j] += (float(comps[mat_idx]) if mixmodel == 'Areal' else 1) * tau_wavel
                        
                    if updates[0]:
                        updates[1].put((i*n_wavels + j + 1)/n_iterations) # Put the total progress so far in the multiprocessing queue
                    # Skip everything else
                    continue

                # Otherwise, use normal routine
                # Compute SU only for the target angle range, and only at theta=0 for Ps0
                _, _, _, SU_small = PyMie.SF_SD(m, wavel*1000, small_diameters*1000, small_dist,
                                            minAngle=theta_min, maxAngle=theta_max,
                                            angularResolution=1.0, space='theta')
                _, _, _, SU_small_0 = PyMie.SF_SD(m, wavel*1000, small_diameters*1000, small_dist,
                                            minAngle=0, maxAngle=0,
                                            angularResolution=1.0, space='theta')

                # Compute normalization constant C_S using precomputed Qsca
                Q_sca_small = np.array([q['Qsca'] for q in Q_small])
                integral_SU_small = np.sum(Q_sca_small * np.pi * (small_diameters/2)**2 * small_dist) * (np.pi / wavel**2)
                C_S = 1 / integral_SU_small if integral_SU_small != 0 else 1.0

                Ps0 = C_S * SU_small_0[0]
                solid_int_Ps = 4 * np.pi
                P_small = C_S * np.asarray(SU_small)

                # Calculations of scattering and absorption efficiencies for small particles
                Q_s_denom = np.trapz(small_dist * np.pi * small_sizes**2, small_sizes) # Calculate denominator for total efficiency first so we don't have to recompute

                if Q_s_denom != 0: # We only update small scattering/absorption efficiencies if denominator is nonzero. Otherwise, Q_SS and Q_SA are left at 0 (a zero denominator means the integral didn't have enough points)
                    # Small scattering efficiency as a function of size parameter
                    Q_ssx = [q['Qsca'] for q in Q_small]
                    # Total small particle Mie scattering efficiency, P&C eq. 6b (They said to reuse it for small particle scattering)
                    Q_SS = np.trapz(Q_ssx * small_dist * np.pi * small_sizes**2, small_sizes) / Q_s_denom

                    # Small absorption efficiency as a function of size parameter
                    Q_sax = [q['Qabs'] for q in Q_small]
                    # Total small particle Mie absorption efficiency, P&C eq. 6b (They said to reuse it for small particle absorption)
                    Q_SA = np.trapz(Q_sax * small_dist * np.pi * small_sizes**2, small_sizes) / Q_s_denom

            # Default parameters in case there are no large particles
            Q_LS = 0
            Q_LA = 0
            if n_large_size > 0:
                # Calculations of scattering and absorption efficiencies for large particles
                Q_l_denom = np.trapz(large_dist * np.pi * large_sizes**2, large_sizes) # Calculate denominator for total efficiency first so we don't have to recompute
                
                if Q_l_denom != 0: # We only update Q_LS and Q_LA if denominator is nonzero. Otherwise, leave them at 0
                    # Large cattering efficiency as a function of size parameter
                    Q_lsx = [q['Qsca'] for q in Q_large]

                    # Total large particle Mie scattering efficiency, P&C eq. 6b
                    Q_LS = np.trapz(Q_lsx * large_dist * np.pi * large_sizes**2, large_sizes) / Q_l_denom
                    # Large absorption efficiency as a function of size parameter
                    Q_lax = [q['Qabs'] for q in Q_large]
                    # Total large particle Mie absorptionefficiency, P&C eq. 6b (They said to reuse it for absorption)
                    Q_LA = np.trapz(Q_lax * large_dist * np.pi * large_sizes**2, large_sizes) / Q_l_denom

                # --- LARGE REGIME: DIFFRACTION ---
                # We only compute the diffraction regime for one material
                if 'Semi-Empirical' in section or 'Diffraction' in section:
                    if i == 0:
                        # Since irregular particles have larger cross-sectional radii than their equivalent volume spheres, must multiply by sqrt(r)
                        rescaled_sizes = large_sizes * np.sqrt(r)
                        # We also need to rescale size distributions for later
                        rescaled_sizedist=(large_diameters/2)**(-1*powlaw)

                        # Cast theta and sizes as row/column arrays
                        # Theta needs to be split up at pi/4 for later
                        theta_2d_small = solid_angles[solid_angles < np.pi / 4, np.newaxis]
                        theta_2d_large = solid_angles[solid_angles >= np.pi / 4, np.newaxis]
                        theta_2d = np.concatenate((theta_2d_small, theta_2d_large))
                        sizes_2d = rescaled_sizes[np.newaxis, :]

                        # Create the z variable for the Bessel function (z = x*sin(theta), Pollack & Cuzzi eqn. 2b)
                        # Note this assumption is only valid when sin(theta) ~ theta, so we add only the part where that assumption is true (theta < pi/4)
                        z_small = sizes_2d * np.sin(theta_2d_small)
                        # J.R. Hodkinson and I. Greenleaves (1962) state that at large angles of scatter where theta !~ sin(theta), z=x*theta
                        z_large = sizes_2d * theta_2d_large
                        # Stack so that z_2d is a 2d array of shape (len(solid_angles), len(rescaled_sizes))
                        z_2d = np.vstack((z_small, z_large))

                        # Pollack & Cuzzi use a first-order Bessel function in calculating the diffraction
                        # However there is an indeterminate at theta=0, so the L'Hospital's rule limit must be applied
                        with np.errstate(divide='ignore', invalid='ignore'):
                            j1_term = sp.special.j1(z_2d) / z_2d # This makes another 2d array of same shape as z_2d
                            j1_term[z_2d == 0] = 0.5  # L'Hopital's rule limit as stated by Hodkinson and Greenleaves

                        # Calculating unnormalized phase function for diffraction as reported in Hodkinson and Greenleaves, 1963
                        # Again, creates a 2d array of same shape as z_2d
                        d_x_unnormalized = (sizes_2d**2) * (2 * j1_term)**2 * 0.5 * (1 + np.cos(theta_2d)**2)

                        # Get exctinction coefficient and optical depth for large sizes
                        Q_lex = np.asarray([q['Qext'] for q in Q_large])
                        tau_large = np.trapz(np.pi * rescaled_sizes**2 * rescaled_sizedist, rescaled_sizes)

                        # Getting the integrand over which to integrate for all rescaled sizes
                        # Again, creates a 2d array of same shape as z_2d. Column axis (len(sizes)) gets multiplied by sizes_2d^2 again and then by a row vector for sizedists
                        # Because sizes_2d and sizedists are row vectors with the same length as sizes_2d, they get copied for each row across d_x_unnormalized
                        # Note that this deviates from the methods of Pollack & Cuzzi. This is an experimental modification made by M. Hedman.
                        integrand = Q_lex[np.newaxis, :]**2/16/tau_large * d_x_unnormalized * np.pi * sizes_2d**2 * rescaled_sizedist[np.newaxis, :]
                        # Integration takes place
                        # We integrate over axis 1, the size axis. Result is a list with shape len(angles)
                        # The array is already normalized
                        P_large_diff_2d[j] = np.trapz(integrand, rescaled_sizes, axis=1)

                        Pd0 = P_large_diff_2d[j][0] # Get the first element, for normalizing P_T
                        solid_int_Pd = np.trapz(2*np.pi*P_large_diff_2d[j]*np.sin(solid_angles), solid_angles) # Get the integral over all solid angles
                    if 'Diffraction' in section:
                        # Normalize so that integral over all solid angles equals 4*pi
                        C_D = 4 * np.pi / solid_int_Pd
                        # Get the diffraction phase function between valid angles
                        P_large_diff = C_D * P_large_diff_2d[j][solid_min_idx:solid_max_idx + 1]
                    else:
                        # Semi-Empirical Mie uses its own normalization elsewhere
                        # Here we just get the phase function between valid angles
                        P_large_diff = P_large_diff_2d[j][solid_min_idx:solid_max_idx + 1]

                # --- LARGE REGIME: EXTERNAL REFLECTION ---
                if 'Semi-Empirical' in section or 'External Reflection' in section:
                    # Placeholder array to hold external reflection component
                    P_large_extref_unsliced = np.zeros_like(solid_angles)
                    m_bar_sqr = np.abs(m)**2 # Taking the magnitude of the index of refraction squared
                    # Get the unnormalized phase function for external reflection, which is a function of the solid angle theta
                    nan_idxs = (m_bar_sqr - 1 + np.sin(solid_angles / 2)**2) < 0
                    # No normalization is needed because it cancels out in the f
                    P_large_extref_unsliced[nan_idxs] = 1 # Clamp the m_bar_sqr - 1 + np.sin(solid_angles / 2)**2 to 0 to avoid negative square root. This yields an intensity of 1 (see Ellison & Peetz, 1959, section 3)
                    P_large_extref_unsliced[~nan_idxs] = 0.5*((np.sin(solid_angles[~nan_idxs] / 2) - (m_bar_sqr - 1 + np.sin(solid_angles[~nan_idxs] / 2)**2)**0.5)/(np.sin(solid_angles[~nan_idxs] / 2) + (m_bar_sqr - 1 + np.sin(solid_angles[~nan_idxs] / 2)**2)**0.5))**2 + 0.5 * ((m_bar_sqr * np.sin(solid_angles[~nan_idxs] / 2) - (m_bar_sqr - 1 + np.sin(solid_angles[~nan_idxs] / 2)**2)**0.5)/(m_bar_sqr * np.sin(solid_angles[~nan_idxs] / 2) + (m_bar_sqr - 1 + np.sin(solid_angles[~nan_idxs] / 2)**2)**0.5))**2

                    Pr0 = P_large_extref_unsliced[0] # Get the first element, for normalizing P_T later
                    solid_int_Pr = np.trapz(2*np.pi*P_large_extref_unsliced*np.sin(solid_angles), solid_angles) # Get the integral over all solid angles
                    if 'External Reflection' in section:
                        # Normalize so that integral over all solid angles equals 4*pi
                        C_R = 4 * np.pi / solid_int_Pr
                        # Get the external reflection phase function between valid angles
                        P_large_extref = C_R * P_large_extref_unsliced[solid_min_idx:solid_max_idx + 1]
                    else:
                        # Semi-Empirical Mie uses its own normalization elsewhere
                        # Here we just get the phase function between valid angles
                        P_large_extref = P_large_extref_unsliced[solid_min_idx:solid_max_idx + 1]

                        # If intensity is the desired output, we can skip the rest of this iteration
                        if output == 'Intensity':
                            output_arr[j] += (float(comps[mat_idx]) if mixmodel == 'Areal' else 1) * P_large_extref
                            # Pass the progress to the queue if requested
                            if updates[0]:
                                updates[1].put((i*n_wavels + j + 1)/n_iterations) # Put the total progress so far in the multiprocessing queue
                            continue

                # --- LARGE REGIME: TRANSMISSION ---
                if 'Semi-Empirical' in section or 'Transmission' in section:
                    b = -2 * np.log(G) / np.pi # np.log is the natural log
                    P_large_trans_unsliced = np.e**(1 + b * solid_angles)

                    if 'Transmission' in section:
                        # If intensity is the desired output, we can skip the rest of this iteration
                        if output == 'Intensity':
                            output_arr[j] += (float(comps[mat_idx]) if mixmodel == 'Areal' else 1) * P_large_trans_unsliced[solid_min_idx:solid_max_idx + 1]
                            # Pass the progress to the queue if requested
                            if updates[0]:
                                updates[1].put((i*n_wavels + j + 1)/n_iterations) # Put the total progress so far in the multiprocessing queue
                            continue

                        # Otherwise, normalize so that integral over all solid angles equals 4*pi
                        C_T = 1 / np.trapz(0.5*P_large_trans_unsliced*np.sin(solid_angles), solid_angles)
                        # Get the transmission phase function between valid angles
                        P_large_trans = C_T * P_large_trans_unsliced[solid_min_idx:solid_max_idx + 1]
                    else:
                        # In semi-empirical Mie mode, we will normalize all three modes such that the sum of all their integrals over all solid angles equals Q_LS. This requires us to determine a normalization constant for P_large_trans.
                        C_T = (Q_LS - np.trapz(0.5*P_large_diff_2d[j]*np.sin(solid_angles), solid_angles) - np.trapz(0.5*P_large_extref_unsliced*np.sin(solid_angles), solid_angles)) / np.trapz(0.5*P_large_trans_unsliced*np.sin(solid_angles), solid_angles)
                        P_large_trans_normalized = C_T * P_large_trans_unsliced # Scaling by its normalization constant
                        # Get the first value of this array, as well as the integral over all solid angles, for further normalization later
                        Pt0 = P_large_trans_normalized[0]
                        solid_int_Pt = np.trapz(2*np.pi*P_large_trans_normalized*np.sin(solid_angles), solid_angles)
                        P_large_trans = P_large_trans_normalized[solid_min_idx:solid_max_idx + 1] # Slice with appropriate angle range

            # Note: Now we have to compute reflectances as if the whole thing were pure Mie. In order to normalize the transmission component, the pure Mie normalized peak and the semi-empirical normalized peak must be set equal
            if 'Semi-Empirical' in section:
                # Calculate scattering intensity over ALL sizes ONLY at theta=0
                _, _, _, SU_small_full_0 = PyMie.SF_SD(m, wavel*1000, diameters*1000, sizedist,
                                            minAngle=0, maxAngle=0,
                                            angularResolution=1.0, space='theta')
                
                # Compute normalization analytically using the already computed Q_full array
                Q_sca_all = np.array([q['Qsca'] for q in Q_full])
                integral_SU_full = np.sum(Q_sca_all * np.pi * (diameters/2)**2 * sizedist) * (np.pi / wavel**2)
                C_S_full = 1 / integral_SU_full if integral_SU_full != 0 else 1.0
                
                # To get the phase function peak, normalize the scattering intensity at theta=0
                Ps0_full = C_S_full * SU_small_full_0[0]
                solid_int_Ps_full = 4 * np.pi
                
            # Total efficiencies between both regimes, weighted by ratio of large to small particles
            Q_star_S = Q_LS * F * r + Q_SS * (1 - F) # Total scattering efficiency, P&C eq. 7a
            Q_star_A = (Q_LA * F) * (r if 2*n*np.mean(large_sizes) > 1 else 1) + Q_SA * (1 - F) # Total absorption efficiency, P&C eq. 7b. They mention a case where the first term needs to be multiplied by r if 2nx>1. I take that to mean average large x.
            Q_star_E = Q_star_S + Q_star_A # Total extinction efficiency, P&C eq. 7c

            # Weights for small and large terms, according to P&C
            P_small_fact = (1 - F) * Q_SS / Q_star_S
            P_large_fact = r * F * Q_LS / Q_star_S

            # Compute the combined phase function, P&C eq. 8.
            P_star = np.zeros_like(angle_range)
            # Add on small size components as appropriate
            if n_small_size > 0:
                P_star += P_small * P_small_fact
            # Add on large size components as appropriate
            if n_large_size > 0:
                if 'Semi-Empirical' in section or 'Diffraction' in section:
                    P_star += P_large_diff * P_large_fact
                if 'Semi-Empirical' in section or 'External Reflection' in section:
                    P_star += P_large_extref * P_large_fact
                if 'Semi-Empirical' in section or 'Transmission' in section:
                    # Now here is where normalization of transmission gets tricky. In the semi-empirical method, we want the normalized peak of the phase function to equal that of pure Mie
                    if 'Semi-Empirical' in section:
                        # We have to weight the phase function values with their weights in the overall phase function (because small and large terms are mixed, they don't cancel out)
                        Ps0, solid_int_Ps = Ps0*P_small_fact, solid_int_Ps*P_small_fact
                        Pd0, Pr0, Pt0, solid_int_Pd, solid_int_Pr, solid_int_Pt = P_large_fact*Pd0, P_large_fact*Pr0, P_large_fact*Pt0, solid_int_Pd*P_large_fact, solid_int_Pr*P_large_fact, solid_int_Pt*P_large_fact
                        
                        # For convenience, we combine the small, diffraction, and external reflection components
                        Psdr0 = Ps0 + Pd0 + Pr0
                        solid_int_Psdr = solid_int_Ps + solid_int_Pd + solid_int_Pr

                        # Then we calculate the k value
                        kmie = (Psdr0 - solid_int_Psdr*Ps0_full/solid_int_Ps_full)/(solid_int_Pt*Ps0_full/solid_int_Ps_full - Pt0)

                    else:
                        kmie = 1 # Set to 1 as default multiplicative identity
                    
                    # Add on transmission term
                    P_star += kmie * P_large_fact * P_large_trans

            # Now check the output and calculate phase or reflectance if requested
            if output == 'Phase Function':
                output_arr[j] += (float(comps[mat_idx]) if mixmodel == 'Areal' else 1) * P_star
            elif output == 'Reflectance':
                omega_0 = Q_star_S / Q_star_E

                # Calculate tau from first available reflectance if not provided
                if tau is None:
                    # For a single scattering cloud, Reflectance = 0.25 * (Q_star_S / Q_star_E) * P_star * tau
                    tau_wavel = ref_if / (0.25 * omega_0 * P_star[0])
                else:
                    tau_wavel = tau

                # Calculate reflectance using scattering data
                refl = 0.25 * omega_0 * P_star * tau_wavel
                output_arr[j] += (float(comps[mat_idx]) if mixmodel == 'Areal' else 1) * refl
                taus[j] += (float(comps[mat_idx]) if mixmodel == 'Areal' else 1) * tau_wavel

            if updates[0]:
                updates[1].put((i*n_wavels + j + 1)/n_iterations) # Put the total progress so far in the multiprocessing queue

    # Return a reflectances/phase functions as well as tau
    return output_arr.tolist(), taus.tolist()

def angle_mie_reflectances_old(smin, smax, powlaw, r=1, G=1, x0=np.inf, theta_min=vars.ANGLE_LOWERBOUND, theta_max=vars.ANGLE_UPPERBOUND, wavels=[vars.WAVEL], nsize=vars.NSIZE, comps={vars.COMP: 1}, mixmodel='Molecular', tau=None, ref_if=0, section='Semi-Empirical Mie', output='Reflectance', updates=[False, None]):
    """
    Computes a 2D array of reflectances given arrays of angles and wavelengths. By default, a semi-empirical Mie
    theory is used based on methods from Pollack & Cuzzi's 'Scattering of Nonspherical Particles of Size Comparable 
    to a Wavelength: A New Semi-Empirical Theory and Its Application to Troposhperic Aerosols' (1979), which uses
    pure Mie theory for small particles and a combination of diffraction, external reflection, and transmission
    theories for large particles.

    :param smin: Float minimum size in microns used in size distribution
    :param smax: Float maximum size in microns used in size distribution
    :param powlaw: Float power law used in size distribution
    :param G: Float constant to use in Mie transmission equation
    :param r: Float ratio of the surface area of an irregular particle to the surface area of a sphere of the same 
              volume (unitless)
    :param x0: Float size cutoff above which to use semi-empirical methods and below which to use pure Mie 
               (unitless)
    :param theta_min: Integer lower angle bound in degrees
    :param theta_max: Integer upper angle bound in degrees
    :param wavel: Array of wavelengths in microns at which to evaluate Mie
    :param nsize: List [size mode, integer number of sizes to use in size distribution]
    :param comps: Dictionary with keys as mixture materials and values as volume fractions of materials
    :param mixmodel: String mixture model to use, either 'Areal' or 'Molecular'  
    :param tau: Float allowing for optical detph tau (unitless) to be input, rather than calculated
    :param ref_if: Float reflectance from original data (unitless) from which to calculate optical depth tau. 
                   ref_if MUST correspond to theta_min.
    :param section: String indicating which term of semi-empirical Mie methods to use
    :param output: String indicating what to return ('Reflectance' for reflectances or 'Phase Function' for phase 
                   function)
    :param updates: List containing boolean of whether or not to print progress update fraction after each 
                    wavelength iteration as well as multiprocessing queue to pass it to
    :returns: 2D Array of reflectance values (unitless) between theta_min, theta_max, and the min and max of 
                 wavels. Shape (len(wavels), len(angles))
              Array for calculated optical depths (unitless) at each wavelength
    """

    angle_range = np.radians(np.arange(theta_min, theta_max + 1, 1)) # Range of given angles in radians
    solid_angles_degs = np.arange(0, 181, 1) # All colatitude angles in the range of solid angles, for integrals
    solid_angles = np.radians(solid_angles_degs) # All colatitude angles in radians
    solid_min_idx = np.where(solid_angles_degs == theta_min)[0][0] # Locate given angle bounds within solid angles
    solid_max_idx = np.where(solid_angles_degs == theta_max)[0][0]

    # Figure out whether to iterate over all materials and sum weighted reflectances (areal model)
    # or all at once and get the averaged index of refraction (molecular model)
    comp_names = comps.keys() if mixmodel == 'Areal' else [0]

    # Empty array to hold reflectances/phase function and taus
    output_arr = np.zeros((len(wavels), len(angle_range)))
    taus = np.zeros((len(wavels)))

    # Diffraction term actually doesn't depend on material or index of refraction
    # We create placeholders here for both phase and intensity
    SU_large_diff_2d = np.zeros((len(wavels), len(angle_range)))
    P_large_diff_2d = np.zeros((len(wavels), len(angle_range)))

    # Get the number of compositions/iterations (for progress updates)
    n_comps = len(comp_names)
    n_wavels = len(wavels)
    n_iterations = n_comps * n_wavels

    # Iterate through each material
    for i, mat_idx in enumerate(comp_names):
        # Compute index of refraction across all wavelengths
        ns, ks = get_nk(wavels, ({mat_idx: comps[mat_idx]} if mixmodel == 'Areal' else comps), mixmodel=mixmodel)
        # Iterate through all wavelengths
        for j, wavel in enumerate(wavels):
            # Get the smallest and largest available size parameters
            min_size = 2 * np.pi * smin / wavel
            max_size = 2 * np.pi * smax / wavel

            # Ideally, we will divide our small and large sizes such that small sizes are 1/4 the distribution, large sizes are 3/4
            # But if x0 is placed such that there are no small or large sizes, this will change
            n_small_size = round(nsize[1] / 4) # Linspace only takes integers
            n_large_size = nsize[1] - n_small_size # So we keep total sizes equal to nsize[1] to stay true to user specification

            # Create placeholder variables
            small_radii = np.asarray([])
            large_radii = np.asarray([])
            small_sizes = np.asarray([])
            large_sizes = np.asarray([])

            # Now we figure out whether or not there are small parameters/large parameters based on the position of x0
            if x0 < min_size:
                n_small_size = 0
                if x0 < max_size:
                    n_large_size = nsize[1] # If there are no small sizes, we will let large sizes take up all of them
            if x0 >= max_size:
                n_large_size = 0
                if x0 >= min_size:
                    n_small_size = nsize[1] # If there are no large sizes, we will let small sizes take up all of them

            if nsize[0] == 'Linear':
                # Linearly spaced arrays for both small and large sizes, assuming there's more than 0
                if n_small_size > 0:
                    small_radii = np.linspace(smin, min(x0, max_size) * wavel / (2*np.pi), n_small_size)
                if n_large_size > 0:
                    large_radii = np.linspace(max(x0 + 0.01, min_size) * wavel / (2*np.pi), smax, n_large_size)
            elif nsize[0] == 'Logarithmic':
                # Logarithmically spaced distributions will have linear spacing for small sizes and log spacing for large sizes
                # Whatever base is used for the large sizes will be the number of sizes in the small size array.
                if n_small_size > 0:
                    small_radii = np.linspace(smin, min(x0, max_size) * wavel / (2*np.pi), nsize[1])
                if n_large_size > 0:
                    large_radii = log_dist(max(x0 + 0.01, min_size) * wavel / (2*np.pi), smax, nsize[1])

            # Regardless of what was computed before, we calculate diameters and sizes normally and size distributions using a power law
            small_sizes = 2 * np.pi * small_radii / wavel
            large_sizes = 2 * np.pi * large_radii / wavel
            small_diameters = 2*small_radii
            large_diameters = 2*large_radii
            small_dist = small_radii**(-1*powlaw)
            large_dist = large_radii**(-1*powlaw)
            # Then we concatenate to get the full distributions, important for determining F and other things
            diameters = np.concatenate(((small_diameters if n_small_size > 0 else []), (large_diameters if n_large_size > 0 else [])))
            sizedist = np.concatenate(((small_dist if n_small_size > 0 else []), (large_dist if n_large_size > 0 else [])))
            sizeparams = np.concatenate(((small_sizes if n_small_size > 0 else []), (large_sizes if n_large_size > 0 else [])))

            # Grab specific index of refraction for this wavelength
            n, k = ns[j], ks[j]
            m = complex(n, k)

            # Calculating a ratio of the size of the large particle regime vs the size of the small particle regime
            # See Pollack & Cuzzi, 1980, eq. 7d
            F = np.trapz(large_dist * np.pi * large_sizes**2, large_sizes) / np.trapz(sizedist * np.pi * sizeparams**2, sizeparams)

            # --- SMALL REGIME PARTICLES ---
            # Default parameters in case there are no small particles
            SU_small = np.zeros_like(solid_angles)
            P_small = np.zeros_like(angle_range)
            Q_SS = 0
            Q_SA = 0
            if n_small_size > 0:
                # There is a specific case for which this needs to be optimized, otherwise the code will take eons to run
                # If only the mie theory is needed (no large sizes) and the output is requested in reflectances, there is a shortcut we can take.
                if n_large_size == 0 and (output == 'Reflectance' or output == 'Intensity'):
                    # Use PyMieScatt to compute scattering angles and intensities in the small regime
                    # Takes wavelength & diameters in nanometers, so need to convert from microns
                    # Note how this shortcut allows us to calculate the scattering intensity only within the angle range
                    # This is because reflectance doesn't have to be normalized by integrating over all solid angles
                    theta1, SL, SR, SU_small = PyMie.SF_SD(m, wavel*1000, small_diameters*1000, small_dist,
                                            minAngle=theta_min, maxAngle=theta_max,
                                            angularResolution=1.0, space='theta')
                        
                    # Calculate Q coefficients given the data, also applying conversions
                    qdict = PyMie.Mie_SD(m, wavel*1000., small_diameters*1000, small_dist, asDict=True)

                    # Extract extinction coefficient. PyMieScatt applies 10^-6 scale factor assuming
                    # we input size distribution in inverse cubic centimeters, and so outputs inverse
                    # megameters. But we input inverse cubic microns, and we want our output in
                    # inverse meters. This will work perfectly, but we do need to undo the 10^-6 
                    # scale factor.
                    bext = qdict['Bext'] * 1.e6

                    # Calculate tau for this wavelength from first available reflectance if tau is not provided
                    if tau is None:
                        tau_wavel = ref_if * bext * 4 * np.pi / (SU_small[0] * (wavel*1000)**2)
                    else:
                        # Otherwise, we use a fixed tau for all wavelengths
                        tau_wavel = tau

                    if output == 'Intensity':
                        # Append to relevant arrays
                        output_arr[j] += (float(comps[mat_idx]) if mixmodel == 'Areal' else 1) * SU_small
                        taus[j] += (float(comps[mat_idx]) if mixmodel == 'Areal' else 1) * tau_wavel

                        if updates[0]:
                            updates[1].put((i*n_wavels + j + 1)/n_iterations) # Put the total progress so far in the multiprocessing queue
                        # Skip everything else
                        continue
                    else:
                        # Calculate reflectance, converting wavelength to nanometers. SU_small is in the
                        # same units as sizedists, meaning we have square nanometers over cubic microns,
                        # which is the same as inverse meters. Our units thus cancel assuming tau and 
                        # reflectances are unitless.
                        # If this is being curve-fit to data and tau is input as None, this should make
                        # the output reflectance and input reflectance match at theta=angle_range[0]
                        refl=np.asarray(SU_small)*(wavel*1000)**2/(4*np.pi)*tau_wavel/bext

                        # Append to relevant arrays
                        output_arr[j] += (float(comps[mat_idx]) if mixmodel == 'Areal' else 1) * refl
                        taus[j] += (float(comps[mat_idx]) if mixmodel == 'Areal' else 1) * tau_wavel
                        
                        if updates[0]:
                            updates[1].put((i*n_wavels + j + 1)/n_iterations) # Put the total progress so far in the multiprocessing queue
                        # Skip everything else
                        continue

                # Use PyMieScatt to compute scattering angles and intensities in the small regime
                # Takes wavelength & diameters in nanometers, so need to convert from microns
                theta1, SL, SR, SU_small = PyMie.SF_SD(m, wavel*1000, small_diameters*1000, small_dist,
                                            minAngle=solid_angles_degs[0], maxAngle=solid_angles_degs[-1],
                                            angularResolution=1.0, space='theta')

                # SU is not normalized, but the phase function must be normalized to have an integral over all solid angles of 4*pi
                C_S = 1/(0.5 * np.trapz(SU_small * np.sin(solid_angles), solid_angles))
                # To get the phase function, we normalize the scattering intensity by multiplying by the constant
                P_small = (C_S * np.asarray(SU_small))[solid_min_idx:solid_max_idx+1]

                # Calculations of scattering and absorption efficiencies for small particles
                Q_s_denom = np.trapz(small_dist * np.pi * small_sizes**2, small_sizes) # Calculate denominator for total efficiency first so we don't have to recompute

                # Grab Qs for small particles across all diameters
                Q_small = [PyMie.MieQ(m, wavel*1000, diameter, asDict=True) for diameter in small_diameters*1000]
                # Small scattering efficiency as a function of size parameter
                Q_ssx = [q['Qsca'] for q in Q_small]
                # Total small particle Mie scattering efficiency, P&C eq. 6b (They said to reuse it for small particle scattering)
                Q_SS = np.trapz(Q_ssx * small_dist * np.pi * small_sizes**2, small_sizes) / Q_s_denom

                # Small absorption efficiency as a function of size parameter
                Q_sax = [q['Qabs'] for q in Q_small]
                # Total small particle Mie absorption efficiency, P&C eq. 6b (They said to reuse it for small particle absorption)
                Q_SA = np.trapz(Q_sax * small_dist * np.pi * small_sizes**2, small_sizes) / Q_s_denom

            # Default parameters in case there are no large particles (diffraction arrays are declared earlier)
            SU_large_extref = np.zeros_like(solid_angles)
            SU_large_trans = np.zeros_like(solid_angles)
            P_large_extref = np.zeros_like(angle_range)
            P_large_trans = np.zeros_like(angle_range)
            Q_D = 0
            Q_R = 0
            Q_T = 0
            Q_LS = 0
            Q_LA = 0
            if n_large_size > 0:
                # Calculations of scattering and absorption efficiencies for large particles
                Q_l_denom = np.trapz(large_dist * np.pi * large_sizes**2, large_sizes) # Calculate denominator for total efficiency first so we don't have to recompute

                # Grab Qs for large particles across all diameters
                Q_large = [PyMie.MieQ(m, wavel*1000, diameter, asDict=True) for diameter in large_diameters*1000]

                # Large cattering efficiency as a function of size parameter
                Q_lsx = [q['Qsca'] for q in Q_large]
                # Total large particle Mie scattering efficiency, P&C eq. 6b
                Q_LS = np.trapz(Q_lsx * large_dist * np.pi * large_sizes**2, large_sizes) / Q_l_denom

                # Large absorption efficiency as a function of size parameter
                Q_lax = [q['Qabs'] for q in Q_large]
                # Total large particle Mie absorptionefficiency, P&C eq. 6b (They said to reuse it for absorption)
                Q_LA = np.trapz(Q_lax * large_dist * np.pi * large_sizes**2, large_sizes) / Q_l_denom

                # --- LARGE REGIME: DIFFRACTION ---
                # We only compute the diffraction regime for one material
                if 'Semi-Empirical' in section or 'Diffraction' in section:
                    if i == 0:
                        # Since irregular particles have larger cross-sectional radii than their equivalent volume spheres, must multiply by sqrt(r)
                        rescaled_sizes = large_sizes * np.sqrt(r)
                        # We also need to rescale size distributions for later
                        rescaled_sizedist=(large_diameters/2)**(-1*powlaw)

                        # Cast theta and sizes as row/column arrays
                        # Theta needs to be split up at pi/4 for later
                        theta_2d_small = solid_angles[solid_angles < np.pi / 4, np.newaxis]
                        theta_2d_large = solid_angles[solid_angles >= np.pi / 4, np.newaxis]
                        theta_2d = np.concatenate((theta_2d_small, theta_2d_large))
                        sizes_2d = rescaled_sizes[np.newaxis, :]

                        # Create the z variable for the Bessel function (z = x*sin(theta), Pollack & Cuzzi eqn. 2b)
                        # Note this assumption is only valid when sin(theta) ~ theta, so we add only the part where that assumption is true (theta < pi/4)
                        z_small = sizes_2d * np.sin(theta_2d_small)
                        # J.R. Hodkinson and I. Greenleaves (1962) state that at large angles of scatter where theta !~ sin(theta), z=x*theta
                        z_large = sizes_2d * theta_2d_large
                        # Stack so that z_2d is a 2d array of shape (len(solid_angles), len(rescaled_sizes))
                        z_2d = np.vstack((z_small, z_large))

                        # Pollack & Cuzzi use a first-order Bessel function in calculating the diffraction
                        # However there is an indeterminate at theta=0, so the L'Hospital's rule limit must be applied
                        with np.errstate(divide='ignore', invalid='ignore'):
                            j1_term = sp.special.j1(z_2d) / z_2d # This makes another 2d array of same shape as z_2d
                            j1_term[z_2d == 0] = 0.5  # L'Hopital's rule limit as stated by Hodkinson and Greenleaves

                        # Calculating unnormalized phase function for diffraction as reported in Hodkinson and Greenleaves, 1963
                        # Again, creates a 2d array of same shape as z_2d
                        d_x_unnormalized = (sizes_2d**2 / 4 / np.pi) * (2 * j1_term)**2 * 0.5 * (1 + np.cos(theta_2d)**2)

                        # Getting the integrand over which to integrate for all rescaled sizes
                        # Again, creates a 2d array of same shape as z_2d. Column axis (len(sizes)) gets multiplied by sizes_2d^2 again and then by a row vector for sizedists
                        # Because sizes_2d and sizedists are row vectors with the same length as sizes_2d, they get copied for each row across d_x_unnormalized
                        integrand_unnormalized = d_x_unnormalized * np.pi * sizes_2d**2 * rescaled_sizedist[np.newaxis, :]

                        # Integration takes place without normalization constant
                        # We integrate over axis 1, the size axis. Result is a list with shape len(angles)
                        SU_large_diff = np.trapz(integrand_unnormalized, rescaled_sizes, axis=1)
                        
                        # Normalizing so that integral over all solid angles equals 4*pi
                        # We calculate C_D, the normalization constant, by taking the inverse of the integral of the unnormalized phase function
                        # divided by 4 * pi over dOmega, the solid angles. dOmega = sin(theta)dthetadphi, but there is no dependence on phi since theta is our colatitude scattering angle
                        # So we integrate with respect to phi to get 2pi in the numerator which cancels out with 4pi to leave 0.5
                        C_D = 1/(0.5 * np.trapz(SU_large_diff * np.sin(solid_angles), solid_angles))
                        # We now normalize to get the phase function by multiplying by the constant, getting only valid angles
                        P_large_diff_2d[j] = (C_D * np.asarray(SU_large_diff))[solid_min_idx:solid_max_idx+1]
                        SU_large_diff_2d[j] = SU_large_diff[solid_min_idx:solid_max_idx+1]

                    if 'Semi-Empirical' in section:
                        Q_D = 1 # Diffraction efficiency according to Pollack & Cuzzi
                    else: # If 'Mie Diffraction'
                        Q_D = Q_LS # Transmission term will take up entire phase function
                        # For intensity, we already have the output array; we can just append that and skip everything else
                        if output == 'Intensity':
                            # Append to relevant arrays
                            output_arr[j] += (float(comps[mat_idx]) if mixmodel == 'Areal' else 1) * SU_large_diff_2d[j] # Already sliced
                            
                            if updates[0]:
                                updates[1].put((i*n_wavels + j + 1)/n_iterations) # Put the total progress so far in the multiprocessing queue
                            # Skip everything else
                            continue

                # --- LARGE REGIME: EXTERNAL REFLECTION ---
                if 'Semi-Empirical' in section or 'External Reflection' in section:
                    m_bar_sqr = np.abs(m)**2 # Taking the magnitude of the index of refraction squared
                    # Get the unnormalized phase function for external reflection, which is a function of the solid angle theta
                    nan_idxs = (m_bar_sqr - 1 + np.sin(solid_angles / 2)**2) < 0
                    SU_large_extref[nan_idxs] = 1 # Clamp the m_bar_sqr - 1 + np.sin(solid_angles / 2)**2 to 0 to avoid negative square root. This yields an intensity of 1 (see Ellison & Peetz, 1959, section 3)
                    SU_large_extref[~nan_idxs] = 0.5*((np.sin(solid_angles[~nan_idxs] / 2) - (m_bar_sqr - 1 + np.sin(solid_angles[~nan_idxs] / 2)**2)**0.5)/(np.sin(solid_angles[~nan_idxs] / 2) + (m_bar_sqr - 1 + np.sin(solid_angles[~nan_idxs] / 2)**2)**0.5))**2 + 0.5 * ((m_bar_sqr * np.sin(solid_angles[~nan_idxs] / 2) - (m_bar_sqr - 1 + np.sin(solid_angles[~nan_idxs] / 2)**2)**0.5)/(m_bar_sqr * np.sin(solid_angles[~nan_idxs] / 2) + (m_bar_sqr - 1 + np.sin(solid_angles[~nan_idxs] / 2)**2)**0.5))**2

                    # Calculating the normalization constant, same process as diffraction
                    C_R = 1/(0.5 * np.trapz(SU_large_extref * np.sin(solid_angles), solid_angles))
                    # We now normalize the phase function by multiplying by the constant, getting it between valid angles
                    P_large_extref = (C_R * np.asarray(SU_large_extref))[solid_min_idx:solid_max_idx+1]
                    
                    if 'Semi-Empirical' in section:
                        Q_R = 1 / C_R # External reflection efficiency according to Pollack & Cuzzi, eq. 4
                    else: # If 'Mie External Reflection'
                        Q_R = Q_LS # Transmission term will take up entire phase function
                        # For intensity, we already have the output array; we can just append that and skip everything else
                        if output == 'Intensity':
                            # Append to relevant arrays
                            output_arr[j] += (float(comps[mat_idx]) if mixmodel == 'Areal' else 1) * SU_large_extref[solid_min_idx:solid_max_idx+1]
                            
                            if updates[0]:
                                updates[1].put((i*n_wavels + j + 1)/n_iterations) # Put the total progress so far in the multiprocessing queue
                            # Skip everything else
                            continue

                # --- LARGE REGIME: TRANSMISSION ---
                if 'Semi-Empirical' in section or 'Transmission' in section:
                    b = -2 * np.log(G) / np.pi # np.log is the natural log
                    SU_large_trans = np.e**(1 + b * solid_angles)

                    # Calculating the normalization constant, same process as diffraction
                    C_T = 1/(0.5 * np.trapz(SU_large_trans * np.sin(solid_angles), solid_angles))
                    # Now we need to get the scattering intensities between the valid angles (no more integrals are taken of SU_large_trans)
                    # We now normalize the phase function by multiplying by the constant, getting it between valid angles
                    P_large_trans = (C_T * np.asarray(SU_large_trans))[solid_min_idx:solid_max_idx+1]

                    if 'Semi-Empirical' in section:
                        Q_T = Q_LS - Q_D - Q_R # Transmission efficiency according to Pollack & Cuzzi eq. 6a
                    else: # If 'Mie Transmission'
                        Q_T = Q_LS # Transmission term will take up entire phase function
                        # For intensity, we already have the output array; we can just append that and skip everything else
                        if output == 'Intensity':
                            # Append to relevant arrays
                            output_arr[j] += (float(comps[mat_idx]) if mixmodel == 'Areal' else 1) * SU_large_trans[solid_min_idx:solid_max_idx+1]
                            
                            if updates[0]:
                                updates[1].put((i*n_wavels + j + 1)/n_iterations) # Put the total progress so far in the multiprocessing queue
                            # Skip everything else
                            continue

            # Total efficiencies between both regimes, weighted by ratio of large to small particles
            Q_star_S = Q_LS * F * r + Q_SS * (1 - F) # Total scattering efficiency, P&C eq. 7a
            Q_star_A = (Q_LA * F) * (r if 2*n*np.mean(large_sizes) > 1 else 1) + Q_SA * (1 - F) # Total absorption efficiency, P&C eq. 7b. They mention a case where the first term needs to be multiplied by r if 2nx>1. I take that to mean average large x.
            Q_star_E = Q_star_S + Q_star_A # Total extinction efficiency, P&C eq. 7c

            # Compute the combined phase function, P&C eq. 8. Note we cancel out Q_LS, avoiding 0/0 errors if Q_LS = 0
            P_star = P_small * (1 - F) * Q_SS / Q_star_S + r * F / Q_star_S * (P_large_diff_2d[j] * Q_D + P_large_extref * Q_R + P_large_trans * Q_T)
            
            # Large absorption efficiency as a function of size parameter
            Q_lex = [q['Qext'] for q in Q_large]
            
            # Now check the output and calculate phase or reflectance if requested
            if output == 'Phase Function':
                output_arr[j] += (float(comps[mat_idx]) if mixmodel == 'Areal' else 1) * P_star

            elif output == 'Intensity':
                output_arr[j] = np.asarray(SU_small)[solid_min_idx:solid_max_idx+1] * (1 - F) * Q_SS / Q_star_S + r * F / Q_star_S * (SU_large_diff_2d[j] * Q_D + SU_large_extref[solid_min_idx:solid_max_idx+1] * Q_R + SU_large_trans[solid_min_idx:solid_max_idx+1] * Q_T)

            elif output == 'Reflectance':
                omega_0 = Q_star_S / Q_star_E

                # Calculate tau from first available reflectance if not provided
                if tau is None:
                    # For a single scattering cloud, Reflectance = 0.25 * (Q_star_S / Q_star_E) * P_star * tau
                    tau_wavel = ref_if / (0.25 * omega_0 * P_star[0])
                else:
                    tau_wavel = tau

                # Calculate reflectance using scattering data
                refl = 0.25 * omega_0 * P_star * tau_wavel
                output_arr[j] += (float(comps[mat_idx]) if mixmodel == 'Areal' else 1) * refl
                taus[j] += (float(comps[mat_idx]) if mixmodel == 'Areal' else 1) * tau_wavel

            # Progress updates -- assuming the loop gets to this point
            if updates[0]:
                updates[1].put((i*n_wavels + j + 1)/n_iterations) # Put the total progress so far in the multiprocessing queue
            
    # Return a reflectances/phase functions as well as tau
    return output_arr.tolist(), taus.tolist()

def henyey_greenstein(angle, gweights):
    """
    Computes probability density of light scattering at a given angle using a two-term Henyey-Greenstein function
    with provided weights g (asymmetry parameter) and w (unnormalized term weight)

    :param angle: Integer angle in radians at which to calculate probability density
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

def fresn_surface_reflectances(wave_um, param, comps={vars.COMP, 1}, mixmodel='Molecular', updates=[False, None]):
    """
    Compute the surface albedo of a material over various wavelengths from Fresnel reflectances.

    :param wave_um: List of float wavelengths at which to evaluate in microns
    :param comps: Dictionary with keys as materials and values as their respective v/v fractions
    :param param: Float effective grain size (average particle radius, microns)
    :param mixmodel: String, either 'Molecular' (averaging optical constants) or 'Areal' (averaging reflectances)
    :param updates: List containing boolean of whether or not to print progress update fraction after each 
                    wavelength iteration as well as multiprocessing queue to pass it to
    :returns: List of reflectances evaluated. Shape len(wavels)
    """
    # Figure out whether to iterate over all materials and sum weighted reflectances (areal model)
    # or all at once and get the averaged index of refraction (molecular model)
    comp_names = comps.keys() if mixmodel == 'Areal' else range(1)
    reflectances = np.zeros_like(wave_um, dtype=float)

    # Convert effective grain size & wavelengths to meters for calculations
    s = param * 1e-6
    wave = (np.asarray(wave_um) * 1e-6).tolist()

    # Get length of arrays for progress tracking
    n_comps = len(comp_names)
    n_wave = len(wave)
    n_iterations = n_comps * n_wave

    # Loop over all materials and average reflectance if areal
    # Otherwise, evaluate all at once and average optical constants
    for idx, mat in enumerate(comp_names):
        # Get optical constants for these wavelengths
        n, k = get_nk(wave_um, ({mat: comps[mat]} if mixmodel == 'Areal' else comps), mixmodel=mixmodel)

        # Compute surface albedo
        nnx=np.size(wave)
        theta1=np.linspace(0,np.pi/4,100)
        dtheta1=theta1[1]-theta1[0]
        theta2=theta1+np.pi/4
        alb1=np.zeros(nnx)
        for i in range(nnx):
            aS=4*np.pi*k[i]/wave[i]*s
            rs1=(np.cos(theta1)-(n[i]+k[i]*1.0j)*np.sqrt(1-np.sin(theta1)**2/n[i]**2))/(np.cos(theta1)+(n[i]+k[i]*1.0j)*np.sqrt(1-np.sin(theta1)**2/n[i]**2))
            rp1=((n[i]+k[i]*1.0j)*np.cos(theta1)-np.sqrt(1-np.sin(theta1)**2/n[i]**2))/((n[i]+k[i]*1.0j)*np.cos(theta1)+np.sqrt(1-np.sin(theta1)**2/n[i]**2))
            ro1=.5*(np.abs(rs1)**2+np.abs(rp1)**2)
            rb=np.sum(2*ro1*np.cos(theta2)*np.sin(theta2))*dtheta1
            rs2=(np.cos(theta2[np.sin(theta2)<n[i]])-(n[i]+k[i]*1.0j)*np.sqrt(1-np.sin(theta2[np.sin(theta2)<n[i]])**2/n[i]**2))/(np.cos(theta2[np.sin(theta2)<n[i]])+(n[i]+k[i]*1.0j)*np.sqrt(1-np.sin(theta2[np.sin(theta2)<n[i]])**2/n[i]**2))
            rp2=((n[i]+k[i]*1.0j)*np.cos(theta2[np.sin(theta2)<n[i]])-np.sqrt(1-np.sin(theta2[np.sin(theta2)<n[i]])**2/n[i]**2))/((n[i]+k[i]*1.0j)*np.cos(theta2[np.sin(theta2)<n[i]])+np.sqrt(1-np.sin(theta2[np.sin(theta2)<n[i]])**2/n[i]**2))
            ro2=.5*(np.abs(rs2)**2+np.abs(rp2)**2)
            rf=np.sum(2*ro2*np.cos(theta2[np.sin(theta2)<n[i]])*np.sin(theta2[np.sin(theta2)<n[i]]))*dtheta1
            re=rb+rf
            ri=1-(1-re)/n[i]**2
            rrb=rb+0.5*(1-re)*(1-ri)*ri*np.exp(-2*aS)/(1-ri*np.exp(-aS))
            rrf=rf+(1-re)*(1-ri)*np.exp(-aS)+0.5*(1-re)*(1-ri)*ri*np.exp(-2*aS)/(1-ri*np.exp(-aS))
            alb1[i]=(1+rrb**2-rrf**2)/2/rrb-np.sqrt((1+rrb**2-rrf**2)**2/4/rrb**2-1)

            if updates[0]:
                updates[1].put((idx * n_wave + i + 1)/n_iterations)

            

        reflectances += (float(comps[mat]) if mixmodel == 'Areal' else 1) * alb1

    return reflectances

def surface_albedo(wave,n,k,s,mat_idx,updates=[False, None]):
    """
    Calculates albedo from Fresnel reflectances for a given range of wavelengths and optical constants n and k 
    with effective grain size s.

    :param wave: List of float wavelengths at which to evaluate, in meters
    :param n: Float index of refraction of material
    :param k: Float index of absorption of material
    :param s: Float effective grain size, in meters
    :returns: List of unitless reflectances (albedos) evaluated. Shape len(wave)
    """
    
    return alb1

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
    :param dirs_removed: Integer, how many parent directories to go out to before creating the graph folder
    :returns: String path of resultant directory
    """
    # Define the path for the output directory
    graph_dir = os.path.join("../graphs", dirname)
    # Adds ../ for each parent directory indicated by dirs_removed
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
    # Plots that use this function have log-log scales anyway, so this makes the report more accurate
    # to what users see on the graph. However, it is not taking into account in MAPE.
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
    if r_squared < 0.5: # Report if fit has very low goodness-of-fit
        print(f"CRITICAL WARNING: Terrible global fit! Log R^2 is incredibly low ({r_squared:.2f}).")
    else:
        print(f"Fit Quality: Log R^2 = {r_squared:.4f}")
        
    if mape > 25.0: # Report if fit deviates from the original data by a massive percentage
        print(f"CRITICAL WARNING: Fit deviates by an average of {mape:.1f}%. Massive inaccuracy detected.")
    else:
        print(f"Average Error: {mape:.1f}%")

    # Determine if there are spikes in error (spikes in fit)
    print("--- Local Spike Analysis ---")
    raw_diffs = log_act - log_pred
    mean_raw = np.mean(raw_diffs) # Get mean
    std_raw = np.std(raw_diffs) # Get standard deviation

    if std_raw == 0:
        print("No variance in error. Perfect relative smoothness.")
        return

    # Calculate Z-scores. If Z-score is greater than 3, that usually indicates a spike
    zscores = np.abs((raw_diffs - mean_raw) / std_raw)
    points = np.where(zscores >= 3)[0]
    
    if len(points) > 0: # Indicate which thetas have spikes in the fit
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
    :param if_orig: Array of original reflectances (unitless), interpolated to thetas
    :param if_fit: Array of reflectances (unitless) from line of best fit
    :param params: Array of parameters required to produce given fit. Shape varies based on model used.
    :param use_mie: Boolean for determining whether Mie or Henyey-Greenstein methods were used.
    """
    # Plot setup - using a log scale plot, plus a space for text to display parameters
    fig, (iflogplt, txt) = plt.subplots(1, 2, figsize=(12, 8), layout='constrained')
    plt_RMSE = RMSE(if_orig, if_fit) # Calculate overall error between datasets
    print(f"\nRMSE ({vars.ANGLE_LOWERBOUND}-{vars.ANGLE_UPPERBOUND} degrees): {plt_RMSE}")
    diff_analysis(if_orig, if_fit, theta) # Run error analysis to determine accuracy of fit
    # Plotting the given datasets
    iflogplt.plot(theta, if_orig, color='black', label=f'Reflectance from data')
    iflogplt.plot(theta, if_fit, color='blue', label=f"Computed {'Mie' if use_mie else 'Henyey-Greenstein'} (RMSE={plt_RMSE:.2e})")

    # Cosmetic setup
    iflogplt.set_xlabel('Scattering Angle (degrees)')
    iflogplt.set_ylabel('Reflectance (I/F)')
    iflogplt.set_title(f'{vars.DATA_FILE[:-4]} Reflectance vs Scattering Angle')
    iflogplt.grid()
    iflogplt.legend()
    iflogplt.set_yscale('log') # Only y should be the log scale

    # Fill in textbox with fit parameters
    if use_mie:
        txt.text(0.25, 0.5, f"Mie Params:\nsmin={params[0]:.4f}\nsmax={params[1]:.4f}\npowlaw={params[2]:.4f}\nr={params[3]:.4f}\nG={params[4]:.4f}\nx0={params[5]:.4f}\ntau={params[6]:.2e}", bbox=dict(facecolor='lightblue', alpha=0.5, edgecolor='black', pad=10), fontsize=15)
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
    :param wavels: Array of wavelengths in microns at which to extract reflectance data
    :param params: Array of particle distribution parameters calculated from Mie theory.
    :param angle: Integer scattering angle in degrees at which to loop over wavelengths
    """
    reflectances = []
    # Iterate through all given wavelengths to get a reflectance at the given angle, pass existing tau
    for w in wavels:
        reflectance_wavel = angle_mie_reflectances(params[0], params[1], params[2], r=params[3], G=params[4], x0=params[5], theta_min=angle, theta_max=angle, wavels=[w], tau=params[6])[0][0][0]
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
    txt.text(0.25, 0.5, f"Mie Params:\nsmin={params[0]:.4f}\nsmax={params[1]:.4f}\npowlaw={params[2]:.4f}\nr={params[3]:.4f}\nG={params[4]:.4f}\nx0={params[5]:.4f}\ntau={params[6]:.2e}", bbox=dict(facecolor='lightblue', alpha=0.5, edgecolor='black', pad=10), fontsize=15)

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
    :param mie_ifs: Array of reflectances (unitless) computed from Mie theory
    :param hg_ifs: Array of reflectances (unitless) computed from Henyey-Greenstein function
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
    iflogplt.set_yscale('log') # Only y should have a log scale

    # Fill in textbox with necessary parameters
    txt.text(0.3, 0.6, f"Mie Params:\nsmin={params[0]:.4f}\nsmax={params[1]:.4f}\npowlaw={params[2]:.4f}\ntau={params[3]:.2e}", bbox=dict(facecolor='lightblue', alpha=0.5, edgecolor='black', pad=10), fontsize=15)
    txt.text(0.3, 0.3, f"Henyey-Greenstein Params:\nw1={hg_params[0]:.2f}\ng1={hg_params[1]:.2f}\nw2={hg_params[2]:.2f}\ng2={hg_params[3]:.2f}\nwscale={hg_params[4]:.2e}", bbox=dict(facecolor='pink', alpha=0.5, edgecolor='black', pad=10), fontsize=15)
    
    # Save to file
    plt.savefig(fig_path)
    print(f"Angle I/F plot saved to {fig_path}")

def plt_wavel(fig_path, wavels, angles, plt_angles, reflectances, params):
    """
    Plots and saves several graphs of reflectances varied over different wavelenghts at different scattering angles
    in the same window, using the size distributions given. As only Mie theory is dependent on wavelength, 
    this graph will only be generated if Mie theory is used.

    :param fig_path: String path to which to save graph, including image name
    :param wavels: Array of wavelengths in microns comprising row-axis of reflectances
    :param angles: Array of angles in degrees comprising column-axis of reflectances
    :param plt_angles: Array of angles in degrees for which to produce graphs
    :param reflectances: 2D Numpy array of reflectances (unitless) at different angles and wavelengths. Shape 
                         (len(wavels), len(angles))
    :param params: Array of particle distribution parameters calculated from Mie theory.
    """

    # Plot setup, plus a space for text to display parameters
    fig, (ifplt, txt) = plt.subplots(1, 2, figsize=(12, 8), layout='constrained')

    # Iterate through each angle and plot an array of reflectances vs. wavelengths for that angle
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
    A function that takes input reflectance, angle, and wavelength data and outputs it to a csv file, with rows 
    being wavelengths and columns being reflectances.

    :param file_path: String path to which to save csv
    :param wavels: Row-axis float array for data in microns
    :param angles: Column-axis integer array for data in degrees
    :param reflectances: 2D Numpy array containing unitless reflectance data. Shape (len(wavels), len(angles))
    """

    # Open the csv writer
    with open(file_path, 'w') as f:
        writer = csv.writer(f)
        header = ["Wavelength (μm) | Scattering Angle (°)"] # Left element is for row headers, right is for column headers
        header.extend(angles) # Since array is two-dimensional, top left box will be title, and top axis will be angles
        writer.writerow(header)
        # Iterate through each wavelength and append the reflectances at different scattering angles in new rows
        for idx, wavel_slice in enumerate(reflectances):
            row = [wavels[idx]]
            row.extend(wavel_slice) # Left axis will be wavelengths, plus all values to the right
            writer.writerow(row)
    print(f"Wavelength-angle reflectance table saved to {file_path}")

    