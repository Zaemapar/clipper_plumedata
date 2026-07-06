"""
A file containing various helper methods to organize and read files, perform major computations relating to
scattering theories, or output graphs given an input dataset.
Author: Parker A. Zaemann
Date: 06 Jul 2026
"""

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
        next(reader) # Skip column header row
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

def get_nk(wavels, comps, mixmodel='Molecular', returnmode=['full']):
    """
    Determines complex index of refraction n + ki of a material given its composition and the incident wavelength.

    :param wavel: Float wavelength in microns
    :param comps: Dictionary of string compositions of material and given volume fractions
    :param mixmodel: String mixture model to use, only 'Molecular' is supported
    :param returnmode: List indicating whether to return full wavels, ns, ks array (['full']), or whether to 
                       return only one n, k pair (['single', wavel])
    :returns: Float n (refractive index) at the given wavelength
              Float k (absorption coefficient) at the given wavelength
    """

    # Create empty arrays to hold all the data
    dielectrics, fs, temp_ns, temp_ks = [], [], [], []

    for comp in comps.keys():
        # Material files MUST be formatted with "_constants.csv" at the end!
        idx_refraction_path = os.path.join("data", comp + "_constants.csv")
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
    if returnmode[0] == 'full':
        return ns, ks
    else:
        # Return an n,k pair at the given wavelength
        wavel_idx = np.where(np.asarray(wavels) == returnmode[1])[0][0]
        return ns[wavel_idx], ks[wavel_idx]

def get_minmax_wavel(comp):
    """
    Reads the same file as get_nk, but searches for min and max wavelength supported.

    :param comp: String composition of material
    :returns: Float minimum wavelength supported
              Float maximum wavelength supported
    """
    idx_refraction_path = os.path.join("data", comp + "_constants.csv")
    
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
    files = os.listdir("data") # List all files
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

def get_data(src, altitude=None):
    """
    Reads data from a given sample and extracts scattering angle (or wavelength) and corresponding I/F reflectance.
    Parses header using string search to determine how to format the data.

    :param src: String path to data file
    :param altitude: Float altitude if reading data that depends on it
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
            reflectance_idx = np.where(np.asarray(header) == str(altitude))[0] # Find the column corresponding to the input altitude
            if_scale_factor = -2 # Equivalent width is usually listed in those tables, to convert to tau divide by 100
        else:
            for idx, headstring in enumerate(header):
                lc_headstring = headstring.lower()
                # Check to see if it's an angle column
                if ("theta" in lc_headstring or "angle" in lc_headstring) and "range" not in lc_headstring:
                    theta_idx = idx
                    # Phase vs scattering angle needs to be tracked so operations can be performed in the end
                    # Ultimately if both are provided the code will use whichever one comes last
                    if "phase" in lc_headstring:
                        phase_theta = True
                    else:
                        phase_theta = False
                # Check to see if it's a reflectance column
                elif "reflect" in lc_headstring or "contr" in lc_headstring:
                    reflectance_idx = idx
                    # Find where it is multiplied by a scale factor, if any. Reflectance often is.
                    modifier_idx = lc_headstring.find("10^")
                    if modifier_idx > 0:
                        if_scale_factor = int(lc_headstring[(modifier_idx + 3):])

        # Iterate through each line in the rest of the document
        for line in reader:
            # Append scattering angle (convert to scattering if phase)
            thetas.append(float(line[theta_idx]) if not phase_theta else 180 - float(line[theta_idx]))
            # Append reflectance & scale by scale factor
            reflectances.append(float(line[reflectance_idx]) * 10**if_scale_factor)

    return thetas, reflectances

def angle_mie_reflectances(smin, smax, powlaw, r=0, G=1, x0=np.inf, theta_min=vars.ANGLE_LOWERBOUND, theta_max=vars.ANGLE_UPPERBOUND, wavels=[vars.WAVEL], nsize=vars.NSIZE, comps={vars.COMP: 1}, mixmodel='Molecular', tau=None, ref_if=0, section='Semi-Empirical Mie', output='Reflectance'):
    """
    Computes a 2D array of reflectances given arrays of angles and wavelengths. By default, a semi-empirical Mie
    theory is used based on methods from Pollack & Cuzzi's 'Scattering of Nonspherical Particles of Size Comparable 
    to a Wavelength: A New Semi-Empirical Theory and Its Application to Troposhperic Aerosols' (1979), which uses
    pure Mie theory for small particles and a combination of diffraction, external reflection, and transmission
    theories for large particles.

    :param smin: Float minimum size used in size distribution
    :param smax: Float maximum size used in size distribution
    :param powlaw: Float power law used in size distribution
    :param G: Float constant to use in Mie transmission equation
    :param r: Float ratio of the surface area of an irregular particle to the surface area of a sphere of the same 
              volume
    :param x0: Float size cutoff above which to use semi-empirical methods and below which to use pure Mie
    :param theta_min: Integer minimum angle with which to curve fit to original data
    :param theta_max: Integer maximum angle with which to curve fit to original data
    :param wavel: Array of wavelengths at which to evaluate Mie
    :param nsize: Integer number of particle sizes to use (resolution)
    :param comps: Dictionary with keys as mixture materials and values as volume fractions of materials
    :param mixmodel: String mixture model to use, either 'Areal' or 'Molecular'  
    :param tau: Float allowing for optical detph tau to be input, rather than calculated
    :param ref_if: Float reflectance from original data from which to calculate optical depth tau. ref_if MUST 
                   correspond to theta_min.
    :param section: String indicating which term of semi-empirical Mie methods to use
    :param output: String indicating what to return ('Reflectance' for reflectances or 'Phase Function' for phase 
                   function)
    :returns: 2D Array of reflectance values between theta_min, theta_max, and the min and max of wavels. Shape 
                (len(wavels), len(angles))
              Array for calculated optical depths at each wavelength
    """

    # Compute the nominal (unnormalized) particle size distribution
    radii=np.linspace(smin, smax, nsize)
    diameters=radii*2
    sizedists=radii**(-1*powlaw)

    # Empty array to hold reflectances/phase function and taus
    output_arr = []
    taus = []

    angle_range = np.radians(np.arange(theta_min, theta_max + 1, 1)) # Range of given angles in radians
    solid_angles_degs = np.arange(0, 181, 1) # All colatitude angles in the range of solid angles, for integrals
    solid_angles = np.radians(solid_angles_degs) # All colatitude angles in radians
    solid_min_idx = np.where(solid_angles_degs == theta_min)[0][0] # Locate given angle bounds within solid angles
    solid_max_idx = np.where(solid_angles_degs == theta_max)[0][0]

    # Figure out whether to iterate over all materials and sum weighted reflectances (areal model)
    # or all at once and get the averaged index of refraction (molecular model)
    outer_range = comps.keys() if mixmodel == 'Areal' else range(1)

    # Iterate through each provdied wavelength
    for wavel in wavels:
        wavel_tau = 0 # Create placeholders for summing up all taus
        mat_output = np.zeros_like(angle_range) # Placeholder for reflectances/phase function for this material at this wavelength
        for mat_idx in outer_range:
            sizeparams = 2 * np.pi * radii / wavel # This is the size parameter by which we gague small and large particles
            # We split up our size distribution arrays into small and large sections
            small_idxs = np.where(sizeparams <= x0)[0]
            large_idxs = np.where(sizeparams > x0)[0]
            small_sizes = sizeparams[small_idxs]
            large_sizes = sizeparams[large_idxs]
            small_diameters = diameters[small_idxs]
            large_diameters = diameters[large_idxs]
            small_dist = sizedists[small_idxs]
            large_dist = sizedists[large_idxs]

            # Compute index of refraction at given wavelength
            n, k = get_nk(wavels, ({mat_idx: comps[mat_idx]} if mixmodel == 'Areal' else comps), mixmodel=mixmodel, returnmode=['single', wavel])
            m = complex(n, k)

            # Calculating a ratio of the size of the large particle regime vs the size of the small particle regime
            # See Pollack & Cuzzi, 1980, eq. 7d
            F = np.trapz(large_dist * np.pi * large_sizes**2, large_sizes) / np.trapz(sizedists * np.pi * sizeparams**2, sizeparams)

            # --- SMALL REGIME PARTICLES ---
            # Default parameters in case there are no small particles
            SU_small = np.zeros_like(angle_range)
            P_small = np.zeros_like(angle_range)
            Q_SS = 0
            Q_SA = 0
            if len(small_idxs) > 0:
                # Use PyMieScatt to compute scattering angles and intensities in the small regime
                # Takes wavelength & diameters in nanometers, so need to convert from microns
                theta1, SL, SR, SU_small = PyMie.SF_SD(m, wavel*1000, small_diameters*1000, small_dist,
                                            minAngle=solid_angles_degs[0], maxAngle=solid_angles_degs[-1],
                                            angularResolution=1.0, space='theta')

                # Calculate coefficients given the data, also applying conversions
                qdict_small = PyMie.Mie_SD(m, wavel*1000., small_diameters*1000, small_dist, asDict=True)
                bsca_small = qdict_small['Bsca'] * 1.e6 # Extract extinction coefficient bext

                # Extract extinction coefficient. PyMieScatt applies 10^-6 scale factor assuming we
                # input size distribution in inverse cubic centimeters, and so outputs inverse
                # megameters. But we input inverse cubic microns, and we want our output in inverse
                # meters. This will work perfectly, but we do need to undo the 10^-6 scale factor.
                bext_small = qdict_small['Bext'] * 1.e6

                # SU is not normalized, but the phase function must be normalized to have an integral over all solid angles of 4*pi
                C_S = 1/(0.5 * np.trapz(SU_small * np.sin(solid_angles), solid_angles))
                # To get the phase function, we normalize the scattering intensity by multiplying by the constant
                P_small = (C_S * np.asarray(SU_small))[solid_min_idx:solid_max_idx+1]

                # Calculations of scattering and absorption efficiencies for small particles
                Q_s_denom = np.trapz(small_dist * np.pi * small_sizes**2, small_sizes) # Calculate denominator for total efficiency first so we don't have to recompute

                # Small scattering efficiency as a function of size parameter
                Q_ssx = [PyMie.MieQ(m, wavel*1000, diameter, asDict=True)['Qsca'] for diameter in small_diameters*1000]
                # Total small particle Mie scattering efficiency, P&C eq. 6b (They said to reuse it for small particle scattering)
                Q_SS = np.trapz(Q_ssx * small_dist * np.pi * small_sizes**2, small_sizes) / Q_s_denom

                # Small absorption efficiency as a function of size parameter
                Q_sax = [PyMie.MieQ(m, wavel*1000, diameter, asDict=True)['Qabs'] for diameter in small_diameters*1000]
                # Total small particle Mie absorption efficiency, P&C eq. 6b (They said to reuse it for small particle absorption)
                Q_SA = np.trapz(Q_sax * small_dist * np.pi * small_sizes**2, small_sizes) / Q_s_denom

            # Default parameters in case there are no large particles
            SU_large_diff = np.zeros_like(angle_range)
            SU_large_extref = np.zeros_like(angle_range)
            SU_large_trans = np.zeros_like(angle_range)
            P_large_diff = np.zeros_like(angle_range)
            P_large_extref = np.zeros_like(angle_range)
            P_large_trans = np.zeros_like(angle_range)
            Q_D = 0
            Q_R = 0
            Q_T = 0
            Q_LS = 0
            Q_LA = 0
            if len(large_idxs) > 0:
                # --- LARGE REGIME: DIFFRACTION ---
                SU_large_diff = []
                # Since irregular particles have larger cross-sectional radii than their equivalent volume spheres, must multiply by sqrt(r)
                rescaled_sizes = large_sizes * np.sqrt(r)
                # We also need to rescale size distributions for later
                rescaled_sizedists=(large_diameters/2)**(-1*powlaw)
                for theta in solid_angles:
                    # For z = x * sin(theta)
                    z = rescaled_sizes * np.sin(theta)

                    # Pollack & Cuzzi use a first-order Bessel function in calculating the diffraction
                    # However there is an indeterminate at theta=0, so the L'Hospital's rule limit must be applied
                    with np.errstate(divide='ignore', invalid='ignore'):
                        j1_term = sp.special.j1(z) / z
                        j1_term[z == 0] = 0.5  # L'Hopital's rule limit

                    # Calculating unnormalized phase function for diffraction as reported in Hodkinson and Greenleaves, 1963
                    d_x_unnormalized = (rescaled_sizes**2 / 4 / np.pi) * (2 * j1_term)**2 * 0.5 * (1 + np.cos(theta)**2)

                    # Getting the integrand over which to integrate for all rescaled sizes
                    integrand_unnormalized = d_x_unnormalized * np.pi * rescaled_sizes**2 * rescaled_sizedists
                    
                    # Integration takes place without normalization constant
                    if section == 'Mie Diffraction' or theta < np.pi / 2:
                        SU_large_diff.append(np.trapz(integrand_unnormalized, rescaled_sizes))
                    else:
                        # Bessel function has a periodic spike. Pollack and Cuzzi assume this term drops to 0 in the backscatter region anyways
                        # So we clamp it here
                        SU_large_diff.append(0)
                
                # Normalizing so that integral over all solid angles equals 4*pi
                # We calculate C_D, the normalization constant, by taking the inverse of the integral of the unnormalized phase function
                # divided by 4 * pi over dOmega, the solid angles. dOmega = sin(theta)dthetadphi, but there is no dependence on phi since theta is our colatitude scattering angle
                # So we integrate with respect to phi to get 2pi in the numerator which cancels out with 4pi to leave 0.5
                C_D = 1/(0.5 * np.trapz(SU_large_diff * np.sin(solid_angles), solid_angles))
                # Now we need to get the scattering intensities between the valid angles (no more integrals are taken of SU_large_diff)
                SU_large_diff = SU_large_diff[solid_min_idx:solid_max_idx+1]
                # We now normalize to get the phase function by multiplying by the constant
                P_large_diff = (C_D * np.asarray(SU_large_diff))

                # --- LARGE REGIME: EXTERNAL REFLECTION ---
                m_bar_sqr = np.abs(m)**2 # Taking the magnitude of the index of refraction squared
                # Get the unnormalized phase function for external reflection, which is a function of the solid angle theta
                SU_large_extref = 0.5*((np.sin(solid_angles / 2) - (m_bar_sqr - 1 + np.sin(solid_angles / 2)**2)**0.5)/(np.sin(solid_angles / 2) + (m_bar_sqr - 1 + np.sin(solid_angles / 2)**2)**0.5))**2 + 0.5 * ((m_bar_sqr * np.sin(solid_angles / 2) - (m_bar_sqr - 1 + np.sin(solid_angles / 2)**2)**0.5)/(m_bar_sqr * np.sin(solid_angles / 2) + (m_bar_sqr - 1 + np.sin(solid_angles / 2)**2)**0.5))**2
                
                # Calculating the normalization constant, same process as diffraction
                C_R = 1/(0.5 * np.trapz(SU_large_extref * np.sin(solid_angles), solid_angles))
                # Now we need to get the scattering intensities between the valid angles (no more integrals are taken of SU_large_extref)
                SU_large_extref = SU_large_extref[solid_min_idx:solid_max_idx+1]
                # We now normalize the phase function by multiplying by the constant
                P_large_extref = (C_R * np.asarray(SU_large_extref))

                # --- LARGE REGIME: TRANSMISSION ---
                b = -2 * np.log(G) / np.pi # np.log is the natural log
                SU_large_trans = np.e**(1 + b * solid_angles)

                # Calculating the normalization constant, same process as diffraction
                C_T = 1/(0.5 * np.trapz(SU_large_trans * np.sin(solid_angles), solid_angles))
                # Now we need to get the scattering intensities between the valid angles (no more integrals are taken of SU_large_trans)
                SU_large_trans = SU_large_trans[solid_min_idx:solid_max_idx+1]
                # We now normalize the phase function by multiplying by the constant
                P_large_trans = (C_T * np.asarray(SU_large_trans))

                Q_D = 1 # Diffraction efficiency according to Pollack & Cuzzi
                Q_R = 1 / C_R # External reflection efficiency according to Pollack & Cuzzi, eq. 4

                # Calculations of scattering and absorption efficiencies for large particles
                Q_l_denom = np.trapz(large_dist * np.pi * large_sizes**2, large_sizes) # Calculate denominator for total efficiency first so we don't have to recompute

                # Scattering efficiency as a function of size parameter
                Q_lsx = [PyMie.MieQ(m, wavel*1000, diameter, asDict=True)['Qsca'] for diameter in large_diameters*1000]
                # Total large particle Mie scattering efficiency, P&C eq. 6b
                Q_LS = np.trapz(Q_lsx * large_dist * np.pi * large_sizes**2, large_sizes) / Q_l_denom
                Q_T = Q_LS - Q_D - Q_R # Transmission efficiency according to Pollack & Cuzzi eq. 6a

                # Large absorption efficiency as a function of size parameter
                Q_lax = [PyMie.MieQ(m, wavel*1000, diameter, asDict=True)['Qabs'] for diameter in large_diameters*1000] # Scattering efficiency as a function of size parameter
                # Total large particle Mie absorptionefficiency, P&C eq. 6b (They said to reuse it for absorption)
                Q_LA = np.trapz(Q_lax * large_dist * np.pi * large_sizes**2, large_sizes) / Q_l_denom
            # Check through the provided section to see which terms to include in the final reflectance/phase function. Void all terms not included
            # Pure Mie is already taken care of by the fact that x0 is set to infinity, so all particles are small
            if section == 'Mie Diffraction':
                P_small = np.zeros_like(angle_range)
                P_large_extref = np.zeros_like(angle_range)
                P_large_trans = np.zeros_like(angle_range)
                Q_T = 0
                Q_R = 0
                Q_D = Q_LS
            elif section == 'Mie External Reflection':
                P_small = np.zeros_like(angle_range)
                P_large_diff = np.zeros_like(angle_range)
                P_large_trans = np.zeros_like(angle_range)
                Q_R = Q_LS
                Q_D = 0
                Q_T = 0
            elif section == 'Mie Transmission':
                P_small = np.zeros_like(angle_range)
                P_large_diff = np.zeros_like(angle_range)
                P_large_extref = np.zeros_like(angle_range)
                Q_T = Q_LS
                Q_D = 0
                Q_R = 0

            # Total efficiencies between both regimes, weighted by ratio of large to small particles
            Q_star_S = Q_LS * F * r + Q_SS * (1 - F) # Total scattering efficiency, P&C eq. 7a
            Q_star_A = Q_LA * F + Q_SA * (1 - F) # Total absorption efficiency, P&C eq. 7b
            Q_star_E = Q_star_S + Q_star_A # Total extinction efficiency, P&C eq. 7c

            # Compute the combined phase function, P&C eq. 8. Note we cancel out Q_LS, avoiding 0/0 errors if Q_LS = 0
            P_star = P_small * (1 - F) * Q_SS / Q_star_S + r * F / Q_star_S * (P_large_diff * Q_D + P_large_extref * Q_R + P_large_trans * Q_T)

            # Now check the output and calculate phase or reflectance if requested
            if output == 'Phase Function':
                mat_output += P_star
 
            elif output == 'Reflectance':
                # Calculate tau from first available reflectance if not provided
                if tau is None:
                    # For a single scattering cloud, Reflectance = 0.25 * (Q_star_S / Q_star_E) * P_star * tau
                    omega_0 = Q_star_S / Q_star_E
                    tau = ref_if / (0.25 * omega_0 * P_star[0])

                # Calculate reflectance using scattering data
                omega_0 = Q_star_S / Q_star_E
                refl = 0.25 * omega_0 * P_star * tau
                mat_output += (float(comps[mat_idx]) if mixmodel == 'Areal' else 1) * refl
                wavel_tau += (float(comps[mat_idx]) if mixmodel == 'Areal' else 1) * tau
        
        output_arr.append(mat_output.tolist()) # Stack the material-weighted reflectances for this wavelength
        taus.append(wavel_tau)

    # Return a slice of reflectances/phase functions at the appropriate angles, as well as tau
    return output_arr, taus

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

def output_graph(graphmode, sensor, comps, mixmodel, datamodel, fitmodel, param, tau, wavelbounds=(0, np.inf), output='Reflectance'):
    """
    A function that takes input scattering data as might be observed by one of Europa Clipper's instruments,
    as well as additional data about the fit and the plot type desired, and outputs the corresponding x and y
    arrays.

    :param graphmode: Integer, indicates whether to output reflectance vs. wavelength (0), surface reflectance
                     vs. wavelength (1), or reflectance vs. scattering angle (2)
    :param sensor: String sensor from which to read wavelength range
    :param comps: Dictionary with keys as materials and values as volume fractions of materials
    :param mixmodel: String mixture model to use, either 'Areal' or 'Molecular'
    :param datamodel: Array of size distribution strings [min particle size, max particle size, power law,
                      r, G, x0]
    :param fitmodel: String for fit model to use, either 'Mie' or 'Henyey-Greenstein'
    :param param: Fixed parameter to base reflectances on. Float wavelength if graphmode = 0, float effective
                  grain size if graphmode = 1, int scattering angle if graphmode = 2
    :param tau: Float optical depth
    :param wavelbounds: Tuple (float, float) wavelength bounds as established by the composition, representing
                        data available for index of refraction interpolation
    :param output: String indicating the type of output to return, either 'Reflectance' or 'Phase Function'
    :returns: List containing x data values
              List containing float reflectance/phase function data values
    """

    # Try to extract r, G, and tau from the data; if they are NaNs, set them to their default values
    if datamodel[3] == 'NaN':
        r = 0
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

    # Angle mode
    if graphmode == 1:
        wavel_range = pvars.INSTRUMENTS[sensor] # Plot over range of all available wavelengths in the sensor
        xarr = np.arange(round(max(wavelbounds[0][1], wavel_range[0]), 5), round(min(wavelbounds[1][1], wavel_range[1]), 5) + 1e-20, 0.01) # Assume a 10 nm resolution. Get the most limiting bounds
        reflectances = fresn_surface_reflectances(xarr, param, comps=comps, mixmodel=mixmodel)
    elif graphmode == 2:
        xarr = np.arange(0, 181, 1) # Declare array of angles 0 to 180 degrees
        # Get reflectances. Programmed so that angles start, stop, and step the same way as xarr
        reflectances = angle_mie_reflectances(float(datamodel[0]), float(datamodel[1]), float(datamodel[2]), r=r, G=G, x0=x0, theta_min=0, theta_max=180, wavels=[param], comps=comps, mixmodel=mixmodel, nsize=pvars.NSIZE, tau=tau, section=fitmodel, output=output)[0][0]

        # For Henyey-Greenstein, curve fitting is needed to determine global scale factor
        if fitmodel == 'Henyey-Greenstein':
            # Iterate through given data models and see if input data model matches.
            # If so, get the HG parameters for that data model
            init_hg_params = []
            for model in pvars.DATA_MODELS.values():
                if datamodel == model[0]:
                    init_hg_params = model[1]
                    break
            
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
    elif graphmode == 0:
        wavel_range = pvars.INSTRUMENTS[sensor] # Plot over range of all available wavelengths in the sensor
        xarr = np.arange(round(max(wavelbounds[0][1], wavel_range[0]), 5), round(min(wavelbounds[1][1], wavel_range[1]), 5) + 1e-20, 0.01) # Assume a 10 nm resolution. Get the most limiting bounds
        reflectances = angle_mie_reflectances(float(datamodel[0]), float(datamodel[1]), float(datamodel[2]), r=r, G=G, x0=x0, theta_min=param, theta_max=param, wavels=xarr, comps=comps, mixmodel=mixmodel, nsize=pvars.NSIZE, tau=tau)[0]
        # angle_mie_reflectances returns a weird column array shape [[element1], [element2], [element3]]
        # Need to make array not as deep
        reflectances = [element[0] for element in reflectances]

    return xarr.tolist(), reflectances # xarr should not be a numpy array because PyQt6 doesn't know what numpy is for some reason

def fresn_surface_reflectances(wavels, param, comps={vars.COMP, 1}, mixmodel='Molecular'):
    """
    Compute the surface albedo of a material over various wavelengths using the Fresnel equations.

    :param wavels: List of float wavelengths at which to evaluate
    :param comps: Dictionary with keys as materials and values as their respective v/v fractions
    :param param: Float effective grain size (average particle radius)
    :param mixmodel: String, either 'Molecular' (averaging optical constants) or 'Areal' (averaging reflectances)
    :returns: List of reflectances evaluated. Shape len(wavels)
    """
    # Figure out whether to iterate over all materials and sum weighted reflectances (areal model)
    # or all at once and get the averaged index of refraction (molecular model)
    outer_range = comps.keys() if mixmodel == 'Areal' else range(1)
    reflectances = np.zeros_like(wavels)

    # Loop over all materials and average reflectance if areal
    # Otherwise, evaluate all at once and average optical constants
    for mat_idx in outer_range:
        n, k = get_nk(wavels, ({mat_idx: comps[mat_idx]} if mixmodel == 'Areal' else comps), mixmodel=mixmodel)
        reflectances += (float(comps[mat_idx]) if mixmodel == 'Areal' else 1) * surface_albedo(wavels, n, k, param)

    return reflectances

def surface_albedo(wave,n,k,s):
    """
    Solves the Fresnel equations for a given range of wavelengths and optical constants n and k with effective
    grain size s to compute the surface albedo of a material.

    :param wave: List of float wavelengths at which to evaluate
    :param n: Float index of refraction of material
    :param k: Float index of absorption of material
    :param s: Float effective grain size
    :returns: List of reflectances (albedos) evaluated. Shape len(wave)
    """
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
        header.extend(angles) # Since array is two-dimensional, top left box will be title, and top axis will be angles
        writer.writerow(header)
        for idx, wavel_slice in enumerate(reflectances):
            row = [wavels[idx]]
            row.extend(wavel_slice) # Left axis will be wavelengths, plus all values to the right
            writer.writerow(row)
    print(f"Wavelength-angle reflectance table saved to {file_path}")

    