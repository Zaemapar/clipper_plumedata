#import all the required libraries
#If PyMieScatt not installed add following line:
#!pip install PyMieScatt
#QUESTION: Should we copy PyMieScatt's routine into the workbook as a way to stabilize the code?

import numpy as np
import scipy as sp
import csv
import matplotlib.pyplot as plt
import PyMieScatt as PyMie
import sys
import os

# Get the path to the directory above the current one
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

# Add it to sys.path
sys.path.append(parent_dir)

# Import utils file
import utils

#Define Input Parameters:
#COMMENT: These should eventually be provided by an input interface or text file

# The VIMS IR spectra detects these sizes (https://mmhedman.github.io/papers_published/encplume_apj.pdf)
smin = 6  #Minimum particle size (radius in microns)
smax = 11 #Maximum particle size (radius in microns)
phase_angle = 161 # As reported in https://mmhedman.github.io/papers_published/encplume_apj.pdf
powlaw_seed = 3 # Seed around which to try different power laws
nsize = 51  #Number of particle sizes to evaluate size distribution
#NOTE: In the current code this is the observed optical depth, not the geometrical optical depth.
comp = 'Water Ice' #Composition of particles (currently only option is water ice)

#Define function for reading optical constants of water ice from Warren and Brandt 2008
#Optical constants of ice from the ultraviolet to the microwave: A revised compilation
#JGR Atmospheres 113: D14 https://doi.org/10.1029/2007JD009744
#I'm using this version since is covers the full UV-Visible-Near IR spectral range
#COMMENT: Requires external csv file to be in same directory, maybe should update to be in stable other directory

def wb08read():
    """
    Extract information on complex index of refraction from provided datasheets for given material.

    :returns: Numpy array for given wavelength,
              Numpy array for n values (refractive indices) at each wavelength,
              Numpy array for k values (absorbtion coefficients) at each wavelength
    """
    wavew, nw, kw, temp = [], [], [], []

    with open('../data/WB08_Iceconstants.csv', 'r') as f:
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

#Define function to obtain optical constants of particles from specified wavelength and composition

def get_nk(wavel,comp):
    if comp == 'Water Ice':
        # Calling function to read appropriate tables of optical constants for water ice
        wavex,nx,kx=wb08read()

    # Interpolate to given wavelength
    nxfunc=sp.interpolate.interp1d(wavex,nx)
    kxfunc=sp.interpolate.interp1d(wavex,kx)
    n=nxfunc(wavel)
    k=kxfunc(wavel)
    return n,k

def get_reflectances(theta, wavel, diameters, sizedists, tau, comp, arr=False):
    reflectances = []
    bexts = []
    SUs = []
    for idx, w in enumerate(wavel):
        # Get the complex refractive index at the given wavelength
        n,k=get_nk(w,comp)
        #Compute the reflectance (I/F) of the particle population.
        #Note this uses the same normalization process as described in Appendix B of
        #de Pater et al. 2026 Characterization of the Outer Uranian Rings.... in JGR Planets 131 e2025JE009404.
        #doi.org/10.1029/2025JE00404
        m = complex(n, k) # Compute complex refractive index

        # Use PyMieScatt to compute scattering angles and intensities
        # Takes wavelength & diameters in nanometers, so need to convert from microns        
        theta1, SL, SR, SU = PyMie.SF_SD(m, w*1000, diameters*1000, sizedists,
                                         minAngle=theta, maxAngle=theta, 
                                         angularResolution=1.0, space='theta')
        # Calculate coefficients given the data, also applying conversions
        qdict = PyMie.Mie_SD(m, w*1000., diameters*1000, sizedists, asDict=True)
        # Extract extinction coefficient. PyMieScatt applies 10^-6 scale factor assuming we
        # input size distribution in inverse cubic centimeters, and so outputs inverse
        # megameters. But we input inverse cubic microns, and we want our output in inverse
        # meters. This will work perfectly, but we do need to undo the 10^-6 scale factor.
        bext = qdict['Bext'] * 1.e6
        # Calculate reflectance using scattering data, converting wavelength to nanometers. SU
        # is in the same units as sizedists, meaning we have square nanometers over cubic microns,
        # which is the same as inverse meters. Our units thus cancel assuming tau and reflectances
        # should be unitless.
        reflectances.append(SU[0]*(w*1000)**2/(4*np.pi)*(tau if not arr else tau[idx])/bext)
        SUs.append(SU[0])
        bexts.append(bext)
    return SUs, bexts, np.asarray(reflectances) # Return all appropriate data across all wavelengths (reflectances needs to be numpy to use for RSME later)

def get_plt_data(src):
    wavelength = []
    eqwidths = []

    with open(src, "r") as f:
        reader = csv.reader(f)
        header = next(reader) # Skip header row
        altitude = header[1:] # Remove title element & extract altitude values

        for line in reader:
            # Append wavelength and equivalent width (will be a 2D array since there are multiple altitudes)
            wavelength.append(float(line[0]))
            eqwidths.append([float(x) for x in line[1:]]) # Convert equivalent width values to floats and store as list

    return wavelength, np.asarray(eqwidths), altitude # Need to cast reflectance as a numpy array for slicing

def compute_density(idx, reflectances, wavel, SU, bext):
    # Given the appropriate data, reverse calculate the particle densityfrom reflectance
    N = reflectances[idx] * 4 * np.pi / (SU[idx] * (wavel[idx]*1000)**2)
    return N

def RMSE(y_act, y_pred):
    return np.sqrt(np.mean((y_act-y_pred)**2))

def mkifplot(alt, graph_dir, given_theta, reflectances, powlaw_seed, powlaw_tests, MIN_RMSE=False):
    RMSEs = []

    # Name of plot needs to contain altitude to distinguish it
    alt_path = os.path.join(graph_dir, f"altitude_{alt}.png")

    # Plot setup - using a linear and log scale plot
    fig, (iflogplt, ifplt) = plt.subplots(1, 2, figsize=(12, 8), layout='constrained')

    # Plot original dataset
    ifplt.plot(given_theta, reflectances[0], color='black', label='Given Reflectance')
    iflogplt.plot(given_theta, reflectances[0], color='black', label='Given Reflectance')

    # Plot seed powerlaw
    RMSEs.append(RMSE(reflectances[0], reflectances[1]))
    ifplt.plot(given_theta, reflectances[1], color='red', label=f'Computed (powlaw={powlaw_seed}, RMSE={RMSEs[0]:.3e})')
    iflogplt.plot(given_theta, reflectances[1], color='red', label=f'Computed (powlaw={powlaw_seed}, RMSE={RMSEs[0]:.3e})')

    # Plot each power law
    for idx, powlaw in enumerate(powlaw_tests):
        RMSEs.append(RMSE(reflectances[0], reflectances[idx + 2]))
        ifplt.plot(given_theta, reflectances[idx + 2], label=f'Computed (powlaw={powlaw}, RMSE={RMSEs[idx + 1]:.3e})')
        iflogplt.plot(given_theta, reflectances[idx + 2], label=f'Computed (powlaw={powlaw}, RMSE={RMSEs[idx + 1]:.3e})')
        
    # Labels and cosmetic setup
    ifplt.set_xlabel('Wavelength (microns)')
    ifplt.set_ylabel('Reflectance (I/F)')
    ifplt.set_title(f'Reflectance vs Wavelength (Altitude {alt} km)')
    ifplt.legend()
    ifplt.grid()
    iflogplt.set_xlabel('Wavelength (microns)')
    iflogplt.set_ylabel('Reflectance (I/F)')
    iflogplt.set_title(f'Reflectance vs Wavelength (Log Scale, altitude {alt} km)')
    iflogplt.grid()
    iflogplt.set_yscale('log')

    # Save to file
    plt.savefig(alt_path if not MIN_RMSE else os.path.join(graph_dir, f"model_encleadus_{alt}.png"))

    if not MIN_RMSE:
        RMSE_min = min(RMSEs)
        powlaw_idx = RMSEs.index(RMSE_min) - 1
        model_accurate = reflectances[powlaw_idx + 2]

        # Define a path for a third directory for most accurate models
        model_dir = "models"

        # Create the additional directory
        try:
            os.makedirs(os.path.join(graph_dir, model_dir), exist_ok=True)
        except OSError as error:
            print(f"Error creating directory: {error}") # Error handling

        mkifplot(alt, os.path.join(graph_dir, model_dir), given_theta, [reflectances[0], model_accurate], (powlaw_tests[powlaw_idx] if powlaw_idx >= 0 else powlaw_seed), [], MIN_RMSE=True)

if __name__ == "__main__":
    scatter_angle = 180 - phase_angle # Convert to scattering angle
    #Compute the nominal (unnormalized) particle size distribution
    radii=np.linspace(smin,smax,nsize)
    dr=radii[1]-radii[0]
    diameters=radii*2
    # Extract the desired wavelengths, altitudes, and observed reflectances from the data table
    given_wavelengths, given_eqwidths, given_altitudes = get_plt_data("../data/group1_encleadus.csv")

    # Define the path for the output directory
    graph_dir = utils.create_dirpath("powlaw_varwavel_encleadus", dirs_removed=1)

    # Create the directory to contain the plots
    try:
        os.makedirs(graph_dir, exist_ok=True)
    except OSError as error:
        print(f"Error creating directory: {error}") # Error handling

    # Setup values to test for power laws, everywhere around the seed
    powlaw_tests = [i for i in range(max(0, powlaw_seed - 5), powlaw_seed + 5)]
    powlaw_tests.remove(powlaw_seed)

    # Define a path for a second directory, for these specific initial conditions
    conditions_dir = f"powlaw{powlaw_tests[0]}-{powlaw_tests[-1]}_wavel_test"

    # Create the additional directory
    try:
        os.makedirs(os.path.join(graph_dir, conditions_dir), exist_ok=True)
    except OSError as error:
        print(f"Error creating directory: {error}") # Error handling

    # Loop through each altitude. We will use a separate graph for each. Higher altitudes should be more accurate.
    for idx, alt in enumerate(given_altitudes):
        reflectances = [] # Empty array to hold reflectances
        
        # Perform calculations with the seed power law. This will allow us to fix a value of tau.
        sizedists=radii**(-1 * powlaw_seed)
        #Compute the Geometric optical depth of the nominal particle size distribution (Not currently Used)
        taunom=np.sum(sizedists*np.pi*radii**2*dr)

        # Get a slice of the reflectance from the data table along the given altitude
        eqwidth_slice = given_eqwidths[:, idx]

        # To find optical depth, we divide equivalent width by 100 km
        taus_0 = eqwidth_slice / 100
        tau = eqwidth_slice[0] / 100

        # Get initial reflectance data & seed powerlaw data
        _, _, tau0_reflectances = get_reflectances(scatter_angle, given_wavelengths, diameters, sizedists, taus_0, comp, arr=True)
        _, _, tau_reflectances = get_reflectances(scatter_angle, given_wavelengths, diameters, sizedists, tau, comp)

        # Add to reflectances array
        reflectances.append(tau0_reflectances)
        reflectances.append(tau_reflectances)

        # Loop through power law tests
        for powlaw in powlaw_tests:
            # Compute new distribution
            sizedists=radii**(-1 * powlaw)

            # Use the optical depths to compute reflectances at given wavelengths, to compare with data table
            _, _, tau_reflectances = get_reflectances(scatter_angle, given_wavelengths, diameters, sizedists, tau, comp)

            # Add to reflectances array
            reflectances.append(tau_reflectances)

        # Make plot
        mkifplot(alt, os.path.join(graph_dir, conditions_dir), given_wavelengths, reflectances, powlaw_seed, powlaw_tests)

        # Update status
        print(f"Completed plot for altitude {alt} km")
