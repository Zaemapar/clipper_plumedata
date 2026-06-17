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

# The following particle sizes dominate the G ring (https://academic.oup.com/mnras/article/538/2/1118/8043278)
smin = 8 #Minimum particle size (radius in microns)
smax_seed = smin + 5 # Seed around which to try different maximum particle size (radius in microns)

powlaw = 3 # Power law for size distribution
nsize = 51  #Number of particle sizes to evaluate size distribution
#NOTE: In the current code this is the observed optical depth, not the geometrical optical depth.
wavel = 0.647 # Wavelength in microns from RED filter on WAC camera as reported in https://iopscience.iop.org/article/10.1088/0004-637X/811/1/67/meta#apj519262fn5
comp = 'Water Ice' #Composition of particles (currently only option is water ice)

#Define function for reading optical constants of water ice from Warren and Brandt 2008
#Optical constants of ice from the ultraviolet to the microwave: A revised compilation
#JGR Atmospheres 113: D14 https://doi.org/10.1029/2007JD009744
#I'm using this version since is covers the full UV-Visible-Near IR spectral range
#COMMENT: Requires external csv file to be in same directory, maybe should update to be in stable other directory

def wb08read():
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

def get_reflectances(theta_min, theta_max, wavel, diameters, sizedists, m, tau):
    # Use PyMieScatt to compute scattering angles and intensities
    # Takes wavelength & diameters in nanometers, so need to convert from microns
    theta1, SL, SR, SU = PyMie.SF_SD(m, wavel*1000, diameters*1000, sizedists,
                                     minAngle=theta_min, maxAngle=theta_max,
                                     angularResolution=1.0, space='theta')
    # Calculate coefficients given the data, also applying conversions
    qdict = PyMie.Mie_SD(m, wavel*1000., diameters*1000, sizedists, asDict=True)
    # Extract extinction coefficient
    bsca = qdict['Bsca'] * 1.e6
    bext = qdict['Bext'] * 1.e6 # Why is this here?
    # Calculate reflectance using scattering data, converting wavelength to nanometers
    reflectances=np.asarray(SU)*(wavel*1000)**2/(4*np.pi)*tau/bext
    return SU, bext, theta1, np.asarray(reflectances) # Return all appropriate data across all wavelengths (reflectances needs to be numpy to use for RSME later)

def get_plt_data(src):
    avg_thet = []
    reflectance = []
    scatter = []

    with open("../data/gring.csv", "r") as f:
        reader = csv.reader(f)
        next(reader) # Skip header row

        for line in reader:
            # Append scattering angle and reflectance
            avg_thet.append(float(line[1])*np.pi/180) # Convert from degrees to radians
            reflectance.append(float(line[3]) * 10**(-6))

    return avg_thet, reflectance

def compute_tau(idx, reflectances, wavel, SU, bext):
    # Given the appropriate data, reverse calculate the optical depth from reflectance
    tau = reflectances[idx] * bext * 4 * np.pi / (SU[idx] * (wavel*1000)**2)
    return tau

def RMSE(y_act, y_pred):
    return np.sqrt(np.mean((y_act-y_pred)**2))

def mkifplot(graph_dir, given_theta, reflectances, smin, smax_seed, smax_tests):
    # Plot setup - using a linear and log scale plot
    fig, (iflogplt, ifplt) = plt.subplots(1, 2, figsize=(12, 8), layout='constrained')

    # Plot original dataset

    ifplt.plot(given_theta, reflectances[0], color='black', label='Given Reflectance')
    iflogplt.plot(given_theta, reflectances[0], color='black', label='Given Reflectance')

    # Plot seed maximum particle size
    RMSE_seed = RMSE(reflectances[0], reflectances[1])
    ifplt.plot(given_theta, reflectances[1], color='red', label=f'Computed (smax={smax_seed}, RMSE={RMSE_seed:.3e})')
    iflogplt.plot(given_theta, reflectances[1], color='red', label=f'Computed (smax={smax_seed}, RMSE={RMSE_seed:.3e})')

    # Plot each maximum particle size
    for idx, smax in enumerate(smax_tests):
        RMSE_smax = RMSE(reflectances[0], reflectances[idx + 2])
        ifplt.plot(given_theta, reflectances[idx + 2], label=f'Computed (smax={smax}, RMSE={RMSE_smax:.3e}))')
        iflogplt.plot(given_theta, reflectances[idx + 2], label=f'Computed (smax={smax}, RMSE={RMSE_smax:.3e}))')

    # Labels and cosmetic setup
    ifplt.set_xlabel('Scattering Angle (degrees)')
    ifplt.set_ylabel('Reflectance (I/F)')
    ifplt.set_title('Reflectance vs Scattering Angle')
    ifplt.legend()
    ifplt.grid()
    iflogplt.set_xlabel('Scattering Angle (degrees)')
    iflogplt.set_ylabel('Reflectance (I/F)')
    iflogplt.set_title('Reflectance vs Scattering Angle (Log Scale)')
    iflogplt.grid()
    iflogplt.set_yscale('log')
    iflogplt.set_xscale('log')

    # Save to file
    plt.savefig(os.path.join(graph_dir, f"smin{smin}_smax{smax_tests[0]}-{smax_tests[-1]}_varangle.png"))

if __name__ == "__main__":
    # Compute the nominal (unnormalized) particle size distribution around powerlaw ss
    radii=np.linspace(smin,smax_seed,nsize)
    dr=radii[1]-radii[0]
    diameters=radii*2
    # Obtain the optical constants at the desired wavelength
    n,k=get_nk(wavel,comp)
    # Compute the reflectance (I/F) of the particle population.
    #Note this uses the same normalization process as described in Appendix B of
    #de Pater et al. 2026 Characterization of the Outer Uranian Rings.... in JGR Planets 131 e2025JE009404.
    #doi.org/10.1029/2025JE00404
    m = complex(n, k) # Compute complex refractive index
    
    # Define the path for the output directory
    graph_dir = utils.create_dirpath("smax_varangle_gring", dirs_removed=1)

    # Extract the desired angles and observed reflectances from the data table
    given_theta, given_if = get_plt_data("../data/gring.csv")

    reflectances = [] # Empty array to hold reflectances
    
    # Perform calculations with the seed max particle size. This will allow us to fix a value of tau.
    sizedists=radii**(-1 * powlaw)
    #Compute the Geometric optical depth of the nominal particle size distribution for this max particle size
    taunom=np.sum(sizedists*np.pi*radii**2*dr)
    # Compute the unpolarized light scattering intensity, extintion coefficient, and angles based on calculated particle distribution and tau
    SU, bext, theta1, _ = get_reflectances(0, 180, wavel, diameters, sizedists, m, taunom)
    
    # Interpolate calculated data to given theta values
    SUxfunc = sp.interpolate.interp1d(theta1, SU)
    SU_interp = SUxfunc(given_theta) # Interpolating SU to match angles in data table

    # Compute optical depths off of first scattering angle
    tau = compute_tau(0, given_if, wavel, SU_interp, bext)

    # Use optical depths to recompute reflectances at given angles, to compare with data table
    _, _, _, tau_reflectances = get_reflectances(0, 180, wavel, diameters, sizedists, m, tau)

    # Interpolate these reflectances to match the given angles in the data table
    tau_refxfunc = sp.interpolate.interp1d(theta1, tau_reflectances)
    tau_reflectances_interp = tau_refxfunc(given_theta)

    # Display results
    print(f"tau_nom: {taunom:.1e}, tau: {tau:.1e}")

    # Add interpolated reflectances to array. Also add original data
    reflectances.append(given_if)
    reflectances.append(tau_reflectances_interp)

    # Setup values to test for maximum particle sizes. Everywhere from smin to smin + 10, except the seed.
    smax_tests = [i for i in range(smin, smin + 11)]
    smax_tests.remove(smax_seed)

    # Iterate through each test
    for smax in smax_tests:
        radii=np.linspace(smin,smax,nsize)
        diameters=radii*2
        sizedists=radii**(-1 * powlaw) # Get size distribution for this maximum particle size
        # Use optical depths & minimum size to recompute reflectances at given angles and fixed tau
        _, _, _, tau_reflectances = get_reflectances(0, 180, wavel, diameters, sizedists, m, tau)
        
        # Interpolate these reflectances to match the given angles in the data table
        tau_refxfunc = sp.interpolate.interp1d(theta1, tau_reflectances)
        tau_reflectances_interp = tau_refxfunc(given_theta)
        reflectances.append(tau_reflectances_interp)

        # Update once finished
        print(f"Added max size {smax:.1f}")

    # Make plot
    mkifplot(graph_dir, given_theta, reflectances, smin, smax_seed, smax_tests)
