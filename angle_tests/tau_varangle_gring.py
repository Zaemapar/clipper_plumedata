#import all the required libraries
#If PyMieScatt not installed add following line:
#!pip install PyMieScatt
#QUESTION: Should we copy PyMieScatt's routine into the workbook as a way to stabilize the code?

import numpy as np
import scipy as sp
import csv
import matplotlib.pyplot as plt
import PyMieScatt as PyMie
import os
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
smin = 5  #Minimum particle size (radius in microns)
smax = 10   #Maximum particle size (radius in microns)

powlaw = -2.8 # Power law index for particle size distribution of G Ring, as reported in https://academic.oup.com/mnras/article/538/2/1118/8043278
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
    return SU, bext, theta1, reflectances

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

def RMSE(y_act, y_pred):
    return np.sqrt(np.mean((y_act-y_pred)**2))

def compute_tau(idx, reflectances, wavel, SU, bext):
    # Given the appropriate data, reverse calculate the optical depth from reflectance
    tau = reflectances[idx] * bext * 4 * np.pi / (SU[idx] * (wavel*1000)**2)
    return tau

def mkifplot(given_theta, given_if, reflectances_interp, tau0_reflectances_interp, tauf_reflectances_interp, tau_0, tau_f):
    # Plot setup - using a linear and log scale plot
    fig, (iflogplt, ifplt) = plt.subplots(1, 2, figsize=(12, 8), layout='constrained')

    # Compute mean absolute values of dataset
    RMSE_tau0 = RMSE(given_if, tau0_reflectances_interp)
    RMSE_tauf = RMSE(given_if, tauf_reflectances_interp)

    # Plotting the given datasets
    ifplt.plot(given_theta, given_if, color='black', label='Given Reflectance')
    #ifplt.plot(given_theta, reflectances_interp, color='blue',label='Computed (tau=tau_nom)')
    ifplt.plot(given_theta, tau0_reflectances_interp, color='red', label=f'Computed (tau={tau_0:.1e}, RMSE={RMSE_tau0:.3e})')
    ifplt.plot(given_theta, tauf_reflectances_interp, color='blue', label=f'Computed (tau={tau_f:.1e}, RMSE={RMSE_tauf:.3e})')
    iflogplt.plot(given_theta, given_if, color='black', label='Given Reflectance')
    #iflogplt.plot(given_theta, reflectances_interp, color='blue', label='Computed (tau=tau_nom)')
    iflogplt.plot(given_theta, tau0_reflectances_interp, color='red', label=f'Computed (tau={tau_0:.1e}), RMSE={RMSE_tau0:.3e}')
    iflogplt.plot(given_theta, tauf_reflectances_interp, color='blue', label=f'Computed (tau={tau_f:.1e}), RMSE={RMSE_tauf:.3e}')

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

    graph_dir = utils.create_dirpath("tau_varangle_gring", dirs_removed=1)
    # Save to file
    plt.savefig(os.path.join(graph_dir, "tau_test.png"))

if __name__ == "__main__":
    # Compute the nominal (unnormalized) particle size distribution
    radii=np.linspace(smin,smax,nsize)
    dr=radii[1]-radii[0]
    diameters=radii*2
    sizedists=radii**(powlaw)
    #C ompute the Geometric optical depth of the nominal particle size distribution (Not currently Used)
    taunom=np.sum(sizedists*np.pi*radii**2*dr)
    # Obtain the optical constants at the desired wavelength
    n,k=get_nk(wavel,comp)
    # Compute the reflectance (I/F) of the particle population.
    #Note this uses the same normalization process as described in Appendix B of
    #de Pater et al. 2026 Characterization of the Outer Uranian Rings.... in JGR Planets 131 e2025JE009404.
    #doi.org/10.1029/2025JE00404
    m = complex(n, k) # Compute complex refractive index

    # Compute the unpolarized light scattering intensity, extintion coefficient, angles, and reflectances based on calculated particle distribution and tau
    SU, bext, theta1, reflectances = get_reflectances(0, 180, wavel, diameters, sizedists, m, taunom)
    
    # Extract the desired angles and observed reflectances from the data table
    given_theta, given_if = get_plt_data("../data/gring.csv")
    
    # Interpolate calculated data to given theta values
    refxfunc = sp.interpolate.interp1d(theta1, reflectances)
    SUxfunc = sp.interpolate.interp1d(theta1, SU)
    reflectances_interp = refxfunc(given_theta) # Interpolating computed reflectances to match angles in data table
    SU_interp = SUxfunc(given_theta) # Interpolating SU to match angles in data table

    # Compute optical depths off of first and last scattering angles
    tau_0 = compute_tau(0, given_if, wavel, SU_interp, bext)
    tau_f = compute_tau(-1, given_if, wavel, SU_interp, bext)

    # Use optical depths to recompute reflectances at given angles, to compare with data table
    _, _, _, tau0_reflectances = get_reflectances(0, 180, wavel, diameters, sizedists, m, tau_0)
    _, _, _, tauf_reflectances = get_reflectances(0, 180, wavel, diameters, sizedists, m, tau_f)

    # Interpolate these reflectances to match the given angles in the data table
    tau0_refxfunc = sp.interpolate.interp1d(theta1, tau0_reflectances)
    tauf_refxfunc = sp.interpolate.interp1d(theta1, tauf_reflectances)
    tau0_reflectances_interp = tau0_refxfunc(given_theta)
    tauf_reflectances_interp = tauf_refxfunc(given_theta)

    # Display results
    print(f"tau_nom: {taunom:.1e}, tau_0: {tau_0:.1e}, tau_f: {tau_f:.1e}")

    # Make plot
    mkifplot(given_theta, given_if, reflectances_interp, tau0_reflectances_interp, tauf_reflectances_interp, tau_0, tau_f)
