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

smin = 5  #Minimum particle size (radius in microns)
smax = 10   #Maximum particle size (radius in microns)
powlaw = -2.8 # Power law index for particle size distribution
tau = 1e-6 # Fixed tau, for the purpose of plotting
nsize = 51  #Number of particle sizes to evaluate size distribution
#NOTE: In the current code this is the observed optical depth, not the geometrical optical depth.
wavel = 0.647 # Detector reflected emitted wavelength in microns
comp = 'Water Ice' # Composition of particles (currently only option is water ice)

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
    bext = qdict['Bext'] * 1.e6
    # Calculate reflectance using scattering data, converting wavelength to nanometers
    reflectances=np.asarray(SU)*(wavel*1000)**2/(4*np.pi)*tau/bext
    return theta1, reflectances

def mkifplot(given_theta, reflectances_interp, conditions):
    # Plot setup - using a linear and log scale plot
    fig, (iflogplt, ifplt) = plt.subplots(1, 2, figsize=(12, 8), layout='constrained')

    # Plotting the given datasets
    ifplt.plot(given_theta, reflectances_interp, color='blue', label=f'Computed (smin={conditions[0]},smax={conditions[1]},powlaw={conditions[2]},tau={conditions[3]:.1e})')
    iflogplt.plot(given_theta, reflectances_interp, color='blue', label=f'Computed (smin={conditions[0]},smax={conditions[1]},powlaw={conditions[2]},tau={conditions[3]:.1e})')

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

    graph_dir = utils.create_dirpath("gring_Mie_plot", dirs_removed=1)
    # Save to file
    plt.savefig(os.path.join(graph_dir, "gring_Mie_test.png"))

if __name__ == "__main__":
    # Compute the nominal (unnormalized) particle size distribution
    radii=np.linspace(smin,smax,nsize)
    dr=radii[1]-radii[0]
    diameters=radii*2
    sizedists=radii**(powlaw)
    # Obtain the optical constants at the desired wavelength
    n,k=get_nk(wavel,comp)
    # Compute the reflectance (I/F) of the particle population.
    #Note this uses the same normalization process as described in Appendix B of
    #de Pater et al. 2026 Characterization of the Outer Uranian Rings.... in JGR Planets 131 e2025JE009404.
    #doi.org/10.1029/2025JE00404
    m = complex(n, k) # Compute complex refractive index

    # Compute the unpolarized light scattering intensity, extintion coefficient, angles, and reflectances based on calculated particle distribution and tau
    theta1, reflectances = get_reflectances(0, 180, wavel, diameters, sizedists, m, tau)

    conditions = [smin, smax, powlaw, tau]
    # Make plot
    mkifplot(theta1 * 180 / np.pi, reflectances, conditions) # Convert angles to degrees because PyMieScatt outputs radians
