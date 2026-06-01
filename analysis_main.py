#import all the required libraries
#If PyMieScatt not installed add following line:
#!pip install PyMieScatt
#QUESTION: Should we copy PyMieScatt's routine into the workbook as a way to stabilize the code?

import numpy as np
import scipy as sp
import csv
import matplotlib.pyplot as plt
import PyMieScatt as PyMie

#Define Input Parameters:
#COMMENT: These should eventually be provided by an input interface or text file

smin = 0.1  #Minimum particle size (radius in microns)
smax = 5    #Maximum particle size (radius in microns)
powlaw = -3 #Power-law index for particle size distribution
nsize = 51  #Number of particle sizes to evaluate size distribution
tau = 1e-3  #Optical depth of partcle plume
#NOTE: In the current code this is the observed optical depth, not the geometrical optical depth.
wavel = 0.5 #Observed wavelength (in microns)
phase = 145 #Observed phase angle (in degrees)
comp = 'Water Ice' #Composition of particles (currently only option is water ice)

#Define function for reading optical constants of water ice from Warren and Brandt 2008
#Optical constants of ice from the ultraviolet to the microwave: A revised compilation
#JGR Atmospheres 113: D14 https://doi.org/10.1029/2007JD009744
#I'm using this version since is covers the full UV-Visible-Near IR spectral range
#COMMENT: Requires external csv file to be in same directory, maybe should update to be in stable other directory

def wb08read():
    wavew, nw, kw, temp = [], [], [], []

    with open('WB08_Iceconstants.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader) # Skip header row
        #line=line.strip(',')
        #print(line)
        for line in reader:
            wavew.append(float(line[0]))
            nw.append(float(line[1]))
            kw.append(float(line[2]))

    wavew=np.array(wavew)
    nw=np.array(nw)
    kw=np.array(kw)
    return wavew,nw,kw

#Define function to obtain optical constants of particles from specified wavelength and composition

def get_nk(wavel,comp):
    if comp == 'Water Ice':
        wavex,nx,kx=wb08read()
    nxfunc=sp.interpolate.interp1d(wavex,nx)
    kxfunc=sp.interpolate.interp1d(wavex,kx)
    n=nxfunc(wavel)
    k=kxfunc(wavel)
    return n,k

if __name__ == "__main__":
    #Compute the nominal (unnormalized) particle size distribution
    radii=np.linspace(smin,smax,nsize)
    dr=radii[1]-radii[0]
    diameters=radii*2
    sizedists=radii**(powlaw)
    #Compute the Geometric optical depth of the nominal particle size distribution (Not currently Used)
    taunom=np.sum(sizedists*np.pi*radii**2*dr)
    #Obtain the optical constants at the desired wavelength
    n,k=get_nk(wavel,comp)
    #Compute the reflectance (I/F) of the particle population.
    #Note this uses the same normalization process as described in Appendix B of
    #de Pater et al. 2026 Characterization of the Outer Uranian Rings.... in JGR Planets 131 e2025JE009404.
    #doi.org/10.1029/2025JE00404
    m = complex(n, k) # Compute complex refractive index
    theta1, SL, SR, SU = PyMie.SF_SD(m, wavel*1000, diameters*1000, sizedists,
                                     minAngle=(180-phase), maxAngle=(180-phase),
                                     angularResolution =1.0, space='theta')
    qdict = PyMie.Mie_SD(m, wavel*1000., diameters*1000, sizedists, asDict=True)
    bsca = qdict['Bsca'] * 1.e6
    bext = qdict['Bext'] * 1.e6
    reflectance=np.asarray(SU[0])*(wavel*1000)**2/(4*3.14159)*tau/bext
    print(reflectance)
