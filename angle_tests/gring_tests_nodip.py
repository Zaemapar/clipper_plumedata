#import all the required libraries
#If PyMieScatt not installed add following line:
#!pip install PyMieScatt
#QUESTION: Should we copy PyMieScatt's routine into the workbook as a way to stabilize the code?

import numpy as np
import scipy as sp
import scipy.optimize
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

# Setup parameters, calculated for optimization in gring_tests.py
s_min = 1 # Minimum particle size to scan for
s_max = 10 # Maximum particle size to scan for
powlaw_min = 2 # Min and max power law to scan for
powlaw_max = 4
nsize=51
comp = 'Water Ice'
wavel = 0.647
tau = 1.17e-6
angle_lowerbound = 1
angle_upperbound = 160

# Scan bounds to fit
gmin = 0
gmax = 1
ng = 100
wmin = 0
wmax = 10e-5
nw = 100

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

def get_reflectances(smin, smax, powlaw, m, given_ifs, theta_min=angle_lowerbound, theta_max=angle_upperbound, wavel=wavel, nsize=nsize, tau=None):
    # Compute the nominal (unnormalized) particle size distribution
    radii=np.linspace(smin,smax,nsize)
    #dr=radii[1]-radii[0]
    diameters=radii*2
    sizedists=radii**(-1*powlaw)

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

    # Calculate tau from first available reflectance if not provided
    if tau is None:
        tau = given_ifs[0] * bext * 4 * np.pi / (SU[0] * (wavel*1000)**2)

    # Calculate reflectance using scattering data, converting wavelength to nanometers
    reflectances=np.asarray(SU)*(wavel*1000)**2/(4*np.pi)*tau/bext

    # Clamp all reflectances below the last one
    last_reflectance = reflectances[-1]
    dip_idxs = np.where(reflectances < last_reflectance)[0]
    reflectances[dip_idxs] = [last_reflectance for i in dip_idxs]
    return reflectances, tau

def get_plt_data(src):
    avg_thet = []
    reflectance = []
    scatter = []

    with open("../data/gring.csv", "r") as f:
        reader = csv.reader(f)
        next(reader) # Skip header row

        for line in reader:
            # Append scattering angle and reflectance
            avg_thet.append(float(line[1]))
            reflectance.append(float(line[3]) * 10**(-6))

    return np.asarray(avg_thet), np.asarray(reflectance)

def RMSE(y_act, y_pred):
    # Root mean square error
    return np.sqrt(np.mean((y_act-y_pred)**2))

def mkplot(given_theta, orig_if, model_if, params):
    # Plot setup - using a linear and log scale plot
    fig, (iflogplt, txt) = plt.subplots(1, 2, figsize=(12, 8), layout='constrained')

    # Compute mean absolute values of dataset
    RMSE_plt = RMSE(orig_if, model_if)

    # Plotting the given datasets
    iflogplt.plot(given_theta, orig_if, color='black', label=f'G-Ring Reflectances')
    iflogplt.plot(given_theta, model_if, color='blue', label=f'Computed Mie')

    # Cosmetic setup
    iflogplt.set_xlabel('Scattering Angle (degrees)')
    iflogplt.set_ylabel('I/F Reflectance')
    iflogplt.set_title(f'Reflectance vs Scattering Angle')
    iflogplt.grid()
    iflogplt.set_yscale('log')
    iflogplt.set_xscale('log')
    #bottom, _ = iflogplt.get_ylim()
    #iflogplt.set_ylim(bottom, SU_start*10)

    txt.text(0.25, 0.3, f"Params:\nsmin={params[0]:.4f}\nsmax={params[1]:.4f}\npowlaw={params[2]:.4f}\ntau={params[3]:.2e}", bbox=dict(facecolor='lightblue', alpha=0.5, edgecolor='black', pad=10), fontsize=15)
    txt.text(0.25, 0.6, f"Initial Conditions:\nRMSE={RMSE_plt:.2e}", bbox=dict(facecolor='pink', alpha=0.5, edgecolor='black', pad=10), fontsize=15)

    graph_dir = utils.create_dirpath("gring_tests_nodip", dirs_removed=1)
    # Save to file
    plt.savefig(os.path.join(graph_dir, f"Mie_model_gring.png"))

if __name__ == "__main__":
    # Obtain the optical constants at the desired wavelength
    n,k=get_nk(wavel,comp)

    # Compute the reflectance (I/F) of the particle population.
    #Note this uses the same normalization process as described in Appendix B of
    #de Pater et al. 2026 Characterization of the Outer Uranian Rings.... in JGR Planets 131 e2025JE009404.
    #doi.org/10.1029/2025JE00404
    m = complex(n, k) # Compute complex refractive index

    # Extract the desired angles and observed reflectances from the data table
    given_theta, given_if = get_plt_data("../data/gring.csv")

    # Create an array of valid angles and interpolate given points to it
    max_theta = np.floor(given_theta[-1])
    angle_arr = np.arange(1, max_theta + 1, 1)
    valid_angles = np.arange(angle_lowerbound, angle_upperbound + 1, 1)
    valid_idxs = np.where((angle_arr >= valid_angles[0]) & (angle_arr <= valid_angles[-1]))[0]
    given_interp = sp.interpolate.interp1d(given_theta, given_if)
    ifs_init = given_interp(angle_arr) # Interpolate data to valid angles
    if_interp = ifs_init[valid_idxs]

    # Provide initial guesses and bounds (with slight asymmetry to avoid singular jacobian)
    p0 = [s_min + (s_max-s_min)/3, s_min + (s_max-s_min)/2, powlaw_min + (powlaw_max-powlaw_min)/4]
    bounds = ([s_min, s_min, powlaw_min], 
              [s_max, s_max, powlaw_max])
    
    def mie_iterator(thetas, smin, smax, powlaw):
        if s_max < s_min:
            return None
        reflectances, _ = get_reflectances(smin, smax, powlaw, m, if_interp)
        return np.log10(reflectances)

    try:
        # Fit in log space so the tail isn't ignored
        popt, _ = sp.optimize.curve_fit(
            mie_iterator, 
            valid_angles, 
            np.log10(if_interp), 
            p0=p0, 
            bounds=bounds,
            maxfev=10000,
            diff_step=0.01,
            x_scale=[s_max, s_max, powlaw_max] # help the optimizer understand the scale
        )
        params = popt.tolist()
    except Exception as e:
        print(f"Curve fitting failed: {e}")
        params = p0

    # Retrieve the tau corresponding to the fitted parameters
    _, optimal_tau = get_reflectances(params[0], params[1], params[2], m, if_interp, theta_min=angle_lowerbound, theta_max=angle_upperbound)
    params.append(optimal_tau) # append to parameter list

    plt_reflectances, _ = get_reflectances(params[0], params[1], params[2], m, if_interp, theta_min=angle_arr[0], theta_max=angle_arr[-1], tau=optimal_tau)

    mkplot(angle_arr, ifs_init, plt_reflectances, params)
