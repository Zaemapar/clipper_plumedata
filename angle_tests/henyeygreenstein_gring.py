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

angle_lowerbound = 1
angle_upperbound = 180

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

def henyey_greenstein(angle, gweights):
    hg_term = 0   
    
    # Loop over gweights (each contains info about a term of the function)
    for term in gweights:
        g = term[0]
        w = term[1]
        hg_term += w/(4*np.pi)*(1-g**2)/(1+g**2-2*g*np.cos(angle))**1.5
    return hg_term

def RMSE(y_act, y_pred):
    # Root mean square error
    return np.sqrt(np.mean((y_act-y_pred)**2))

def mksuplot(given_theta, SU, SU_HG, hgparams):
    # Plot setup - using a linear and log scale plot
    fig, (iflogplt, txt) = plt.subplots(1, 2, figsize=(12, 8), layout='constrained')

    # Compute mean absolute values of dataset
    RMSE_plt = RMSE(SU, SU_HG)

    SU_start = SU_HG[1]

    # Plotting the given datasets
    iflogplt.plot(given_theta, SU, color='black', label=f'G-Ring Reflectance)')
    iflogplt.plot(given_theta, SU_HG, color='blue', label=f'Computed Henyey-Greenstein')

    # Cosmetic setup
    iflogplt.set_xlabel('Scattering Angle (degrees)')
    iflogplt.set_ylabel('Reflectance (I/F)')
    iflogplt.set_title(f'G-Ring Reflectance vs Scattering Angle')
    iflogplt.grid()
    iflogplt.set_yscale('log')
    iflogplt.set_xscale('log')
    bottom, _ = iflogplt.get_ylim()
    iflogplt.set_ylim(bottom, SU_start*10)

    txt.text(0.25, 0.3, f"HG Params:\nw1={hgparams[0]:.2f}\ng1={hgparams[1]:.2f}\nw2={hgparams[2]:.2f}\ng2={hgparams[3]:.2f}\nwscale={hgparams[4]:.2e}", bbox=dict(facecolor='lightblue', alpha=0.5, edgecolor='black', pad=10), fontsize=15)
    txt.text(0.25, 0.6, f"Initial Conditions:\nRMSE={RMSE_plt:.2e}", bbox=dict(facecolor='pink', alpha=0.5, edgecolor='black', pad=10), fontsize=15)

    # Save to file
    plt.savefig(os.path.join(graph_dir, f"gring_hgfit.png"))

if __name__ == "__main__":
    # Define the path for the output directory
    graph_dir = utils.create_dirpath("henyeygreenstein_gring", dirs_removed=1)

    # Create the directory to contain the plots
    try:
        os.makedirs(graph_dir, exist_ok=True)
    except OSError as error:
        print(f"Error creating directory: {error}") # Error handling

    def hg_model(angle, w1, g1, w2, g2):
        return np.log10(henyey_greenstein(angle, [[g1, w1], [g2, w2]]))

    # Compute the unpolarized light scattering intensity, extintion coefficient, angles, and reflectances based on calculated particle distribution and tau
    theta1_orig, reflectances_orig = get_plt_data("saturnring_scatter.csv")
    theta_max = np.floor(theta1_orig[-1])
    theta1 = np.arange(1, theta_max + 1, 1)
    orig_interp = sp.interpolate.interp1d(theta1_orig, reflectances_orig)
    reflectances = orig_interp(theta1)

    # Get a list of valid angles to test, within angle bounds
    valid_tests = np.where((theta1 <= angle_upperbound) & (theta1 >= angle_lowerbound))[0]
    
    # Convert theta to radians for henyey_greenstein
    theta1_rad = theta1 * np.pi / 180
    valid_angles = theta1_rad[valid_tests]
    valid_reflectances = reflectances[valid_tests]

    # Provide initial guesses and bounds (with slight asymmetry to avoid singular jacobian)
    p0 = [wmax/2, 0.9, wmax/3, 0.5]
    bounds = ([wmin+1e-10, gmin, wmin+1e-10, gmin], 
                [wmax, gmax, wmax, gmax])
        
    try:
        # Fit in log space so the tail isn't ignored
        popt, _ = sp.optimize.curve_fit(
            hg_model, 
            valid_angles, 
            np.log10(valid_reflectances), 
            p0=p0, 
            bounds=bounds,
            maxfev=10000,
            x_scale=[wmax, gmax, wmax, gmax] # help the optimizer understand the scale
        )
        hgparams = popt.tolist()
    except Exception as e:
        print(f"Curve fitting failed: {e}")
        hgparams = p0
    
    # Compute full array of fitted values
    reflectance_HG = henyey_greenstein(theta1_rad, [[hgparams[1], hgparams[0]], [hgparams[3], hgparams[2]]])

    # Normalize weights
    w1 = hgparams[0]
    w2 = hgparams[2]
    scale_factor = w1 + w2

    hgparams[0] = w1 / scale_factor
    hgparams[2] = w2 / scale_factor
    hgparams.append(scale_factor)

    min_RMSE = RMSE(valid_reflectances, reflectance_HG[valid_tests])

    print(f"HG Params: {hgparams}")
    print(f"RMSE ({angle_lowerbound}-{angle_upperbound}): {min_RMSE}")

    # Make plot
    mksuplot(theta1, reflectances, reflectance_HG, hgparams)
