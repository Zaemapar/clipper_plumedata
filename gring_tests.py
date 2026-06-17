import concurrent.futures
import itertools
import functools
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

# smin, smax, and powlaw will be determined by the code

s_lowerbound = 1 # Set the bounds for the s model to try
s_upperbound = 15
powlaw_seed = 3 # Seed around which to try different power laws

theta_min = 1
theta_max = 160 # Min and max scattering angles to evaluate at
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

def get_plt_data(src):
    avg_thet = []
    reflectance = []

    with open("../data/gring.csv", "r") as f:
        reader = csv.reader(f)
        next(reader) # Skip header row

        for line in reader:
            # Append scattering angle and reflectance
            avg_thet.append(float(line[1])) # Values are in degrees
            reflectance.append(float(line[3]) * 10**(-6)) # I/F (Dimensionless)

    return avg_thet, reflectance

def RMSE(y_act, y_pred):
    # Root mean square error
    return np.sqrt(np.mean(((y_act)-(y_pred))**2))

def mkifplot(given_theta, reflectances, fit_reflectances, fit_tuple, fit_RMSE):
    # Plot setup - using a linear and log scale plot
    fig, (iflogplt, ifplt) = plt.subplots(1, 2, figsize=(12, 8), layout='constrained')

    # Plot original dataset
    ifplt.plot(given_theta, reflectances, color='black', label='Given Reflectance')
    iflogplt.plot(given_theta, reflectances, color='black', label='Given Reflectance')

    # Plot fit dataset
    ifplt.plot(given_theta, fit_reflectances, color='red', label=f'Computed (smin, smax, powlaw = {fit_tuple[0]:.1f}, {fit_tuple[1]:.1f}, {fit_tuple[2]}; RMSE={fit_RMSE:.2e})')
    iflogplt.plot(given_theta, fit_reflectances, color='red', label=f'Computed (smin, smax, powlaw = {fit_tuple[0]:.1f}, {fit_tuple[1]:.1f}, {fit_tuple[2]}; RMSE={fit_RMSE:.2e})')

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

    graph_dir = utils.create_dirpath("gring_tests", dirs_removed=1)
    # Save to file
    plt.savefig(os.path.join(graph_dir, f"model_gring.png"))

def evaluate_smin_smax(params, wavel, m, nsize, if_interp, theta_min=0, theta_max=180):
    smin, smax, powlaws = params
    
    # Reject invalid size combinations
    if smax < smin:
        return []
        
    # Calculate radii and diameters with nsize steps
    radii = np.linspace(smin, smax, nsize)
    diameters = radii * 2
    
    # Do single-particle Mie calculations once for these diameters
    single_bexts = []
    single_SUs = []
    diameters_nm = diameters * 1000.0 # Mie takes measurements in nm, not microns
    wavel_nm = wavel * 1000.0
    
    # Loop through scattering function for each particle size to get unpolarized
    # light intensities and extinction coefficients
    for d in diameters_nm:
        q = PyMie.MieQ(m, wavel_nm, d, asDict=True)
        bext_single = q['Qext'] * np.pi * ((d/2)**2) * 1e-6
        single_bexts.append(bext_single)
        
        t, sl, sr, su = PyMie.ScatteringFunction(m, wavel_nm, d, minAngle=theta_min, maxAngle=theta_max, angularResolution=1.0, space='theta')
        single_SUs.append(su)
        
    single_bexts = np.array(single_bexts)
    single_SUs = np.array(single_SUs)
    
    # Loop through all power laws
    results = []
    for powlaw in powlaws:
        # Size distriution function for particles
        sizedists = radii**(-1 * powlaw)
        
        # Do the integration step for this powlaw
        # Note that no integration is reqired, only a sum because dr cancels out
        bext = np.sum(single_bexts * sizedists) * 1e6
        SU = np.sum(single_SUs * sizedists[:, np.newaxis], axis=0)
        
        # Calculation of tau using reflectance formula and FIRST unpolarized intensity
        tau = if_interp[0] * bext * 4 * np.pi / (SU[0] * wavel_nm**2)

        # Reversing the formula for ALL intensities and angles
        test_reflectances = np.asarray(SU) * wavel_nm**2 / (4*np.pi) * tau / bext
        
        # Getting the error
        test_RMSE = RMSE(if_interp, test_reflectances)
        
        # Attaching results
        results.append((test_RMSE, tau, test_reflectances, (smin, smax, powlaw)))
        
    return results

if __name__ == "__main__":
    # Obtain the optical constants at the desired wavelength
    n,k=get_nk(wavel,comp)

    # Compute the reflectance (I/F) of the particle population.
    #Note this uses the same normalization process as described in Appendix B of
    #de Pater et al. 2026 Characterization of the Outer Uranian Rings.... in JGR Planets 131 e2025JE009404.
    #doi.org/10.1029/2025JE00404
    m = complex(n, k) # Compute complex refractive index
    
    # Extract the desired angles and observed reflectances from the data table
    given_theta, given_if = get_plt_data("saturnring_scatter.csv")
    thetax_func = sp.interpolate.interp1d(given_theta, given_if)
    max_theta = int(np.floor(np.max(given_theta)))
    min_theta = int(np.ceil(np.min(given_theta)))
    thetas_raw = np.arange(min_theta, max_theta + 1, 1)
    if_interp_raw = thetax_func(thetas_raw)

    # Index the points between min and max thetas
    thetas = thetas_raw[theta_min:theta_max + 1]
    if_interp = if_interp_raw[theta_min:theta_max + 1]

    smins = list(range(s_lowerbound, s_upperbound + 1))
    smaxs = smins
    # Setup values to test for power laws, everywhere around the seed
    powlaws = list(range(max(0, powlaw_seed - 5), powlaw_seed + 6))
    
    # Loop through all combinations of sizes and power laws
    param_combinations = []
    for smin in smins:
        for smax in smaxs:
            if smax >= smin:
                param_combinations.append((smin, smax, powlaws))

    RMSE_min = np.inf
    fit_tuple = (smins[0], smaxs[0], powlaws[0])
    fit_tau = 0
    fit_reflectances = []

    # Bind the fixed arguments to the worker function
    worker_func = functools.partial(
        evaluate_smin_smax, 
        wavel=wavel, 
        m=m, 
        nsize=nsize, 
        if_interp=if_interp, 
        theta_min=theta_min,
        theta_max=theta_max
    )

    print("Starting grid search across {} (smin, smax) pairs...".format(len(param_combinations)))

    # Computes results in parallel, saving compute time
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for batch_results in executor.map(worker_func, param_combinations):
            for result in batch_results:
                test_RMSE, tau, test_refs, params = result
                print(f"Checked smin={params[0]}, smax={params[1]}, powlaw={params[2]}")
                
                # Notes the most accurate result
                if test_RMSE < RMSE_min:
                    RMSE_min = test_RMSE
                    fit_tau = tau
                    fit_reflectances = test_refs
                    fit_tuple = params

    # Make plot & print the tau
    print(f"Calculated tau: {fit_tau}")
    mkifplot(thetas, if_interp, fit_reflectances, fit_tuple, RMSE_min)
