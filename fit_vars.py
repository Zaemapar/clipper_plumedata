import os

# --- FIT FILE PARAMETERS ---
# Simulation params
DATA_FILE = "ering.csv"
MODEL = "mie" # Set to either mie (for Mie scattering) or hg (for Henyey-Greenstein function)
ALTITUDE = None # Set to an altitude value in km if you want to curve-fit to altitude-dependent reflectance-vs-wavelength data. Not functional currently.

# Angle scan bounds. For use in fitting only, final plots will plot over all available theta
ANGLE_LOWERBOUND = 1
ANGLE_UPPERBOUND = 166

# MIE DEFAULT PARAMETERS
# Min and max particle sizes to scan for in Mie distribution
S_MIN = 1
S_MAX = 10
# Min and max power law to scan for in Mie distribution
POWLAW_MIN = 2
POWLAW_MAX = 4
NSIZE=51 # Number of sizes to include in distribution (resolution)
COMP = 'Water Ice' # Composition of material. Currently only this is supported.
WAVEL = 0.647 # Given reading wavelength of instrument
WAVEL_SNAPSHOT_ANGLE = 20 # Once size distribution is determined at WAVEL, angle will be fixed to this and wavelength varied to examine effects

# WAVELENGTH PLOT PARAMETERS. FOR MIE TESTS ONLY, AFTER APPROPRIATE PARAMETERS HAVE BEEN COMPUTED.
MIN_WAVEL_TEST = 0.05
MAX_WAVEL_TEST = 5
WAVEL_STEP = 0.05

# HENYEY-GREENSTEIN PARAMETER BOUNDS
# Gs are the asymmetry parameters
GMIN = 0
GMAX = 1
# Ws are the term weights
WMIN = 0
WMAX = 10e-5

# Path to data file
DATA_PATH = os.path.join("data", DATA_FILE)