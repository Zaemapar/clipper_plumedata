# --- PREDICTION FILE PARAMETERS ---
TAU = 1e-6 # Optical depth
INSTRUMENTS = {'NAC': [0.358, 1.050], 'WAC': [0.370, 1.050], 'MISE': [0.8, 5], 'Europa-UVS': [0.055, 0.206], 'E-THERMIS': [7, 80]}
COMP = 'Water Ice' # Composition of material. Currently only this is supported.
TYPE = "ering-like" # Either "gring-like" or "ering-like"

# Angle scan bounds
ANGLE_LOWERBOUND = 1
ANGLE_UPPERBOUND = 160
WAVEL_SNAPSHOT_ANGLE = 20 # Once size distribution is determined at WAVEL, angle will be fixed to this and wavelength varied to examine effects

# Other distribution parameters
NSIZE=51 # Number of sizes to include in distribution (resolution)