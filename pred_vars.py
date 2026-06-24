# --- PREDICTION FILE PARAMETERS ---
TAU = 1e-6 # Optical depth
INSTRUMENTS = {'NAC': [0.358, 1.050], 'WAC': [0.370, 1.050], 'MISE': [0.8, 5], 'Europa-UVS': [0.055, 0.206]}
DATA_MODELS = ['G-ring-like', 'E-ring-like']
COMP = 'Water Ice'

# Other distribution parameters
NSIZE=51 # Number of sizes to include in Mie distribution (resolution)
# Arrays of parameters for Mie and Henyey-Greenstein functions for different data models.
# 1st is G-ring-like, 2nd is E-ring-like
DISTTYPE_ARR = [[4.5357, 8.4725, 2.3550],
                [1.0647, 4.9519, 3.9954]]
HG_PARAMS_ARR = [[0.46, 0.98, 0.54, 0.34], # w1, g1, w2, g2
                 [0.91, 0.87, 0.09, 0.10]]