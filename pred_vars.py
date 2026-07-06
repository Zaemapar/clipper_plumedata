import numpy as np

# --- PREDICTION FILE PARAMETERS ---
TAU = 1e-6 # Optical depth
INSTRUMENTS = {'NAC': [0.358, 1.050], 'WAC': [0.370, 1.050], 'MISE': [0.8, 5], 'Europa-UVS': [0.055, 0.206]}
# For each data model, the first index of the 2d array is the size distribution parameters for the semi-empirical Mie theory [smin, smax, powlaw, r, G, x0]
# The second is the Henyey-Greenstein function parameters [w1, g1, w2, g2]
# NOTE: PC Figures are the figures from Pollack & Cuzzi's "Scattering of Nonspherical Particles of 
# Size Comparable to a Wavelength: A New Semi-Empirical Theory and Its Application to Troposhperic Aerosols" (1979)
# For all figures, use 5 um wavelength.
# Use CUBES material for figure 3, OCTAHEDRA for figure 6, and CONCAVE_CONVEX for figure 7
DATA_MODELS = {'G-ring-like': [[4.5357, 8.4725, 2.3550, 1.3, 1.5, 3], [0.46, 0.98, 0.54, 0.34]], 'E-ring-like': [[1.0647, 4.9519, 3.9954, 1.3, 1.5, 3], [0.91, 0.87, 0.09, 0.10]], 'PC Cubes Fig 3': [[1.51197195937, 14.1647899352, 2.5, 1.3, 1.5, 4], [0, 0, 0, 0]], 'PC Octahedrons Fig 6': [[4.67119757975, 7.24154991068, 2.5, 1.1, 2, 8]], 'PC Concave-Convex Fig 7': [[4.69507082121, 14.1647899352, 2.5, 1.3, 4, 10]]}

# Other distribution parameters
NSIZE=51 # Number of sizes to include in Mie distribution (resolution)