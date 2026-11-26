"""
A script to set up the main GUI for plotting plume spectra dynamically against various initial conditions. Takes
input size distribution, instrument wavelength range, optical depth, and material composition, among other 
parameters, and creates plots for each combination of parameters provided, allowing for multiple plots on the 
same window and switching between plotting reflectance vs. wavelength, surface albedo vs. wavelength, and 
reflectance vs. angle (phase or scattering). Also allows for saving plots and metadata and loading external data,
provided it is formatted correctly.

Throughout this program, references to 'PC80' refer to Pollack & Cuzzi's "Scattering of Nonspherical Particles of 
Size Comparable to a Wavelength: A New Semi-Empirical Theory and Its Application to Troposhperic Aerosols" (1979).
References to 'HG' refer to Henyey-Greenstein functions.

To clarify in case of confusion, the code labels the graph mode button as "Reflectance vs. <x-axis units>" for 
simplicity; however, there are options to change the y-axis output to something other than reflectance.

Authors: Matthew M. Hedman, Parker A. Zaemann
Date: 25 November 2026
Source: https://github.com/Zaemapar/clipper_plumedata
Contact: mhedman@uidaho.edu
"""

import sys
import pyqtgraph as pg
import pyqtgraph.exporters
import PyQt5.QtWidgets as qt
import PyQt5.QtCore as qc
import pred_vars as vars
import utils
import os
import csv
import copy
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

COLOR_ARR = ['red', 'magenta', 'orange', 'green', 'blue', 'violet'] # Colors for new graphs, cycled through when a plot is made permanent
COMPS = utils.get_available_materials() # Call function to parse data directory and get all available materials to display
# Create a style sheet to use for every single input box. This is so that if the gui happens to be in dark mode, the text will render as white.
BKSHEET = """
            QComboBox {
                color: black; /* Color of the selected/displayed text */
            }
            QComboBox QAbstractItemView {
                color: black; /* Color of the text in the dropdown list */
                selection-background-color: lightgray;
                selection-color: black;
            }
          """

def output_graph(graphmode, sensor, comps, mixmodel, datamodel, fitmodel, param, tau, nsizes, wavelbounds, output, result_queue):
    """
    Output x and y arrays for a plot given various input parameters and spectra-related x and y axis units

    :param graphmode: Integer, indicates whether to output reflectance vs. wavelength (0), surface albedo
                    vs. wavelength (1), or reflectance vs. angle (2)
    :param sensor: String sensor from which to read wavelength range
    :param comps: Dictionary with keys as materials and values as volume fractions of materials
    :param mixmodel: String mixture model to use, either 'Areal' or 'Molecular'
    :param datamodel: List of size distribution parameters [float min particle size, float max particle size, float
                      power law, float r, float G, float x0]
    :param fitmodel: String for scattering theory to use, either 'Semi-Empirical Mie', 'Pure Mie',
                    'Mie Diffraction', 'Mie External Reflection', or 'Mie Transmission'.
    :param param: Primary fixed parameter for the given graphmode. Float scattering or phase angle in degrees 
                  if graphmode = 0, float effective grain size in microns if graphmode = 1, float wavelength in 
                  microns if graphmode = 2
    :param tau: Float optical depth (unitless)
    :param nsizes: List [string size mode, integer number of sizes to use in size distribution]
    :param wavelbounds: Tuple (float, float) wavelength bounds in microns as established by the composition, 
                        representing data available for index of refraction interpolation
    :param output: List of strings indicating the type of output to return. First element is x axis, either
                   'Phase Angle', 'Scattering Angle', or 'Wavelength'. Second element is y axis, either 
                   'Reflectance', 'Phase Function', 'Intensity', or 'Albedo'.
    :returns: List containing float wavelength/angle data values (microns/radians/degrees)
              List containing float reflectance/albedo/phase function data values (unitless)
    """

    # --- PREPARE VARIABLES ---
    # Try to extract r, G, and tau from the datamodel array; if they are NaNs, set them to their default values
    # NaN means the other parameters are configured in such a way that they will not be used.
    # So we set to a default to prevent errors.
    if datamodel[3] == 'NaN':
        r = 1 # R is a ratio and should be 1 by default
    else:
        r = float(datamodel[3])

    if datamodel[4] == 'NaN':
        G = 1
    else:
        G = float(datamodel[4])

    if tau == 'NaN':
        tau = 0
    else:
        tau = float(tau)
    
    if fitmodel == 'Semi-Empirical Mie':
        x0 = float(datamodel[5])
    elif fitmodel == 'Pure Mie':
        x0 = np.inf # Pure Mie treats all particles as small, using Mie theory for perfect spheres
    else:
        x0 = 0 # All other components of Mie treat all particles as large to use only the selected components

    # nsizes also needs to be checked for nans as it only applies to semi-empirical Mie, pure Mie, and Mie diffraction
    if nsizes == ['NaN', 'NaN']:
        nsizes = vars.NSIZE

    # --- GRAPH MODE 0: REFLECTANCE VS WAVELENGTH ---
    # If reflectances vs. wavelengths mode, compare the wavelength range of the sensor with the wavelength range of the material and use the most limiting bounds as the x array
    if graphmode == 0:
        wavel_range = vars.INSTRUMENTS[sensor] # Range of all available wavelengths in the sensor
        min_bound = float(round(max(wavelbounds[0], wavel_range[0]), 5)) # Round most limiting wavelength bounds to 5 decimals to clamp floating-point precision loss
        max_bound = float(round(min(wavelbounds[1], wavel_range[1]), 5))
        xarr = np.arange(min_bound, max_bound + 0.01, 0.01) # Assume a 10 nm resolution
        if xarr[-1] > max_bound:
            xarr[-1] = max_bound # Clamp the last element if it goes out of bounds
        
        # Use the old model from PC80 if requested
        if 'PC80' in fitmodel:
            reflectances = utils.angle_mie_reflectances_old(float(datamodel[0]), float(datamodel[1]), float(datamodel[2]), r=r, G=G, x0=x0, theta_min=param, theta_max=param, wavels=xarr, comps=comps, mixmodel=mixmodel, nsize=nsizes, tau=tau, section=fitmodel[:-4], output=output[1], updates=[True, result_queue])[0]
        else:
            reflectances = utils.angle_mie_reflectances(float(datamodel[0]), float(datamodel[1]), float(datamodel[2]), r=r, G=G, x0=x0, theta_min=param, theta_max=param, wavels=xarr, comps=comps, mixmodel=mixmodel, nsize=nsizes, tau=tau, section=fitmodel, output=output[1], updates=[True, result_queue])[0]

        # angle_mie_reflectances returns an array of shape [wavels, angles]
        # Because input angle is fixed, reflectances is a column array with shape [[element1], [element2], [element3]]
        # Here we flatten the array to remove the extra dimension
        reflectances = [element[0] for element in reflectances]

    # --- GRAPH MODE 1: SURFACE ALBEDO ---
    # If surface albedo mode, compare the wavelength range of the sensor with the wavelength range of the material and use the most limiting bounds as the x array
    elif graphmode == 1:
        wavel_range = vars.INSTRUMENTS[sensor] # Range of all available wavelengths in the sensor
        min_bound = round(max(wavelbounds[0], wavel_range[0]), 5) # Round most limiting wavelength bounds to 5 decimals to clamp floating-point precision loss
        max_bound = round(min(wavelbounds[1], wavel_range[1]), 5)
        xarr = np.arange(min_bound, max_bound + 1e-10, 0.01) # Assume a 10 nm resolution
        if xarr[-1] > max_bound:
            xarr = xarr[:len(xarr) - 1] # Clamp the last element if it goes out of bounds
        reflectances = utils.fresn_surface_reflectances(xarr, param, comps=comps, mixmodel=mixmodel, updates=[True, result_queue])
    
    # --- GRAPH MODE 2: REFLECTANCE VS ANGLE ---
    # If reflectance vs. angle mode, create an array of all colatitude angles and use that as x array
    elif graphmode == 2:
        xarr = np.arange(0, 181, 1) if output[0] == 'Scattering Angle' else np.arange(180, -1, -1) # Create list of angles from 0 to 180 degrees if scattering, or 180 to 0 degrees if phase
       
        # Use the old PC80 model if requested
        # Note that the angle_mie_reflectances function does not use xarr directly; it takes only the first and last values. By default, it steps the angles by 1 degree, making its computations identical to those here.
        if 'PC80' in fitmodel:
            reflectances = utils.angle_mie_reflectances_old(float(datamodel[0]), float(datamodel[1]), float(datamodel[2]), r=r, G=G, x0=x0, theta_min=0, theta_max=180, wavels=[param], comps=comps, mixmodel=mixmodel, nsize=nsizes, tau=tau, section=fitmodel, output=output[1], updates=[True, result_queue])[0][0]
        else:
            reflectances = utils.angle_mie_reflectances(float(datamodel[0]), float(datamodel[1]), float(datamodel[2]), r=r, G=G, x0=x0, theta_min=0, theta_max=180, wavels=[param], comps=comps, mixmodel=mixmodel, nsize=nsizes, tau=tau, section=fitmodel, output=output[1], updates=[True, result_queue])[0][0]

        # For Henyey-Greenstein, curve fitting is needed to determine global scale factor
        if fitmodel == 'Henyey-Greenstein':
            # Iterate through preset size distributions and see if input size distribution matches.
            # If so, get the HG parameters for that size distribution
            init_hg_params = []
            for model in vars.SIZEDISTS.values():
                if datamodel == model[0]:
                    init_hg_params = model[1]
                    break
            
            # Define a function to use in curve-fitting HG function to match Mie via scale factor
            def hg_iterator(angles, scale_factor):
                hg_ifs = utils.henyey_greenstein(xarr, [[init_hg_params[1], init_hg_params[0] * scale_factor], [init_hg_params[3], init_hg_params[2] * scale_factor]])
                return np.log10(hg_ifs) # Will be fit on a log scale

            # Try to best-fit the Henyey-Greenstein function to the Mie function via a scale factor
            try:
                popt, _ = sp.optimize.curve_fit(
                    hg_iterator, 
                    xarr, 
                    np.log10(reflectances), # Log of original dataset is being fit with log of best-fit dataset
                    p0=[1e-6], # Standard scale for HG weights
                    maxfev=10000,
                )
                factor = popt.tolist()[0]
            except Exception as e:
                print(f"Curve fitting failed: {e}")
                factor = 1e-6

            # Final Henyey-Greenstein reflectances
            reflectances = utils.henyey_greenstein(np.radians(xarr), [[init_hg_params[1], init_hg_params[0] * factor], [init_hg_params[3], init_hg_params[2] * factor]])

    # --- CLEANUP AND RETURN ---
    # Drain any remaining progress items in the queue
    try:
        while not result_queue.empty():
            item = result_queue.get_nowait()
    except Exception:
        pass # The queue is already probably empty

    # Return final arrays by putting them in queue
    result_queue.put({
        "x_data": xarr.tolist(),
        "y_data": reflectances
    })

class MainWindow(qt.QMainWindow):
    """
    Sets up the main GUI for plotting plume particle scattering data, containing widgets to modify core parameters,
    adjust graph settings, and dynamically update results. Used by analysis_main.py in startup. Extends PyQt5's
    QMainWindow.
    """

    def __init__(self):
        # Initialize the parent class (QMainWindow)
        super().__init__()

        # --- MAIN WINDOW PROPERTIES ---
        # Title and sizing
        self.setWindowTitle("Europa Clipper Scattering Prediction Application")
        self.resize(1500, 1100)

        # Create the data input screen widget
        self.data_input_widget = qt.QWidget()
        
        # Use a Vertical Box Layout (QVBoxLayout) to stack widgets top-to-bottom
        self.layout = qt.QGridLayout()
        self.layout.setContentsMargins(30, 30, 30, 30) # Making sure things don't overlap each other
        self.layout.setSpacing(20)
        self.data_input_widget.setLayout(self.layout)

        # --- WIDGETS ---
        # Push button to toggle between reflectance vs wavelength graphs, surface albedo vs wavelength graphs, and reflectance vs angle graphs
        # Indicates what type of graph is being shown. 0 is reflectance vs wavelength, 1 is surface albedo vs wavelength, 2 is reflectance vs angle
        self.graphmode = 0
        self.graph_mode_button = qt.QPushButton("Reflectance vs. Wavelength") # Note that by default, a wavelength spectrum is shown. The other input fields are initialized accordingly.

        # Dropdown list for instruments on Europa Clipper
        self.instrument_label = qt.QLabel("Instrument:")
        self.instrument = qt.QComboBox()
        self.instrument.addItems(vars.INSTRUMENTS.keys()) # Add all available instruments from variables file
        self.instrument.setStyleSheet(BKSHEET)

        # Dropdown list for available material compositions
        # Labels form the header row for the materials table if multiple are added
        self.comp_labels = [qt.QLabel("Composition:"), qt.QLabel("Volume Fraction (dim):")]
        self.mat_table = None # A table will be added if we add multiple materials to the composition
        self.mixture_model = None # A mixture model toggle list will be added along with the table
        # Each material in a composition will have a dropdown list to specify the material,
        # plus an entry field for the corresponding volume fraction. An entry will be added to this array for each new material added.
        self.comps = [[qt.QComboBox(), qt.QLineEdit()]]
        self.comps[0][0].addItems(COMPS) # Each box allows for selection of a new material
        self.comps[0][0].setStyleSheet(BKSHEET)
        self.comps[0][1].setText('1.0') # The default volume fraction (not shown) is 1.0, since there is only one material
        self.comps[0][1].setPlaceholderText("Input volume fraction") # In case the user clears the box
        # Determining the minimum and maximum bounds for wavelength based on available material data
        # Instrument wavelength ranges will be far more limiting, but good to include this nonetheless
        self.comp_min_wavel, self.comp_max_wavel = utils.get_minmax_wavel(self.comps[0][0].currentText())

        # Dropdown list for available preset size distributions (G-ring, E-ring, etc)
        self.dist_label = qt.QLabel("Size Distribution:")
        self.dist_model = qt.QComboBox()
        self.dist_model.addItems(vars.SIZEDISTS.keys())
        self.dist_model.setStyleSheet(BKSHEET)
        self.dist_model.addItems(['Custom'])
        self.dist_model.setVisible(True) # Redundant, but needed for checks. Apparently this isn't true by default.
        # These objects display when the 'Custom' option for size distributions is selected; essential for Mie computations.
        self.smin_label = qt.QLabel("Min size (μm):")
        self.smin_label.setVisible(False)
        self.smin_box = qt.QLineEdit()
        self.smin_box.setPlaceholderText("Input min size in microns")
        self.smin_box.setStyleSheet("border: 1px solid black;")
        self.smin_box.setVisible(False)
        self.smax_label = qt.QLabel("Max size (μm):")
        self.smax_label.setVisible(False)
        self.smax_box = qt.QLineEdit()
        self.smax_box.setPlaceholderText("Input max size in microns")
        self.smax_box.setStyleSheet("border: 1px solid black;")
        self.smax_box.setVisible(False)
        self.powlaw_label = qt.QLabel("Power Law (dim):")
        self.powlaw_label.setVisible(False)
        self.powlaw_box = qt.QLineEdit()
        self.powlaw_box.setPlaceholderText("Input power law")
        self.powlaw_box.setStyleSheet("border: 1px solid black;")
        self.powlaw_box.setVisible(False)

        # These objects conditionally display when the custom size distribution is selected depending on the scattering theory; necessary for the semi-empirical theory and some variants.
        self.G_label = qt.QLabel("G (dim):")
        self.G_label.setVisible(False)
        self.G_box = qt.QLineEdit()
        self.G_box.setPlaceholderText("Input G parameter")
        self.G_box.setStyleSheet("border: 1px solid black;")
        self.G_box.setVisible(False)
        self.r_label = qt.QLabel("r (dim):")
        self.r_label.setVisible(False)
        self.r_box = qt.QLineEdit()
        self.r_box.setPlaceholderText("Input r parameter")
        self.r_box.setStyleSheet("border: 1px solid black;")
        self.r_box.setVisible(False)
        self.x0_label = qt.QLabel("x0 (dim):")
        self.x0_label.setVisible(False)
        self.x0_box = qt.QLineEdit()
        self.x0_box.setPlaceholderText("Input size parameter")
        self.x0_box.setStyleSheet("border: 1px solid black;")
        self.x0_box.setVisible(False)

        # Dropdown list to select which scattering theory to use in graphs
        self.fit_label = qt.QLabel("Scattering Theory:")
        self.fit_model = qt.QComboBox()
        # By default, semi-empirical Mie (with modifications from the original PC80 methods) is used. The others are to get different components of the Mie function.
        # The original PC80 semi-empirical model and its components are also listed
        # Henyey-Greenstein is omitted initially since it's not available in reflectance vs. wavelength plots
        self.fit_model.addItems(["Semi-Empirical Mie", "Pure Mie", "Mie Diffraction", "Mie External Reflection", "Mie Transmission", "Semi-Empirical Mie (PC80)", "Mie Diffraction (PC80)", "Mie External Reflection (PC80)", "Mie Transmission (PC80)"])
        self.fit_model.setStyleSheet(BKSHEET)
        self.fit_model.setVisible(True) # Redundant, but needed for checks

        # Button to change between input window and resolution settings window
        self.window_button = qt.QPushButton("Open Resolution Settings")
        self.window_button.setVisible(True) # Redundant, but needed for checks

        # Entry box for manual optical depth input. Will only be shown if the y-axis of the graph is 'Reflectance'
        self.tau_label = qt.QLabel("Optical Depth (dim):")
        self.tau_box = qt.QLineEdit()
        self.tau_box.setPlaceholderText("Input optical depth")
        self.tau_box.setText("1e-6") # Standard order of magnitude for optical depth
        self.tau_box.setVisible(True) # Redundant, but needed for checks

        # Entry box for primary varying parameter, either angle (phase or scattering, for ref vs. wavel), effective grain size (surface albedo vs. wavelength), or wavelength (for ref vs. theta)
        self.param_str = '' # Variable to hold string value of this parameter
        # If an angle is input, it will always have two options: scattering or phase. Angle is an input by default.
        self.input_label = qt.QComboBox()
        self.input_label.addItems(["Scattering Angle (°):", "Phase Angle (°):"])
        self.input_label.setStyleSheet(BKSHEET)
        self.input_box = qt.QLineEdit()
        self.input_box.setPlaceholderText("Input scattering angle in degrees") # Switching graphs clears this box, so this needs to display
        self.input_box.setText("90") # By default, 90 degrees scattering angle is selected

        # X axis dropdown box, can switch between plotting phase angle and scattering angle
        self.x_axis_label = qt.QLabel("X Axis:")
        self.x_axis = qt.QComboBox()
        self.x_axis.addItems(["Wavelength"])
        self.x_axis.setStyleSheet(BKSHEET)

        # Y axis dropdown box, can switch between plotting reflectance and phase function (as used in Pollack & Cuzzi, 1980)
        self.y_axis_label = qt.QLabel("Y Axis:")
        self.y_axis = qt.QComboBox()
        self.y_axis.addItems(["Reflectance", "Phase Function"]) # Intensity is not supported for semi-empirical Mie theory
        self.y_axis.setStyleSheet(BKSHEET)

        # Set up radio buttons for X and Y scales, change between linear and logarithmic
        self.x_scale_label = qt.QLabel("X Scale:")
        self.x_linear = qt.QRadioButton("Linear")
        self.x_linear.setChecked(True) # Linear by default
        self.x_log = qt.QRadioButton("Logarithmic")
        self.x_scale = qt.QButtonGroup() # Setting up a group to ensure only one can be checked
        self.x_scale.addButton(self.x_linear)
        self.x_scale.addButton(self.x_log)

        self.y_scale_label = qt.QLabel("Y Scale:")
        self.y_linear = qt.QRadioButton("Linear")
        self.y_linear.setChecked(True) # Linear by default
        self.y_log = qt.QRadioButton("Logarithmic")
        self.y_scale = qt.QButtonGroup() # Setting up a group to ensure only one can be checked
        self.y_scale.addButton(self.y_linear)
        self.y_scale.addButton(self.y_log)

        # Set up a pushbutton to save the current plot on the graph and start a new plot in the same window
        self.add_graph_button = qt.QPushButton("Add new plot")
        # Set up a pushbutton to add materials
        self.add_material_button = qt.QPushButton("Add new material (mixture)")
        # Set up a pushbutton to clear the whole plot
        self.clear_plot_button = qt.QPushButton("Clear plots")
        # Set up a pushbutton to save the plot to file
        self.save_plot_button = qt.QPushButton("Save plot")
        # Set up a pushbutton to load outside data
        self.load_button = qt.QPushButton("Load new data")

        # Set up a label at the bottom of the screen giving status updates/error messages
        self.load_label = qt.QLabel("Up to date")
        self.load_label.setStyleSheet("color: red;")
        # The load label will handle errors, as such it needs to wrap since it can get pretty long
        self.load_label.setWordWrap(True)

        # Set up a progress bar. Won't always be visible, but for long computations, it is handy
        self.prog = qt.QProgressBar(self)
        self.prog.setGeometry(0, 0, 300, 25)
        self.prog.setMaximum(100) # Set maximum value
        self.prog.setVisible(False) # Won't be shown by default (when not loading)

        # Set up a cancel button to stop the graph from loading. Only visible when the progress bar is visible
        self.cancel = qt.QPushButton("Cancel")
        self.cancel.setVisible(False) # Won't be shown by default

        # Set up the graph window
        self.multiple = 0 # Multiple will be used to indicate if multiple datasets are in the same graph and what the index of the latest one is. It's like a current plot index
        self.loaded_idxs = {} # Dictionary to hold indexes of plots that were loaded via load data
        self.last_graph_loaded = False # Boolean, to determine whether or not the most recently plotted graph was a loaded graph
        self.x_data = [] # Arrays to hold the current data
        self.y_data = []
        grid_pen = pg.mkPen(color='k', width=1, style=pg.QtCore.Qt.PenStyle.SolidLine) # Pen to use on axes
        self.graph_widget = pg.PlotWidget()
        self.graph_widget.setBackground('w') # Set background to white
        self.graph_widget.showGrid(x=True, y=True) # Grid is helpful to identify points on displayed graphs
        # Setup axes for default reflectance vs. wavelength plot
        self.graph_widget.setLabel('left', self.y_axis.currentText(), units='I/F', color='k')
        self.graph_widget.setLabel('bottom', 'Wavelength', units='μm', color='k')
        # Disable axis autoscale to prevent misleading prefix additions based on zoom
        # Axis displays numbers in scientific notation automatically for clarity
        self.graph_widget.getAxis('bottom').enableAutoSIPrefix(False)
        self.graph_widget.getAxis('left').enableAutoSIPrefix(False)
        # Pen ensures axes and numbers are black and thick
        self.graph_widget.getAxis('bottom').setPen(grid_pen)
        self.graph_widget.getAxis('left').setPen(grid_pen)
        self.graph_widget.getAxis('bottom').setTextPen(grid_pen)
        self.graph_widget.getAxis('left').setTextPen(grid_pen)
        self.legend = None # Declare a legend object, will be populated when multiple > 0

        # Set up a pen to plot the window
        self.last_color_idx = 0 # Color from which to start. Will count up in multiple plotting
        self.plt_pen = pg.mkPen(color=COLOR_ARR[self.last_color_idx], width=3)
        self.ifs = self.graph_widget.plot(self.x_data, self.y_data, pen=self.plt_pen) # Plot a blank dataset. Modifying self.x_data or self.y_data will modify this plot.

        # Set up the 'home zoom' button to reset the zoom
        self.home_button = qt.QPushButton("Home")

        # Set up the min and max bounds for x and y, shown in the top right corner of the graph
        # Allows user to set graph bounds manually
        self.xmin_label = qt.QLabel("Min X:")
        self.xmin_label.setStyleSheet("color: black;") # We have to set the text to black in these style sheets because Mac will sometimes put the GUI in dark mode and attempt to switch the text to white, where it will be invisible
        self.xmin_box = qt.QLineEdit()
        self.xmin_box.setPlaceholderText("Input min x")
        self.xmin_box.setMaximumWidth(50) # Makes sure input boxes are wide enough to display whatever text is entered in (placeholder text crops off and only shows the end, sorry, don't know how to fix that)
        self.xmax_label = qt.QLabel("Max X:")
        self.xmax_label.setStyleSheet("color: black;")
        self.xmax_box = qt.QLineEdit()
        self.xmax_box.setPlaceholderText("Input max x")
        self.xmax_box.setMaximumWidth(50)
        self.ymin_label = qt.QLabel("Min Y:")
        self.ymin_label.setStyleSheet("color: black;")
        self.ymin_box = qt.QLineEdit()
        self.ymin_box.setPlaceholderText("Input min y")
        self.ymin_box.setMaximumWidth(50)
        self.ymax_label = qt.QLabel("Max Y:")
        self.ymax_label.setStyleSheet("color: black;")
        self.ymax_box = qt.QLineEdit()
        self.ymax_box.setPlaceholderText("Input max y")
        self.ymax_box.setMaximumWidth(50)

        # Creating a stack widget so we can cycle between resolution settings window and main window
        self.stack = qt.QStackedWidget()

        # Set up the settings screen for modifying graph resolution. This is purely for performance.
        self.settings_widget = qt.QWidget()
        # Use a Vertical Box Layout (QVBoxLayout) to stack widgets top-to-bottom
        self.settings_layout = qt.QGridLayout()
        self.settings_layout.setContentsMargins(30, 30, 30, 30) # Making sure things don't overlap each other
        self.settings_layout.setSpacing(20)
        self.settings_widget.setLayout(self.settings_layout)

        # The following widgets will be added to this screen
        # Back button to go back to data input screen
        self.back_button = qt.QPushButton("Back")

        # Dropdown box to use in selecting linear-distributed or log-distributed sizes
        self.spacing_label = qt.QLabel("Spacing Mode:")
        self.spacing = qt.QComboBox()
        self.spacing.addItems(["Linear", "Logarithmic"])
        self.spacing.setCurrentText(vars.NSIZE[0]) # Set to the default spacing from vars
        self.spacing.setStyleSheet(BKSHEET)

        # Slider to use in selecting size distribution size for Semi-Empirical Mie, Pure Mie, and Mie Diffraction scattering theories
        self.dist_slider = qt.QSlider(qc.Qt.Orientation.Horizontal)
        self.dist_slider.setMinimum(2)
        self.dist_slider.setMaximum(100)
        self.dist_slider.setValue(vars.NSIZE[1])
        self.dist_slider.setMinimumWidth(210)
        self.slider_label = qt.QLabel(f"Number of Sizes: {self.dist_slider.value()}")
        self.slider_label.setMinimumWidth(125)

        # Add these widgets to the layout
        self.settings_layout.addWidget(self.back_button, 0, 0, 1, 3)
        self.settings_layout.addWidget(self.spacing_label, 1, 0, 1, 1)
        self.settings_layout.addWidget(self.spacing, 1, 1, 1, 2)
        self.settings_layout.addWidget(self.slider_label, 2, 0, 1, 1)
        self.settings_layout.addWidget(self.dist_slider, 2, 1, 1, 2)

        # --- BASE PARAMETER SETUP ---
        # Base parameters hold initial conditions of each plot on the graph, used for tracking changes
        # If ANY of these values are not modifiable by the user in the GUI (i.e. not displayed), they
        # will be set to 'NaN' or None.
        self.base_params = [[None, # Instrument on Clipper
                            {None: 'NaN'}, # Materials dictionary of the form {name: volume fraction}
                            "Molecular", # Mixture model is not displayed at first, so must set to default. If not displayed, this is Molecular by default, not NaN. That is because it influences computations even if not displayed.
                            'NaN', # Minimum particle size parameter in microns, varies with size distribution
                            'NaN', # Maximum particle size parameter in microns, varies with size distribution
                            'NaN', # Power law parameter, varies with size distribution
                            'NaN', # r parameter for semi-empirical Mie defining ratio of surface areas of irregular particles relative to equal volume spheres, varies with size distribution
                            'NaN', # G parameter for semi-empirical Mie defining slope of transmission component, varies with size distribution
                            'NaN', # x0 parameter for semi-empirical Mie defining boundary between small (Mie) and large sizes, varies with size distribution
                            None, # Which scattering theory to use
                            [None, 'NaN'], # Size distribution resolution (spacing, n_sizes)
                            'NaN', # Optical depth
                            ['NaN', None]]] # Varying parameter (wavelength, grain size, scattering angle, phase angle). Second index indicates parameter type.
        self.legend_idxs = [] # Array to hold base parameter indexes to include in the legends, i.e. those that change at some point in base_param's history
        self.latest_params, _ = self.track_changes() # This grabs the initial state of the GUI
        self.most_recent_update = copy.deepcopy(self.latest_params) # This is only in case the user cancels loading at the start, so we have something to revert to. The GUI will revert to the parameters it had before the update.
        self.most_recent_update[-1][0] = '' # In case the user cancels before the first graph loads, the input box will clear

        # --- WIDGET FUNCTIONALITY ---
        # Pressing the graph mode button toggles between graph modes and clears the window
        self.graph_mode_button.clicked.connect(self.mode_toggle)

        # Changing the x-axis clears the window and changes the displayed horizontal axis
        self.x_axis.currentTextChanged.connect(self.change_axis)
        # Changing the y-axis clears the window and changes the displayed vertical axis
        self.y_axis.currentTextChanged.connect(self.change_axis)

        # Pressing scale radio buttons updates the scale
        self.x_scale.buttonClicked.connect(self.update_scale)
        self.y_scale.buttonClicked.connect(self.update_scale)

        # Changing the input label dropdown will modify things like the input box placeholder text
        self.input_label.currentTextChanged.connect(lambda: self.change_input(update=True))

        # Changing any dataset parameter (instrument, tau, input) re-plots the data
        self.instrument.currentTextChanged.connect(self.update_graph)
        self.tau_box.returnPressed.connect(self.update_graph)
        self.input_box.returnPressed.connect(self.update_graph)

        # The size distribution model and scattering theory require special functions to handle custom distribution inputs but also re-plot the data
        self.dist_model.currentTextChanged.connect(self.change_distmodel)
        self.fit_model.currentTextChanged.connect(self.change_fitmodel)

        # Connect the custom sizedist boxes to the update function
        self.smin_box.returnPressed.connect(self.update_graph)
        self.smax_box.returnPressed.connect(self.update_graph)
        self.powlaw_box.returnPressed.connect(self.update_graph)
        self.r_box.returnPressed.connect(self.update_graph)
        self.G_box.returnPressed.connect(self.update_graph)
        self.x0_box.returnPressed.connect(self.update_graph)

        # The window button connects to a function that switches to the resolution settings window
        self.window_button.clicked.connect(self.change_window)
        # The back button connects to the same function to switch back to the data input window
        self.back_button.clicked.connect(self.change_window)

        # The slider updates the number of sizes in the size distribution used by utils.py
        self.dist_slider.valueChanged.connect(self.update_slider)
        # The corresponding dropdown box updates a separate function
        self.spacing.currentTextChanged.connect(self.update_spacing)

        # Changing composition parameters re-plots the data. Any further items added to comps will be connected to update_graph as well
        self.comps[0][0].currentTextChanged.connect(self.update_graph)
        self.comps[0][1].returnPressed.connect(self.update_graph)

        # Pressing the add new plot button saves the dataset in the plot and starts a new one
        self.add_graph_button.clicked.connect(self.plot_graph)
        # Pressing the add material button adds new materials to the composition list
        self.add_material_button.clicked.connect(self.add_material)
        # Pressing the clear plot button clears the plot and resets the title and GUI
        self.clear_plot_button.clicked.connect(self.clear_graph)
        # Pressing the save plot button saves the graph and metadata to file
        self.save_plot_button.clicked.connect(self.save_graph)
        # Pressing the home zoom button auto-zooms the graph
        self.home_button.clicked.connect(lambda: self.rst_boxes(autozoom=True))
        # Pressing the load button loads external data into the graph
        self.load_button.clicked.connect(self.load_data)
        # Pressing the cancel button kills the subprocess updating the graph
        self.cancel.clicked.connect(lambda: self.kill_process(rst_gui=True, rst_proc=False))

        # Pressing return in any bounds field triggers a bounds update (will not update the plotted data)
        self.xmin_box.returnPressed.connect(self.update_bounds)
        self.xmax_box.returnPressed.connect(self.update_bounds)
        self.ymin_box.returnPressed.connect(self.update_bounds)
        self.ymax_box.returnPressed.connect(self.update_bounds)

        # --- GRAPH DRAG & RELEASE LOGIC ---
        # Create a flag to track if the user dragged the graph, triggering an update of the bounds
        # Dragging the graph connects to an event, but ensuring the graph is dragged and released
        # requires intercepting a mouse release event.
        self.graph_was_dragged = False
        # Listen for manual range changes that fire continuously during a drag
        def flag_dragged():
            self.graph_was_dragged = True
        self.graph_widget.getViewBox().sigRangeChangedManually.connect(flag_dragged)

        # Store the original mouse release/scroll wheel to avoid breaking PyQt5's internal math
        self._original_mouse_release = self.graph_widget.mouseReleaseEvent
        # The scroll wheel allows the graph to be zoomed in/out, also requiring a bounds update
        self._original_wheel = self.graph_widget.wheelEvent

        def click_release(event):
            """
            Tracks when the user releases a mouse drag on the graph, signaling an update for the bounds boxes.

            :param event: Keyboard input event to watch for
            """
            # Let PyQtGraph finish its native panning/zooming logic first
            self._original_mouse_release(event)
            # Check if this release was the end of a drag, or just a normal click
            if self.graph_was_dragged:
                # Reset the flag for the next time
                self.graph_was_dragged = False
                self.rst_boxes() # Reset bounds boxes

        def scroll_wheel(event):
            """
            Intercepts a scroll zoom update to the graph to update the bounds boxes dynamically.

            :param event: Keyboard input event to watch for
            """
            self._original_wheel(event) # Keep the original scrolling
            self.rst_boxes() # Reset bounds boxes
            
        # Override default mouse release and scroll events to allow dynamic updating of bounds boxes
        self.graph_widget.mouseReleaseEvent = click_release
        self.graph_widget.wheelEvent = scroll_wheel

        # --- LAYOUT ---
        # Items are arranged as (row, column, rowspan, columnspan)
        # Buttons are at the top alongside instruments
        self.layout.addWidget(self.graph_mode_button, 0, 0, 1, 3)
        self.layout.addWidget(self.add_material_button, 1, 0, 1, 3)
        self.layout.addWidget(self.add_graph_button, 2, 0, 1, 3)
        self.layout.addWidget(self.clear_plot_button, 3, 0, 1, 3)
        self.layout.addWidget(self.save_plot_button, 4, 0, 1, 3)
        self.layout.addWidget(self.load_button, 5, 0, 1, 3)
        self.layout.addWidget(self.instrument_label, 6, 0, 1, 1)
        self.layout.addWidget(self.instrument, 6, 1, 1, 2)

        # Widgets are overlaid on top of each other in graph section (home in top left, xmin/max and ymin/max in top right)
        self.layout.addWidget(self.graph_widget, 0, 3, 24, 24)
        self.layout.addWidget(self.home_button, 1, 4, 1, 1)
        self.layout.addWidget(self.xmin_label, 1, 23, 1, 1)
        self.layout.addWidget(self.xmin_box, 1, 24, 1, 1)
        self.layout.addWidget(self.xmax_label, 1, 25, 1, 1)
        self.layout.addWidget(self.xmax_box, 1, 26, 1, 1)
        self.layout.addWidget(self.ymin_label, 2, 23, 1, 1)
        self.layout.addWidget(self.ymin_box, 2, 24, 1, 1)
        self.layout.addWidget(self.ymax_label, 2, 25, 1, 1)
        self.layout.addWidget(self.ymax_box, 2, 26, 1, 1)

        # These are replaced by a table if a multi-material mixture is enabled
        self.layout.addWidget(self.comp_labels[0], 7, 0, 1, 1)
        self.layout.addWidget(self.comps[0][0], 7, 1, 1, 2)

        # Widget visibility updates depending on the combination of parameters selected.
        # This makes up the majority of the input section of the GUI
        self.layout.addWidget(self.dist_label,  8, 0, 1, 1)
        self.layout.addWidget(self.dist_model, 8, 1, 1, 2)
        self.layout.addWidget(self.fit_label, 9, 0, 1, 1)
        self.layout.addWidget(self.fit_model, 9, 1, 1, 2)
        self.layout.addWidget(self.window_button, 10, 0, 1, 3)
        self.layout.addWidget(self.tau_label, 11, 0, 1, 1)
        self.layout.addWidget(self.tau_box, 11, 1, 1, 2)
        self.layout.addWidget(self.input_label, 12, 0, 1, 1)
        self.layout.addWidget(self.input_box, 12, 1, 1, 2)
        self.layout.addWidget(self.x_axis_label, 13, 0, 1, 1)
        self.layout.addWidget(self.x_axis, 13, 1, 1, 2)
        self.layout.addWidget(self.y_axis_label, 14, 0, 1, 1)
        self.layout.addWidget(self.y_axis, 14, 1, 1, 2)
        self.layout.addWidget(self.x_scale_label, 15, 0, 1, 1)
        self.layout.addWidget(self.x_linear, 15, 1, 1, 1)
        self.layout.addWidget(self.x_log, 15, 2, 1, 1)
        self.layout.addWidget(self.y_scale_label, 16, 0, 1, 1)
        self.layout.addWidget(self.y_linear, 16, 1, 1, 1)
        self.layout.addWidget(self.y_log, 16, 2, 1, 1)
        self.layout.addWidget(self.load_label, 17, 0, 1, 1)
        self.layout.addWidget(self.prog, 17, 1, 1, 2)
        self.layout.addWidget(self.cancel, 18, 1, 1, 2)

        # Add both the main window and the resolution settings window to the stack
        self.stack.addWidget(self.data_input_widget)
        self.stack.addWidget(self.settings_widget)
        
        # Set stack as the central widget
        self.setCentralWidget(self.stack)
        self.stack.setCurrentIndex(0) # Show data entry screen by default

        # Set up a processor variable for heavy computing via subprocesses
        self.proc = None
        # Set up a blank queue variable. Will be changed at each compute step.
        self.result_queue = None

        # Create a timer to check if the subprocess is done
        self.monitor_timer = qc.QTimer()
        self.monitor_timer.setInterval(10) # Update time in milliseconds
        self.monitor_timer.timeout.connect(self.update_process)

        # Output the first graph using default settings in input boxes
        self.change_input() # First update the placeholder texts
        self.update_graph()

    def mode_toggle(self):
        """
        Defines the behavior of the graph mode button when clicked. Toggles between 'Reflectance vs. Wavelength', 'Surface Albedo vs. Wavelength', and 'Reflectance vs. Angle'. Clears graph, resets input fields, and updates title.

        :param self: MainWindow object
        """
        self.clear_graph() # Clear graph first. This will stop any pending updates.
        old_graphmode = self.graphmode
        old_dist = self.dist_model.currentText()
        old_fit = self.fit_model.currentText()
        self.graphmode = int((self.graphmode + 1) % 3) # Cycle graph mode between 0 and 2
        self.list_updates() # Call to ensure all lists display the appropriate items

        # Ensure missing widgets are displayed upon a mode change (relevant from mode 1 to mode 2)
        if old_graphmode == 1:
            # Create new rows to add the missing widgets we know will always be present in this graph mode
            self.mknewrow(self.row_widget(self.input_label), 2)
            self.layout.addWidget(self.dist_label, 8, 0, 1, 1)
            self.layout.addWidget(self.dist_model, 8, 1, 1, 2)
            self.layout.addWidget(self.fit_label, 9, 0, 1, 1)
            self.layout.addWidget(self.fit_model, 9, 1, 1, 2)
            self.dist_label.setVisible(True) # Make sure they are displayed so track_changes can update them properly
            self.dist_model.setVisible(True)
            self.fit_label.setVisible(True)
            self.fit_model.setVisible(True)

            # Check if the nsizes slider should display and add it after fit_model if so
            if self.window_button is not None:
                window_button_row = self.row_widget(self.fit_model) + 1
                self.mknewrow(window_button_row, 1)
                self.layout.addWidget(self.window_button, window_button_row, 0, 1, 3)
                self.window_button.setVisible(True)

            # Re-add size distribution boxes if the distribution was set to custom before
            if self.dist_model.currentText() == "Custom" and self.smin_box.isHidden():
                self.mknewrow(9, 3)
                self.layout.addWidget(self.smin_label, 9, 0, 1, 1)
                self.layout.addWidget(self.smin_box, 9, 1, 1, 2)
                self.layout.addWidget(self.smax_label, 10, 0, 1, 1)
                self.layout.addWidget(self.smax_box, 10, 1, 1, 2)
                self.layout.addWidget(self.powlaw_label, 11, 0, 1, 1)
                self.layout.addWidget(self.powlaw_box, 11, 1, 1, 2)
                self.smin_label.setVisible(True)
                self.smin_box.setVisible(True)
                self.smax_label.setVisible(True)
                self.smax_box.setVisible(True)
                self.powlaw_label.setVisible(True)
                self.powlaw_box.setVisible(True)

                # Add G, r, and x0 boxes if the scattering theory allows it
                if (self.fit_model.currentText() == "Semi-Empirical Mie" or self.fit_model.currentText() == "Mie Diffraction") and self.r_box.isHidden():
                    self.mknewrow(12, 1)
                    self.layout.addWidget(self.r_label, 12, 0, 1, 1)
                    self.layout.addWidget(self.r_box, 12, 1, 1, 2)
                    self.r_label.setVisible(True)
                    self.r_box.setVisible(True)
                if (self.fit_model.currentText() == "Semi-Empirical Mie" or self.fit_model.currentText() == "Mie Diffraction") and self.G_box.isHidden():
                    # Check if both G and r exist (Semi-Empirical Mie only) and add them on separate rows, otherwise add G to row 11.
                    if self.fit_model.currentText() == "Semi-Empirical Mie":
                        self.mknewrow(13, 1)
                        self.layout.addWidget(self.G_label, 13, 0, 1, 1)
                        self.layout.addWidget(self.G_box, 13, 1, 1, 2)
                    else:
                        self.mknewrow(12, 1)
                        self.layout.addWidget(self.G_label, 12, 0, 1, 1)
                        self.layout.addWidget(self.G_box, 12, 1, 1, 2)
                    self.G_label.setVisible(True)
                    self.G_box.setVisible(True)
                if self.fit_model.currentText() == "Semi-Empirical Mie" and self.x0_box.isHidden():
                    # x0 displays only when both r and G are present (Semi-Empirical Mie)
                    self.mknewrow(14, 1)
                    self.layout.addWidget(self.x0_label, 14, 0, 1, 1)
                    self.layout.addWidget(self.x0_box, 14, 1, 1, 2)
                    self.x0_label.setVisible(True)
                    self.x0_box.setVisible(True)

        if self.graphmode == 0:
            self.graph_mode_button.setText("Reflectance vs. Wavelength")   

        elif self.graphmode == 1:
            self.graph_mode_button.setText("Surface Albedo vs. Wavelength")

            # Remove size distribution boxes if shown and switch sizedist to G-ring-like
            if old_dist == "Custom":
                # Shift rows up
                self.mknewrow(self.row_widget(self.smin_label) + 3, -3)

                # Remove widgets from display; visibility is set to false instead of setting smin_label
                # to None to trigger a reset of size distribution boxes elsewhere.
                # Boxes that disappear reappear with their previous values when cycling through modes.
                # The code does not read these boxes while they are invisible
                self.layout.removeWidget(self.smin_label)
                self.smin_label.setVisible(False)
                self.layout.removeWidget(self.smin_box)
                self.smin_box.setVisible(False)
                self.layout.removeWidget(self.smax_label)
                self.smax_label.setVisible(False)
                self.layout.removeWidget(self.smax_box)
                self.smax_box.setVisible(False)
                self.layout.removeWidget(self.powlaw_label)
                self.powlaw_label.setVisible(False)
                self.layout.removeWidget(self.powlaw_box)
                self.powlaw_box.setVisible(False)
                
                # Hide G, r, and x0 boxes if created by the scattering theory
                if old_fit == "Semi-Empirical Mie" or old_fit == "Mie Diffraction": # Removing r
                    # Deleting rows involves going to the next row and shifting it up one
                    self.mknewrow(self.row_widget(self.r_label) + 1, -1)
                    self.layout.removeWidget(self.r_label)
                    self.r_label.setVisible(False)
                    self.layout.removeWidget(self.r_box)
                    self.r_box.setVisible(False)
                if old_fit == "Semi-Empirical Mie" or old_fit == "Mie Transmission": # Removing G
                    self.mknewrow(self.row_widget(self.G_label) + 1, -1)
                    self.layout.removeWidget(self.G_label)
                    self.G_label.setVisible(False)
                    self.layout.removeWidget(self.G_box)
                    self.G_box.setVisible(False)
                if old_fit == "Semi-Empirical Mie": # Removing x0
                    self.mknewrow(self.row_widget(self.x0_label) + 1, -1)
                    self.layout.removeWidget(self.x0_label)
                    self.x0_label.setVisible(False)
                    self.layout.removeWidget(self.x0_box)
                    self.x0_box.setVisible(False)

            # Hide unnecessary widgets, leaving only the input param
            self.mknewrow(self.row_widget(self.dist_label) + 2, -2)
            self.layout.removeWidget(self.dist_label)
            self.dist_label.setVisible(False)
            self.layout.removeWidget(self.dist_model)
            self.dist_model.setVisible(False)
            self.layout.removeWidget(self.fit_label)
            self.fit_label.setVisible(False)
            self.layout.removeWidget(self.fit_model)
            self.fit_model.setVisible(False)

            # Hide window button if present
            if self.window_button is not None:
                self.mknewrow(self.row_widget(self.window_button) + 1, -1)
                self.layout.removeWidget(self.window_button)
                self.window_button.setVisible(False)

            # Hide optical depth (tau) if visible
            if not self.tau_box.isHidden():
                self.mknewrow(self.row_widget(self.input_box), -1)
                self.layout.removeWidget(self.tau_label)
                self.tau_label.setVisible(False)
                self.layout.removeWidget(self.tau_box)
                self.tau_box.setVisible(False)

        elif self.graphmode == 2:
            self.graph_mode_button.setText("Reflectance vs. Angle")

        self.change_axis() # Call just to make sure all shown parameters correspond to the y-axis of the plot. This will also update the dropdown lists.
        self.change_input() # Call to ensure input box placeholder texts update appropriately

    def update_graph(self):
        """
        Called when data parameters change. Invokes the plotter function from the main program and outputs new temporary data to the graph, replacing old data if not made permanent. If parameters are invalid, updates status message to the
        appropriate error message and turns the appropriate box red. If multiple graphs involved, updates legend 
        accordingly.

        :param self: MainWindow object
        """
        self.all_borders('black') # Reset any previous errors

        # --- ERROR CHECKING ---
        # Error checking for materials table if available
        err_message = '' # Will dynamically update to support multiple errors at once
        if len(self.comps) > 1:
            # Update materials checks to make sure there aren't errors specifically in the materials table
            # Append any error message returned by update_materials
            err_message += self.update_materials()

        # Error checking for smin, smax, powlaw boxes if available
        if not self.smin_box.isHidden():
            smin_txt = self.smin_box.text()
            smax_txt = self.smax_box.text()
            powlaw_txt = self.powlaw_box.text()

            # Smin checks
            try:
                smin = float(smin_txt)
                if np.isnan(smin): # Check for NaN
                    err_message += "Min size must be a number. "
                elif smin <= 0:
                    # Include a space after each error message since they can stack
                    err_message += "Min size must be greater than zero. "
            except ValueError:
                err_message += "Min size must be a float. "

            # Smax checks
            try:
                smax = float(smax_txt)
                if np.isnan(smax): # Check for NaN
                    err_message += "Max size must be a number. "
                elif 'Min size' not in err_message and smax < smin:
                    err_message += "Max size must be greater than min size. "
                elif smax <= 0:
                    err_message += "Max size must be greater than zero. "
            except ValueError:
                err_message += "Max size must be a float. "

            # Powlaw checks
            try:
                powlaw = float(powlaw_txt)
                if np.isnan(powlaw): # Check for NaN
                    err_message += "Power law must be a number. "
            except ValueError:
                err_message += "Power law must be a float. "

            # If r or G are included, check those
            if not self.r_box.isHidden():
                try:
                    r = float(self.r_box.text())
                    if np.isnan(r): # Check for NaN
                        err_message += "R must be a number. "
                    elif r < 0:
                        err_message += "R must be nonnegative. "
                except ValueError:
                    err_message += "R must be a float. "
            if not self.G_box.isHidden():
                try:
                    G = float(self.G_box.text())
                    if np.isnan(G): # Check for NaN
                        err_message += "G must be a number. "
                    elif G <= 0:
                        err_message += "G must be greater than 0. " # G gets passed into a natural log and must be greater than 0
                except ValueError:
                    err_message += "G must be a float. "
            if not self.x0_box.isHidden():
                try:
                    x0 = float(self.x0_box.text())
                    if np.isnan(x0): # Check for NaN
                        err_message += "x0 must be a number. "
                    elif x0 < 0:
                        err_message += "x0 must be nonnegative. "
                except ValueError:
                    err_message += "x0 must be a float. "
            
        # --- READ PARAMETERS ---
        # Read each parameter from the appropriate widget
        sensor = self.instrument.currentText()

        # Check for updated comps here to catch , though the error message displays at the end
        # 
        compositions = {} # This will be used to update the graph/legend
        if "Volume fractions" not in err_message: # Only compute if no comp errors were detected
            for i, item in enumerate(self.comps):
                if float(item[1].text()) != 0: # Compositions with v/v equal to zero don't count in legend updates
                    compositions[item[0].currentText()] = item[1].text()

            # Composition changes require getting supported wavelengths for that material
            # Set up extreme cases
            min_limiting_comp = ['', 0] # First index will be material, second will be limiting wavelength bound
            max_limiting_comp = ['', 99999999999]
            # Check which nonzero v/v material has the greatest minimum and least maximum wavelength
            # Get a baseline wavelength range bound
            for comp in compositions.keys():
                comp_min_wavel, comp_max_wavel = utils.get_minmax_wavel(comp)
                if comp_min_wavel[1] > min_limiting_comp[1]:
                    min_limiting_comp[0] = comp
                    min_limiting_comp[1] = comp_min_wavel[1]
                if comp_max_wavel[1] < max_limiting_comp[1]:
                    max_limiting_comp[0] = comp
                    max_limiting_comp[1] = comp_max_wavel[1]
            self.comp_min_wavel = min_limiting_comp
            self.comp_max_wavel = max_limiting_comp
        
        # Optical depth and the parameter are initially read as strings
        tau_str = self.tau_box.text()
        self.param_str = self.input_box.text() # param_str is a self variable because it needs to be referenced elsewhere

        # Error checking for parameter if it is a wavelength
        if self.graphmode == 2:
            try:
                param = float(self.param_str)
                if np.isnan(param): # Check for NaN
                    err_message += 'Wavelength must be a number. '
                # Ensure wavelength is within sensor range and index of refraction data for the material
                # Ensure no prior composition errors so self.comp_min_wavel updates correctly
                elif "Volume fractions" not in err_message:
                    if param < self.comp_min_wavel[1]:
                        err_message += f'Wavelength is too low for {self.comp_min_wavel[0]}. '
                    elif param > self.comp_max_wavel[1]:
                        err_message += f'Wavelength is too high for {self.comp_max_wavel[0]}. '
                if param < vars.INSTRUMENTS[sensor][0]:
                    err_message += f'Wavelength is too low for {sensor}. '
                elif param > vars.INSTRUMENTS[sensor][1]:
                    err_message += f'Wavelength is too high for {sensor}. '
            except ValueError:
                # Allow MIN and MAX input to get the bounds of the instrument range
                if self.param_str == 'MIN':
                    param = max(vars.INSTRUMENTS[sensor][0], self.comp_min_wavel[1]) # Set it to the minimum possible wavelength with the given initial conditions
                    self.param_str = str(param) # Param str is used in title updates, ensure it doesn't display MAX but an actual number
                    # Update input box to display the parameter value instead of MIN for legend/title clarity
                    self.input_box.setText(self.param_str)
                elif self.param_str == 'MAX':
                    param = min(vars.INSTRUMENTS[sensor][1], self.comp_max_wavel[1]) # Set it to the maximum possible wavelength with the given initial conditions
                    self.param_str = str(param)
                    self.input_box.setText(self.param_str)
                else:
                    err_message += f'Wavelength must be a float, MIN, or MAX. '

        # Error checking for parameter if it is a scattering angle
        elif self.graphmode == 0:
            # Check if the code expects a scattering or phase angle
            angle_str = 'Scattering angle' if 'Scattering' in self.input_label.currentText() else 'Phase angle'
            # NaN check is omitted since NaN is a float, not an integer
            try:
                param = int(self.param_str)
                if param < 0 or param > 180:
                    err_message += f'{angle_str} must be between 0 and 180 degrees. '
            except ValueError:
                err_message += f'{angle_str} must be an integer. '

        # Error checking for parameter if it is an effective grain size
        elif self.graphmode == 1:
            try:
                param = float(self.param_str)
                if np.isnan(param): # Check for NaN
                    err_message += f'Effective grain size must be a number. '
                elif param <= 0:
                    err_message += f'Effective grain size must be greater than zero. '
            except ValueError:
                err_message += f'Effective grain size must be a float. '

        # Error checking for lack of data in wavelength range for wavelength-dependent plots
        if self.graphmode == 0 or self.graphmode == 1:
            # Trigger error if max data wavelength is below min sensor wavelength or vice versa
            # Only one condition can be true since min <= max wavelength
            if self.comp_min_wavel[1] > vars.INSTRUMENTS[sensor][1]:
                err_message += f'No data available for {self.comp_min_wavel[0]} in {sensor} wavelength range. '
            elif self.comp_max_wavel[1] < vars.INSTRUMENTS[sensor][0]:
                err_message += f'No data available for {self.comp_max_wavel[0]} in {sensor} wavelength range. '

        # Error checking for optical depth
        try:
            tau = float(tau_str)
            if np.isnan(tau):
                err_message += 'Optical depth must be a number. '
            elif tau < 0:
                err_message += 'Optical depth must be nonnegative. '
        except ValueError:
            err_message += 'Optical depth must be a float. '

        # --- UPDATE LOGIC ---
        # Update only if there are no input errors
        if len(err_message) == 0:
            # Get current parameters and indices differing from original data
            latest_params, temp_legend_idxs = self.track_changes()
            if self.latest_params == latest_params and self.proc is not None: # If the most recent update is identical to a previously requested update and a process is running, don't bother restarting it
                self.load_label.setText("Graph is still updating")
                # Generic errors turn all boxes red
                self.all_borders('red')
            else:
                self.latest_params = latest_params
                # Check if base_params has only one non-loaded plot before storing parameters for legend updates.
                # If so, parameter variation is absent and legend tracking is unnecessary.
                # If the last graph was loaded and base_param size reduced to 1, this update implies the size is
                # expanding to 2 with potential parameter variance; temp_legend_idxs is not clamped.
                if len(self.base_params) == 1 and not self.last_graph_loaded:
                    self.temp_legend_idxs = []
                else:
                    self.temp_legend_idxs = temp_legend_idxs # Temp legend idxs needs to be global because it is used in update_process

                # Skip calculations if fields have not changed and ensure the previous graph was not loaded before throwing an error
                if not self.last_graph_loaded and self.latest_params == self.base_params[-1]:
                    self.load_label.setText("Plot is identical to previous graph")
                    # Generic errors turn all boxes red
                    self.all_borders('red')
                # temp_legend_idxs is None only if multiple plots exist and the current plot
                # matches a previous plot in the window history
                # 
                elif temp_legend_idxs is None:
                        self.load_label.setText("Plot is identical to existing data")
                        # Remove graph to clarify erroneous data is not plotted and GUI display differs from graph
                        self.ifs.setData([], [])
                        self.legend.removeItem(self.ifs)
                        # Replace last base_params element with a NaN array if the spot exists
                        if not self.last_graph_loaded:
                            self.base_params[-1] = [None, {None: 'NaN'}, "Molecular", 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', None, [None, 'NaN'], 'NaN', ['NaN', None]] # Reset base params to null to ensure graph is fully cleared
                        self.all_borders('red')
                else:
                    if self.proc is not None:
                        self.kill_process(rst_gui=False, rst_proc=True) # Terminate any pending update, but DO NOT revert the GUI because the user updated something. Also, keep the loading bar, as we are only resetting the process
                    self.load_label.setText("Updating...") # Status label changes to indicate loading
                    qt.QApplication.processEvents() # Force a screen update to render the status label and the box borders during computation

                    # Create a queue for the computation
                    self.result_queue = mp.Queue()

                    # Store previous parameters in most_recent_update to revert GUI on cancel
                    # Ensure instrument is not None before updating most_recent_update
                    if len(self.base_params) > 0 and self.base_params[-1][0] != None:
                        self.most_recent_update = copy.deepcopy(self.base_params[-1]) # Get current state of GUI. We have to deep copy else updating most_recent_update will update base_params for some reason.

                    # Reuse the same pen color for all updates to the current plot
                    self.plt_pen.setColor(pg.mkColor(COLOR_ARR[self.last_color_idx]))

                    # Stop timer if running
                    self.monitor_timer.stop()
                    # Check to ensure a process isn't already running, and if it is, kill it
                    if self.proc is not None:
                        self.proc.terminate() # Send the SIGTERM signal to the process
                        self.proc.join()

                    # Show progress bar to assure user code is updating
                    if self.prog.isHidden():
                        self.prog.setVisible(True)
                        self.cancel.setVisible(True)
                    self.prog.setValue(0) # Start at 0% complete
                        
                    # Set up a multiprocessing process to do the heavy calculations on a separate core
                    self.proc = mp.Process(target=output_graph, args=(self.graphmode, self.latest_params[0], self.latest_params[1], self.latest_params[2], [self.latest_params[3], self.latest_params[4], self.latest_params[5], self.latest_params[6], self.latest_params[7], self.latest_params[8]], self.latest_params[9], (float(self.latest_params[12][0]) if 'Phase Angle' not in self.input_label.currentText() else 180 - float(self.latest_params[12][0])), self.latest_params[11], self.latest_params[10], (self.comp_min_wavel[1], self.comp_max_wavel[1]), [self.x_axis.currentText(), self.y_axis.currentText()], self.result_queue))
                    self.proc.start()
                    self.monitor_timer.start()

        else:
            if self.proc is not None:
                self.kill_process(rst_gui=False) # Terminate any pending update, but DO NOT revert the GUI so that the user can see exactly what was wrong with their input
            self.load_label.setText(err_message)
            # Scan the error message for problematic variables and turn their boxes red
            if 'Min size' in err_message or 'min size' in err_message:
                self.smin_box.setStyleSheet(f"border: 1px solid red;")
            if 'Max size' in err_message:
                self.smax_box.setStyleSheet(f"border: 1px solid red;")
            if 'Power law' in err_message:
                self.powlaw_box.setStyleSheet(f"border: 1px solid red;")    
            if 'R ' in err_message: # Should include a space, just in case since R is a very generic thing to find in a string
                self.r_box.setStyleSheet(f"border: 1px solid red;")
            if 'G ' in err_message:
                self.G_box.setStyleSheet(f"border: 1px solid red;")
            if 'x0' in err_message:
                self.x0_box.setStyleSheet(f"border: 1px solid red;")
            if 'Wavelength' in err_message or 'angle' in err_message or 'Effective grain size' in err_message:
                self.input_box.setStyleSheet("border: 1px solid red;")
            if 'Optical depth' in err_message:
                self.tau_box.setStyleSheet("border: 1px solid red;")
            # Composition and size distribution errors get handled earlier

    def plot_graph(self):
        """
        Saves current x_data and y_data to the graph as a permanent plot and resets the program for a new plot, accommodating multiple datasets. Sets up the legend 
        and updates the title.

        :param self: MainWindow object
        """
        if self.proc is not None:
            self.kill_process() # Stop any current updates

        # If the data haven't been plotted for this graph yet, throw a general error
        if len(self.x_data) == 0 or len(self.y_data) == 0:
            self.load_label.setText("No plot to add")
            self.all_borders('red')
        else:
            # Create a legend if none already. Needs to be here since legends are for multiple graphs only
            if self.legend == None:
                self.legend = self.graph_widget.addLegend()
                self.legend.setLabelTextColor('k') # Default is gray and hard to see; we set it to black

            self.legend.removeItem(self.ifs) # If temporary plot has a legend item, remove it to establish a permanent one
            # Turn the temporary plot into a permanent one not associated with a temporary self.ifs variable.
            self.graph_widget.plot(self.x_data, self.y_data, pen=pg.mkPen(color=COLOR_ARR[self.last_color_idx], width=3), name='')
            
            # Update the legend and title to reflect the new dataset
            self.legend_idxs = copy.deepcopy(self.temp_legend_idxs) # We grab whatever is currently in the temporary legend. Now it is the official legend.
            self.generate_legend(self.legend_idxs)
            self.update_title(self.legend_idxs)
            # Expand the size of base_params, which tracks history of graphs. The last two entries are now identical,
            # but update_graph will change the last entry before this gets called again
            # We also first check that the last plot wasn't loaded data
            if not self.last_graph_loaded:
                self.most_recent_update = copy.deepcopy(self.base_params[-1]) # Get the current state of the GUI
                self.most_recent_update[-1][0] = '' # Set the parameter string to '', so that when we update the graph, if we hit 'Cancel', the parameter box clears
                # Append a blank array to represent the new plot
                self.base_params.append([None, {None: 'NaN'}, "Molecular", 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', None, [None, 'NaN'], 'NaN', ['NaN', None]])
            self.multiple += 1 # Set the global variable to indicate multiple graphs stored on plot
            # Cycle through to the next color for the next dataset
            self.last_color_idx = (self.last_color_idx + 1) % len(COLOR_ARR)
            # self.ifs becomes ready for the next dataset
            self.x_data = []
            self.y_data = []
            self.ifs.setData(self.x_data, self.y_data)
            self.all_borders('black') # Remove all outstanding errors
            self.load_label.setText("Up to date")

    def clear_graph(self):
        """
        Clears all graphs from the current plot, as well as the parameter in the parameter box. Removes any
        material table and replaces it with a single material dropdown box.

        :param self: MainWindow object
        """
        if self.proc is not None:
            self.kill_process() # Stop any current updates

        self.multiple = 0 # No graphs means no multiple
        if len(self.base_params) > 0:
            self.most_recent_update = copy.deepcopy(self.base_params[-1]) # Get the most recently fully plotted non-loaded plot
        else:
            self.most_recent_update, _ = self.track_changes() # This isn't ideal, but if we truly have no non-loaded plots displayed in the graph, we grab the crrent state of the GUI
        self.most_recent_update[-1][0] = '' # Set the parameter string to '', so that when we update the graph, if we hit 'Cancel', the parameter box clears
        self.param_str = '' # Reset input box
        self.base_params = [[None, {None: 'NaN'}, "Molecular", 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', None, [None, 'NaN'], 'NaN', ['NaN', None]]] # Reset to a blank entry, signaling to the next update graph NOT to update the most recent parameters
        self.input_box.setText(self.param_str)
        # --- RESET GUI ---
        # Reset temporary plot
        self.x_data = []
        self.y_data = []
        self.ifs.setData(self.x_data, self.y_data)
        self.last_color_idx = 0 # Set temporary plot color to red
        self.graph_widget.clearPlots() # Erase all permanent plots
        self.legend = None # Erase stored legend
        # Erase all base parameters and varying parameters
        self.legend_idxs = []
        self.temp_legend_idxs = [] # This is for holding temporary information about the legend
        # Erase all loaded data indices
        self.loaded_idxs = {}
        self.last_graph_loaded = False
        self.update_title([]) # Update title to default. [] ensures that no parameters are listed as varying

        # Remove material table and mixture model toggle if present
        if self.mat_table is not None:
            # Erase all compositions except the first
            self.comps = [[qt.QComboBox(), qt.QLineEdit()]]
            self.comps[0][0].addItems(COMPS)
            self.comps[0][0].setStyleSheet(BKSHEET)
            self.comps[0][1].setText('1.0')
            self.comps[0][1].setPlaceholderText("Input volume fraction")
            # Reconnect combo box & line edit to update function
            self.comps[0][0].currentTextChanged.connect(self.update_graph)
            self.comps[0][1].returnPressed.connect(self.update_graph)
            # Recompute material wavelength bounds
            self.comp_min_wavel, self.comp_max_wavel = utils.get_minmax_wavel(self.comps[0][0].currentText())

            self.layout.removeWidget(self.mat_table)

            # Call row function to shift all elements below toggle up one
            self.mknewrow(self.row_widget(self.mixture_model_label) + 1, -1)
            # Remove the elements of the toggle
            self.layout.removeWidget(self.mixture_model_label)
            self.layout.removeWidget(self.molecular)
            self.layout.removeWidget(self.areal)

            # For some reason, if we don't set these to None, "ghost versions" will appear in the window even after removed
            # So we must clear them here.
            self.mat_table = None
            self.mixture_model_label = None
            self.molecular = None
            self.areal = None

            # Add singular label and composition dropdown
            self.layout.addWidget(self.comp_labels[0], 7, 0, 1, 2)
            self.layout.addWidget(self.comps[0][0], 7, 1, 1, 2)

        self.rst_boxes(autozoom=True) # Re-zoom to fit
        self.all_borders('black') # Remove all outstanding errors
        self.load_label.setText("Up to date")

    def update_scale(self):
        """
        Updates the X and Y scales based on the current values of the corresponding radio button groups.

        :param self: MainWindow object
        """
        # Read scale text
        xscale = self.x_scale.checkedButton().text()
        yscale = self.y_scale.checkedButton().text()
        # Set scale of graph
        self.graph_widget.setLogMode(True if xscale == "Logarithmic" else False, True if yscale == "Logarithmic" else False)

    def save_graph(self):
        """
        Saves the current plot to file. Allows user to specify path through popup box

        :param self: MainWindow object
        (:param dialog_title:, :param starting_directory:, :param file_extension: Handled automatically)
        """
        # First check to ensure something is even plotted
        if len(self.graph_widget.listDataItems()) == 0:
            self.load_label.setText("No plot to save")
            # No boxes will be turned red
        else:
            # Open the file save dialogue box. Here user can specify the path they want to save the graph to
            file_path, _ = qt.QFileDialog.getSaveFileName(
                self, 
                "Save Plot As", 
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), # We will default the file to the PARENT directory, not gui.py's directory
                "PNG Files (*.png);;All Files (*)"
            )

            # Check if the user actually selected a path (or hit 'Cancel')
            if file_path:
                file_path = os.path.abspath(file_path)
                # Remove any file extension
                file_path = file_path.replace('.png', '')
                file_path = file_path.replace('.jpg', '')

                # Use PyQtGraph's exporter to save the widget content
                self.exporter = pg.exporters.ImageExporter(self.graph_widget.plotItem)
                self.exporter.export(file_path + '.png')

                caption = {}
                # Write information about the constants used to generate the graphs to a CSV in the same directory
                # But we only do it if there is non-loaded data in the plot
                if len(self.base_params) > 0:
                    caption = self.generate_caption() # Get all the unchanging parameters
                    with open(file_path + '_params.csv', mode='w', newline='') as file:
                        writer = csv.writer(file)
                        for key, value in caption.items():
                            writer.writerow([key, value])

                self.all_borders('black') # Reset all input boxes to black border
                
                # Confirm whether file saved successfully & set load label to display that
                save_txt = ""
                # Check to see if the graph saved correctly
                if os.path.exists(file_path + '.png'):
                    save_txt += f"Graph saved to {file_path}.png. "
                else:
                    save_txt += f"File failed to save. "

                # Check to see if the parameters saved correctly (and were expected to)
                if len(caption) > 0:
                    if os.path.exists(file_path + '_params.csv'):
                        save_txt += f"Parameters saved to {file_path}_params.csv. "
                    else:
                        save_txt += f"Parameters failed to save. "
                # Update the load label with the status message
                self.load_label.setText(save_txt)

    def update_title(self, title_elements):
        """
        Updates the graph title dynamically based on current parameter values.

        :param self: MainWindow object
        :param title_elements: Array of integer indices to indicate which columns from base_params 
                               to include in the title. Shape len(self.legend_idxs)
        """
        title_str = ''

        title_str += self.y_axis.currentText() + " vs. " # First add the y-axis variable being changed
        title_str += f"{('Scattering Angle ' if self.x_axis.currentText() == 'Scattering Angle' else 'Phase Angle ') if self.graphmode == 2 else 'Wavelength '}" # Then add the x-axis variable

        # Check to see if there are only loaded datasets in the graph
        if self.multiple == len(self.loaded_idxs) - 1: # If there are as many graphs as there are loaded graphs, i.e. no nonloaded graphs
            if len(self.loaded_idxs.keys()) == 1:
                title_str += f'for {self.loaded_idxs[next(iter(self.loaded_idxs.keys()))]} Data'
            elif len(self.loaded_idxs.keys()) > 1:
                title_str += f'for Various Data'
        else: # There are nonloaded plots
            if self.multiple > len(self.loaded_idxs) and title_elements is not None and len(title_elements) > 0: # If there are more than two nonloaded graphs
                # If multiple, legend_idxs likely has a nonzero length and will usually be the parameter passed to this
                # program. Loop through and determine where the plots are varying, then include those in the title.
                if self.multiple > 0 and self.multiple:
                    title_str += 'Across Various '
                    i = 0

                    # Check to make sure every plot shown is a named distribution, and if not, indicate that size distribution parameters should be shown in title
                    known_params = True
                    for plot in self.base_params:
                        sizedist_params = [float(x) for x in plot[3:9]] # Cast each size distribution parameter of the plot to a float
                        model_found = False # We only need one instance of 'False' to conclude something other than named distributions exists
                        # Iterate through each distribution and try to find size distribution in parameters
                        for arr in vars.SIZEDISTS.values():
                            if sizedist_params == arr[0]:
                                model_found = True
                                break
                        # If no matching distributions were found, distribution contains custom models
                        if not model_found:
                            known_params = False
                            break

                    # Now we loop over each element to add to the title
                    while i < len(title_elements):
                        idx = title_elements[i] # Get the title element index
                        if idx == 0:
                            title_str += 'Sensors'
                        elif idx == 1:
                            title_str += 'Compositions'
                        elif idx == 2:
                            title_str += 'Mixture Models'
                        # These are the distribution parameters
                        elif idx >= 3 and idx <= 8:
                            # If all of the parameters have known size distributions, include that
                            if known_params:
                                title_str += 'Size Distributions'
                                # Skip max sizes, powerlaws, Rs, Gs, and x0s if all known size distributions
                                remaining_idxs = np.where(np.asarray(title_elements) > 8)[0]
                                if len(remaining_idxs) > 0:
                                    i = remaining_idxs[0] - 1 # Will be incremented by 1 in the end, so need to decrement to keep constant
                                else:
                                    break # The title is done
                                # We can't just 'continue' because we might miss a comma and the grammar would be weird
                            elif idx == 3:
                                title_str += 'Minimum Sizes'
                            elif idx == 4:
                                title_str += 'Maximum Sizes'
                            elif idx == 5:
                                title_str += 'Power Laws'
                            elif idx == 6:
                                title_str += 'Rs'
                            elif idx == 7:
                                title_str += 'Gs'
                            elif idx == 8:
                                title_str += 'x0s'
                        elif idx == 9:
                            title_str += 'Scattering Theories'
                        elif idx == 10:
                            title_str += 'Resolutions'
                        elif idx == 11:
                            title_str += 'Optical Depths'
                        elif idx == 12:
                            if self.graphmode == 0:
                                # Check to see whether there are different kinds of angles being plotted
                                angle_slice = [entry[-1][1] for entry in self.base_params]
                                if 'Scattering Angle (°):' in angle_slice and 'Phase Angle (°):' in angle_slice:
                                    title_str += 'Angles'
                                elif 'Phase Angle (°):' in angle_slice:
                                    title_str += 'Phase Angles'
                                elif 'Scattering Angle (°):' in angle_slice:
                                    title_str += 'Scattering Angles'
                            elif self.graphmode == 1:
                                title_str += 'Effective Grain Sizes'
                            else:
                                title_str += 'Wavelengths'

                        # Adding commas and an 'and' after the second-to-last element for grammatical accuracy
                        if i < len(title_elements) - 2:
                            title_str += ', '
                        elif i == len(title_elements) - 2:
                            if ',' in title_str: # ALWAYS add the Oxford comma if 3+ items!
                                title_str += ', and '
                            else:
                                title_str += ' and '
                        i += 1 # Increment
            # A single graph will always have its wavelength (if ref vs angle) or scattering angle (if ref vs wavelength) displayed
            elif len(self.base_params) > 0: # If there is only one nonloaded graph
                title_str += 'at ' + (str(self.base_params[-1][-1][0]) if self.base_params[-1][-1][0] != 'NaN' else '')
                if self.graphmode == 0:
                    title_str += '° ' + ('Scattering Angle' if 'Scattering Angle' in self.input_label.currentText() else 'Phase Angle')
                elif self.graphmode == 1:
                    title_str += ' μm Effective Grain Size'
                else:
                    title_str += ' μm Wavelength'

            # As long as the title does not detect various instruments being represented, it is helpful to include the
            # instrument observing the data
            if 'Sensors' not in title_str:
                title_str = self.instrument.currentText() + ' ' + title_str

            # At the end of the title, we check to see if there is any loaded data and if so, add that to the end
            if len(self.loaded_idxs.keys()) == 1:
                title_str += f", Plotted Against {self.loaded_idxs[next(iter(self.loaded_idxs.keys()))]} Data"
            elif len(self.loaded_idxs.keys()) > 1:
                title_str += f", Plotted Against Various Data"

        self.graph_widget.setTitle(title_str, color='k') # Update the graph title

    def track_changes(self):
        """
        Reads the current state of all input fields, compares it to the previous state, and outputs the current state alongside differences.

        :param self: MainWindow object
        :returns: List of strings representing values in each field. Shape len(self.base_params)
                  List of current values that vary across all of base_params and current fields.
        """
        # Add a dictionary to hold all new composition values
        # If compositions are added (self.comps grows larger), that is handled by add_material
        compositions = {}
        for i, item in enumerate(self.comps):
            key = item[0].currentText()
            value = item[1].text()
            try:
                # update_materials will check the majority of problematic material entries
                # This is just a last line of defense
                if len(value) != 0 and float(value) != 0:
                    compositions[key] = float(value)
            except ValueError: # In case of any error, skip the material
                continue

        # If multiple compositions, read mixture model, else grab the previous value in the base parameters
        if len(self.comps) > 1:
            mix = self.mixture_model.checkedButton().text()
        else:
            # This could be either areal or molecular. In single-material mixtures it doesn't matter which
            # one is selected, so we match it to the first base_params entry to avoid the title saying we are
            # varying mixture model upon adding single-material data.
            mix = self.base_params[-1][2] if len(self.base_params) > 0 else 'Molecular'

        # If no control over sizedist data, then read distribution type and grab from vars file
        if self.smin_box.isHidden() and not self.dist_model.isHidden():
            datamodel_params = vars.SIZEDISTS[self.dist_model.currentText()][0]
        # If control over sizedist data, read from appropriate boxes, but only if the user can change them
        elif not self.smin_box.isHidden() and not self.dist_model.isHidden():
            # We also need to check to make sure values were actually entered, and if not, replace all erroneous values with 'NaN'. For most_recent_update loads, this will get replaced with ''.
            try:
                smin = float(self.smin_box.text())
            except ValueError:
                smin = 'NaN'
            try:
                smax = float(self.smax_box.text())
            except ValueError:
                smax = 'NaN'
            try:
                powlaw = float(self.powlaw_box.text())
            except ValueError:
                powlaw = 'NaN'
            datamodel_params = [smin, smax, powlaw]
            # r, G, and x0 boxes may or may not be visible depending on scattering theory; always append 'NaN' as a backup
            if not self.r_box.isHidden():
                try:
                    datamodel_params.append(float(self.r_box.text()))
                except ValueError:
                    datamodel_params.append('NaN')
            else:
                datamodel_params.append('NaN')
            
            if not self.G_box.isHidden():
                try:
                    datamodel_params.append(float(self.G_box.text()))
                except ValueError:
                    datamodel_params.append('NaN')
            else:
                datamodel_params.append('NaN')
            
            if not self.x0_box.isHidden():
                try:
                    datamodel_params.append(float(self.x0_box.text()))
                except ValueError:
                    datamodel_params.append('NaN')
            else:
                datamodel_params.append('NaN')
        # Default case is all NaNs. This is only really for surface albedo.
        else:
            datamodel_params = ['NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'Nan']

        # Only grab size distribution resolution if resolution is active
        if self.window_button is not None and not self.window_button.isHidden():
            res = [self.spacing.currentText(), self.dist_slider.value() / 100 if self.spacing.currentText() == 'Logarithmic' else self.dist_slider.value()]
        else:
            res = ['NaN', 'NaN']

        # Only grab tau if changeable, i.e. if y-axis is reflectance
        if not self.tau_box.isHidden():
            try:
                tau = float(self.tau_box.text()) # Other checks will be made to ensure tau is a float
            except ValueError:
                tau = 'NaN'
        else:
            tau = 'NaN'
        
        # Always grab input value
        try:
            param = int(self.input_box.text()) if 'Angle' in self.input_label.currentText() else float(self.input_box.text())
        except ValueError:
            param = 'NaN'

        # Create a new array of updated parameters
        updated_params = [self.instrument.currentText(),
                          compositions,
                          mix,
                          datamodel_params[0],
                          datamodel_params[1],
                          datamodel_params[2],
                          datamodel_params[3],
                          datamodel_params[4],
                          datamodel_params[5],
                          self.fit_model.currentText() if not self.fit_model.isHidden() else None,
                          res,
                          tau,
                          [param, self.input_label.currentText()]]

        # Check where the new parameters do not match the base ones
        update_idxs = []
        for i, element in enumerate(updated_params):
            if len(self.base_params) > 0 and element != self.base_params[0][i]:
                update_idxs.append(i)
                
        # If the array is found to match some previous entry in the graph, clear the return array
        # This will trigger a special kind of error in the rest of the code
        if len(self.base_params) > 0 and updated_params in self.base_params:
            return_idxs = None
        else:
            # If the special error was triggered previously, it makes the variable legend_idxs un-appendable. Fix this issue for the new iteration.
            if self.legend_idxs is None:
                self.legend_idxs = []
            # Combine previous varying values with new values found to vary, sorted
            return_idxs = np.unique(np.concatenate((np.asarray(self.legend_idxs), update_idxs))).astype(int).tolist()
        return updated_params, return_idxs


    def generate_legend(self, legend_idxs):
        """
        Loops through an existing legend items object and updates each element based on legend_idxs to account for varying base_params.
        
        :param self: MainWindow object
        :param legend_idxs: Array of indices in base_params indicating which parameters are varying. 
                            Shape (len(self.base_params[0]))
        """
        # First get a slice of base_params to figure out if different angles are being displayed
        angle_slice = [entry[-1][1] for entry in self.base_params]
        multiple_angle_types = False
        # Check for the different kinds of angles. If both exists, we will display them in the legend
        if self.graphmode == 0 and 'Scattering Angle (°):' in angle_slice and 'Phase Angle (°):' in angle_slice:
            multiple_angle_types = True
        i = 0 # A variable that will increment for each non-loaded plot considered, helps index through base_params
        # Loop through each index in legend
        for leg_idx in range(len(self.legend.items)):
            legend_str = ''

            # First check to see if the item is loaded data
            if str(leg_idx) in self.loaded_idxs.keys():
                legend_str += self.loaded_idxs[str(leg_idx)] # If so, add the data name
            elif legend_idxs is not None and len(legend_idxs) > 0: # Otherwise, parse legend_idxs
                name = '' # String to hold a named distribution
                combo = self.base_params[i][3:9] # This is similar to the process used to determine if there is an unnamed distribution for the title updating
                        
                # Set the named distribution based on matching sizedist params, if any
                # This one, unlike in the title updating, gets a name on a plot-by-plot basis
                for model in vars.SIZEDISTS.keys():
                    if vars.SIZEDISTS[model][0] == combo:
                        name = model
                        break

                # Loop through all changing parameters
                idx = 0
                while idx < len(legend_idxs):
                    item = legend_idxs[idx] # Get the item
                    base_item = self.base_params[i][item] # Get the value from base parameters
                    # Instruments, mixture models, and scattering theories can have their strings added directly
                    if item == 0 or item == 2 or item == 9:
                        legend_str += base_item
                    # Compositions are added directly if only one, else each one is added with its corresponding v/v
                    elif item == 1:
                        comps_dict = base_item
                        if len(comps_dict) == 1:
                            legend_str += next(iter(comps_dict)) # Gets first (and only) element of the dictionary
                        else: # Format will be (v/v_MaterialA=X.X, v/v_MaterialB=Y.Y)
                            legend_str += '('
                            for j, key in enumerate(comps_dict.keys()):
                                legend_str += f'v/v_{key}={comps_dict[key]}'
                                # Add commas if before last element
                                if len(comps_dict.keys()) > 1 and j < len(comps_dict.keys()) - 1:
                                    legend_str += ', ' # This needs no 'and' case
                            legend_str += ')'
                    # smin, smax, powlaw, r, G, x0, tau, and the input will be added numerically
                    elif item >= 3 and item <= 8:
                        # Skip the remaining smax and tau if a named distribution
                        if len(name) > 0:
                            legend_str += name # If a named distriubtion, add that and skip ahead
                            # Skip all the sizedist parameters to after index 8
                            remaining_idxs = np.where(np.asarray(legend_idxs) > 8)[0]
                            if len(remaining_idxs) > 0:
                                idx = remaining_idxs[0] - 1 # Will be incremented by 1 in the end, so need to decrement to keep constant
                            else:
                                self.legend.items[leg_idx][1].setText(legend_str) # Update legend item
                                break # The legend is done for this element
                        elif item == 3:
                            # Otherwise, add the minimum size
                            legend_str += f'smin={base_item} μm'
                        # Only add smax/powlaw/r/G/x0 if unnamed distribution
                        elif item == 4:
                            legend_str += f'smax={base_item} μm'
                        elif item == 5:
                            legend_str += f'powlaw={base_item}'
                        elif item == 6:
                            legend_str += f'r={base_item}'
                        elif item == 7:
                            legend_str += f'G={base_item}'
                        elif item == 8:
                            legend_str += f'x0={base_item}'
                    elif item == 10:
                        if base_item[0] == 'Linear':
                            legend_str += 'lin nsizes='
                        elif base_item[0] == 'Logarithmic':
                            legend_str += 'log base='
                        legend_str += f'{base_item[1]}'
                    elif item == 11:
                        legend_str += f'tau={float(base_item):.2e}' # Tau can have exponents. Best to put in scientific notation
                    elif item == 12:
                        legend_str += f"{base_item[0]}{' μm' if (self.graphmode == 1 or self.graphmode == 2) else '°'}"
                        # If there are multiple types of angles, signify which type of angle this is
                        if multiple_angle_types:
                            legend_str += (' scat' if 'Scattering' in base_item[1] else ' phase')

                    # If before the last thing to add to legend, add a comma and a space
                    if len(legend_idxs) > 1 and idx < len(legend_idxs) - 1:
                        legend_str += ', '
                    idx += 1 # Increment counter
                i += 1 # Increment index in base_params
            else:
                # By default, we list the angle in degrees or the wavelength in microns for the plot
                legend_str += f"{self.base_params[i][-1][0]}{' μm' if (self.graphmode == 1 or self.graphmode == 2) else '°'}"
                i += 1 # Increment index in base_params

            self.legend.items[leg_idx][1].setText(legend_str) # Update legend item

    def generate_caption(self):
        """
        Generates a caption-like string of all unmodified parameters across plots. Assumes self.base_params has a nonzero length.

        :param self: MainWindow object
        :returns: Dictionary containing all unmodified base parameters and their values
        """
        # Find all the indices where the legend idxs and the parameter list don't match
        # This is where the unchanged parameters are
        param_idxs = np.arange(0, len(self.base_params[0]), 1)
        mask = ~np.isin(param_idxs, self.temp_legend_idxs) # Filter out which parameters aren't in the temporary legend idxs. Note that we use temporary idxs because we want to include the most recent plot.
        unchanged_idxs = np.where(mask)[0]

        caption = {} # Create an empty dictionary to hold constants
        for idx in unchanged_idxs:
            item = self.base_params[0][idx] # The first entry in base_params will suffice for constant parameters
            if item != 'NaN' and item is not None: # Ensure no NaNs/irrelevant parameters get added
                if idx == 0:
                    caption['Instrument'] = item
                elif idx == 1:
                    caption['Composition'] = item
                elif idx == 2 and len(self.base_params[0][1]) > 1:
                    caption['Mixture Model'] = item
                elif idx == 3:
                    caption['Minimum Particle Radius (μm)'] = item
                elif idx == 4:
                    caption['Maximum Particle Radius (μm)'] = item
                elif idx == 5:
                    caption['Power Law'] = item
                elif idx == 6:
                    caption['Surface Area Ratio of Irregular Particle to Equal Volume Sphere'] = item
                elif idx == 7:
                    caption['Constant for Mie Transmission'] = item
                elif idx == 8:
                    caption['Small Particle Boundary Size'] = item
                elif idx == 9:
                    caption['Scattering Theory'] = item
                elif idx == 10:
                    caption['Resolution'] = item
                elif idx == 11:
                    caption['Optical Depth'] = item
                elif idx == 12:
                    caption['Input'] = item
        return caption

    def add_material(self):
        """
        Adds a material dropdown list and its corresponding v/v entry box to the materials table. Creates the table and mixture model toggle first if not present.

        :param self: MainWindow object
        """ 
        # If no table or radio buttons, add them. Original material will be added to the table first
        if len(self.comps) == 1:
            # Set up radio buttons to toggle between areal and molecular mixture types
            self.mixture_model_label = qt.QLabel("Mixture model:")
            self.molecular = qt.QRadioButton("Molecular")
            self.molecular.setChecked(True) # Molecular mixing (more complex) by default
            self.areal = qt.QRadioButton("Areal")
            self.mixture_model = qt.QButtonGroup() # Setting up a group to ensure only one can be checked
            self.mixture_model.addButton(self.molecular)
            self.mixture_model.addButton(self.areal)
            
            # Set up the radio buttons to automatically update the graph
            self.mixture_model.buttonClicked.connect(self.update_graph)

            # The material table just replaces the old material dropdown menu. However, for the mixture model,
            # we need to make a new row by shifting everything down one.
            mix_row_idx = self.row_widget(self.comps[0][0]) + 1 # Get the row index after the single material dropdown
            self.mknewrow(mix_row_idx, 1)
            self.layout.addWidget(self.mixture_model_label, mix_row_idx, 0, 1, 1)
            self.layout.addWidget(self.molecular, mix_row_idx, 1, 1, 1)
            self.layout.addWidget(self.areal, mix_row_idx, 2, 1, 1)

            # Set up table. Header row is added automatically with setHorizontalHeaderLabels and need not be
            # included in initialization. StableTable ensures table is uninteractable to prevent segmentation
            # faults.
            self.mat_table = StableTable(1, 2)
            self.mat_table.setHorizontalHeaderLabels([self.comp_labels[0].text(), self.comp_labels[1].text()])
            # Allow header rows to expand evenly across full table on startup, removing blank space
            self.mat_table.horizontalHeader().setSectionResizeMode(qt.QHeaderView.ResizeMode.Stretch)
            # Add composition dropdown to left cell and v/v entry box to right cell
            self.mat_table.setCellWidget(0, 0, self.comps[0][0])
            self.mat_table.setCellWidget(0, 1, self.comps[0][1])
            # Stop the table background from ever accepting focus. This helps prevent segmentation faults.
            self.mat_table.setFocusPolicy(qc.Qt.FocusPolicy.NoFocus)

            # Stop the table from drawing selection boxes around cells. Prevents segmentation fault from
            # occurring if space between cells is clicked
            self.mat_table.setSelectionMode(qt.QAbstractItemView.SelectionMode.NoSelection)

            # Remove previous label and single dropdown and add table
            self.layout.removeWidget(self.comps[0][0])
            self.layout.removeWidget(self.comp_labels[0])
            self.layout.addWidget(self.mat_table, 7, 0, 1, 3)

        # Add the additional material whether table was just created or not
        # Ensure composition array is updated
        self.comps.append([qt.QComboBox(), qt.QLineEdit()])
        self.comps[-1][0].addItems(COMPS)
        self.comps[-1][0].setStyleSheet(BKSHEET)
        # Find the index of the composition in the dropdown directly above this one
        mat_idx = np.where(np.asarray(COMPS) == self.comps[-2][0].currentText())[0][0]
        # Select the next dropdown in the list
        self.comps[-1][0].setCurrentText(COMPS[(mat_idx + 1) % len(COMPS)])
        # After changing, connect widgets to update function
        self.comps[-1][0].currentTextChanged.connect(self.update_graph)
        self.comps[-1][1].returnPressed.connect(self.update_graph)
        self.comps[-1][1].setText('0.0') # Will not cause update because return needs to be pressed
        self.comps[-1][1].setPlaceholderText("Input volume fraction")
        # Added comps will match current state of previous entries
        self.comps[-1][1].setStyleSheet("border: 1px solid " + ("red;" if "red" in self.comps[0][1].styleSheet() else "black;"))

        # Add widgets to table by inserting row at end
        self.mat_table.insertRow(self.mat_table.rowCount())
        self.mat_table.setCellWidget(self.mat_table.rowCount() - 1, 0, self.comps[-1][0])
        self.mat_table.setCellWidget(self.mat_table.rowCount() - 1, 1, self.comps[-1][1])

    def update_materials(self):
        """
        Reads the current state of all material input fields to ensure proper formatting.

        :param self: MainWindow object
        :returns: String containing all errors with current material field input
        """
        self.load_label.setText("Updating...")
        fraction_sum = 0 # This will need to add to 1 across all materials
        err_string = ''
        empty_item_idxs = [] # Array to hold indices of input fields without valid values
        for idx, item in enumerate(self.comps):
            # First reset all boxes to black-bordered
            text = item[1].text()
            if text == '':
                empty_item_idxs.append(idx) # Append index if empty
            else:
                try:
                    float_val = float(text)
                    if float_val < 0: # Check to ensure the user didn't input a negative value
                        if 'zero' not in err_string:
                            err_string += 'Volume fractions must be greater than or equal to zero.'
                        self.comps[idx][1].setStyleSheet("border: 1px solid red;") # Turn erroneous box red
                    elif float_val == 0:
                        empty_item_idxs.append(idx) # Append index if some variation of 0
                    else:
                        fraction_sum += float_val # Add to total fraction if nonzero float
                except ValueError:
                    empty_item_idxs.append(idx) # Append index if invalid number
                    if len(err_string) == 0: # Ensures this error doesn't get added repeatedly if multiple fields are invalid
                        err_string += 'Volume fractions must be floats. '
                        self.comps[idx][1].setStyleSheet("border: 1px solid red;") # Turn erroneous box red

         # If there is only one empty/erroneous field and the summed v/v fractions fall short of 1, fill the 
         # empty v/v so that the fractions add to 1
        if len(empty_item_idxs) == 1 and fraction_sum <= 1:
            err_string = err_string.replace('Volume fractions must be floats. ', '')
            self.comps[empty_item_idxs[0]][1].setText(str(round(1 - fraction_sum, 10)))
            self.comps[idx][1].setStyleSheet("border: 1px solid black;") # Empty box now has valid value
            fraction_sum = 1

        # If there are too many empty fields and the fractions don't add to 1, add this error
        if fraction_sum != 1:
            err_string += 'Volume fractions must add to 1.'

        if len(err_string) > 0:
            if 'add to 1' in err_string:
                # Failure to add to 1 is a generic error, turn all material boxes red
                for i in range(len(self.comps)):
                    self.comps[i][1].setStyleSheet("border: 1px solid red;")
        else:
            # If there are too many literally empty fields, but the fractions do add to 1, replace empty fields
            # with 0.0, assuming that's what the user meant
            for i in range(len(self.comps)):
                if len(self.comps[i][1].text()) == 0:
                    self.comps[i][1].setText("0.0")

        return err_string

    def row_widget(self, widget):
        """
        Locates the row index of a given widget in a layout.

        :param self: MainWindow object
        :param widget: Widget to locate row position of in self.layout

        :returns: Integer row index of widget
        """
        for i in range(self.layout.count()):
            item = self.layout.itemAt(i) # Get each item in the layout
            # If item is a widget, get the widget
            if item and item.widget() and widget == item.widget():
                return self.layout.getItemPosition(i)[0] # Return row position of item

    def mknewrow(self, rowstart, rowshift):
        """
        Shifts all widgets at or below rowstart up or down by rowshift rows.

        :param self: MainWindow object containing the widgets
        :param rowstart: Integer index at or below which to modify widget positions
        :param rowshift: Integer indicating how many row indices to shift widgets and in which direction.
                         Positive shifts down, negative shifts up.
        """
        widgets_to_remove = [] # An array to hold all widgets to be modified
        for i in range(self.layout.count()):
            item = self.layout.itemAt(i) # Get each item in the layout
            # If item is a widget, get the widget
            if item and item.widget():
                widget = item.widget()
                row, col, rowSpan, colSpan = self.layout.getItemPosition(i) # Get position of item
                # Check if this widget is in the affected area
                if row >= rowstart and col < 3:
                    # Add to modifiable widgets
                    widgets_to_remove.append((widget, row, col, rowSpan, colSpan))
        
        # For each modifiable widget, remove it, then re-add it rowshift rows up/down
        for widget, row, col, rowSpan, colSpan in widgets_to_remove:
            self.layout.removeWidget(widget)
            self.layout.addWidget(widget, row + rowshift, col, rowSpan, colSpan) 

    def all_borders(self, color):
        """
        Sets all input box borders to a certain color, used in displaying/removing global errors

        :param self: MainWindow object
        :param color: String color to set box outlines to
        """
        self.tau_box.setStyleSheet(f"border: 1px solid {color};")
        self.input_box.setStyleSheet(f"border: 1px solid {color};")
        # Iterate through each composition box. Will work if only one.
        for i in range(len(self.comps)):
            self.comps[i][1].setStyleSheet(f"border: 1px solid {color};")

        # Turn size distribution boxes to the chosen color if present
        if not self.smin_box.isHidden():
            self.smin_box.setStyleSheet(f"border: 1px solid {color};")
            self.smax_box.setStyleSheet(f"border: 1px solid {color};")
            self.powlaw_box.setStyleSheet(f"border: 1px solid {color};")
            if not self.r_box.isHidden():
                self.r_box.setStyleSheet(f"border: 1px solid {color};")
            if not self.G_box.isHidden():
                self.G_box.setStyleSheet(f"border: 1px solid {color};")
            if not self.x0_box.isHidden():
                self.x0_box.setStyleSheet(f"border: 1px solid {color};")

    def rst_boxes(self, autozoom=False):
        """
        Sets the min and max values for X and Y in their respective input boxes according to the current window
        parameters.

        :param self: MainWindow object
        :param autozoom: Boolean, whether or not to reset zoom automatically 
        """
        # Reset all bound borders to erase previous errors
        self.xmin_box.setStyleSheet(f"border: 1px solid black;")
        self.xmax_box.setStyleSheet(f"border: 1px solid black;")
        self.ymin_box.setStyleSheet(f"border: 1px solid black;")
        self.ymax_box.setStyleSheet(f"border: 1px solid black;")

        if autozoom:
            # Check to ensure we are on the first plot and no instrument is selected
            if len(self.base_params) > 0 and self.base_params[-1][0] is None:
                self.graph_widget.setRange(xRange=[-10, 10], yRange=[-10, 10])
            else:
                self.graph_widget.autoRange() # Window auto fit for the new dataset

        # Get bounds of window
        bounds = self.graph_widget.viewRange()

        # Set x and y bound text boxes to reflect current bounds
        self.xmin_box.setText(str(bounds[0][0]))
        self.xmax_box.setText(str(bounds[0][1]))
        self.ymin_box.setText(str(bounds[1][0]))
        self.ymax_box.setText(str(bounds[1][1]))

    def update_bounds(self):
        """
        Updates the bounds of the graph widget based on user input values for xmin, xmax, ymin, and ymax.

        :param self: MainWindow object
        """
        # Reset all bound borders to erase previous errors
        self.xmin_box.setStyleSheet(f"border: 1px solid black;")
        self.xmax_box.setStyleSheet(f"border: 1px solid black;")
        self.ymin_box.setStyleSheet(f"border: 1px solid black;")
        self.ymax_box.setStyleSheet(f"border: 1px solid black;")

        err_string = '' # Error string for dynamic error updating

        # Parse each of the entry fields
        xmin = self.xmin_box.text()
        xmax = self.xmax_box.text()
        ymin = self.ymin_box.text()
        ymax = self.ymax_box.text()

        # Try to convert to floats and update error string if fail, also ensure maxes > mins
        try:
            xmin = float(xmin)
            if np.isnan(xmin):  # Check for NaN
                err_string += 'Min X must be a number. '
        except ValueError:
            err_string += 'Min X must be a float. '

        try:
            xmax = float(xmax)
            # Ensure xmin is a float before checking this error
            if np.isnan(xmax): # Check for NaN
                err_string += 'Max X must be a number. '
            elif 'Min X' not in err_string and xmax <= xmin:
                err_string += 'Max X must be greater than min X. '
        except ValueError:
            err_string += 'Max X must be a float. '

        try:
            ymin = float(ymin)
            if np.isnan(ymin):  # Check for NaN
                err_string += 'Min Y must be a number. '
        except ValueError:
            err_string += 'Min Y must be a float. '

        try:
            ymax = float(ymax)
            if np.isnan(ymax):  # Check for NaN
                err_string += 'Max Y must be a number. '
            elif 'Min Y' not in err_string and ymax <= ymin:
                err_string += 'Max Y must be greater than min Y. '
        except ValueError:
            err_string += 'Max Y must be a float. '

        if len(err_string) == 0:
            # Update range if no errors
            self.graph_widget.setRange(xRange=[xmin, xmax], yRange=[ymin, ymax])
        else:
            self.load_label.setText(err_string)
            # Parse error string to determine which boxes to turn red
            # If we have a min > max error it will turn both problematic boxes red!
            # Note that these errors are caught separately from input parameter errors. As such, a graph can still
            # update with these errors because they don't impede graph functionality, only rescaling
            if 'Min X' in err_string or 'min X' in err_string:
                self.xmin_box.setStyleSheet(f"border: 1px solid red;")
            if 'Max X' in err_string:
                self.xmax_box.setStyleSheet(f"border: 1px solid red;")
            if 'Min Y' in err_string or 'min Y' in err_string:
                self.ymin_box.setStyleSheet(f"border: 1px solid red;")
            if 'Max Y' in err_string:
                self.ymax_box.setStyleSheet(f"border: 1px solid red;")

    def change_distmodel(self):
        """
        Displays boxes for custom size distribution modification. Updates the graph if a non-custom distribution is selected.

        :param self: MainWindow object
        """
        # Check to see if the Custom option is selected and the widgets haven't been created yet
        # Henyey-Greenstein graphs cannot have custom size distributions
        if self.dist_model.currentText() == 'Custom' and self.smin_box.isHidden() and not self.fit_model.currentText() == 'Henyey-Greenstein':
            # Make some new rows and add the widgets
            size_params_row = self.row_widget(self.fit_label)
            self.mknewrow(size_params_row, 3)

            self.layout.addWidget(self.smin_label, size_params_row, 0, 1, 1)
            self.layout.addWidget(self.smin_box, size_params_row, 1, 1, 2)
            self.layout.addWidget(self.smax_label, size_params_row + 1, 0, 1, 1)
            self.layout.addWidget(self.smax_box, size_params_row + 1, 1, 1, 2)
            self.layout.addWidget(self.powlaw_label, size_params_row + 2, 0, 1, 1)
            self.layout.addWidget(self.powlaw_box, size_params_row + 2, 1, 1, 2)

            # Re-show the widgets
            self.smin_label.setVisible(True)
            self.smin_box.setVisible(True)
            self.smax_label.setVisible(True)
            self.smax_box.setVisible(True)
            self.powlaw_label.setVisible(True)
            self.powlaw_box.setVisible(True)
        else:
            if not self.smin_box.isHidden():
                # Shift rows up
                self.mknewrow(self.row_widget(self.smin_box) + 3, -3)
                # Remove widgets from display
                self.layout.removeWidget(self.smin_label)
                self.layout.removeWidget(self.smin_box)
                self.layout.removeWidget(self.smax_label)
                self.layout.removeWidget(self.smax_box)
                self.layout.removeWidget(self.powlaw_label)
                self.layout.removeWidget(self.powlaw_box)

                # Set to invisible so they actually remove
                self.smin_label.setVisible(False)
                self.smin_box.setVisible(False)
                self.smax_label.setVisible(False)
                self.smax_box.setVisible(False)
                self.powlaw_label.setVisible(False)
                self.powlaw_box.setVisible(False)
        # Update regardless of if boxes were removed or not
        # change_fitmodel updates off of change dist model because change_fitmodel does the work of checking if r,
        # G, and x0 should show in the GUI based on what the scattering theory allows
        self.change_fitmodel(update=False)

    def change_fitmodel(self, update=True):
        """
        Intercepts changes in scattering theory to update available options for custom size distribution parameters.

        :param self: MainWindow object
        :param update: Boolean, whether to call update_graph on finish. Is overridden to True if not a custom size
                       distribution.
        """
        self.list_updates() # Update all lists to contain all options available for the current fit model
        # Examine the current scattering theory to see if it allows for modification of G, r, or x0, or if the window button should be present
        fit = self.fit_model.currentText()

        # Remove the change window button if it shouldn't be there. Otherwise, add it after the fit model
        if 'Semi-Empirical' in fit or 'Pure' in fit or 'Diffraction' in fit:
            if self.window_button is None:
                self.window_button = qt.QPushButton("Open Resolution Settings")
                self.window_button.clicked.connect(self.change_window)
                window_button_row = self.row_widget(self.fit_label) + 1
                self.mknewrow(window_button_row, 1)
                self.layout.addWidget(self.window_button, window_button_row, 0, 1, 3)
        else:
            if not self.window_button is None:
                self.mknewrow(self.row_widget(self.window_button) + 1, -1)
                self.layout.removeWidget(self.window_button)
                self.window_button = None

        # First always remove G, r, and x0 boxes. Here we clear them rather than hide them.
        if not self.r_box.isHidden():
            self.mknewrow(self.row_widget(self.r_label) + 1, -1)
            self.layout.removeWidget(self.r_label)
            self.r_label.setVisible(False)
            self.layout.removeWidget(self.r_box)
            self.r_box.setVisible(False)
        if not self.G_box.isHidden():
            self.mknewrow(self.row_widget(self.G_label) + 1, -1)
            self.layout.removeWidget(self.G_label)
            self.G_label.setVisible(False)
            self.layout.removeWidget(self.G_box)
            self.G_box.setVisible(False)
        if not self.x0_box.isHidden():
            self.mknewrow(self.row_widget(self.x0_label) + 1, -1)
            self.layout.removeWidget(self.x0_label)
            self.x0_label.setVisible(False)
            self.layout.removeWidget(self.x0_box)
            self.x0_box.setVisible(False)
        # Re-add any boxes relevant to the scattering theory & size distribution
        if not self.smin_box.isHidden():
            if "Semi-Empirical" in fit:
                # Semi-Empirical Mie requires r, G, and x0
                # Add the row right after the size distribution boxes
                r_row = self.row_widget(self.powlaw_label) + 1
                self.mknewrow(r_row, 3)
                self.layout.addWidget(self.r_label, r_row, 0, 1, 1)
                self.layout.addWidget(self.r_box, r_row, 1, 1, 2)
                self.layout.addWidget(self.G_label, r_row + 1, 0, 1, 1)
                self.layout.addWidget(self.G_box, r_row + 1, 1, 1, 2)
                self.layout.addWidget(self.x0_label, r_row + 2, 0, 1, 1)
                self.layout.addWidget(self.x0_box, r_row + 2, 1, 1, 2)

                # Set fields to visible
                self.r_label.setVisible(True)
                self.r_box.setVisible(True)
                self.G_label.setVisible(True)
                self.G_box.setVisible(True)
                self.x0_label.setVisible(True)
                self.x0_box.setVisible(True)

            elif "Diffraction" in fit:
                # Mie Diffraction only requires r
                # Add the row right after the size distribution boxes
                r_row = self.row_widget(self.powlaw_label) + 1
                self.mknewrow(r_row, 1)
                self.layout.addWidget(self.r_label, r_row, 0, 1, 1)
                self.layout.addWidget(self.r_box, r_row, 1, 1, 2)

                # Set field to visible
                self.r_label.setVisible(True)
                self.r_box.setVisible(True)

            elif "Transmission" in fit:
                # Mie Transmission label only requires G
                # Add the row right after the size distribution boxes
                G_row = self.row_widget(self.powlaw_label) + 1
                self.mknewrow(G_row, 1)
                self.layout.addWidget(self.G_label, G_row, 0, 1, 1)
                self.layout.addWidget(self.G_box, G_row, 1, 1, 2)

                # Set field to visible
                self.G_label.setVisible(True)
                self.G_box.setVisible(True)

        # The update optional parameter is there to prevent update graph from getting called when the new sizedist
        # boxes are added, so the boxes don't instantly turn red. But if those boxes aren't there, we force
        # an update anyways.
        if update or not self.dist_model.currentText() == 'Custom':
            self.update_graph() # Lastly, update the graph

    def change_axis(self):
        """
        Changes the Y axis of the graph between reflectance and phase function (reflectance normalized to integrate to 1 over all solid angles).
        
        :param self: MainWindow object
        """
        self.list_updates() # Update displayed items in lists 
        if self.y_axis.currentText() == 'Reflectance':
            self.graph_widget.setLabel('left', 'Reflectance', units='I/F', color='k') # Units are I/F
            # Show tau again because reflectance depends on it
            if self.tau_box.isHidden():
                tau_row = self.row_widget(self.input_label) # Row at input label (input label will be shifted down one to make room)
                self.mknewrow(tau_row, 1)
                self.layout.addWidget(self.tau_label, tau_row, 0, 1, 1)
                self.layout.addWidget(self.tau_box, tau_row, 1, 1, 2)
                self.tau_label.setVisible(True)
                self.tau_box.setVisible(True)

        elif self.y_axis.currentText() == 'Phase Function':
            self.graph_widget.setLabel('left', 'Phase Function', units='dim', color='k') # Units are dimensionless
            # Hide tau, phase function does not depend on it
            if not self.tau_box.isHidden():
                tau_row = self.row_widget(self.tau_label)
                self.mknewrow(tau_row + 1, -1)
                self.layout.removeWidget(self.tau_label)
                self.layout.removeWidget(self.tau_box)
                self.tau_label.setVisible(False)
                self.tau_box.setVisible(False)

        elif self.y_axis.currentText() == 'Intensity':
            self.graph_widget.setLabel('left', 'Intensity', units='dim', color='k') # Units are dimensionless
            # Hide tau, intensity does not depend on it
            if not self.tau_box.isHidden():
                tau_row = self.row_widget(self.tau_label)
                self.mknewrow(tau_row + 1, -1)
                self.layout.removeWidget(self.tau_label)
                self.layout.removeWidget(self.tau_box)
                self.tau_label.setVisible(False)
                self.tau_box.setVisible(False)

        elif self.y_axis.currentText() == 'Albedo': # Tau and Henyey-Greenstein will get removed elsewhere
            self.graph_widget.setLabel('left', 'Albedo', units='dim', color='k') # Units are dimensionless

        # Check for x-axis changes. Really only x-axis label gets updated here. This will matter more when output_graph gets called.
        if self.x_axis.currentText() == 'Phase Angle':
            self.graph_widget.setLabel('bottom', 'Phase Angle', units='°', color='k') # Units are degrees
        elif self.x_axis.currentText() == 'Scattering Angle':
            self.graph_widget.setLabel('bottom', 'Scattering Angle', units='°', color='k') # Units are degrees
        elif self.x_axis.currentText() == 'Wavelength':
            self.graph_widget.setLabel('bottom', 'Wavelength', units='μm', color='k') # Units are microns

        self.clear_graph() # Clear the graph to avoid any issues with previous data

    def change_input(self, update=False):
        """
        Triggers when the input label is changed, updating the default text of an empty input box. Updates the graph if requested.

        :param self: MainWindow object
        :param update: Boolean. True calls the the update_graph function when finished.
        """
        if 'Phase Angle' in self.input_label.currentText():
            self.input_box.setPlaceholderText("Input phase angle in degrees")
        elif 'Scattering Angle' in self.input_label.currentText():
            self.input_box.setPlaceholderText("Input scattering angle in degrees")
        elif 'Effective Grain Size' in self.input_label.currentText():
            self.input_box.setPlaceholderText("Input effective grain size in microns")
        elif 'Wavelength' in self.input_label.currentText():
            self.input_box.setPlaceholderText("Input wavelength in microns")

        if update:
            self.update_graph()

    def change_window(self):
        """
        Cycles between the data input window and the resolution settings window when the button is available.

        :param self: MainWindow object
        """
        self.stack.setCurrentIndex((self.stack.currentIndex() + 1) % 2) # Show data entry screen by default
        if self.stack.currentIndex() == 0:
            # Add the load label in its proper place
            self.layout.addWidget(self.load_label, self.row_widget(self.y_scale_label) + 1, 0, 1, 3)
            # Add the progress bar in its proper place
            self.layout.addWidget(self.prog, self.row_widget(self.y_scale_label) + 1, 1, 1, 2)
            # Add the cancel button in its proper place
            self.layout.addWidget(self.cancel, self.row_widget(self.y_scale_label) + 2, 1, 1, 2)
            # We add in the graph widgets no matter screen which is being used
            self.layout.addWidget(self.graph_widget, 0, 3, 24, 24)
            self.layout.addWidget(self.home_button, 1, 4, 1, 1)
            self.layout.addWidget(self.xmin_label, 1, 23, 1, 1)
            self.layout.addWidget(self.xmin_box, 1, 24, 1, 1)
            self.layout.addWidget(self.xmax_label, 1, 25, 1, 1)
            self.layout.addWidget(self.xmax_box, 1, 26, 1, 1)
            self.layout.addWidget(self.ymin_label, 2, 23, 1, 1)
            self.layout.addWidget(self.ymin_box, 2, 24, 1, 1)
            self.layout.addWidget(self.ymax_label, 2, 25, 1, 1)
            self.layout.addWidget(self.ymax_box, 2, 26, 1, 1)
        elif self.stack.currentIndex() == 1:
            # Add the load label in its proper place
            self.settings_layout.addWidget(self.load_label, 3, 0, 1, 3)
            # Add the progress bar in its proper place
            self.settings_layout.addWidget(self.prog, 3, 1, 1, 2)
            # Add the cancel button in its proper place
            self.layout.addWidget(self.cancel, 4, 1, 1, 2)
            # We add in the graph widgets no matter screen which is being used
            self.settings_layout.addWidget(self.graph_widget, 0, 3, 24, 24)
            self.settings_layout.addWidget(self.home_button, 1, 4, 1, 1)
            self.settings_layout.addWidget(self.xmin_label, 1, 23, 1, 1)
            self.settings_layout.addWidget(self.xmin_box, 1, 24, 1, 1)
            self.settings_layout.addWidget(self.xmax_label, 1, 25, 1, 1)
            self.settings_layout.addWidget(self.xmax_box, 1, 26, 1, 1)
            self.settings_layout.addWidget(self.ymin_label, 2, 23, 1, 1)
            self.settings_layout.addWidget(self.ymin_box, 2, 24, 1, 1)
            self.settings_layout.addWidget(self.ymax_label, 2, 25, 1, 1)
            self.settings_layout.addWidget(self.ymax_box, 2, 26, 1, 1)
    
    def update_spacing(self):
        """
        Handles updates to the spacing mode of the graph by changing the bounds of the slider.

        :param self: MainWindow object
        """
        # Logarithmic requires the slider to set the base
        if self.spacing.currentText() == 'Logarithmic':
            self.dist_slider.setMinimum(101)
            self.dist_slider.setMaximum(400)
            self.dist_slider.setValue(200)
            # Connect to update slider, except set dec to true
            self.dist_slider.valueChanged.connect(lambda x: self.update_slider(x, dec=True))
            # Update when finished
            self.update_slider(self.dist_slider.value(), dec=True)
        elif self.spacing.currentText() == 'Linear':
            self.dist_slider.setMinimum(2)
            self.dist_slider.setMaximum(100)
            self.dist_slider.setValue(vars.NSIZE[1])
            # Connect to default, dec as false
            self.dist_slider.valueChanged.connect(self.update_slider)
            # Update when finished
            self.update_slider(self.dist_slider.value())
        self.update_graph() # Call a graph update upon changing spacing type

    def update_slider(self, value, dec=False):
        """
        Handles changes in the size distribution slider, updating the displayed value and warning that the graph has not updated.

        :param self: MainWindow object
        :param value: The integer current position of the slider
        :param dec: Whether to display the number divided by 100. Necessary to allow decimal values.
        """
        self.slider_label.setText(f"{'Number of Sizes' if self.spacing.currentText() == 'Linear' else 'Log Base'}: {value if dec == False else value/100}")
        self.update_graph() # Call graph update. This is fine to do because even if the user cranks the resolution to something massive, they can still cancel the update.

    def set_enabled(self, enable):
        """
        Disables or enables all widgets in self.layout based on the boolean value of enable.

        :param self: MainWindow object
        :param enable: Boolean. True to enable all widgets, False to disable them.
        """

        for i in range(self.layout.count()):
            item = self.layout.itemAt(i) # Get each item in the layout
            # If item is a widget, get the widget
            if item and item.widget():
                item.widget().setEnabled(enable)

    def update_process(self):
        """
        Updates the progress bar while the graph is updating, and plots the new dataset when finished.
        
        :param self: MainWindow object
        """
        try:
            value = self.result_queue.get_nowait() # Get whatever is in the result queue at this moment
        except Exception:
            value = None # If the queue is empty
        if isinstance(value, float): # A float is returned, that is a progress update
            # Show progress bar if in the process of updating
            if self.prog.isHidden():
                self.prog.setVisible(True)
                self.cancel.setVisible(True)
            self.prog.setValue(int(value*100)) # Output result to progress bar as a percentage
        elif isinstance(value, dict): # A dictionary has been returned, time to update graph
            self.monitor_timer.stop() # Stop the timer if process is not running
            if self.proc:
                self.proc.join()
                self.proc = None

            # Grab results from the queue
            if value is not None:
                self.x_data = value["x_data"]
                self.y_data = value["y_data"]
                
                # Plot the dataset if not in the graph, else change the current unsaved graph
                if self.ifs not in self.graph_widget.plotItem.items:
                    self.ifs = self.graph_widget.plot(self.x_data, self.y_data, pen=self.plt_pen, name=(f"{self.param_str}{' μm' if (self.graphmode == 1 or self.graphmode == 2) else '°'}" if self.multiple > 0 else None))
                else:
                    self.ifs.setData(self.x_data, self.y_data) # self.ifs variable allows for updating of the temporary graph

                # Store the newly plotted parameters in base_params
                # Note if the last graph was loaded, the last element was popped; as a result, we have to re-add it
                if self.last_graph_loaded:
                    self.base_params.append(self.latest_params)
                else:
                    self.base_params[-1] = self.latest_params
                
                # If the current plot index was in the loaded data dictionary, remove it
                if str(self.multiple) in self.loaded_idxs.keys():
                    del self.loaded_idxs[str(self.multiple)]
                self.last_graph_loaded = False # Indicate the most recent plot is not loaded data

                # Update the legend if multiple graphs
                if self.multiple > 0:
                    self.legend.removeItem(self.ifs) # Need to do remove and add because there's no setter function
                    self.legend.addItem(self.ifs, '') # Add a blank placeholder for self.ifs so legend setter function has a full array to loop through
                    self.generate_legend(self.temp_legend_idxs) # Update the full legend. The last element is our new temporary plot

                self.update_title(self.temp_legend_idxs) # Update the title showing variance across all current plots

                # Clean up process, hide load label/cancel button
                self.proc = None
                if not self.prog.isHidden():
                    self.prog.setVisible(False)
                    self.cancel.setVisible(False)
                    self.prog.setValue(0)

                self.rst_boxes(autozoom=True) # Reset zoom. Note how we do it AFTER we update base params, since rst_boxes default zooms if "None" is detected as the instrument. So we have to make sure that doesn't happen.
                self.all_borders('black') # Remove all outstanding errors
                self.load_label.setText("Up to date")
            else:
                self.load_label.setText("Computation failed. Please try again.")
        
    def kill_process(self, rst_gui=True, rst_proc=False):
        """
        Stops the update graph multiprocess, stopping the timer and hiding the progress bar. self.proc must not be None.

        :param self: MainWindow object
        :param rst_gui: Boolean. True resets GUI to the parameters present before the process started. False 
                        keeps GUI as-is.
        :param rst_proc: Boolean. False hides the progress bar and cancel button, indicating the process is done.
                         True does not hide them but resets them, assuming the process will be restarted.
        """
        # Stop timer if running
        self.monitor_timer.stop()
        self.proc.terminate() # Send the SIGTERM signal to the process
        self.proc.join()
        # Remove progress bar and cancel buttons if visible
        self.prog.setValue(0) # First set to 0
        if not rst_proc and not self.prog.isHidden():
            self.prog.setVisible(False)
            self.cancel.setVisible(False)
        self.proc = None
        self.result_queue = None

        if rst_gui:
            # Reset the GUI to the state it was previously
            # To do that, we must first disable all fields to prevent updates
            self.block_signals(True)

            # Then we iterate through the most recent parameters to make sure all NaNs are replaced with empty strings
            for i in range(len(self.most_recent_update)):
                if self.most_recent_update[i] == 'NaN':
                    self.most_recent_update[i] = ''
                # Note that any element in most_recent_update that is an array/dictionary will never have a NaN element, so we don't need to check inside them
            
            # If base_params has a None in its latest entry, don't bother with replacing it. The graph for that entry still has not been displayed, so the last entry of base_params can have Nones
            # We do want to update the GUI inputs, however
            if len(self.base_params) > 0 and self.base_params[-1][0] is not None:
                self.base_params[-1] = self.most_recent_update

            self.instrument.setCurrentText(self.most_recent_update[0])
            # Loop over materials and reset those
            mat_idx = 0
            for mat in self.most_recent_update[1].keys():
                self.comps[mat_idx][0].setCurrentText(mat)
                self.comps[mat_idx][1].setText(str(self.most_recent_update[1][mat]))
                mat_idx += 1
            # If multiple materials, reset mixture model
            if len(self.comps) > 1:
                if self.most_recent_update[2] == 'Areal':
                    self.areal.setChecked(True)
                elif self.most_recent_update[2] == 'Molecular':
                    self.molecular.setChecked(True)

            # If the size distribution is not custom, check for named size distributions
            # Note how we check the CURRENT state of the size distribution box, rather than the previous state
            # This means that if the custom box is selected, and a named distribution is found, the custom boxes will get filled with that distribution's parameters, rather than that distribution's name showing up in the sizedist box
            if self.dist_model.currentText() != 'Custom':
                for model in vars.SIZEDISTS.keys():
                    if [self.most_recent_update[3], self.most_recent_update[4], self.most_recent_update[5], self.most_recent_update[6], self.most_recent_update[7], self.most_recent_update[8]] == vars.SIZEDISTS[model][0]:
                        self.dist_model.setCurrentText(model)
                        break
            else:
                self.smin_box.setText(str(self.most_recent_update[3]))
                self.smax_box.setText(str(self.most_recent_update[4]))
                self.powlaw_box.setText(str(self.most_recent_update[5]))
                self.r_box.setText(str(self.most_recent_update[6]))
                self.G_box.setText(str(self.most_recent_update[7]))
                self.x0_box.setText(str(self.most_recent_update[8]))

            self.fit_model.setCurrentText(self.most_recent_update[9])
            self.spacing.setCurrentText(self.most_recent_update[10][0])
            self.dist_slider.setValue(self.most_recent_update[10][1])
            self.tau_box.setText(str(self.most_recent_update[11]))
            self.input_label.setCurrentText(self.most_recent_update[12][1])
            self.input_box.setText(str(self.most_recent_update[12][0]))
            
            # --- ENABLE FIELDS ---
            # Now we re-enable all fields
            self.block_signals(False)
            
            self.all_borders('black') # Remove all outstanding errors
            self.load_label.setText("Up to date") # Update load label when finished

    def list_updates(self):
        """
        Updates the fit_model and y_axis lists based on their values to prevent invalid parameter combinations.

        :param self: MainWindow object
        """

        # We must disable ALL UPDATES to ensure we don't trigger recursion loops
        self.block_signals(True)
        
        # Grab the fit model
        fit = self.fit_model.currentText()
        # First update what values are allowed to be displayed in the y-axis box
        if self.graphmode == 1: # For graph mode 1, only albedo is allowed
            if self.y_axis.findText("Albedo") < 0:
                self.y_axis.insertItem(0, "Albedo")
            self.y_axis.removeItem(self.y_axis.findText("Reflectance"))
            self.y_axis.removeItem(self.y_axis.findText("Phase Function"))
            self.y_axis.removeItem(self.y_axis.findText("Intensity"))
        else:
            # Remove albedo from the y-axis dropdown, since it is only supported in graphmode 1
            self.y_axis.removeItem(self.y_axis.findText("Albedo"))
            if 'Semi-Empirical' in fit or 'Diffraction' in fit:
                # Remove intensity from the y-axis dropdown list since it is not supported in semi-empirical Mie/Mie diffraction plots
                # If it's in there, it will switch to reflectance
                # If it's not, the remove will fail silently
                self.y_axis.removeItem(self.y_axis.findText("Intensity"))
                # Reflectance and phase function are supported, so they will be added if not present
                if self.y_axis.findText("Reflectance") < 0:
                    self.y_axis.insertItem(0, "Reflectance")
                if self.y_axis.findText("Phase Function") < 0:
                    self.y_axis.insertItem(self.y_axis.count(), "Phase Function")
            elif 'Pure' in fit or 'Reflection' in fit or 'Transmission' in fit:
                # Reflectance, phase function, and intensity are supported for Pure Mie, Mie External Reflection, and Mie Transmission, so they will be added if not present
                if self.y_axis.findText("Reflectance") < 0:
                    self.y_axis.insertItem(0, "Reflectance")
                if self.y_axis.findText("Phase Function") < 0:
                    self.y_axis.insertItem(self.y_axis.count(), "Phase Function")
                if self.y_axis.findText("Intensity") < 0:
                    self.y_axis.insertItem(1, "Intensity")
            elif fit == 'Henyey-Greenstein':
                # Everything but reflectance
                self.y_axis.removeItem(self.y_axis.findText("Intensity"))
                self.y_axis.removeItem(self.y_axis.findText("Phase Function"))
                # Reflectance is supported, so it will be added if not present
                if self.y_axis.findText("Reflectance") < 0:
                    self.y_axis.insertItem(0, "Reflectance")

        # Grab the y-axis value
        y_val = self.y_axis.currentText()
        # Then update the fit model list based on the y-axis
        if self.graphmode == 0: # Henyey-Greenstein is not available in reflectance vs. wavelength plots
            self.fit_model.removeItem(self.fit_model.findText("Henyey-Greenstein"))
        if y_val == 'Reflectance':
            # Henyey-Greenstein, Semi-Empirical Mie, and Mie Diffraction get added because Reflectance supports them
            if self.fit_model.findText("Semi-Empirical Mie") < 0:
                self.fit_model.insertItem(0, "Semi-Empirical Mie")
            if self.fit_model.findText("Mie Diffraction") < 0:    
                self.fit_model.insertItem(2, "Mie Diffraction")
            if self.graphmode != 0 and self.fit_model.findText("Henyey-Greenstein") < 0:
                self.fit_model.insertItem(self.fit_model.count(), "Henyey-Greenstein") # Add this one last
        elif y_val == 'Phase Function':
            # Henyey-Greenstein is not supported for phase function plots
            self.fit_model.removeItem(self.fit_model.findText("Henyey-Greenstein"))
            # Semi-Empirical Mie and Mie Diffraction get added because Phase Function supports them
            if self.fit_model.findText("Semi-Empirical Mie") < 0:
                self.fit_model.insertItem(0, "Semi-Empirical Mie") # Not possible to have multiple, since text has to change to trigger this
            if self.fit_model.findText("Mie Diffraction") < 0:
                self.fit_model.insertItem(2, "Mie Diffraction")
        elif y_val == 'Intensity':
            # Remove Henyey-Greenstein, Semi-Empirical Mie, and Mie Diffraction from the scattering theory dropdown list since they are not supported in intensity plots
            # If they're in there, the list will switch to Pure Mie
            # If they're not, the removes will fail silently
            self.fit_model.removeItem(self.fit_model.findText("Henyey-Greenstein"))
            self.fit_model.removeItem(self.fit_model.findText("Semi-Empirical Mie"))
            self.fit_model.removeItem(self.fit_model.findText("Mie Diffraction"))

        # Then update the x-axis box
        if self.graphmode == 0 or self.graphmode == 1:
            # Remove scattering and phase angles from x-axis dropdown and add wavelength
            self.x_axis.removeItem(self.x_axis.findText("Phase Angle"))
            self.x_axis.removeItem(self.x_axis.findText("Scattering Angle"))
            if self.x_axis.findText("Wavelength") < 0:
                self.x_axis.addItem("Wavelength")
        elif self.graphmode == 2:
            # Remove wavelength from x-axis dropdown and add scattering and phase angles
            self.x_axis.removeItem(self.x_axis.findText("Wavelength"))
            if self.x_axis.findText("Scattering Angle") < 0:
                self.x_axis.addItem("Scattering Angle")
            if self.x_axis.findText("Phase Angle") < 0:
                self.x_axis.addItem("Phase Angle")

        # Finally, update the input parameter options
        if self.graphmode == 0:
            # Reflectance vs. wavelength plots require phase/scattering angle inputs
            self.input_label.removeItem(self.input_label.findText("Wavelength (μm):"))
            if self.input_label.findText("Scattering Angle (°):") < 0:
                self.input_label.addItem("Scattering Angle (°):")
            if self.input_label.findText("Phase Angle (°):") < 0:
                self.input_label.addItem("Phase Angle (°):")
        elif self.graphmode == 1:
            # Albedo vs. wavelength plots require effective grain size input
            self.input_label.removeItem(self.input_label.findText("Phase Angle (°):"))
            self.input_label.removeItem(self.input_label.findText("Scattering Angle (°):"))
            if self.input_label.findText("Effective Grain Size (μm):") < 0:
                self.input_label.addItem("Effective Grain Size (μm):")
        elif self.graphmode == 2:
            # Reflectance vs. angle plots require wavelength input
            self.input_label.removeItem(self.input_label.findText("Effective Grain Size (μm):"))
            if self.input_label.findText("Wavelength (μm):") < 0:
                self.input_label.addItem("Wavelength (μm):")

        # Now we reenable updates
        self.block_signals(False)

    def block_signals(self, block):
        """
        Blocks/unblocks signals from every changeable input field in the GUI, preventing them from
        triggering updates.
        
        :param self: MainWindow object
        :param block: Boolean. True blocks signals, False unblocks them.
        """
        # Block signals from all available GUI fields
        self.instrument.blockSignals(block)
        # Iterate through materials table and block all fields
        for mat in self.comps:
            mat[0].blockSignals(block)
            mat[1].blockSignals(block)
        # Block mixture model if there is a materials table
        if self.mixture_model is not None:
            self.mixture_model.blockSignals(block)
        self.dist_model.blockSignals(block)
        self.smin_box.blockSignals(block)
        self.smax_box.blockSignals(block)
        self.powlaw_box.blockSignals(block)
        self.r_box.blockSignals(block)
        self.G_box.blockSignals(block)
        self.x0_box.blockSignals(block)
        self.fit_model.blockSignals(block)
        self.spacing.blockSignals(block)
        self.dist_slider.blockSignals(block)
        self.tau_box.blockSignals(block)
        self.input_label.blockSignals(block)
        self.input_box.blockSignals(block)
        self.x_axis.blockSignals(block)
        self.y_axis.blockSignals(block)
        
    def load_data(self):
        """
        Prompts the user to select an import file and loads its data into the temporary plot. Displays an error if formatting is incorrect.
        
        :param self: MainWindow object
        """
        self.all_borders("black") # We need not worry about errors in the GUI for the time being, as we are loading external data
        # Get only csv files, as those usually contain data
        file_path, _ = qt.QFileDialog.getOpenFileName(
            self, 
            "Open File", 
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), # We will default grabbing the file from the PARENT directory, not gui.py's directory
            "All Files (*);;CSV Files (*.csv)"
        )
        if len(file_path) > 0: # Check to make sure a path was actually entered
            # Kill the update process because we are plotting new data
            if self.proc is not None:
                self.kill_process()

            # Get the base file name
            file_name = os.path.splitext(os.path.basename(file_path))[0]

            # Check to ensure this file's data doesn't already exist in the graph, either as the latest plot or a previous one
            if str(self.multiple) in self.loaded_idxs.keys() and self.loaded_idxs[str(self.multiple)] == file_name:
                self.load_label.setText("Data is identical to previous graph")
            elif file_name in self.loaded_idxs.values():
                self.load_label.setText("Data already exists in graph")
            else:
                # Load the data, if possible
                xarr, yarr, errcode = utils.get_data(file_path, target_x=self.x_axis.currentText(), target_y=self.y_axis.currentText())

                # Catch any errors that may have arisen and print them out
                if len(xarr) == 0:
                    if errcode == 0: # This means that no data was found, period
                        self.load_label.setText(f"No data found in {file_path}")
                    elif errcode == 1: # This means that the right type of x data wasn't found
                        self.load_label.setText(f"No {self.x_axis.currentText().lower()} data found in {file_path}")
                    elif errcode == 2: # This means that the right type of y data wasn't found
                        self.load_label.setText(f"No {self.y_axis.currentText().lower()} data found in {file_path}")
                    elif errcode == 3: # This means invalid values were found in the x array
                        self.load_label.setText(f"Invalid values found in {self.x_axis.currentText().lower()} column")
                    elif errcode == 4: # This means invalid values were found in the y array
                        self.load_label.setText(f"Invalid values found in {self.y_axis.currentText().lower()} column")
                    elif errcode == 5: # This means the x and y arrays weren't the same length
                        self.load_label.setText(f"{self.x_axis.currentText().lower().capitalize()} column and {self.y_axis.currentText().lower()} column have different lengths")
                else:
                    # Set x and y data to new array
                    self.x_data = xarr
                    self.y_data = yarr

                    # Reuse the same pen color for all updates to the current plot
                    self.plt_pen.setColor(pg.mkColor(COLOR_ARR[self.last_color_idx]))

                    # Base parameters will have no array added to it
                    # Plot the dataset if not in the graph, else change the current unsaved graph
                    if self.ifs not in self.graph_widget.plotItem.items:
                        self.ifs = self.graph_widget.plot(self.x_data, self.y_data, pen=self.plt_pen)
                    else:
                        self.ifs.setData(self.x_data, self.y_data) # self.ifs variable allows for updating of the temporary graph

                    # Clear the input box to indicate that the current GUI does not reflect what is shown in the graph
                    self.input_box.setText('')
                    self.most_recent_update[-1][0] = '' # Clear the input box in the most recent update array as well, so that if the user loads a plot and then decides to cancel their next one the GUI will revert to having the input box blank again

                    # If the last graph wasn't loaded, pop the last element of base_params because the latest plot no longer applies to base_params
                    if not self.last_graph_loaded:
                        self.base_params.pop(-1)

                    # Add the current plot index to the array indicating which plots are loaded data
                    # If it's already in there, it will be overwritten
                    self.loaded_idxs[str(self.multiple)] = file_name
                    self.last_graph_loaded = True # Indicate the most recent plot is loaded data

                    self.temp_legend_idxs = copy.deepcopy(self.legend_idxs) # This is because we have eliminated the most recent entry, so no new parameters are varying
                    # Update the legend if multiple graphs
                    if self.multiple > 0:
                        self.legend.removeItem(self.ifs) # Need to do remove and add because there's no setter function
                        self.legend.addItem(self.ifs, '') # Add a blank placeholder for self.ifs so legend setter function has a full array to loop through
                        self.generate_legend(self.temp_legend_idxs) # Update the full legend. The last element is our new temporary plot
            
                    self.update_title(self.temp_legend_idxs) # Update the title showing variance across all current plots
                    self.rst_boxes(autozoom=True) # Reset zoom

                    self.all_borders('black')
                    self.load_label.setText("Up to date")

class StableTable(qt.QTableWidget):
    """
    Sets up a safe extension of PyQt5's QTableWidget that disables all potentially problematic keyboard
    interactions while within the table, intercepting any events that could lead to segmentation faults.
    """
    def keyPressEvent(self, event):
        """
        Overrides QTableWidget's normal keyPressEvent to intercept key presses and block navigation keys that cause segmentation faults.

        :param self: QTableWidget object
        :param event: Key press event to intercept
        """
        # List of keys that cause the segmentation fault
        crash_keys = [
            qc.Qt.Key.Key_Tab,
            qc.Qt.Key.Key_Up, 
            qc.Qt.Key.Key_Down,
            qc.Qt.Key.Key_PageUp, 
            qc.Qt.Key.Key_PageDown,
            qc.Qt.Key.Key_Left,
            qc.Qt.Key.Key_Right
        ]
        
        # If the key is one of the dangerous navigation keys, swallow it
        if event.key() in crash_keys:
            event.accept() # Skip dangerous processing
            return
            
        # Otherwise, let the table process the key normally (like typing numbers)
        super().keyPressEvent(event)
