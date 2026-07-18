"""
A script to set up the main GUI for plotting plume spectra dynamically while varying various parameters. Takes
input size distribution, instrument wavelength range, optical depth, and material composition, among others, and
creates plots for each combination of parameters, allowing for multiple plots on the same window and switching 
between plotting reflectance vs. wavelength, surface albedo vs. wavelength, and reflectance vs. scattering angle.

Authors: Parker A. Zaemann, Matthew M. Hedman
Date: 17 July 2026
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
import numpy as np
import multiprocessing as mp

COLOR_ARR = ['red', 'magenta', 'orange', 'green', 'blue', 'violet'] # Update_graph creates a temporary graph that cycles through these colors as they are added to the window
COMPS = utils.get_available_materials() # Call function to parse data directory to get all available materials to display
# Set up a queue to get the information from the subprocess when done
result_queue = mp.Queue()

def output_graph(graphmode, sensor, comps, mixmodel, datamodel, fitmodel, param, tau, nsizes, wavelbounds, output):
    """
    A function that takes input scattering data as might be observed by one of Europa Clipper's instruments,
    as well as additional data about the fit and the plot type desired, and outputs the corresponding x and y
    arrays for a plot.

    :param graphmode: Integer, indicates whether to output reflectance vs. wavelength (0), surface reflectance
                    vs. wavelength (1), or reflectance vs. scattering angle (2)
    :param sensor: String sensor from which to read wavelength range
    :param comps: Dictionary with keys as materials and values as volume fractions of materials
    :param mixmodel: String mixture model to use, either 'Areal' or 'Molecular'
    :param datamodel: Array of size distribution strings [min particle size, max particle size, power law,
                    r, G, x0]
    :param fitmodel: String for scattering theory to use, either 'Semi-Empirical Mie', 'Pure Mie',
                    'Mie Scattering', 'Mie External Reflection', or 'Mie Transmission'.
    :param param: Fixed parameter to base reflectances on. Int scattering angle in degrees if graphmode = 0, 
                float effective grain size in microns if graphmode = 1, float wavelength in microns if 
                graphmode = 2
    :param tau: Float optical depth (unitless)
    :param nsizes: List [size mode, integer number of sizes to use in size distribution]
    :param wavelbounds: Tuple (float, float) wavelength bounds in microns as established by the composition, 
                        representing data available for index of refraction interpolation
    :param output: List of strings indicating the type of output to return. First element is x axis, either
                'Phase Angle', 'Scattering Angle', or 'Wavelength'. Second element is y axis, either 
                'Reflectance' or 'Phase Function'.
    :returns: List containing x data values (wavelengths in microns, angles in degrees)
            List containing float reflectance/albedo/phase function data values (unitless)
    """

    # Try to extract r, G, and tau from the datamodel array; if they are NaNs, set them to their default values
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

    # nsizes also needs to be checked for nans as it only applies to semi-empirical Mie, pure Mie, and Mie scattering
    if nsizes == ['NaN', 'NaN']:
        nsizes = vars.NSIZE # It won't be used anyway

    # If surface reflectance mode, create an array of wavelengths and calculate reflectances using surface formulas
    if graphmode == 1:
        wavel_range = vars.INSTRUMENTS[sensor] # Get range of all available wavelengths in the sensor
        xarr = np.arange(round(max(wavelbounds[0], wavel_range[0]), 5), round(min(wavelbounds[1], wavel_range[1]), 5) + 1e-20, 0.01) # Assume a 10 nm resolution. Get the most limiting bounds
        reflectances = utils.fresn_surface_reflectances(xarr, param, comps=comps, mixmodel=mixmodel)
    # If reflectance vs. angle mode, create an array of all colatitude angles and call the semi-empirical Mie theory
    elif graphmode == 2:
        xarr = np.arange(0, 181, 1) if output[0] == 'Scattering Angle' else np.arange(180, -1, -1) # Declare array of angles 0 to 180 degrees (or 180 to 0 degrees if phase)
        # Get reflectances. Programmed so that angles start, stop, and step the same way as xarr
        reflectances = utils.angle_mie_reflectances(float(datamodel[0]), float(datamodel[1]), float(datamodel[2]), r=r, G=G, x0=x0, theta_min=0, theta_max=180, wavels=[param], comps=comps, mixmodel=mixmodel, nsize=nsizes, tau=tau, section=fitmodel, output=output[1])[0][0]

        # For Henyey-Greenstein, curve fitting is needed to determine global scale factor
        if fitmodel == 'Henyey-Greenstein':
            # Iterate through given size distributions and see if input size distribution matches.
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
    # If reflectances vs. wavelengths mode, get the wavelength range from the sensor and use that in the semi-empirical Mie methods
    elif graphmode == 0:
        wavel_range = vars.INSTRUMENTS[sensor] # Plot over range of all available wavelengths in the sensor
        xarr = np.arange(round(max(wavelbounds[0], wavel_range[0]), 5), round(min(wavelbounds[1], wavel_range[1]), 5) + 1e-20, 0.01) # Assume a 10 nm resolution. Get the most limiting bounds
        reflectances = utils.angle_mie_reflectances(float(datamodel[0]), float(datamodel[1]), float(datamodel[2]), r=r, G=G, x0=x0, theta_min=param, theta_max=param, wavels=xarr, comps=comps, mixmodel=mixmodel, nsize=nsizes, tau=tau, section=fitmodel, output=output[1])[0]
        # angle_mie_reflectances returns a weird column array here, with shape [[element1], [element2], [element3]]
        # Need to make array not as deep
        reflectances = [element[0] for element in reflectances]

    # Return final arrays by putting them in queue
    result_queue.put({
        "x_data": xarr.tolist(),
        "y_data": reflectances
    })

class MainWindow(qt.QMainWindow):
    """
    Sets up the main GUI for plotting plume particle scattering data, containing widgets to modify core parameters,
    adjust graph settings, and dynamically update results. Used by analysis_main.py in startup. Extends PyQt6's
    QMainWindow.
    """

    def __init__(self):
        # Initialize the parent class (QMainWindow)
        super().__init__()

        # --- MAIN WINDOW PROPERTIES ---
        # Title and sizing
        self.setWindowTitle("Scattera Clipper Instrument Spectrum Prediction")
        self.resize(1500, 1100)

        # Create the data input screen widget
        self.data_input_widget = qt.QWidget()
        
        # We will use a Vertical Box Layout (QVBoxLayout) to stack widgets top-to-bottom
        self.layout = qt.QGridLayout()
        self.layout.setContentsMargins(30, 30, 30, 30) # Making sure things don't overlap each other
        self.layout.setSpacing(20)
        self.data_input_widget.setLayout(self.layout)

        # --- WIDGETS ---
        # Push button to toggle between reflectance vs wavelength graphs and reflectance vs scattering angle graphs
        # Indicates what type of graph is being shown. 0 is IFs vs wavelengths, 1 is surface albedos vs wavelengths, 2 is IFs vs scattering angles
        self.graphmode = 0
        self.graph_mode_button = qt.QPushButton("Reflectance vs. Wavelength")

        # Dropdown list for instruments on Europa Clipper
        self.instrument_label = qt.QLabel("Instrument:")
        self.instrument = qt.QComboBox()
        self.instrument.addItems(vars.INSTRUMENTS.keys()) # Add all available instruments from variables file

        # Dropdown list for available material compositions
        # These labels will form the header row for the materials table if multiple are added
        self.comp_labels = [qt.QLabel("Composition:"), qt.QLabel("Volume Fraction (dim):")]
        self.mat_table = None # A table will be added if we add multiple materials to the composition
        # Each material in a composition will have a dropdown list to specify the material,
        # plus an entry field for the corresponding volume fraction. This array will be added to for each material
        self.comps = [[qt.QComboBox(), qt.QLineEdit()]]
        self.comps[0][0].addItems(COMPS) # Each box allows for selection of a new material
        self.comps[0][1].setText('1.0') # The default volume fraction (not shown) is 1.0, since there is only one material
        self.comps[0][1].setPlaceholderText("Input volume fraction")
        # Determining the minimum and maximum bounds for wavelength based on available material data
        # Instrument wavelength ranges will be far more limiting, but good to include this nonetheless
        self.comp_min_wavel, self.comp_max_wavel = utils.get_minmax_wavel(self.comps[0][0].currentText())

        # Dropdown list for available size distributions (G-ring, E-ring, etc)
        self.dist_label = qt.QLabel("Size Distribution:")
        self.dist_model = qt.QComboBox()
        self.dist_model.addItems(vars.SIZEDISTS.keys())
        self.dist_model.addItems(['Custom'])
        # These objects will be shown when the user clicks the 'Custom' option for size distributions. Essential for Mie computations.
        self.smin_label = None
        self.smin_box = None
        self.smax_label = None
        self.smax_box = None
        self.powlaw_label = None
        self.powlaw_box = None
        # These objects may or may not be shown depending on the scattering theory, but they are necessary for the semi-empirical theory and some of its variants
        self.G_label = None
        self.G_box = None
        self.r_label = None
        self.r_box = None
        self.x0_label = None
        self.x0_box = None

        # Dropdown list to select which scattering theory to use in graphs
        self.fit_label = qt.QLabel("Scattering Theory:")
        self.fit_model = qt.QComboBox()
        # By default, semi-empirical Mie (as implemented by Pollack & Cuzzi, 1980) is used. The others are to get different components of the Mie function.
        # Since the default is reflectance vs. wavelength, Henyey-Greenstein won't be added yet
        self.fit_model.addItems(["Semi-Empirical Mie", "Pure Mie", "Mie Diffraction", "Mie External Reflection", "Mie Transmission"])

        # Button to change between input window and resolution settings window
        self.window_button = qt.QPushButton("Open Resolution Settings")

        # Entry box for manual optical depth input. Will only be shown if the y-axis of the graph is 'reflectance'
        self.tau_label = qt.QLabel("Optical Depth (dim):")
        self.tau_box = qt.QLineEdit()
        self.tau_box.setPlaceholderText("Input optical depth") # In case the user clears the box
        self.tau_box.setText("1e-6") # Standard order of magnitude for optical depth

        # Entry box for varying parameter, either scattering angle (for ref vs. wavel), effective grain size (surface reflectance vs. wavelength), or wavelength (for ref vs. theta)
        self.param_str = ''
        # If an angle is input, it will always have two options: scattering or phase
        self.input_label = qt.QComboBox()
        self.input_label.addItems(["Phase Angle (°):", "Scattering Angle (°):"])
        self.input_box = qt.QLineEdit()
        self.input_box.setPlaceholderText("Input scattering angle in degrees") # Switching graphs clears this box, so this needs to display
        self.input_box.setText("90")

        # X axis dropdown box, can switch between plotting phase angle and scattering angle
        self.x_axis_label = qt.QLabel("X Axis:")
        self.x_axis = qt.QComboBox()
        self.x_axis.addItems(["Wavelength"])

        # Y axis dropdown box, can switch between plotting reflectance and phase function (as used in Pollack & Cuzzi, 1980)
        self.y_axis_label = qt.QLabel("Y Axis:")
        self.y_axis = qt.QComboBox()
        self.y_axis.addItems(["Reflectance", "Phase Function", "Intensity"])

        # Setting up radio buttons for X and Y scales, change between linear and logarithmic
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

        # Setting up a pushbutton to plot the current data and start a new dataset on the same plot
        self.add_graph_button = qt.QPushButton("Add new plot")
        # Setting up a pushbutton to add materials
        self.add_material_button = qt.QPushButton("Add new material (mixture)")
        # Setting up a pushbutton to clear the whole plot
        self.clear_plot_button = qt.QPushButton("Clear plots")
        # Setting up a pushbutton to save the plot to file
        self.save_plot_button = qt.QPushButton("Save plot")

        # Setting up a label at the bottom of the screen giving status updates
        self.load_label = qt.QLabel("Up to date")
        self.load_label.setStyleSheet("color: red;")
        # The load label will handle errors, as such it needs to wrap since it can get pretty long
        self.load_label.setWordWrap(True)

        # Setting up the graph window
        self.multiple = False # Multiple will be used to indicate if multiple datasets are in the same graph.
        self.x_data = [] # Arrays to hold the current data
        self.y_data = []
        grid_pen = pg.mkPen(color='k', width=1, style=pg.QtCore.Qt.PenStyle.SolidLine) # Pen to use on axes
        self.graph_widget = pg.PlotWidget()
        self.graph_widget.setBackground('w') # Set background to white
        self.graph_widget.showGrid(x=True, y=True) # Grid is helpful to identify points on displayed graphs
        # Setup axes for default reflectance vs. wavelength plot
        self.graph_widget.setLabel('left', self.y_axis.currentText(), units='I/F', color='k')
        self.graph_widget.setLabel('bottom', 'Wavelength', units='μm', color='k')
        # Disable axis "autoscale" where the program adds a prefix based on the zoom, which can be misleading
        # Axis will display numbers in scientific notation automatically; this is good for clarity
        self.graph_widget.getAxis('bottom').enableAutoSIPrefix(False)
        self.graph_widget.getAxis('left').enableAutoSIPrefix(False)
        # Pen ensures axes and numbers are black and thick
        self.graph_widget.getAxis('bottom').setPen(grid_pen)
        self.graph_widget.getAxis('left').setPen(grid_pen)
        self.graph_widget.getAxis('bottom').setTextPen(grid_pen)
        self.graph_widget.getAxis('left').setTextPen(grid_pen)
        self.legend = None # Declare a legend object, will be populated when multiple is True

        # Setting up a pen to plot the window
        self.last_color_idx = 0 # Color from which to start. Will count up in multiple plotting
        self.plt_pen = pg.mkPen(color=COLOR_ARR[self.last_color_idx], width=3)
        self.ifs = self.graph_widget.plot(self.x_data, self.y_data, pen=self.plt_pen) # Plot a blank dataset

        # Setting up the 'home zoom' button to reset the zoom
        self.home_button = qt.QPushButton("Home")

        # Setting up the min and max bounds for x and y
        self.xmin_label = qt.QLabel("Min X:")
        self.xmin_label.setStyleSheet("color: black;")
        self.xmin_box = qt.QLineEdit()
        self.xmin_box.setPlaceholderText("Input min x")
        self.xmin_box.setMaximumWidth(50)
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

        # Setting up the settings screen
        self.settings_widget = qt.QWidget()
        # We will use a Vertical Box Layout (QVBoxLayout) to stack widgets top-to-bottom
        self.settings_layout = qt.QGridLayout()
        self.settings_layout.setContentsMargins(30, 30, 30, 30) # Making sure things don't overlap each other
        self.settings_layout.setSpacing(20)
        self.settings_widget.setLayout(self.settings_layout)

        # The following widgets will be added to this screen. NONE OF THESE will update the graph
        # Back button to go back to data input screen
        self.back_button = qt.QPushButton("Back")

        # Dropdown box to use in selecting linear-distributed or log-distributed sizes
        self.spacing_label = qt.QLabel("Spacing Mode:")
        self.spacing = qt.QComboBox()
        self.spacing.addItems(["Linear", "Logarithmic"])
        self.spacing.setCurrentText(vars.NSIZE[0]) # Grab the default spacing from vars

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
        # Base parameters track conditions of each plot on the graph, used for tracking changes
        # If ANY of these values are not modifiable by the user in the GUI (i.e. not displayed), they
        # will be set to 'NaN'
        self.base_params = [[self.instrument.currentText(), # Instrument on Clipper
                    {self.comps[0][0].currentText(), self.comps[0][1].text()}, # Materials dictionary, name: volume fraction
                    'Molecular', # Mixture model is not displayed at first, so must set to default. If not displayed, this is Molecular by default, not NaN
                    vars.SIZEDISTS['G-ring-like'][0][0], # S_min parameter, varies with size distribution
                    vars.SIZEDISTS['G-ring-like'][0][1], # S_max parameter, varies with size distribution
                    vars.SIZEDISTS['G-ring-like'][0][2], # Power law parameter, varies with size distribution
                    vars.SIZEDISTS['G-ring-like'][0][3], # r parameter for semi-empirical Mie, varies with size distribution
                    vars.SIZEDISTS['G-ring-like'][0][4], # G parameter for semi-empirical Mie, varies with size distribution
                    vars.SIZEDISTS['G-ring-like'][0][5], # x0 parameter for semi-empirical Mie, varies with size distribution
                    self.fit_model.currentText(), # Which scattering theory to use
                    [self.spacing.currentText(), self.dist_slider.value()], # Size distribution resolution
                    self.tau_box.text(), # Optical depth
                    [self.input_box.text(), self.input_label.currentText()]]] # Varying parameter (wavelength, grain size, scattering angle, phase angle)
        self.legend_idxs = [] # Array to hold base parameter indexes to include in the legends, i.e. those that change at some point in base_param's history

        # --- WIDGET FUNCTIONALITY ---
        # Pressing the graph mode button toggles between graph modes, clearing the window in the process
        self.graph_mode_button.clicked.connect(self.mode_toggle)

        # Changing the x-axis will clear the window and change the displayed horizontal axis
        self.x_axis.currentTextChanged.connect(self.change_axis)
        # Changing the y-axis will clear the window and change the displayed vertical axis
        self.y_axis.currentTextChanged.connect(self.change_axis)

        # Pressing scale radio buttons updates scale
        self.x_scale.buttonClicked.connect(self.update_scale)
        self.y_scale.buttonClicked.connect(self.update_scale)

        # Changing anything about the dataset (instrument, comp, model, tau, etc) replots the data
        self.instrument.currentTextChanged.connect(self.update_graph)
        self.tau_box.returnPressed.connect(self.update_graph)
        self.input_label.currentTextChanged.connect(lambda: self.change_input(update=True))
        self.input_box.returnPressed.connect(self.update_graph)
        # The size distribution model and the scattering theory need special functions to handle custom distribution inputs
        self.dist_model.currentTextChanged.connect(self.change_distmodel)
        self.fit_model.currentTextChanged.connect(self.change_fitmodel)

        # The window button will be connected to a function that switches to the resolution settings window
        self.window_button.clicked.connect(self.change_window)
        # The back button uses this same function to switch back to the data input window
        self.back_button.clicked.connect(self.change_window)

        # The slider will NOT be connected to update_graph because it is purely a performance improvement tool
        # It merely updateds the displayed size count
        self.dist_slider.valueChanged.connect(self.update_slider)
        # Its corresponding dropdown box updates a separate function
        self.spacing.currentTextChanged.connect(self.update_spacing)

        # Changing anything about the composition replots the data
        self.comps[0][0].currentTextChanged.connect(self.update_graph)
        self.comps[0][1].returnPressed.connect(self.update_graph)

        # Pressing add graph button saves the dataset in the plot and starts a new one
        self.add_graph_button.clicked.connect(self.plot_graph)
        # Pressing add material button will add new materials to the composition list
        self.add_material_button.clicked.connect(self.add_material)
        # Pressing clear plot button clears the plot and resets the title
        self.clear_plot_button.clicked.connect(self.clear_graph)
        # Pressing save plot button saves the graph to file
        self.save_plot_button.clicked.connect(self.save_graph)
        # Pressing the home zoom button auto-zooms the graph
        self.home_button.clicked.connect(lambda: self.rst_boxes(autozoom=True))

        # Hitting return in any of the bounds fields will cause a bounds update
        self.xmin_box.returnPressed.connect(self.update_bounds)
        self.xmax_box.returnPressed.connect(self.update_bounds)
        self.ymin_box.returnPressed.connect(self.update_bounds)
        self.ymax_box.returnPressed.connect(self.update_bounds)

        # --- GRAPH DRAG & RELEASE LOGIC ---
        # Create a flag to track if the user dragged the graph, which would cause an update of the bounds
        # Dragging the graph can connect to an event itself, but ensuring the graph is dragged and released
        # requires intercepting a mouse release event.
        self.graph_was_dragged = False
        # Listen for manual range changes (fires continuously during a drag)
        def flag_dragged():
            self.graph_was_dragged = True
        self.graph_widget.getViewBox().sigRangeChangedManually.connect(flag_dragged)

        # Store the original mouse release/scroll wheel to avoid breaking PyQt6's internal math
        self._original_mouse_release = self.graph_widget.mouseReleaseEvent
        # The scroll wheel allows the graph to be zoomed in/out, also requiring a bounds update
        self._original_wheel = self.graph_widget.wheelEvent

        def click_release(event):
            """
            A function that tracks when the user has let go after dragging the graph with their mouse, signals
            to update the bounds boxes.

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
            A function that intercepts a scroll zoom update to the graph and updates the bound boxes
            dynamically.

            :param event: Keyboard input event to watch for
            """
            self._original_wheel(event) # Keep the original scrolling
            self.rst_boxes() # Reset bounds boxes
            
        # Override default mouse release and scroll events to allow dynamic updating of bounds boxes
        self.graph_widget.mouseReleaseEvent = click_release
        self.graph_widget.wheelEvent = scroll_wheel

        # --- LAYOUT ---
        # Items are arranged in (row, column, rowspan, columnspan)
        # Buttons are at the top, plus instruments since that box never causes weird formatting
        self.layout.addWidget(self.graph_mode_button, 0, 0, 1, 3)
        self.layout.addWidget(self.add_material_button, 1, 0, 1, 3)
        self.layout.addWidget(self.add_graph_button, 2, 0, 1, 3)
        self.layout.addWidget(self.clear_plot_button, 3, 0, 1, 3)
        self.layout.addWidget(self.save_plot_button, 4, 0, 1, 3)
        self.layout.addWidget(self.instrument_label, 5, 0, 1, 1)
        self.layout.addWidget(self.instrument, 5, 1, 1, 2)

        # All these widgets are overlaid on top of each other (home in top left, xmin/max & ymin/max in top right)
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

        # These two will get replaced by a table if multi-material mixture is enabled
        # Only one new row will be inserted for the mix model if this is the case
        self.layout.addWidget(self.comp_labels[0], 6, 0, 1, 1)
        self.layout.addWidget(self.comps[0][0], 6, 1, 1, 2)

        # The rest of the widgets may update in visibility depending on the combination of parameters selected.
        self.layout.addWidget(self.dist_label,  7, 0, 1, 1)
        self.layout.addWidget(self.dist_model, 7, 1, 1, 2)
        self.layout.addWidget(self.fit_label, 8, 0, 1, 1)
        self.layout.addWidget(self.fit_model, 8, 1, 1, 2)
        self.layout.addWidget(self.window_button, 9, 0, 1, 3)
        self.layout.addWidget(self.tau_label, 10, 0, 1, 1)
        self.layout.addWidget(self.tau_box, 10, 1, 1, 2)
        self.layout.addWidget(self.input_label, 11, 0, 1, 1)
        self.layout.addWidget(self.input_box, 11, 1, 1, 2)
        self.layout.addWidget(self.x_axis_label, 12, 0, 1, 1)
        self.layout.addWidget(self.x_axis, 12, 1, 1, 2)
        self.layout.addWidget(self.y_axis_label, 13, 0, 1, 1)
        self.layout.addWidget(self.y_axis, 13, 1, 1, 2)
        self.layout.addWidget(self.x_scale_label, 14, 0, 1, 1)
        self.layout.addWidget(self.x_linear, 14, 1, 1, 1)
        self.layout.addWidget(self.x_log, 14, 2, 1, 1)
        self.layout.addWidget(self.y_scale_label, 15, 0, 1, 1)
        self.layout.addWidget(self.y_linear, 15, 1, 1, 1)
        self.layout.addWidget(self.y_log, 15, 2, 1, 1)
        self.layout.addWidget(self.load_label, 16, 0, 2, 2)

        # Add the two screens to the stack
        self.stack.addWidget(self.data_input_widget)
        self.stack.addWidget(self.settings_widget)
        
        # Set the stack as the central widget
        self.setCentralWidget(self.stack)
        self.stack.setCurrentIndex(0) # Show data entry screen by default

        # Set up a processor variable for heavy computing later
        self.proc = None

        # Create a timer for when we check if the subprocess is done later
        self.monitor_timer = qc.QTimer()
        self.monitor_timer.setInterval(100) # Update ever 100 ms
        self.monitor_timer.timeout.connect(self.finish_process)

        # There are default settings in the input boxes, so output the first graph
        self.change_input() # First update the placeholder texts
        self.update_graph()

    def mode_toggle(self):
        """
        A function that defines the behavior of the graph mode button when clicked. Toggles between
        'Reflectance vs. Wavelength', 'Surface Reflectance vs. Wavelength', and 'Reflectance vs. Scattering 
        Angle.' Clears graph in the process, resets input fields, and updates title

        :param self: MainWindow object
        """
        self.clear_graph() # Clear graph first

        # First ensure all missing widgets are displayed upon a mode change. This is only relevant from mode 1 to mode 2
        if self.graphmode == 1:
            # Make a couple new rows to add the missing widgets we know will always be present in this graph mode
            self.mknewrow(self.row_widget(self.input_label), 2)
            self.layout.addWidget(self.dist_label, 7, 0, 1, 1)
            self.layout.addWidget(self.dist_model, 7, 1, 1, 2)
            self.layout.addWidget(self.fit_label, 8, 0, 1, 1)
            self.layout.addWidget(self.fit_model, 8, 1, 1, 2)
            self.dist_label.setVisible(True) # Make sure they are displayed so track_changes can update them properly
            self.dist_model.setVisible(True)
            self.fit_label.setVisible(True)
            self.fit_model.setVisible(True)
            self.y_axis.removeItem(self.y_axis.findText("Albedo")) # Albedo is special to graph mode 1
            # Add back the options for phase function & reflectance for the y-axis
            # This will switch back to reflectance by default
            self.y_axis.addItems(["Reflectance", "Phase Function", "Intensity"])

            # Check to see if the nsizes slider should be displayed. If so, add it after fit_model
            if self.window_button is not None:
                window_button_row = self.row_widget(self.fit_model) + 1
                self.mknewrow(window_button_row, 1)
                self.layout.addWidget(self.window_button, window_button_row, 0, 1, 3)
                self.window_button.setVisible(True)

            # If we set the distribution to custom before, we re-add all the size distribution boxes
            if self.smin_label is not None:
                self.mknewrow(8, 3)
                self.layout.addWidget(self.smin_label, 8, 0, 1, 1)
                self.layout.addWidget(self.smin_box, 8, 1, 1, 2)
                self.layout.addWidget(self.smax_label, 9, 0, 1, 1)
                self.layout.addWidget(self.smax_box, 9, 1, 1, 2)
                self.layout.addWidget(self.powlaw_label, 10, 0, 1, 1)
                self.layout.addWidget(self.powlaw_box, 10, 1, 1, 2)
                self.smin_label.setVisible(True)
                self.smin_box.setVisible(True)
                self.smax_label.setVisible(True)
                self.smax_box.setVisible(True)
                self.powlaw_label.setVisible(True)
                self.powlaw_box.setVisible(True)

                # If the scattering theory allows it, add G, r, and x0 boxes
                if self.r_label is not None:
                    self.mknewrow(11, 1)
                    self.layout.addWidget(self.r_label, 11, 0, 1, 1)
                    self.layout.addWidget(self.r_box, 11, 1, 1, 2)
                    self.r_label.setVisible(True)
                    self.r_box.setVisible(True)
                if self.G_label is not None:
                    # Check to see if both G and r exist, and if so, add them on separate rows. Otherwise, add G to row 11.
                    if self.r_label is not None:
                        self.mknewrow(12, 1)
                        self.layout.addWidget(self.G_label, 12, 0, 1, 1)
                        self.layout.addWidget(self.G_box, 12, 1, 1, 2)
                    else:
                        self.mknewrow(11, 1)
                        self.layout.addWidget(self.G_label, 11, 0, 1, 1)
                        self.layout.addWidget(self.G_box, 11, 1, 1, 2)
                    self.G_label.setVisible(True)
                    self.G_box.setVisible(True)
                if self.x0_label is not None:
                    # x0 will ONLY show up in the case where both r and G show up, i.e. in the semi-empirical Mie
                    self.mknewrow(13, 1)
                    self.layout.addWidget(self.x0_label, 13, 0, 1, 1)
                    self.layout.addWidget(self.x0_box, 13, 1, 1, 2)
                    self.x0_label.setVisible(True)
                    self.x0_box.setVisible(True)

        self.graphmode = int((self.graphmode + 1) % 3) # Cycle graph mode between 0 and 2
        if self.graphmode == 0:
            self.graph_mode_button.setText("Reflectance vs. Wavelength")
            # Replace wavelength input with angle inputs. If wavelength is not in the combo box, this will fail silently.
            self.input_label.removeItem(self.input_label.findText("Wavelength (μm):"))
            self.input_label.addItems(["Phase Angle (°):", "Scattering Angle (°):"])
            # Remove Henyey-Greenstein from the dropdown list since it is not supported in ref. vs. wavel. plots
            # If it's not in there, this will fail silently and not cause any issues
            self.fit_model.removeItem(self.fit_model.findText("Henyey-Greenstein"))

            # Remove angle options from x-axis dropdown, add only wavelength option
            self.x_axis.removeItem(self.x_axis.findText("Phase Angle"))
            self.x_axis.removeItem(self.x_axis.findText("Scattering Angle"))
            self.x_axis.addItem("Wavelength")            

        elif self.graphmode == 1:
            self.graph_mode_button.setText("Surface Reflectance vs. Wavelength")
            # Replace angle inputs with wavelength input. If angles are not in the combo box, remove calls will fail silently.
            self.input_label.removeItem(self.input_label.findText("Phase Angle (°):"))
            self.input_label.removeItem(self.input_label.findText("Scattering Angle (°):"))
            self.input_label.addItem("Effective Grain Size (μm):")

            # If size distribution boxes are shown, get rid of them and switch sizedist to G-ring-like
            if self.smin_label is not None:
                # Shift rows up
                self.mknewrow(self.row_widget(self.smin_label) + 3, -3)

                # Remove widgets from display. Note how we set visibility to false instead of setting smin_label
                # to None. That is done elsewhere to trigger a RESET of the size distribution boxes. As it stands,
                # when the graph cycles through its modes, boxes that disappear will reappear with the values they
                # had before in them. It is important to note the code does NOT read these boxes while they are
                # invisible.
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
                
                # If the scattering theory created them, make G, r, and x0 boxes not visible
                if self.r_label is not None:
                    # Deleting rows invovles going to the next row and shifting it up one
                    self.mknewrow(self.row_widget(self.r_label) + 1, -1)
                    self.layout.removeWidget(self.r_label)
                    self.r_label.setVisible(False)
                    self.layout.removeWidget(self.r_box)
                    self.r_box.setVisible(False)
                if self.G_label is not None:
                    self.mknewrow(self.row_widget(self.G_label) + 1, -1)
                    self.layout.removeWidget(self.G_label)
                    self.G_label.setVisible(False)
                    self.layout.removeWidget(self.G_box)
                    self.G_box.setVisible(False)
                if self.x0_label is not None:
                    # Delete this row
                    self.mknewrow(self.row_widget(self.x0_label) + 1, -1)
                    self.layout.removeWidget(self.x0_label)
                    self.x0_label.setVisible(False)
                    self.layout.removeWidget(self.x0_box)
                    self.x0_box.setVisible(False)

            # Hide all unnecessary widgets (only the input param is needed)
            self.mknewrow(self.row_widget(self.dist_label) + 2, -2)
            self.layout.removeWidget(self.dist_label)
            self.dist_label.setVisible(False)
            self.layout.removeWidget(self.dist_model)
            self.dist_model.setVisible(False)
            self.layout.removeWidget(self.fit_label)
            self.fit_label.setVisible(False)
            self.layout.removeWidget(self.fit_model)
            self.fit_model.setVisible(False)

            # Also hide slider if present
            if self.window_button is not None:
                self.mknewrow(self.row_widget(self.window_button) + 1, -1)
                self.layout.removeWidget(self.window_button)
                self.window_button.setVisible(False)

            # Also hide tau if visible
            if not self.tau_label.isHidden():
                self.mknewrow(self.row_widget(self.input_box), -1)
                self.layout.removeWidget(self.tau_label)
                self.tau_label.setVisible(False)
                self.layout.removeWidget(self.tau_box)
                self.tau_box.setVisible(False)


            # Remove phase function, reflectance, and intensity from the y-axis dropdown list since it is not supported in surface spectra plots
            # Albedo is currently the only supported y-axis value for those plots
            # If they're not in there, this will fail silently and not cause any issues
            self.y_axis.removeItem(self.y_axis.findText("Reflectance"))
            self.y_axis.removeItem(self.y_axis.findText("Phase Function"))
            self.y_axis.removeItem(self.y_axis.findText("Intensity"))
            self.y_axis.addItem("Albedo")
            # x axis doesn't change because, once again, only wavelength is available

        elif self.graphmode == 2:
            self.graph_mode_button.setText("Reflectance vs. Scattering Angle")
            self.input_label.removeItem(self.input_label.findText("Effective Grain Size (μm):"))
            self.input_label.addItem("Wavelength (μm):")
            # Remove wavelength from x-axis dropdown and add scattering and phase angles
            self.x_axis.removeItem(self.x_axis.findText("Wavelength"))
            self.x_axis.addItems(["Phase Angle", "Scattering Angle"])

        self.change_input() # Call just to make sure input placeholder text updates
        self.change_axis() # Call just to make sure all shown parameters correspond to the y-axis of the plot

    def update_graph(self):
        """
        A function called if any of the data parameters are changed. Calls the plotter function from the main
        program and outputs the new temporary data on the graph, replacing the old data (assuming that data
        wasn't made permanent via 'Add new plot'). If parameters are invalid, updates status message to the
        appropriate error message and turns the appropriate box red. If multiple graphs involved, updates legend 
        accordingly.

        :param self: MainWindow object
        """
        self.all_borders('black') # Reset any previous errors
        self.load_label.setText("Updating...") # Status label changes to indicate loading
        
        qt.QApplication.processEvents() # Force a screen update to render the status label and the box borders during computation

        # Error checking for materials table if available
        err_message = '' # Will dynamically update to support multiple errors at once
        if len(self.comps) > 1:
            # Update materials checks to make sure there aren't errors specifically in the materials table
            # update_materials returns an error message if any, tack that on whether empty or not
            err_message += self.update_materials()

        # Error checking for smin, smax, powlaw boxes if available
        if self.smin_label is not None:
            smin_txt = self.smin_box.text()
            smax_txt = self.smax_box.text()
            powlaw_txt = self.powlaw_box.text()

            # Smin checks
            try:
                smin = float(smin_txt)
                if smin <= 0:
                    # A space must be included after each error message because they can stack
                    err_message += "Min size must be greater than zero. "
            except ValueError:
                err_message += "Min size must be a float. "

            # Smax checks
            try:
                smax = float(smax_txt)
                if 'Min size' not in err_message and smax < smin:
                    err_message += "Max size must be greater than min size. "
                elif smax <= 0:
                    err_message += "Max size must be greater than zero. "
            except ValueError:
                err_message += "Max size must be a float. "

            # Powlaw checks
            try:
                powlaw = float(powlaw_txt)
                if powlaw < 0:
                    err_message += "Power law must be nonnegative (will be negated in calculations). "
            except ValueError:
                err_message += "Power law must be a float. "

            # If r or G are included, check those
            if self.r_label is not None:
                try:
                    r = float(self.r_box.text())
                    if r < 0:
                        err_message += "R must be nonnegative. "
                except ValueError:
                    err_message += "R must be a float. "
            if self.G_label is not None:
                try:
                    G = float(self.G_box.text())
                    if G <= 0:
                        err_message += "G must be greater than 0. " # G gets passed into a natural log and must be greater than 0
                except ValueError:
                    err_message += "G must be a float. "
            if self.x0_label is not None:
                try:
                    x0 = float(self.x0_box.text())
                    if x0 < 0:
                        err_message += "x0 must be nonnegative. "
                except ValueError:
                    err_message += "x0 must be a float. "

            # Turn appropriate boxes red
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
            
        # Read each parameter from the appropriate widget
        sensor = self.instrument.currentText()

        # Error message is displayed at the end; however, we need to check for updated comps here to check for
        # other errors later
        compositions = {} # This will be used to update the graph/legend
        if "Volume fractions" not in err_message: # Only compute if no comp errors were detected
            for i, item in enumerate(self.comps):
                if not float(item[1].text()) == 0: # Compositions with v/v equal to zero don't count in legend updates
                    compositions[item[0].currentText()] = item[1].text()

            # Composition changes require getting which wavelengths are supported for that material
            # First set up extreme cases
            min_limiting_comp = ['', 0] # First index will be material, second will be limiting wavelength bound
            max_limiting_comp = ['', 99999999999]
            # A loop to check which material with a nonzero v/v has the greatest minimum wavelength, and which has the least maximum
            # Getting a baseline wavelength range bound
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
        
        # Tau and the parameter are both initially read as strings
        tau_str = self.tau_box.text()
        self.param_str = self.input_box.text() # param_str is a self variable because it needs to be referenced elsewhere

        # Error checking for parameter if it's a wavelength
        if self.graphmode == 2:
            try:
                param = float(self.param_str)
                # Check to make sure it's in range of the sensor AND the index of refraction data for that material
                # material has to make sure no composition errors are before it, else self.comp_min_wavel won't be updated
                if "Volume fractions" not in err_message:
                    if param < self.comp_min_wavel[1]:
                        err_message += f'Wavelength is too low for {self.comp_min_wavel[0]}. '
                    elif param > self.comp_max_wavel[1]:
                        err_message += f'Wavelength is too high for {self.comp_max_wavel[0]}. '
                if param < vars.INSTRUMENTS[sensor][0]:
                    err_message += f'Wavelength is too low for {sensor}. '
                elif param > vars.INSTRUMENTS[sensor][1]:
                    err_message += f'Wavelength is too high for {sensor}. '
            except ValueError:
                # Allow for input of MIN and MAX to get the top and bottom of the instrument's range
                if self.param_str == 'MIN':
                    param = vars.INSTRUMENTS[sensor][0]
                    self.param_str = str(param) # Param str is used in title updates, ensure it doesn't display MAX but an actual number
                    # Set the user input box to display the parameter. This is mainly so the title or legend doesn't display MIN, which is weird
                    self.input_box.setText(self.param_str)
                    # Check material bounds again (sensor bounds automatically pass)
                    if "Volume fractions" not in err_message:
                        if param < self.comp_min_wavel[1]:
                            err_message += f'Wavelength is too low for {self.comp_min_wavel[0]}. '
                        elif param > self.comp_max_wavel[1]:
                            err_message += f'Wavelength is too high for {self.comp_max_wavel[0]}. '
                elif self.param_str == 'MAX':
                    param = vars.INSTRUMENTS[sensor][1]
                    self.param_str = str(param)
                    self.input_box.setText(self.param_str)
                    if "Volume fractions" not in err_message:
                        if param < self.comp_min_wavel[1]:
                            err_message += f'Wavelength is too low for {self.comp_min_wavel[0]}. '
                        elif param > self.comp_max_wavel[1]:
                            err_message += f'Wavelength is too high for {self.comp_max_wavel[0]}. '
                else:
                    err_message += f'Wavelength must be a float, MIN, or MAX. '

        # Error checking for parameter if it's a scattering angle
        elif self.graphmode == 0:
            # First check to see if the code expects a scattering angle or a phase angle
            angle_str = 'Scattering angle' if 'Scattering' in self.input_label.currentText() else 'Phase angle'
            try:
                param = int(self.param_str)
                if param < 0 or param > 180:
                    err_message += f'{angle_str} must be between 0 and 180 degrees. '
            except ValueError:
                err_message += f'{angle_str} must be an integer. '

        # Error checking for parameter if it's an effective grain size
        elif self.graphmode == 1:
            try:
                param = float(self.param_str)
                if param <= 0:
                    err_message += f'Effective grain size must be greater than zero. '
            except ValueError:
                err_message += f'Effective grain size must be a float. '

        # Error checking for if no data will be plotted in the wavelength range for plots that depend on wavelength
        if self.graphmode == 0 or self.graphmode == 1:
            # Trigger an error if the max data wavelength is below the min sensor wavelength or vice versa so that nothing will be plotted
            # Only one will ever be true because both true would mean min wavelength > max wavelength
            if self.comp_min_wavel[1] > vars.INSTRUMENTS[sensor][1]:
                err_message += f'No data available for {self.comp_min_wavel[0]} in {sensor} wavelength range. '
            elif self.comp_max_wavel[1] < vars.INSTRUMENTS[sensor][0]:
                err_message += f'No data available for {self.comp_max_wavel[0]} in {sensor} wavelength range. '

        # Error checking for tau
        try:
            tau = float(tau_str)
            if tau < 0:
                err_message += 'Optical depth must be nonnegative. '
        except ValueError:
            err_message += 'Optical depth must be a float. '

        # Error checking to ensure that x0 is not divided such that the length of small sizes or large sizes is 1 and cannot be integrated
        if self.x0_label is not None and self.graphmode != 0 and 'x0' not in err_message and 'Wavelength' not in err_message and 'wavelength' not in err_message and 'size' not in err_message:
            sizes = self.dist_slider.value()
            stype = self.spacing.currentText()

            # This is the radius creation and size parameter conversion used in utils.py
            if stype == 'Linear':
                radii = np.linspace(smin, smax, sizes)
            elif stype == 'Logarithmic':
                radii = utils.log_dist(smin, smax, sizes / 100)
            sizeparams = 2 * np.pi * radii / param
            small_sizes = sizeparams[sizeparams <= x0]
            large_sizes = sizeparams[sizeparams > x0]

            # Check to ensure neither part has length 1, otherwise we won't be able to integrate it later
            if len(small_sizes) == 1:
                err_message += 'x0 is too low for given number of sizes (cannot integrate small sizes). '
            if len(large_sizes) == 1:
                err_message += 'x0 is too high for given number of sizes (cannot integrate large sizes). '

        # We only update if there are no input errors
        if len(err_message) == 0:
            # Get the current text box values (updated_params), plus any indexes where they're different from the original data
            updated_params, self.temp_legend_idxs = self.track_changes() # temp_legend_idxs is used in the worker_results function and thus needs to be a global variable
            # temp_legend_idxs is only ever None if there are multiple plots in the window and the current plot
            # is found to match a plot somewhere in the window's history. Not necessarily checking for change
            # but for duplicate plots
            if self.temp_legend_idxs is None:
                    self.load_label.setText("Plot is identical to existing data")
                    self.ifs.setData([], [])
                    self.legend.removeItem(self.ifs)
                    self.all_borders('red')
            # update_graph runs whenever a field is clicked. But if it hasn't changed, don't bother with calculations
            elif len(self.temp_legend_idxs) == 0:
                self.load_label.setText("Plot is identical to previous graph")
                # Generic errors are generally speaking going to turn all the boxes red
                self.all_borders('red')
            else:
                # If the text box values are in fact unique and have no errors, update the last element in the base parameter array
                # The last element represents the current plot (every time a plot gets added later, a new element gets tacked on)
                self.base_params[-1] = updated_params

                # Reuse the same pen color for all updates to the current plot
                self.plt_pen.setColor(pg.mkColor(COLOR_ARR[self.last_color_idx]))

                # Stop timer if running
                self.monitor_timer.stop()
                # Check to ensure a process isn't already running, and if it is, kill it
                if self.proc is not None:
                    self.proc.terminate() # Send the SIGTERM signal to the process
                    self.proc.join()

                # Set up a multiprocessing process to do the heavy calculations on a separate core
                self.proc = mp.Process(target=output_graph, args=(self.graphmode, updated_params[0], updated_params[1], updated_params[2], [updated_params[3], updated_params[4], updated_params[5], updated_params[6], updated_params[7], updated_params[8]], updated_params[9], (float(updated_params[12][0]) if 'Phase Angle' not in self.input_label.currentText() else 180 - float(updated_params[12][0])), updated_params[11], updated_params[10], (self.comp_min_wavel[1], self.comp_max_wavel[1]), [self.x_axis.currentText(), self.y_axis.currentText()]))
                self.proc.start()
                self.monitor_timer.start()

        else:
            self.load_label.setText(err_message)
            # Scan the error message for problematic variables and turn their boxes red
            if 'Wavelength' in err_message or 'angle' in err_message or 'Effective grain size' in err_message:
                self.input_box.setStyleSheet("border: 1px solid red;")
            if 'Optical depth' in err_message:
                self.tau_box.setStyleSheet("border: 1px solid red;")
            # Composition and size distribution errors get handled earlier

    def plot_graph(self):
        """
        A function that saves the current x_data, y_data of the temporary plot to the graph as a permanent plot 
        and resets the program for a new plot, setting the plot up to hold multiple datasets. Sets up the legend 
        and updates the title.

        :param self: MainWindow object
        """
        # Stop timer if running
        self.monitor_timer.stop()
        # Check to ensure a process isn't already running, and if it is, kill it
        if self.proc is not None:
            self.load_label.setText("Cancelling...")
            qt.QApplication.processEvents() # Force a screen update to render the status label
            self.proc.terminate() # Send the SIGTERM signal to the process
            self.proc.join()
            self.proc = None

        # Track all changes in the current dataset. Here updated_params is likely not going to be different than
        # the last row of self.base_params if the dataset has already been plotted, which it must be for this
        # to run
        updated_params, self.legend_idxs = self.track_changes()

        # If the data haven't been plotted for this graph yet, throw a general error
        if len(self.x_data) == 0 or len(self.y_data) == 0:
            self.load_label.setText("No plot to add")
            self.all_borders('red')
        # If the data have been found in a previous dataset, throw a general error
        # Should be caught by the update_graph, but just in case
        elif self.legend_idxs is None:
            self.load_label.setText("Plot is identical to existing data")
            self.all_borders('red')
        else:
            self.all_borders('black') # Remove all outstanding errors

            # Create a legend if none already. Needs to be here since legends are for multiple graphs only
            if self.legend == None:
                self.legend = self.graph_widget.addLegend()
                self.legend.setLabelTextColor('k') # Default is gray and hard to see; we set it to black

            self.legend.removeItem(self.ifs) # If temporary plot has a legend item, remove it to establish a permanent one
            # Turn the temporary plot into a permanent one not associated with a temporary self.ifs variable.
            self.graph_widget.plot(self.x_data, self.y_data, pen=pg.mkPen(color=COLOR_ARR[self.last_color_idx], width=3), name='')
            
            # Expand the size of base_params, which tracks history of graphs. The last two entries are now identical,
            # but update_graph will change the last entry before this gets called again
            self.base_params.append(updated_params)
            self.generate_legend(self.legend_idxs)

            # self.generate_legend looks for changes across datasets, but if there's only one in the window,
            # use default legend label
            if not self.multiple:
                self.legend.items[-1][1].setText(f"{self.base_params[0][-1][0]}{' μm' if (self.graphmode == 1 or self.graphmode == 2) else '°'}") # Default legend string for first item
            
            self.multiple = True # Set the global variable to indicate multiple graphs stored on plot
            self.update_title(self.legend_idxs)
            self.rst_boxes(autozoom=True) # Reset zoom
            # Cycle through to the next color for the next dataset
            self.last_color_idx = (self.last_color_idx + 1) % len(COLOR_ARR)
            # self.ifs becomes ready for the next dataset
            self.x_data = []
            self.y_data = []
            self.ifs.setData(self.x_data, self.y_data)
            self.load_label.setText("Up to date")

    def clear_graph(self):
        """
        Clears all graphs from the current plot, as well as the parameter in the parameter box. Removes any
        material table and replaces it with a single material dropdown box.

        :param self: MainWindow object
        """
        # Stop timer if running
        self.monitor_timer.stop()
        # Check to ensure a process isn't already running, and if it is, kill it
        if self.proc is not None:
            self.load_label.setText("Cancelling...")
            qt.QApplication.processEvents() # Force a screen update to render the status label
            self.proc.terminate() # Send the SIGTERM signal to the process
            self.proc.join()
            self.proc = None

        self.multiple = False # No graphs means no multiple
        self.param_str = '' # Reset input box
        self.input_box.setText(self.param_str)
        self.update_title([]) # Update title to default. [] ensures that no parameters are listed as varying
        # Reset temporary plot
        self.x_data = []
        self.y_data = []
        self.ifs.setData(self.x_data, self.y_data)
        self.last_color_idx = 0 # Set temporary plot color to red
        self.graph_widget.clearPlots() # Erase all permanent plots
        self.legend = None # Erase stored legend
        # Erase all base parameters and varying parameters
        self.legend_idxs = []
        self.base_params = [[None, None, 'Molecular', None, None, None, None, None, None, None, [None, None], None, [None, None]]]

        self.all_borders('black') # Remove all error red borders

        # Remove material table and mixture model toggle if present
        if self.mat_table is not None:
            # Erase all compositions except the first
            self.comps = [[qt.QComboBox(), qt.QLineEdit()]]
            self.comps[0][0].addItems(COMPS)
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
            self.layout.addWidget(self.comp_labels[0], 6, 0, 1, 2)
            self.layout.addWidget(self.comps[0][0], 6, 1, 1, 2)

        self.rst_boxes(autozoom=True) # Re-zoom to fit
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
        # Open the file save dialogue box. Here user can specify the path they want to save the graph to
        file_path, _ = qt.QFileDialog.getSaveFileName(
            self, 
            "Save Plot As", 
            "", 
            "PNG Files (*.png);;All Files (*)"
        )

        # Check if the user actually selected a path (or hit 'Cancel')
        if file_path:
            file_path = os.path.abspath(file_path)
            # Remove any file extension
            file_path.replace('.png', '')
            file_path.replace('.jpg', '')

            # Use PyQtGraph's exporter to save the widget content
            self.exporter = pg.exporters.ImageExporter(self.graph_widget.plotItem)
            self.exporter.export(file_path + '.png')

            # Write information about the constants used to generate the graphs to a CSV in the same directory
            caption = self.generate_caption() # Get all the unchanging parameters
            with open(file_path + '_params.csv', mode='w', newline='') as file:
                writer = csv.writer(file)
                for key, value in caption.items():
                    writer.writerow([key, value])

            self.all_borders('black') # Reset all input boxes to black border
            
            # Confirm whether file saved successfully & set load label to display that
            if os.path.exists(file_path):
                self.load_label.setText(f"Graph saved to {file_path}.png. Parameters saved to {file_path}_params.csv.")
            else:
                self.load_label.setText(f"File failed to save.")

    def update_title(self, title_elements):
        """
        A function that updates the graph title dynamically based on current parameter values

        :param self: MainWindow object
        :param title_elements: Array of integer indices to indicate which columns from base_params 
                               to include in the title. Shape len(self.legend_idxs)
        """
        title_str = ''

        title_str += self.y_axis.currentText() + " vs. " # First add the y-axis variable being changed
        title_str += f"{('Scattering Angle ' if self.x_axis.currentText() == 'Scattering Angle' else 'Phase Angle ') if self.graphmode == 2 else 'Wavelength '}" # Then add the x-axis variable

        # If multiple, legend_idxs likely has a nonzero length and will usually be the parameter passed to this
        # program. Loop through and determine where the plots are varying, then include those in the title.
        if self.multiple:
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
        else:
            title_str += 'at ' + self.param_str
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

        self.graph_widget.setTitle(title_str, color='k') # Update the graph title

    def track_changes(self):
        """
        A function that reads the current state of all input fields, compares it to the previous state, and outputs
        the current state and the difference.

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
                if not len(value) == 0 and not float(value) == 0:
                    compositions[key] = value
            except ValueError: # In case of any error, skip the material
                continue

        # If multiple compositions, read mixture model, else grab the previous value in the base parameters
        if len(self.comps) > 1:
            mix = self.mixture_model.checkedButton().text()
        else:
            # This could be either areal or molecular. In single-material mixtures it doesn't matter which
            # one is selected, so we match it to the first base_params entry to avoid the title saying we are
            # varying mixture model upon adding single-material data.
            mix = self.base_params[-1][2]

        # If no control over sizedist data, then read distribution type and grab from vars file
        if self.smin_label is None:
            datamodel_params = vars.SIZEDISTS[self.dist_model.currentText()][0]
        # If control over sizedist data, read from appropriate boxes, but only if the user can change them
        elif not self.smin_label.isHidden():
            datamodel_params = [float(self.smin_box.text()), float(self.smax_box.text()), float(self.powlaw_box.text())]
            # r, G, and x0 boxes may or may not be visible depending on scattering theory; always append 'NaN' as a backup
            if self.r_label is not None and not self.r_label.isHidden():
                datamodel_params.append(float(self.r_box.text()))
            else:
                datamodel_params.append('NaN')
            
            if self.G_label is not None and not self.G_label.isHidden():
                datamodel_params.append(float(self.G_box.text()))
            else:
                datamodel_params.append('NaN')
            
            if self.x0_label is not None and not self.x0_label.isHidden():
                datamodel_params.append(float(self.x0_box.text()))
            else:
                datamodel_params.append('NaN')
        # Default case is all NaNs. This is only really for surface reflectance.
        else:
            datamodel_params = ['NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'Nan']

        # Only grab size distribution resolution if resolution is active
        if self.window_button is not None:
            res = [self.spacing.currentText(), self.dist_slider.value() / 100 if self.spacing.currentText() == 'Logarithmic' else self.dist_slider.value()]
        else:
            res = ['NaN', 'NaN']

        # Only grab tau if changeable, i.e. if y-axis is reflectance
        if not self.tau_label.isHidden():
            tau = self.tau_box.text()
        else:
            tau = 'NaN'

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
                          self.fit_model.currentText(),
                          res,
                          tau,
                          [self.input_box.text(), self.input_label.currentText()]]

        # Check where the new parameters do not match the base ones
        update_idxs = []
        for i, element in enumerate(updated_params):
            if not element == self.base_params[0][i]:
                update_idxs.append(i)

        # Search all but the last part of base_params (which could match updated_params)
        # If the array is found to match some previous entry in the graph, clear the return array
        # This will trigger a special kind of error in the rest of the code
        if updated_params in self.base_params[:-1]:
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
        A function that loops through an existing legend items object and updates each element in the legend based
        on the current state of legend_idxs, i.e. what variables in base_params are changing and need to be
        accounted for.
        
        :param self: MainWindow object
        :param legend_idxs: Array of indices in base_params indicating which parameters are varying. 
                            Shape (len(self.base_params[0]))
        """
        if len(legend_idxs) > 0: # Only loop if there are parameters to include in legend
            # First get a slice of base_params to figure out if different angles are being displayed
            angle_slice = [entry[-1][1] for entry in self.base_params]
            multiple_angle_types = False
            # Check for the different kinds of angles. If both exists, we will display them in the legend
            if self.graphmode == 0 and 'Scattering Angle (°):' in angle_slice and 'Phase Angle (°):' in angle_slice:
                multiple_angle_types = True
            # Loop through each index in legend
            for i in range(len(self.legend.items)):
                legend_str = ''
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
                                self.legend.items[i][1].setText(legend_str) # Update legend item
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
                    self.legend.items[i][1].setText(legend_str) # Update legend item
                    idx += 1 # Increment counter

    def generate_caption(self):
        """
        A method that takes all parameters that were not modified across all plots and lists
        them and their values in a caption-like string.

        :param self: MainWindow object
        :returns: Dictionary containing all unmodified base parameters and their values
        """
        # Find all the indices where the legend idxs and the parameter list don't match
        # This is where the unchanged parameters are
        param_idxs = np.arange(0, len(self.base_params[0]), 1)
        mask = ~np.isin(param_idxs, self.legend_idxs) # Filter out which parameters aren't in the legend idxs
        unchanged_idxs = np.where(mask)[0]

        caption = {} # Create an empty dictionary to hold constants
        for idx in unchanged_idxs:
            item = self.base_params[0][idx] # The first entry in base_params will suffice for constant parameters
            if not (item == 'NaN' or item == 'None'): # Ensure no NaNs/irrelevant parameters get added
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
                    caption['Minimum Particle Radius (μm)'] = item
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
        A function that adds a material dropdown list and its corresponding v/v entry box to the table of
        materials. If no table is present, create it and the mixture model toggle first.

        :param self: MainWindow object
        """ 
        # If no table or radio buttons, add them. Original material will be added to the table first
        if len(self.comps) == 1:
            # Setting up radio buttons to toggle between areal and molecular mixture types
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
            mix_row_idx = self.row_widget(self.dist_label)
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
            self.layout.addWidget(self.mat_table, 6, 0, 1, 3)

        # Add the additional material whether table was just created or not
        # Ensure composition array is updated
        self.comps.append([qt.QComboBox(), qt.QLineEdit()])
        self.comps[-1][0].addItems(COMPS)
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
        A function that reads the current state of all material input fields and ensures they follow proper
        formatting.

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
                    if float_val == 0:
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
        if not fraction_sum == 1:
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
        A function that locates the row index of a given widget in a layout.

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
        A function that shifts all widgets at or below rowstart up or down a number of rows indicated by rowshift.

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
        if self.smin_label is not None:
            self.smin_box.setStyleSheet(f"border: 1px solid {color};")
            self.smax_box.setStyleSheet(f"border: 1px solid {color};")
            self.powlaw_box.setStyleSheet(f"border: 1px solid {color};")
            if self.r_label is not None:
                self.r_box.setStyleSheet(f"border: 1px solid {color};")
            if self.G_label is not None:
                self.G_box.setStyleSheet(f"border: 1px solid {color};")
            if self.x0_label is not None:
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
            if self.base_params[-1][0] == None:
                # Default zoom if graph has just been cleared (base_params will have 1 row full of NoneTypes)
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
        A function that updates the bounds of the graph widget based on user input values for xmin, xmax, ymin,
        and ymax.

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
        except ValueError:
            err_string += 'Min X must be a float. '

        try:
            xmax = float(xmax)
            # Ensure xmin is a float before checking this error
            if 'Min X' not in err_string and xmax <= xmin:
                err_string += 'Max X must be greater than min X. '
        except ValueError:
            err_string += 'Max X must be a float. '

        try:
            ymin = float(ymin)
        except ValueError:
            err_string += 'Min Y must be a float. '

        try:
            ymax = float(ymax)
            if 'Min Y' not in err_string and ymax <= ymin:
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
        A method to display boxes for modification of minimum and maximum particle sizes and power law when
        the 'Custom' size distribution is selected. If another size distribution is selected, update graph instead.

        :param self: MainWindow object
        """
        # Check to see if the Custom option is selected and the widgets haven't been created yet
        # Henyey-Greenstein graphs cannot have custom size distributions
        if self.dist_model.currentText() == 'Custom' and self.smin_label is None and not self.fit_model.currentText() == 'Henyey-Greenstein':
            self.smin_label = qt.QLabel("Min size (μm):")
            self.smin_box = qt.QLineEdit()
            self.smin_box.setPlaceholderText("Input min size in microns")
            self.smax_label = qt.QLabel("Max size (μm):")
            self.smax_box = qt.QLineEdit()
            self.smax_box.setPlaceholderText("Input max size in microns")
            self.powlaw_label = qt.QLabel("Power Law (dim):")
            self.powlaw_box = qt.QLineEdit()
            self.powlaw_box.setPlaceholderText("Input power law")

            # Connect the boxes to the update function
            self.smin_box.returnPressed.connect(self.update_graph)
            self.smax_box.returnPressed.connect(self.update_graph)
            self.powlaw_box.returnPressed.connect(self.update_graph)
            self.smin_box.setStyleSheet("border: 1px solid black;")
            self.smax_box.setStyleSheet("border: 1px solid black;")
            self.powlaw_box.setStyleSheet("border: 1px solid black;")

            # Make some new rows and add the widgets
            size_params_row = self.row_widget(self.fit_label)
            self.mknewrow(size_params_row, 3)

            self.layout.addWidget(self.smin_label, size_params_row, 0, 1, 1)
            self.layout.addWidget(self.smin_box, size_params_row, 1, 1, 2)
            self.layout.addWidget(self.smax_label, size_params_row + 1, 0, 1, 1)
            self.layout.addWidget(self.smax_box, size_params_row + 1, 1, 1, 2)
            self.layout.addWidget(self.powlaw_label, size_params_row + 2, 0, 1, 1)
            self.layout.addWidget(self.powlaw_box, size_params_row + 2, 1, 1, 2)
        # Switching away from Custom model will not hide but CLEAR sizedist boxes
        else:
            if self.smin_label is not None:
                # Shift rows up
                self.mknewrow(self.row_widget(self.smin_label) + 3, -3)
                # Remove widgets from display
                self.layout.removeWidget(self.smin_label)
                self.layout.removeWidget(self.smin_box)
                self.layout.removeWidget(self.smax_label)
                self.layout.removeWidget(self.smax_box)
                self.layout.removeWidget(self.powlaw_label)
                self.layout.removeWidget(self.powlaw_box)

                # Set to none so they actually remove
                self.smin_label = None
                self.smin_box = None
                self.smax_label = None
                self.smax_box = None
                self.powlaw_label = None
                self.powlaw_box = None
        # Update regardless of if boxes were removed or not
        # change_fitmodel updates off of change dist model because change_fitmodel does the work of checking if r,
        # G, and x0 should show in the GUI based on what the scattering theory allows
        self.change_fitmodel(update=False)

    def change_fitmodel(self, update=True):
        """
        A function to intercept a change in scattering theory, mainly to update available options for parameters to vary in the size
        distributions model if a custom distribution is selected.

        :param self: MainWindow object
        :param update: Boolean, whether to call update_graph on finish. Is overrided to True if not a custom size
                       distribution.
        """
        # If Henyey-Greenstein is selected and Custom sizedist is selected, revert to G-ring-like and remove custom boxes
        if self.fit_model.currentText() == 'Henyey-Greenstein' and self.dist_model.currentText() == 'Custom':
            self.dist_model.setCurrentText('G-ring-like')
            self.change_distmodel() 

        # Remove the window change button if it shouldn't be there. Otherwise, add it after the fit model
        if self.fit_model.currentText() == 'Semi-Empirical Mie' or self.fit_model.currentText() == 'Pure Mie' or self.fit_model.currentText() == 'Mie Diffraction':
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
        if self.r_label is not None:
            self.mknewrow(self.row_widget(self.r_label) + 1, -1)
            self.layout.removeWidget(self.r_label)
            self.r_label = None
            self.layout.removeWidget(self.r_box)
            self.r_box = None
        if self.G_label is not None:
            self.mknewrow(self.row_widget(self.G_label) + 1, -1)
            self.layout.removeWidget(self.G_label)
            self.G_label = None
            self.layout.removeWidget(self.G_box)
            self.G_box = None
        if self.x0_label is not None:
            self.mknewrow(self.row_widget(self.x0_label) + 1, -1)
            self.layout.removeWidget(self.x0_label)
            self.x0_label = None
            self.layout.removeWidget(self.x0_box)
            self.x0_box = None
        # Re-add any boxes relevant to the scattering theory & size distribution
        if self.smin_label is not None:

            # Examine the current scattering theory to see if it allows for modification of G, r, or x0
            fit = self.fit_model.currentText()
            if fit == "Semi-Empirical Mie":
                # Semi-Empirical Mie requires r, G, and x0
                self.r_label = qt.QLabel("R (dim):")
                self.r_box = qt.QLineEdit()
                self.r_box.setPlaceholderText("Input r value")
                self.G_label = qt.QLabel("G (dim):")
                self.G_box = qt.QLineEdit()
                self.G_box.setPlaceholderText("Input G value")
                self.x0_label = qt.QLabel("x0 (dim):")
                self.x0_box = qt.QLineEdit()
                self.x0_box.setPlaceholderText("Input x0 value")
                self.r_box.returnPressed.connect(self.update_graph)
                self.G_box.returnPressed.connect(self.update_graph)
                self.x0_box.returnPressed.connect(self.update_graph)
                self.r_box.setStyleSheet("border: 1px solid black;")
                self.G_box.setStyleSheet("border: 1px solid black;")
                self.x0_box.setStyleSheet("border: 1px solid black;")

                # Add the row right after the size distribution boxes
                r_row = self.row_widget(self.powlaw_label) + 1
                self.mknewrow(r_row, 3)
                self.layout.addWidget(self.r_label, r_row, 0, 1, 1)
                self.layout.addWidget(self.r_box, r_row, 1, 1, 2)
                self.layout.addWidget(self.G_label, r_row + 1, 0, 1, 1)
                self.layout.addWidget(self.G_box, r_row + 1, 1, 1, 2)
                self.layout.addWidget(self.x0_label, r_row + 2, 0, 1, 1)
                self.layout.addWidget(self.x0_box, r_row + 2, 1, 1, 2)

            elif fit == "Mie Diffraction":
                # Mie Diffraction only requires r
                self.r_label = qt.QLabel("R:")
                self.r_box = qt.QLineEdit()
                self.r_box.setPlaceholderText("Input r value")
                self.r_box.returnPressed.connect(self.update_graph)
                self.r_box.setStyleSheet("border: 1px solid black;")

                # Add the row right after the size distribution boxes
                r_row = self.row_widget(self.powlaw_label) + 1
                self.mknewrow(r_row, 1)
                self.layout.addWidget(self.r_label, r_row, 0, 1, 1)
                self.layout.addWidget(self.r_box, r_row, 1, 1, 2)

            elif fit == "Mie Transmission":
                # Mie Transmission only requires G
                self.G_label = qt.QLabel("G:")
                self.G_box = qt.QLineEdit()
                self.G_box.setPlaceholderText("Input G value")
                self.G_box.returnPressed.connect(self.update_graph)
                self.G_box.setStyleSheet("border: 1px solid black;")

                # Add the row right after the size distribution boxes
                G_row = self.row_widget(self.powlaw_label) + 1
                self.mknewrow(G_row, 1)
                self.layout.addWidget(self.G_label, G_row, 0, 1, 1)
                self.layout.addWidget(self.G_box, G_row, 1, 1, 2)

        # The update optional parameter is there to prevent update graph from getting called when the new sizedist
        # boxes are added, so the boxes don't instantly turn red. But if those boxes aren't there, we force
        # an update anyways.
        if update or not self.dist_model.currentText() == 'Custom':
            self.update_graph() # Lastly, update the graph

    def change_axis(self):
        """
        A method to change the Y axis of the graph between reflectance and phase function (phase function is 
        reflectance normalized such that the integral over all solid angles is 1).
        
        :param self: MainWindow object
        """
        if self.y_axis.currentText() == 'Reflectance':
            self.graph_widget.setLabel('left', 'Reflectance', units='I/F', color='k') # Units are I/F
            # Henyey-Greenstein gets added because Reflectance supports it
            if not self.graphmode == 0 and self.fit_model.findText("Henyey-Greenstein") < 0:
                self.fit_model.addItems(["Henyey-Greenstein"]) # Not possible to have multiple, since text has to change to trigger this
            # Show tau again because reflectance depends on it
            if self.tau_label.isHidden():
                tau_row = self.row_widget(self.fit_label) + 1
                self.mknewrow(tau_row, 1)
                self.layout.addWidget(self.tau_label, tau_row, 0, 1, 1)
                self.layout.addWidget(self.tau_box, tau_row, 1, 1, 2)
                self.tau_label.setVisible(True)
                self.tau_box.setVisible(True)

        elif self.y_axis.currentText() == 'Phase Function':
            self.graph_widget.setLabel('left', 'Phase Function', units='dim', color='k') # Units are dimensionless
            # Remove Henyey-Greenstein from the y-axis dropdown list since it is not supported in phase function plots
            # If it's in there, it will switch to the first option in the list (by default it would switch to the option above HG)
            # If it's not, the remove will fail silently
            if self.fit_model.currentText() == "Henyey-Greenstein":
                self.fit_model.setCurrentText("Semi-Empirical Mie")
            self.fit_model.removeItem(self.fit_model.findText("Henyey-Greenstein"))

            # Hide tau, phase function does not depend on it
            if not self.tau_label.isHidden():
                tau_row = self.row_widget(self.tau_label)
                self.mknewrow(tau_row + 1, -1)
                self.layout.removeWidget(self.tau_label)
                self.layout.removeWidget(self.tau_box)
                self.tau_label.setVisible(False)
                self.tau_box.setVisible(False)

        elif self.y_axis.currentText() == 'Intensity':
            self.graph_widget.setLabel('left', 'Intensity', units='dim', color='k') # Units are dimensionless
            # Remove Henyey-Greenstein from the y-axis dropdown list since it is not supported in intensity plots
            # If it's in there, it will switch to the first option in the list (by default it would switch to the option above HG)
            # If it's not, the remove will fail silently
            if self.fit_model.currentText() == "Henyey-Greenstein":
                self.fit_model.setCurrentText("Semi-Empirical Mie")
            self.fit_model.removeItem(self.fit_model.findText("Henyey-Greenstein"))

            # Hide tau, intensity does not depend on it
            if not self.tau_label.isHidden():
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
        A function that triggers whenever the input label is changed. All it does is change the default text when
        the input box is empty. Then it updates the graph if requested.

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
        A function that cycles between the data input window and the resolution settings window
        when the button is available.

        :param self: MainWindow object
        """
        self.stack.setCurrentIndex((self.stack.currentIndex() + 1) % 2) # Show data entry screen by default
        if self.stack.currentIndex() == 0:
            # Add the load label in its proper place
            self.layout.addWidget(self.load_label, self.row_widget(self.y_scale_label) + 1, 0, 1, 3)
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
        A function that handles updates to the spacing mode of the graph, mainly just to change
        the bounds of the slider.

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

    def update_slider(self, value, dec=False):
        """
        A function that handles changes in the size distribution slider, updating the displayed value in
        slider_label and displaying a warning message that the graph has not been updated.

        :param self: MainWindow object
        :param value: The integer current position of the slider
        :param dec: Whether to display the number divided by 100. Necessary to allow decimal values.
        """
        self.slider_label.setText(f"{'Number of Sizes' if self.spacing.currentText() == 'Linear' else 'Log Base'}: {value if dec == False else value/100}")
        self.load_label.setText("Graph is not updated. Edit another field to update graph.")

    def set_enabled(self, enable):
        """
        A function that disables/enables all widgets in self.layout based on the boolean value of enable.

        :param self: MainWindow object
        :param enable: Boolean. True to enable all widgets, False to disable them.
        """

        for i in range(self.layout.count()):
            item = self.layout.itemAt(i) # Get each item in the layout
            # If item is a widget, get the widget
            if item and item.widget():
                item.widget().setEnabled(enable)

    def finish_process(self):
        if self.proc is None or not self.proc.is_alive():
            # If the process is no longer alive, signal it has ended
            self.monitor_timer.stop() # Stop the timer if process is not running
            if self.proc:
                self.proc.join()
                self.proc = None

            # Grab results from the queue
            if not result_queue.empty():
                results = result_queue.get()
                self.x_data = results["x_data"]
                self.y_data = results["y_data"]
                
                # Plot the dataset if not in the graph, else change the current unsaved graph
                if self.ifs not in self.graph_widget.plotItem.items:
                    self.ifs = self.graph_widget.plot(self.x_data, self.y_data, pen=self.plt_pen, name=(f"{self.param_str}{' μm' if (self.graphmode == 1 or self.graphmode == 2) else '°'}" if self.multiple else None))
                else:
                    self.ifs.setData(self.x_data, self.y_data) # self.ifs variable allows for updating of the temporary graph

                # Update the legend if multiple graphs
                if self.multiple:
                    self.legend.removeItem(self.ifs) # Need to do remove and add because there's no setter function
                    self.legend.addItem(self.ifs, '') # Add a blank placeholder for self.ifs so legend setter function has a full array to loop through
                    self.generate_legend(self.temp_legend_idxs) # Update the full legend. The last element is our new temporary plot

                self.update_title(self.temp_legend_idxs) # Update the title showing variance across all current plots
                self.rst_boxes(autozoom=True) # Reset zoom

                # Clean up process
                self.proc = None

                self.load_label.setText("Up to date")
            else:
                self.load_label.setText("Computation failed. Please try again.")
        
class StableTable(qt.QTableWidget):
    """
    Sets up a safe extension of PyQt6's QTableWidget that disables all potentially problematic keyboard
    interactions while within the table, intercepting any events that could lead to segmentation faults.
    """
    def keyPressEvent(self, event):
        """
        A function overriding QTableWidget's normal keyPressEvent function, intercepting any key press made while
        inside the table and blocking it if it would attempt to move between cells and cause a segmentation fault.

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
