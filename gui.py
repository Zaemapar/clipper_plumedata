import sys
import pyqtgraph as pg
import pyqtgraph.exporters
import PyQt6.QtWidgets as qt
import PyQt6.QtCore as qc
import pred_vars as vars
import utils
import os
import numpy as np

COLOR_ARR = ['red', 'magenta', 'orange', 'green', 'blue', 'violet'] # Array of graph colors to cycle through
COMPS = utils.get_available_materials() # Call function to parse data directory to get all available materials

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
        self.setWindowTitle("Clipper Instrument Spectrum Prediction")
        self.resize(1200, 800)

        # QMainWindow requires a central widget to hold everything else
        self.central_widget = qt.QWidget()
        self.setCentralWidget(self.central_widget)

        # We will use a Vertical Box Layout (QVBoxLayout) to stack widgets top-to-bottom
        self.layout = qt.QGridLayout()
        self.layout.setContentsMargins(30, 30, 30, 30) # Making sure things don't overlap each other
        self.layout.setSpacing(20)
        self.central_widget.setLayout(self.layout)

        # --- WIDGETS ---

        # Push button to toggle between reflectance vs wavelength graphs and reflectance vs scattering angle graphs
        # Indicates what type of graph is being shown. 0 is IFs vs wavelengths, 1 is surface albedos vs wavelengths, 2 is IFs vs scattering angles
        self.graphmode = 0
        self.graph_mode_button = qt.QPushButton("Reflectance vs. Wavelength")

        # Dropdown list for instruments on Europa Clipper
        self.instrument_label = qt.QLabel("Instrument:")
        self.instrument = qt.QComboBox()
        self.instrument.addItems(vars.INSTRUMENTS.keys()) # Add from variables file

        # Dropdown list for available material compositions
        self.comp_labels = [qt.QLabel("Composition:"), qt.QLabel("Volume Fraction:")]
        self.mat_table = None # A table will be added if we add multiple materials to the composition
        # Each material in a composition will have a dropdown list to specify the material,
        # plus an entry field for the corresponding volume fraction
        self.comps = [[qt.QComboBox(), qt.QLineEdit()]]
        self.comps[0][0].addItems(COMPS)
        self.comps[0][1].setText('1.0') # The default volume fraction (not shown) is 1.0, since there is only one material
        # Determining the minimum and maximum bounds for wavelength based on available material data
        # Instrument wavelength ranges will be far more limiting, but good to include this nonetheless
        self.comp_min_wavel, self.comp_max_wavel = utils.get_minmax_wavel(self.comps[0][0].currentText())

        # Dropdown list for available data models (G-ring, E-ring, etc)
        self.dist_label = qt.QLabel("Data Model:")
        self.dist_model = qt.QComboBox()
        self.dist_model.addItems(vars.DATA_MODELS)

        # Radio buttons to toggle between Mie and Henyey-Greenstein theory, or just Mie for reflectance vs wavelength plots
        self.fit_label = qt.QLabel("Fit Model:")
        self.model_a = qt.QRadioButton("Mie")
        self.model_a.setChecked(True) # By default, Mie is used, mainly because ref vs. wavel can't use HG
        self.model_b = qt.QRadioButton("Henyey-Greenstein")
        self.fit_model = qt.QButtonGroup() # Setting up a group to ensure only one can be checked
        self.fit_model.addButton(self.model_a)
        self.fit_model.addButton(self.model_b)
        self.model_b.setVisible(False) # Default is reflectance vs wavelenght, therefore Henyey-Greenstein is not visible

        # Entry box for manual optical depth input
        self.tau_label = qt.QLabel("Optical Depth:")
        self.tau_box = qt.QLineEdit()
        self.tau_box.setPlaceholderText("Input optical depth") # Switching graph modes will clear this box, so this needs to display
        self.tau_box.setText("1e-6") # Standard order of magnitude for optical depth

        # Entry box for varying parameter, either scattering angle (for ref vs. wavel) or wavelength (for ref vs. theta)
        self.param_str = ''
        self.input_label = qt.QLabel("Scattering Angle:")
        self.input_box = qt.QLineEdit()
        self.input_box.setPlaceholderText("Input scattering angle in degrees")
        self.input_box.setText("90")

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
        self.graph_widget.showGrid(x=True, y=True)
        self.graph_widget.setLabel('left', 'Reflectance', units='I/F', color='k')
        self.graph_widget.setLabel('bottom', 'Wavelength', units='μm', color='k')
        self.graph_widget.getAxis('bottom').setPen(grid_pen) # Pen ensures axes and numbers are black and thick
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
        self.xmin_box = qt.QLineEdit()
        self.xmin_box.setMaximumWidth(50)
        self.xmax_label = qt.QLabel("Max X:")
        self.xmax_box = qt.QLineEdit()
        self.xmax_box.setMaximumWidth(50)
        self.ymin_label = qt.QLabel("Min Y:")
        self.ymin_box = qt.QLineEdit()
        self.ymin_box.setMaximumWidth(50)
        self.ymax_label = qt.QLabel("Max Y:")
        self.ymax_box = qt.QLineEdit()
        self.ymax_box.setMaximumWidth(50)

        # --- WIDGET FUNCTIONALITY ---
        # Pressing the graph mode button toggles between ref vs wavel and ref vs angle
        self.graph_mode_button.clicked.connect(self.mode_toggle)

        # Pressing scale radio buttons updates scale
        self.x_scale.buttonClicked.connect(self.update_scale)
        self.y_scale.buttonClicked.connect(self.update_scale)

        # Changing anything about the dataset (instrument, comp, model, tau, etc) replots the data
        self.instrument.currentTextChanged.connect(self.update_graph)
        self.dist_model.currentTextChanged.connect(self.update_graph)
        self.fit_model.buttonClicked.connect(self.update_graph)
        self.tau_box.returnPressed.connect(self.update_graph)
        self.input_box.returnPressed.connect(self.update_graph)

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
        
        # Create a flag to track if the user actually dragged the graph
        self.graph_was_dragged = False
        # Listen for manual range changes (fires continuously during a drag)
        def flag_dragged():
            self.graph_was_dragged = True
        self.graph_widget.getViewBox().sigRangeChangedManually.connect(flag_dragged)
        # Store the original mouse release to avoid breaking PyQt6's internal math
        self._original_mouse_release = self.graph_widget.mouseReleaseEvent

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
            
        # Override default mouse release event
        self.graph_widget.mouseReleaseEvent = click_release

        # --- LAYOUT ---
        # Items are arranged in (row, column, rowspan, columnspan)
        self.layout.addWidget(self.graph_mode_button, 0, 0, 1, 3)
        self.layout.addWidget(self.add_material_button, 1, 0, 1, 3)
        self.layout.addWidget(self.add_graph_button, 2, 0, 1, 3)
        self.layout.addWidget(self.clear_plot_button, 3, 0, 1, 3)
        self.layout.addWidget(self.save_plot_button, 4, 0, 1, 3)
        self.layout.addWidget(self.instrument_label, 5, 0, 1, 1)
        self.layout.addWidget(self.instrument, 5, 1, 1, 2)
        self.layout.addWidget(self.graph_widget, 0, 3, 16, 16)
        self.layout.addWidget(self.home_button, 1, 4, 1, 1)
        self.layout.addWidget(self.xmin_label, 1, 15, 1, 1)
        self.layout.addWidget(self.xmin_box, 1, 16, 1, 1)
        self.layout.addWidget(self.xmax_label, 1, 17, 1, 1)
        self.layout.addWidget(self.xmax_box, 1, 18, 1, 1)
        self.layout.addWidget(self.ymin_label, 2, 15, 1, 1)
        self.layout.addWidget(self.ymin_box, 2, 16, 1, 1)
        self.layout.addWidget(self.ymax_label, 2, 17, 1, 1)
        self.layout.addWidget(self.ymax_box, 2, 18, 1, 1)

        # These two will get replaced by a table if multi-material mixture is enabled
        self.layout.addWidget(self.comp_labels[0], 6, 0, 1, 1)
        self.layout.addWidget(self.comps[0][0], 6, 1, 1, 2)

        self.layout.addWidget(self.dist_label,  7, 0, 1, 1)
        self.layout.addWidget(self.dist_model, 7, 1, 1, 2)
        self.layout.addWidget(self.fit_label, 8, 0, 1, 1)
        self.layout.addWidget(self.model_a, 8, 1, 1, 1)
        self.layout.addWidget(self.model_b, 8, 2, 1, 1)
        self.layout.addWidget(self.tau_label, 9, 0, 1, 1)
        self.layout.addWidget(self.tau_box, 9, 1, 1, 2)
        self.layout.addWidget(self.input_label, 10, 0, 1, 1)
        self.layout.addWidget(self.input_box, 10, 1, 1, 2)
        self.layout.addWidget(self.x_scale_label, 11, 0, 1, 1)
        self.layout.addWidget(self.x_linear, 11, 1, 1, 1)
        self.layout.addWidget(self.x_log, 11, 2, 1, 1)
        self.layout.addWidget(self.y_scale_label, 12, 0, 1, 1)
        self.layout.addWidget(self.y_linear, 12, 1, 1, 1)
        self.layout.addWidget(self.y_log, 12, 2, 1, 1)
        self.layout.addWidget(self.load_label, 13, 0, 2, 2)
        
        # Base parameters track conditions of each plot on the graph, used for tracking changes
        self.base_params = [[self.instrument.currentText(),
                    {self.comps[0][0].currentText(), self.comps[0][1].text()},
                    'Molecular', # Mixture model is not displayed at first, so must set to default
                    self.dist_model.currentText(),
                    self.fit_model.checkedButton().text(),
                    self.tau_box.text(),
                    self.input_box.text()]]
        self.legend_idxs = [] # Array to hold base parameter indexes to include in the legends, i.e. those that have changed

        self.update_graph() # There are default settings in the input boxes, so output the first graph
        self.update_title(self.legend_idxs)
        self.all_borders('black') # Set all input box borders to black

    def mode_toggle(self):
        """
        A function that defines the behavior of the graph mode button when clicked. Toggles between
        'Reflectance vs. Wavelength' and 'Reflectance vs. Scattering Angle.' Clears graph in the process, 
        resets input fields, and updates title

        :param self: MainWindow object
        """
        # First ensure all missing widgets are displayed upon a mode change
        if self.graphmode == 1:
            self.mknewrow(7, 3)
            self.layout.addWidget(self.dist_label,  7, 0, 1, 1)
            self.layout.addWidget(self.dist_model, 7, 1, 1, 2)
            self.layout.addWidget(self.fit_label, 8, 0, 1, 1)
            self.layout.addWidget(self.model_a, 8, 1, 1, 1)
            self.layout.addWidget(self.model_b, 8, 2, 1, 1)
            self.layout.addWidget(self.tau_label, 9, 0, 1, 1)
            self.layout.addWidget(self.tau_box, 9, 1, 1, 2)
            self.dist_label.setVisible(True)
            self.dist_model.setVisible(True)
            self.fit_label.setVisible(True)
            self.model_a.setVisible(True)
            self.model_b.setVisible(True)
            self.tau_label.setVisible(True)
            self.tau_box.setVisible(True)


        self.graphmode = int((self.graphmode + 1) % 3) # Switch the graph mode variable
        if self.graphmode == 0:
            self.graph_mode_button.setText("Reflectance vs. Wavelength")
            self.input_label.setText("Scattering Angle:")
            self.input_box.setPlaceholderText("Input scattering angle in degrees")
            self.graph_widget.setLabel('bottom', 'Wavelength', units='μm')
            self.model_b.setVisible(False) # Reflectance vs. wavelength does not have Henyey-Greenstein functionality
            self.model_a.setChecked(True) # So we set it automatically to the remaining Mie option
        elif self.graphmode == 1:
            self.graph_mode_button.setText("Surface Reflectance vs. Wavelength")
            self.input_label.setText("Effective Grain Size:")
            self.input_box.setPlaceholderText("Input effective grain size in microns")
            self.graph_widget.setLabel('bottom', 'Wavelength', units='μm')

            # Hide all unnecessary widgets (only the input param is needed)
            self.layout.removeWidget(self.dist_label)
            self.dist_label.setVisible(False)
            self.layout.removeWidget(self.dist_model)
            self.dist_model.setVisible(False)
            self.layout.removeWidget(self.fit_label)
            self.fit_label.setVisible(False)
            self.layout.removeWidget(self.model_a)
            self.model_a.setVisible(False)
            self.layout.removeWidget(self.model_b)
            self.model_b.setVisible(False)
            self.layout.removeWidget(self.tau_label)
            self.tau_label.setVisible(False)
            self.layout.removeWidget(self.tau_box)
            self.tau_box.setVisible(False)
            self.mknewrow(10, -3)

        elif self.graphmode == 2:
            self.graph_mode_button.setText("Reflectance vs. Scattering Angle")
            self.input_label.setText("Wavelength:")
            self.input_box.setPlaceholderText("Input wavelength in microns")
            self.graph_widget.setLabel('bottom', 'Scattering Angle', units='°') # Change bottom axis label
            self.model_b.setVisible(True) # Reflectance vs. scattering angle has Henyey-Greenstein functionality
        
        self.clear_graph() # Clear graph when finished

    def update_graph(self):
        """
        A function called if any of the data parameters are changed. Calls the plotter function from the main
        program and outputs the new temporary function on the graph, replacing the old function (assuming that function
        wasn't made permanent via 'Add new plot'). If parameters are invalid, updates status message to error message
        and turns the appropriate box red. If multiple graphs involved, updates legend accordingly.

        :param self: MainWindow object
        """
        self.all_borders('black') # Reset any previous errors
        self.load_label.setText("Updating...") # Status label changes to indicate loading
        qt.QApplication.processEvents() # Force a screen update to render the status label and the box borders

        err_message = '' # Will dynamically update to support multiple errors at once
        if len(self.comps) > 1:
            # Update materials checks to make sure there aren't errors specifically in the materials table
            err_message += self.update_materials()


        # Read each parameter from the appropriate widget
        compositions = {}
        sensor = self.instrument.currentText()

        # Error message is displayed at the end; however, we need to check for updated comps here to check for
        # other errors later
        if len(err_message) == 0:
            for i, item in enumerate(self.comps):
                if not float(item[1].text()) == 0: # Compositions with v/v equal to zero don't count in legend updates
                    compositions[item[0].currentText()] = item[1].text()

            # Composition changes require getting which wavelengths are supported for that material
            # First set up extreme cases
            min_limiting_comp = ['', 0]
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

        # Error handling - checking to make sure that all parameters are in bounds
        # Error checking for parameter if it's a wavelength
        if self.graphmode == 2:
            try:
                param = float(self.param_str)
                # Check to make sure it's in range of the sensor AND the index of refraction data for that material
                # material is first because it has to make sure no errors are before it, the only errors before it
                # would be composition errors which would prevent us from getting accurate wavelength bounds anyway
                if len(err_message) == 0:
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
                    if len(err_message) == 0:
                        if param < self.comp_min_wavel[1]:
                            err_message += f'Wavelength is too low for {self.comp_min_wavel[0]}. '
                        elif param > self.comp_max_wavel[1]:
                            err_message += f'Wavelength is too high for {self.comp_max_wavel[0]}. '
                elif self.param_str == 'MAX':
                    param = vars.INSTRUMENTS[sensor][1]
                    self.param_str = str(param)
                    self.input_box.setText(self.param_str)
                    if len(err_message) == 0:
                        if param < self.comp_min_wavel[1]:
                            err_message += f'Wavelength is too low for {self.comp_min_wavel[0]}. '
                        elif param > self.comp_max_wavel[1]:
                            err_message += f'Wavelength is too high for {self.comp_max_wavel[0]}. '
                else:
                    err_message += f'Wavelength must be a float, MIN, or MAX. '
        # Error checking for parameter if it's a scattering angle
        elif self.graphmode == 0:
            try:
                param = int(self.param_str)
                if param < 0 or param > 180:
                    err_message += f'Scattering angle must be between 0 and 180 degrees. '
            except ValueError:
                err_message += f'Scattering angle must be an integer. '
        # Error checking for parameter if it's an effective grain size
        elif self.graphmode == 1:
            try:
                param = float(self.param_str)
                if param < 0:
                    err_message += f'Effective grain size must be nonnegative. '
            except ValueError:
                err_message += f'Effective grain size must be a float. '

        # Error checking for tau
        try:
            tau = float(tau_str)
            if tau < 0:
                err_message += 'Optical depth must be nonnegative. '
        except ValueError:
            err_message += 'Optical depth must be a float. '

        if len(err_message) == 0:
            # Get the current text box values (updated_params), plus any indexes where they're different from the original data
            updated_params, temp_legend_idxs = self.track_changes()
            # temp_legend_idxs is only ever None if there are multiple plots in the window and the current plot
            # is found to match a plot somewhere in the window's history. Not necessarily checking for change
            # but for duplicate plots
            if temp_legend_idxs is None:
                    self.load_label.setText("Plot is identical to existing data")
                    self.ifs.setData([], [])
                    self.legend.removeItem(self.ifs)
                    self.all_borders('red')
            # This function runs whenever a field is clicked. But if it hasn't changed, don't bother with calculations
            elif len(temp_legend_idxs) == 0:
                self.load_label.setText("Plot is identical to previous graph")
                # Generic errors are generally speaking going to turn all the boxes red
                self.all_borders('red')
            else:
                # If the text box values are in fact unique and have no errors, update the last element in the base parameter array
                # The last element represents the current plot
                self.base_params[-1] = updated_params

                # Reuse the same pen color for all updates to the current plot
                self.plt_pen.setColor(pg.mkColor(COLOR_ARR[self.last_color_idx]))
                
                # Get the new x and y datasets if no errors
                self.x_data, self.y_data = utils.output_graph(self.graphmode, updated_params[0], updated_params[1], updated_params[2], updated_params[3], updated_params[4], float(updated_params[6]), float(updated_params[5]), wavelbounds=(self.comp_min_wavel, self.comp_max_wavel))
                
                # Plot the dataset if not in the graph, else change the current unsaved graph
                if self.ifs not in self.graph_widget.plotItem.items:
                    self.ifs = self.graph_widget.plot(self.x_data, self.y_data, pen=self.plt_pen, name=(f"{self.param_str}{' μm' if (self.graphmode == 1 or self.graphmode == 2) else '°'}" if self.multiple else None))
                else:
                    self.ifs.setData(self.x_data, self.y_data)

                # Update the legend if multiple graphs
                if self.multiple:
                    self.legend.removeItem(self.ifs) # Need to do remove + add because there's no setter function
                    self.legend.addItem(self.ifs, '') # Add a blank placeholder for self.ifs so legend setter function has a full array to loop through
                    self.generate_legend(temp_legend_idxs)

                self.update_title(temp_legend_idxs)
                self.rst_boxes(autozoom=True) # Reset zoom
                self.load_label.setText("Up to date")
        else:
            self.load_label.setText(err_message)
            # Scan the error message for problematic variables and turn their boxes red
            if 'Wavelength' in err_message or 'Scattering angle' in err_message or 'Effective grain size':
                self.input_box.setStyleSheet("border: 1px solid red;")
            if 'Optical depth' in err_message:
                self.tau_box.setStyleSheet("border: 1px solid red;")

    def plot_graph(self):
        """
        A function that saves the current x_data, y_data to the graph as a permanent plot and resets the dataset
        for a new plot, setting the plot up to hold multiple datasets. Sets up the legend and updates the title.

        :param self: MainWindow object
        """
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
            self.all_borders('black')

            # Create a legend if none already. Needs to be here since legends are for multiple graphs only
            if self.legend == None:
                self.legend = self.graph_widget.addLegend()
                self.legend.setLabelTextColor('k') # Default is gray, but hard to see

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
                self.legend.items[-1][1].setText(f"{self.base_params[0][-1]}{' μm' if (self.graphmode == 1 or self.graphmode == 2) else '°'}") # Default legend string for first item
            
            self.multiple = True # Set the global variable to indicate multiple graphs stored on plot
            self.update_title(self.legend_idxs)
            self.graph_widget.autoRange() # Auto fit for the shown datasets
            # Cycle through to the next color for the next dataset
            self.last_color_idx = (self.last_color_idx + 1) % len(COLOR_ARR)
            # self.ifs becomes ready for the next dataset
            self.x_data = []
            self.y_data = []
            self.ifs.setData(self.x_data, self.y_data)
            self.load_label.setText("Up to date")

    def clear_graph(self):
        """
        Clears all graphs from the current plot, as well as the parameter in the parameter box.

        :param self: MainWindow object
        """
        self.multiple = False # No graphs means no multiple
        self.param_str = '' # Reset input box
        self.input_box.setText(self.param_str)
        self.update_title([]) # Update title to default
        # Reset temporary plot
        self.x_data = []
        self.y_data = []
        self.ifs.setData(self.x_data, self.y_data)
        self.last_color_idx = 0 # Set temporary plot color to red
        self.graph_widget.clearPlots() # Erase all permanent plots
        self.legend = None # Erase stored legend
        # Erase all base parameters and varying parameters
        self.legend_idxs = []
        self.base_params = [[None, None, 'Molecular', None, None, None, None]]

        # Erase all compositions except the first
        self.comps = [[qt.QComboBox(), qt.QLineEdit()]]
        self.comps[0][0].addItems(COMPS)
        self.comps[0][1].setText('1.0')
        # Reconnect combo box & line edit to update function
        self.comps[0][0].currentTextChanged.connect(self.update_graph)
        self.comps[0][1].returnPressed.connect(self.update_graph)
        # Recompute material wavelength bounds
        self.comp_min_wavel, self.comp_max_wavel = utils.get_minmax_wavel(self.comps[0][0].currentText())

        self.all_borders('black') # Remove all error red borders

        # Remove material table and mixture model toggle if present
        if self.mat_table is not None:
            self.layout.removeWidget(self.mat_table)
            self.layout.removeWidget(self.mixture_model_label)
            self.layout.removeWidget(self.molecular)
            self.layout.removeWidget(self.areal)

            # For some reason, if we don't set these to None, "ghost versions" will appear in the window even after reomved
            # So we must clear them here.
            self.mat_table = None
            self.mixture_model_label = None
            self.molecular = None
            self.areal = None

            # Call row function to shift all elements below toggle up one
            self.mknewrow(7, -1)

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
        # Open the file save dialogue box
        file_path, _ = qt.QFileDialog.getSaveFileName(
            self, 
            "Save Plot As", 
            "", 
            "PNG Files (*.png);;All Files (*)"
        )

        # Check if the user actually selected a path (or hit 'Cancel')
        if file_path:
            file_path = os.path.abspath(file_path)
            if not file_path.endswith('.png'): # Ensure png extension
                file_path += '.png'

            # Use PyQtGraph's exporter to save the widget content
            self.exporter = pg.exporters.ImageExporter(self.graph_widget.plotItem)
            self.exporter.export(file_path)

            # Confirm whether file saved successfully
            if os.path.exists(file_path):
                print(f"File saved to {file_path}")
            else:
                print(f"File failed to save.")

    def update_title(self, title_elements):
        """
        A function that updates the graph title dynamically based on current parameter values

        :param self: MainWindow object
        :param title_elements: Array of integer indices to indicate which columns from base_params 
                               to include in the title. Shape len(self.legend_idxs)
        """
        title_str = ''
        # Base of title changes with graph mode
        if self.graphmode == 0 or self.graphmode == 1:
            title_str += f"{'Surface ' if self.graphmode == 1 else ''}Reflectance vs. Wavelength "
        else:
            title_str += f"Reflectance vs. Scattering Angle "

        # If multiple, legend_idxs likely has a nonzero length and will usually be the parameter passed to this
        # program. Loop through and determine where the plots are varying, then include those in the title.
        if self.multiple:
            title_str += 'Across Various '
            for i, idx in enumerate(title_elements):
                if idx == 0:
                    title_str += 'Sensors'
                elif idx == 1:
                    title_str += 'Compositions'
                elif idx == 2:
                    title_str += 'Mixture Models'
                elif idx == 3:
                    title_str += 'Data Models'
                elif idx == 4:
                    title_str += 'Fit Models'
                elif idx == 5:
                    title_str += 'Optical Depths'
                elif idx == 6:
                    if self.graphmode == 0:
                        title_str += 'Scattering Angles'
                    elif self.graphmode == 1:
                        title_str += 'Effective Grain Sizes'
                    else:
                        title_str += 'Wavelengths'

                # Adding commas and an 'and' after the second-to-last element for grammatical accuracy
                if i < len(title_elements) - 2:
                    title_str += ', '
                elif i == len(title_elements) - 2:
                    if len(title_elements) > 2:
                        title_str += ', and '
                    else:
                        title_str += ' and '
        # A single graph will always have its wavelength (if ref vs angle) or scattering angle (if ref vs wavelength) displayed
        else:
            title_str += 'at ' + self.param_str
            if self.graphmode == 0:
                title_str += '° Scattering Angle'
            elif self.graphmode == 1:
                title_str += ' μm Effective Grain Size'
            else:
                title_str += ' μm Wavelength'

        # As long as the title does not detect various instruments being represented, it is helpful to include the
        # instrument observing the data
        if 'Sensors' not in title_str:
            title_str = self.instrument.currentText() + ' ' + title_str

        self.graph_widget.setTitle(title_str, color='k')

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
            mix = self.base_params[-1][2]

        # Check to see what to put in the legend, i.e. what has changed
        updated_params = [self.instrument.currentText(),
                          compositions,
                          mix,
                          self.dist_model.currentText(),
                          self.fit_model.checkedButton().text(),
                          self.tau_box.text(),
                          self.input_box.text()] # Get the new condition values

        # Check where the new parameters do not match the base ones
        update_idxs = np.where(~(np.asarray(updated_params) == np.asarray(self.base_params[0])))[0]

        # Search all but the last part of base_params (which could match updated_params)
        # If the array is found to match some previous entry in the graph, clear the return array
        # This will trigger a special kind of error in the rest of the code
        if updated_params in self.base_params[:-1]:
            return_idxs = None
        else:
            # If the special error was triggered previously, it makes the array un-appendable. Fix this issue for
            # the new iteration.
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
                            Shape (len(self.base_params))
        """
        if len(legend_idxs) > 0: # Only loop if there are parameters to include in legend
            # Loop through each index in legend
            for i in range(len(self.legend.items)):
                legend_str = ''
                # Loop through all changing parameters
                for idx, item in enumerate(legend_idxs):
                    # Instruments, mixture models, data models, and fit models can have their strings added directly
                    if item == 0 or item == 2 or item == 3 or item == 4:
                        legend_str += self.base_params[i][item]
                    # Compositions are added directly if only one, else each one is added with its corresponding v/v
                    elif item == 1:
                        comps_dict = self.base_params[i][item]
                        if len(comps_dict) == 1:
                            legend_str += next(iter(comps_dict)) # Gets first (and only) element of the dictionary
                        else:
                            legend_str += '('
                            for j, key in enumerate(comps_dict.keys()):
                                legend_str += f'v/v_{key}={comps_dict[key]}'
                                # Add commas if before last element
                                if len(comps_dict.keys()) > 1 and j < len(comps_dict.keys()) - 1:
                                    legend_str += ', '
                            legend_str += ')'
                    # Tau and the input will be added numerically
                    elif item == 5:
                        legend_str += f'tau={self.base_params[i][item]:.2s}'
                    elif item == 6:
                        legend_str += f"{self.base_params[i][item]}{' μm' if (self.graphmode == 1 or self.graphmode == 2) else '°'}"

                    # If before the last thing to add to legend, add a comma and a space
                    if len(legend_idxs) > 1 and idx < len(legend_idxs) - 1:
                        legend_str += ', '
                    self.legend.items[i][1].setText(legend_str) # Update legend item

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
            self.mknewrow(7, 1)
            self.layout.addWidget(self.mixture_model_label, 7, 0, 1, 1)
            self.layout.addWidget(self.molecular, 7, 1, 1, 1)
            self.layout.addWidget(self.areal, 7, 2, 1, 1)

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

         # If there is only one empty field and the summed v/v fractions fall short of 1, fill the empty v/v
         # such that the fractions add to 1
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
            if 'Min X' in err_string or 'min X' in err_string:
                self.xmin_box.setStyleSheet(f"border: 1px solid red;")
            if 'Max X' in err_string:
                self.xmax_box.setStyleSheet(f"border: 1px solid red;")
            if 'Min Y' in err_string or 'min Y' in err_string:
                self.ymin_box.setStyleSheet(f"border: 1px solid red;")
            if 'Max Y' in err_string:
                self.ymax_box.setStyleSheet(f"border: 1px solid red;")

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
