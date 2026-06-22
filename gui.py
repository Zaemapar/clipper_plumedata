import sys
import pyqtgraph as pg
import pyqtgraph.exporters
import PyQt6.QtWidgets as qt
import pred_vars as vars
import utils
import os

COLOR_ARR = ['red', 'magenta', 'orange', 'green', 'blue', 'violet'] # Array of graph colors to cycle through

class MainWindow(qt.QMainWindow):
    def __init__(self):
        # Initialize the parent class (QMainWindow)
        super().__init__()

        # --- MAIN WINDOW PROPERTIES ---
        # Title and sizing
        self.setWindowTitle("Clipper Instrument Spectrum Prediction")
        self.resize(1200, 800)

        # QMainWindow requires a central widget to hold everything else
        central_widget = qt.QWidget()
        self.setCentralWidget(central_widget)

        # We will use a Vertical Box Layout (QVBoxLayout) to stack widgets top-to-bottom
        layout = qt.QGridLayout()
        layout.setContentsMargins(30, 30, 30, 30) # Making sure things don't overlap each other
        layout.setSpacing(20)
        central_widget.setLayout(layout)

        # --- WIDGETS ---
        # Push button to toggle between reflectance vs wavelength graphs and reflectance vs scattering angle graphs
        self.ifvangle = False # Indicates what type of graph is being shown
        self.graph_mode_button = qt.QPushButton("Reflectance vs. Wavelength")

        # Dropdown list for instruments on Europa Clipper
        self.instrument_label = qt.QLabel("Instrument:")
        self.instrument = qt.QComboBox()
        self.instrument.addItems(vars.INSTRUMENTS.keys())

        # Dropdown list for available material compositions
        self.comp_label = qt.QLabel("Composition:")
        self.comp = qt.QComboBox()
        self.comp.addItems(vars.COMPS)
        self.comp_min_wavel, self.comp_max_wavel = utils.get_minmax_wavel(self.comp.currentText())

        # Dropdown list for available data models
        self.dist_label = qt.QLabel("Data Model:")
        self.dist_model = qt.QComboBox()
        self.dist_model.addItems(vars.DATA_MODELS)

        # Radio buttons to toggle between Mie and Henyey-Greenstein theory, or just Mie for reflectance vs wavelength plots
        self.fit_label = qt.QLabel("Fit Model:")
        self.model_a = qt.QRadioButton("Mie")
        self.model_a.setChecked(True)
        self.model_b = qt.QRadioButton("Henyey-Greenstein")
        self.fit_model = qt.QButtonGroup() # Setting up a group to ensure only one can be checked
        self.fit_model.addButton(self.model_a)
        self.fit_model.addButton(self.model_b)
        self.model_b.setVisible(False) # Default is reflectance vs wavelenght, therefore Henyey-Greenstein is not visible

        # Entry box for manual optical depth input
        self.tau_label = qt.QLabel("Optical Depth:")
        self.tau_box = qt.QLineEdit()
        self.tau_box.setPlaceholderText("Input optical depth") # Switching graph modes will clear this box, so this needs to display
        self.tau_box.setText("1e-6")
        self.tau_box.setStyleSheet("border: 1px solid black;") # Declaring border type for future modification

        # Entry box for varying parameter, either scattering angle (for ref vs. wavel) or wavelength (for ref vs. theta)
        self.param_str = ''
        self.input_label = qt.QLabel("Scattering Angle:")
        self.input_box = qt.QLineEdit()
        self.input_box.setPlaceholderText("Input scattering angle in degrees")
        self.input_box.setText("90")
        self.input_box.setStyleSheet("border: 1px solid black;") # Declaring border type for future modification

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
        # Setting up a pushbutton to clear the whole plot
        self.clear_plot_button = qt.QPushButton("Clear plots")
        # Setting up a pushbutton to save the plot to file
        self.save_plot_button = qt.QPushButton("Save plot")

        # Setting up a label at the bottom of the screen giving status updates
        self.load_label = qt.QLabel("Up to date")
        self.load_label.setStyleSheet("color: red;")
        self.load_label.setWordWrap(True)

        # Setting up the graph window
        self.multiple = False # Multiple will be used to indicate if multiple datasets are in the same graph
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
        self.update_title()
        self.legend = None # Declare a legend object, will be populated when multiple is True

        # Setting up a pen to plot the window
        self.last_color_idx = 0 # Color from which to start. Will count up in multiple plotting
        self.plt_pen = pg.mkPen(color=COLOR_ARR[self.last_color_idx], width=3)
        self.ifs = self.graph_widget.plot(self.x_data, self.y_data, pen=self.plt_pen) # Plot a blank dataset

        # Setting up the 'home zoom' button to reset the zoom
        self.home_button = qt.QPushButton("Home")

        # --- WIDGET FUNCTIONALITY ---
        # Pressing the graph mode button toggles between ref vs wavel and ref vs angle
        self.graph_mode_button.clicked.connect(self.mode_toggle)

        # Pressing scale radio buttons updates scale
        self.x_scale.buttonClicked.connect(self.update_scale)
        self.y_scale.buttonClicked.connect(self.update_scale)

        # Changing anything about the dataset (instrument, comp, model, tau, etc) replots the data
        self.instrument.currentTextChanged.connect(self.update_graph)
        self.comp.currentTextChanged.connect(self.update_graph)
        self.dist_model.currentTextChanged.connect(self.update_graph)
        self.fit_model.buttonClicked.connect(self.update_graph)
        self.tau_box.returnPressed.connect(self.update_graph)
        self.input_box.returnPressed.connect(self.update_graph)

        # Pressing add graph button saves the dataset in the plot and starts a new one
        self.add_graph_button.clicked.connect(self.plot_graph)
        # Pressing clear plot button clears the plot and resets the title
        self.clear_plot_button.clicked.connect(self.clear_graph)
        # Pressing save plot button saves the graph to file
        self.save_plot_button.clicked.connect(self.save_graph)
        # Pressing the home zoom button auto-zooms the graph
        self.home_button.clicked.connect(lambda: self.graph_widget.autoRange())

        # --- LAYOUT ---
        # Items are arranged in (row, column, rowspan, columnspan)
        layout.addWidget(self.graph_mode_button, 0, 0, 1, 3)
        layout.addWidget(self.instrument_label, 1, 0, 1, 1)
        layout.addWidget(self.instrument, 1, 1, 1, 2)
        layout.addWidget(self.comp_label, 2, 0, 1, 1)
        layout.addWidget(self.comp, 2, 1, 1, 2)
        layout.addWidget(self.dist_label,  3, 0, 1, 1)
        layout.addWidget(self.dist_model, 3, 1, 1, 2)
        layout.addWidget(self.fit_label, 4, 0, 1, 1)
        layout.addWidget(self.model_a, 4, 1, 1, 1)
        layout.addWidget(self.model_b, 4, 2, 1, 1)
        layout.addWidget(self.tau_label, 5, 0, 1, 1)
        layout.addWidget(self.tau_box, 5, 1, 1, 2)
        layout.addWidget(self.input_label, 6, 0, 1, 1)
        layout.addWidget(self.input_box, 6, 1, 1, 2)
        layout.addWidget(self.x_scale_label, 7, 0, 1, 1)
        layout.addWidget(self.x_linear, 7, 1, 1, 1)
        layout.addWidget(self.x_log, 7, 2, 1, 1)
        layout.addWidget(self.y_scale_label, 8, 0, 1, 1)
        layout.addWidget(self.y_linear, 8, 1, 1, 1)
        layout.addWidget(self.y_log, 8, 2, 1, 1)
        layout.addWidget(self.add_graph_button, 9, 0, 1, 3)
        layout.addWidget(self.clear_plot_button, 10, 0, 1, 3)
        layout.addWidget(self.save_plot_button, 11, 0, 1, 3)
        layout.addWidget(self.load_label, 12, 0, 2, 3)
        layout.addWidget(self.graph_widget, 0, 3, 15, 15)
        layout.addWidget(self.home_button, 1, 4, 1, 1)

        self.update_graph() # There are default settings in the input boxes, so output the first graph

    def mode_toggle(self):
        """
        A function that defines the behavior of the graph mode button when clicked. Toggles between
        'Reflectance vs. Wavelength' and 'Reflectance vs. Scattering Angle.' Clears graph in the process, 
        resets input fields, and updates title

        :param self: MainWindow object
        """
        self.ifvangle = not self.ifvangle # Switch the graph mode variable
        if self.ifvangle:
            self.graph_mode_button.setText("Reflectance vs. Scattering Angle")
            self.input_label.setText("Wavelength:")
            self.input_box.setPlaceholderText("Input wavelength in microns")
            self.graph_widget.setLabel('bottom', 'Scattering Angle', units='°') # Change bottom axis label
            self.model_b.setVisible(True) # Reflectance vs. scattering angle has Henyey-Greenstein functionality
            self.clear_graph() # Clear graph when finished
        else:
            self.graph_mode_button.setText("Reflectance vs. Wavelength")
            self.input_label.setText("Scattering Angle:")
            self.input_box.setPlaceholderText("Input scattering angle in degrees")
            self.graph_widget.setLabel('bottom', 'Wavelength', units='μm')
            self.model_b.setVisible(False) # Reflectance vs. wavelength does not have Henyey-Greenstein functionality
            self.model_a.setChecked(True) # So we set it automatically to the remaining Mie option
            self.clear_graph()

    def update_graph(self):
        """
        A function called if any of the data parameters are changed. Calls the plotter function from the main
        program and outputs the new function on the graph, replacing the old function (assuming that function
        wasn't stored via 'Add new plot'). If parameters are invalid, updates status message to error message
        and turns the appropriate box red. If multiple graphs involved, updates legend accordingly.

        :param self: MainWindow object
        """
        # Reset the borders of the input boxes to black in case they were red before
        self.tau_box.setStyleSheet("border: 1px solid black;")
        self.input_box.setStyleSheet("border: 1px solid black;")
        self.load_label.setText("Updating...") # Status label changes to indicate loading
        qt.QApplication.processEvents() # Force a screen update to render the status label and the box borders

        # Read each parameter from the appropriate widget
        ifvanglebool = self.ifvangle
        sensor = self.instrument.currentText()
        composition = self.comp.currentText()
        # Composition changes require getting which wavelengths are supported for that material
        self.comp_min_wavel, self.comp_max_wavel = utils.get_minmax_wavel(self.comp.currentText())
        distmodel = self.dist_model.currentText()
        fitmodel = self.fit_model.checkedButton().text()
        # Tau and the parameter are both initially read as strings
        tau_str = self.tau_box.text()
        self.param_str = self.input_box.text() # param_str is a self variable because it needs to be referenced elsewhere

        # Error handling - checking to make sure that all parameters are in bounds
        err_message = '' # Will dynamically update to support multiple errors at once
        # Error checking for tau
        try:
            tau = float(tau_str)
            if tau < 0:
                err_message += 'Optical depth must be nonnegative. '
        except ValueError:
            err_message += 'Optical depth must be a float. '
        # Error checking for parameter if it's a wavelength
        if ifvanglebool:
            try:
                param = float(self.param_str)
                # Check to make sure it's in range of the sensor AND the index of refraction data for that material
                if param < vars.INSTRUMENTS[sensor][0]:
                    err_message += f'Wavelength is too low for {sensor}. '
                elif param > vars.INSTRUMENTS[sensor][1]:
                    err_message += f'Wavelength is too high for {sensor}. '
                if param < self.comp_min_wavel:
                    err_message += f'Wavelength is too low for {composition}'
                elif param > self.comp_max_wavel:
                    err_message += f'Wavelength is too high for {composition}'
            except ValueError:
                # Allow for input of MIN and MAX to get the top and bottom of the instrument's range
                if self.param_str == 'MIN':
                    param = vars.INSTRUMENTS[sensor][0]
                    self.param_str = str(param) # Param str is used in title updates, ensure it doesn't display MAX but an actual number
                elif self.param_str == 'MAX':
                    param = vars.INSTRUMENTS[sensor][1]
                    self.param_str = str(param)
                else:
                    err_message += f'Wavelength must be a float, MIN, or MAX. '
        # Error checking for parameter if it's a scattering angle
        else:
            try:
                param = int(self.param_str)
                if param < 0 or param > 180:
                    err_message += f'Scattering angle must be between 0 and 180 degrees. '
            except ValueError:
                err_message += f'Scattering angle must be an integer. '

        if len(err_message) == 0:
            # Get the new x and y datasets if no errors
            self.x_data, self.y_data = utils.output_graph(ifvanglebool, sensor, composition, distmodel, fitmodel, float(param), float(tau))
            
            # Plot the dataset if not in the graph, else change the current unsaved graph
            if self.ifs not in self.graph_widget.plotItem.items:
                self.ifs = self.graph_widget.plot(self.x_data, self.y_data, pen=self.plt_pen, name=(f"{self.param_str}{' μm' if self.ifvangle else '°'}" if self.multiple else None))
            else:
                self.ifs.setData(self.x_data, self.y_data)
                # Update the legend if multiple graphs
                if self.multiple:
                    new_name = f"{self.param_str}{' μm' if self.ifvangle else '°'}"
                    self.ifs.opts['name'] = new_name
                    self.legend.removeItem(self.ifs) # Need to do remove + add because there's no setter function
                    self.legend.addItem(self.ifs, new_name)
            self.load_label.setText("Up to date")
            self.graph_widget.autoRange() # Window auto fit for the new dataset
            self.update_title()
        else:
            self.load_label.setText(err_message)
            # Scan the error message for problematic variables and turn their boxes red
            if 'Wavelength' in err_message or 'Scattering angle' in err_message:
                self.input_box.setStyleSheet("border: 1px solid red;")
            if 'Optical depth' in err_message:
                self.tau_box.setStyleSheet("border: 1px solid red;")

    def plot_graph(self):
        """
        A function that saves the current x_data, y_data to the graph as a permanent plot and resets the dataset
        for a new plot, setting the plot up to hold multiple datasets. Sets up the legend and updates the title.
        Also clears the input box for the new dataset.

        :param self: MainWindow object
        """
        # Handle the case if there is no temporary graph
        if len(self.x_data) == 0 or len(self.y_data) == 0:
            self.load_label.setText("No plot to add")
            self.input_box.setStyleSheet("border: 1px solid red;")
        else:
            self.multiple = True # Set the global variable to indicate multiple graphs stored on plot
            self.update_title()
            if self.legend == None:
                self.legend = self.graph_widget.addLegend() # Add legend if none
                self.legend.setLabelTextColor('k')
            self.legend.removeItem(self.ifs) # If temporary plot has a legend item, remove it to establish a permanent one
            # By not setting it to a variable, the new plot will not be changed
            self.graph_widget.plot(self.x_data, self.y_data, pen=pg.mkPen(color=COLOR_ARR[self.last_color_idx], width=3), name=f"{self.param_str}{' μm' if self.ifvangle else '°'}")
            self.graph_widget.autoRange() # Auto fit for the shown datasets
            # Cycle through to the next color for the next temporary plot
            self.last_color_idx = (self.last_color_idx + 1) % len(COLOR_ARR)
            self.plt_pen.setColor(pg.mkColor(COLOR_ARR[self.last_color_idx]))
            # Reset data arrays and temporary plot
            self.x_data = []
            self.y_data = []
            self.ifs.setData(self.x_data, self.y_data)
            self.param_str = '' # Reset input box
            self.input_box.setText('')

    def clear_graph(self):
        """
        Clears all graphs from the current plot, as well as the parameter in the parameter box.

        :param self: MainWindow object
        """
        self.multiple = False # No graphs means no multiple
        self.param_str = '' # Reset input box
        self.input_box.setText(param_str)
        self.update_title() # Update title to default
        # Reset temporary plot
        self.x_data = []
        self.y_data = []
        self.ifs.setData(self.x_data, self.y_data)
        self.last_color_idx = 0 # Set temporary plot color to red
        self.graph_widget.clearPlots() # Erase all permanent plots

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

    def update_title(self):
        """
        A function that updates the graph title dynamically based on current parameter values

        :param self: MainWindow object
        """
        # Title changes based on graph mode
        if not self.ifvangle:
            # Will say 'Various Scattering Angles' if multiple graphs or just the scattering angle if one
            self.graph_widget.setTitle(f"{self.instrument.currentText()} Reflectance vs. Wavelength at {'Various' if self.multiple else self.param_str + '°'} Scattering Angle{'s' if self.multiple else ''}", color='k')
        else:
            # Will say 'Various Wavelengths' if multiple graphs or just the wavelength if one
            self.graph_widget.setTitle(f"{self.instrument.currentText()} Reflectance vs. Scattering Angle at {'Various' if self.multiple else self.param_str + ' μm'} Wavelength{'s' if self.multiple else ''}", color='k')