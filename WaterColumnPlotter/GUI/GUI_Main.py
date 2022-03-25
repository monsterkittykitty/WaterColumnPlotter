# Lynette Davis
# ldavis@ccom.unh.edu
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# November 2021

# Description: Main class for Water Column Plotter. Initiates GUI and all other processes.

import json
import multiprocessing
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow
from PyQt5.QtCore import QTimer
import sys
from WaterColumnPlotter.Plotter.WaterColumn import WaterColumn
from WaterColumnPlotter.GUI.Dialogs.PYFiles.AllSettingsDialog import AllSettingsDialog
from WaterColumnPlotter.GUI.Widgets.GUI_MDI import GUI_MDI
from WaterColumnPlotter.GUI.Widgets.GUI_StatusBar import GUI_StatusBar
from WaterColumnPlotter.GUI.Widgets.GUI_Toolbar import GUI_Toolbar

__appname__ = "Water Column Plotter"


class MainWindow(QMainWindow):
    count = 0

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        # Default settings:
        self.settings = {'system_settings': {'system': "Kongsberg"},
                         'ip_settings': {'ip': '127.0.0.1', 'port': 6020, 'protocol': "UDP",
                                         'socketBufferMultiplier': 4},
                         'processing_settings': {'binSize_m': 0.20, 'acrossTrackAvg_m': 10, 'depth_m': 2,
                                                 'depthAvg_m': 2, 'alongTrackAvg_ping': 5, 'maxHeave_m': 2.5},
                         'buffer_settings': {'maxGridCells': 500, 'maxBufferSize_ping': 1000}}

        # Shared queue to contain pie objects:
        self.queue_pie = multiprocessing.Queue()
        self.temp_queue = multiprocessing.Queue()

        self.PLOT_UPDATE_INTERVAL = 1000  # Milliseconds

        # Window setup:
        self.resize(1300, 820)
        self.setWindowTitle("Water Column Plotter")

        self.waterColumn = WaterColumn(self.settings)
        self.update_timer = QTimer()

        # Tool Bar
        self.toolBar = self._initToolBar()

        # Status Bar
        self.status = self._initStatusBar()
        self.setStatusBar(self.status)

        # Multiple Document Interface
        self.mdi = self._initMDI()
        self.setCentralWidget(self.mdi)

        self.show()

        self.displaySettingsDialog()  # This will block until OK or Close / Cancel is selected in settings dialog

        print("Initialization: self.watercolumn.ip_settings_edited: {}".format(self.waterColumn.ip_settings_edited))

        # Must wait for OK or Close on settings dialog to initialize shared memory and ring buffers.
        self.waterColumn.initRingBuffers(create_shmem=True)
        self.mdi.setSharedRingBufferProcessed(self.waterColumn.shared_ring_buffer_processed)

    def playProcesses(self):
        """
        This method called when toolbar's play button is pressed. Activates SonarMain and PlotterMain processes in
        WaterColumn class; initiates receipt of data from sonar and processing and plotting of data;
        starts update_timer, which controls update of plots and status bar.
        """
        self.waterColumn.playProcesses()
        self.update_timer.start(self.PLOT_UPDATE_INTERVAL)

    def pauseProcesses(self):
        """
        This method called when toolbar's pause button is pressed. Deactivates SonarMain and PlotterMain processes in
        WaterColumn class; stops receipt of data from sonar, but continues processing and plotting data in queues.
        """
        self.waterColumn.pauseProcesses()

    def stopProcesses(self):
        """
        This method is called when toolbar's stop button is pressed. Deactivates SonarMain and PlotterMain processes in
        WaterColumn class; stops receipt of data from sonar and processing and plotting of data in queue; clears queues;
        stops update_timer, which controls update of plots and status bar.
        """
        self.waterColumn.stopProcesses()
        self.update_timer.stop()

    def updateStatusBar(self):
        """
        Updates GUI MainWindow's StatusBar.
        """
        with self.waterColumn.full_ping_count.get_lock() and self.waterColumn.discard_ping_count.get_lock():
            self.status.set_ping_counts(self.waterColumn.full_ping_count.value,
                                        self.waterColumn.discard_ping_count.value)

    def updatePlot(self):
        """
        Updates GUI MainWindow's MDI windows' plots.
        """
        if self.waterColumn.get_raw_buffer_length() > 0:

            # UPDATE PIE PLOT
            temp_pie = self.waterColumn.get_pie()
            if temp_pie is not None:
                # if temp_pie.any():  # For debugging
                    # print("temp_pie.shape", temp_pie.shape)

                self.mdi.pieWidget.pie_plot.setImage(temp_pie.T, autoRange=False,
                                                     autoLevels=False, autoHistogramRange=False,
                                                     pos=(-(int(temp_pie.shape[1] / 2)),
                                                          -(self.settings['processing_settings']['maxHeave_m'] /
                                                            self.settings['processing_settings']['binSize_m'])))
                # pos = (-x_zero,
                #        -(self.settings['processing_settings']['maxHeave_m'] /
                #          self.settings['processing_settings']['binSize_m']))

                self.mdi.pieWidget.updateTimestampAndIntensity()
                # # Plots vertical line
                # y = [0, 50]
                # x = [0, 0]
                # self.mdi.verticalWidget.plot.plot(x, y)
            else:
                print("Temp pie is none.")

        if self.waterColumn.get_processed_buffer_length() > 0:

            # UPDATE VERTICAL PLOT
            temp_vertical = self.waterColumn.get_vertical_slice()
            # temp_vertical, index_heave = self.waterColumn.get_vertical_slice()
            if temp_vertical is not None:
                # if temp_vertical.any():  # For debugging
                #     print("temp_vertical.shape", temp_vertical.shape)

                self.mdi.verticalWidget.vertical_plot.setImage(temp_vertical, autoRange=False,
                                                               autoLevels=False, autoHistogramRange=False,
                                                               pos=(-temp_vertical.shape[0],
                                                                    -(self.settings
                                                                       ['processing_settings']['maxHeave_m'] /
                                                                       self.settings
                                                                       ['processing_settings']['binSize_m'])))

                # self.mdi.verticalWidget.vertical_plot.setImage(temp_vertical, autoRange=False,
                #                                                autoLevels=False, autoHistogramRange=False,
                #                                                pos=(-temp_vertical.shape[0],
                #                                                     -((self.settings
                #                                                        ['processing_settings']['maxHeave_m'] /
                #                                                        self.settings
                #                                                        ['processing_settings'][
                #                                                            'binSize_m']) - index_heave)))

                self.mdi.verticalWidget.updateTimestampAndIntensity()

            # UPDATE HORIZONTAL PLOT
            temp_horizontal = self.waterColumn.get_horizontal_slice()
            if temp_horizontal is not None:
                # if temp_horizontal.any():  # For debugging
                #     print("temp_horizontal.shape", temp_horizontal.shape)

                self.mdi.horizontalWidget.horizontal_plot.setImage(temp_horizontal, autoRange=False,
                                                                   autoLevels=False, autoHistogramRange=False,
                                                                   pos=(-temp_horizontal.shape[0],
                                                                        -int(temp_horizontal.shape[1] / 2)))
                self.mdi.horizontalWidget.updateTimestampAndIntensity()

    # SYSTEM SETTINGS SLOTS:
    def systemEdited(self):
        """
        Updates system settings.
        """
        # NOTE: Currently only one system (Kongsberg) supported. If/when other systems supported, the sonarMain
        # processes will have to be terminated and restarted with the new system's capture and processing code.

        self.toolBar.setSystem(self.settings['system_settings']['system'])

    # IP SETTINGS SLOTS:
    def ipEdited(self):
        """
        Updates IP settings.
        """
        print("ip_edited")
        self.toolBar.setIPPort(self.settings['ip_settings']['ip'], self.settings['ip_settings']['port'])

        # NOTE: IP address stored as multiprocessing Array
        with self.waterColumn.ip.get_lock():
            self.waterColumn.ip[:] = self.waterColumn.editIP(self.settings['ip_settings']['ip'], append=True)

        if self.waterColumn.sonarMain:
            self.waterColumn.ip_settings_edited = True

    def portEdited(self):
        """
        Updates port settings.
        """
        print("port_edited")
        self.toolBar.setIPPort(self.settings['ip_settings']['ip'], self.settings['ip_settings']['port'])

        with self.waterColumn.port.get_lock():
            self.waterColumn.port.value = self.settings['ip_settings']['port']

        if self.waterColumn.sonarMain:
            self.waterColumn.ip_settings_edited = True

    # TODO: Link to other processes
    def protocolEdited(self):
        """
        Updates protocol settings.
        """
        print("protocol_edited: ", self.settings['ip_settings']['protocol'][0])
        with self.waterColumn.protocol.get_lock():
            self.waterColumn.protocol.value = self.settings['ip_settings']['protocol'][0]

        if self.waterColumn.sonarMain:
            self.waterColumn.ip_settings_edited = True

    # TODO: Link to other processes
    def socketBufferEdited(self):
        """
        Updates socket buffer settings.
        """
        print("socket_buffer_edited")
        with self.waterColumn.socket_buffer_multiplier.get_lock():
            self.waterColumn.socket_buffer_multiplier.value = self.settings['ip_settings']['socketBufferMultiplier']

        if self.waterColumn.sonarMain:
            self.waterColumn.ip_settings_edited = True

    # PROCESSING SETTINGS SLOTS:
    def binSizeEdited(self, fromSettingsDialog=False):
        """
        Updates bin size settings.
        :param fromSettingsDialog: Indicates whether update was made from settings dialog (rather than MDI window).
        """
        # Only need to update MDI windows if setting was updated in settings dialog:
        if fromSettingsDialog:
            self.mdi.pieWidget.setBinSize(self.settings['processing_settings']['binSize_m'])

        with self.waterColumn.bin_size.get_lock():
            self.waterColumn.bin_size.value = self.settings['processing_settings']['binSize_m']

        self.mdi.pieWidget.pie_plot.clear()
        self.mdi.verticalWidget.vertical_plot.clear()
        self.mdi.horizontalWidget.horizontal_plot.clear()

        # TODO: This will get called twice if both depthEdited and binSizeEdited...
        # Move location of depth indicator in vertical slice and pie slice windows:
        self.mdi.setDepthIndicator()
        # Omitted to decrease clutter over plots
        # self.mdi.setDepthAvgIndicators()

    def acrossTrackAvgEdited(self, fromSettingsDialog=False):
        """
        Updates across-track average settings.
        :param fromSettingsDialog: Indicates whether update was made from settings dialog (rather than MDI window).
        """
        print("Across-track average edited. self.watercolumn.ip_settings_edited: {}".format(self.waterColumn.ip_settings_edited))

        # Only need to update MDI windows if setting was updated in settings dialog:
        if fromSettingsDialog:
            self.mdi.verticalWidget.setAcrossTrackAvg(self.settings['processing_settings']['acrossTrackAvg_m'])

        with self.waterColumn.across_track_avg.get_lock():
            self.waterColumn.across_track_avg.value = self.settings['processing_settings']['acrossTrackAvg_m']

    def depthEdited(self, fromSettingsDialog=False):
        """
        Updates depth settings.
        :param fromSettingsDialog: Indicates whether update was made from settings dialog (rather than MDI window).
        """
        # Only need to update MDI windows if setting was updated in settings dialog:
        if fromSettingsDialog:
            self.mdi.horizontalWidget.setDepth(self.settings['processing_settings']['depth_m'])

        with self.waterColumn.depth.get_lock():
            self.waterColumn.depth.value = self.settings['processing_settings']['depth_m']

        # TODO: This will get called twice if both depthEdited and binSizeEdited...
        # Move location of depth indicator in vertical slice and pie slice windows:
        self.mdi.setDepthIndicator()
        # Omitted to decrease clutter over plots
        # self.mdi.setDepthAvgIndicators()

    def depthAvgEdited(self, fromSettingsDialog=False):
        """
        Updates depth average settings.
        :param fromSettingsDialog: Indicates whether update was made from settings dialog (rather than MDI window).
        """
        # Only need to update MDI windows if setting was updated in settings dialog:
        if fromSettingsDialog:
            self.mdi.horizontalWidget.setDepthAvg(self.settings['processing_settings']['depthAvg_m'])

        with self.waterColumn.depth_avg.get_lock():
            self.waterColumn.depth_avg.value = self.settings['processing_settings']['depthAvg_m']

        # Omitted to decrease clutter over plots
        # self.mdi.setDepthAvgIndicators()

    def alongTrackAvgEdited(self):
        """
        Updates along-track average settings.
        """
        print("Along-track average edited. self.watercolumn.ip_settings_edited: {}".format(self.waterColumn.ip_settings_edited))
        print("New along_track_average: ", self.settings['processing_settings']['alongTrackAvg_ping'])

        with self.waterColumn.along_track_avg.get_lock():
            self.waterColumn.along_track_avg.value = self.settings['processing_settings']['alongTrackAvg_ping']

    def heaveEdited(self):
        """
        Updates heave settings.
        """
        with self.waterColumn.max_heave.get_lock():
            self.waterColumn.max_heave.value = self.settings['processing_settings']['maxHeave_m']

    # BUFFER SETTINGS SLOTS:
    # TODO: Link to other processes
    def gridCellsEdited(self):
        """
        Updates grid cell settings. (Indicates maximum size of square matrix for pie plots.)
        """
        with self.waterColumn.max_grid_cells.get_lock():
            self.waterColumn.max_grid_cells.value = self.settings['buffer_settings']['maxGridCells']

    # TODO: Link to other processes
    def pingBufferEdited(self):
        """
        Updates ping buffer settings. (Indicates maximum number of pings to store in raw buffer.)
        """
        with self.waterColumn.max_ping_buffer.get_lock():
            self.waterColumn.max_ping_buffer.value = self.settings['buffer_settings']['maxBufferSize_ping']

    def settingsEdited(self):
        """
        Signals WaterColumn class that settings have been updated / changed.
        """
        print("Settings edited.")
        self.waterColumn.settingsChanged()

    def displaySettingsDialog(self):
        """
        Initializes and launches settings dialog; connects signals / slots.
        """
        settingsDialog = AllSettingsDialog(self.settings, parent=self)

        # Once raw and processed ring buffers are initialized,
        # changing max_grid_cells and max_ping_buffer is no longer allowable.
        if self.waterColumn.shared_ring_buffer_raw and self.waterColumn.shared_ring_buffer_processed:
            settingsDialog.spinBoxMaxGridCells.setDisabled(True)
            settingsDialog.spinBoxMaxGridCells.setToolTip("Editing of this field is not allowed after initialization.")
            settingsDialog.spinBoxMaxPingBuffer.setDisabled(True)
            settingsDialog.spinBoxMaxPingBuffer.setToolTip("Editing of this field is not allowed after initialization.")

        # Signals / Slots
        settingsDialog.pushButtonLoadSettings.clicked.connect(lambda: self.displayLoadSettingsDialog(settingsDialog))
        settingsDialog.pushButtonSaveSettings.clicked.connect(self.displaySaveSettingsDialog)
        settingsDialog.signalSystemEdited.connect(self.systemEdited)
        settingsDialog.signalIPEdited.connect(self.ipEdited)
        settingsDialog.signalPortEdited.connect(self.portEdited)
        settingsDialog.signalProtocolEdited.connect(self.protocolEdited)
        settingsDialog.signalSocketBufferEdited.connect(self.socketBufferEdited)
        settingsDialog.signalBinSizeEdited.connect(lambda: self.binSizeEdited(fromSettingsDialog=True))
        settingsDialog.signalAcrossTrackAvgEdited.connect(lambda: self.acrossTrackAvgEdited(fromSettingsDialog=True))
        settingsDialog.signalDepthEdited.connect(lambda: self.depthEdited(fromSettingsDialog=True))
        settingsDialog.signalDepthAvgEdited.connect(lambda: self.depthAvgEdited(fromSettingsDialog=True))
        print("displaySettingsDialog, before alongTrackSignal, ip: {}".format(self.waterColumn.ip_settings_edited))
        settingsDialog.signalAlongTrackAvgEdited.connect(self.alongTrackAvgEdited)
        print("displaySettingsDialog, after alongTrackSignal, ip: {}".format(self.waterColumn.ip_settings_edited))
        settingsDialog.signalHeaveEdited.connect(self.heaveEdited)
        settingsDialog.signalGridCellsEdited.connect(self.gridCellsEdited)
        settingsDialog.signalPingBufferEdited.connect(self.pingBufferEdited)
        settingsDialog.signalsettingsEdited.connect(self.settingsEdited)

        settingsDialog.exec_()

    def displaySaveSettingsDialog(self):
        """
        Launches file browser to enable user to save settings.
        """
        saveDialog = QFileDialog(self)
        filePath = saveDialog.getSaveFileName(self, __appname__, directory=".\Settings", filter="JSON (*.json)")

        if filePath[0]:
            with open(filePath[0], 'w') as f:
                json.dump(self.settings, f, indent=4)

    def displayLoadSettingsDialog(self, settingsDialog):
        """
        Launches file browser to enable user to load previously saved settings.
        :param settingsDialog: An instance of AllSettingsDialog to load settings into.
        """
        openDialog = QFileDialog(self)
        filePath = openDialog.getOpenFileName(self,  __appname__, directory=".\Settings", filter="JSON (*.json)")

        if filePath[0]:
            with open(filePath[0], 'r') as f:
                tempSettings = json.load(f)

            settingsDialog.validateAndSetValuesFromFile(tempSettings)

    def closeEvent(self, event):
        """
        This method is called when GUI MainWindows close (X) button is clicked. Deactivates SonarMain and PlotterMain
        processes in WaterColumn class, closes and unlinks shared memory in WaterColumn class.
        :param event: Close event emitted by GUI MainWindow's close (X) button.
        """
        # with self.waterColumn.process_flag.get_lock():
        #     self.waterColumn.process_flag.value = False
        self.waterColumn.closeSharedMemory()
        # Release shared memory definitely
        self.waterColumn.unlinkSharedMemory()

        event.accept()

    def _initToolBar(self):
        """
        Initializes GUI MainWindow's Toolbar; connects signals / slots for toolbar buttons.
        :return: GUI MainWindow's Toolbar.
        """
        toolBar = GUI_Toolbar(self.settings, parent=self)
        self.addToolBar(toolBar)

        # Signals / Slots
        toolBar.signalPlay.connect(self.playProcesses)
        toolBar.signalPause.connect(self.pauseProcesses)
        toolBar.signalStop.connect(self.stopProcesses)
        toolBar.signalSettings.connect(self.displaySettingsDialog)

        return toolBar

    def _initStatusBar(self):
        """
        Initializes GUI MainWindow's StatusBar; connects signals / slots for updating status bar text.
        :return: GUI MainWindow's StatusBar.
        """
        statusBar = GUI_StatusBar()

        # Signals / Slots
        self.update_timer.timeout.connect(self.updateStatusBar)

        return statusBar

    def _initMDI(self):
        """
        Initializes GUI MainWindow's Multiple Document Interface (MDI) with vertical, horizontal, and pie plots;
        connects signals / slots for windows' settings buttons.
        :return: GUI MainWindow's MDI
        """
        mdi = GUI_MDI(self.settings, self.waterColumn.shared_ring_buffer_processed, parent=self)

        # Signals / Slots
        mdi.verticalWidget.signalAcrossTrackAvgEdited.connect(self.acrossTrackAvgEdited)
        mdi.verticalWidget.signalsettingsEdited.connect(self.settingsEdited)
        mdi.pieWidget.signalbinSizeEdited.connect(self.binSizeEdited)
        mdi.pieWidget.signalsettingsEdited.connect(self.settingsEdited)
        mdi.horizontalWidget.signalDepthEdited.connect(self.depthEdited)
        mdi.horizontalWidget.signalDepthAvgEdited.connect(self.depthAvgEdited)
        mdi.horizontalWidget.signalsettingsEdited.connect(self.settingsEdited)

        self.update_timer.timeout.connect(self.updatePlot)

        return mdi


def main():
    """
    Launches GUI MainWindow.
    """
    app = QApplication(sys.argv)
    form = MainWindow()
    form.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()