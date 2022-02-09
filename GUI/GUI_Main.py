# Main class for Water Column Plotter.

# Lynette Davis
# ldavis@ccom.unh.edu
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# November 2021

import ctypes
import json
import multiprocessing
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow
from PyQt5.QtCore import QTimer
import sys
from WaterColumn import WaterColumn

from GUI.Dialogs.PYFiles.AllSettingsDialog2 import AllSettingsDialog2
from GUI_MDI import GUI_MDI
from GUI.GUI_StatusBar import GUI_StatusBar
from GUI.GUI_Toolbar import GUI_Toolbar

__appname__ = "Water Column Plotter"


class MainWindow(QMainWindow):
    count = 0

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        # Default settings:
        self.settings = {'system_settings': {'system': "Kongsberg"},
                         'ip_settings': {'ip': '127.0.0.1', 'port': 8080, 'protocol': "UDP",
                                         'socketBufferMultiplier': 4},
                         'processing_settings': {'binSize_m': 0.20, 'acrossTrackAvg_m': 10, 'depth_m': 2,
                                                 'depthAvg_m': 2, 'alongTrackAvg_ping': 5, 'maxHeave_m': 5},
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
        # Look into this: https://groups.google.com/g/pyqtgraph/c/HAFx-wIpmGA
        # test = np.ones((50, 100))
        # test[:] = 50
        # self.mdi.pieWidget.pie_plot.setImage(test, autoRange=False,
        #                                      autoLevels=False, autoHistogramRange=False,
        #                                      pos=(-(test.shape[1] / 2), 0))
        # This does kind of work! Find way to make linear overlay!
        # # circle overlay
        # pen = QPen(Qt.red, 0.1)
        # r = MyCircleOverlay(pos=(-20, 20), size=10, pen=pen, movable=False)
        # self.mdi.pieWidget.pie_plot.getView().addItem(r)

        # This just replaces image...
        # test2 = np.empty((50, 100))
        # test2[:] = np.nan
        # test2[:, 20] = 100
        # self.mdi.pieWidget.pie_plot.setImage(test2, autoRange=False,
        #                                      autoLevels=False, autoHistogramRange=False,
        #                                      pos=(-(test2.shape[1] / 2), 0))

        print("updating plots")

        # with self.waterColumn.processing_settings_edited.get_lock():
        #     if self.waterColumn.processing_settings_edited.value:
        #         print("self.waterColumn.processing_settings_edited: ", self.waterColumn.processing_settings_edited.value)
        #         self.waterColumn.update_buffers()

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

        if self.waterColumn.get_processed_buffer_length() > 0:

            # UPDATE VERTICAL PLOT
            temp_vertical  = self.waterColumn.get_vertical_slice()
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
        # else:
        #     self.mdi.pieWidget.pie_plot.clear()
        #     self.mdi.verticalWidget.vertical_plot.clear()
        #     self.mdi.horizontalWidget.horizontal_plot.clear()

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
        print("ipEdited")
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
        print("portEdited")
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
        print("protocolEdited")
        with self.waterColumn.protocol.get_lock():
            self.waterColumn.protocol.value = self.settings['ip_settings']['protocol']

        if self.waterColumn.sonarMain:
            self.waterColumn.ip_settings_edited = True

    # TODO: Link to other processes
    def socketBufferEdited(self):
        """
        Updates socket buffer settings.
        """
        print("socketBufferEdited")
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
        :return:
        """
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
        print("depthEdited")
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
        print("along track edited")
        with self.waterColumn.along_track_avg.get_lock():
            self.waterColumn.along_track_avg.value = self.settings['processing_settings']['alongTrackAvg_ping']

    def heaveEdited(self):
        """
        Updates heave settings.
        """
        print("In gui main; heaveEdited")
        with self.waterColumn.max_heave.get_lock():
            self.waterColumn.max_heave.value = self.settings['processing_settings']['maxHeave_m']

    # BUFFER SETTINGS SLOTS:
    # TODO: Link to other processes
    def gridCellsEdited(self):
        """
        Updates grid cell settings. (Indicates maximum size of square matrix for pie plots.)
        """
        # pass
        with self.waterColumn.max_grid_cells.get_lock():
            self.waterColumn.max_grid_cells.value = self.settings['buffer_settings']['maxGridCells']

    # TODO: Link to other processes
    def pingBufferEdited(self):
        """
        Updates ping buffer settings. (Indicates maximum number of pings to store in raw buffer.)
        """
        pass

    # def ipSettingsEdited(self):
    #     self.waterColumn.ip_settings_edited = True

    def processingSettingsEdited(self):
        """
        Signals WaterColumn class that settings have been updated / changed.
        """
        print("in gui main; processingSettingsEdited")
        self.waterColumn.settingsChanged()

    # def settingsEdited(self, list):
    #     print("in gui main; settingsEdited", list)
    #     if self.settings['system_settings']['system'] == "Kongsberg":
    #         assert len(list) == 3
    #         settings_edited = {'Kongsberg': {'capture': list[0], 'process': list[1], 'plotter': list[2]}}
    #     self.waterColumn.settingsChanged(settings_edited)

    # def settingsEdited(self, list):
    #     """
    #     Signals WaterColumn class which settings have been updated / changed.
    #     :param list: A list of booleans corresponding to keys in self.settings dictionary.
    #     True indicates a value has been edited; False indicates that it has not.
    #
    #     self.settings = {'system_settings': {'system': _0_},
    #                      'ip_settings': {'ip': _1_, 'port': _2_, 'protocol': _3_, 'socketBufferMultiplier': _4_},
    #                      'processing_settings': {'binSize_m': _5_, 'acrossTrackAvg_m': _6_, 'depth_m': _7_,
    #                                              'depthAvg_m': _8_, 'alongTrackAvg_ping': _9_, 'maxHeave_m': _10_},
    #                      'buffer_settings': {'maxGridCells': _11_, 'maxBufferSize_ping': _12_}}
    #     """
    #     capture_settings_edited = False
    #     process_settings_edited = False
    #     plotter_settings_edited = False
    #
    #     if list[1] or list[2] or list[3] or list[4]:
    #         capture_settings_edited = True
    #     if list[5] or list[10] or list[11]:
    #         process_settings_edited = True
    #     if list[5] or list[6] or list[7] or list[8] or list[9] or list[10]:
    #         plotter_settings_edited = True
    #
    #     self.waterColumn.settingsChanged(capture_settings_edited, process_settings_edited, plotter_settings_edited)

    def displaySettingsDialog(self):
        """
        Initializes and launches settings dialog; connects signals / slots.
        """
        settingsDialog = AllSettingsDialog2(self.settings, parent=self)

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
        settingsDialog.signalAlongTrackAvgEdited.connect(self.alongTrackAvgEdited)
        settingsDialog.signalHeaveEdited.connect(self.heaveEdited)
        settingsDialog.signalGridCellsEdited.connect(self.gridCellsEdited)
        settingsDialog.signalPingBufferEdited.connect(self.pingBufferEdited)

        # settingsDialog.signalKongsbergCaptureSettingsEdited.connect(self.kongsbergCaptureSettingsEdited)
        # settingsDialog.signalKongsbergProcessSettingsEdited.connect(self.kongsbergProcessSettingsEdited)
        # settingsDialog.signalKongsbergPlotterSettingsEdited.connect(self.kongsbergPlotterSettingsEdited)

        # settingsDialog.signalIPSettingsEdited.connect(self.ipSettingsEdited)
        settingsDialog.signalProcessingSettingsEdited.connect(self.processingSettingsEdited)
        # settingsDialog.signalSettingsEdited.connect(self.settingsEdited)

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
        with self.waterColumn.process_flag.get_lock():
            self.waterColumn.process_flag.value = False
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
        mdi.verticalWidget.signalProcessingSettingsEdited.connect(self.processingSettingsEdited)
        mdi.pieWidget.signalbinSizeEdited.connect(self.binSizeEdited)
        mdi.pieWidget.signalProcessingSettingsEdited.connect(self.processingSettingsEdited)
        mdi.horizontalWidget.signalDepthEdited.connect(self.depthEdited)
        mdi.horizontalWidget.signalDepthAvgEdited.connect(self.depthAvgEdited)
        mdi.horizontalWidget.signalProcessingSettingsEdited.connect(self.processingSettingsEdited)

        self.update_timer.timeout.connect(self.updatePlot)

        return mdi

# # TODO: Test:
# import pyqtgraph as pg
# class MyCircleOverlay(pg.EllipseROI):
#     def __init__(self, pos, size, **args):
#         pg.ROI.__init__(self, pos, size, **args)
#         self.aspectLocked = True

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