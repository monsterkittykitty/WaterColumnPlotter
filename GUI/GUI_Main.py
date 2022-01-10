# Main class for Water Column Plotter.

# Lynette Davis
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# November 2021

import datetime
import json
import multiprocessing
import numpy as np
# import psutil
from PyQt5.QtWidgets import QAction, QApplication, QFileDialog, QMainWindow, QMdiSubWindow, QStatusBar, QTextEdit
from PyQt5.QtCore import QTimer
import sys
from WaterColumn import WaterColumn

from GUI.Dialogs.PYFiles.AllSettingsDialog2 import AllSettingsDialog2
from GUI_MDI import GUI_MDI
from GUI_StatusBar import GUI_StatusBar
# from GUI_StatusBar_Kongsberg import GUI_StatusBar_Kongsberg
from GUI_Toolbar import GUI_Toolbar

# TODO: Testing
from PyQt5.QtGui import QPen
from PyQt5.QtCore import Qt

from KongsbergDGMain import KongsbergDGMain

__appname__ = "Water Column Plotter"

# if hasattr(Qt, 'AA_EnableHighDpiScaling'):
#     QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
#
# if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
#     QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

class MainWindow(QMainWindow):
    count = 0

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        # Default settings:
        # TODO: Are we doing anything with maxHeave?
        # maxBufferSize based on ~1000 MWC datagrams per minute for 10 minutes (~16 per second).
        self.settings = {'system_settings': {'system': "Kongsberg"},
                         'ip_settings': {'ip': '127.0.0.1', 'port': 8080, 'protocol': "UDP",
                                         'socketBufferMultiplier': 4},
                         'processing_settings': {'binSize_m': 0.20, 'acrossTrackAvg_m': 10, 'depth_m': 2,
                                                 'depthAvg_m': 2, 'alongTrackAvg_ping': 5, 'dualSwathPolicy': 0,
                                                 'maxHeave_m': 5},
                         'buffer_settings': {'maxGridCells': 500, 'maxBufferSize_ping': 1000}}

        # TODO: Check available memory to assign buffer sizes?
        # available_mem_gb = psutil.virtual_memory().available / 1024 / 1024 / 1024

        # Shared queue to contain pie objects:
        self.queue_pie = multiprocessing.Queue()
        self.temp_queue = multiprocessing.Queue()

        # TODO: Set this in settings dialog?
        self.PLOT_UPDATE_INTERVAL = 1000  # Milliseconds

        # Window setup:
        self.resize(1200, 820)
        self.setWindowTitle("Water Column Plotter")

        self.waterColumn = WaterColumn(self.settings)
        self.update_timer = QTimer()

        # Menu Bar
        #self._initMenuBar()

        # Tool Bar
        self.toolBar = self._initToolBar()

        # Status Bar
        #self.status = None
        # self.setStatusBar(self.status)
        #self.status_update_timer = QTimer()
        self.status = self._initStatusBar()
        self.setStatusBar(self.status)

        # Multiple Document Interface
        #self.plot_update_timer = QTimer()
        self.mdi = self._initMDI()
        self.setCentralWidget(self.mdi)

        self.show()

        self.displaySettingsDialog()  # This will block until OK or Close / Cancel is selected in settings dialog

        # For testing:
        self.plot_update_count = 0

    def startProcesses(self):
        """
        This method called when toolbar's play button is pressed. Activates processes in WaterColumn class.
        """
        self.waterColumn.startProcesses()
        self.update_timer.start(self.PLOT_UPDATE_INTERVAL)
        #self.update_timer.start(self.PLOT_UPDATE_INTERVAL)

    def pauseProcesses(self):
        """
        This method called when toolbar's stop button is pressed. Deactivates processes in WaterColumn class.
        """
        self.waterColumn.pauseProcesses()
        self.update_timer.stop()
        #self.update_timer.stop()

    def updateStatusBar(self):
        # This implementation probably doesn't need to be system specific...
        # def updateStatusBarKongsberg(self):
        print("updateStatusBar")
        self.status.set_ping_counts(self.waterColumn.full_ping_count.value, self.waterColumn.discard_ping_count.value)

    def updatePlot(self):
        # self.toolBar.labelRxToLostValues.setText("BI")
        # print("update discard: ", self.waterColumn.discard_ping_count.value)
        # self.toolBar.labelRxToLostValues.setText(str(self.waterColumn.full_ping_count.value) + ":"
        #                                          + str(self.waterColumn.discard_ping_count.value))

        print("updatePlot")
        # TODO: Check that this is working! Should I make an update plot function
        #  specifically for Kongsberg, so this doesn't need to be checked every time?
        # if isinstance(self.status, GUI_StatusBar_Kongsberg):
        #     self.status.set_ping_counts(self.waterColumn.full_ping_count.value, self.waterColumn.discard_ping_count.value)

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


        if self.waterColumn.get_raw_buffer_length() > 0:
            print("raw buffer greater than zero")
            temp_pie = self.waterColumn.get_pie()
            if temp_pie is not None:
                if temp_pie.any():  # For debugging
                    print("temp_pie.shape", temp_pie.shape)
                #if temp_pie.any():
                self.mdi.pieWidget.pie_plot.setImage(temp_pie.T, autoRange=False,
                                                     autoLevels=False, autoHistogramRange=False,
                                                     pos=(-(temp_pie.shape[1] / 2),
                                                          -(self.settings['processing_settings']['maxHeave_m'] /
                                                            self.settings['processing_settings']['binSize_m'])))
                # # Plots vertical line
                # y = [0, 50]
                # x = [0, 0]
                # self.mdi.verticalWidget.plot.plot(x, y)

        print("proc buffer length: ", self.waterColumn.get_processed_buffer_length())
        if self.waterColumn.get_processed_buffer_length() > 0:
            self.plot_update_count += 1

            if self.plot_update_count == 300:
                dgTime_proc = self.waterColumn.shared_ring_buffer_processed.view_recent_pings(
                    self.waterColumn.shared_ring_buffer_processed.timestamp_buffer_avg, 1)
                dgTime_raw = self.waterColumn.shared_ring_buffer_raw.view_recent_pings(
                    self.waterColumn.shared_ring_buffer_raw.timestamp_buffer, 1)
                print("Current time: {}; dgTime_proc: {}, {}; dgTime_raw: {}, {}".format(datetime.datetime.utcnow(),
                                                                                         dgTime_proc[0],
                                                datetime.datetime.utcfromtimestamp(float(dgTime_proc[0])), dgTime_raw[0],
                                                datetime.datetime.utcfromtimestamp(float(dgTime_raw[0]))))

            temp_vertical = self.waterColumn.get_vertical_slice()
            if temp_vertical is not None:
                if temp_vertical.any():  # For debugging
                    print("temp_vertical.shape", temp_vertical.shape)
                # if temp_vertical.shape[0] > 0:
                # print("plotting vertical")
                self.mdi.verticalWidget.vertical_plot.setImage(temp_vertical, autoRange=False,
                                                               autoLevels=False, autoHistogramRange=False,
                                                               pos=(-temp_vertical.shape[0],
                                                                    -(self.settings['processing_settings']['maxHeave_m'] /
                                                                    self.settings['processing_settings']['binSize_m'])))
                print("updating vertical plot: maxHeave: {}, binSize: {}".format(self.settings['processing_settings']['maxHeave_m'], self.settings['processing_settings']['binSize_m']))

            temp_horizontal = self.waterColumn.get_horizontal_slice()
            if temp_horizontal is not None:
                if temp_horizontal.any():  # For debugging
                    print("temp_horizontal.shape", temp_horizontal.shape)
                # if temp_horizontal.shape[0] > 0:
                # print("plotting horizontal")
                self.mdi.horizontalWidget.horizontal_plot.setImage(temp_horizontal, autoRange=False,
                                                                   autoLevels=False, autoHistogramRange=False,
                                                                   pos=(-temp_horizontal.shape[0],
                                                                        -temp_horizontal.shape[1] / 2))
            else:
                print("temp_horizontal is none")

    # SYSTEM SETTINGS SLOTS:
    # TODO: Link to other processes
    def systemEdited(self):
        self.toolBar.setSystem(self.settings['system_settings']['system'])
        # TODO: Reset sonar main:
        # 1. end all currently running processing and relauch watercolumn?

    # IP SETTINGS SLOTS:
    # TODO: Link to other processes
    def ipEdited(self):
        # print("IP HAS BEEN EDITED: {}".format(self.settings["ip_settings"]["ip"]))
        # print("default ip: {}".format(self.defaultSettings["ip_settings"]["ip"]))
        # print("type(self.settings): ", type(self.settings))
        # print("ipEdited")
        #self.mdi.subwindowSettingsDisplay.setIPPort(self.settings)
        self.toolBar.setIPPort(self.settings['ip_settings']['ip'], self.settings['ip_settings']['port'])

    def portEdited(self):
        #self.mdi.subwindowSettingsDisplay.setIPPort(self.settings)
        self.toolBar.setIPPort(self.settings['ip_settings']['ip'], self.settings['ip_settings']['port'])

    def protocolEdited(self):
        pass

    def socketBufferEdited(self):
        pass

    # PROCESSING SETTINGS SLOTS:
    def binSizeEdited(self, fromSettingsDialog=False):
        print("binSizeEdited")
        #print(fromSettingsDialog)
        # self.mdi.subwindowSettingsDisplay.setBinSize(self.settings)
        # Only need to update MDI windows if setting was updated in settings dialog:
        if fromSettingsDialog:
            self.mdi.pieWidget.setBinSize(self.settings['processing_settings']['binSize_m'])
        with self.waterColumn.bin_size.get_lock():
            self.waterColumn.bin_size.value = self.settings['processing_settings']['binSize_m']

    def acrossTrackAvgEdited(self, fromSettingsDialog=False):
        print("acrossTrackAvgEdited")
        #self.mdi.subwindowSettingsDisplay.setAcrossTrackAvg(self.settings)
        # Only need to update MDI windows if setting was updated in settings dialog:
        if fromSettingsDialog:
            self.mdi.verticalWidget.setAcrossTrackAvg(self.settings['processing_settings']['acrossTrackAvg_m'])
        with self.waterColumn.across_track_avg.get_lock():
            self.waterColumn.across_track_avg.value = self.settings['processing_settings']['acrossTrackAvg_m']

    def depthEdited(self, fromSettingsDialog=False):
        print("depthEdited")
        #self.mdi.subwindowSettingsDisplay.setDepth(self.settings)
        # Only need to update MDI windows if setting was updated in settings dialog:
        if fromSettingsDialog:
            self.mdi.horizontalWidget.setDepth(self.settings['processing_settings']['depth_m'])
        with self.waterColumn.depth.get_lock():
            self.waterColumn.depth.value = self.settings['processing_settings']['depth_m']

    def depthAvgEdited(self, fromSettingsDialog=False):
        print("depthAvgEdited")
        #self.mdi.subwindowSettingsDisplay.setDepthAvg(self.settings)
        # Only need to update MDI windows if setting was updated in settings dialog:
        if fromSettingsDialog:
            self.mdi.horizontalWidget.setDepthAvg(self.settings['processing_settings']['depthAvg_m'])
        with self.waterColumn.depth_avg.get_lock():
            self.waterColumn.depth_avg.value = self.settings['processing_settings']['depthAvg_m']

    def alongTrackAvgEdited(self):
        print("alongTrackAvgEdited")
        #self.mdi.subwindowSettingsDisplay.setAlongTrackAvg(self.settings)
        with self.waterColumn.along_track_avg.get_lock():
            self.waterColumn.along_track_avg.value = self.settings['processing_settings']['alongTrackAvg_ping']
        pass

    # TODO: Remove this?
    def dualSwathAvgEdited(self):
        #self.mdi.subwindowSettingsDisplay.setDualSwathPolicy(self.settings)
        pass

    def heaveEdited(self):
        with self.waterColumn.max_heave.get_lock():
            self.waterColumn.max_heave.value = self.settings['processing_settings']['maxHeave_m']
        pass

    def processingSettingsEdited(self):
        print("IN PROCESSING SETTINGS EDITED SLOT")
        with self.waterColumn.processing_settings_edited.get_lock():
            self.waterColumn.processing_settings_edited.value = True

        self.waterColumn.settingsChanged()
        pass

    # BUFFER SETTINGS SLOTS:
    # TODO: Link to other processes
    def gridCellsEdited(self):
        pass

    def pingBufferEdited(self):
        pass

    # def newActionSlot(self):
    #     sub = QMdiSubWindow()
    #     sub.setWidget(QTextEdit())
    #     sub.setWindowTitle("subwindow" + str(MainWindow.count))
    #     self.mdi.addSubWindow(sub)
    #     sub.show()
    #
    # def cascadeActionSlot(self):
    #     self.mdi.cascadeSubWindows()
    #
    # def tileActionSlot(self):
    #     self.mdi.tileSubWindows()

    def displaySettingsDialog(self):
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
        settingsDialog.signalDualSwathPolicyEdited.connect(self.dualSwathAvgEdited)
        settingsDialog.signalHeaveEdited.connect(self.heaveEdited)
        settingsDialog.signalGridCellsEdited.connect(self.gridCellsEdited)
        settingsDialog.signalPingBufferEdited.connect(self.pingBufferEdited)

        settingsDialog.signalProcessingSettingsEdited.connect(self.processingSettingsEdited)

        settingsDialog.exec_()

    def displaySaveSettingsDialog(self):
        saveDialog = QFileDialog(self)
        filePath = saveDialog.getSaveFileName(self, __appname__, directory=".\Settings", filter="JSON (*.json)")

        if filePath[0]:
            with open(filePath[0], 'w') as f:
                json.dump(self.settings, f, indent=4)

    def displayLoadSettingsDialog(self, settingsDialog):
        openDialog = QFileDialog(self)
        filePath = openDialog.getOpenFileName(self,  __appname__, directory=".\Settings", filter="JSON (*.json)")

        if filePath[0]:
            with open(filePath[0], 'r') as f:
                tempSettings = json.load(f)

            settingsDialog.validateAndSetValuesFromFile(tempSettings)

    def closeEvent(self, event):
        with self.waterColumn.process_flag.get_lock():
            self.waterColumn.process_flag.value = False
        # if self.waterColumn.process_flag.value:
        #     self.waterColumn.process_flag.value = False
        #     self.waterColumn.sonarMain.dg_capture.join()
        #     self.waterColumn.sonarMain.dg_process.join()
        #     self.waterColumn.plotterMain.plotter.join()
        # if self.waterColumn.sonarMain:
        #     self.waterColumn.sonarMain.dg_capture.sock_in.close()
        # Quit using shared memory in the frontend
        self.waterColumn.closeSharedMemory()
        # Release shared memory definitely
        self.waterColumn.unlinkSharedMemory()

        event.accept()

    def _initMenuBar(self):
        menuBar = self.menuBar()

        # (Sample menu bar...)
        # Menu bar - File:
        # file = menuBar.addMenu("File")
        # newAction = QAction("New", self)
        # cascadeAction = QAction("Cascade", self)
        # tileAction = QAction("Tile", self)
        # file.addAction(newAction)
        # file.addAction(cascadeAction)
        # file.addAction(tileAction)
        # newAction.triggered.connect(self.newActionSlot)
        # cascadeAction.triggered.connect(self.cascadeActionSlot)
        # tileAction.triggered.connect(self.tileActionSlot)

        # Menu bar - Settings:
        settings = menuBar.addMenu("Settings")

        allSettingsAction = QAction("All Settings", self)
        saveSettingsAction = QAction("Save Settings", self)
        loadSettingsAction = QAction("Load Settings", self)

        settings.addAction(allSettingsAction)
        settings.addSeparator()
        settings.addAction(saveSettingsAction)
        settings.addAction(loadSettingsAction)

        # Signals / SLots
        allSettingsAction.triggered.connect(self.testSettingsDialog)
        #allSettingsAction.triggered.connect(self.displaySettingsDialog)
        saveSettingsAction.triggered.connect(self.displaySaveSettingsDialog)
        self.tempdialog = AllSettingsDialog2(self.settings)
        loadSettingsAction.triggered.connect(lambda: self.displayLoadSettingsDialog(self.tempdialog))

    def _initToolBar(self):
        toolBar = GUI_Toolbar(self.settings, parent=self)
        self.addToolBar(toolBar)

        # Signals / Slots
        toolBar.signalPlay.connect(self.startProcesses)
        toolBar.signalPause.connect(self.pauseProcesses)
        toolBar.signalSettings.connect(self.displaySettingsDialog)

        return toolBar

    def _initStatusBar(self):
        statusBar = GUI_StatusBar()
        self.update_timer.timeout.connect(self.updateStatusBar)

        # This implementation probably doesn't need to be system specific...
        # if self.settings['system_settings']['system'] == "Kongsberg":
        #     statusBar = GUI_StatusBar_Kongsberg()
        #     self.status_update_timer.timeout.connect(self.updateStatusBarKongsberg)
        # else:  # Another system
        #     statusBar = QStatusBar()
        #     # self.status_update_timer.timeout.connect([<some_other_system_update_function>])

        return statusBar

    # def _initKongsberStatusBar(self):
    #     statusBar = GUI_StatusBar_Kongsberg()
    #
    #     return statusBar

    def _initMDI(self):
        mdi = GUI_MDI(self.settings, parent=self)

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

# TODO: Test:
import pyqtgraph as pg
class MyCircleOverlay(pg.EllipseROI):
    def __init__(self, pos, size, **args):
        pg.ROI.__init__(self, pos, size, **args)
        self.aspectLocked = True

def main():
    app = QApplication(sys.argv)
    form = MainWindow()
    form.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()