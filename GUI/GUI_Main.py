# Main class for Water Column Plotter.

# Lynette Davis
# Center for Coastal and Ocean Mapping
# November 2021

import ctypes
import json
import multiprocessing
from PlotterMain import PlotterMain
from PyQt5.QtWidgets import QAction, QApplication, QFileDialog, QGroupBox, QLabel, QMainWindow, QMdiArea, QMdiSubWindow, QMessageBox, QTextEdit, QToolBar, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt, QThread, QThreadPool, QTimer
import pyqtgraph as pg
import sys

from GUI.Dialogs.PYFiles.AllSettingsDialog2 import AllSettingsDialog2
from GUI_MDI import GUI_MDI
from GUI_Toolbar import GUI_Toolbar


from KongsbergDGMain import KongsbergDGMain

__appname__ = "Water Column Plotter"


class MainWindow(QMainWindow):
    count = 0

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        # Default settings:
        # TODO: Are we doing anything with maxHeave?
        self.settings = {'system_settings': {'system': "Kongsberg"},
                         'ip_settings': {'ip': '127.0.0.1', 'port': 8080},
                         'processing_settings': {'binSize_m': 0.20, 'acrossTrackAvg_m': 10, 'depth_m': 10,
                                                 'depthAvg_m': 10, 'alongTrackAvg_ping': 5, 'dualSwathPolicy': 0},
                         'pie_settings': {'maxHeave_m': 5, 'maxGridCells': 500, 'maxBufferSize': 10000}}

        # Shared queue to contain pie objects:
        self.queue_pie = multiprocessing.Queue()
        # Shared value to communicate 'play' (True) or 'stop' (False) status between main and multiprocessing processes:
        self.processBoolean = multiprocessing.Value(ctypes.c_bool, True)

        self.threadPool = QThreadPool(parent=self)
        print("Multithreading with maximum %d threads" % self.threadPool.maxThreadCount())

        self.PLOT_UPDATE_INTERVAL = 1000  # Milliseconds

        # Window setup:
        self.resize(1200, 800)
        self.setWindowTitle("Water Column Plotter")

        # Menu Bar
        #self.__initMenuBar()

        # Tool Bar
        self.toolBar = self.__initToolBar()

        # Multiple Document Interface
        self.mdi = self.__initMDI()
        self.setCentralWidget(self.mdi)

        # To be set via signal/slot of SettingsDialog.py, when system_settings:system is changed.
        # TODO: Some sort of error handling and graceful closing of threads
        #  if system is changed while another system thread is running!
        #self.sonarProcess = None
        self.sonarMain = None

        # TODO: Note to self: Plotter has nothing to plot until pies are made and put in queue_pie;
        #  nothing is placed in queue_pie until KongsbergDGMain is initiated by selecting a sonar system.
        self.plotterMain = PlotterMain(self.settings, self.queue_pie, self.processBoolean)
        # self.threadPool.start(self.plotterMain)

        self.show()

        # self.settingsDialog = AllSettingsDialog(self.settings, self)
        # self.__connectSettingsSignalsSlots()
        #self.settingsDialog = self.__initSettingsDialog()
        self.displaySettingsDialog()  # This will block until OK or Close / Cancel is selected in settings dialog

        #self.workerThread = WorkerThread(self.settings)
        #self.workerThread.start()
        self.timer = QTimer()
        self.timer.timeout.connect(self.updatePlot)
        # self.timer.start(self.PLOT_UPDATE_INTERVAL)

    def playProcesses(self):
        """
        Initiates self.sonarMain and self.plotterMain processes in new threads from QThreadPool.
        """
        # Play button is pressed. Disable play button; enable stop button
        self.toolBar.toolButtonPlay.setDisabled(True)
        # self.toolBar.toolButtonPlay.setStyleSheet("background-color : rgb(158, 158, 158)")  # Grey
        self.toolBar.toolButtonPlay.setStyleSheet("background-color : rgb(154, 171, 155)")  # Green

        self.toolBar.toolButtonStop.setEnabled(True)
        self.toolBar.toolButtonStop.setStyleSheet("background-color : rgb(240, 240, 240)")

        # Ensure shared process_boolean is set to True
        self.processBoolean.value = True

        # Initiate processes
        self.__playSystemProcess()
        self.__playPlotterProcess()

    def __playSystemProcess(self):
        """
        Initiates self.sonarMain processe in new threads from QThreadPool.
        """
        if self.sonarMain is None:
            if self.settings["system_settings"]["system"] == "Kongsberg":  # Kongsberg system
                # print("Launching Kongsberg thread.")
                # Note: Must maintain reference to sonarMain with 'self.':
                self.sonarMain = KongsbergDGMain(self.settings, self.queue_pie, self.processBoolean)
                self.threadPool.start(self.sonarMain)
            else:  # Other system
                # TODO: Alter code when additional systems supported.
                QMessageBox.warning(self, "Warning", "Currently supporting only Kongsberg systems.")
                # self.sonarMain = <SystemMain>
                # self.threadPool.start(self.sonarMain)
        else:  # self.sonarMain is not None
            if isinstance(self.sonarMain, KongsbergDGMain):  # Kongsberg system
                self.threadPool.start(self.sonarMain)
            else:  # Other system
                # TODO: Alter code when additional systems supported.
                QMessageBox.warning(self, "Warning", "Currently supporting only Kongsberg systems.")
                # self.threadPool.start(self.sonarMain)

    def __playPlotterProcess(self):
        """Initiates self.plotterMain processe in new threads from QThreadPool."""
        # NOTE: This method of launching plotterMain and timer results in a weird error, but it still works *shrug*:
        # ImportError: cannot import name 'Popen' from partially initialized module
        # 'multiprocessing.popen_spawn_win32' (most likely due to a circular import)
        # Documented here: https://github.com/dask/distributed/issues/4168
        self.threadPool.start(self.plotterMain)
        self.timer.start(self.PLOT_UPDATE_INTERVAL)

    def stopProcesses(self):
        """
        Allows self.sonarMain and self.plotterMain processes to end by setting self.process_boolean flag to False.
        """
        # Stop button is pressed. Disable stop button; enable play button
        self.toolBar.toolButtonStop.setDisabled(True)
        # self.toolBar.toolButtonStop.setStyleSheet("background-color : rgb(158, 158, 158)")  # Grey
        self.toolBar.toolButtonStop.setStyleSheet("background-color : rgb(219, 141, 141)")  # Red

        self.toolBar.toolButtonPlay.setEnabled(True)
        self.toolBar.toolButtonPlay.setStyleSheet("background-color : rgb(240, 240, 240)")

        # Ensure shared process_boolean is set to True
        # TODO: Unsure whether it's necessary to lock this? Ask Steve?
        with self.processBoolean.get_lock():
            self.processBoolean.value = False

        self.__stopSystemProcess()
        self.__stopPlotterProcess()

    def __stopSystemProcess(self):
        """
        Probably unnecessary. self.process_boolean = False should allow these processes to end.
        """
        pass

    def __stopPlotterProcess(self):
        """
        Probably unnecessary. self.process_boolean = False should allow these processes to end.
        """
        pass

    def updatePlot(self):
        print("Update settings: ", self.settings['processing_settings']['binSize_m'])
        print("LEN ", len(self.plotterMain.plotter.vertical_slice_buffer))
        images = self.plotterMain.plotter.retrieve_plot_matrices()

        self.mdi.pieWidget.pie_plot.setImage(images[0], autoRange=False,
                                             autoLevels=False, levels=(-95, 35),
                                             autoHistogramRange=False,
                                             pos=(-(self.settings['pie_settings']['maxGridCells'] / 2), 0))

        self.mdi.verticalWidget.vertical_plot.setImage(images[1], autoRange=False,
                                                       autoLevels=False, levels=(-95, 35),
                                                       autoHistogramRange=False,
                                                       pos=(-len(images[1]), 0))

        # TODO: Confirm that setPos y position is correct:
        self.mdi.horizontalWidget.horizontal_plot.setImage(images[2], autoRange=False,
                                                           autoLevels=False, levels=(-95, 35),
                                                           autoHistogramRange=False,
                                                           pos=(-len(images[2]),
                                                                -(self.settings['pie_settings']['maxGridCells'] / 2)))

    def __initMenuBar(self):
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

    def __initToolBar(self):
        toolBar = GUI_Toolbar(self.settings, parent=self)
        self.addToolBar(toolBar)

        # Signals / Slots
        toolBar.signalPlay.connect(self.playProcesses)
        toolBar.signalStop.connect(self.stopProcesses)
        toolBar.signalSettings.connect(self.displaySettingsDialog)

        return toolBar

    def __initMDI(self):
        mdi = GUI_MDI(self.settings, parent=self)

        # Signals / Slots
        mdi.verticalWidget.signalAcrossTrackAvgEdited.connect(self.acrossTrackAvgEdited)
        mdi.pieWidget.signalbinSizeEdited.connect(self.binSizeEdited)
        mdi.horizontalWidget.signalDepthEdited.connect(self.depthEdited)
        mdi.horizontalWidget.signalDepthAvgEdited.connect(self.depthAvgEdited)

        return mdi

    # def __initSettingsDialog(self):
    #     settingsDialog = AllSettingsDialog(self.settings, parent=self)
    #
    #     # Signals / Slots
    #     settingsDialog.pushButtonLoadSettings.clicked.connect(lambda: self.displayLoadSettingsDialog(settingsDialog))
    #     settingsDialog.signalSystemEdited.connect(self.systemEdited)
    #     settingsDialog.signalIPEdited.connect(self.ipEdited)
    #     settingsDialog.signalPortEdited.connect(self.portEdited)
    #     settingsDialog.signalBinSizeEdited.connect(lambda: self.binSizeEdited(fromSettingsDialog=True))
    #     settingsDialog.signalAcrossTrackAvgEdited.connect(lambda: self.acrossTrackAvgEdited(fromSettingsDialog=True))
    #     settingsDialog.signalDepthEdited.connect(lambda: self.depthEdited(fromSettingsDialog=True))
    #     settingsDialog.signalDepthAvgEdited.connect(lambda: self.depthAvgEdited(fromSettingsDialog=True))
    #     settingsDialog.signalAlongTrackAvgEdited.connect(self.alongTrackAvgEdited)
    #     settingsDialog.signalDualSwathPolicyEdited.connect(self.dualSwathAvgEdited)
    #
    #     return settingsDialog


    def systemEdited(self):
        print("SYSTEM EDITED")
        # self.mdi.subwindowSettingsDisplay.setSystem(self.settings)
        self.toolBar.setSystem(self.settings['system_settings']['system'])

        # if self.settings["system_settings"]["system"] == "Kongsberg":
        #     # Launch Kongsberg thread:
        #     if self.sonarMain is None:
        #         # Note: Must maintain reference to this with 'self.':
        #         # self.sonarProcess = LaunchSonarProcess(KongsbergDGMain(self.settings, self.queue_pie))
        #         # self.sonarProcess.start()
        #         self.sonarMain = KongsbergDGMain(self.settings, self.queue_pie)
        #         self.threadPool.start(self.sonarMain)
        #
        #         print(self.settings["ip_settings"]["ip"])
        #         print("Launching KongsbergMain")
        #     else:
        #         # TODO: Error checking. Do you really want to change systems? If yes, close previous thread.
        #         pass
        #     # while True:
        #     #     print(self.queue_pie.qsize())
        #     pass
        # else:  # self.settings["system_settings"]["system"] == "Other"
        #     # Launch other processing code: XXX
        #     # Note: This is currently disabled by error checks in
        #     # SettingsDialog.py that do now allow selection of "Other"
        #     # EX: self.sonarProcess = ResonThread(), self.sonarProcess = r2SonicThread()
        #     pass

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

    def binSizeEdited(self, fromSettingsDialog=False):
        print("in binsize edited slot")
        print(fromSettingsDialog)
        # self.mdi.subwindowSettingsDisplay.setBinSize(self.settings)
        # Only need to update MDI windows if setting was updated in settings dialog:
        if fromSettingsDialog:
            self.mdi.pieWidget.setBinSize(self.settings['processing_settings']['binSize_m'])

    def acrossTrackAvgEdited(self, fromSettingsDialog=False):
        #self.mdi.subwindowSettingsDisplay.setAcrossTrackAvg(self.settings)
        # Only need to update MDI windows if setting was updated in settings dialog:
        if fromSettingsDialog:
            self.mdi.verticalWidget.setAcrossTrackAvg(self.settings['processing_settings']['acrossTrackAvg_m'])

    def depthEdited(self, fromSettingsDialog=False):
        #self.mdi.subwindowSettingsDisplay.setDepth(self.settings)
        # Only need to update MDI windows if setting was updated in settings dialog:
        if fromSettingsDialog:
            self.mdi.horizontalWidget.setDepth(self.settings['processing_settings']['depth_m'])

    def depthAvgEdited(self, fromSettingsDialog=False):
        #self.mdi.subwindowSettingsDisplay.setDepthAvg(self.settings)
        # Only need to update MDI windows if setting was updated in settings dialog:
        if fromSettingsDialog:
            self.mdi.horizontalWidget.setDepthAvg(self.settings['processing_settings']['depthAvg_m'])

    def alongTrackAvgEdited(self):
        #self.mdi.subwindowSettingsDisplay.setAlongTrackAvg(self.settings)
        pass

    def dualSwathAvgEdited(self):
        #self.mdi.subwindowSettingsDisplay.setDualSwathPolicy(self.settings)
        pass

    def newActionSlot(self):
        sub = QMdiSubWindow()
        sub.setWidget(QTextEdit())
        sub.setWindowTitle("subwindow" + str(MainWindow.count))
        self.mdi.addSubWindow(sub)
        sub.show()

    def cascadeActionSlot(self):
        self.mdi.cascadeSubWindows()

    def tileActionSlot(self):
        self.mdi.tileSubWindows()

    def displaySettingsDialog(self):
        settingsDialog = AllSettingsDialog2(self.settings, parent=self)

        # Signals / Slots
        settingsDialog.pushButtonLoadSettings.clicked.connect(lambda: self.displayLoadSettingsDialog(settingsDialog))
        settingsDialog.pushButtonSaveSettings.clicked.connect(self.displaySaveSettingsDialog)
        settingsDialog.signalSystemEdited.connect(self.systemEdited)
        settingsDialog.signalIPEdited.connect(self.ipEdited)
        settingsDialog.signalPortEdited.connect(self.portEdited)
        settingsDialog.signalBinSizeEdited.connect(lambda: self.binSizeEdited(fromSettingsDialog=True))
        settingsDialog.signalAcrossTrackAvgEdited.connect(lambda: self.acrossTrackAvgEdited(fromSettingsDialog=True))
        settingsDialog.signalDepthEdited.connect(lambda: self.depthEdited(fromSettingsDialog=True))
        settingsDialog.signalDepthAvgEdited.connect(lambda: self.depthAvgEdited(fromSettingsDialog=True))
        settingsDialog.signalAlongTrackAvgEdited.connect(self.alongTrackAvgEdited)
        settingsDialog.signalDualSwathPolicyEdited.connect(self.dualSwathAvgEdited)

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

class LaunchSonarProcess(QThread):
    def __init__(self, sonarMain, parent=None):
        super(LaunchSonarProcess, self).__init__(parent)

        self.sonarMain = sonarMain

    def run(self):
        self.sonarMain.run()

class LaunchPlotterProcess(QThread):
    def __init__(self, plotterMain, parent=None):
        super(LaunchPlotterProcess, self).__init__(parent)

        self.plotterMain = plotterMain

    def run(self):
        self.plotterMain.run()

class LaunchProcesses(QThread):

    def __init__(self, sonarMain, plottingMain, parent=None):
        super(LaunchProcesses, self).__init__(parent)
        print("LaunchProcesses")
        self.sonarMain = sonarMain
        self.plottingProcess = plottingMain

    def run(self):
        print("LaunchProcesses run")
        self.sonarMain.run()
        self.plottingMain.run()

def main():
    app = QApplication(sys.argv)
    form = MainWindow()
    form.show()
    #form.displaySettingsDialog()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()