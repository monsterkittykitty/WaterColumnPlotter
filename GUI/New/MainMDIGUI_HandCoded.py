import copy
import json
#from LaunchProcesses import LaunchProcesses
import multiprocessing
import numpy as np
from Plotter import Plotter
from PlotterMain import PlotterMain
from PyQt5.QtWidgets import QAction, QApplication, QFileDialog, QMainWindow, QMdiArea, QMdiSubWindow, QMessageBox, \
    QTextEdit
from PyQt5.QtCore import pyqtSignal, QJsonDocument, Qt, QThread, QThreadPool, QTimer
import pyqtgraph as pg
from pyvistaqt import BackgroundPlotter, QtInteractor
from qtrangeslider import QRangeSlider
import sys

from GUI.New.PYFiles.SettingsDialog import SettingsDialog
from GUI.New.SubwindowPieSliceWidget import SubwindowPieSliceWidget
from GUI.New.SubwindowVerticalSliceWidget import SubwindowVerticalSliceWidget
from GUI.New.SubwindowHorizontalSliceWidget import SubwindowHorizontalSliceWidget
from GUI.New.SubwindowSettingsDisplay import SubwindowSettingsDisplay

from KongsbergDGMain import KongsbergDGMain

__appname__ = "Water Column Plotter"

class MainWindow(QMainWindow):
    count = 0

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        # Default settings:
        # TODO: Make this loadable from / savable to file:
        self.defaultSettings = {"system_settings": {"system": ""},
                                "ip_settings": {"ip": "127.0.0.1", "port": 8080},
                                "processing_settings": {"binSize_m": 0.25, "acrossTrackAvg_m": 10, "depthAvg_m": 10,
                                                        "alongTrackAvg_ping": 5, "depthDisplay_m": 10,
                                                        "dualSwathPolicy": 0, "maxHeave_m": 5}}

        # TODO: Make this loadable from / savable to file:
        self.settings = copy.deepcopy(self.defaultSettings)

        # Shared queue to contain pie objects:
        self.queue_pie = multiprocessing.Queue()

        self.threadPool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadPool.maxThreadCount())

        self.PLOT_UPDATE_INTERVAL = 1000  # Milliseconds

        # Window setup:
        self.resize(1200, 800)

        self.setWindowTitle("Water Column Plotter")

        # Menu Bar
        self.setupMenuBar()

        self.mdi = QMdiArea()
        self.setCentralWidget(self.mdi)

        # self.verticalPlot = pg.PlotWidget()
        # self.piePlot = pg.PlotWidget()
        # self.horizontalPlot = pg.PlotWidget()

        # To be set via signal/slot of SettingsDialog.py, when system_settings:system is changed.
        # TODO: Some sort of error handling and graceful closing of threads
        #  if system is changed while another system thread is running!
        #self.sonarProcess = None
        self.sonarMain = None

        # TODO: Note to self: Plotter has nothing to plot until pies are made and put in queue_pie;
        #  nothing is placed in queue_pie until KongsbergDGMain is initiated by selecting a sonar system.
        self.plotterMain = PlotterMain(self.settings, self.queue_pie)
        self.threadPool.start(self.plotterMain)

        # Open 3 subwindows for plots; 1 subwindow to display settings:
        # VERTICAL:
        # self.verticalWidget = SubwindowVerticalSliceWidget(self.verticalPlot)
        self.verticalWidget = SubwindowVerticalSliceWidget()
        # PIE
        # self.pieWidget = SubwindowPieSliceWidget(self.piePlot)
        self.pieWidget = SubwindowPieSliceWidget()
        # HORIZONTAL:
        # self.horizontalWidget = SubwindowHorizontalSliceWidget(self.horizontalPlot)
        self.horizontalWidget = SubwindowHorizontalSliceWidget()
        # SETTINGS:
        self.subwindowSettingsDisplay = SubwindowSettingsDisplay(self.settings)

        self.openAllSubwindows()

        self.show()

        self.settingsDialog = SettingsDialog(self.settings, self)
        self.settingsDialog.signalSystemEdited.connect(self.systemEdited)
        self.settingsDialog.signalIPEdited.connect(self.ipEdited)
        self.settingsDialog.signalPortEdited.connect(self.portEdited)
        self.settingsDialog.signalBinSizeEdited.connect(self.binSizeEdited)
        self.settingsDialog.signalAcrossTrackAvgEdited.connect(self.acrossTrackAvgEdited)
        self.settingsDialog.signalDepthAvgEdited.connect(self.depthAvgEdited)
        self.settingsDialog.signalAlongTrackAvgEdited.connect(self.alongTrackAvgEdited)
        self.settingsDialog.signalDualSwathPolicyEdited.connect(self.dualSwathAvgEdited)

        self.displaySettingsDialog()

        #self.workerThread = WorkerThread(self.settings)
        #self.workerThread.start()
        self.timer = QTimer()
        self.timer.timeout.connect(self.updatePlot)
        self.timer.start(self.PLOT_UPDATE_INTERVAL)

    def updatePlot(self):
        print("LEN ", len(self.plotterMain.plotter.vertical_slice_buffer))
        images = self.plotterMain.plotter.retrieve_plot_matrices()
        self.pieWidget.pie_plot.setImage(images[0])
        print("shape pie: ", images[0].shape)
        self.verticalWidget.vertical_plot.setImage(images[1])
        print("shape vert: ", images[1].shape)
        #self.horizontalWidget.horizontal_plot.setImage(images[2])

    def openAllSubwindows(self):
        self.setupVerticalPlotSubwindow()
        self.setupPiePlotSubwindow()
        self.setupHorizontalPlotSubwindow()
        self.setupSettingsSubwindow()

    def setupVerticalPlotSubwindow(self):
        verticalSubwindowMDI = self.mdi.addSubWindow(self.verticalWidget)
        verticalSubwindowMDI.resize(800, 400)
        # Disable close button:
        verticalSubwindowMDI.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint)

    def setupPiePlotSubwindow(self):
        pieSubwindowMDI = self.mdi.addSubWindow(self.pieWidget)
        pieSubwindowMDI.resize(400, 400)
        # Disable close button:
        pieSubwindowMDI.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint)

    def setupHorizontalPlotSubwindow(self):
        horizontalSubwindowMDI = self.mdi.addSubWindow(self.horizontalWidget)
        horizontalSubwindowMDI.resize(800, 400)
        # Disable close button:
        horizontalSubwindowMDI.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint)

    def setupSettingsSubwindow(self):
        settingsSubwindowMDI = self.mdi.addSubWindow(self.subwindowSettingsDisplay)
        settingsSubwindowMDI.resize(400, 400)
        # Disable close button:
        settingsSubwindowMDI.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint)

    def setupMenuBar(self):

        menuBar = self.menuBar()

        # Menu bar - File:
        file = menuBar.addMenu("File")
        newAction = QAction("New", self)
        cascadeAction = QAction("Cascade", self)
        tileAction = QAction("Tile", self)
        file.addAction(newAction)
        file.addAction(cascadeAction)
        file.addAction(tileAction)
        newAction.triggered.connect(self.newActionSlot)
        cascadeAction.triggered.connect(self.cascadeActionSlot)
        tileAction.triggered.connect(self.tileActionSlot)

        # Menu bar - Settings:
        settings = menuBar.addMenu("Settings")

        allSettingsAction = QAction("All Settings", self)
        saveSettingsAction = QAction("Save Settings", self)
        loadSettingsAction = QAction("Load Settings", self)

        settings.addAction(allSettingsAction)
        settings.addSeparator()
        settings.addAction(saveSettingsAction)
        settings.addAction(loadSettingsAction)

        allSettingsAction.triggered.connect(self.displaySettingsDialog)
        saveSettingsAction.triggered.connect(self.displaySaveSettingsDialog)
        loadSettingsAction.triggered.connect(self.displayLoadSettingsDialog)

    def systemEdited(self):
        self.subwindowSettingsDisplay.setSystem(self.settings)

        if self.settings["system_settings"]["system"] == "Kongsberg":
            # Launch Kongsberg thread:
            if self.sonarMain is None:
                # Note: Must maintain reference to this with 'self.':
                # self.sonarProcess = LaunchSonarProcess(KongsbergDGMain(self.settings, self.queue_pie))
                # self.sonarProcess.start()
                self.sonarMain = KongsbergDGMain(self.settings, self.queue_pie)
                self.threadPool.start(self.sonarMain)

                print(self.settings["ip_settings"]["ip"])
                print("Launching KongsbergMain")
            else:
                # TODO: Error checking. Do you really want to change systems? If yes, close previous thread.
                pass
            # while True:
            #     print(self.queue_pie.qsize())
            pass
        else:  # self.settings["system_settings"]["system"] == "Other"
            # Launch other processing code: XXX
            # Note: This is currently disabled by error checks in
            # SettingsDialog.py that do now allow selection of "Other"
            # EX: self.sonarProcess = ResonThread(), self.sonarProcess = r2SonicThread()
            pass

    def ipEdited(self):
        print("IP HAS BEEN EDITED: {}".format(self.settings["ip_settings"]["ip"]))
        print("default ip: {}".format(self.defaultSettings["ip_settings"]["ip"]))
        print("type(self.settings): ", type(self.settings))
        # print("ipEdited")
        self.subwindowSettingsDisplay.setIPPort(self.settings)

    def portEdited(self):
        self.subwindowSettingsDisplay.setIPPort(self.settings)

    def binSizeEdited(self):
        self.subwindowSettingsDisplay.setBinSize(self.settings)

    def acrossTrackAvgEdited(self):
        self.subwindowSettingsDisplay.setAcrossTrackAvg(self.settings)

    def depthAvgEdited(self):
        self.subwindowSettingsDisplay.setDepthAvg(self.settings)

    def alongTrackAvgEdited(self):
        self.subwindowSettingsDisplay.setAlongTrackAvg(self.settings)

    def dualSwathAvgEdited(self):
        self.subwindowSettingsDisplay.setDualSwathPolicy(self.settings)

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
        self.settingsDialog.exec_()

    def displaySaveSettingsDialog(self):
        saveDialog = QFileDialog(self)
        filePath = saveDialog.getSaveFileName(self, __appname__, directory=".\Settings", filter="JSON (*.json)")

        if filePath[0]:
            with open(filePath[0], 'w') as f:
                json.dump(self.settings, f, indent=4)

    def displayLoadSettingsDialog(self):
        openDialog = QFileDialog(self)
        filePath = openDialog.getOpenFileName(self,  __appname__, directory=".\Settings", filter="JSON (*.json)")

        if filePath[0]:
            with open(filePath[0], 'r') as f:
                tempSettings = json.load(f)

            self.settingsDialog.validateAndloadValues(tempSettings)

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


# class LaunchProcesses(QThread):
#
#     def __init__(self, settings, queue_pie, vertical_plot, parent=None):
#         super(LaunchProcesses, self).__init__(parent)
#         print("LaunchProcesses")
#
#         #self.processes = LaunchProcesses(settings, queue_pie, vertical_plot)
#         self.kongsbergMain = KongsbergDGMain(settings, queue_pie)
#         print("LaunchProcesses plotter")
#         self.plotterThread = PlotterThread(vertical_plot)
#
#     def run(self):
#         print("LaunchProcesses.run")
#         #self.threadDone.emit("string1", "string2")
#         self.kongsbergMain.run()
#         self.plotterThread.run()
#         #self.processes.run()
#
#         self.kongsbergMain.process_consumer.join()
#         self.kongsbergMain.process_producer.join()
#         self.plotterThread.process_plotter.join()

# class KongsbergThread(QThread):
#     def __init__(self, settings, queue, parent=None):
#         super(KongsbergThread, self).__init__(parent)
#         self.kg = KongsbergDGMain(settings, queue)
#
#     def run(self):
#         print("kongsthread running")
#         self.kg.run()

def main():
    app = QApplication(sys.argv)
    form = MainWindow()
    form.show()
    #form.displaySettingsDialog()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()