import copy
import json
import multiprocessing
from PyQt5.QtWidgets import QAction, QApplication, QFileDialog, QMainWindow, QMdiArea, QMdiSubWindow, QMessageBox, \
    QTextEdit
from PyQt5.QtCore import pyqtSignal, QJsonDocument, Qt, QThread
import pyqtgraph as pg
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
                                                        "alongTrackAvg_ping": 5, "dualSwathPolicy": 0}}

        # TODO: Make this loadable from / savable to file:
        self.settings = copy.deepcopy(self.defaultSettings)

        # Shared queue to contain pie matrices:
        self.queue_pie = multiprocessing.Queue()

        # Window setup:
        self.resize(1000, 800)

        self.setWindowTitle("Water Column Plotter")

        # Menu Bar
        self.setupMenuBar()

        self.mdi = QMdiArea()
        self.setCentralWidget(self.mdi)

        # To be set via signal/slot of SettingsDialog.py, when system_settings:system is changed.
        # TODO: Some sort of error handling and graceful closing of threads
        #  if system is changed while another system thread is running!

        self.sonarThread = None

        self.plotterThread = PlotterThread()
        self.plotterThread.start()

        # Open 3 subwindows for plots:
        # VERTICAL:
        self.verticalPlot = pg.PlotWidget()
        self.verticalWidget = SubwindowVerticalSliceWidget(self.verticalPlot)
        self.verticalSubwindowMDI = self.mdi.addSubWindow(self.verticalWidget)
        self.verticalSubwindowMDI.resize(600, 300)
        # Disable close button:
        self.verticalSubwindowMDI.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint)

        # PIE
        self.piePlot = pg.PlotWidget()
        self.pieWidget = SubwindowPieSliceWidget(self.piePlot)
        self.pieSubwindowMDI = self.mdi.addSubWindow(self.pieWidget)
        self.pieSubwindowMDI.resize(300, 300)
        # Disable close button:
        self.pieSubwindowMDI.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint)

        # HORIZONTAL:
        self.horizontalPlot = pg.PlotWidget()
        self.horizontalWidget = SubwindowHorizontalSliceWidget(self.horizontalPlot)
        self.horizontalSubwindowMDI = self.mdi.addSubWindow(self.horizontalWidget)
        self.horizontalSubwindowMDI.resize(600, 300)
        # Disable close button:
        self.horizontalSubwindowMDI.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint)

        # Open 1 subwindow to display settings:
        self.subwindowSettingsDisplay = SubwindowSettingsDisplay(self.settings)
        self.settingsSubwindowMDI = self.mdi.addSubWindow(self.subwindowSettingsDisplay)
        self.settingsSubwindowMDI.resize(300, 300)
        self.settingsSubwindowMDI.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint)

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
            if self.sonarThread is None:
                self.sonarThread = KongsbergThread(self.settings, self.queue_pie)
                self.sonarThread.start()

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
            # EX: self.sonarThread = ResonThread(), self.sonarThread = r2SonicThread()
            pass

    def ipEdited(self):
        # print("IP HAS BEEN EDITED: {}".format(self.settings["ip_settings"]["ip"]))
        # print("default ip: {}".format(self.defaultSettings["ip_settings"]["ip"]))
        # print("type(self.settings): ", type(self.settings))
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

class KongsbergThread(QThread):

    def __init__(self, settings, queue_pie, parent=None):
        super(KongsbergThread, self).__init__(parent)

        self.kongsbergMain = KongsbergDGMain(settings, queue_pie)

    def run(self):
        #self.threadDone.emit("string1", "string2")
        self.kongsbergMain.run()

class PlotterThread(QThread):

    def __init__(self, parent=None):
        super(PlotterThread, self).__init__(parent)

    # TODO: Initiate plotter! This should probably be its own file!
    def run(self):
        print("Plotting!")
        pass

def main():
    app = QApplication(sys.argv)
    form = MainWindow()
    form.show()
    #form.displaySettingsDialog()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()