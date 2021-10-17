import copy
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import pyqtgraph as pg
from qtrangeslider import QRangeSlider
import sys

from GUI.New.PYFiles.SettingsDialog import SettingsDialog
from GUI.New.SubwindowPieSliceWidget import SubwindowPieSliceWidget
from GUI.New.SubwindowVerticalSliceWidget import SubwindowVerticalSliceWidget
from GUI.New.SubwindowHorizontalSliceWidget import SubwindowHorizontalSliceWidget
from GUI.New.SubwindowSettingsDisplay import SubwindowSettingsDisplay

from KongsbergDGMain import KongsbergDGMain


class MainWindow(QtWidgets.QMainWindow):
    count = 0

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        # Default settings:
        # TODO: Make this loadable from / savable to file:
        self.defaultSettings = {"system_settings": {"system": "Kongsberg"},
                                "ip_settings": {"ip": "127.0.0.1", "port": 8080},
                                "processing_settings": {"binSize_m": 0.25, "acrossTrackAvg_m": 10, "depthAvg_m": 10,
                                                        "alongTrackAvg_ping": 5, "dualSwathPolicy": 0}}

        # TODO: Make this loadable from / savable to file:
        self.settings = copy.deepcopy(self.defaultSettings)

        self.resize(1000, 800)

        self.setWindowTitle("Water Column Plotter")

        self.mdi = QtWidgets.QMdiArea()
        self.setCentralWidget(self.mdi)

        # Menu bar:
        menuBar = self.menuBar()

        # Menu bar - File:
        file = menuBar.addMenu("File")
        newAction = QtWidgets.QAction("New", self)
        cascadeAction = QtWidgets.QAction("Cascade", self)
        tileAction = QtWidgets.QAction("Tile", self)
        file.addAction(newAction)
        file.addAction(cascadeAction)
        file.addAction(tileAction)
        newAction.triggered.connect(self.newActionSlot)
        cascadeAction.triggered.connect(self.cascadeActionSlot)
        tileAction.triggered.connect(self.tileActionSlot)

        # Menu bar - Settings:
        settings = menuBar.addMenu("Settings")
        allSettingsAction = QtWidgets.QAction("All Settings", self)
        settings.addAction(allSettingsAction)
        allSettingsAction.triggered.connect(self.displaySettingsDialog)

        # Open 3 subwindows for plots:
        self.verticalSubwindow = self.mdi.addSubWindow(SubwindowVerticalSliceWidget())
        self.verticalSubwindow.resize(600, 300)
        # Disable close button:
        self.verticalSubwindow.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint)

        self.pieSubwindow = self.mdi.addSubWindow(SubwindowPieSliceWidget())
        self.pieSubwindow.resize(300, 300)
        # Disable close button:
        self.pieSubwindow.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint)

        self.horizontalSubwindow = self.mdi.addSubWindow(SubwindowHorizontalSliceWidget())
        self.horizontalSubwindow.resize(600, 300)
        # Disable close button:
        self.horizontalSubwindow.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint)

        # Open 1 subwindow to display settings:
        self.subwindowSettingsDisplay = SubwindowSettingsDisplay(self.settings)
        self.settingsSubwindow = self.mdi.addSubWindow(self.subwindowSettingsDisplay)
        self.settingsSubwindow.resize(300, 300)
        self.settingsSubwindow.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint)

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

        print("after displaysettingsdialog")

        # self.kongsbergDG = KongsbergDGMain(rx_ip=self.settings["ip_settings"]["ip"],
        #                                    rx_port=self.settings["ip_settings"]["port"],
        #                                    bin_size=self.settings["processing_settings"]["binSize_m"])

        #self.workerThread = WorkerThread(self.settings)
        #self.workerThread.start()

    def systemEdited(self):
        self.subwindowSettingsDisplay.setSystem(self.settings)

    def ipEdited(self):
        # print("IP HAS BEEN EDITED: {}".format(self.settings["ip_settings"]["ip"]))
        # print("default ip: {}".format(self.defaultSettings["ip_settings"]["ip"]))
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
        sub = QtWidgets.QMdiSubWindow()
        sub.setWidget(QtWidgets.QTextEdit())
        sub.setWindowTitle("subwindow" + str(MainWindow.count))
        self.mdi.addSubWindow(sub)
        sub.show()

    def cascadeActionSlot(self):
        self.mdi.cascadeSubWindows()

    def tileActionSlot(self):
        self.mdi.tileSubWindows()

    def displaySettingsDialog(self):
        self.settingsDialog.exec_()


class WorkerThread(QThread):

    def __init__(self, settings, parent=None):
        super(WorkerThread, self).__init__(parent)

        self.kongsbergDG = KongsbergDGMain(rx_ip=settings["ip_settings"]["ip"],
                                           rx_port=settings["ip_settings"]["port"],
                                           bin_size=settings["processing_settings"]["binSize_m"])

    def run(self):
        #self.threadDone.emit("string1", "string2")
        self.kongsbergDG.run()

def main():
    app = QtWidgets.QApplication(sys.argv)
    form = MainWindow()
    form.show()
    #form.displaySettingsDialog()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()