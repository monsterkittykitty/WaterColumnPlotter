from PyQt5.QtWidgets import QFrame, QGridLayout, QLabel, QWidget
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from qtrangeslider import QRangeSlider
import sys


class SubwindowSettingsDisplay(QWidget):
    def __init__(self, settings, parent=None):
        super(SubwindowSettingsDisplay, self).__init__(parent)
        self.setWindowTitle("Current Settings")

        self.resize(200, 200)

        #layout = QtWidgets.QVBoxLayout()
        layout = QGridLayout()
        layout.setColumnMinimumWidth(0, 25)
        layout.setColumnMinimumWidth(1, 25)
        # layout.setColumnMinimumWidth(2, 25)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        # layout.setColumnStretch(2, 0)

        separator1 = QFrame()
        separator1.setFrameShape(QFrame.HLine)
        separator1.setLineWidth(1)
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setLineWidth(1)

        self.labelSystem = QLabel(self)
        self.labelIPPort = QLabel(self)
        self.labelBinSize = QLabel(self)
        self.labelAcrossTrackAvg = QLabel(self)
        self.labelDepthAvg = QLabel(self)
        self.labelAlongTrackAvg = QLabel(self)
        self.labelDualSwathPolicy = QLabel(self)
        self.setAllValues(settings)

        layout.addWidget(QLabel("System:"), 0, 0)
        layout.addWidget(self.labelSystem, 0, 1)
        layout.addWidget(separator1, 1, 0, 1, 2)
        layout.addWidget(QLabel("IP:Port:"), 2, 0)
        layout.addWidget(self.labelIPPort, 2, 1)
        layout.addWidget(separator2, 3, 0, 1, 2)
        layout.addWidget(QLabel("Bin Size (m):"), 4, 0)
        layout.addWidget(self.labelBinSize, 4, 1)
        #layout.addWidget(separator, 5, 0, 1, 2)
        layout.addWidget(QLabel("Across-Track Average (m):"), 6, 0)
        layout.addWidget(self.labelAcrossTrackAvg, 6, 1)
        #layout.addWidget(separator, 7, 0, 1, 2)
        layout.addWidget(QLabel("Depth Average (m):"), 8, 0)
        layout.addWidget(self.labelDepthAvg, 8, 1)
        #layout.addWidget(separator, 9, 0, 1, 2)
        layout.addWidget(QLabel("Along-Track Average (pings):"), 10, 0)
        layout.addWidget(self.labelAlongTrackAvg, 10, 1)
        #layout.addWidget(separator, 11, 0, 1, 2)
        layout.addWidget(QLabel("Dual Swath Policy:"), 12, 0)
        layout.addWidget(self.labelDualSwathPolicy, 12, 1)

        self.setLayout(layout)

    def setAllValues(self, settings):
        self.setSystem(settings)
        self.setIPPort(settings)
        self.setBinSize(settings)
        self.setAcrossTrackAvg(settings)
        self.setDepthAvg(settings)
        self.setAlongTrackAvg(settings)
        self.setDualSwathPolicy(settings)

    def setSystem(self, settings):
        self.labelSystem.setText(settings["system_settings"]["system"])

    def setIPPort(self, settings):
        self.labelIPPort.setText(settings["ip_settings"]["ip"] + " : " + str(settings["ip_settings"]["port"]))

    def setBinSize(self, settings):
        self.labelBinSize.setText(str(settings["processing_settings"]["binSize_m"]))

    def setAcrossTrackAvg(self, settings):
        self.labelAcrossTrackAvg.setText(str(settings["processing_settings"]["acrossTrackAvg_m"]))

    def setDepthAvg(self, settings):
        self.labelDepthAvg.setText(str(settings["processing_settings"]["depthAvg_m"]))

    def setAlongTrackAvg(self, settings):
        self.labelAlongTrackAvg.setText(str(settings["processing_settings"]["alongTrackAvg_ping"]))

    def setDualSwathPolicy(self, settings):
        self.labelDualSwathPolicy.setText(str(settings["processing_settings"]["dualSwathPolicy"]))