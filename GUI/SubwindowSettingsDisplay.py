# Lynette Davis
# Center for Coastal and Ocean Mapping
# October 2021

from PyQt5.QtWidgets import QComboBox, QGridLayout, QLabel, QPushButton, QSpinBox, QStyle, QWidget
from PyQt5.QtCore import pyqtSignal, Qt
import pyqtgraph as pg
from qtrangeslider import QRangeSlider
import sys


# class SubwindowSettingsDisplay(QWidget):
#     def __init__(self, settings, parent=None):
#         super(SubwindowSettingsDisplay, self).__init__(parent)
#
#         self.settings = settings
#
#         self.setWindowTitle("Additional Settings")
#
#         self.resize(200, 200)
#
#         layout = QGridLayout()
#
#         labelAlongTrackAvg = QLabel("Along-Track Avg (pings):", parent=self)
#         layout.addWidget(labelAlongTrackAvg, 0, 0)
#
#         self.spinboxAlongTrackAvg = QSpinBox()
#         self.spinboxAlongTrackAvg.setToolTip("Number of pings to average along-track.")
#         self.spinboxAlongTrackAvg.setRange(1, 100)
#         self.spinboxAlongTrackAvg.setSingleStep(1)
#         self.setAlongTrackAvg(self.settings)
#         layout.addWidget(self.spinboxAlongTrackAvg, 0, 1)
#
#         labelDualSwath = QLabel("Dual Swath Policy:", parent=self)
#         layout.addWidget(labelDualSwath, 1, 0)
#
#         self.comboboxDualSwath = QComboBox()
#         self.comboboxDualSwath.addItems(["Keep all pings.", "Keep first ping only.", "Keep second ping only."])
#         self.setDualSwath(self.settings)
#         layout.addWidget(self.comboboxDualSwath, 1, 1)
#
#         # pushButtonApply = QPushButton("Apply")
#         iconApply = self.style().standardIcon(QStyle.SP_DialogApplyButton)
#         pushButtonApply = QPushButton()
#         pushButtonApply.setToolTip("Apply")
#         pushButtonApply.setIcon(iconApply)
#         #pushButtonApply.clicked.connect(self.acrossTrackAvgEditedFunction)
#         layout.addWidget(pushButtonApply, 2, 0)
#
#         # pushButtonCancel = QPushButton("Cancel")
#         iconCancel = self.style().standardIcon(QStyle.SP_DialogCancelButton)
#         pushButtonCancel = QPushButton()
#         pushButtonCancel.setToolTip("Cancel")
#         pushButtonCancel.setIcon(iconCancel)
#         #pushButtonCancel.clicked.connect(self.resetAcrossTrackAvg)
#         layout.addWidget(pushButtonCancel, 2, 1)
#
#         self.setLayout(layout)
#
#     def setAlongTrackAvg(self, settings):
#         self.spinboxAlongTrackAvg.setValue(settings["processing_settings"]["alongTrackAvg_ping"])
#
#     def setDualSwath(self, settings):
#         self.comboboxDualSwath.setCurrentIndex(settings["processing_settings"]["dualSwathPolicy"])


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
        self.labelDepth = QLabel(self)
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
        layout.addWidget(QLabel("Depth (m):"), 7, 0)
        layout.addWidget(self.labelDepth, 7, 1)
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
        self.setDepth(settings)
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

    def setDepth(self, settings):
        self.labelDepth.setText(str(settings["processing_settings"]["depth_m"]))

    def setDepthAvg(self, settings):
        self.labelDepthAvg.setText(str(settings["processing_settings"]["depthAvg_m"]))

    def setAlongTrackAvg(self, settings):
        self.labelAlongTrackAvg.setText(str(settings["processing_settings"]["alongTrackAvg_ping"]))

    def setDualSwathPolicy(self, settings):
        if settings["processing_settings"]["dualSwathPolicy"] == 0:
            self.labelDualSwathPolicy.setText("Keep all pings.")
        elif settings["processing_settings"]["dualSwathPolicy"] == 1:
            self.labelDualSwathPolicy.setText("Keep first ping only.")
        elif settings["processing_settings"]["dualSwathPolicy"] == 2:
            self.labelDualSwathPolicy.setText("Keep second ping only.")
        else:
            self.labelDualSwathPolicy.setText("")

        #self.labelDualSwathPolicy.setText(str(settings["processing_settings"]["dualSwathPolicy"]))
