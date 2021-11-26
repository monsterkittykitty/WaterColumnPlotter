# Toolbar class for Water Column Plotter MainWindow.

# Lynette Davis
# Center for Coastal and Ocean Mapping
# November 2021

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QSpinBox, QStyle, QToolBar, QToolButton, QVBoxLayout, QWidget

class GUI_Toolbar(QToolBar):

    signalPlay = pyqtSignal(name="playClicked")
    signalStop = pyqtSignal(name="stopClicked")
    signalSettings = pyqtSignal(name="settingsClicked")

    def __init__(self, settings, parent=None):
        super(GUI_Toolbar, self).__init__(parent)

        self.settings = settings

        # IP:
        groupBoxIPPort = QGroupBox("Listening on: ", parent=self)
        # Set default value:
        self.labelIPPort = QLabel(settings['ip_settings']['ip'] + ":"
                                  + str(settings['ip_settings']['port']))  # This field needs to be editable!
        layoutIPPort = QVBoxLayout()
        layoutIPPort.addWidget(self.labelIPPort)
        groupBoxIPPort.setLayout(layoutIPPort)

        # System:
        groupBoxSystem = QGroupBox("System: ", parent=self)
        # Set default value:
        self.labelSystem = QLabel(settings['system_settings']['system'])  # This field needs to be editable!
        layoutSystem = QVBoxLayout()
        layoutSystem.addWidget(self.labelSystem)
        groupBoxSystem.setLayout(layoutSystem)

        # Play / Stop Icons:
        iconPlay = self.style().standardIcon(QStyle.SP_MediaPlay)
        self.toolButtonPlay = QToolButton(self)
        self.toolButtonPlay.setToolTip("Begin listening / plotting data.")
        self.toolButtonPlay.setIcon(iconPlay)
        self.toolButtonPlay.setStyleSheet("background-color : rgb(240, 240, 240)")
        self.toolButtonPlay.clicked.connect(self.playClicked.emit)

        iconStop = self.style().standardIcon(QStyle.SP_MediaStop)
        self.toolButtonStop = QToolButton(self)
        self.toolButtonStop.setToolTip("Stop listening / plotting data.")
        self.toolButtonStop.setIcon(iconStop)
        self.toolButtonStop.setStyleSheet("background-color : rgb(240, 240, 240)")
        # Initialize stop button as disabled:
        self.toolButtonStop.setDisabled(True)
        self.toolButtonStop.clicked.connect(self.stopClicked.emit)

        # Add widgets to toolbar:
        self.addWidget(groupBoxIPPort)
        self.addWidget(groupBoxSystem)
        self.addWidget(self.toolButtonPlay)
        self.addWidget(self.toolButtonStop)

        # For adding additional settngs to toolbar. Note: This is incomplete and signals / slots are not enabled.
        # # Settings:
        # groupBoxSettings = QGroupBox("Additional Settings: ", parent=self)
        #
        # layoutSettings = QGridLayout()
        #
        # labelAlongTrackAvg = QLabel("Along-Track Avg (pings):", parent=self)
        # layoutSettings.addWidget(labelAlongTrackAvg, 0, 0)
        #
        # self.spinboxAlongTrackAvg = QSpinBox()
        # self.spinboxAlongTrackAvg.setToolTip("Number of pings to average along-track.")
        # self.spinboxAlongTrackAvg.setRange(1, 100)
        # self.spinboxAlongTrackAvg.setSingleStep(1)
        # self.setAlongTrackAvg(self.settings)
        # layoutSettings.addWidget(self.spinboxAlongTrackAvg, 0, 1)
        #
        # layoutSettings.setColumnMinimumWidth(2, 5)
        #
        # labelDualSwath = QLabel("Dual Swath Policy:", parent=self)
        # layoutSettings.addWidget(labelDualSwath, 0, 3)
        #
        # self.comboboxDualSwath = QComboBox()
        # self.comboboxDualSwath.addItems(["Keep all pings.", "Keep first ping only.", "Keep second ping only."])
        # self.setDualSwath(self.settings)
        # layoutSettings.addWidget(self.comboboxDualSwath, 0, 4)
        #
        # layoutSettings.setColumnMinimumWidth(5, 5)
        #
        # # pushButtonApply = QPushButton("Apply")
        # iconApply = self.style().standardIcon(QStyle.SP_DialogApplyButton)
        # pushButtonApply = QPushButton()
        # pushButtonApply.setToolTip("Apply")
        # pushButtonApply.setIcon(iconApply)
        # # pushButtonApply.clicked.connect(self.acrossTrackAvgEditedFunction)
        # layoutSettings.addWidget(pushButtonApply, 0, 6)
        #
        # # pushButtonCancel = QPushButton("Cancel")
        # iconCancel = self.style().standardIcon(QStyle.SP_DialogCancelButton)
        # pushButtonCancel = QPushButton()
        # pushButtonCancel.setToolTip("Cancel")
        # pushButtonCancel.setIcon(iconCancel)
        # # pushButtonCancel.clicked.connect(self.resetAcrossTrackAvg)
        # layoutSettings.addWidget(pushButtonCancel, 0, 7)
        #
        # groupBoxSettings.setLayout(layoutSettings)
        #
        # # Spacer Widget:
        # spacer = QWidget()
        # spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # self.addWidget(spacer)
        #
        # self.addWidget(groupBoxSettings)

        # Instead of above, add icon to toolbar that brings up settings dialog
        iconSettings = self.style().standardIcon(QStyle.SP_FileDialogDetailedView)
        self.toolButtonSettings = QToolButton(self)
        self.toolButtonSettings.setToolTip("Settings")
        self.toolButtonSettings.setIcon(iconSettings)
        self.toolButtonSettings.setStyleSheet("QToolButton {background-color : rgb(240, 240, 240)}"
                                              "QToolButton:pressed {background-color : rgb(158, 158, 158)}")
        self.toolButtonSettings.clicked.connect(self.settingsClicked.emit)

        # Spacer Widget:
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.addWidget(spacer)

        self.addWidget(self.toolButtonSettings)

    def setIPPort(self, ip, port):
        self.labelIPPort.setText(ip + ":" + str(port))

    def setSystem(self, system):
        self.labelSystem.setText(system)

    # def setAlongTrackAvg(self, settings):
    #     self.spinboxAlongTrackAvg.setValue(settings['processing_settings']['alongTrackAvg_ping'])
    #
    # def setDualSwath(self, settings):
    #     self.comboboxDualSwath.setCurrentIndex(settings['processing_settings']['dualSwathPolicy'])
