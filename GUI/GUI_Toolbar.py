# Toolbar class for Water Column Plotter MainWindow.

# Lynette Davis
# Center for Coastal and Ocean Mapping
# November 2021

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QGridLayout, QGroupBox, QLabel, QSizePolicy, QStyle, QToolBar, QToolButton, QVBoxLayout, QWidget

class GUI_Toolbar(QToolBar):

    signalPlay = pyqtSignal(name="playClicked")
    signalPause = pyqtSignal(name="pauseClicked")
    signalSettings = pyqtSignal(name="settingsClicked")

    def __init__(self, settings, parent=None):
        super(GUI_Toolbar, self).__init__(parent)

        self.settings = settings

        # IP:
        groupBoxIPPort = QGroupBox("Listening on: ", parent=self)
        # Set default value:
        self.labelIPPort = QLabel(settings['ip_settings']['ip'] + ":" + str(settings['ip_settings']['port']))
        layoutIPPort = QVBoxLayout()
        layoutIPPort.addWidget(self.labelIPPort)
        groupBoxIPPort.setLayout(layoutIPPort)

        # System:
        groupBoxSystem = QGroupBox("System: ", parent=self)
        # Set default value:
        self.labelSystem = QLabel(settings['system_settings']['system'])
        layoutSystem = QVBoxLayout()
        layoutSystem.addWidget(self.labelSystem)
        groupBoxSystem.setLayout(layoutSystem)

        # Play / Pause Icons:
        iconPlay = self.style().standardIcon(QStyle.SP_MediaPlay)
        self.toolButtonPlay = QToolButton(self)
        self.toolButtonPlay.setToolTip("Begin listening / plotting data.")
        self.toolButtonPlay.setIcon(iconPlay)
        self.toolButtonPlay.setStyleSheet("background-color : rgb(240, 240, 240)")
        # Connect signals / slots
        self.toolButtonPlay.clicked.connect(self.playButtonClicked)

        iconPause = self.style().standardIcon(QStyle.SP_MediaPause)
        self.toolButtonPause = QToolButton(self)
        self.toolButtonPause.setToolTip("Pause listening / plotting data.")
        self.toolButtonPause.setIcon(iconPause)
        self.toolButtonPause.setStyleSheet("background-color : rgb(240, 240, 240)")
        # Initialize pause button as disabled:
        self.toolButtonPause.setDisabled(True)
        # Connect signals / slots
        self.toolButtonPause.clicked.connect(self.pauseButtonClicked)

        # Add widgets to toolbar:
        self.addWidget(groupBoxIPPort)
        self.addWidget(groupBoxSystem)
        self.addWidget(self.toolButtonPlay)
        self.addWidget(self.toolButtonPause)

        # Spacer Widget:
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.addWidget(spacer)

        # Create display mechanism for rxed to lost pings
        # groupBoxPings = QGroupBox("Pings: ", parent=self)
        # layoutPings = QGridLayout()
        #
        # labelRxToLost = QLabel("Rxed:Lost", parent=self)
        # layoutPings.addWidget(labelRxToLost, 0, 0)
        # self.labelRxToLostValues = QLabel("0:0", parent=self)
        # layoutPings.addWidget(self.labelRxToLostValues, 0, 1)
        #
        # groupBoxPings.setLayout(layoutPings)
        #
        # self.addWidget(groupBoxPings)

        # For adding additional settings to toolbar. Note: This is incomplete and signals / slots are not enabled.
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
        # Connect signals / slots
        self.toolButtonSettings.clicked.connect(self.settingsClicked.emit)

        # # Spacer Widget:
        # spacer = QWidget()
        # spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # self.addWidget(spacer)

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

    def playButtonClicked(self):
        # Play button is pressed.
        # Disable play button
        self.toolButtonPlay.setDisabled(True)
        # self.toolBar.toolButtonPlay.setStyleSheet("background-color : rgb(158, 158, 158)")  # Grey
        self.toolButtonPlay.setStyleSheet("background-color : rgb(154, 171, 155)")  # Green

        # Enable pause button
        self.toolButtonPause.setEnabled(True)
        self.toolButtonPause.setStyleSheet("background-color : rgb(240, 240, 240)")

        self.playClicked.emit()

    def pauseButtonClicked(self):
        # Pause button is pressed.
        # Disable pause button
        self.toolButtonPause.setDisabled(True)
        # self.toolBar.toolButtonPause.setStyleSheet("background-color : rgb(158, 158, 158)")  # Grey
        # self.toolButtonPause.setStyleSheet("background-color : rgb(219, 141, 141)")  # Red
        # self.toolButtonPause.setStyleSheet("background-color : rgb(225, 161, 53)")  # Yellow
        self.toolButtonPause.setStyleSheet("background-color : rgb(212, 170, 65)")  # Yellow

        # Enable play button
        self.toolButtonPlay.setEnabled(True)
        self.toolButtonPlay.setStyleSheet("background-color : rgb(240, 240, 240)")

        self.pauseClicked.emit()
