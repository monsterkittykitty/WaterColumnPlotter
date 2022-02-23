# Lynette Davis
# ldavis@ccom.unh.edu
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# November 2021

# Description: Description: Toolbar class for WaterColumnPlotter MainWindow;
# initializes toolbar to display current IP, port, and buttons (play, pause, stop, settings).

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QGroupBox, QLabel, QSizePolicy, QStyle, QToolBar, QToolButton, QVBoxLayout, QWidget


class GUI_Toolbar(QToolBar):

    signalPlay = pyqtSignal(name="playClicked")
    signalPause = pyqtSignal(name="pauseClicked")
    signalStop = pyqtSignal(name="stopClicked")
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
        # self.toolButtonPlay.setToolTip("Begin listening / plotting data.")
        self.toolButtonPlay.setToolTip("Begins receiving, processing, and plotting data.")
        self.toolButtonPlay.setIcon(iconPlay)
        self.toolButtonPlay.setStyleSheet("background-color : rgb(240, 240, 240)")
        # Connect signals / slots
        self.toolButtonPlay.clicked.connect(self.playButtonClicked)

        iconPause = self.style().standardIcon(QStyle.SP_MediaPause)
        self.toolButtonPause = QToolButton(self)
        # self.toolButtonPause.setToolTip("Pause listening / plotting data.")
        self.toolButtonPause.setToolTip("Stops receiving data; continues processing and plotting queued data.")
        self.toolButtonPause.setIcon(iconPause)
        self.toolButtonPause.setStyleSheet("background-color : rgb(240, 240, 240)")
        # Initialize pause button as disabled:
        self.toolButtonPause.setDisabled(True)
        # Connect signals / slots
        self.toolButtonPause.clicked.connect(self.pauseButtonClicked)

        iconStop = self.style().standardIcon(QStyle.SP_MediaStop)
        self.toolButtonStop = QToolButton(self)
        self.toolButtonStop.setToolTip("Stops receiving, processing, and plotting data; discards queued data.")
        self.toolButtonStop.setIcon(iconStop)
        self.toolButtonStop.setStyleSheet("background-color : rgb(240, 240, 240)")
        # Initialize stop button as disabled:
        self.toolButtonStop.setDisabled(True)
        # Connect signals / slots
        self.toolButtonStop.clicked.connect(self.stopButtonClicked)

        # Add widgets to toolbar:
        self.addWidget(groupBoxIPPort)
        self.addWidget(groupBoxSystem)
        self.addWidget(self.toolButtonPlay)
        self.addWidget(self.toolButtonPause)
        self.addWidget(self.toolButtonStop)

        # Spacer Widget:
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.addWidget(spacer)

        # Add icon to toolbar that brings up settings dialog
        iconSettings = self.style().standardIcon(QStyle.SP_FileDialogDetailedView)
        self.toolButtonSettings = QToolButton(self)
        self.toolButtonSettings.setToolTip("Settings")
        self.toolButtonSettings.setIcon(iconSettings)
        self.toolButtonSettings.setStyleSheet("QToolButton {background-color : rgb(240, 240, 240)}"
                                              "QToolButton:pressed {background-color : rgb(158, 158, 158)}")
        # Connect signals / slots
        self.toolButtonSettings.clicked.connect(self.settingsClicked.emit)

        self.addWidget(self.toolButtonSettings)

    def setIPPort(self, ip, port):
        """
        Sets toolbar labels with current port and ip address.
        :param ip: Current IP address used for listening for incoming datagrams
        :param port: Current port used for listening for incoming datagrams
        """
        self.labelIPPort.setText(ip + ":" + str(port))

    def setSystem(self, system):
        """
        Sets toolbar labels with current sonar system.
        :param system: Current sonar system that is providing data
        """
        self.labelSystem.setText(system)

    def playButtonClicked(self):
        """
        Enables and disables appropriate buttons when play button is clicked;
        emits signal indicating play button clicked.
        """
        # Play button is pressed.
        # Disable play button
        self.toolButtonPlay.setDisabled(True)
        # self.toolBar.toolButtonPlay.setStyleSheet("background-color : rgb(158, 158, 158)")  # Grey
        self.toolButtonPlay.setStyleSheet("background-color : rgb(154, 171, 155)")  # Green

        # Enable pause button
        self.toolButtonPause.setEnabled(True)
        self.toolButtonPause.setStyleSheet("background-color : rgb(240, 240, 240)")

        # Enable stop button
        self.toolButtonStop.setEnabled(True)
        self.toolButtonStop.setStyleSheet("background-color : rgb(240, 240, 240)")

        self.playClicked.emit()

    def pauseButtonClicked(self):
        """
        Enables and disables appropriate buttons when pause button is clicked;
        emits signal indicating pause button clicked.
        """
        # Pause button is pressed.
        # Disable pause button
        self.toolButtonPause.setDisabled(True)
        # self.toolBar.toolButtonPause.setStyleSheet("background-color : rgb(158, 158, 158)")  # Grey
        self.toolButtonPause.setStyleSheet("background-color : rgb(212, 170, 65)")  # Yellow

        # Enable play button
        self.toolButtonPlay.setEnabled(True)
        self.toolButtonPlay.setStyleSheet("background-color : rgb(240, 240, 240)")

        self.pauseClicked.emit()

    def stopButtonClicked(self):
        """
        Enables and disables appropriate buttons when stop button is clicked;
        emits signal indicating stop button clicked.
        """
        # Stop button is pressed.
        # Disable pause button
        self.toolButtonPause.setDisabled(True)
        self.toolButtonPause.setStyleSheet("background-color : rgb(240, 240, 240)")
        # Disable stop button
        self.toolButtonStop.setDisabled(True)
        self.toolButtonStop.setStyleSheet("background-color : rgb(219, 141, 141)")

        # Enable play button
        self.toolButtonPlay.setEnabled(True)
        self.toolButtonPlay.setStyleSheet("background-color : rgb(240, 240, 240)")

        self.stopClicked.emit()
