# Toolbar class for Water Column Plotter MainWindow.

# Lynette Davis
# Center for Coastal and Ocean Mapping
# November 2021

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QStatusBar, QGridLayout, QGroupBox, QLabel, QSizePolicy, QStyle, QToolBar, QToolButton, QVBoxLayout, QWidget

class GUI_StatusBar(QStatusBar):

    def __init__(self, parent=None):
        super(GUI_StatusBar, self).__init__(parent)

        # Spacer Widget:
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        #self.addWidget(spacer)

        layoutPings = QGridLayout()

        labelRxToLost = QLabel("WC Pings, Rxed:Lost", parent=self)
        #layoutPings.addWidget(labelRxToLost, 0, 0)
        self.labelRxToLostValues = QLabel("0:0", parent=self)
        #layoutPings.addWidget(self.labelRxToLostValues, 0, 1)

        self.addPermanentWidget(labelRxToLost)
        self.addPermanentWidget(self.labelRxToLostValues)

    def set_ping_counts(self, full_count, discard_count):
        self.labelRxToLostValues.setText(str(full_count) + ":" + str(discard_count))