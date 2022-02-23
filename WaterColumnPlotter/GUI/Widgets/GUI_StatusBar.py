# Lynette Davis
# ldavis@ccom.unh.edu
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# November 2021

# Description: Status Bar class for WaterColumnPlotter MainWindow;
# initializes status bar to display number of received and lost pings.

from PyQt5.QtWidgets import QStatusBar, QGridLayout, QLabel, QSizePolicy, QWidget


class GUI_StatusBar(QStatusBar):

    def __init__(self, parent=None):
        super(GUI_StatusBar, self).__init__(parent)

        # Spacer Widget:
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layoutPings = QGridLayout()

        labelRxToLost = QLabel("WC Pings, Rxed:Lost", parent=self)
        self.labelRxToLostValues = QLabel("0:0", parent=self)

        self.addPermanentWidget(labelRxToLost)
        self.addPermanentWidget(self.labelRxToLostValues)

    def set_ping_counts(self, full_count, discard_count):
        """
        Sets status bar labels with number of received (full_count) and lost (discard_count) pings.
        :param full_count: Integer indicating number of successfully reconstructed pings (from partitioned datagrams)
        :param discard_count: Integer indicating number of unsuccessfully reconstructed pings
        (from partitioned datagrams)
        """
        self.labelRxToLostValues.setText(str(full_count) + ":" + str(discard_count))
