from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from qtrangeslider import QRangeSlider
import sys


class SubwindowVerticalSliceWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(SubwindowVerticalSliceWidget, self).__init__(parent)
        self.setWindowTitle("Vertical Slice")

        #self.resize(400, 200)

        #layout = QtWidgets.QVBoxLayout()
        layout = QtWidgets.QGridLayout()
        layout.setColumnMinimumWidth(0, 25)
        layout.setColumnMinimumWidth(1, 25)
        layout.setColumnMinimumWidth(2, 25)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 0)
        layout.setColumnStretch(2, 0)

        layout.addWidget(QtWidgets.QPushButton("T"), 0, 0)
        layout.addWidget(QtWidgets.QPushButton("To"), 0, 1)
        layout.addWidget(QtWidgets.QPushButton("Top"), 0, 2)
        layout.addWidget(pg.PlotWidget(), 1, 0, 3, 3)
        #layout.addWidget(pg.PlotWidget())
        layout.addWidget(QRangeSlider(Qt.Horizontal), 4, 1, 1, 2)
        # layout.addWidget(QtWidgets.QPushButton("Bottom"))

        self.setLayout(layout)
