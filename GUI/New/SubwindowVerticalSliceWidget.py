from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from qtrangeslider import QRangeSlider
import sys


class SubwindowVerticalSliceWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(SubwindowVerticalSliceWidget, self).__init__(parent)

        self.vertical_plot = pg.ImageView(self)
        # Disable ROI button:
        self.vertical_plot.ui.roiBtn.hide()
        # Disable Norm button:
        self.vertical_plot.ui.menuBtn.hide()
        # Disable right-click context menu:
        self.vertical_plot.view.setMenuEnabled(False)
        self.vertical_plot.ui.histogram.item.vb.setMenuEnabled(False)
        #self.vertical_plot.ui.histogram.item.vb.setRange(disableAutoRange=True)

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
        layout.addWidget(self.vertical_plot, 1, 0, 3, 3)
        #layout.addWidget(pg.PlotWidget())
        layout.addWidget(QRangeSlider(Qt.Horizontal), 4, 1, 1, 2)
        # layout.addWidget(QtWidgets.QPushButton("Bottom"))

        self.setLayout(layout)
