from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import pyvista as pv
from pyvistaqt import BackgroundPlotter, QtInteractor
from qtrangeslider import QRangeSlider
import sys


class SubwindowHorizontalSliceWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(SubwindowHorizontalSliceWidget, self).__init__(parent)

        self.horizontal_plot = pg.ImageView(self)
        # Disable ROI button:
        self.horizontal_plot.ui.roiBtn.hide()
        # Disable Norm button:
        self.horizontal_plot.ui.menuBtn.hide()
        # Disable right-click context menu:
        self.horizontal_plot.view.setMenuEnabled(False)
        self.horizontal_plot.ui.histogram.item.vb.setMenuEnabled(False)
        #self.horizontal_plot.ui.histogram.item.vb.setRange(disableAutoRange=True)

        self.setWindowTitle("Horizontal Slice")

        layout = QtWidgets.QGridLayout()
        layout.setColumnMinimumWidth(0, 25)
        layout.setColumnMinimumWidth(1, 25)
        layout.setColumnMinimumWidth(2, 25)
        layout.setColumnMinimumWidth(3, 25)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 0)
        layout.setColumnStretch(2, 0)
        layout.setColumnStretch(3, 0)

        # layout.addWidget(QtWidgets.QPushButton("T"), 0, 0)
        labelDepth = QtWidgets.QLabel("Depth:")
        layout.addWidget(labelDepth, 0, 1)

        spinboxDepth = QtWidgets.QDoubleSpinBox()
        spinboxDepth.setDecimals(2)
        spinboxDepth.setRange(0, 1000)
        spinboxDepth.setSingleStep(0.5)
        spinboxDepth.setValue(10.00)
        layout.addWidget(spinboxDepth, 0, 2)

        pushButtonApply = QtWidgets.QPushButton("Apply")
        layout.addWidget(pushButtonApply, 0, 3)

        layout.addWidget(self.horizontal_plot, 1, 0, 3, 4)

        rangeSlider = QRangeSlider(Qt.Horizontal)
        layout.addWidget(rangeSlider, 4, 1, 1, 2)

        # layout.addWidget(QtWidgets.QPushButton("Bottom"))

        self.setLayout(layout)
