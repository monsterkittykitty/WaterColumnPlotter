# Lynette Davis
# Center for Coastal and Ocean Mapping
# October 2021

import numpy as np
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QDoubleSpinBox, QGridLayout, QLabel, QPushButton, QStyle, QWidget
import pyqtgraph as pg
# import pyvista as pv
# from pyvistaqt import BackgroundPlotter, QtInteractor
# from qtrangeslider import QRangeSlider
import sys


class SubwindowHorizontalSliceWidget(QWidget):

    signalDepthEdited = pyqtSignal(name="depthEdited")
    signalDepthAvgEdited = pyqtSignal(name="depthAvgEdited")

    def __init__(self, settings, parent=None):
        super(SubwindowHorizontalSliceWidget, self).__init__(parent)

        self.settings = settings

        self.setWindowTitle("Horizontal Slice")

        self.plot = GUI_PlotItem(self.settings)
        # self.plot = pg.PlotItem()
        # self.plot.hideButtons()
        self.plot.vb.state['aspectLocked'] = False
        self.plot.setXRange(-(self.settings['buffer_settings']['maxBufferSize'] /
                              self.settings['processing_settings']['alongTrackAvg_ping']), 0)
        self.plot.setYRange(self.settings['buffer_settings']['maxGridCells'], 0)

        # Adding axis labels based on:
        # https://stackoverflow.com/questions/58516639/add-axes-labels-and-title-to-pyqtgraph-imageview
        self.plot.setLabel(axis='left', text='Across-Track')
        self.plot.setLabel(axis='bottom', text='Pings')

        self.horizontal_plot = pg.ImageView(self, view=self.plot)

        self.horizontal_plot.ui.histogram.setLevels(min=-95, max=35)

        # Disable ROI button:
        self.horizontal_plot.ui.roiBtn.hide()
        # Disable Norm button:
        self.horizontal_plot.ui.menuBtn.hide()
        # Disable right-click context menu:
        self.horizontal_plot.view.setMenuEnabled(False)
        self.horizontal_plot.ui.histogram.item.vb.setMenuEnabled(False)
        #self.horizontal_plot.ui.histogram.item.vb.setRange(disableAutoRange=True)

        # Initialize plot with blank matrix of max size:
        # temp_image = np.empty((self.settings['buffer_settings']['maxGridCells'],
        #                       self.settings['buffer_settings']['maxBufferSize'], ))
        # temp_image.fill(np.nan)
        # temp_image = np.zeros((self.settings['buffer_settings']['maxBufferSize'],
        #                        self.settings['buffer_settings']['maxGridCells'],))
        # autoRange = Auto scale and pan the view around the image such that the image fills the view.
        # autoLevels = Set the min/max intensity levels automatically to match the image data.
        # autoHistogramRange = If True, the histogram y-range is automatically scaled to fit the image data.
        # self.horizontal_plot.setImage(temp_image, autoRange=False, autoLevels=False, levels=(-95, 35), autoHistogramRange=False)

        layout = QGridLayout()
        layout.setColumnMinimumWidth(0, 25)
        layout.setColumnMinimumWidth(1, 25)
        layout.setColumnMinimumWidth(2, 25)
        layout.setColumnMinimumWidth(3, 25)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 0)
        layout.setColumnStretch(2, 0)
        layout.setColumnStretch(3, 0)

        # layout.addWidget(QtWidgets.QPushButton("T"), 0, 0)
        labelDepth = QLabel("Depth (m):")
        layout.addWidget(labelDepth, 0, 1)

        self.spinboxDepth = QDoubleSpinBox()
        self.spinboxDepth.setToolTip("Depth of horizontal slice.")
        self.spinboxDepth.setDecimals(2)
        self.spinboxDepth.setRange(1, 1000)
        self.spinboxDepth.setSingleStep(0.5)
        self.setDepth(self.settings['processing_settings']['depth_m'])
        layout.addWidget(self.spinboxDepth, 0, 2)

        layout.setColumnMinimumWidth(3, 5)

        labelDepthAvg = QLabel("Depth Average (m):")
        layout.addWidget(labelDepthAvg, 0, 4)

        self.spinboxDepthAvg = QDoubleSpinBox()
        self.spinboxDepthAvg.setToolTip("Depth to average.")
        self.spinboxDepthAvg.setDecimals(2)
        self.spinboxDepthAvg.setRange(1, 100)
        self.spinboxDepthAvg.setSingleStep(0.5)
        self.setDepthAvg(self.settings['processing_settings']['depthAvg_m'])
        layout.addWidget(self.spinboxDepthAvg, 0, 5)

        layout.setColumnMinimumWidth(6, 5)

        # pushButtonApply = QPushButton("Apply")
        iconApply = self.style().standardIcon(QStyle.SP_DialogApplyButton)
        pushButtonApply = QPushButton()
        pushButtonApply.setToolTip("Apply")
        pushButtonApply.setIcon(iconApply)
        pushButtonApply.clicked.connect(self.editedAllFunction)
        layout.addWidget(pushButtonApply, 0, 7)

        # pushButtonCancel = QPushButton("Cancel")
        iconCancel = self.style().standardIcon(QStyle.SP_DialogCancelButton)
        pushButtonCancel = QPushButton()
        pushButtonCancel.setToolTip("Cancel")
        pushButtonCancel.setIcon(iconCancel)
        pushButtonCancel.clicked.connect(self.resetAll)
        layout.addWidget(pushButtonCancel, 0, 8)

        layout.addWidget(self.horizontal_plot, 1, 0, 3, 9)

        # rangeSlider = QRangeSlider(Qt.Horizontal)
        # layout.addWidget(rangeSlider, 4, 1, 1, 2)

        # layout.addWidget(QtWidgets.QPushButton("Bottom"))

        self.setLayout(layout)

    def setDepth(self, depth):
        self.spinboxDepth.setValue(depth)

    def setDepthAvg(self, depthAvg):
        self.spinboxDepthAvg.setValue(depthAvg)

    def resetAll(self):
        self.resetDepth()
        self.resetDepthAvg()

    def resetDepth(self):
        self.spinboxDepth.setValue(self.settings['processing_settings']['depth_m'])

    def resetDepthAvg(self):
        self.spinboxDepthAvg.setValue(self.settings['processing_settings']['depthAvg_m'])

    def editedAllFunction(self):
        self.depthEditedFunction()
        self.depthAvgEditedFunction()

    def depthEditedFunction(self):
        if self.settings['processing_settings']['depth_m'] != self.spinboxDepth.value():
            self.settings['processing_settings']['depth_m'] = round(self.spinboxDepth.value(), 2)
            self.depthEdited.emit()

    def depthAvgEditedFunction(self):
        if self.settings['processing_settings']['depthAvg_m'] != self.spinboxDepthAvg.value():
            self.settings['processing_settings']['depthAvg_m'] = round(self.spinboxDepthAvg.value(), 2)
            self.depthAvgEdited.emit()


class GUI_PlotItem(pg.PlotItem):
    def __init__(self, settings, parent=None):
        super(GUI_PlotItem, self).__init__(parent)

        self.settings = settings

    def autoBtnClicked(self):
        """
        Overrides method in pyqtgraph.PlotItem. Instead of enabling autoRange when clicked,
        this button will now return image to default home view.
        """
        # This ensures that full, specified x, y range will be displayed, but 1:1 aspect ratio may not be maintained.
        self.vb.state['aspectLocked'] = False

        # self.setXRange(-(self.settings['buffer_settings']['maxBufferSize'] /
        #                  self.settings['processing_settings']['alongTrackAvg_ping']), 0)
        # self.setYRange(self.settings['buffer_settings']['maxGridCells'], 0)

        self.enableAutoRange()
        self.setXRange(-(self.settings['buffer_settings']['maxBufferSize'] /
                         self.settings['processing_settings']['alongTrackAvg_ping']), 0)
        self.autoBtn.hide()
