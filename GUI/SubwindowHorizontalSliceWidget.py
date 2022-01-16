# Lynette Davis
# Center for Coastal and Ocean Mapping
# October 2021

import datetime
import math
import numpy as np
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QStyle, QVBoxLayout, QWidget
import pyqtgraph as pg
# import pyvista as pv
# from pyvistaqt import BackgroundPlotter, QtInteractor
# from qtrangeslider import QRangeSlider
import sys


class SubwindowHorizontalSliceWidget(QWidget):

    signalDepthEdited = pyqtSignal(name="depthEdited")
    signalDepthAvgEdited = pyqtSignal(name="depthAvgEdited")
    signalProcessingSettingsEdited = pyqtSignal(name="processingSettingsEdited")

    def __init__(self, settings, shared_ring_buffer_processed, parent=None):
        super(SubwindowHorizontalSliceWidget, self).__init__(parent)

        self.settings = settings
        self.shared_ring_buffer_processed = shared_ring_buffer_processed

        # Mouse position
        self.pos_x = None
        self.pos_y = None

        self.setWindowTitle("Horizontal Slice")

        self.plot = GUI_PlotItem(self.settings)
        # self.plot = pg.PlotItem()
        # self.plot.hideButtons()
        self.plot.vb.state['aspectLocked'] = False
        self.plot.setXRange(-(self.settings['buffer_settings']['maxBufferSize_ping'] /
                              self.settings['processing_settings']['alongTrackAvg_ping']), 0)
        self.plot.setYRange(self.settings['buffer_settings']['maxGridCells'], 0)

        # Adding axis labels based on:
        # https://stackoverflow.com/questions/58516639/add-axes-labels-and-title-to-pyqtgraph-imageview
        self.plot.setLabel(axis='left', text='Across-Track')
        self.plot.setLabel(axis='bottom', text='Pings')

        # Crosshair
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.plot.addItem(self.vLine, ignoreBounds=True)
        self.plot.addItem(self.hLine, ignoreBounds=True)

        # ImageView
        self.horizontal_plot = pg.ImageView(self, view=self.plot)

        self.horizontal_plot.ui.histogram.setLevels(min=-95, max=35)
        # Based on: https://stackoverflow.com/questions/38021869/getting-imageitem-values-from-pyqtgraph
        self.horizontal_plot.scene.sigMouseMoved.connect(self.mouseMoved)

        # Disable ROI button:
        self.horizontal_plot.ui.roiBtn.hide()
        # Disable Norm button:
        self.horizontal_plot.ui.menuBtn.hide()
        # Disable right-click context menu:
        self.horizontal_plot.view.setMenuEnabled(False)
        self.horizontal_plot.ui.histogram.item.vb.setMenuEnabled(False)
        #self.horizontal_plot.ui.histogram.item.vb.setRange(disableAutoRange=True)

        # OVERALL LAYOUT:
        layout = QVBoxLayout()

        # TOP ROW:
        top_row_layout = QHBoxLayout()

        # Spacer Widget:
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        top_row_layout.addWidget(spacer)

        labelDepth = QLabel("Depth (m):")
        top_row_layout.addWidget(labelDepth)

        self.spinboxDepth = QDoubleSpinBox()
        self.spinboxDepth.setToolTip("Depth of horizontal slice.")
        self.spinboxDepth.setDecimals(2)
        self.spinboxDepth.setRange(1, 1000)
        self.spinboxDepth.setSingleStep(0.5)
        self.setDepth(self.settings['processing_settings']['depth_m'])
        top_row_layout.addWidget(self.spinboxDepth)

        labelDepthAvg = QLabel("Depth Average (m):")
        top_row_layout.addWidget(labelDepthAvg)

        self.spinboxDepthAvg = QDoubleSpinBox()
        self.spinboxDepthAvg.setToolTip("Depth to average.")
        self.spinboxDepthAvg.setDecimals(2)
        self.spinboxDepthAvg.setRange(1, 100)
        self.spinboxDepthAvg.setSingleStep(0.5)
        self.setDepthAvg(self.settings['processing_settings']['depthAvg_m'])
        top_row_layout.addWidget(self.spinboxDepthAvg)

        # pushButtonApply = QPushButton("Apply")
        iconApply = self.style().standardIcon(QStyle.SP_DialogApplyButton)
        pushButtonApply = QPushButton()
        pushButtonApply.setToolTip("Apply")
        pushButtonApply.setIcon(iconApply)
        pushButtonApply.clicked.connect(self.editedAllFunction)
        top_row_layout.addWidget(pushButtonApply)

        # pushButtonCancel = QPushButton("Cancel")
        iconCancel = self.style().standardIcon(QStyle.SP_DialogCancelButton)
        pushButtonCancel = QPushButton()
        pushButtonCancel.setToolTip("Cancel")
        pushButtonCancel.setIcon(iconCancel)
        pushButtonCancel.clicked.connect(self.resetAll)
        top_row_layout.addWidget(pushButtonCancel)

        # BOTTOM ROW
        bottom_row_layout = QHBoxLayout()

        # Time Label
        labelMousePosTime = QLabel("Time: ", parent=self)
        bottom_row_layout.addWidget(labelMousePosTime)
        self.labelMousePosTimeValue = QLabel("nan", parent=self)
        bottom_row_layout.addWidget(self.labelMousePosTimeValue)

        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        bottom_row_layout.addWidget(line)

        # Ping Label
        labelMousePosPing = QLabel("Ping:", parent=self)
        bottom_row_layout.addWidget(labelMousePosPing)
        self.labelMousePosPingValue = QLabel("nan", parent=self)
        bottom_row_layout.addWidget(self.labelMousePosPingValue)

        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        bottom_row_layout.addWidget(line)

        # Across-track Label
        labelMousePosAcrossTrack = QLabel("Across-Track (m):", parent=self)
        bottom_row_layout.addWidget(labelMousePosAcrossTrack)
        self.labelMousePosAcrossTrackValue = QLabel("nan", parent=self)
        bottom_row_layout.addWidget(self.labelMousePosAcrossTrackValue)

        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        bottom_row_layout.addWidget(line)

        # Intensity Label
        labelMousePosIntensity = QLabel("Intensity (dB):", parent=self)
        bottom_row_layout.addWidget(labelMousePosIntensity)
        self.labelMousePosIntensityValue = QLabel("nan", parent=self)
        bottom_row_layout.addWidget(self.labelMousePosIntensityValue)

        # Spacer Widget:
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # layout.addWidget(spacer, 0, 0)
        bottom_row_layout.addWidget(spacer)

        layout.addLayout(top_row_layout)
        layout.addWidget(self.horizontal_plot)
        layout.addLayout(bottom_row_layout)

        # rangeSlider = QRangeSlider(Qt.Horizontal)
        # layout.addWidget(rangeSlider, 4, 1, 1, 2)

        # layout.addWidget(QtWidgets.QPushButton("Bottom"))

        self.setLayout(layout)


        # TODO: Old implementation (before crosshair with ping and depth display).
        # layout = QGridLayout()
        # layout.setColumnMinimumWidth(0, 25)
        # layout.setColumnMinimumWidth(1, 25)
        # layout.setColumnMinimumWidth(2, 25)
        # layout.setColumnMinimumWidth(3, 25)
        # layout.setColumnStretch(0, 1)
        # layout.setColumnStretch(1, 0)
        # layout.setColumnStretch(2, 0)
        # layout.setColumnStretch(3, 0)
        #
        # # layout.addWidget(QtWidgets.QPushButton("T"), 0, 0)
        # labelDepth = QLabel("Depth (m):")
        # layout.addWidget(labelDepth, 0, 1)
        #
        # self.spinboxDepth = QDoubleSpinBox()
        # self.spinboxDepth.setToolTip("Depth of horizontal slice.")
        # self.spinboxDepth.setDecimals(2)
        # self.spinboxDepth.setRange(1, 1000)
        # self.spinboxDepth.setSingleStep(0.5)
        # self.setDepth(self.settings['processing_settings']['depth_m'])
        # layout.addWidget(self.spinboxDepth, 0, 2)
        #
        # layout.setColumnMinimumWidth(3, 5)
        #
        # labelDepthAvg = QLabel("Depth Average (m):")
        # layout.addWidget(labelDepthAvg, 0, 4)
        #
        # self.spinboxDepthAvg = QDoubleSpinBox()
        # self.spinboxDepthAvg.setToolTip("Depth to average.")
        # self.spinboxDepthAvg.setDecimals(2)
        # self.spinboxDepthAvg.setRange(1, 100)
        # self.spinboxDepthAvg.setSingleStep(0.5)
        # self.setDepthAvg(self.settings['processing_settings']['depthAvg_m'])
        # layout.addWidget(self.spinboxDepthAvg, 0, 5)
        #
        # layout.setColumnMinimumWidth(6, 5)
        #
        # # pushButtonApply = QPushButton("Apply")
        # iconApply = self.style().standardIcon(QStyle.SP_DialogApplyButton)
        # pushButtonApply = QPushButton()
        # pushButtonApply.setToolTip("Apply")
        # pushButtonApply.setIcon(iconApply)
        # pushButtonApply.clicked.connect(self.editedAllFunction)
        # layout.addWidget(pushButtonApply, 0, 7)
        #
        # # pushButtonCancel = QPushButton("Cancel")
        # iconCancel = self.style().standardIcon(QStyle.SP_DialogCancelButton)
        # pushButtonCancel = QPushButton()
        # pushButtonCancel.setToolTip("Cancel")
        # pushButtonCancel.setIcon(iconCancel)
        # pushButtonCancel.clicked.connect(self.resetAll)
        # layout.addWidget(pushButtonCancel, 0, 8)
        #
        # layout.addWidget(self.horizontal_plot, 1, 0, 3, 9)
        #
        # # rangeSlider = QRangeSlider(Qt.Horizontal)
        # # layout.addWidget(rangeSlider, 4, 1, 1, 2)
        #
        # # layout.addWidget(QtWidgets.QPushButton("Bottom"))
        #
        # self.setLayout(layout)

    def mouseMoved(self, pos):
        try:
            position = self.horizontal_plot.getImageItem().mapFromScene(pos)

            self.pos_x = position.x()
            self.pos_y = position.y()

            # Correct for shifted position of image
            x = position.x() - self.horizontal_plot.image.shape[0]
            y = position.y() - (self.horizontal_plot.image.shape[1] / 2)

            self.vLine.setPos(x)
            self.hLine.setPos(y)

            if 0 <= round(position.x()) < self.horizontal_plot.image.shape[0] and \
                    0 <= round(position.y()) < self.horizontal_plot.image.shape[1]:
                intensity = self.horizontal_plot.image[round(position.x())][round(position.y())]
            else:
                intensity = float('nan')

            if math.isnan(intensity):
                x = y = float('nan')

            self.setMousePositionLabels(x, y, intensity)

        except AttributeError:  # Triggered when nothing is plotted
            pass

    def setMousePositionLabels(self, ping, across_track, intensity):
        timestamp = float('nan')

        if not math.isnan(intensity):
            timestamp_epoch_sec = self.shared_ring_buffer_processed.view_buffer_elements(
                self.shared_ring_buffer_processed.timestamp_buffer_avg)[round(self.pos_x)]

            timestamp = datetime.datetime.utcfromtimestamp(timestamp_epoch_sec).time()
            ping = round(ping * self.settings['processing_settings']['alongTrackAvg_ping'])
            across_track = round((across_track * self.settings['processing_settings']['binSize_m']), 2)
            intensity = round(intensity, 2)

        self.labelMousePosTimeValue.setText(str(timestamp))
        self.labelMousePosPingValue.setText(str(ping))
        self.labelMousePosAcrossTrackValue.setText(str(across_track))
        self.labelMousePosIntensityValue.setText(str(intensity))

    def getMousePosition(self):
        return self.pos_x, self.pos_y

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
        self.processingSettingsEdited.emit()

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
        self.setXRange(-(self.settings['buffer_settings']['maxBufferSize_ping'] /
                         self.settings['processing_settings']['alongTrackAvg_ping']), 0)
        self.autoBtn.hide()
