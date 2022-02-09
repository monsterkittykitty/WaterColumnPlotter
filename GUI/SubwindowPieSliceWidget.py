# Lynette Davis
# Center for Coastal and Ocean Mapping
# October 2021

import datetime
import math
import numpy as np
from PyQt5.QtWidgets import QDoubleSpinBox, QFrame, QGraphicsSceneMouseEvent, QGridLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QStyle, QVBoxLayout, QWidget
from PyQt5.QtCore import pyqtSignal, Qt
import pyqtgraph as pg
# from qtrangeslider import QRangeSlider
import sys

class SubwindowPieSliceWidget(QWidget):

    signalbinSizeEdited = pyqtSignal(name="binSizeEdited")
    signalProcessingSettingsEdited = pyqtSignal(name="processingSettingsEdited")

    def __init__(self, settings, shared_ring_buffer_processed, parent=None):
        super(SubwindowPieSliceWidget, self).__init__(parent)

        self.settings = settings
        self.shared_ring_buffer_processed = shared_ring_buffer_processed

        # Mouse position
        self.matrix_x = None
        self.matrix_y = None
        self.intensity = None
        self.plot_x = None
        self.plot_y = None

        self.setWindowTitle("Pie Slice")

        self.plot = GUI_PlotItem(self.settings)

        self.plot.vb.state['aspectLocked'] = False
        self.plot.setXRange(-(self.settings['buffer_settings']['maxGridCells'] / 2),
                            (self.settings['buffer_settings']['maxGridCells'] / 2))
        self.plot.setYRange(self.settings['buffer_settings']['maxGridCells'], 0)

        # Adding axis labels based on:
        # https://stackoverflow.com/questions/58516639/add-axes-labels-and-title-to-pyqtgraph-imageview
        self.plot.setLabel(axis='left', text='Depth')
        self.plot.setLabel(axis='bottom', text='Across-Track')

        # Crosshair
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.plot.addItem(self.vLine, ignoreBounds=True)
        self.plot.addItem(self.hLine, ignoreBounds=True)

        # ImageView
        self.pie_plot = pg.ImageView(self, view=self.plot)  # &&&  <class 'pyqtgraph.GraphicsScene.GraphicsScene.GraphicsScene'>

        self.depthIndicator = pg.InfiniteLine(angle=0, pen=pg.mkPen('m', width=1, style=Qt.DotLine), movable=False)
        self.depthIndicator.setPos(self.settings['processing_settings']['depth_m'] /
                                   self.settings['processing_settings']['binSize_m'])
        self.pie_plot.getView().addItem(self.depthIndicator)

        self.pie_plot.ui.histogram.setLevels(min=-95, max=35)
        # Based on https://stackoverflow.com/questions/38021869/getting-imageitem-values-from-pyqtgraph
        self.pie_plot.scene.sigMouseMoved.connect(self.mouseMoved)

        # Disable ROI button:
        self.pie_plot.ui.roiBtn.hide()
        # Disable Norm button:
        self.pie_plot.ui.menuBtn.hide()
        # Disable right-click context menu:
        self.pie_plot.view.setMenuEnabled(False)
        self.pie_plot.ui.histogram.item.vb.setMenuEnabled(False)

        # OVERALL LAYOUT
        layout = QVBoxLayout()

        # TOP ROW:
        top_row_layout = QHBoxLayout()

        # Spacer Widget:
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        top_row_layout.addWidget(spacer)

        labelBinSize = QLabel("Bin Size (m):")
        top_row_layout.addWidget(labelBinSize)

        self.spinboxBinSize = QDoubleSpinBox()
        self.spinboxBinSize.setToolTip("Raw data to be placed in square bins of this dimension.")
        self.spinboxBinSize.setDecimals(2)
        self.spinboxBinSize.setRange(0.05, 100)
        self.spinboxBinSize.setSingleStep(0.05)
        self.setBinSize(self.settings['processing_settings']['binSize_m'])
        top_row_layout.addWidget(self.spinboxBinSize)

        # pushButtonApply = QPushButton("Apply")
        iconApply = self.style().standardIcon(QStyle.SP_DialogApplyButton)
        pushButtonApply = QPushButton()
        pushButtonApply.setToolTip("Apply (Note: Changes in bin size cannot be applied retroactively.)")
        pushButtonApply.setIcon(iconApply)
        pushButtonApply.clicked.connect(self.binSizeEditedFunction)
        top_row_layout.addWidget(pushButtonApply)

        # pushButtonCancel = QPushButton("Cancel")
        iconCancel = self.style().standardIcon(QStyle.SP_DialogCancelButton)
        pushButtonCancel = QPushButton()
        pushButtonCancel.setToolTip("Cancel")
        pushButtonCancel.setIcon(iconCancel)
        pushButtonCancel.clicked.connect(self.resetBinSize)
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

        # Across-track Label
        labelMousePosAcrossTrack = QLabel("Across-Track (m):", parent=self)
        bottom_row_layout.addWidget(labelMousePosAcrossTrack)
        self.labelMousePosAcrossTrackValue = QLabel("nan", parent=self)
        bottom_row_layout.addWidget(self.labelMousePosAcrossTrackValue)

        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        bottom_row_layout.addWidget(line)

        # Depth Label
        labelMousePosDepth = QLabel("Depth (m):", parent=self)
        bottom_row_layout.addWidget(labelMousePosDepth)
        self.labelMousePosDepthValue = QLabel("nan", parent=self)
        bottom_row_layout.addWidget(self.labelMousePosDepthValue)

        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        bottom_row_layout.addWidget(line)

        # self.Intensity Label
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
        layout.addWidget(self.pie_plot)
        layout.addLayout(bottom_row_layout)

        self.setLayout(layout)

    def setSharedRingBufferProcessed(self, shared_ring_buffer_processed):
        self.shared_ring_buffer_processed = shared_ring_buffer_processed

    def mouseMoved(self, pos):
        try:
            position = self.pie_plot.getImageItem().mapFromScene(pos)

            # These values provide indices into image matrix that corresponds with cursor position
            self.matrix_x = position.x()
            self.matrix_y = position.y()
            # print("pie mousemoved, posx: {}, posy: {}".format(self.matrix_x, self.matrix_y))

            # These values provide x and y position of plot that corresponds with cursor position
            self.plot_x = self.matrix_x - (self.pie_plot.image.shape[0] / 2)
            self.plot_y = self.matrix_y - (self.settings['processing_settings']['maxHeave_m'] /
                                           self.settings['processing_settings']['binSize_m'])

            # print("pie mousemoved, corrected x: {}, corrected y: {}".format(self.plot_x, self.plot_y))

            self.vLine.setPos(self.plot_x)
            self.hLine.setPos(self.plot_y)

            if 0 <= round(self.matrix_x) < self.pie_plot.image.shape[0] and \
                    0 <= round(self.matrix_y) < self.pie_plot.image.shape[1]:
                self.intensity = self.pie_plot.image[round(self.matrix_x)][round(self.matrix_y)]
            else:
                self.intensity = float('nan')

            if math.isnan(self.intensity):
                self.matrix_x = self.matrix_y = float('nan')
                self.plot_x = self.plot_y = float('nan')

            self.setMousePositionLabels(self.plot_x, self.plot_y, self.intensity)

        except AttributeError:  # Triggered when nothing is plotted
            pass

    def setMousePositionLabels(self, across_track, depth, intensity):
        # timestamp = float('nan')
        # try:
        #     timestamp_epoch_sec = self.shared_ring_buffer_processed.view_buffer_elements(
        #         self.shared_ring_buffer_processed.timestamp_buffer_avg)[-1]
        #     timestamp = datetime.datetime.utcfromtimestamp(timestamp_epoch_sec).time()
        # except TypeError:  # Triggered when self.shared_ring_buffer_processed not fully initialized?
        #     pass

        if not math.isnan(intensity):
            across_track = round((across_track * self.settings['processing_settings']['binSize_m']), 2)
            depth = round((depth * self.settings['processing_settings']['binSize_m']), 2)
            self.intensity = round(intensity, 2)

        # self.labelMousePosTimeValue.setText(str(timestamp))
        self.labelMousePosAcrossTrackValue.setText(str(across_track))
        self.labelMousePosDepthValue.setText(str(depth))
        self.labelMousePosIntensityValue.setText(str(self.intensity))

    def getMousePosition(self):
        return self.matrix_x, self.matrix_y

    def updateTimestamp(self):
        """
        Called when plot is updated. Displays an updated value for timestamp.
        """
        timestamp = float('nan')
        try:
            if self.intensity:  # Ensure that self.intensity is not None
                timestamp_epoch_sec = self.shared_ring_buffer_processed.view_buffer_elements(
                    self.shared_ring_buffer_processed.timestamp_buffer_avg)[-1]
                timestamp = datetime.datetime.utcfromtimestamp(timestamp_epoch_sec).time()
        except TypeError:  # Triggered when self.shared_ring_buffer_processed not fully initialized?
            pass
        self.labelMousePosTimeValue.setText(str(timestamp))

    def updateIntensity(self):
        """
        Called when plot is updated. Displays an updated value for intensity at cursor hover position.
        """
        # NOTE: When cursor is not moving, last known plot coordinates
        # (self.plot_x, self.plot_y) will remain valid. Use these!

        if self.intensity:  # Ensure that self.intensity is not None
            self.intensity = float('nan')
            if not math.isnan(self.plot_x) and not math.isnan(self.plot_y):
                x = self.plot_x + self.pie_plot.image.shape[0] / 2
                y = self.plot_y + (self.settings['processing_settings']['maxHeave_m'] /
                                   self.settings['processing_settings']['binSize_m'])

                if 0 <= round(x) < self.pie_plot.image.shape[0] and \
                        0 <= round(y) < self.pie_plot.image.shape[1]:
                    self.intensity = self.pie_plot.image[round(x)][round(y)]
                # else:
                #     self.intensity = float('nan')

            self.labelMousePosIntensityValue.setText(str(self.intensity))

    def updateTimestampAndIntensity(self):
        self.updateTimestamp()
        self.updateIntensity()

    def setDepthIndicator(self, y):
        self.depthIndicator.setPos(y)

    def setBinSize(self, binSize):
        self.spinboxBinSize.setValue(binSize)

    def resetBinSize(self):
        self.spinboxBinSize.setValue(self.settings['processing_settings']['binSize_m'])

    def binSizeEditedFunction(self):
        if self.settings['processing_settings']['binSize_m'] != self.spinboxBinSize.value():
            self.settings['processing_settings']['binSize_m'] = round(self.spinboxBinSize.value(), 2)
            print("pie slice settings: ", self.settings['processing_settings']['binSize_m'])
            self.binSizeEdited.emit()
            self.processingSettingsEdited.emit()


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

        # self.setXRange(-(self.settings['buffer_settings']['maxGridCells'] / 2),
        #                (self.settings['buffer_settings']['maxGridCells'] / 2))
        self.setYRange(self.settings['buffer_settings']['maxGridCells'], 0)
        self.enableAutoRange()

        self.autoBtn.hide()
