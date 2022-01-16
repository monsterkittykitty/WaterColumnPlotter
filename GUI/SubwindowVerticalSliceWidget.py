# Lynette Davis
# Center for Coastal and Ocean Mapping
# October 2021

import datetime
import math
import numpy as np
from PyQt5.QtWidgets import QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QStyle, \
    QVBoxLayout, QWidget
from PyQt5.QtCore import pyqtSignal, Qt
import pyqtgraph as pg
# from qtrangeslider import QRangeSlider
import sys


class SubwindowVerticalSliceWidget(QWidget):
    signalAcrossTrackAvgEdited = pyqtSignal(name="acrossTrackAvgEdited")
    signalProcessingSettingsEdited = pyqtSignal(name="processingSettingsEdited")

    def __init__(self, settings, shared_ring_buffer_processed, parent=None):
        super(SubwindowVerticalSliceWidget, self).__init__(parent)

        self.settings = settings
        self.shared_ring_buffer_processed = shared_ring_buffer_processed

        # Mouse position
        self.pos_x = None
        self.pos_y = None
        self.intensity = None

        self.setWindowTitle("Vertical Slice")

        self.plot = GUI_PlotItem(self.settings)
        self.plot.vb.state['aspectLocked'] = False
        self.plot.setXRange(-(self.settings['buffer_settings']['maxBufferSize_ping'] /
                              self.settings['processing_settings']['alongTrackAvg_ping']), 0)
        self.plot.setYRange(self.settings['buffer_settings']['maxGridCells'], 0)

        # Adding axis labels based on:
        # https://stackoverflow.com/questions/58516639/add-axes-labels-and-title-to-pyqtgraph-imageview
        self.plot.setLabel(axis='left', text='Depth')
        self.plot.setLabel(axis='bottom', text='Pings')

        # Crosshair
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.plot.addItem(self.vLine, ignoreBounds=True)
        self.plot.addItem(self.hLine, ignoreBounds=True)

        # ImageView
        self.vertical_plot = pg.ImageView(self, view=self.plot)

        self.vertical_plot.ui.histogram.setLevels(min=-95, max=35)
        # Based on https://stackoverflow.com/questions/38021869/getting-imageitem-values-from-pyqtgraph
        self.vertical_plot.scene.sigMouseMoved.connect(self.mouseMoved)

        # # TODO: TEST
        # https://stackoverflow.com/questions/63619065/pyqtgraph-use-arbitrary-values-for-axis-with-imageitem
        # self.xval = np.linspace(0, self.settings['processing_settings']['alongTrackAvg_ping'],
        #                           self.settings['buffer_settings']['maxBufferSize'])
        # self.yval = np.linspace(0, self.settings['processing_settings']['binSize_m'],
        #                         self.settings['buffer_settings']['maxGridCells'])
        #
        # # image_width = abs(self.xval_h[0]-self.xval_h[0])
        # # image_height = abs(self.xval_h[0]-self.xval_h[0])  # if x and y-scales are the same
        # image_
        # pixel_size = image_width/(self.xval_h.size-1)
        # self.image_h.setRect(QRectF(self.xval_h[0]-pixel_size/2, self.xval_h[0]-pixel_size/2, image_width, image_height))
        # # TODO: END TEST

        # Disable ROI button:
        self.vertical_plot.ui.roiBtn.hide()
        # Disable Norm button:
        self.vertical_plot.ui.menuBtn.hide()
        # Disable right-click context menu:
        self.vertical_plot.view.setMenuEnabled(False)
        self.vertical_plot.ui.histogram.item.vb.setMenuEnabled(False)

        # OVERALL LAYOUT:
        layout = QVBoxLayout()

        # TOP ROW:
        top_row_layout = QHBoxLayout()

        # Spacer Widget:
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        top_row_layout.addWidget(spacer)

        labelAcrossTrackAvg = QLabel("Across-Track Avg (m):")
        top_row_layout.addWidget(labelAcrossTrackAvg)

        self.spinboxAcrossTrackAvg = QDoubleSpinBox()
        self.spinboxAcrossTrackAvg.setToolTip("Width to average across-track.")
        self.spinboxAcrossTrackAvg.setDecimals(2)
        self.spinboxAcrossTrackAvg.setRange(1, 100)
        self.spinboxAcrossTrackAvg.setSingleStep(0.5)
        self.setAcrossTrackAvg(self.settings['processing_settings']['acrossTrackAvg_m'])
        top_row_layout.addWidget(self.spinboxAcrossTrackAvg)

        # pushButtonApply = QPushButton("Apply")
        iconApply = self.style().standardIcon(QStyle.SP_DialogApplyButton)
        pushButtonApply = QPushButton()
        pushButtonApply.setToolTip("Apply")
        pushButtonApply.setIcon(iconApply)
        pushButtonApply.clicked.connect(self.acrossTrackAvgEditedFunction)
        top_row_layout.addWidget(pushButtonApply)

        # pushButtonCancel = QPushButton("Cancel")
        iconCancel = self.style().standardIcon(QStyle.SP_DialogCancelButton)
        pushButtonCancel = QPushButton()
        pushButtonCancel.setToolTip("Cancel")
        pushButtonCancel.setIcon(iconCancel)
        pushButtonCancel.clicked.connect(self.resetAcrossTrackAvg)
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

        # Depth Label
        labelMousePosDepth = QLabel("Depth (m):", parent=self)
        bottom_row_layout.addWidget(labelMousePosDepth)
        self.labelMousePosDepthValue = QLabel("nan", parent=self)
        bottom_row_layout.addWidget(self.labelMousePosDepthValue)

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
        layout.addWidget(self.vertical_plot)
        layout.addLayout(bottom_row_layout)

        self.setLayout(layout)

        # TODO: Old implementation (before crosshair with ping and depth display).
        # layout = QGridLayout()
        # layout.setColumnMinimumWidth(0, 25)
        # layout.setColumnMinimumWidth(1, 25)
        # layout.setColumnMinimumWidth(2, 25)
        # layout.setColumnStretch(0, 1)
        # layout.setColumnStretch(1, 0)
        # layout.setColumnStretch(2, 0)
        #
        # # Dummy buttons:
        # # layout.addWidget(QtWidgets.QPushButton("T"), 0, 0)
        # # layout.addWidget(QtWidgets.QPushButton("To"), 0, 1)
        # # layout.addWidget(QtWidgets.QPushButton("Top"), 0, 2)
        #
        # labelAcrossTrackAvg = QLabel("Across-Track Avg (m):")
        # layout.addWidget(labelAcrossTrackAvg, 0, 1)
        #
        # self.spinboxAcrossTrackAvg = QDoubleSpinBox()
        # self.spinboxAcrossTrackAvg.setToolTip("Width to average across-track.")
        # self.spinboxAcrossTrackAvg.setDecimals(2)
        # self.spinboxAcrossTrackAvg.setRange(1, 100)
        # self.spinboxAcrossTrackAvg.setSingleStep(0.5)
        # self.setAcrossTrackAvg(self.settings['processing_settings']['acrossTrackAvg_m'])
        # layout.addWidget(self.spinboxAcrossTrackAvg, 0, 2)
        #
        # layout.setColumnMinimumWidth(3, 5)
        #
        # # pushButtonApply = QPushButton("Apply")
        # iconApply = self.style().standardIcon(QStyle.SP_DialogApplyButton)
        # pushButtonApply = QPushButton()
        # pushButtonApply.setToolTip("Apply")
        # pushButtonApply.setIcon(iconApply)
        # pushButtonApply.clicked.connect(self.acrossTrackAvgEditedFunction)
        # layout.addWidget(pushButtonApply, 0, 4)
        #
        # # pushButtonCancel = QPushButton("Cancel")
        # iconCancel = self.style().standardIcon(QStyle.SP_DialogCancelButton)
        # pushButtonCancel = QPushButton()
        # pushButtonCancel.setToolTip("Cancel")
        # pushButtonCancel.setIcon(iconCancel)
        # pushButtonCancel.clicked.connect(self.resetAcrossTrackAvg)
        # layout.addWidget(pushButtonCancel, 0, 5)
        #
        # layout.addWidget(self.vertical_plot, 1, 0, 3, 6)
        # # layout.addWidget(pg.PlotWidget())
        # # layout.addWidget(QRangeSlider(Qt.Horizontal), 4, 1, 1, 2)
        # # layout.addWidget(QtWidgets.QPushButton("Bottom"))
        #
        # self.setLayout(layout)

    def mouseMoved(self, pos):
        try:
            position = self.vertical_plot.getImageItem().mapFromScene(pos)

            # These values provide indices into image matrix that corresponds with cursor position
            self.pos_x = position.x()
            self.pos_y = position.y()

            # These values provide x and y position of plot that corresponds with cursor position
            x = position.x() - self.vertical_plot.image.shape[0]
            y = position.y() - (self.settings['processing_settings']['maxHeave_m'] /
                                self.settings['processing_settings']['binSize_m'])

            self.vLine.setPos(x)
            self.hLine.setPos(y)

            if 0 <= round(self.pos_x) < self.vertical_plot.image.shape[0] and \
                    0 <= round(self.pos_y) < self.vertical_plot.image.shape[1]:
                self.intensity = self.vertical_plot.image[round(self.pos_x)][round(self.pos_y)]
            else:
                self.intensity = float('nan')

            if math.isnan(self.intensity):
                x = y = float('nan')
                self.pos_x = self.pos_y = float('nan')

            self.setMousePositionLabels(x, y, self.intensity)

        except AttributeError:  # Triggered when nothing is plotted
            pass

    def setMousePositionLabels(self, ping, depth, intensity):
        timestamp = float('nan')
        try:
            if not math.isnan(self.pos_x):
                timestamp_epoch_sec = self.shared_ring_buffer_processed.view_buffer_elements(
                    self.shared_ring_buffer_processed.timestamp_buffer_avg)[round(self.pos_x)]
                timestamp = datetime.datetime.utcfromtimestamp(timestamp_epoch_sec).time()
        except TypeError:  # Triggered when self.shared_ring_buffer_processed not fully initialized?
            pass

        if not math.isnan(intensity):
            ping = round(ping * self.settings['processing_settings']['alongTrackAvg_ping'])
            depth = round((depth * self.settings['processing_settings']['binSize_m']), 2)
            self.intensity = round(intensity, 2)

        self.labelMousePosTimeValue.setText(str(timestamp))
        self.labelMousePosPingValue.setText(str(ping))
        self.labelMousePosDepthValue.setText(str(depth))
        self.labelMousePosIntensityValue.setText(str(self.intensity))

    def getMousePosition(self):
        return self.pos_x, self.pos_y

    def updateTimestamp(self):
        timestamp = float('nan')
        try:
            if self.intensity:  # Ensure that self.intensity is not None
                if not math.isnan(self.pos_x):
                    timestamp_epoch_sec = self.shared_ring_buffer_processed.view_buffer_elements(
                        self.shared_ring_buffer_processed.timestamp_buffer_avg)[round(self.pos_x)]
                    timestamp = datetime.datetime.utcfromtimestamp(timestamp_epoch_sec).time()
        except TypeError:  # Triggered when self.shared_ring_buffer_processed not fully initialized?
            pass
        self.labelMousePosTimeValue.setText(str(timestamp))

    def updateIntensity(self):
        x, y = self.getMousePosition()
        if x and y:
            if not math.isnan(x) and math.isnan(y):
                if 0 <= round(x) < self.pie_plot.image.shape[0] and \
                        0 <= round(y) < self.pie_plot.image.shape[1]:
                    self.intensity = self.pie_plot.image[round(x)][round(y)]
                else:
                    self.intensity = float('nan')

        self.labelMousePosIntensityValue.setText(str(self.intensity))

    def updateTimestampAndIntensity(self):
        self.updateTimestamp()
        self.updateTimestamp()

    def setAcrossTrackAvg(self, acrossTrackAvg):
        self.spinboxAcrossTrackAvg.setValue(acrossTrackAvg)

    def resetAcrossTrackAvg(self):
        self.spinboxAcrossTrackAvg.setValue(self.settings['processing_settings']['acrossTrackAvg_m'])

    def acrossTrackAvgEditedFunction(self):
        if self.settings['processing_settings']['acrossTrackAvg_m'] != self.spinboxAcrossTrackAvg.value():
            self.settings['processing_settings']['acrossTrackAvg_m'] = round(self.spinboxAcrossTrackAvg.value(), 2)
            self.acrossTrackAvgEdited.emit()
            self.processingSettingsEdited.emit()


# class GUI_PlotItem(pg.PlotWidget):
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
