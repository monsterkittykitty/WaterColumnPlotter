# Lynette Davis
# Center for Coastal and Ocean Mapping
# October 2021

import datetime
import math
import numpy as np
from PyQt5.QtWidgets import QDoubleSpinBox, QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QStyle, \
    QVBoxLayout, QWidget
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QPen
import pyqtgraph as pg


class SubwindowVerticalSliceWidget(QWidget):
    signalAcrossTrackAvgEdited = pyqtSignal(name="acrossTrackAvgEdited")
    signalProcessingSettingsEdited = pyqtSignal(name="processingSettingsEdited")

    def __init__(self, settings, shared_ring_buffer_processed, parent=None):
        super(SubwindowVerticalSliceWidget, self).__init__(parent)

        self.settings = settings
        self.shared_ring_buffer_processed = shared_ring_buffer_processed

        # Mouse position
        self.matrix_x = None
        self.matrix_y = None
        self.intensity = None
        self.plot_x = None
        self.plot_y = None

        self.mouseClickedFlag = False

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

        # Crosshair - behind image
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.plot.addItem(self.vLine, ignoreBounds=True)
        self.plot.addItem(self.hLine, ignoreBounds=True)

        # ImageView
        self.vertical_plot = pg.ImageView(self, view=self.plot)

        # Horizontal line to indicate position of horizontal slice
        self.depthIndicator = pg.InfiniteLine(angle=0, pen=pg.mkPen('m', width=1, style=Qt.DotLine), movable=False)
        self.depthIndicator.setPos(self.settings['processing_settings']['depth_m'] /
                                   self.settings['processing_settings']['binSize_m'])
        self.vertical_plot.getView().addItem(self.depthIndicator)

        # Omitted to decrease clutter over plot
        # # Horizontal lines to indicate width and position of horizontal slice
        # self.depthAvgIndicator1 = pg.InfiniteLine(angle=0, pen=pg.mkPen('c', width=1, style=Qt.DotLine), movable=False)
        # self.depthAvgIndicator2 = pg.InfiniteLine(angle=0, pen=pg.mkPen('c', width=1, style=Qt.DotLine), movable=False)
        # self.depthAvgIndicator1.setPos((self.settings['processing_settings']['depth_m'] /
        #                                 self.settings['processing_settings']['binSize_m']) -
        #                                ((self.settings['processing_settings']['depthAvg_m'] /
        #                                  self.settings['processing_settings']['binSize_m']) / 2))
        # self.depthAvgIndicator2.setPos((self.settings['processing_settings']['depth_m'] /
        #                                 self.settings['processing_settings']['binSize_m']) +
        #                                ((self.settings['processing_settings']['depthAvg_m'] /
        #                                  self.settings['processing_settings']['binSize_m']) / 2))
        # # self.vertical_plot.getView().addItem(self.depthAvgIndicator1)
        # # self.vertical_plot.getView().addItem(self.depthAvgIndicator2)
        # self.plot.addItem(self.depthAvgIndicator1)
        # self.plot.addItem(self.depthAvgIndicator2)

        self.vertical_plot.ui.histogram.setLevels(min=-95, max=35)
        # Based on https://stackoverflow.com/questions/38021869/getting-imageitem-values-from-pyqtgraph
        self.vertical_plot.scene.sigMouseMoved.connect(self.mouseMoved)
        # self.vertical_plot.scene.sigMouseClicked.connect(self.mouseClicked)

        # # TODO: TEST
        # https://stackoverflow.com/questions/63619065/pyqtgraph-use-arbitrary-values-for-axis-with-imageitem
        # self.vertical_plot.getImageItem().setRect()

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

    def setSharedRingBufferProcessed(self, shared_ring_buffer_processed):
        self.shared_ring_buffer_processed = shared_ring_buffer_processed

    def setCoordinates(self):
        # https://stackoverflow.com/questions/63619065/pyqtgraph-use-arbitrary-values-for-axis-with-imageitem
        image_x = self.vertical_plot.image.shape[0]
        image_y = self.vertical_plot.image.shape[1]

        num_bins_heave = self.settings['processing_settings']['maxHeave_m'] / \
                         self.settings['processing_settings']['binSize_m']

        num_bins_depth = image_y - num_bins_heave

        meters_at_max_depth = num_bins_depth * self.settings['processing_settings']['binSize_m']

        print("meters_at_max_depth:", meters_at_max_depth)

        # We want our scale to go from -heave through max depth
        y_val = np.linspace(-self.settings['processing_settings']['maxHeave_m'], 5, int(meters_at_max_depth))

        x_pos = 0
        # y_pos = (self.settings['processing_settings']['maxHeave_m'] /
        #           self.settings['processing_settings']['binSize_m'])
        y_pos = 0
        # I don't want to change x_scale
        x_scale = self.settings['buffer_settings']['maxBufferSize_ping'] / \
                  self.settings['processing_settings']['alongTrackAvg_ping']
        y_scale = self.vertical_plot.image.shape[1] * self.settings['processing_settings']['binSize_m']

        print("x_scale, y_scale: ", x_scale, y_scale)

        self.vertical_plot.getImageItem().setRect(x_pos, y_pos, x_scale, y_scale)

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

    def mouseMoved(self, pos):
        try:
            position = self.vertical_plot.getImageItem().mapFromScene(pos)

            # These values provide indices into image matrix that corresponds with cursor position
            self.matrix_x = position.x()
            self.matrix_y = position.y()

            # These values provide x and y position of plot that corresponds with cursor position
            self.plot_x = self.matrix_x - self.vertical_plot.image.shape[0]
            self.plot_y = self.matrix_y - (self.settings['processing_settings']['maxHeave_m'] /
                                self.settings['processing_settings']['binSize_m'])
            # x, y = self.calcCursorPositionInPlot(self.matrix_x, self.matrix_y)

            self.vLine.setPos(self.plot_x)
            self.hLine.setPos(self.plot_y)

            if 0 <= round(self.matrix_x) < self.vertical_plot.image.shape[0] and \
                    0 <= round(self.matrix_y) < self.vertical_plot.image.shape[1]:
                self.intensity = self.vertical_plot.image[round(self.matrix_x)][round(self.matrix_y)]
            else:
                self.intensity = float('nan')

            if math.isnan(self.intensity):
                self.matrix_x = self.matrix_y = float('nan')
                self.plot_x = self.plot_y = float('nan')

            self.setMousePositionLabels(self.plot_x, self.plot_y, self.intensity)

        except AttributeError:  # Triggered when nothing is plotted
            pass

    # def cursorPositionInPlotCoordinates(self, x, y):
    #     plot_x = x - self.vertical_plot.image.shape[0]
    #     plot_y = y - (self.settings['processing_settings']['maxHeave_m'] /
    #                             self.settings['processing_settings']['binSize_m'])

    def setMousePositionLabels(self, ping, depth, intensity):
        timestamp = float('nan')
        try:
            if not math.isnan(self.matrix_x):
                temp_timestamp_buffer_elements = self.shared_ring_buffer_processed.view_buffer_elements(
                    self.shared_ring_buffer_processed.timestamp_buffer_avg)

                # Ensure that index is less than length of buffer.
                # Issues may occur here when bin_size is changed and shared_ring_buffer_processed is cleared.
                if round(self.matrix_x) < len(temp_timestamp_buffer_elements):
                    timestamp_epoch_sec = temp_timestamp_buffer_elements[round(self.matrix_x)]
                    timestamp = datetime.datetime.utcfromtimestamp(timestamp_epoch_sec).time()

                # timestamp_epoch_sec = self.shared_ring_buffer_processed.view_buffer_elements(
                #     self.shared_ring_buffer_processed.timestamp_buffer_avg)[round(self.matrix_x)]
                # timestamp = datetime.datetime.utcfromtimestamp(timestamp_epoch_sec).time()
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
        return self.matrix_x, self.matrix_y

    def updateTimestamp(self):
        # NOTE: When cursor is not moving, last known plot coordinates
        # (self.plot_x, self.plot_y) will remain valid. Use these!
        timestamp = float('nan')
        try:
            if self.intensity:  # Ensure that self.intensity is not None
                if not math.isnan(self.plot_x):
                    # TODO: It might be more correct to us self.shared_ring_buffer_processed... here
                    #  instead of self.vertical_plot...

                    # Ensure indices fall within matrix bounds
                    if 0 <= abs(round(self.plot_x)) < self.vertical_plot.image.shape[0]:
                        timestamp_epoch_sec = self.shared_ring_buffer_processed.view_buffer_elements(
                            self.shared_ring_buffer_processed.timestamp_buffer_avg)[round(self.plot_x)]
                        timestamp = datetime.datetime.utcfromtimestamp(timestamp_epoch_sec).time()
        except TypeError:  # Triggered when self.shared_ring_buffer_processed not fully initialized?
            pass
        self.labelMousePosTimeValue.setText(str(timestamp))

    def updateIntensity(self):
        # NOTE: When cursor is not moving, last known plot coordinates
        # (self.plot_x, self.plot_y) will remain valid. Use these!
        if self.intensity:  # Ensure that self.intensity is not None
            self.intensity = float('nan')
            if not math.isnan(self.plot_x) and not math.isnan(self.plot_y):
                # Because right-most edge of image is always positioned at x = 0,
                # we can simply use self.plot_x as our x coordinate.
                y = self.plot_y + (self.settings['processing_settings']['maxHeave_m'] /
                                   self.settings['processing_settings']['binSize_m'])

                # Ensure indices fall within matrix bounds
                if 0 <= abs(round(self.plot_x)) < self.vertical_plot.image.shape[0] and \
                        0 <= round(y) < self.vertical_plot.image.shape[1]:
                    self.intensity = self.vertical_plot.image[round(self.plot_x)][round(y)]
                # else:
                #     self.intensity = float('nan')

        self.labelMousePosIntensityValue.setText(str(self.intensity))

    def updateTimestampAndIntensity(self):
        self.updateTimestamp()
        self.updateIntensity()

    # def mouseClicked(self):
    #     self.mouseClickedFlag = not self.mouseClickedFlag
    #
    #     if self.mouseClickedFlag:
    #         # circle overlay
    #         pen = QPen(Qt.darkYellow, 0.05)
    #         r = CircleOverlay(pos=(self.plot_x, self.plot_y), size=3, pen=pen, movable=False)
    #         self.vertical_plot.getView().addItem(r)

    def setDepthIndicator(self, y):
        self.depthIndicator.setPos(y)

    def setDepthAvgIndicators(self, y1, y2):
        self.depthAvgIndicator1.setPos(y1)
        self.depthAvgIndicator2.setPos(y2)

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
        # self.vb.setLimits(yMin=-10)

        self.autoBtn.hide()


class CircleOverlay(pg.EllipseROI):
    def __init__(self, pos, size, **args):
        pg.ROI.__init__(self, pos, size, **args)
        self.aspectLocked = True