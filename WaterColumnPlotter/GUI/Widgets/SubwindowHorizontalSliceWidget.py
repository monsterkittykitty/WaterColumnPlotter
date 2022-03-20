# Lynette Davis
# ldavis@ccom.unh.edu
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# October 2021

# Description: A widget to be used as an MDI (Multiple Document Interface) subwindow;
# displays plots and settings relevant to the horizontal slice display.

import datetime
import math
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QDoubleSpinBox, QFrame, QHBoxLayout, QLabel, \
    QPushButton, QSizePolicy, QStyle, QVBoxLayout, QWidget
import pyqtgraph as pg

# TODO: Consider use of pyvista for data visualization rather than pyqtgraph?
# import pyvista as pv  # "Easier Pythonic interface to VTK"


class SubwindowHorizontalSliceWidget(QWidget):

    signalDepthEdited = pyqtSignal(name="depthEdited")
    signalDepthAvgEdited = pyqtSignal(name="depthAvgEdited")
    signalsettingsEdited = pyqtSignal(name="settingsEdited")

    def __init__(self, settings, shared_ring_buffer_processed, parent=None):
        super(SubwindowHorizontalSliceWidget, self).__init__(parent)

        self.settings = settings
        self.shared_ring_buffer_processed = shared_ring_buffer_processed

        # Mouse position
        self.matrix_x = None
        self.matrix_y = None
        self.intensity = None
        self.plot_x = None
        self.plot_y = None

        self.setWindowTitle("Horizontal Slice")

        self.plot = GUI_PlotItem(self.settings)
        # self.plot.hideButtons()
        self.plot.vb.state['aspectLocked'] = False
        self.plot.setXRange(-(self.settings['buffer_settings']['maxBufferSize_ping'] /
                              self.settings['processing_settings']['alongTrackAvg_ping']), 0)
        self.plot.setYRange(self.settings['buffer_settings']['maxGridCells'], 0)

        # Adding axis labels based on:
        # https://stackoverflow.com/questions/58516639/add-axes-labels-and-title-to-pyqtgraph-imageview
        self.plot.setLabel(axis='left', text='Across-Track')
        self.plot.setLabel(axis='bottom', text='Pings')

        # Crosshair - behind image
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.plot.addItem(self.vLine, ignoreBounds=True)
        self.plot.addItem(self.hLine, ignoreBounds=True)

        # ImageView
        self.horizontal_plot = pg.ImageView(self, view=self.plot)
        self.horizontal_plot.ui.histogram.setLevels(min=-95, max=35)
        # Based on: https://stackoverflow.com/questions/38021869/getting-imageitem-values-from-pyqtgraph
        self.horizontal_plot.scene.sigMouseMoved.connect(self.mouseMoved)

        # TODO: Change axis labels to indicate meters and pings rather than bins???
        # https://stackoverflow.com/questions/63619065/pyqtgraph-use-arbitrary-values-for-axis-with-imageitem
        # self.horizontal_plot.getImageItem().setRect()

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
        # TODO: END

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

        # Depth settings
        labelDepth = QLabel("Depth (m):")
        top_row_layout.addWidget(labelDepth)

        self.spinboxDepth = QDoubleSpinBox()
        self.spinboxDepth.setToolTip("Depth of horizontal slice.")
        self.spinboxDepth.setDecimals(2)
        self.spinboxDepth.setRange(1, 1000)
        self.spinboxDepth.setSingleStep(0.5)
        self.setDepth(self.settings['processing_settings']['depth_m'])
        top_row_layout.addWidget(self.spinboxDepth)

        # Depth average settings
        labelDepthAvg = QLabel("Depth Average (m):")
        top_row_layout.addWidget(labelDepthAvg)

        self.spinboxDepthAvg = QDoubleSpinBox()
        self.spinboxDepthAvg.setToolTip("Depth to average.")
        self.spinboxDepthAvg.setDecimals(2)
        self.spinboxDepthAvg.setRange(1, 100)
        self.spinboxDepthAvg.setSingleStep(0.5)
        self.setDepthAvg(self.settings['processing_settings']['depthAvg_m'])
        top_row_layout.addWidget(self.spinboxDepthAvg)

        # Apply button
        iconApply = self.style().standardIcon(QStyle.SP_DialogApplyButton)
        pushButtonApply = QPushButton()
        pushButtonApply.setToolTip("Apply")
        pushButtonApply.setIcon(iconApply)
        pushButtonApply.clicked.connect(self.editedAllFunction)
        top_row_layout.addWidget(pushButtonApply)

        # Cancel button
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

        self.setLayout(layout)

    def setSharedRingBufferProcessed(self, shared_ring_buffer_processed):
        """
        Sets reference to shared_ring_buffer_processed
        :param shared_ring_buffer_processed: An instance of SharedRingBufferProcessed
        """
        self.shared_ring_buffer_processed = shared_ring_buffer_processed
        self.plot.set_ring_buffer_processed_length(self.shared_ring_buffer_processed.SIZE_BUFFER)

    def mouseMoved(self, pos):
        """
        Function is called upon movement of mouse over plot.
        Determines x, y, z values at cursor position and updates labels.
        :param pos: Passed when sigMouseMoved signal is emitted; contains position data.
        """
        try:
            position = self.horizontal_plot.getImageItem().mapFromScene(pos)

            self.matrix_x = position.x()
            self.matrix_y = position.y()

            # Correct for shifted position of image
            self.plot_x = self.matrix_x - self.horizontal_plot.image.shape[0]
            self.plot_y = self.matrix_y - (self.horizontal_plot.image.shape[1] / 2)

            self.vLine.setPos(self.plot_x)
            self.hLine.setPos(self.plot_y)

            if 0 <= round(self.matrix_x) < self.horizontal_plot.image.shape[0] and \
                    0 <= round(self.matrix_y) < self.horizontal_plot.image.shape[1]:
                self.intensity = self.horizontal_plot.image[round(self.matrix_x)][round(self.matrix_y)]
            else:
                self.intensity = float('nan')

            if math.isnan(self.intensity):
                self.matrix_x = self.matrix_y = float('nan')
                self.plot_x = self.plot_y = float('nan')

            self.setMousePositionLabels(self.plot_x, self.plot_y, self.intensity)

        except AttributeError:  # Triggered when nothing is plotted
            pass

    def setMousePositionLabels(self, ping, across_track, intensity):
        """
        Updates window's timestamp, ping number, across-track distance,
        and intensity according to given (x, y) position.
        :param ping: x-position of mouse over plot
        :param across_track: y-position of mouse over plot
        :param intensity: Intensity at given (x, y) position
        """
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

        except TypeError:  # Triggered when self.shared_ring_buffer_processed not fully initialized?
            pass

        if not math.isnan(intensity):
            ping = round(ping * self.settings['processing_settings']['alongTrackAvg_ping'])
            across_track = round((across_track * self.settings['processing_settings']['binSize_m']), 2)
            intensity = round(intensity, 2)

        self.labelMousePosTimeValue.setText(str(timestamp))
        self.labelMousePosPingValue.setText(str(ping))
        self.labelMousePosAcrossTrackValue.setText(str(across_track))
        self.labelMousePosIntensityValue.setText(str(intensity))

    def getMousePosition(self):
        """
        Return cursor position over plot.
        :return: (x, y) position of cursor over plot in bin number
        """
        return self.matrix_x, self.matrix_y

    def updateTimestamp(self):
        """
        Provides a mechanism to update window's timestamp value when
        cursor is *not* moving using last recorded mouse position.
        """
        # NOTE: When cursor is not moving, last known plot coordinates
        # (self.plot_x, self.plot_y) will remain valid. Use these!
        timestamp = float('nan')
        try:
            if self.intensity:  # Ensure that self.intensity is not None
                if not math.isnan(self.plot_x):
                    # TODO: It might be more correct to us self.shared_ring_buffer_processed... here
                    #  instead of self.horizontal_plot...
                    # Ensure indices fall within matrix bounds
                    if 0 <= abs(round(self.plot_x)) < self.horizontal_plot.image.shape[0]:
                        timestamp_epoch_sec = self.shared_ring_buffer_processed.view_buffer_elements(
                            self.shared_ring_buffer_processed.timestamp_buffer_avg)[round(self.plot_x)]
                        timestamp = datetime.datetime.utcfromtimestamp(timestamp_epoch_sec).time()
        except TypeError:  # Triggered when self.shared_ring_buffer_processed not fully initialized?
            pass
        self.labelMousePosTimeValue.setText(str(timestamp))

    def updateIntensity(self):
        """
        Provides a mechanism to update window's intensity value when
        cursor is *not* moving using last recorded mouse position.
        """
        # NOTE: When cursor is not moving, last known plot coordinates
        # (self.plot_x, self.plot_y) will remain valid. Use these!
        if self.intensity:  # Ensure that self.intensity is not None
            self.intensity = float('nan')
            if not math.isnan(self.plot_x) and not math.isnan(self.plot_y):
                # Because right-most edge of image is always positioned at x = 0,
                # we can simply use self.plot_x as our x coordinate.
                y = self.plot_y + (self.horizontal_plot.image.shape[1] / 2)

                # Ensure indices fall within matrix bounds
                if 0 <= abs(round(self.plot_x)) < self.horizontal_plot.image.shape[0] and \
                        0 <= round(y) < self.horizontal_plot.image.shape[1]:
                    self.intensity = self.horizontal_plot.image[round(self.plot_x)][round(y)]

        self.labelMousePosIntensityValue.setText(str(self.intensity))

    def updateTimestampAndIntensity(self):
        """
        Called when plot is updated. Displays an updated timestamp and intensity value at cursor hover position.
        """
        self.updateTimestamp()
        self.updateIntensity()

    def setDepth(self, depth):
        """
        Sets value of depth spinbox.
        :param depth: Depth value to assign to depth spinbox
        """
        self.spinboxDepth.setValue(depth)

    def setDepthAvg(self, depthAvg):
        """
        Sets value of depth average spinbox.
        :param depthAvg: Depth average value to assign to depth average spinbox
        """
        self.spinboxDepthAvg.setValue(depthAvg)

    def resetAll(self):
        """
        Resets values of both depth and depth average spinboxes to the values contained in current settings.
        """
        self.resetDepth()
        self.resetDepthAvg()

    def resetDepth(self):
        """
        Resets value of depth spinbox to value contained in current settings.
        """
        self.spinboxDepth.setValue(self.settings['processing_settings']['depth_m'])

    def resetDepthAvg(self):
        """
        Resets value of depth average spinbox to value contained in current settings.
        """
        self.spinboxDepthAvg.setValue(self.settings['processing_settings']['depthAvg_m'])

    def editedAllFunction(self):
        """
        Updates current settings to reflect values in depth and depth average spinboxes;
        emits depth edited, depth average edited, and processing settings edited signals.
        """
        self.depthEditedFunction()
        self.depthAvgEditedFunction()
        self.settingsEdited.emit()

    def depthEditedFunction(self):
        """
        Updates current settings to reflect value in depth spinbox; emits depth edited signal.
        """
        if self.settings['processing_settings']['depth_m'] != self.spinboxDepth.value():
            self.settings['processing_settings']['depth_m'] = round(self.spinboxDepth.value(), 2)
            self.depthEdited.emit()

    def depthAvgEditedFunction(self):
        """
        Updates current settings to reflect value in depth average spinbox; emits depth average edited signal.
        """
        if self.settings['processing_settings']['depthAvg_m'] != self.spinboxDepthAvg.value():
            self.settings['processing_settings']['depthAvg_m'] = round(self.spinboxDepthAvg.value(), 2)
            self.depthAvgEdited.emit()


class GUI_PlotItem(pg.PlotItem):
    def __init__(self, settings, parent=None):
        super(GUI_PlotItem, self).__init__(parent)

        self.settings = settings
        self.ring_buffer_processed_length = None

    def set_ring_buffer_processed_length(self, ring_buffer_processed_length):
        self.ring_buffer_processed_length = ring_buffer_processed_length

    def autoBtnClicked(self):
        """
        Overrides method in pyqtgraph.PlotItem. Instead of enabling autoRange when clicked,
        this button will now return image to default home view.
        """
        # This ensures that full, specified x, y range will be displayed, but 1:1 aspect ratio may not be maintained.
        self.vb.state['aspectLocked'] = False

        self.enableAutoRange()

        if self.ring_buffer_processed_length:
            self.setXRange(-self.ring_buffer_processed_length, 0)
        else:
            self.setXRange(-(self.settings['buffer_settings']['maxBufferSize_ping'] /
                             self.settings['processing_settings']['alongTrackAvg_ping']), 0)
        # self.setYRange(self.settings['buffer_settings']['maxGridCells'], 0)

        self.autoBtn.hide()
