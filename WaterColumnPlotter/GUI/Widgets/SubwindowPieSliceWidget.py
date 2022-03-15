# Lynette Davis
# ldavis@ccom.unh.edu
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# October 2021

# Description: A widget to be used as an MDI (Multiple Document Interface) subwindow;
# displays plots and settings relevant to the pie slice display.

import datetime
import math
from PyQt5.QtWidgets import QDoubleSpinBox, QFrame, QHBoxLayout, QLabel, \
    QPushButton, QSizePolicy, QStyle, QVBoxLayout, QWidget
from PyQt5.QtCore import pyqtSignal, Qt
import pyqtgraph as pg

# TODO: Consider use of pyvista for data visualization rather than pyqtgraph?
# import pyvista as pv  # "Easier Pythonic interface to VTK"


class SubwindowPieSliceWidget(QWidget):

    signalbinSizeEdited = pyqtSignal(name="binSizeEdited")
    signalsettingsEdited = pyqtSignal(name="settingsEdited")

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
        # self.plot.hideButtons()
        self.plot.vb.state['aspectLocked'] = False
        self.plot.setXRange(-(self.settings['buffer_settings']['maxGridCells'] / 2),
                            (self.settings['buffer_settings']['maxGridCells'] / 2))
        self.plot.setYRange(self.settings['buffer_settings']['maxGridCells'], 0)

        # Adding axis labels based on:
        # https://stackoverflow.com/questions/58516639/add-axes-labels-and-title-to-pyqtgraph-imageview
        self.plot.setLabel(axis='left', text='Depth')
        self.plot.setLabel(axis='bottom', text='Across-Track')

        # Crosshair - behind image
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.plot.addItem(self.vLine, ignoreBounds=True)
        self.plot.addItem(self.hLine, ignoreBounds=True)

        # ImageView
        self.pie_plot = pg.ImageView(self, view=self.plot)

        # Horizontal line to indicate position of horizontal slice
        self.depthIndicator = pg.InfiniteLine(angle=0, pen=pg.mkPen('m', width=1, style=Qt.DotLine), movable=False)
        self.depthIndicator.setPos(self.settings['processing_settings']['depth_m'] /
                                   self.settings['processing_settings']['binSize_m'])
        self.pie_plot.getView().addItem(self.depthIndicator)

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
        # self.pie_plot.getView().addItem(self.depthAvgIndicator1)
        # self.pie_plot.getView().addItem(self.depthAvgIndicator2)

        self.pie_plot.ui.histogram.setLevels(min=-95, max=35)
        # Based on https://stackoverflow.com/questions/38021869/getting-imageitem-values-from-pyqtgraph
        self.pie_plot.scene.sigMouseMoved.connect(self.mouseMoved)

        # TODO: Change axis labels to indicate meters and pings rather than bins???
        # https://stackoverflow.com/questions/63619065/pyqtgraph-use-arbitrary-values-for-axis-with-imageitem
        # self.pie_plot.getImageItem().setRect()

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

        # Bin size settings
        labelBinSize = QLabel("Bin Size (m):")
        top_row_layout.addWidget(labelBinSize)

        self.spinboxBinSize = QDoubleSpinBox()
        self.spinboxBinSize.setToolTip("Raw data to be placed in square bins of this dimension.")
        self.spinboxBinSize.setDecimals(2)
        self.spinboxBinSize.setRange(0.05, 100)
        self.spinboxBinSize.setSingleStep(0.05)
        self.setBinSize(self.settings['processing_settings']['binSize_m'])
        top_row_layout.addWidget(self.spinboxBinSize)

        # Apply button
        iconApply = self.style().standardIcon(QStyle.SP_DialogApplyButton)
        pushButtonApply = QPushButton()
        pushButtonApply.setToolTip("Apply (Note: Changes in bin size cannot be applied retroactively.)")
        pushButtonApply.setIcon(iconApply)
        pushButtonApply.clicked.connect(self.binSizeEditedFunction)
        top_row_layout.addWidget(pushButtonApply)

        # Cancel button
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
        """
        Sets reference to shared_ring_buffer_processed
        :param shared_ring_buffer_processed: An instance of SharedRingBufferProcessed
        """
        self.shared_ring_buffer_processed = shared_ring_buffer_processed

    def mouseMoved(self, pos):
        """
        Function is called upon movement of mouse over plot.
        Determines x, y, z values at cursor position and updates labels.
        :param pos: Passed when sigMouseMoved signal is emitted; contains position data.
        """
        try:
            position = self.pie_plot.getImageItem().mapFromScene(pos)

            # These values provide indices into image matrix that corresponds with cursor position
            self.matrix_x = position.x()
            self.matrix_y = position.y()

            # These values provide x and y position of plot that corresponds with cursor position
            self.plot_x = self.matrix_x - (self.pie_plot.image.shape[0] / 2)
            self.plot_y = self.matrix_y - (self.settings['processing_settings']['maxHeave_m'] /
                                           self.settings['processing_settings']['binSize_m'])

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
        """
        Updates window's across-track distance, depth, and intensity according to given (x, y) position.
        :param aross_track: x-position of mouse over plot
        :param depth: y-position of mouse over plot
        :param intensity: Intensity at given (x, y) position
        """
        if not math.isnan(intensity):
            across_track = round((across_track * self.settings['processing_settings']['binSize_m']), 2)
            depth = round((depth * self.settings['processing_settings']['binSize_m']), 2)
            self.intensity = round(intensity, 2)

        self.labelMousePosAcrossTrackValue.setText(str(across_track))
        self.labelMousePosDepthValue.setText(str(depth))
        self.labelMousePosIntensityValue.setText(str(self.intensity))

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
        # Note that unlike vertical and horizontal plots, timestamp over all of
        # pie display is constant and does not change with mouse position.
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
        Provides a mechanism to update window's intensity value when
        cursor is *not* moving using last recorded mouse position.
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

            self.labelMousePosIntensityValue.setText(str(self.intensity))

    def updateTimestampAndIntensity(self):
        """
        Called when plot is updated. Displays an updated timestamp and intensity value at cursor hover position.
        """
        self.updateTimestamp()
        self.updateIntensity()

    def setDepthIndicator(self, y):
        """
        Sets horizontal line to indicate depth of horizontal slice.
        :param y: Depth (by bin number) at which to set depth indicator
        """
        self.depthIndicator.setPos(y)

    def setDepthAvgIndicators(self, y1, y2):
        """
        Sets horizontal lines to indicate range of depth average of horizontal slice.
        :param y1: Depth (by bin number) at which to set first depth average indicator
        :param y2: Depth (by bin number) at which to set second depth average indicator
        """
        self.depthAvgIndicator1.setPos(y1)
        self.depthAvgIndicator2.setPos(y2)

    def setBinSize(self, binSize):
        """
        Sets value of bin size spinbox.
        :param binSize: Bin size value to assign to bin size spinbox
        """
        self.spinboxBinSize.setValue(binSize)

    def resetBinSize(self):
        """
        Resets values of bin size spinbox to the value contained in current settings.
        """
        self.spinboxBinSize.setValue(self.settings['processing_settings']['binSize_m'])

    def binSizeEditedFunction(self):
        """
        Updates current settings to reflect value in bin size spinbox;
        emits bin size edited and processing settings edited signals.
        """
        if self.settings['processing_settings']['binSize_m'] != self.spinboxBinSize.value():
            self.settings['processing_settings']['binSize_m'] = round(self.spinboxBinSize.value(), 2)
            self.binSizeEdited.emit()
            self.settingsEdited.emit()


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
