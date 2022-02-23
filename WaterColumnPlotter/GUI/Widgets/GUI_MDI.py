# Lynette Davis
# ldavis@ccom.unh.edu
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# November 2021

# Description: MDI (Multiple Document Interface) class for WaterColumnPlotter MainWindow;
# initializes widgets for vertical, pie, and horizontal slices.

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMdiArea

from WaterColumnPlotter.GUI.Widgets.SubwindowPieSliceWidget import SubwindowPieSliceWidget
from WaterColumnPlotter.GUI.Widgets.SubwindowVerticalSliceWidget import SubwindowVerticalSliceWidget
from WaterColumnPlotter.GUI.Widgets.SubwindowHorizontalSliceWidget import SubwindowHorizontalSliceWidget


class GUI_MDI(QMdiArea):

    def __init__(self, settings, shared_ring_buffer_processed, parent=None):
        super(GUI_MDI, self).__init__(parent)

        self.settings = settings
        self.shared_ring_buffer_processed = shared_ring_buffer_processed

        SUBWINDOW_WIDTH_LARGE = 800
        SUBWINDOW_WIDTH_SMALL = 500
        SUBWINDOW_HEIGHT_LARGE = 360
        SUBWINDOW_HEIGHT_SMALL = 360

        # Create 3 widgets for plots
        # VERTICAL
        self.verticalWidget = SubwindowVerticalSliceWidget(self.settings, self.shared_ring_buffer_processed, self)
        # PIE
        self.pieWidget = SubwindowPieSliceWidget(self.settings, self.shared_ring_buffer_processed, self)
        # HORIZONTAL
        self.horizontalWidget = SubwindowHorizontalSliceWidget(self.settings, self.shared_ring_buffer_processed, self)

        # Add widgets to subwindow in QMdiArea
        self.__setupAndAddSubwindow(self.verticalWidget, SUBWINDOW_WIDTH_LARGE, SUBWINDOW_HEIGHT_LARGE)
        self.__setupAndAddSubwindow(self.pieWidget, SUBWINDOW_WIDTH_SMALL, SUBWINDOW_HEIGHT_SMALL)
        self.__setupAndAddSubwindow(self.horizontalWidget, SUBWINDOW_WIDTH_LARGE, SUBWINDOW_HEIGHT_LARGE)

    def setSharedRingBufferProcessed(self, shared_ring_buffer_processed):
        """
        Provides each widget (vertical, pie, horizontal) with a reference to shared_ring_buffer_processed.
        :param shared_ring_buffer_processed: An instance of SharedRingBufferProcessed
        """
        self.shared_ring_buffer_processed = shared_ring_buffer_processed
        self.verticalWidget.setSharedRingBufferProcessed(self.shared_ring_buffer_processed)
        self.pieWidget.setSharedRingBufferProcessed(self.shared_ring_buffer_processed)
        self.horizontalWidget.setSharedRingBufferProcessed(self.shared_ring_buffer_processed)


    def __setupAndAddSubwindow(self, widget, width, height):
        """
        Add QtWidget to subwindow in QMdiArea, resize subwindow, and disable window close button.
        :param widget: QtWidget to be added to QMdiArea
        :param width: Width of subwindow in QMdiArea
        :param height: Height of subwindow in QMdiArea
        """
        subwindow = self.addSubWindow(widget)
        subwindow.resize(width, height)
        # Disable close button:
        subwindow.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowCloseButtonHint)

    def setDepthIndicator(self):
        """
        Sets horizontal line on vertical and pie plots to indicate depth of horizontal slice.
        """
        bin_num_at_depth = round((self.settings['processing_settings']['depth_m'] /
                                  self.settings['processing_settings']['binSize_m']), 2)

        self.verticalWidget.setDepthIndicator(bin_num_at_depth)
        self.pieWidget.setDepthIndicator(bin_num_at_depth)

    # Omitted from implementation to decrease clutter over plot
    def setDepthAvgIndicators(self):
        """
        Sets horizontal lines on vertical and pie plots to indicate range of depth average of horizontal slice.
        """
        bin_num_at_depth = (self.settings['processing_settings']['depth_m'] /
                            self.settings['processing_settings']['binSize_m'])

        num_bins_depth_avg = (self.settings['processing_settings']['depthAvg_m'] /
                              self.settings['processing_settings']['binSize_m'])

        self.verticalWidget.setDepthAvgIndicators(round((bin_num_at_depth - (num_bins_depth_avg / 2)), 2),
                                                  round((bin_num_at_depth + (num_bins_depth_avg / 2)), 2))
        self.pieWidget.setDepthAvgIndicators(round((bin_num_at_depth - (num_bins_depth_avg / 2)), 2),
                                             round((bin_num_at_depth + (num_bins_depth_avg / 2)), 2))
