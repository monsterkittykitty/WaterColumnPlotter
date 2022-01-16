# MDI (Multiple Document Interface) class for Water Column Plotter MainWindow.

# Lynette Davis
# Center for Coastal and Ocean Mapping
# November 2021

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QAction, QApplication, QFileDialog, QGroupBox, QLabel, QMdiArea, QPushButton, QStyle, QToolBar, QToolButton, QVBoxLayout, QWidget

from GUI.SubwindowPieSliceWidget import SubwindowPieSliceWidget
from GUI.SubwindowVerticalSliceWidget import SubwindowVerticalSliceWidget
from GUI.SubwindowHorizontalSliceWidget import SubwindowHorizontalSliceWidget
from GUI.SubwindowSettingsDisplay import SubwindowSettingsDisplay

class GUI_MDI(QMdiArea):

    def __init__(self, settings, shared_ring_buffer_processed, parent=None):
        super(GUI_MDI, self).__init__(parent)

        self.settings = settings
        self.shared_ring_buffer_processed = shared_ring_buffer_processed

        SUBWINDOW_WIDTH_LARGE = 800
        SUBWINDOW_WIDTH_SMALL = 500
        SUBWINDOW_HEIGHT_LARGE = 360
        SUBWINDOW_HEIGHT_SMALL = 360

        # Create 3 widgets for plots; 1 widget to display settings:
        # VERTICAL:
        self.verticalWidget = SubwindowVerticalSliceWidget(self.settings, self.shared_ring_buffer_processed, self)
        # PIE
        self.pieWidget = SubwindowPieSliceWidget(self.settings, self.shared_ring_buffer_processed, self)
        # HORIZONTAL:
        self.horizontalWidget = SubwindowHorizontalSliceWidget(self.settings, self.shared_ring_buffer_processed, self)
        # SETTINGS:
        #self.subwindowSettingsDisplay = SubwindowSettingsDisplay(self.settings, self)

        # Add widgets to subwindow in QMdiArea
        self.__setupAndAddSubwindow(self.verticalWidget, SUBWINDOW_WIDTH_LARGE, SUBWINDOW_HEIGHT_LARGE)
        self.__setupAndAddSubwindow(self.pieWidget, SUBWINDOW_WIDTH_SMALL, SUBWINDOW_HEIGHT_SMALL)
        self.__setupAndAddSubwindow(self.horizontalWidget, SUBWINDOW_WIDTH_LARGE, SUBWINDOW_HEIGHT_LARGE)
        #self.__setupAndAddSubwindow(self.subwindowSettingsDisplay, SUBWINDOW_WIDTH_SMALL, SUBWINDOW_HEIGHT_SMALL)

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