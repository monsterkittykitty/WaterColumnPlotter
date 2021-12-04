# Lynette Davis
# Center for Coastal and Ocean Mapping
# October 2021

import numpy as np
from PyQt5.QtWidgets import QDoubleSpinBox, QGridLayout, QLabel, QPushButton, QStyle, QWidget
from PyQt5.QtCore import pyqtSignal, Qt
import pyqtgraph as pg
# from qtrangeslider import QRangeSlider
import sys


class SubwindowVerticalSliceWidget(QWidget):

    signalAcrossTrackAvgEdited = pyqtSignal(name="acrossTrackAvgEdited")

    def __init__(self, settings, parent=None):
        super(SubwindowVerticalSliceWidget, self).__init__(parent)

        self.settings = settings

        self.setWindowTitle("Vertical Slice")

        self.plot = GUI_PlotItem(self.settings)
        # self.plot = pg.PlotItem()
        # self.plot.hideButtons()
        self.plot.vb.state['aspectLocked'] = False
        self.plot.setXRange(-(self.settings['buffer_settings']['maxBufferSize'] /
                         self.settings['processing_settings']['alongTrackAvg_ping']), 0)
        self.plot.setYRange(self.settings['buffer_settings']['maxGridCells'], 0)

        # Adding axis labels based on:
        # https://stackoverflow.com/questions/58516639/add-axes-labels-and-title-to-pyqtgraph-imageview
        self.plot.setLabel(axis='left', text='Depth')
        self.plot.setLabel(axis='bottom', text='Along-Track')

        self.vertical_plot = pg.ImageView(self, view=self.plot)

        # temp_image = np.empty((500, 100))
        # temp_image[:] = np.nan
        # self.vertical_plot.setImage(temp_image, autoRange=False, autoLevels=False, levels=(-95, 35), autoHistogramRange=False)
        # Disable ROI button:
        self.vertical_plot.ui.roiBtn.hide()
        # Disable Norm button:
        self.vertical_plot.ui.menuBtn.hide()
        # Disable right-click context menu:
        self.vertical_plot.view.setMenuEnabled(False)
        self.vertical_plot.ui.histogram.item.vb.setMenuEnabled(False)
        #self.vertical_plot.ui.histogram.item.vb.setRange(disableAutoRange=True)

        # Initialize plot with blank matrix of max size:
        # temp_image = np.empty((self.settings['buffer_settings']['maxGridCells'],
        #                        self.settings['buffer_settings']['maxBufferSize'],))
        # temp_image.fill(np.nan)
        # temp_image = np.zeros((self.settings['buffer_settings']['maxBufferSize'],
        #                        self.settings['buffer_settings']['maxGridCells'], ))
        # autoRange = Auto scale and pan the view around the image such that the image fills the view.
        # autoLevels = Set the min/max intensity levels automatically to match the image data.
        # autoHistogramRange = If True, the histogram y-range is automatically scaled to fit the image data.
        # self.vertical_plot.setImage(temp_image, autoRange=False, autoLevels=False, levels=(-95, 35),
        #                               autoHistogramRange=False)

        #self.resize(400, 200)

        #layout = QtWidgets.QVBoxLayout()
        layout = QGridLayout()
        layout.setColumnMinimumWidth(0, 25)
        layout.setColumnMinimumWidth(1, 25)
        layout.setColumnMinimumWidth(2, 25)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 0)
        layout.setColumnStretch(2, 0)

        # Dummy buttons:
        # layout.addWidget(QtWidgets.QPushButton("T"), 0, 0)
        # layout.addWidget(QtWidgets.QPushButton("To"), 0, 1)
        # layout.addWidget(QtWidgets.QPushButton("Top"), 0, 2)

        labelAcrossTrackAvg = QLabel("Across-Track Avg (m):")
        layout.addWidget(labelAcrossTrackAvg, 0, 1)

        self.spinboxAcrossTrackAvg = QDoubleSpinBox()
        self.spinboxAcrossTrackAvg.setToolTip("Width to average across-track.")
        self.spinboxAcrossTrackAvg.setDecimals(2)
        self.spinboxAcrossTrackAvg.setRange(1, 100)
        self.spinboxAcrossTrackAvg.setSingleStep(0.5)
        self.setAcrossTrackAvg(self.settings['processing_settings']['acrossTrackAvg_m'])
        layout.addWidget(self.spinboxAcrossTrackAvg, 0, 2)

        layout.setColumnMinimumWidth(3, 5)

        # pushButtonApply = QPushButton("Apply")
        iconApply = self.style().standardIcon(QStyle.SP_DialogApplyButton)
        pushButtonApply = QPushButton()
        pushButtonApply.setToolTip("Apply")
        pushButtonApply.setIcon(iconApply)
        pushButtonApply.clicked.connect(self.acrossTrackAvgEditedFunction)
        layout.addWidget(pushButtonApply, 0, 4)

        # pushButtonCancel = QPushButton("Cancel")
        iconCancel = self.style().standardIcon(QStyle.SP_DialogCancelButton)
        pushButtonCancel = QPushButton()
        pushButtonCancel.setToolTip("Cancel")
        pushButtonCancel.setIcon(iconCancel)
        pushButtonCancel.clicked.connect(self.resetAcrossTrackAvg)
        layout.addWidget(pushButtonCancel, 0, 5)

        layout.addWidget(self.vertical_plot, 1, 0, 3, 6)
        # layout.addWidget(pg.PlotWidget())
        # layout.addWidget(QRangeSlider(Qt.Horizontal), 4, 1, 1, 2)
        # layout.addWidget(QtWidgets.QPushButton("Bottom"))

        self.setLayout(layout)

    def setAcrossTrackAvg(self, acrossTrackAvg):
        self.spinboxAcrossTrackAvg.setValue(acrossTrackAvg)

    def resetAcrossTrackAvg(self):
        self.spinboxAcrossTrackAvg.setValue(self.settings['processing_settings']['acrossTrackAvg_m'])

    def acrossTrackAvgEditedFunction(self):
        if self.settings['processing_settings']['acrossTrackAvg_m'] != self.spinboxAcrossTrackAvg.value():
            self.settings['processing_settings']['acrossTrackAvg_m'] = round(self.spinboxAcrossTrackAvg.value(), 2)
            self.acrossTrackAvgEdited.emit()


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
        self.setXRange(-(self.settings['buffer_settings']['maxBufferSize'] /
                         self.settings['processing_settings']['alongTrackAvg_ping']), 0)
        self.setYRange(self.settings['buffer_settings']['maxGridCells'], 0)
