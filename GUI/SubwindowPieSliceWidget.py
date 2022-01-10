# Lynette Davis
# Center for Coastal and Ocean Mapping
# October 2021

import numpy as np
from PyQt5.QtWidgets import QDoubleSpinBox, QGridLayout, QLabel, QPushButton, QStyle, QWidget
from PyQt5.QtCore import pyqtSignal, Qt
import pyqtgraph as pg
# from qtrangeslider import QRangeSlider
import sys


class SubwindowPieSliceWidget(QWidget):

    signalbinSizeEdited = pyqtSignal(name="binSizeEdited")
    signalProcessingSettingsEdited = pyqtSignal(name="processingSettingsEdited")

    def __init__(self, settings, parent=None):
        super(SubwindowPieSliceWidget, self).__init__(parent)

        self.settings = settings

        self.setWindowTitle("Pie Slice")

        self.resize(200, 200)

        self.plot = GUI_PlotItem(self.settings)
        # self.plot = pg.PlotItem()
        self.plot.vb.state['aspectLocked'] = False
        self.plot.setXRange(-(self.settings['buffer_settings']['maxGridCells'] / 2),
                            (self.settings['buffer_settings']['maxGridCells'] / 2))
        self.plot.setYRange(self.settings['buffer_settings']['maxGridCells'], 0)

        # Adding axis labels based on:
        # https://stackoverflow.com/questions/58516639/add-axes-labels-and-title-to-pyqtgraph-imageview
        self.plot.setLabel(axis='left', text='Depth')
        self.plot.setLabel(axis='bottom', text='Across-Track')

        self.pie_plot = pg.ImageView(self, view=self.plot)

        self.pie_plot.ui.histogram.setLevels(min=-95, max=35)

        # Disable ROI button:
        self.pie_plot.ui.roiBtn.hide()
        # Disable Norm button:
        self.pie_plot.ui.menuBtn.hide()
        # Disable right-click context menu:
        self.pie_plot.view.setMenuEnabled(False)
        self.pie_plot.ui.histogram.item.vb.setMenuEnabled(False)
        #self.pie_plot.ui.histogram.item.vb.setRange(disableAutoRange=True)

        # Initialize plot with blank matrix of max size:
        # temp_image = np.empty((self.settings['buffer_settings']['maxGridCells'],
        #                        self.settings['buffer_settings']['maxBufferSize'],))
        # temp_image.fill(np.nan)
        # temp_image = np.zeros((self.settings['buffer_settings']['maxBufferSize'],
        #                        self.settings['buffer_settings']['maxGridCells'],))
        # autoRange = Auto scale and pan the view around the image such that the image fills the view.
        # autoLevels = Set the min/max intensity levels automatically to match the image data.
        # autoHistogramRange = If True, the histogram y-range is automatically scaled to fit the image data.
        # self.pie_plot.setImage(temp_image, autoRange=False, autoLevels=False, levels=(-95, 35),
        #                               autoHistogramRange=False)

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

        labelBinSize = QLabel("Bin Size (m):")
        layout.addWidget(labelBinSize, 0, 1)

        self.spinboxBinSize = QDoubleSpinBox()
        self.spinboxBinSize.setToolTip("Raw data to be placed in square bins of this dimension.")
        self.spinboxBinSize.setDecimals(2)
        self.spinboxBinSize.setRange(0, 100)
        self.spinboxBinSize.setSingleStep(0.1)
        self.setBinSize(self.settings['processing_settings']['binSize_m'])
        layout.addWidget(self.spinboxBinSize, 0, 2)

        layout.setColumnMinimumWidth(3, 5)

        # pushButtonApply = QPushButton("Apply")
        iconApply = self.style().standardIcon(QStyle.SP_DialogApplyButton)
        pushButtonApply = QPushButton()
        pushButtonApply.setToolTip("Apply (Note: Changes in bin size cannot be applied retroactively.)")
        pushButtonApply.setIcon(iconApply)
        pushButtonApply.clicked.connect(self.binSizeEditedFunction)
        layout.addWidget(pushButtonApply, 0, 4)

        # pushButtonCancel = QPushButton("Cancel")
        iconCancel = self.style().standardIcon(QStyle.SP_DialogCancelButton)
        pushButtonCancel = QPushButton()
        pushButtonCancel.setToolTip("Cancel")
        pushButtonCancel.setIcon(iconCancel)
        pushButtonCancel.clicked.connect(self.resetBinSize)
        layout.addWidget(pushButtonCancel, 0, 5)

        layout.addWidget(self.pie_plot, 1, 0, 3, 6)
        # layout.addWidget(pg.PlotWidget())
        # layout.addWidget(QRangeSlider(Qt.Horizontal), 4, 1, 1, 2)
        # layout.addWidget(QtWidgets.QPushButton("Bottom"))

        self.setLayout(layout)

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
        # self.setYRange(self.settings['buffer_settings']['maxGridCells'], 0)

        self.enableAutoRange()
        self.autoBtn.hide()
