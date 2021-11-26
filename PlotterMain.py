from KongsbergDGPlot import KongsbergDGPlot
import multiprocessing
from Plotter import Plotter
from PyQt5.QtCore import QRunnable, QThread
import pyqtgraph as pg


class PlotterMain(QRunnable):

    def __init__(self, settings, queue_pie, process_boolean=None, parent=None):
        super(PlotterMain, self).__init__()

        self.settings = settings
        self.queue_pie = queue_pie

        self.plotter = Plotter(self.settings, self.queue_pie, process_boolean)

    def run(self):
        print("In PlotterMain.run()")

        # self.process_plotter = multiprocessing.Process(target=self.plotter.run())
        self.process_plotter = multiprocessing.Process(target=self.plotter.get_and_plot_pie())
        self.process_plotter.daemon = True
        self.process_plotter.start()
        #process_plotter.join()
