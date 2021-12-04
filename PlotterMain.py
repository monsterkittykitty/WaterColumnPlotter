from KongsbergDGPlot import KongsbergDGPlot
import multiprocessing
#from Plotter import Plotter
from Plotter2 import Plotter2
from PyQt5.QtCore import QRunnable, QThread
import pyqtgraph as pg

import TestProcess


class PlotterMain(QRunnable):

    def __init__(self, settings, queue_pie, process_boolean=None, parent=None):
        super(PlotterMain, self).__init__()

        self.settings = settings
        self.queue_pie = queue_pie

        #self.plotter = Plotter(self.settings, self.queue_pie, process_boolean)
        self.plotter = Plotter2(self.settings, self.queue_pie, process_boolean)

        # TODO: Delete this
        self.test_process = TestProcess.TestProcess(process_boolean)

    def run(self):
        print("In PlotterMain.run()")

        # self.process_plotter = multiprocessing.Process(target=self.plotter.run())
        #self.process_plotter = multiprocessing.Process(target=self.plotter.get_and_plot_pie())

        # This does not print the true conditions, instead it results in a pickle error:
        # _pickle.UnpicklingError: pickle data was truncated
        # And GUI_Main does not exit if play is pressed, (then stop pressed or unpressed), then main window closed.
        self.process_plotter = multiprocessing.Process(target=self.plotter.get_and_buffer_pie)

        # This does print the true conditions, and GUI_Main does exit when main window closed if stop has been pressed.
        # But this keeps printing if main window is closed and stop is not pressed.
        #self.process_plotter = multiprocessing.Process(target=self.plotter.get_and_buffer_pie())

        # TODO: Delete this
        # No pickle error with this. This works great. No problems with or without stop button, GUI_Main closes.
        #self.process_plotter =multiprocessing.Process(target=self.test_process.run)


        self.process_plotter.daemon = True
        self.process_plotter.start()
        #self.process_plotter.join()
