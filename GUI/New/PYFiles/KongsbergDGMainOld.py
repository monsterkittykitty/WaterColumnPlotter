# Lynette Davis
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# August 2021

# Description:


from GUI.Dialogs import IPSettingsDialog
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QDialog
from PyQt5.QtWidgets import QDialogButtonBox
from PyQt5.QtWidgets import QFormLayout
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QStatusBar
from PyQt5.QtWidgets import QToolBar
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QWidget
import sys
import time


class WaterColumnGUI(QMainWindow):

    def __init__(self, parent=None):
        # Using as model: https://github.com/hydroffice/hyo2_soundspeed/blob/master/hyo2/soundspeedmanager/mainwin.py

        # NOTE: All of these options work:
        #super(WaterColumnGUI, self).__init__(parent)
        #QMainWindow.__init__(self)
        super().__init__()

        self.setWindowTitle('Water Column Plotter')

        self.setMinimumSize(640, 480)
        self.resize(920, 640)

        # TODO: Include icon?
        # self.setWindowIcon()

        # TODO: Allow saving / loading settings?

        main_window = QWidget()
        # TODO: How to set main window geometry?
        main_window.setGeometry(100, 100, 500, 250)
        layout = QGridLayout()
        layout.addWidget(QPushButton('Vertical Slice'), 0, 0, 1, 2)
        layout.addWidget(QPushButton('Pie Slice'), 0, 2)
        layout.addWidget(QPushButton('Horizontal Slice'), 1, 0, 1, 2)
        layout.addWidget(QPushButton('Settings'), 1, 2)
        main_window.setLayout(layout)

        self.setCentralWidget(main_window)
        self._createMenu()
        self._createToolBar()
        self._createStatusBar()

        self.show()

        # # waiting for 2 second
        # time.sleep(2)
        #
        # # closing the window
        # self.close()

    def _createMenu(self):
        self.menu = self.menuBar().addMenu("&Menu")
        self.menu.addAction('&Settings', self._launchSettingsDialogue)
        self.menu.addAction('&Exit', self.close)

    def _createToolBar(self):
        tools = QToolBar()
        self.addToolBar(tools)
        tools.addAction('Exit', self.close)

    def _createStatusBar(self):
        status = QStatusBar()
        status.showMessage("I'm the Status Bar")
        self.setStatusBar(status)

    def _launchSettingsDialogue(self):
        #dlg = SettingsDialogue(parent=self)
        dlg = IPSettingsDialog.SettingsDialogue(parent=self)
        dlg.exec_()

    def test(self):
        self.close()


class SettingsDialogue(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('IP:Port Settings')
        dlgLayout = QVBoxLayout()
        formLayout = QFormLayout()
        formLayout.addRow('IP:', QLineEdit())
        formLayout.addRow('Port:', QLineEdit())
        dlgLayout.addLayout(formLayout)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)



        button_box.accepted.connect(self.test)

        dlgLayout.addWidget(button_box)
        self.setLayout(dlgLayout)

    def test(self):
        self.close()
        #pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = WaterColumnGUI()
    #win.show()
    sys.exit(app.exec_())

    # # waiting for 2 second
    # time.sleep(2)
    #
    # # closing the window
    # win.test()





# import sys
# from PyQt5.QtWidgets import QApplication
# from PyQt5.QtWidgets import QLabel
# from PyQt5.QtWidgets import QPushButton
# from PyQt5.QtWidgets import QGridLayout
# from PyQt5.QtWidgets import QWidget
#
# # Create an instance of QApplication
# app = QApplication(sys.argv)
#
# # Create an instance of your application's GUI
# window = QWidget()
# window.setWindowTitle('Water Column Plotter')
# # You can use .setGeometry() to define the size of the window and where to place it on your screen.
# # The first two parameters are the x and y coordinates at which the window will be placed on the screen.
# # The third and fourth parameters are the width and height of the window.
# window.setGeometry(100, 100, 500, 250)
# # window.move(60, 15)
# # helloMsg = QLabel('<h1>Hello World!</h1>', parent=window)
# # helloMsg.move(60, 15)
#
# layout = QGridLayout()
# layout.addWidget(QPushButton('Vertical Slice'), 0, 0, 1, 2)
# layout.addWidget(QPushButton('Pie Slice'), 0, 2)
# layout.addWidget(QPushButton('Horizontal Slice'), 1, 0, 1, 2)
# layout.addWidget(QPushButton('Settings'), 1, 2)
#
# window.setLayout(layout)
#
#
# # Show your application's GUI
# window.show()
#
# # Run your application's event loop (or main loop)
# sys.exit(app.exec_())












# import argparse
# from KongsbergDGCaptureFromSonar import KongsbergDGCaptureFromSonar
# from KongsbergDGPlot import KongsbergDGPlot
# from KongsbergDGProcess import KongsbergDGProcess
# import logging
# import multiprocessing
# import os
# from PyQt5 import QtWidgets
# from pyqtgraph import PlotWidget, plot
# import pyqtgraph as pg
# import sys
#
# logger = logging.getLogger(__name__)


# class MainWindow(QtWidgets.QMainWindow):
#     def __init__(self, *args, **kwargs):
#         super(MainWindow, self).__init__(*args, **kwargs)
#
#         self.graphWidget = pg.PlotWidget()
#         self.setCentralWidget(self.graphWidget)
#
#         hour = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#         temperature = [30, 32, 34, 32, 33, 31, 29, 32, 35, 45]
#
#         # plot data: x, y values
#         self.graphWidget.plot(hour, temperature)
#
#
# def main():
#     app = QtWidgets.QApplication(sys.argv)
#     main = MainWindow()
#     main.show()
#     sys.exit(app.exec_())
#
#
# if __name__ == '__main__':
#     main()


# import time
# import random
# import pyqtgraph as pg
# from collections import deque
# from pyqtgraph.Qt import QtGui, QtCore
#
#
# class Graph:
#     def __init__(self, ):
#         self.dat = deque()
#         self.maxLen = 50  # max number of data points to show on graph
#         self.app = QtGui.QApplication([])
#         self.win = pg.GraphicsWindow()
#
#         self.p1 = self.win.addPlot(colspan=2)
#         self.win.nextRow()
#         self.p2 = self.win.addPlot(colspan=2)
#         self.win.nextRow()
#         self.p3 = self.win.addPlot(colspan=2)
#
#         self.curve1 = self.p1.plot()
#         self.curve2 = self.p2.plot()
#         self.curve3 = self.p3.plot()
#
#         graphUpdateSpeedMs = 50
#         timer = QtCore.QTimer()  # to create a thread that calls a function at intervals
#         timer.timeout.connect(self.update)  # the update function keeps getting called at intervals
#         timer.start(graphUpdateSpeedMs)
#         QtGui.QApplication.instance().exec_()
#
#     def update(self):
#         if len(self.dat) > self.maxLen:
#             self.dat.popleft()  # remove oldest
#         self.dat.append(random.randint(0, 100));
#
#         self.curve1.setData(self.dat)
#         self.curve2.setData(self.dat)
#         self.curve3.setData(self.dat)
#         self.app.processEvents()
#
#
# if __name__ == '__main__':
#     g = Graph()



# class KongsbergDGMain:
#     def __init__(self, rx_ip, rx_port, bin_size, connection="UDP"):
#         self.connection = connection
#         self.rx_ip = rx_ip
#         self.rx_port = rx_port
#
#         self.bin_size = bin_size
#
#         self.queue_data = multiprocessing.Queue()
#         self.queue_pie = multiprocessing.Queue()
#         self.dg_capture = KongsbergDGCaptureFromSonar(rx_ip, rx_port, connection, queue_data=self.queue_data)
#         self.dg_process = KongsbergDGProcess(bin_size=self.bin_size, water_depth=10, max_heave=1, queue_data=self.queue_data,
#                                              queue_pie=self.queue_pie)
#
#         # TODO: Experiment to launch KongsbergDGPlot from WaterColumnGUI.
#         #  Is it better to create KongsbergDGPlot object here and pass it as argument to WaterColumnGUI?
#         #  I think it's OK to do it this way because the format of everything in self.queue_pie
#         #  should be standard regardless of sonar system...
#         self.dg_plot = KongsbergDGPlot(bin_size=self.bin_size, max_heave=1, vertical_slice_width_m=1,
#                                        horizontal_slice_width_m=1, horizontal_slice_depth_m=1, num_pings_to_average=10,
#                                        queue_pie=self.queue_pie)
#         # self.gui = WaterColumnGUI(queue_pie=self.queue_pie)
#
#     def run(self):
#         # TODO: Do I need to set process_consumer daemon value to True?
#         #  https://stonesoupprogramming.com/2017/09/11/python-multiprocessing-producer-consumer-pattern/comment-page-1/
#
#         process_producer = multiprocessing.Process(target=self.dg_capture.receive_dg_and_queue)
#         process_producer.daemon = True
#         process_producer.start()
#         print("producer started")
#
#         process_consumer = multiprocessing.Process(target=self.dg_process.get_and_process_dg)
#         process_consumer.daemon = True
#         process_consumer.start()
#         print("consumer started")
#
#         process_plotter = multiprocessing.Process(target=self.dg_plot.get_and_plot_pie())
#         process_plotter.start()
#         print("plotter started")
#
#         # process_gui = multiprocessing.Process(target=self.gui.run())
#         # #process_gui.daemon = True
#         # process_gui.start()
#         # # TODO: This doesn't print...
#         # print("********************************************************************************************gui started")
#
#         process_producer.join()
#         process_consumer.join()
#         process_plotter.join()
#         # process_gui.join()
#         print("after join")
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument("rx_ip", help="IP address to receive Kongsberg datagrams.")
#     parser.add_argument("rx_port", help="Port to receive Kongsberg datagrams.", type=int)
#     parser.add_argument("--connection", default="UDP", help="Connection type: TCP or UDP.", choices={"TCP", "UDP"})
#     parser.add_argument("bin_size", help="Bin size.", type=float)
#
#     args = parser.parse_args()
#
#     dg_main = KongsbergDGMain(rx_ip=args.rx_ip, rx_port=args.rx_port, connection=args.connection, bin_size=args.bin_size)
#     dg_main.run()
