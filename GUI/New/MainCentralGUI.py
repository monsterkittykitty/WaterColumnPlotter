from PyQt5 import QtCore, QtGui, QtWidgets, uic
from pyqtgraph import PlotWidget
import pyqtgraph as pg
import sys

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        # Load UI page
        uic.loadUi("UIFiles/MainCentralGUI.ui", self)  # This file attempts to use Dock Objects, but is kind of a pain in the ass.


app = QtWidgets.QApplication(sys.argv)
main = MainWindow()
main.show()
sys.exit(app.exec_())