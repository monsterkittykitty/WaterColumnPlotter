# Lynette Davis
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# April 2021

# Description:

from KongsbergDGPlot import KongsbergDGPlot
import sys
from PyQt5 import QtWidgets

class WaterColumnGUI(QtWidgets.QDialog):
    def __init__(self, queue_pie=None, parent=None):
        # From: https://stackoverflow.com/questions/12459811/how-to-embed-matplotlib-in-pyqt-for-dummies
        super(WaterColumnGUI, self).__init__(parent)  # IDK what this does



        # TODO: I think this should initiate the plotting:
        # queue_pie should be multiprocessing queue that pie matrices are added to
        # TODO: Unsure if queue_pie needs to be kept as a variable...
        self.queue_pie = queue_pie
        self.dg_plot = KongsbergDGPlot(queue_pie=self.queue_pie)

        self.app = QtWidgets.QApplication(sys.argv)
        self.window = QtWidgets.QWidget()

        self.layout = QtWidgets.QGridLayout()



    def setup_gui(self):
        self.window.setWindowTitle("Water Column Plotter")
        self.layout.addWidget(QtWidgets.QPushButton("Meow"), 0, 0)
        self.window.setLayout(self.layout)


    def display_gui(self):
        self.window.show()
        self.dg_plot.get_and_plot_pie()
        print("333333333333333333333333333333333333333333333333333333333333333AFTER SYS EXIT")  # Prints
        sys.exit(self.app.exec_())
        print("####################################################################AFTER SYS EXIT")  # Does not print

    def run(self):
        self.setup_gui()
        self.display_gui()

if __name__ == "__main__":
    gui = WaterColumnGUI()
    gui.run()



# import sys
#
# from PyQt5.QtWidgets import QApplication
# from PyQt5.QtWidgets import QHBoxLayout
# from PyQt5.QtWidgets import QPushButton
# from PyQt5.QtWidgets import QWidget
#
# app = QApplication(sys.argv)
# window = QWidget()
# window.setWindowTitle('QHBoxLayout')
# layout = QHBoxLayout()
# layout.addWidget(QPushButton('Left'))
# layout.addWidget(QPushButton('Center'))
# layout.addWidget(QPushButton('Right'))
# window.setLayout(layout)
# window.show()
# sys.exit(app.exec_())


# from PyQt5 import QtWidgets # import PyQt5 widgets
# import sys
#
# # Create the application object
# app = QtWidgets.QApplication(sys.argv)
#
# # Create the form object
# first_window = QtWidgets.QWidget()
#
# # Set window size
# first_window.resize(400, 300)
#
# # Set the form title
# first_window.setWindowTitle("The first pyqt program")
#
# # Show form
# first_window.show()
#
# # Run the program
# sys.exit(app.exec())



