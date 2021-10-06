from PyQt5 import QtCore, QtGui, QtWidgets, uic
import sys

from TestDialog import TestDialog


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        uic.loadUi('testmain.ui', self)

        self.actionIP_Settings.triggered.connect(self.printhi)

    def printhi(self):
        self.newTestDialog = TestDialog()
        self.newTestDialog.exec_()


def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()