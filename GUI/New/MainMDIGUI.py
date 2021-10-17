###
from PyQt5 import QtWidgets, uic
import sys

from GUI.New.PYFiles.IPSettingsDialog import IPSettingsDialog
from GUI.New.PYFiles.SettingsDialog import SettingsDialog


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        uic.loadUi('UIFiles/MainMDIGUI.ui', self)

        #self.subwindow_3.show()

        self.actionIP_Settings.triggered.connect(self.displayIPSettingsDialog)
        print("hellp")
        self.actionSettings.triggered.connect(self.displaySettingsDialog)

    def displayIPSettingsDialog(self):
        self.ipSettingsDialog = IPSettingsDialog()
        self.ipSettingsDialog.exec_()

    def displaySettingsDialog(self):
        self.settingsDialog = SettingsDialog()
        self.settingsDialog.exec_()


def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()

    main.subwindowVertical.show()
    main.subwindowPie.show()
    main.subwindowHorizontal.show()

    main.displaySettingsDialog()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()