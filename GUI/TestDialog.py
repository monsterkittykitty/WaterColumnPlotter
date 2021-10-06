from PyQt5 import QtCore, QtGui, QtWidgets, uic
import sys


class TestDialog(QtWidgets.QDialog):

    def __init__(self, *args, **kwargs):
        super(TestDialog, self).__init__(*args, **kwargs)

        uic.loadUi('testdialog.ui', self)

        self.buttonBox.accepted.connect(self.setIPandPort)

    def setIPandPort(self):
        print("hi")
        print("Setting IP and port: " + self.lineListenIP.text() +
              ":" + self.lineListenPort.text())


