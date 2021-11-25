###
from PyQt5 import QtWidgets, uic
import sys


class IPSettingsDialog(QtWidgets.QDialog):

    def __init__(self, *args, **kwargs):
        super(IPSettingsDialog, self).__init__(*args, **kwargs)

        uic.loadUi('UIFiles/IPSettingsDialog.ui', self)

        self.ip = None
        self.port = None

        self.buttonBox.accepted.connect(self.setIPandPort)

    def setIPandPort(self):
        self.ip = self.lineListenIP.text()
        self.port = self.lineListenPort.text()
        print("Setting IP and port: " + self.ip + ": " + self.port)


