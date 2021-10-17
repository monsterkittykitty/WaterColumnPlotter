from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QIntValidator, QRegExpValidator
from PyQt5.QtCore import pyqtSignal, QRegExp
import sys

import ipaddress

__appname__ = "Settings Dialog"

class SettingsDialog(QtWidgets.QDialog):

    signalSystemEdited = pyqtSignal(name="systemEdited")
    signalIPEdited = pyqtSignal(name="ipEdited")
    signalPortEdited = pyqtSignal(name="portEdited")
    signalBinSizeEdited = pyqtSignal(name="binSizeEdited")
    signalAcrossTrackAvgEdited = pyqtSignal(name="acrossTrackAvgEdited")
    signalDepthAvgEdited = pyqtSignal(name="depthAvgEdited")
    signalAlongTrackAvgEdited = pyqtSignal(name="alongTrackAvgEdited")
    signalDualSwathPolicyEdited = pyqtSignal(name="dualSwathPolicyEdited")

    # def __init__(self, *args, **kwargs):
    #     super(SettingsDialog, self).__init__(*args, **kwargs)
    def __init__(self, settings, parent=None):
        super(SettingsDialog, self).__init__(parent)

        self.settings = settings
        self.parent = parent

        self.ui = uic.loadUi('UIFiles/SettingsDialog.ui', self)

        # Preventative validation:
        self.setValidators()

        #self.ui.buttonBox.accepted.connect(self.validateAndSetValues)
        #self.ui.lineEditIP.editingFinished.connect(self.bye)
        #self.ui.lineEditIP.returnPressed.connect(self.hi)

    def setValidators(self):
        # Preventative validation:

        # IP
        ipRange = "(?:[0-1]?[0-9]?[0-9]|2[0-4][0-9]|25[0-5])"
        ipRegex = QRegExp("^" + ipRange + "\\." + ipRange + "\\." + ipRange + "\\." + ipRange + "$")
        ipValidator = QRegExpValidator(ipRegex, self)
        self.ui.lineEditIP.setValidator(ipValidator)

        # Port
        # TODO: For some reason this is allowing 5-digit integers greater than 65535...
        portValidator = QIntValidator(1, 65535, self)
        self.ui.lineEditPort.setValidator(portValidator)

    def done(self, i):
        """
        Overriding done() method of QDialog.
        Based on this solution: https://www.qtcentre.org/threads/8048-Validate-Data-in-QDialog
        :param i: 0 when 'Cancel' clicked; 1 when 'OK' clicked.
        """
        print("done")
        if QtWidgets.QDialog.Accepted == i:  # OK clicked
            if self.validateAndSetValues():
                QtWidgets.QDialog.done(self, i)
            else:
                pass
        else:  # Cancel / Close clicked
            QtWidgets.QDialog.done(self, i)

    def hi(self):
        print("HI")

    def bye(self):
        print("BYE")

    def validateAndSetValues(self):
        print("validateAndSetValues")

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # System Settings:
        # If system value has changed:
        if self.ui.radioButtonKongsberg.isChecked() and self.settings["system_settings"]["system"] != "Kongsberg":
            self.settings["system_settings"]["system"] = "Kongsberg"
            self.systemEdited.emit()
        # Enable these lines when additional systems supported:
        # elif self.ui.radioButtonOther.isChecked() and self.settings["system_settings"]["system"] != self.ui.lineEditOther.text():
            # self.settings["system_settings"]["system"] = self.ui.lineEditOther.text()
            # self.systemEdited.emit()
        # Disable these lines when additional systems supported:
        elif self.ui.radioButtonOther.isChecked():
            QtWidgets.QMessageBox.warning(self, "Warning", "Currently supporting only Kongsberg systems.")
            return False

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # IP Settings:
        # If IP value has changed:
        if self.settings["ip_settings"]["ip"] != self.ui.lineEditIP.text():
            try:
                # Check for valid IP address:
                ipaddress.ip_address(self.lineEditIP.text())
                self.settings["ip_settings"]["ip"] = self.lineEditIP.text()
                self.ipEdited.emit()
            except ValueError:
                # (NOTE: This should be avoided with preventative validation.)
                # TODO: (NOTE: This is still activated when nothing or incomplete IP address is entered!)
                self.ui.lineEditIP.clear()
                self.ui.lineEditIP.setText(self.settings["ip_settings"]["ip"])
                #self.ui.lineEditIP.setPlaceholderText(str(self.settings["ip_settings"]["ip"]))
                QtWidgets.QMessageBox.warning(self, "Warning", "Enter valid IP address.")
                return False

        # If port value has changed:
        if self.settings["ip_settings"]["port"] != self.ui.lineEditPort.text():
            try:
                self.settings["ip_settings"]["port"] = int(self.lineEditPort.text())
                self.portEdited.emit()
            except ValueError:
                # (NOTE: This should be avoided with preventative validation.)
                # TODO: (NOTE: This is still activated when nothing is entered!)
                self.ui.lineEditPort.clear()
                self.ui.lineEditPort.setText(self.settings["ip_settings"]["port"])
                #self.ui.lineEditPort.setPlaceholderText(str(self.settings["ip_settings"]["port"]))
                QtWidgets.QMessageBox.warning(self, "Warning", "Enter valid port number.")
                return False

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Processing Settings:
        # bin_size_m
        if self.settings["processing_settings"]["binSize_m"] != self.ui.doubleSpinBoxBinSize.value():
            self.settings["processing_settings"]["binSize_m"] = round(self.ui.doubleSpinBoxBinSize.value(), 2)
            self.binSizeEdited.emit()
        # acrossTrackAvg_m
        if self.settings["processing_settings"]["acrossTrackAvg_m"] != self.ui.doubleSpinBoxAcrossTrackAvg.value():
            self.settings["processing_settings"]["acrossTrackAvg_m"] = self.ui.doubleSpinBoxAcrossTrackAvg.value()
            self.acrossTrackAvgEdited.emit()
        # depthAvg_m:
        if self.settings["processing_settings"]["depthAvg_m"] != self.ui.doubleSpinBoxDepthAvg.value():
            self.settings["processing_settings"]["depthAvg_m"] = self.ui.doubleSpinBoxDepthAvg.value()
            self.depthAvgEdited.emit()
        # alongTrackAvg_ping:
        if self.settings["processing_settings"]["alongTrackAvg_ping"] != self.ui.spinBoxAlongTrackAvg.value():
            self.settings["processing_settings"]["alongTrackAvg_ping"] = self.ui.spinBoxAlongTrackAvg.value()
            self.alongTrackAvgEdited.emit()
        # dualSwathPolicy (0 to keep all pings; 1 to keep first ping only; 2 to keep second ping only):
        if self.radioButtonAllPings.isChecked() and self.settings["processing_settings"]["dualSwathPolicy"] != 0:
            self.settings["processing_settings"]["dualSwathPolicy"] = 0
            self.dualSwathPolicyEdited.emit()
        elif self.radioButtonFirstPing.isChecked() and self.settings["processing_settings"]["dualSwathPolicy"] != 1:
            self.settings["processing_settings"]["dualSwathPolicy"] = 1
            self.dualSwathPolicyEdited.emit()
        elif self.radioButtonSecondPing.isChecked() and self.settings["processing_settings"]["dualSwathPolicy"] != 2:
            self.settings["processing_settings"]["dualSwathPolicy"] = 2
            self.dualSwathPolicyEdited.emit()

        print(self.doubleSpinBoxBinSize.value())
        print(round(self.doubleSpinBoxBinSize.value(), self.doubleSpinBoxBinSize.decimals()))
        print(self.doubleSpinBoxAcrossTrackAvg.value())
        print(self.spinBoxAlongTrackAvg.value())
        print(self.radioButtonAllPings.isChecked())

        return True
