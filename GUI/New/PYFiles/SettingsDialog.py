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
        # print("done")
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

    def validateAndloadValues(self, loadedSettings):
        systemEdited = False
        ipEdited = False
        portEdited = False
        binSizeEdited = False
        acrossTrackAvgEdited = False
        depthAvgEdited = False
        alongTrackAvgEdited = False
        dualSwathPolicyEdited = False

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # System Settings:
        # If system value has changed:
        if loadedSettings["system_settings"]["system"] != self.settings["system_settings"]["system"]:

            # Enable these lines when additional systems supported:
            # self.settings["system_settings"]["system"] = settings["system_settings"]["system"]
            # if settings["system_settings"]["system"] == "Kongsberg":
            #     self.ui.radioButtonKongsberg.setChecked(True)
            # else:
            #     self.ui.radioButtonOther.setChecked(True)
            #     self.ui.lineEditOther.setText(settings["system_settings"]["system"])
            # # self.systemEdited.emit()
            # systemEdited = True

            # Disable these lines when additional systems supported:
            if loadedSettings["system_settings"]["system"] == "Kongsberg":
                self.settings["system_settings"]["system"] = loadedSettings["system_settings"]["system"]
                self.ui.radioButtonKongsberg.setChecked(True)
                # self.systemEdited.emit()
                systemEdited = True
            else:
                self.settings["system_settings"]["system"] = "Kongsberg"
                self.ui.radioButtonKongsberg.setChecked(True)
                QtWidgets.QMessageBox.warning(self, "Warning", "Currently supporting only Kongsberg systems.")

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # IP Settings:
        # If IP value has changed:
        if loadedSettings["ip_settings"]["ip"] != self.settings["ip_settings"]["ip"]:
            try:
                # Check for valid IP address:
                ipaddress.ip_address(settings["ip_settings"]["ip"])
                self.settings["ip_settings"]["ip"] = loadedSettings["ip_settings"]["ip"]
                self.ui.lineEditIP.setText(self.settings["ip_settings"]["ip"])
                # self.ipEdited.emit()
                ipEdited = True
            except ValueError:
                # NOTE: This is activated when nothing or incomplete IP address is entered!
                self.ui.lineEditIP.setText(self.settings["ip_settings"]["ip"])
                QtWidgets.QMessageBox.warning(self, "Warning", "Invalid IP address.")

        # If port value has changed:
        if loadedSettings["ip_settings"]["port"] != self.settings["ip_settings"]["port"]:
            try:
                self.settings["ip_settings"]["port"] = int(loadedSettings["ip_settings"]["port"])
                self.ui.lineEditPort.setText(str(self.settings["ip_settings"]["port"]))
                # self.portEdited.emit()
                portEdited = True
            except ValueError:
                # NOTE: This is still activated when nothing is entered!
                self.ui.lineEditPort.setText(self.settings["ip_settings"]["port"])
                QtWidgets.QMessageBox.warning(self, "Warning", "Invalid port number.")

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Processing Settings:
        # bin_size_m
        if loadedSettings["processing_settings"]["binSize_m"] != self.settings["processing_settings"]["binSize_m"]:
            self.settings["processing_settings"]["binSize_m"] = round(loadedSettings["processing_settings"]["binSize_m"], 2)
            self.ui.doubleSpinBoxBinSize.setValue(self.settings["processing_settings"]["binSize_m"])
            # self.binSizeEdited.emit()
            binSizeEdited = True
        # acrossTrackAvg_m
        if loadedSettings["processing_settings"]["acrossTrackAvg_m"] != self.settings["processing_settings"]["acrossTrackAvg_m"]:
            self.settings["processing_settings"]["acrossTrackAvg_m"] = loadedSettings["processing_settings"]["acrossTrackAvg_m"]
            self.ui.doubleSpinBoxAcrossTrackAvg.setValue(self.settings["processing_settings"]["acrossTrackAvg_m"])
            # self.acrossTrackAvgEdited.emit()
            acrossTrackAvgEdited = True
        # depthAvg_m:
        if loadedSettings["processing_settings"]["depthAvg_m"] != self.settings["processing_settings"]["depthAvg_m"]:
            self.settings["processing_settings"]["depthAvg_m"] = loadedSettings["processing_settings"]["depthAvg_m"]
            self.ui.doubleSpinBoxDepthAvg.setValue(self.settings["processing_settings"]["depthAvg_m"])
            # self.depthAvgEdited.emit()
            depthAvgEdited = True
        # alongTrackAvg_ping:
        if loadedSettings["processing_settings"]["alongTrackAvg_ping"] != self.settings["processing_settings"]["alongTrackAvg_ping"]:
            self.settings["processing_settings"]["alongTrackAvg_ping"] = loadedSettings["processing_settings"]["alongTrackAvg_ping"]
            self.ui.doubleSpinBoxAlongTrackAvg.setValue(self.settings["processing_settings"]["alongTrackAvg_ping"])
            # self.alongTrackAvgEdited.emit()
            alongTrackAvgEdited = True
        # dualSwathPolicy (0 to keep all pings; 1 to keep first ping only; 2 to keep second ping only):
        if loadedSettings["processing_settings"]["dualSwathPolicy"] != self.settings["processing_settings"]["dualSwathPolicy"]:
            self.settings["processing_settings"]["dualSwathPolicy"] = loadedSettings["processing_settings"]["dualSwathPolicy"]
            if self.settings["processing_settings"]["dualSwathPolicy"] == 0:
                self.ui.radioButtonAllPings.setChecked(True)
            elif self.settings["processing_settings"]["dualSwathPolicy"] == 1:
                self.ui.radioButtonFirstPing.setChecked(True)
            elif self.settings["processing_settings"]["dualSwathPolicy"] == 2:
                self.ui.radioButtonSecondPing.setChecked(True)
            # self.dualSwathPolicyEdited.emit()
            dualSwathPolicyEdited = True

        # Only emit signals after all values in dictionary have been updated:
        self.emitSignals(systemEdited, ipEdited, portEdited, binSizeEdited, acrossTrackAvgEdited, depthAvgEdited,
                         alongTrackAvgEdited, dualSwathPolicyEdited)

    def validateAndSetValues(self):
        #print("validateAndSetValues")
        systemEdited = False
        ipEdited = False
        portEdited = False
        binSizeEdited = False
        acrossTrackAvgEdited = False
        depthAvgEdited = False
        alongTrackAvgEdited = False
        dualSwathPolicyEdited = False


        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # System Settings:
        # If system value has changed:
        if self.ui.radioButtonKongsberg.isChecked() and self.settings["system_settings"]["system"] != "Kongsberg":
            self.settings["system_settings"]["system"] = "Kongsberg"
            # self.systemEdited.emit()
            systemEdited = True

        # Enable these lines when additional systems supported:
        # elif self.ui.radioButtonOther.isChecked() and self.settings["system_settings"]["system"] != self.ui.lineEditOther.text():
            # self.settings["system_settings"]["system"] = self.ui.lineEditOther.text()
            # # self.systemEdited.emit()
            # systemEdited = True

        # Disable these lines when additional systems supported:
        elif self.ui.radioButtonOther.isChecked():
            self.ui.radioButtonKongsberg.setChecked(True)
            self.ui.lineEditOther.clear()
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
                # self.ipEdited.emit()
                ipEdited = True
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
                # self.portEdited.emit()
                portEdited = True
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
            # self.binSizeEdited.emit()
            binSizeEdited = True
        # acrossTrackAvg_m
        if self.settings["processing_settings"]["acrossTrackAvg_m"] != self.ui.doubleSpinBoxAcrossTrackAvg.value():
            self.settings["processing_settings"]["acrossTrackAvg_m"] = self.ui.doubleSpinBoxAcrossTrackAvg.value()
            # self.acrossTrackAvgEdited.emit()
            acrossTrackAvgEdited = True
        # depthAvg_m:
        if self.settings["processing_settings"]["depthAvg_m"] != self.ui.doubleSpinBoxDepthAvg.value():
            self.settings["processing_settings"]["depthAvg_m"] = self.ui.doubleSpinBoxDepthAvg.value()
            # self.depthAvgEdited.emit()
            depthAvgEdited = True
        # alongTrackAvg_ping:
        if self.settings["processing_settings"]["alongTrackAvg_ping"] != self.ui.spinBoxAlongTrackAvg.value():
            self.settings["processing_settings"]["alongTrackAvg_ping"] = self.ui.spinBoxAlongTrackAvg.value()
            # self.alongTrackAvgEdited.emit()
            alongTrackAvgEdited = True
        # dualSwathPolicy (0 to keep all pings; 1 to keep first ping only; 2 to keep second ping only):
        if self.radioButtonAllPings.isChecked() and self.settings["processing_settings"]["dualSwathPolicy"] != 0:
            self.settings["processing_settings"]["dualSwathPolicy"] = 0
            # self.dualSwathPolicyEdited.emit()
            dualSwathPolicyEdited = True
        elif self.radioButtonFirstPing.isChecked() and self.settings["processing_settings"]["dualSwathPolicy"] != 1:
            self.settings["processing_settings"]["dualSwathPolicy"] = 1
            #self.dualSwathPolicyEdited.emit()
            dualSwathPolicyEdited = True
        elif self.radioButtonSecondPing.isChecked() and self.settings["processing_settings"]["dualSwathPolicy"] != 2:
            self.settings["processing_settings"]["dualSwathPolicy"] = 2
            #self.dualSwathPolicyEdited.emit()
            dualSwathPolicyEdited = True

        # Only emit signals after all values in dictionary have been updated:
        self.emitSignals(systemEdited, ipEdited, portEdited, binSizeEdited, acrossTrackAvgEdited, depthAvgEdited,
                    alongTrackAvgEdited, dualSwathPolicyEdited)

        return True

    def emitSignals(self, systemEdited, ipEdited, portEdited, binSizeEdited, acrossTrackAvgEdited, depthAvgEdited,
                    alongTrackAvgEdited, dualSwathPolicyEdited):
        if systemEdited:
            self.systemEdited.emit()
        if ipEdited:
            self.ipEdited.emit()
        if portEdited:
            self.portEdited.emit()
        if binSizeEdited:
            self.binSizeEdited.emit()
        if acrossTrackAvgEdited:
            self.acrossTrackAvgEdited.emit()
        if depthAvgEdited:
            self.depthAvgEdited.emit()
        if alongTrackAvgEdited:
            self.alongTrackAvgEdited.emit()
        if dualSwathPolicyEdited:
            self.dualSwathPolicyEdited.emit()
