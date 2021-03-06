# Lynette Davis
# ldavis@ccom.unh.edu
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# November 2021

# Description: Provides functions to update internal settings through a settings dialog.

import ipaddress
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QIntValidator, QRegExpValidator
from PyQt5.QtCore import pyqtSignal, QRegExp

__appname__ = "Settings Dialog"


class AllSettingsDialog(QtWidgets.QDialog):

    signalSystemEdited = pyqtSignal(name="systemEdited")
    signalIPEdited = pyqtSignal(name="ipEdited")
    signalPortEdited = pyqtSignal(name="portEdited")
    signalProtocolEdited = pyqtSignal(name="protocolEdited")
    signalSocketBufferEdited = pyqtSignal(name="socketBufferEdited")
    signalBinSizeEdited = pyqtSignal(name="binSizeEdited")
    signalAcrossTrackAvgEdited = pyqtSignal(name="acrossTrackAvgEdited")
    signalDepthEdited = pyqtSignal(name="depthEdited")
    signalDepthAvgEdited = pyqtSignal(name="depthAvgEdited")
    signalAlongTrackAvgEdited = pyqtSignal(name="alongTrackAvgEdited")
    signalHeaveEdited = pyqtSignal(name="heaveEdited")
    signalGridCellsEdited = pyqtSignal(name="gridCellsEdited")
    signalPingBufferEdited = pyqtSignal(name="pingBufferEdited")
    signalsettingsEdited = pyqtSignal(name="settingsEdited")

    def __init__(self, settings, parent=None):
        super(AllSettingsDialog, self).__init__(parent)

        self.settings = settings

        self.ui = uic.loadUi('Dialogs/UIFiles/AllSettingsDialog.ui', self)

        self.setValues()

        # Preventative validation:
        self.VALID_PORT_MIN = 1
        self.VALID_PORT_MAX = 65535
        self.setValidators()

    def setValidators(self):
        """
        Enables preventative validation to ensure valid entries in select fields of setting dialog.
        """

        # IP
        ipRange = "(?:[0-1]?[0-9]?[0-9]|2[0-4][0-9]|25[0-5])"
        ipRegex = QRegExp("^" + ipRange + "\\." + ipRange + "\\." + ipRange + "\\." + ipRange + "$")
        ipValidator = QRegExpValidator(ipRegex, self)
        self.ui.lineEditIP.setValidator(ipValidator)

        # Port
        # TODO: For some reason this is allowing 5-digit integers greater than 65535...
        portValidator = QIntValidator(self.VALID_PORT_MIN, self.VALID_PORT_MAX, self)
        self.ui.lineEditPort.setValidator(portValidator)

    def done(self, i):
        """
        Overriding done() method of QDialog.
        Based on this solution: https://www.qtcentre.org/threads/8048-Validate-Data-in-QDialog
        :param i: 0 when 'Cancel' clicked; 1 when 'OK' clicked.
        """
        if QtWidgets.QDialog.Accepted == i:  # OK clicked
            if self.validateAndSetValuesFromDialog():
                QtWidgets.QDialog.done(self, i)
            else:
                pass
        else:  # Cancel / Close clicked
            QtWidgets.QDialog.done(self, i)

    def setValues(self):
        """
        Assigns values in self.settings (current settings) to fields in settings dialog.
        *** NOTE: There is no error checking in this method!
        If settings are changed, make sure they are correct / valid at time of change! ***
        """
        # System Settings:
        if self.settings['system_settings']['system'] == "Kongsberg":
            self.ui.radioButtonKongsberg.setChecked(True)
        else:
            self.ui.radioButtonOther.setChecked(True)
            self.ui.lineEditOther.setText(self.settings['system_settings']['system'])

        # IP Settings:
        self.ui.lineEditIP.setText(self.settings['ip_settings']['ip'])
        self.ui.lineEditPort.setText(str(self.settings['ip_settings']['port']))
        if self.settings['ip_settings']['protocol'] == "UDP":
            self.ui.radioButtonUDP.setChecked(True)
        else:  # Multicast
            self.ui.radioButtonMulticast.setChecked(True)
        self.ui.spinBoxSocketBuffer.setValue(int(self.settings['ip_settings']['socketBufferMultiplier']))

        # Processing Settings:
        self.ui.doubleSpinBoxBinSize.setValue(round(self.settings['processing_settings']['binSize_m'], 2))
        self.ui.doubleSpinBoxAcrossTrackAvg.setValue(round(self.settings['processing_settings']['acrossTrackAvg_m'], 2))
        self.ui.doubleSpinBoxDepth.setValue(round(self.settings['processing_settings']['depth_m'], 2))
        self.ui.doubleSpinBoxDepthAvg.setValue(round(self.settings['processing_settings']['depthAvg_m'], 2))
        self.ui.spinBoxAlongTrackAvg.setValue(int(self.settings['processing_settings']['alongTrackAvg_ping']))
        self.ui.doubleSpinBoxMaxHeave.setValue(round(self.settings['processing_settings']['maxHeave_m'], 2))

        # Buffer Settings
        self.ui.spinBoxMaxGridCells.setValue(int(self.settings['buffer_settings']['maxGridCells']))
        self.ui.spinBoxMaxPingBuffer.setValue(int(self.settings['buffer_settings']['maxBufferSize_ping']))

    def validateAndSetValuesFromFile(self, loadSettings):
        """
        Assigns values in loadSettings to fields in settings dialog and self.settings; emits signals for changed fields.
        :param loadedSettings: A python dictionary of the same format as self.settings,
        containing settings loaded from a file.
        """
        systemEdited = False
        ipEdited = False
        portEdited = False
        protocolEdited = False
        socketBufferEdited = False
        binSizeEdited = False
        acrossTrackAvgEdited = False
        depthEdited = False
        depthAvgEdited = False
        alongTrackAvgEdited = False
        heaveEdited = False
        gridCellsEdited = False
        pingBufferEdited = False

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # System Settings:

        # If system value has changed:
        if self.settings['system_settings']['system'] != loadSettings['system_settings']['system']:

            # TODO: Enable these lines when additional systems supported:
            # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
            # self.settings['system_settings']['system'] = loadSettings['system_settings']['system']
            # if self.settings['system_settings']['system'] == "Kongsberg":
            #     self.ui.radioButtonKongsberg.setChecked(True)
            # else:
            #     self.ui.radioButtonOther.setChecked(True)
            #     self.ui.lineEditOther.setText(self.settings['system_settings']['system'])
            # systemEdited = True
            # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

            # TODO: Disable these lines when additional systems supported:
            # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
            if loadSettings['system_settings']['system'] == "Kongsberg":
                self.settings['system_settings']['system'] = loadSettings['system_settings']['system']
                self.ui.radioButtonKongsberg.setChecked(True)
                systemEdited = True
            else:
                self.settings['system_settings']['system'] = "Kongsberg"
                self.ui.radioButtonKongsberg.setChecked(True)
                QtWidgets.QMessageBox.warning(self, "Warning", "Currently supporting only Kongsberg systems.")
            # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # IP Settings:

        # If IP value has changed:
        if self.settings['ip_settings']['ip'] != loadSettings['ip_settings']['ip']:
            try:
                # Check for valid IP address:
                ipaddress.ip_address(loadSettings['ip_settings']['ip'])
                self.settings['ip_settings']['ip'] = loadSettings['ip_settings']['ip']
                self.ui.lineEditIP.setText(self.settings['ip_settings']['ip'])
                ipEdited = True
            except ValueError:
                # NOTE: This is activated when nothing or incomplete IP address is entered!
                self.ui.lineEditIP.setText(self.settings['ip_settings']['ip'])
                QtWidgets.QMessageBox.warning(self, "Warning", "Invalid IP address."
                                                               "\nIP reset to {}."
                                              .format(self.settings['ip_settings']['ip']))

        # If port value has changed:
        if self.settings['ip_settings']['port'] != int(loadSettings['ip_settings']['port']):
            try:
                self.settings['ip_settings']['port'] = int(loadSettings['ip_settings']['port'])
                self.ui.lineEditPort.setText(str(self.settings['ip_settings']['port']))
                portEdited = True
            except ValueError:
                # NOTE: This is still activated when nothing is entered!
                self.ui.lineEditPort.setText(str(self.settings['ip_settings']['port']))
                QtWidgets.QMessageBox.warning(self, "Warning", "Invalid port number."
                                                               "\nPort reset to {}."
                                              .format(str(self.settings['ip_settings']['port'])))

        # If IP protocol has changed:
        if self.settings['ip_settings']['protocol'] != loadSettings['ip_settings']['protocol']:
            self.settings['ip_settings']['protocol'] = loadSettings['ip_settings']['protocol']
            if self.settings['ip_settings']['protocol'] == "UDP":
                self.ui.radioButtonUDP.setChecked(True)
            else:
                self.ui.radioButtonMulticast.setChecked(True)
            protocolEdited = True

        # If socket buffer multiplier has changed:
        if self.settings['ip_settings']['socketBufferMultiplier'] != \
                int(loadSettings['ip_settings']['socketBufferMultiplier']):
            self.settings['ip_settings']['socketBufferMultiplier'] = \
                int(loadSettings['ip_settings']['socketBufferMultiplier'])
            self.ui.spinBoxSocketBuffer.setValue(self.settings['ip_settings']['socketBufferMultiplier'])
            socketBufferEdited = True

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Processing Settings:

        # bin_size_m
        if self.settings['processing_settings']['binSize_m'] != \
                loadSettings['processing_settings']['binSize_m']:

            self.settings['processing_settings']['binSize_m'] = \
                loadSettings['processing_settings']['binSize_m']
            self.ui.doubleSpinBoxBinSize.setValue(self.settings['processing_settings']['binSize_m'])
            binSizeEdited = True

        # acrossTrackAvg_m
        if self.settings['processing_settings']['acrossTrackAvg_m'] != \
                loadSettings['processing_settings']['acrossTrackAvg_m']:

            self.settings['processing_settings']['acrossTrackAvg_m'] = \
                loadSettings['processing_settings']['acrossTrackAvg_m']
            self.ui.doubleSpinBoxAcrossTrackAvg.setValue(self.settings['processing_settings']['acrossTrackAvg_m'])
            acrossTrackAvgEdited = True

        if self.settings['processing_settings']['depth_m'] != \
                loadSettings['processing_settings']['depth_m']:

            self.settings['processing_settings']['depth_m'] = \
                loadSettings['processing_settings']['depth_m']
            self.ui.doubleSpinBoxDepth.setValue(self.settings['processing_settings']['depth_m'])
            depthEdited = True

        # depthAvg_m:
        if self.settings['processing_settings']['depthAvg_m'] != \
                loadSettings['processing_settings']['depthAvg_m']:

            self.settings['processing_settings']['depthAvg_m'] = \
                loadSettings['processing_settings']['depthAvg_m']
            self.ui.doubleSpinBoxDepthAvg.setValue(self.settings['processing_settings']['depthAvg_m'])
            depthAvgEdited = True

        # alongTrackAvg_ping:
        if self.settings['processing_settings']['alongTrackAvg_ping'] != \
                loadSettings['processing_settings']['alongTrackAvg_ping']:

            self.settings['processing_settings']['alongTrackAvg_ping'] = \
                loadSettings['processing_settings']['alongTrackAvg_ping']
            self.ui.doubleSpinBoxAlongTrackAvg.setValue(self.settings['processing_settings']['alongTrackAvg_ping'])
            alongTrackAvgEdited = True

        # maxHeave_m
        if self.settings['processing_settings']['maxHeave_m'] != loadSettings['processing_settings']['maxHeave_m']:
            self.settings['processing_settings']['maxHeave_m'] = loadSettings['processing_settings']['maxHeave_m']
            self.ui.doubleSpinBoxMaxHeave.setValue(round(self.settings['processing_settings']['maxHeave_m'], 2))
            heaveEdited = True

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Buffer Settings:

        # maxGridCells
        if self.settings['buffer_settings']['maxGridCells'] != loadSettings['buffer_settings']['maxGridCells']:
            self.settings['buffer_settings']['maxGridCells'] = loadSettings['buffer_settings']['maxGridCells']
            self.ui.spinBoxMaxGridCells.setValue(int(self.settings['buffer_settings']['maxGridCells']))
            gridCellsEdited = True

        # maxBufferSize_ping
        if self.settings['buffer_settings']['maxBufferSize_ping'] != \
                loadSettings['buffer_settings']['maxBufferSize_ping']:
            self.settings['buffer_settings']['maxBufferSize_ping'] = \
                loadSettings['buffer_settings']['maxBufferSize_ping']
            self.ui.spinBoxMaxPingBuffer.setValue(int(self.settings['buffer_settings']['maxBufferSize_ping']))
            pingBufferEdited = True

        # Only emit signals after all values in dictionary have been updated:
        self.emitSignals(systemEdited, ipEdited, portEdited, protocolEdited, socketBufferEdited, binSizeEdited,
                         acrossTrackAvgEdited, depthEdited, depthAvgEdited, alongTrackAvgEdited, heaveEdited,
                         gridCellsEdited, pingBufferEdited)

    def validateAndSetValuesFromDialog(self):
        """
        Assigns values in settings dialog to self.settings; emits signals for changed fields.
        """
        systemEdited = False
        ipEdited = False
        portEdited = False
        protocolEdited = False
        socketBufferEdited = False
        binSizeEdited = False
        acrossTrackAvgEdited = False
        depthEdited = False
        depthAvgEdited = False
        alongTrackAvgEdited = False
        heaveEdited = False
        gridCellsEdited = False
        pingBufferEdited = False

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # System Settings:
        # If system value has changed:
        if self.ui.radioButtonKongsberg.isChecked() and self.settings['system_settings']['system'] != "Kongsberg":
            self.settings['system_settings']['system'] = "Kongsberg"
            systemEdited = True

        # TODO: Enable these lines when additional systems supported:
        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        # elif self.ui.radioButtonOther.isChecked() and self.settings['system_settings']['system'] != self.ui.lineEditOther.text():
            # self.settings['system_settings']['system'] = self.ui.lineEditOther.text()
            # systemEdited = True
        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

        # TODO: Disable these lines when additional systems supported:
        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        elif self.ui.radioButtonOther.isChecked():
            self.ui.radioButtonKongsberg.setChecked(True)
            QtWidgets.QMessageBox.warning(self, "Warning", "Currently supporting only Kongsberg systems.")
            return False
        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # IP Settings:
        # If IP value has changed:
        if self.settings['ip_settings']['ip'] != self.ui.lineEditIP.text():
            try:
                # Check for valid IP address:
                ipaddress.ip_address(self.ui.lineEditIP.text())
                self.settings['ip_settings']['ip'] = self.ui.lineEditIP.text()
                ipEdited = True
            except ValueError:
                # (NOTE: This should be avoided with preventative validation.)
                # TODO: (NOTE: This is still activated when nothing or incomplete IP address is entered!)
                self.ui.lineEditIP.setText(self.settings['ip_settings']['ip'])
                QtWidgets.QMessageBox.warning(self, "Warning", "Invalid IP address."
                                                               "\nIP reset to {}."
                                              .format(self.settings['ip_settings']['ip']))
                return False

        # If port value has changed:
        if self.settings['ip_settings']['port'] != int(self.ui.lineEditPort.text()):
            try:
                self.settings['ip_settings']['port'] = int(self.ui.lineEditPort.text())
                portEdited = True
            except ValueError:
                # (NOTE: This should be avoided with preventative validation.)
                # TODO: (NOTE: This is still activated when nothing is entered!)
                self.ui.lineEditPort.setText(str(self.settings['ip_settings']['port']))
                QtWidgets.QMessageBox.warning(self, "Warning", "Invalid port number."
                                                               "\nPort reset to {}."
                                              .format(str(self.settings['ip_settings']['port'])))
                return False

        # If IP protocol has changed:
        if self.ui.radioButtonUDP.isChecked() and self.settings['ip_settings']['protocol'] != "UDP":
            self.settings['ip_settings']['protocol'] = "UDP"
            protocolEdited = True
        elif self.ui.radioButtonMulticast.isChecked() and self.settings['ip_settings']['protocol'] != "Multicast":
            self.settings['ip_settings']['protocol'] = "Multicast"
            protocolEdited = True

        # If socket buffer multiplier has changed:
        if self.settings['ip_settings']['socketBufferMultiplier'] != self.ui.spinBoxSocketBuffer.value():
            self.settings['ip_settings']['socketBufferMultiplier'] = self.ui.spinBoxSocketBuffer.value()
            socketBufferEdited = True

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Processing Settings:

        # bin_size_m
        if self.settings['processing_settings']['binSize_m'] != self.ui.doubleSpinBoxBinSize.value():
            self.settings['processing_settings']['binSize_m'] = \
                round(self.ui.doubleSpinBoxBinSize.value(), 2)
            binSizeEdited = True

        # acrossTrackAvg_m
        if self.settings['processing_settings']['acrossTrackAvg_m'] != self.ui.doubleSpinBoxAcrossTrackAvg.value():
            self.settings['processing_settings']['acrossTrackAvg_m'] = \
                round(self.ui.doubleSpinBoxAcrossTrackAvg.value(), 2)
            acrossTrackAvgEdited = True

        # depth_m:
        if self.settings['processing_settings']['depth_m'] != self.ui.doubleSpinBoxDepth.value():
            self.settings['processing_settings']['depth_m'] = \
                round(self.ui.doubleSpinBoxDepth.value(), 2)
            depthEdited = True

        # depthAvg_m:
        if self.settings['processing_settings']['depthAvg_m'] != self.ui.doubleSpinBoxDepthAvg.value():
            self.settings['processing_settings']['depthAvg_m'] = \
                round(self.ui.doubleSpinBoxDepthAvg.value(), 2)
            depthAvgEdited = True

        # alongTrackAvg_ping:
        if self.settings['processing_settings']['alongTrackAvg_ping'] != self.ui.spinBoxAlongTrackAvg.value():
            self.settings['processing_settings']['alongTrackAvg_ping'] = \
                int(self.ui.spinBoxAlongTrackAvg.value())
            alongTrackAvgEdited = True

        # maxHeave_m
        if self.settings['processing_settings']['maxHeave_m'] != self.ui.doubleSpinBoxMaxHeave.value():
            self.settings['processing_settings']['maxHeave_m'] = round(self.ui.doubleSpinBoxMaxHeave.value(), 2)
            heaveEdited = True

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Buffer Settings:

        # maxGridCells
        if self.settings['buffer_settings']['maxGridCells'] != self.ui.spinBoxMaxGridCells.value():
            self.settings['buffer_settings']['maxGridCells'] = int(self.ui.spinBoxMaxGridCells.value())
            gridCellsEdited = True

        # maxBufferSize_ping
        if self.settings['buffer_settings']['maxBufferSize_ping'] != self.ui.spinBoxMaxPingBuffer.value():
            self.settings['buffer_settings']['maxBufferSize_ping'] = int(self.ui.spinBoxMaxPingBuffer.value())
            pingBufferEdited = True

        # Only emit signals after all values in dictionary have been updated:
        self.emitSignals(systemEdited, ipEdited, portEdited, protocolEdited, socketBufferEdited, binSizeEdited,
                         acrossTrackAvgEdited, depthEdited, depthAvgEdited, alongTrackAvgEdited, heaveEdited,
                         gridCellsEdited, pingBufferEdited)

        return True

    def emitSignals(self, systemEdited, ipEdited, portEdited, protocolEdited, socketBufferEdited, binSizeEdited,
                    acrossTrackAvgEdited, depthEdited, depthAvgEdited, alongTrackAvgEdited, heaveEdited,
                    gridCellsEdited, pingBufferEdited):
        """
        Emits signals for all True parameters.
        :param systemEdited: Boolean indicating whether system field was edited.
        :param ipEdited: Boolean indicating whether IP field was edited.
        :param portEdited: Boolean indicating whether port field was edited.
        :param protocolEdited: Boolean indicating whether protocol field was edited.
        :param binSizeEdited: Boolean indicating whether binSize field was edited.
        :param acrossTrackAvgEdited: Boolean indicating whether acrossTrackAvg field was edited.
        :param depthEdited: Boolean indicating whether depth field was edited.
        :param depthAvgEdited: Boolean indicating whether depthAvg field was edited.
        :param alongTrackAvgEdited: Boolean indicating whether alongTrackAvg field was edited.
        :param heaveEdited: Boolean indicating whether heave field was edited.
        :param gridCellsEdited: Boolean indicating whether gridCells field was edited.
        :param pingBufferEdited: Boolean indicating whether pingBuffer field was edited.
        """
        if systemEdited:
            self.systemEdited.emit()
        if ipEdited:
            self.ipEdited.emit()
        if portEdited:
            self.portEdited.emit()
        if protocolEdited:
            self.protocolEdited.emit()
        if socketBufferEdited:
            self.socketBufferEdited.emit()
        if binSizeEdited:
            self.binSizeEdited.emit()
        if acrossTrackAvgEdited:
            self.acrossTrackAvgEdited.emit()
        if depthEdited:
            self.depthEdited.emit()
        if depthAvgEdited:
            self.depthAvgEdited.emit()
        if alongTrackAvgEdited:
            self.alongTrackAvgEdited.emit()
        if heaveEdited:
            self.heaveEdited.emit()
        if gridCellsEdited:
            self.gridCellsEdited.emit()
        if pingBufferEdited:
            self.pingBufferEdited.emit()

        # If any settings edited, emit settingsEdited signal
        if systemEdited or ipEdited or portEdited or protocolEdited or socketBufferEdited or \
                binSizeEdited or acrossTrackAvgEdited or depthEdited or depthAvgEdited or \
                alongTrackAvgEdited or heaveEdited or gridCellsEdited or pingBufferEdited:
            self.settingsEdited.emit()
