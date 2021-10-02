# Lynette Davis
# Center for Coastal and Ocean Mapping
# University of New Hampshire
# August 2021

# Description:

import sys

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

class SettingsDialogue(QDialog):
    def __init__(self, parent=None):
        #QDialog.__init__()
        super().__init__(parent)

        self.ip = None
        self.port = None

        self.setWindowTitle('IP:Port Settings')
        dlgLayout = QVBoxLayout()
        formLayout = QFormLayout()
        self.ip_qle = QLineEdit()
        self.port_qle = QLineEdit()
        formLayout.addRow('IP:', self.ip_qle)
        formLayout.addRow('Port:', self.port_qle)
        dlgLayout.addLayout(formLayout)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)

        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        dlgLayout.addWidget(button_box)
        self.setLayout(dlgLayout)

    def accept(self):
        self.ip = self.ip_qle.text()
        self.port = self.port_qle.text()
        print("ip: ", self.ip)
        print("port: ", self.port)
        self.close()
        #pass

    def reject(self):
        self.close()
