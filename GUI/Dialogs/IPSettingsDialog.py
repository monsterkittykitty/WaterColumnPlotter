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
        super().__init__(parent)
        self.setWindowTitle('IP:Port Settings')
        dlgLayout = QVBoxLayout()
        formLayout = QFormLayout()
        formLayout.addRow('IP:', QLineEdit())
        formLayout.addRow('Port:', QLineEdit())
        dlgLayout.addLayout(formLayout)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)



        button_box.accepted.connect(self.test)

        dlgLayout.addWidget(button_box)
        self.setLayout(dlgLayout)

    def test(self):
        self.close()
        #pass