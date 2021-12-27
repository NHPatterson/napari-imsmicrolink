from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QFormLayout,
    QSizePolicy,
)
from qtpy.QtGui import QDoubleValidator
from qtpy.QtCore import Qt


class AddMicroWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setLayout(QVBoxLayout())
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.MinimumExpanding)

        self.add_micro_data = QPushButton("Select microscopy data")
        self.layout().addWidget(self.add_micro_data)

        self.info_grid = QWidget()
        self.info_grid.setLayout(QFormLayout())
        self.info_grid.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.MinimumExpanding)

        self.res_info_label = QLabel()
        self.res_info_label.setText("Microscopy spacing (µm):")

        self.res_info_input = QLineEdit()
        self.res_info_input.setValidator(QDoubleValidator())
        self.res_info_input.setMaximumWidth(50)

        self.file_info_label = QLabel()
        self.file_info_label.setText("Microscopy file name:")

        self.file_info_input = QLabel()
        self.file_info_input.setText("[not selected]")

        self.info_grid.layout().addRow(self.file_info_label, self.file_info_input)
        self.info_grid.layout().addRow(self.res_info_label, self.res_info_input)

        self.info_grid.layout().setAlignment(Qt.AlignmentFlag.AlignTop)

        self.layout().addWidget(self.info_grid)

    def _reset(self):
        self.file_info_input.setText("[not selected]")
        self.res_info_input.setText("")
