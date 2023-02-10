from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QFormLayout,
    QSizePolicy,
    QComboBox,
)
from qtpy.QtGui import QDoubleValidator
from qtpy.QtCore import Qt


class AddIMSWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setLayout(QVBoxLayout())
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.MinimumExpanding)

        self.add_ims_data = QPushButton("Select IMS data")
        self.layout().addWidget(self.add_ims_data)

        self.info_grid = QWidget()
        self.info_grid.setLayout(QFormLayout())

        self.res_info_label = QLabel()
        self.res_info_label.setText("Imaging MS spacing (µm):")

        self.res_info_input = QLineEdit()
        self.res_info_input.setValidator(QDoubleValidator())
        self.res_info_input.setMaximumWidth(50)

        self.file_info_label = QLabel()
        self.file_info_label.setText("Imaging MS file name:")

        self.file_info_input = QLabel()
        self.file_info_input.setText("[not selected]")
        # self.file_info_input.setMaximumWidth(80)
        self.file_info_input.setWordWrap(True)

        self.delete_box = QComboBox()
        self.delete_btn = QPushButton("Delete ROI")

        self.info_grid.layout().addRow(self.file_info_label, self.file_info_input)
        self.info_grid.layout().addRow(self.res_info_label, self.res_info_input)
        # self.info_grid.layout().addRow(self.del_label,)
        self.info_grid.layout().addRow(self.delete_box, self.delete_btn)

        self.info_grid.layout().setAlignment(Qt.AlignmentFlag.AlignTop)
        self.layout().addWidget(self.info_grid)

    def _reset(self):
        self.file_info_input.setText("[not selected]")
        self.res_info_input.setText("")
