from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QGridLayout,
    QFrame,
    QCheckBox,
    QPushButton,
)
from qtpy.QtGui import QIntValidator, QFont
from qtpy.QtCore import Qt


class QVertLine(QFrame):
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.setFrameShape(QFrame.VLine)
        self.setFrameShadow(QFrame.Sunken)
        self.setStyleSheet(
            """
        background-color: #5a626c;
        """
        )


class CntrlIMS(QWidget):
    def __init__(self):
        super().__init__()
        self.setLayout(QVBoxLayout())

        # self.add_ims_data = QPushButton("Select microscopy data")
        # self.layout().addWidget(self.add_ims_data)

        self.pad_grid = QWidget()
        self.pad_grid.setLayout(QGridLayout())
        pad_box_width = 40

        label_font = QFont()
        label_font.setBold(True)
        label_font.setWeight(16)

        self.add_pad_label = QLabel()
        self.add_pad_label.setText("Pad Imaging MS canvas")
        self.add_pad_label.setFont(label_font)
        self.add_pad_label.setAlignment(Qt.AlignCenter)

        self.cur_pad_label = QLabel()
        self.cur_pad_label.setText("Current padding")
        self.cur_pad_label.setFont(label_font)
        self.cur_pad_label.setAlignment(Qt.AlignCenter)

        self.set_pad_grid = QWidget()
        self.set_pad_grid.setLayout(QGridLayout())
        self.cur_pad_grid = QWidget()
        self.cur_pad_grid.setLayout(QGridLayout())

        self.pad_left = QLineEdit()
        self.pad_top = QLineEdit()
        self.pad_bottom = QLineEdit()
        self.pad_right = QLineEdit()

        self.pad_left.setValidator(QIntValidator())
        self.pad_top.setValidator(QIntValidator())
        self.pad_bottom.setValidator(QIntValidator())
        self.pad_right.setValidator(QIntValidator())

        self.pad_left.setMaximumWidth(pad_box_width)
        self.pad_top.setMaximumWidth(pad_box_width)
        self.pad_bottom.setMaximumWidth(pad_box_width)
        self.pad_right.setMaximumWidth(pad_box_width)

        self.cur_pad_left = QLabel()
        self.cur_pad_top = QLabel()
        self.cur_pad_bottom = QLabel()
        self.cur_pad_right = QLabel()

        self.cur_pad_left.setText("0")
        self.cur_pad_top.setText("0")
        self.cur_pad_bottom.setText("0")
        self.cur_pad_right.setText("0")

        self.cur_pad_left.setMaximumWidth(pad_box_width)
        self.cur_pad_top.setMaximumWidth(pad_box_width)
        self.cur_pad_bottom.setMaximumWidth(pad_box_width)
        self.cur_pad_right.setMaximumWidth(pad_box_width)

        self.set_pad_grid.layout().addWidget(self.pad_top, 0, 1)
        self.set_pad_grid.layout().addWidget(self.pad_left, 1, 0)
        self.set_pad_grid.layout().addWidget(self.pad_right, 1, 2)
        self.set_pad_grid.layout().addWidget(self.pad_bottom, 2, 1)

        self.cur_pad_grid.layout().addWidget(self.cur_pad_top, 0, 1)
        self.cur_pad_grid.layout().addWidget(self.cur_pad_left, 1, 0)
        self.cur_pad_grid.layout().addWidget(self.cur_pad_right, 1, 2)
        self.cur_pad_grid.layout().addWidget(self.cur_pad_bottom, 2, 1)

        self.pad_grid.layout().addWidget(self.add_pad_label, 0, 0)
        self.pad_grid.layout().addWidget(self.cur_pad_label, 0, 2)

        self.pad_grid.layout().addWidget(QVertLine(self.pad_grid), 0, 1, 0, 2)

        self.pad_grid.layout().addWidget(self.set_pad_grid, 1, 0)
        self.pad_grid.layout().setAlignment(Qt.AlignmentFlag.AlignTop)
        self.pad_grid.layout().addWidget(self.cur_pad_grid, 1, 2)
        self.pad_grid.layout().setAlignment(Qt.AlignmentFlag.AlignTop)

        self.pad_spacing = QWidget()
        self.pad_spacing.setLayout(QHBoxLayout())
        self.micron_check = QCheckBox("Pad in µm")
        self.pixel_check = QCheckBox("Pad in pixels")
        self.micron_check.setChecked(False)
        self.pixel_check.setChecked(True)
        self.pad_spacing.layout().addWidget(self.micron_check)
        self.pad_spacing.layout().addWidget(self.pixel_check)

        self.pad_grid.layout().addWidget(self.pad_spacing, 2, 0)

        self.do_padding_btn = QPushButton("Pad IMS Canvas")

        self.pad_grid.layout().addWidget(self.do_padding_btn, 3, 0)

        self.layout().addWidget(self.pad_grid)

    def _reset(self):
        self.pixel_check.setChecked(True)
        self.cur_pad_left.setText("0")
        self.cur_pad_top.setText("0")
        self.cur_pad_bottom.setText("0")
        self.cur_pad_right.setText("0")
