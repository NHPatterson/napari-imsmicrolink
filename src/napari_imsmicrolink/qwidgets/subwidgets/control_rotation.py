from qtpy.QtWidgets import (
    QWidget,
    QPushButton,
    QHBoxLayout,
    QGroupBox,
    QGridLayout,
)


class CntrlRotation(QWidget):
    def __init__(self):
        super().__init__()
        self.setLayout(QHBoxLayout())

        self.ims_rot_box = QGroupBox()
        self.ims_rot_box.setTitle("IMS rotation")
        self.ims_rot_box.setLayout(QGridLayout())
        self.micro_rot_box = QGroupBox()
        self.micro_rot_box.setTitle("Microscopy rotation")
        self.micro_rot_box.setLayout(QGridLayout())

        self.rot_mi_90ccw = QPushButton("<< 90° CCW")
        self.rot_mi_180ccw = QPushButton("<< 180° CCW")
        self.rot_mi_90cw = QPushButton("90° CW >>")
        self.rot_mi_180cw = QPushButton("180° CW >>")

        self.rot_ims_90ccw = QPushButton("<< 90° CCW")
        self.rot_ims_90cw = QPushButton("90° CW >>")

        self.ims_rot_box.layout().addWidget(self.rot_ims_90ccw, 0, 0)
        self.ims_rot_box.layout().addWidget(self.rot_ims_90cw, 0, 1)

        self.micro_rot_box.layout().addWidget(self.rot_mi_90ccw, 0, 0)
        self.micro_rot_box.layout().addWidget(self.rot_mi_180ccw, 1, 0)
        self.micro_rot_box.layout().addWidget(self.rot_mi_90cw, 0, 1)
        self.micro_rot_box.layout().addWidget(self.rot_mi_180cw, 1, 1)

        self.layout().addWidget(self.micro_rot_box)
        self.layout().addWidget(self.ims_rot_box)
