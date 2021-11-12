from qtpy.QtWidgets import QWidget, QVBoxLayout, QGroupBox
from napari_imsmicrolink.qwidgets.subwidgets.control_transform import CntrlTransform


class TformControl(QWidget):
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.tform_ctrl_frame = QGroupBox()
        self.tform_ctrl_frame.setTitle("Transformation control")
        self.tform_ctrl_frame.setLayout(QVBoxLayout())
        self.tform_ctl = CntrlTransform()
        self.tform_ctrl_frame.layout().addWidget(self.tform_ctl)