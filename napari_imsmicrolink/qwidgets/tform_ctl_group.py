from qtpy.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QTabWidget
from napari_imsmicrolink.qwidgets.subwidgets.control_transform import CntrlTransform
from napari_imsmicrolink.qwidgets.subwidgets.control_rotation import CntrlRotation


class TformControl(QWidget):
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.tform_ctrl_frame = QGroupBox()
        self.tform_ctrl_frame.setTitle("Transformation control")
        self.tform_ctrl_frame.setLayout(QVBoxLayout())
        self.tform_tabs = QTabWidget()
        self.tform_ctl = CntrlTransform()
        self.rot_ctl = CntrlRotation()

        self.tform_tabs.addTab(self.tform_ctl, "Transform")
        self.tform_tabs.addTab(self.rot_ctl, "Orientation")
        self.tform_ctrl_frame.layout().addWidget(self.tform_tabs)
        self.tform_ctrl_frame.layout().addStretch(1)
