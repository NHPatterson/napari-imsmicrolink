from qtpy.QtWidgets import QWidget, QTabWidget, QSizePolicy
from superqt import QCollapsible
from napari_imsmicrolink.qwidgets.subwidgets.control_transform import CntrlTransform
from napari_imsmicrolink.qwidgets.subwidgets.control_rotation import CntrlRotation


class TformControl(QWidget):
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.tform_ctrl_frame = QCollapsible(title="Transformation control")
        self.tform_ctrl_frame.setSizePolicy(
            QSizePolicy.Minimum, QSizePolicy.MinimumExpanding
        )

        self.tform_tabs = QTabWidget()
        self.tform_ctl = CntrlTransform()
        self.rot_ctl = CntrlRotation()

        self.tform_tabs.addTab(self.tform_ctl, "Transform")
        self.tform_tabs.addTab(self.rot_ctl, "Orientation")
        self.tform_ctrl_frame.addWidget(self.tform_tabs)
        self.tform_ctrl_frame.expand(animate=False)
