from qtpy.QtWidgets import QWidget, QGroupBox, QVBoxLayout, QSizePolicy
from superqt import QCollapsible
from napari_imsmicrolink.qwidgets.subwidgets.control_ims_info import CntrlIMS


class IMSControl(QWidget):
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.ims_ctrl_frame = QCollapsible(title="Imaging MS padding")
        self.ims_ctrl_frame.setSizePolicy(
            QSizePolicy.Minimum, QSizePolicy.MinimumExpanding
        )

        self.ims_ctrl_box = QGroupBox()

        self.ims_ctrl_box.setLayout(QVBoxLayout())
        self.ims_ctl = CntrlIMS()
        self.ims_ctrl_box.layout().addWidget(self.ims_ctl)
        self.ims_ctrl_frame.addWidget(self.ims_ctrl_box)
