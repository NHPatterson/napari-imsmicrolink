from qtpy.QtWidgets import QWidget, QVBoxLayout, QGroupBox
from napari_imsmicrolink.qwidgets.subwidgets.control_ims_info import CntrlIMS


class IMSControl(QWidget):
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.ims_ctrl_frame = QGroupBox()
        self.ims_ctrl_frame.setTitle("Imaging MS control")
        self.ims_ctrl_frame.setLayout(QVBoxLayout())
        self.ims_ctl = CntrlIMS()
        self.ims_ctrl_frame.layout().addWidget(self.ims_ctl)
