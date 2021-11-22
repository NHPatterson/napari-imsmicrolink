from qtpy.QtWidgets import QWidget, QVBoxLayout, QGroupBox
from napari_imsmicrolink.qwidgets.subwidgets.control_save import CntrlSave


class SaveControl(QWidget):
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.save_ctrl_frame = QGroupBox()
        self.save_ctrl_frame.setTitle("Save / reset data")
        self.save_ctrl_frame.setLayout(QVBoxLayout())
        self.save_ctl = CntrlSave()
        self.save_ctrl_frame.layout().addWidget(self.save_ctl)
