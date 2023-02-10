from qtpy.QtWidgets import QWidget, QGroupBox, QVBoxLayout, QSizePolicy
from superqt import QCollapsible
from napari_imsmicrolink.qwidgets.subwidgets.control_save import CntrlSave


class SaveControl(QWidget):
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.save_ctrl_frame = QCollapsible(title="Save / reset data")
        self.save_ctrl_frame.setSizePolicy(
            QSizePolicy.Minimum, QSizePolicy.MinimumExpanding
        )

        self.save_ctrl_box = QGroupBox()
        self.save_ctrl_box.setLayout(QVBoxLayout())
        self.save_ctl = CntrlSave()
        self.save_ctrl_box.layout().addWidget(self.save_ctl)
        self.save_ctrl_frame.addWidget(self.save_ctrl_box)
