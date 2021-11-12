from qtpy.QtWidgets import QWidget, QVBoxLayout, QSizePolicy, QGroupBox, QTabWidget
from napari_imsmicrolink.qwidgets.subwidgets.ims_add_info import AddIMSWidget
from napari_imsmicrolink.qwidgets.subwidgets.add_micro_info import AddMicroWidget


class DataControl(QWidget):
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.data_frame = QGroupBox()
        self.data_frame.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.MinimumExpanding)
        self.data_frame.setTitle("Data import")
        self.data_frame.setLayout(QVBoxLayout())

        self.data_tabs = QTabWidget()

        self.ims_d = AddIMSWidget()
        self.micro_d = AddMicroWidget()

        self.data_tabs.addTab(self.ims_d, "IMS")
        self.data_tabs.addTab(self.micro_d, "Microscopy")
        self.data_frame.layout().addWidget(self.data_tabs)
        self.data_frame.layout().addStretch(1)
