from qtpy.QtWidgets import QWidget, QSizePolicy, QTabWidget
from superqt import QCollapsible
from napari_imsmicrolink.qwidgets.subwidgets.ims_add_info import AddIMSWidget
from napari_imsmicrolink.qwidgets.subwidgets.add_micro_info import AddMicroWidget


class DataControl(QWidget):
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.data_frame = QCollapsible(title="Data import")
        self.data_frame.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.MinimumExpanding)

        self.data_tabs = QTabWidget()
        self.ims_d = AddIMSWidget()
        self.micro_d = AddMicroWidget()

        self.data_tabs.addTab(self.ims_d, "IMS")
        self.data_tabs.addTab(self.micro_d, "Microscopy")
        self.data_frame.addWidget(self.data_tabs)
        self.data_frame.expand()
