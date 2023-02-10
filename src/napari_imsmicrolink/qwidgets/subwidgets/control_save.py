from qtpy.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QPushButton,
)


class CntrlSave(QWidget):
    def __init__(self):
        super().__init__()
        self.setLayout(QHBoxLayout())
        self.reset_data = QPushButton("Reset all data")
        self.save_data = QPushButton("Save transform data")
        self.save_data.setStyleSheet(
            "QPushButton:disabled{background-color:rgb(130, 82, 82);}"
        )

        self.layout().addWidget(self.reset_data)
        self.layout().addWidget(self.save_data)
