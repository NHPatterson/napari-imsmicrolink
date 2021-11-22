from typing import NamedTuple, Optional
from qtpy.QtWidgets import (
    QPushButton,
    QLabel,
    QLineEdit,
    QFormLayout,
    QDialog,
    QErrorMessage,
    QFileDialog,
    QComboBox,
)
from qtpy.QtCore import Qt


class SaveData(NamedTuple):
    project_name: str
    output_dir: str
    output_filetype: str


class SavePopUp(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.output_dir = None
        self.project_name = None
        self.completed = False
        self.project_data: Optional[SaveData] = None

        self.setMinimumSize(400, 75)
        self.setLayout(QFormLayout())

        self.project_lbl = QLabel(self)
        self.project_lbl.setText("Set project (file) name: ")
        self.project_line = QLineEdit(self)

        self.set_output_dir_btn = QPushButton("Select directory")
        self.output_dir_lbl = QLabel(self)
        self.output_dir_lbl.setText("[not selected]")

        self.output_ft_lbl = QLabel("Output format")

        self.output_filetype = QComboBox()
        self.output_filetype.addItem(".h5")
        self.output_filetype.addItem(".csv")

        self.cancel_btn = QPushButton("Cancel save")
        self.save_btn = QPushButton("Save data")

        self.layout().addRow(self.set_output_dir_btn, self.output_dir_lbl)
        self.layout().addRow(self.project_lbl, self.project_line)
        self.layout().addRow(self.output_ft_lbl, self.output_filetype)
        self.layout().addRow(self.cancel_btn, self.save_btn)

        self.set_output_dir_btn.clicked.connect(self.get_output_dir)
        self.save_btn.clicked.connect(self._save_config)
        self.cancel_btn.clicked.connect(self._cancel_save)

    def get_output_dir(self):
        self.output_dir = QFileDialog.getExistingDirectory(
            self, "Select output directory"
        )
        if self.output_dir is not None:
            self.output_dir_lbl.setText(self.output_dir)

    def _gather_data(self) -> SaveData:
        return SaveData(
            output_dir=self.output_dir,
            project_name=self.project_name,
            output_filetype=self.output_filetype.currentText(),
        )

    def _save_config(self):
        if self.output_dir is None:
            error_message = QErrorMessage(self)
            error_message.showMessage("Output directory must be set")
            return

        if self.project_line.text() == "":
            error_message = QErrorMessage(self)
            error_message.showMessage("Project name must be provided")
            return

        self.project_name = self.project_line.text()
        self.project_data = self._gather_data()
        self.completed = True
        self.close()

    def _cancel_save(self):
        self.completed = False
        self.close()

    @staticmethod
    def show_dialog(qt_viewer, parent=None):
        d = SavePopUp(parent)
        d.setWindowTitle("Enter project (file name) information...")
        d.setWindowModality(Qt.ApplicationModal)
        d.exec_()
        return d
