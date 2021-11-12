from qtpy.QtWidgets import (
    QPushButton,
    QLabel,
    QLineEdit,
    QGridLayout,
    QDialog,
    QErrorMessage,
    QFileDialog,
)
from qtpy.QtCore import Qt

class SavePopUp(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.output_dir = None
        self.project_name = None
        self.completed = False
        self.setMinimumSize(400,75)
        self.grid = QGridLayout(self)
        self.grid.setSpacing(10)

        self.project_lbl = QLabel(self)
        self.project_lbl.setText("Set project (file) name: ")
        self.project_line = QLineEdit(self)

        self.set_output_dir_btn = QPushButton("Select directory")
        self.output_dir_lbl = QLabel(self)
        self.output_dir_lbl.setText("[not selected]")

        self.save_btn = QPushButton("Save data")

        self.grid.addWidget(self.set_output_dir_btn, 0, 0)
        self.grid.addWidget(self.output_dir_lbl, 0, 1)

        self.grid.addWidget(self.project_lbl, 1, 0)
        self.grid.addWidget(self.project_line, 1, 1)

        self.grid.addWidget(self.save_btn, 2, 1)

        self.setLayout(self.grid)

        self.set_output_dir_btn.clicked.connect(self.get_output_dir)
        self.save_btn.clicked.connect(self._save_config)

    def get_output_dir(self):
        self.output_dir = QFileDialog.getExistingDirectory(
            self, "Select output directory"
        )
        if self.output_dir is not None:
            self.output_dir_lbl.setText(self.output_dir)

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
        self.completed = True
        self.close()

    @staticmethod
    def show_dialog(qt_viewer, parent=None):
        d = SavePopUp(parent)
        d.setWindowTitle("Enter project (file name) information...")
        d.setWindowModality(Qt.ApplicationModal)
        d.exec_()
        return d