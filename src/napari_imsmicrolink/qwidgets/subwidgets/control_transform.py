from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QComboBox,
)


class PointTable(QTableWidget):
    def __init__(self):
        QTableWidget.__init__(self, 0, 8)
        row_headers = [
            "x-i(px)",
            "y-i(px)",
            "x-m(px)",
            "y-m(px)",
            "x-i(µm)",
            "y-i(µm)",
            "x-m(µm)",
            "y-m(µm)",
        ]
        self.ims_cols = [0, 1, 4, 5]
        self.micro_cols = [2, 3, 6, 7]

        self.setHorizontalHeaderLabels(row_headers)
        self.resizeColumnsToContents()

    def add_point_data(self, pt_data, ims_or_micro="ims", image_res=1):
        n_row = self.rowCount()
        if n_row < pt_data.shape[0]:
            n_rows_to_insert = pt_data.shape[0] - n_row
            if n_row != 0:
                for i in range(n_rows_to_insert):
                    self.insertRow(n_row + i)
            else:
                self.insertRow(0)

        if ims_or_micro == "ims":
            col_idx = self.ims_cols
        else:
            col_idx = self.micro_cols

        for idx, pt in enumerate(pt_data):
            py, px = pt[0], pt[1]

            if ims_or_micro == "ims":
                p_x_um = QTableWidgetItem(str(round(px * image_res, 2)))
                p_y_um = QTableWidgetItem(str(round(py * image_res, 2)))
                p_x_px = QTableWidgetItem(str(round(px, 0)))
                p_y_px = QTableWidgetItem(str(round(py, 0)))
            else:
                p_x_um = QTableWidgetItem(str(round(px, 2)))
                p_y_um = QTableWidgetItem(str(round(py, 2)))
                p_x_px = QTableWidgetItem(str(round(px / image_res, 0)))
                p_y_px = QTableWidgetItem(str(round(py / image_res, 0)))

            self.setItem(idx, col_idx[0], p_x_px)
            self.setItem(idx, col_idx[1], p_y_px)
            self.setItem(idx, col_idx[2], p_x_um)
            self.setItem(idx, col_idx[3], p_y_um)

        updated_n_row = self.rowCount()
        if updated_n_row > pt_data.shape[0]:
            n_items_to_delete = updated_n_row - pt_data.shape[0]
            for idx in range(n_items_to_delete):
                self.takeItem(pt_data.shape[0] + idx, col_idx[0])
                self.takeItem(pt_data.shape[0] + idx, col_idx[1])
                self.takeItem(pt_data.shape[0] + idx, col_idx[2])
                self.takeItem(pt_data.shape[0] + idx, col_idx[3])

        self.resizeColumnsToContents()
        self.resizeRowsToContents()

    def _reset_table(self):
        self.setRowCount(0)


class CntrlTransform(QWidget):
    def __init__(self):
        super().__init__()
        self.setLayout(QVBoxLayout())
        self.pt_table = PointTable()

        self.sub_area = QWidget()
        self.sub_area.setLayout(QHBoxLayout())

        self.error_area = QWidget()
        self.error_area.setLayout(QFormLayout())

        self.run_transform = QPushButton("Visualize transformation")
        self.reset_transform = QPushButton("Reset transformation")

        self.run_transform.setStyleSheet(
            "QPushButton:disabled{background-color:rgb(130, 82, 82);}"
        )
        self.run_transform.setMaximumWidth(140)

        self.error_label = QLabel("Transformation error (μm):")
        self.tform_error = QLabel("")

        self.target_mod_label = QLabel("Target modality:")
        self.target_mode_combo = QComboBox()
        self.target_mode_combo.addItem("IMS")
        self.target_mode_combo.addItem("Microscopy")

        self.error_area.layout().addRow(self.error_label, self.tform_error)
        self.error_area.layout().addRow(self.target_mod_label, self.target_mode_combo)

        self.sub_area.layout().addWidget(self.run_transform)
        self.sub_area.layout().addWidget(self.reset_transform)

        self.layout().addWidget(self.pt_table)
        self.layout().addWidget(self.sub_area)
        self.layout().addWidget(self.error_area)

    def _reset(self):
        self.pt_table._reset_table()
        self.tform_error.setText("")
