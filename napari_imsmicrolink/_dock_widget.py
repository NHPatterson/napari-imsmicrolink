from typing import List, Tuple, Optional, Union
from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import SimpleITK as sitk

from napari_plugin_engine import napari_hook_implementation
import napari
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import QWidget, QVBoxLayout, QErrorMessage
from qtpy.QtCore import QTimer

from napari_imsmicrolink.qwidgets.data_ctl_group import DataControl
from napari_imsmicrolink.qwidgets.ims_ctl_group import IMSControl
from napari_imsmicrolink.qwidgets.tform_ctl_group import TformControl
from napari_imsmicrolink.qwidgets.save_ctl_group import SaveControl
from napari_imsmicrolink.qwidgets.save_dialog import SavePopUp
from napari_imsmicrolink.data.ims_pixel_map import PixelMapIMS
from napari_imsmicrolink.data.microscopy_reader import MicroRegImage
from napari_imsmicrolink.data.microscopy_writer import OmeTiffWriter
from napari_imsmicrolink.data.image_transform import ImageTransform
from napari_imsmicrolink.utils.file import open_file_dialog
from napari_imsmicrolink.utils.color import COLOR_HEXES
from napari_imsmicrolink.utils.image import centered_transform
from napari_imsmicrolink.utils.coords import pmap_coords_to_h5


class IMSMicroLink(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        """
        Main class of the IMS MicroLink to perform registration.

        Parameters
        ----------
        napari_viewer: napari.Viewer
            napari's viewer

        Attributes
        ----------
        ims_pixel_map: Optional[PixelMapIMS]
            IMS Pixel Map data container
        microscopy_image: Optional[MicroRegImage]
            Microscopy image data container via aicsimiageio bioformats reader
        micro_image_names: Optional[List[str, ...]]
            list of channel names from microscopy image
        image_transformer: ImageTransform
            Data class to manage points and compute transformation matrices
        last_transform: np.ndarray
            Most recent transformation
        output_size: Optional[Tuple[int, int]]
            Size, in pixels, of the output image
        """
        super().__init__()
        self.viewer = napari_viewer

        # data classes
        self.ims_pixel_map: Optional[PixelMapIMS] = None
        self.microscopy_image: Optional[MicroRegImage] = None
        self.micro_image_names: Optional[List[str, ...]] = None
        self.image_transformer: ImageTransform = ImageTransform()
        self.last_transform: np.ndarray = np.eye(3)
        self.output_size: Optional[Tuple[int, int]] = None

        # private attrs
        self._micro_rot = 0
        self._ims_rot = 0

        # controller widgets
        self.data: DataControl = DataControl(self)
        self.ims_c: IMSControl = IMSControl(self)
        self.tform_c: TformControl = TformControl(self)
        self.save_c: SaveControl = SaveControl(self)

        # layout widgets in GUI
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.data.data_frame)
        self.layout().addWidget(self.ims_c.ims_ctrl_frame)
        self.layout().addWidget(self.tform_c.tform_ctrl_frame)
        self.layout().addWidget(self.save_c.save_ctrl_frame)
        self.layout().addStretch(1)

        # button actions for widgets
        # data reading
        self.data.ims_d.add_ims_data.clicked.connect(self.read_ims_data)
        self.data.micro_d.add_micro_data.clicked.connect(self.read_micro_data)

        # ims canvas control
        self.ims_c.ims_ctl.do_padding_btn.clicked.connect(self.pad_ims_canvas)
        self.ims_c.ims_ctl.micron_check.clicked.connect(
            lambda: self.set_pad_by("micron")
        )
        self.ims_c.ims_ctl.pixel_check.clicked.connect(lambda: self.set_pad_by("pixel"))

        # transform control
        self.tform_c.tform_ctl.run_transform.clicked.connect(self.run_transformation)
        self.tform_c.rot_ctl.rot_mi_90cw.clicked.connect(
            lambda: self._rotate_modality("microscopy", -90)
        )
        self.tform_c.rot_ctl.rot_mi_90ccw.clicked.connect(
            lambda: self._rotate_modality("microscopy", 90)
        )
        self.tform_c.rot_ctl.rot_mi_180cw.clicked.connect(
            lambda: self._rotate_modality("microscopy", 180)
        )
        self.tform_c.rot_ctl.rot_mi_180ccw.clicked.connect(
            lambda: self._rotate_modality("microscopy", 180)
        )
        self.tform_c.rot_ctl.rot_ims_90cw.clicked.connect(
            lambda: self._rotate_modality("ims", -90)
        )
        self.tform_c.rot_ctl.rot_ims_90ccw.clicked.connect(
            lambda: self._rotate_modality("ims", 90)
        )
        self.tform_c.rot_ctl.rot_ims_180cw.clicked.connect(
            lambda: self._rotate_modality("ims", 180)
        )
        self.tform_c.rot_ctl.rot_ims_180ccw.clicked.connect(
            lambda: self._rotate_modality("ims", 180)
        )

        # save control
        self.save_c.save_ctl.reset_data.clicked.connect(self.reset_data)
        self.save_c.save_ctl.save_data.clicked.connect(self.save_data)

        # update spatial resolutions
        self.ims_res_timer = QTimer()
        self.ims_res_timer.setSingleShot(True)
        self.ims_res_timer.timeout.connect(self.set_ims_res)
        self.data.ims_d.res_info_input.textChanged.connect(self._ims_res_timing)

        self.micro_res_timer = QTimer()
        self.micro_res_timer.setSingleShot(True)
        self.micro_res_timer.timeout.connect(self.set_micro_res)
        self.data.micro_d.res_info_input.textChanged.connect(self._micro_res_timing)

        # start with certain buttons disabled
        self.save_c.save_ctl.save_data.setEnabled(False)
        self.tform_c.tform_ctl.run_transform.setEnabled(False)

    def _ims_res_timing(self) -> None:
        """Wait until there are no changes for 0.5 second before making changes."""
        self.ims_res_timer.start(500)

    def _micro_res_timing(self) -> None:
        """Wait until there are no changes for 0.5 second before making changes."""
        self.micro_res_timer.start(500)

    def read_ims_data(self) -> None:
        """
        Collect data path and import IMS data, generating IMS Pixel Map.
        """
        file_paths = open_file_dialog(
            self,
            single=False,
            wd="",
            name="Open IMS Pixel Maps",
            file_types="All Files (*);;Tiff files (*.txt,*.tif);;sqlite files (*.sqlite);;imzML (*.imzML)",
        )
        if file_paths:
            self.ims_pixel_map = PixelMapIMS(file_paths)
            if len(self.data.ims_d.res_info_input.text()) == 0:
                self.data.ims_d.res_info_input.setText("1")

            self._add_ims_data()
            if isinstance(file_paths, list):
                fp_names = [Path(fp).name for fp in file_paths]
                fp_names = ",".join(fp_names)
                fp_names_full = [Path(fp).as_posix() for fp in file_paths]
                fp_names_full = "\n".join(fp_names_full)
            else:
                fp_names = Path(file_paths).name
                fp_names_full = Path(file_paths).as_posix()

            self.data.ims_d.file_info_input.setText(fp_names)
            self.data.ims_d.file_info_input.setToolTip(fp_names_full)

    def _add_ims_fiducials(self) -> None:
        """
        Add IMS fiducals layer
        """
        self.viewer.add_points(
            data=None,
            name="IMS Fiducials",
            face_color="red",
            edge_color="black",
            size=1
            if self.ims_pixel_map.ims_res == 1
            else self.ims_pixel_map.ims_res // 2,
            scale=(self.ims_pixel_map.ims_res, self.ims_pixel_map.ims_res),
            symbol="x",
        )

        self.viewer.layers["IMS Fiducials"].events.data.connect(self._add_target_pts)

    def _add_target_pts(self, _) -> None:
        self.tform_c.tform_ctl.pt_table.add_point_data(
            self.viewer.layers["IMS Fiducials"].data,
            ims_or_micro="ims",
            image_res=self.ims_pixel_map.ims_res,
        )
        self.image_transformer.add_points(
            self.viewer.layers["IMS Fiducials"].data,
            round=True,
            src_or_tgt="target",
            scaling=self.ims_pixel_map.ims_res,
        )
        self._check_for_tforms()

        if self.image_transformer.point_reg_error != float("inf"):
            self.tform_c.tform_ctl.tform_error.setText(
                f"{np.round(self.image_transformer.point_reg_error, 2)}"
            )
            if self.image_transformer.point_reg_error < self.ims_pixel_map.ims_res / 2:
                self.tform_c.tform_ctl.tform_error.setStyleSheet("color: green")
            else:
                self.tform_c.tform_ctl.tform_error.setStyleSheet("color: red")

    def _add_ims_data(self) -> None:
        if len(self.data.ims_d.res_info_input.text()) > 0:
            ims_res = float(self.data.ims_d.res_info_input.text())
        else:
            ims_res = 1

        if "IMS Pixel Map" in self.viewer.layers:
            self.viewer.layers.pop("IMS Pixel Map")

        self.viewer.add_image(
            self.ims_pixel_map.pixelmap_padded,
            name="IMS Pixel Map",
            scale=(ims_res, ims_res),
            colormap="viridis",
        )

        self._add_ims_rois()
        self._add_ims_fiducials()
        self._update_output_size()

        self.viewer.reset_view()

    def _add_ims_rois(self) -> None:
        shapes = self.ims_pixel_map._shape_map_minimized
        shape_names = [shape[0] for shape in shapes]
        shape_data = [shape[1] for shape in shapes]
        shape_props = {"name": shape_names}
        shape_text = {
            "text": "{name}",
            "color": "white",
            "anchor": "center",
            "size": 12,
        }

        face_colors = []
        for i in range(len(shapes)):
            if i > len(COLOR_HEXES):
                idx = np.random.randint(0, len(COLOR_HEXES) - 1)
                face_colors.append(COLOR_HEXES[idx])
            else:
                face_colors.append(COLOR_HEXES[i])

        self.viewer.add_shapes(
            shape_data,
            shape_type="polygon",
            name="IMS ROIs",
            properties=shape_props,
            text=shape_text,
            edge_color="white",
            edge_width=0.5,
            face_color=face_colors,
            opacity=0.55,
            scale=(self.ims_pixel_map.ims_res, self.ims_pixel_map.ims_res),
        )

    def _add_micro_fiducials(self) -> None:
        if self.ims_pixel_map:
            if self.ims_pixel_map.ims_res == 1 or self.ims_pixel_map.ims_res is None:
                fiducial_size = 1
            else:
                fiducial_size = self.ims_pixel_map.ims_res // 2
        else:
            fiducial_size = 1

        self.viewer.add_points(
            data=None,
            name="Microscopy Fiducials",
            face_color="green",
            edge_color="black",
            size=fiducial_size,
            scale=(
                self.microscopy_image.base_layer_pixel_res,
                self.microscopy_image.base_layer_pixel_res,
            ),
            symbol="cross",
        )

        self.viewer.layers["Microscopy Fiducials"].events.data.connect(
            self._add_source_pts
        )

    def _add_source_pts(self, _) -> None:
        self.tform_c.tform_ctl.pt_table.add_point_data(
            self.viewer.layers["Microscopy Fiducials"].data,
            ims_or_micro="micro",
            image_res=self.microscopy_image.base_layer_pixel_res,
        )

        self.image_transformer.add_points(
            self.viewer.layers["Microscopy Fiducials"].data,
            round=False,
            src_or_tgt="source",
            scaling=self.microscopy_image.base_layer_pixel_res,
        )

        if self.image_transformer.point_reg_error != float("inf"):
            self.tform_c.tform_ctl.tform_error.setText(
                f"{np.round(self.image_transformer.point_reg_error, 3)}"
            )

    @thread_worker
    def _process_micro_data(self, file_path: str) -> Tuple[MicroRegImage, np.ndarray]:
        microscopy_image = MicroRegImage(file_path)
        dask_im = microscopy_image.pyr_levels_dask[1]
        micro_data = []
        for i in range(dask_im.shape[0]):
            im = sitk.GetImageFromArray(dask_im[i, :, :].compute())
            im = sitk.RescaleIntensity(im)
            im = sitk.GetArrayFromImage(im)
            micro_data.append(im)

        micro_data = np.stack(micro_data)
        return microscopy_image, micro_data

    def _add_micro_data(self, data: Tuple[MicroRegImage, np.ndarray]) -> None:
        self.microscopy_image = data[0]
        file_path = self.microscopy_image.image_filepath
        fp_name = Path(file_path).name
        fp_name_full = Path(file_path).as_posix()
        self.viewer.add_image(
            data[1],
            name=self.microscopy_image.cnames,
            scale=(
                self.microscopy_image.base_layer_pixel_res,
                self.microscopy_image.base_layer_pixel_res,
            ),
            channel_axis=0,
        )

        self._update_output_spacing(self.microscopy_image.base_layer_pixel_res)
        self.micro_image_names = self.microscopy_image.cnames
        self.data.micro_d.file_info_input.setText(fp_name)
        self.data.micro_d.file_info_input.setToolTip(fp_name_full)
        self.data.micro_d.res_info_input.setText(
            str(self.microscopy_image.base_layer_pixel_res)
        )
        self.data.micro_d.res_info_input.setText(
            str(self.microscopy_image.base_layer_pixel_res)
        )
        self._add_micro_fiducials()
        self.viewer.reset_view()

    def read_micro_data(self) -> None:
        file_path = open_file_dialog(
            self,
            single=True,
            wd="",
            name="Open post-acquisition microscopy image",
            file_types="All Files (*);;Tiff files (*.tiff,*.tif);;czi files (*.czi)",
        )
        if file_path:
            micro_reader_worker = self._process_micro_data(file_path)
            micro_reader_worker.returned.connect(self._add_micro_data)
            micro_reader_worker.start()

    def _check_for_tforms(self) -> None:
        if self.image_transformer.affine_transform:
            self.save_c.save_ctl.save_data.setEnabled(True)
            self.tform_c.tform_ctl.run_transform.setEnabled(True)

    def _update_output_size(self) -> None:
        if self.ims_pixel_map:
            if len(self.data.micro_d.res_info_input.text()) > 0:
                micro_res = float(self.data.micro_d.res_info_input.text())
            else:
                micro_res = 1

            # canvas is defined as shape of ims data * ims data in microns
            canvas_size_microns = np.multiply(
                self.ims_pixel_map.pixelmap_padded.shape[::-1],
                self.ims_pixel_map.ims_res,
            )

            # output size is in PIXELS, need to convert to number of pixels based on microscopy
            # resolution
            _output_size = np.ceil(
                np.multiply(canvas_size_microns, (1 / micro_res))
            ).astype(int)

            _output_size = tuple([int(x) for x in _output_size])

            self.image_transformer.output_size = _output_size

    def _update_output_spacing(self, spacing: float) -> None:
        self.image_transformer.output_spacing = (
            spacing,
            spacing,
        )

    def set_pad_by(self, unit: str) -> None:
        if unit == "micron":
            if self.ims_c.ims_ctl.pixel_check.isChecked():
                self.ims_c.ims_ctl.pixel_check.setChecked(False)
            else:
                self.ims_c.ims_ctl.pixel_check.setChecked(True)
        else:
            if self.ims_c.ims_ctl.micron_check.isChecked():
                self.ims_c.ims_ctl.micron_check.setChecked(False)
            else:
                self.ims_c.ims_ctl.micron_check.setChecked(True)

        if self.ims_pixel_map:
            self._update_current_padding()

    def _update_current_padding(self) -> None:
        if self.ims_c.ims_ctl.micron_check.isChecked():
            self.ims_c.ims_ctl.cur_pad_left.setText(
                str(self.ims_pixel_map.padding_microns.get("x_left"))
            )
            self.ims_c.ims_ctl.cur_pad_right.setText(
                str(self.ims_pixel_map.padding_microns.get("x_right"))
            )
            self.ims_c.ims_ctl.cur_pad_top.setText(
                str(self.ims_pixel_map.padding_microns.get("y_top"))
            )
            self.ims_c.ims_ctl.cur_pad_bottom.setText(
                str(self.ims_pixel_map.padding_microns.get("y_bottom"))
            )
        else:
            self.ims_c.ims_ctl.cur_pad_left.setText(
                str(self.ims_pixel_map.padding.get("x_left"))
            )
            self.ims_c.ims_ctl.cur_pad_right.setText(
                str(self.ims_pixel_map.padding.get("x_right"))
            )
            self.ims_c.ims_ctl.cur_pad_top.setText(
                str(self.ims_pixel_map.padding.get("y_top"))
            )
            self.ims_c.ims_ctl.cur_pad_bottom.setText(
                str(self.ims_pixel_map.padding.get("y_bottom"))
            )

    def pad_ims_canvas(self) -> None:

        x_left = (
            int(self.ims_c.ims_ctl.pad_left.text())
            if len(self.ims_c.ims_ctl.pad_left.text()) > 0
            else 0
        )
        x_right = (
            int(self.ims_c.ims_ctl.pad_right.text())
            if len(self.ims_c.ims_ctl.pad_right.text()) > 0
            else 0
        )
        y_top = (
            int(self.ims_c.ims_ctl.pad_top.text())
            if len(self.ims_c.ims_ctl.pad_top.text()) > 0
            else 0
        )
        y_bottom = (
            int(self.ims_c.ims_ctl.pad_bottom.text())
            if len(self.ims_c.ims_ctl.pad_bottom.text()) > 0
            else 0
        )

        ims_res = float(self.data.ims_d.res_info_input.text())
        if self.ims_c.ims_ctl.micron_check.isChecked():
            x_left = np.ceil(x_left / ims_res).astype(int)
            x_right = np.ceil(x_right / ims_res).astype(int)
            y_top = np.ceil(y_top / ims_res).astype(int)
            y_bottom = np.ceil(y_bottom / ims_res).astype(int)

        padding = {
            "x_left": x_left,
            "x_right": x_right,
            "y_top": y_top,
            "y_bottom": y_bottom,
        }

        self.ims_pixel_map.padding = padding

        # update image
        self.viewer.layers["IMS Pixel Map"].data = self.ims_pixel_map.pixelmap_padded

        # update IMS ROIs
        updated_shapes = []
        for idx, shape in enumerate(self.ims_pixel_map._shape_map_minimized):
            _, shape_data = deepcopy(shape)
            shape_data[:, 0] = shape_data[:, 0] + self.ims_pixel_map.padding.get(
                "y_top"
            )
            shape_data[:, 1] = shape_data[:, 1] + self.ims_pixel_map.padding.get(
                "x_left"
            )
            updated_shapes.append(shape_data.astype(np.float32))

        self.viewer.layers["IMS ROIs"].data = updated_shapes

        # update fiducial pts
        updated_ims_fids = deepcopy(self.viewer.layers["IMS Fiducials"].data)
        updated_ims_fids[:, 0] += y_top
        updated_ims_fids[:, 1] += x_left
        updated_ims_fids[updated_ims_fids < 0] = 0
        self.viewer.layers["IMS Fiducials"].data = updated_ims_fids

        self._update_current_padding()
        self.ims_c.ims_ctl.pad_left.setText("")
        self.ims_c.ims_ctl.pad_right.setText("")
        self.ims_c.ims_ctl.pad_top.setText("")
        self.ims_c.ims_ctl.pad_bottom.setText("")

        self._update_output_size()

        # self.viewer.reset_view()

    def run_transformation(self) -> None:

        if (
            np.array_equal(
                self.last_transform, self.image_transformer.inverse_affine_np_mat_yx_um
            )
            is False
            and self.image_transformer.inverse_affine_np_mat_yx_um is not None
        ):
            target_tform_modality = (
                self.tform_c.tform_ctl.target_mode_combo.currentText()
            )
            if target_tform_modality == "Microscopy":
                self.viewer.layers[
                    "IMS Pixel Map"
                ].affine = self.image_transformer.affine_np_mat_yx_um

            elif target_tform_modality == "IMS":
                for im in self.micro_image_names:
                    self.viewer.layers[
                        im
                    ].affine = self.image_transformer.inverse_affine_np_mat_yx_um

                self.last_transform = deepcopy(
                    self.image_transformer.inverse_affine_np_mat_yx_um
                )

    def reset_data(self) -> None:
        while len(self.viewer.layers) > 0:
            self.viewer.layers.pop(0)

        # reset GUI elements
        self.data.ims_d._reset()
        self.data.micro_d._reset()
        self.tform_c.tform_ctl._reset()

        # reset all data objects
        self.microscopy_image = None
        self.ims_pixel_map = None
        self.image_transformer = ImageTransform()
        self.output_size = None
        self.last_transform = np.eye(3)
        self.micro_image_names = None

        # start with certain buttons disabled
        self.save_c.save_ctl.save_data.setEnabled(False)
        self.tform_c.tform_ctl.run_transform.setEnabled(False)

        return

    def _get_save_info(self):
        d = SavePopUp.show_dialog(
            self.viewer.window.qt_viewer, parent=self.viewer.window._qt_window
        )

        if d.completed:
            project_data = d.project_data
        else:
            project_data = None

        return project_data

    def _generate_pmap_coords_and_meta(self, project_name: str) -> None:

        padding_ims_metadata = self.ims_pixel_map.prepare_pmap_metadata()

        pmap_coord_data = self.ims_pixel_map.prepare_pmap_data_csv()

        micro_res = float(self.data.micro_d.res_info_input.text())
        ims_res = int(self.ims_pixel_map.ims_res)

        ims_pts_um = self.image_transformer.target_pts.astype(float)[:, [1, 0]]
        ims_pts_px = ims_pts_um / ims_res

        micro_pts_um = self.image_transformer.source_pts.astype(float)[:, [1, 0]]
        micro_pts_px = micro_pts_um / micro_res

        project_metadata = {
            "Project name": project_name,
            "IMS spatial resolution": ims_res,
            "Microscopy spatial resolution": micro_res,
            "IMS pixel map points (xy, microns)": ims_pts_um.tolist(),
            "IMS pixel map points (xy, px)": ims_pts_px.tolist(),
            "PAQ microscopy points (xy, microns)": micro_pts_um.tolist(),
            "PAQ microscopy points (xy, px)": micro_pts_px.tolist(),
            "Affine transformation matrix (xy,microns)": self.image_transformer.affine_np_mat_xy_um.astype(
                float
            ).tolist(),
            "Affine transformation matrix (yx,microns)": self.image_transformer.affine_np_mat_yx_um.astype(
                float
            ).tolist(),
            "Affine transformation matrix (xy,pixels)": self.image_transformer.affine_np_mat_xy_px.astype(
                float
            ).tolist(),
            "Affine transformation matrix (yx,pixels)": self.image_transformer.affine_np_mat_yx_px.astype(
                float
            ).tolist(),
            "Inverse Affine transformation matrix (xy,microns)": self.image_transformer.inverse_affine_np_mat_xy_um.astype(
                float
            ).tolist(),
            "Inverse Affine transformation matrix (yx,microns)": self.image_transformer.inverse_affine_np_mat_yx_um.astype(
                float
            ).tolist(),
            "Inverse Affine transformation matrix (xy,pixels)": self.image_transformer.inverse_affine_np_mat_xy_um.astype(
                float
            ).tolist(),
            "Inverse Affine transformation matrix (yx,pixels)": self.image_transformer.inverse_affine_np_mat_yx_um.astype(
                float
            ).tolist(),
            "Target output image size (xy, pixels)": self.image_transformer.output_size,
            "Target output image spacing (xy, um)": self.image_transformer.output_spacing,
            "PostIMS microscopy image": self.microscopy_image.image_filepath,
        }
        project_metadata.update(padding_ims_metadata)

        return project_metadata, pmap_coord_data

    def _transform_ims_coords_to_microscopy(self) -> np.ndarray:
        xy_coords = np.column_stack(
            [self.ims_pixel_map.x_coords_pad, self.ims_pixel_map.y_coords_pad]
        )

        # need to scale points to physical space to transform
        ims_res = float(self.data.ims_d.res_info_input.text())
        micro_res = float(self.data.micro_d.res_info_input.text())

        xy_coords_scaled = xy_coords * ims_res

        transformed_coords_micro = self.image_transformer.apply_transform_to_pts(
            xy_coords_scaled,
            self.image_transformer.affine_transform,
            xy_order="xy",
        )

        # get points back in IMS pixel space
        transformed_coords_ims = transformed_coords_micro / ims_res

        # get points back in microscopy pixel space
        transformed_coords_micro_px = transformed_coords_micro / micro_res

        return (
            transformed_coords_ims,
            transformed_coords_micro,
            transformed_coords_micro_px,
        )

    @thread_worker
    def _write_data(
        self, project_name: str, output_dir: str, output_filetype: str
    ) -> None:

        project_metadata, pmap_coord_data = self._generate_pmap_coords_and_meta(
            project_name
        )

        target_tform_modality = self.tform_c.tform_ctl.target_mode_combo.currentText()
        if target_tform_modality == "Microscopy":
            (
                transformed_coords_ims,
                transformed_coords_micro,
                transformed_coords_micro_px,
            ) = self._transform_ims_coords_to_microscopy()

            pmap_coord_data["x_micro_ims_px"] = transformed_coords_ims[:, 0]
            pmap_coord_data["y_micro_ims_px"] = transformed_coords_ims[:, 1]
            pmap_coord_data["x_micro_physical"] = transformed_coords_micro[:, 0]
            pmap_coord_data["y_micro_physical"] = transformed_coords_micro[:, 1]
            pmap_coord_data["x_micro_px"] = transformed_coords_micro_px[:, 0]
            pmap_coord_data["y_micro_px"] = transformed_coords_micro_px[:, 1]

        elif target_tform_modality == "IMS":
            ometiff_writer = OmeTiffWriter(
                self.microscopy_image,
                project_name,
                self.image_transformer.affine_transform,
                self.image_transformer.output_size,
                self.image_transformer.output_spacing,
                tile_size=512,
                output_dir=output_dir,
            )
            ometiff_writer.write_image()

        pmeta_out_fp = Path(output_dir) / f"{project_name}-IMSML-meta.json"

        with open(pmeta_out_fp, "w") as json_out:
            json.dump(project_metadata, json_out, indent=1)

        if output_filetype == ".h5":
            coords_out_fp = Path(output_dir) / f"{project_name}-IMSML-coords.h5"
            # pmap_coord_data.to_hdf(
            #     coords_out_fp, key="imsml-coords", complevel=1, index=False
            # )
            pmap_coords_to_h5(pmap_coord_data, coords_out_fp)
        elif output_filetype == ".csv":
            coords_out_fp = Path(output_dir) / f"{project_name}-IMSML-coords.csv"
            pmap_coord_data.to_csv(coords_out_fp, mode="w", index=False)

    def save_data(self) -> None:

        project_data = self._get_save_info()

        if project_data:
            save_worker = self._write_data(
                project_data.project_name,
                project_data.output_dir,
                project_data.output_filetype,
            )
            save_worker.start()
            return
        else:
            error_message = QErrorMessage(self)
            error_message.showMessage("Project data not provided")
            return

    def set_ims_res(self) -> None:
        if len(self.data.ims_d.res_info_input.text()) > 0:
            ims_res = float(self.data.ims_d.res_info_input.text())
            self.viewer.layers["IMS Pixel Map"].scale = (ims_res, ims_res)
            self.viewer.layers["IMS ROIs"].scale = (ims_res, ims_res)
            self.viewer.layers["IMS Fiducials"].scale = (ims_res, ims_res)
            self.viewer.layers["IMS Fiducials"].size = ims_res // 2
            if "Microscopy Fiducials" in self.viewer.layers:
                self.viewer.layers["Microscopy Fiducials"].size = ims_res // 2
            self.ims_pixel_map.ims_res = ims_res
            self.viewer.reset_view()
            self._update_output_size()

    def set_micro_res(self) -> None:
        if len(self.data.micro_d.res_info_input.text()) > 0:
            micro_res = float(self.data.micro_d.res_info_input.text())
            if self.micro_image_names:
                for micro_im in self.micro_image_names:
                    self.viewer.layers[micro_im].scale = (micro_res, micro_res)

                self.viewer.layers["Microscopy Fiducials"].scale = (
                    micro_res,
                    micro_res,
                )
                self.viewer.reset_view()
                self._update_output_spacing(micro_res)
                self._update_output_size()
            else:
                return None

    def _rotate_modality(self, modality: str, angle: Union[int, float]):
        if modality == "microscopy" and self.microscopy_image:
            image_size = self.viewer.layers[self.micro_image_names[0]].data.shape
            image_spacing = self.viewer.layers[self.micro_image_names[0]].scale
            cum_angle = angle + self._micro_rot
            if cum_angle > 360:
                cum_angle -= 360
            microscopy_transform = centered_transform(
                image_size, image_spacing, cum_angle
            )
            for micro_im in self.micro_image_names:
                self.viewer.layers[micro_im].affine = microscopy_transform
            self._micro_rot += angle

        if modality == "ims" and self.ims_pixel_map:
            cum_angle = angle + self._ims_rot
            if cum_angle > 360:
                cum_angle -= 360
            if len(self.viewer.layers["IMS Fiducials"].data) > 0:
                updated_fiducials = self.ims_pixel_map.rotate_coordinates(
                    rotation_angle=cum_angle,
                    fiducial_pts=self.viewer.layers["IMS Fiducials"].data[:, [1, 0]],
                )
                self.viewer.layers["IMS Fiducials"].data = updated_fiducials[:, [1, 0]]
            else:
                self.ims_pixel_map.rotate_coordinates(
                    rotation_angle=cum_angle,
                )

            self.viewer.layers[
                "IMS Pixel Map"
            ].data = self.ims_pixel_map.pixelmap_padded

        return


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return IMSMicroLink
