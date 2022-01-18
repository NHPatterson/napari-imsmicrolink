from typing import List, Tuple, Optional, Union
from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import dask.array as da
import SimpleITK as sitk

from napari_plugin_engine import napari_hook_implementation
import napari
from napari.qt.threading import thread_worker
from napari.utils import progress
from qtpy.QtWidgets import QWidget, QVBoxLayout, QErrorMessage
from qtpy.QtCore import QTimer

from napari_imsmicrolink.qwidgets import (
    DataControl,
    IMSControl,
    TformControl,
    SaveControl,
    SavePopUp,
)
from napari_imsmicrolink.data import (
    PixelMapIMS,
    MicroRegImage,
    CziRegImage,
    TiffFileRegImage,
    TIFFFILE_EXTS,
    OmeTiffWriter,
    ImageTransform,
)

from napari_imsmicrolink.utils.file import open_file_dialog, _generate_ims_fp_info
from napari_imsmicrolink.utils.color import COLOR_HEXES
from napari_imsmicrolink.utils.image import centered_transform, grayscale
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
        self._data: DataControl = DataControl(self)
        self._ims_c: IMSControl = IMSControl(self)
        self._tform_c: TformControl = TformControl(self)
        self._save_c: SaveControl = SaveControl(self)

        # layout widgets in GUI
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self._data.data_frame)
        self.layout().addWidget(self._ims_c.ims_ctrl_frame)
        self.layout().addWidget(self._tform_c.tform_ctrl_frame)
        self.layout().addWidget(self._save_c.save_ctrl_frame)
        self.layout().addStretch(1)

        # button actions for widgets
        # data reading
        self._data.ims_d.add_ims_data.clicked.connect(self.read_ims_data)
        self._data.ims_d.delete_btn.clicked.connect(self._delete_ims_roi)
        self._data.micro_d.add_micro_data.clicked.connect(self.read_micro_data)

        # ims canvas control
        self._ims_c.ims_ctl.do_padding_btn.clicked.connect(self._pad_ims_canvas)
        self._ims_c.ims_ctl.micron_check.clicked.connect(
            lambda: self._set_pad_by("micron")
        )
        self._ims_c.ims_ctl.pixel_check.clicked.connect(
            lambda: self._set_pad_by("pixel")
        )

        # transform control
        self._tform_c.tform_ctl.reset_transform.clicked.connect(self.reset_transform)
        self._tform_c.tform_ctl.run_transform.clicked.connect(self.run_transformation)
        self._tform_c.rot_ctl.rot_mi_90cw.clicked.connect(
            lambda: self._rotate_modality("microscopy", -90)
        )
        self._tform_c.rot_ctl.rot_mi_90ccw.clicked.connect(
            lambda: self._rotate_modality("microscopy", 90)
        )
        self._tform_c.rot_ctl.rot_mi_180cw.clicked.connect(
            lambda: self._rotate_modality("microscopy", 180)
        )
        self._tform_c.rot_ctl.rot_mi_180ccw.clicked.connect(
            lambda: self._rotate_modality("microscopy", 180)
        )
        self._tform_c.rot_ctl.rot_ims_90cw.clicked.connect(
            lambda: self._rotate_modality("ims", -90)
        )
        self._tform_c.rot_ctl.rot_ims_90ccw.clicked.connect(
            lambda: self._rotate_modality("ims", 90)
        )
        # self.tform_c.rot_ctl.rot_ims_180cw.clicked.connect(
        #     lambda: self._rotate_modality("ims", 180)
        # )
        # self.tform_c.rot_ctl.rot_ims_180ccw.clicked.connect(
        #     lambda: self._rotate_modality("ims", 180)
        # )

        # save control
        self._save_c.save_ctl.reset_data.clicked.connect(self.reset_data)
        self._save_c.save_ctl.save_data.clicked.connect(self.save_data)

        # update spatial resolutions
        self._ims_res_timer = QTimer()
        self._ims_res_timer.setSingleShot(True)
        self._ims_res_timer.timeout.connect(self._set_ims_res)
        self._data.ims_d.res_info_input.textChanged.connect(self._ims_res_timing)

        self._micro_res_timer = QTimer()
        self._micro_res_timer.setSingleShot(True)
        self._micro_res_timer.timeout.connect(self._set_micro_res)
        self._data.micro_d.res_info_input.textChanged.connect(self._micro_res_timing)

        # start with certain buttons disabled
        self._save_c.save_ctl.save_data.setEnabled(False)
        self._tform_c.tform_ctl.run_transform.setEnabled(False)

    def read_ims_data(self, file_paths: Optional[Union[str, List[str]]] = None) -> None:
        """
        Collect data path and import IMS data, generating IMS Pixel Map.
        """
        if not file_paths:
            file_paths = open_file_dialog(
                self,
                single=False,
                wd="",
                name="Open IMS Pixel Maps",
                file_types="All Files (*);;Tiff files (*.txt,*.tif);;sqlite files (*.sqlite);;imzML (*.imzML)",
            )
        # if statement checks that open file dialog returned something
        if file_paths:
            if self.ims_pixel_map:
                self.ims_pixel_map.add_pixel_data(file_paths)
            else:
                self.ims_pixel_map = PixelMapIMS(file_paths)
                if len(self._data.ims_d.res_info_input.text()) == 0:
                    self._data.ims_d.res_info_input.setText("1")

            self._add_ims_data()

            fp_names, fp_names_full = _generate_ims_fp_info(self.ims_pixel_map.data)

            self._data.ims_d.file_info_input.setText(fp_names)
            self._data.ims_d.file_info_input.setToolTip(fp_names_full)

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

        self.viewer.layers["IMS Fiducials"].events.data.connect(self._get_target_pts)

    def _generate_ims_rois(self, map_type="minimized"):
        if map_type == "minimized":
            shapes = self.ims_pixel_map._shape_map_minimized
        else:
            shapes = self.ims_pixel_map._make_shape_map(map_type=map_type)

        shape_names = [shape[0] for shape in shapes]
        shape_data = [shape[1] for shape in shapes]
        shape_props = {"name": shape_names}
        shape_text = {
            "text": "{name}",
            "color": "white",
            "anchor": "center",
            "size": 12,
        }
        return shape_data, shape_names, shape_props, shape_text

    def _add_shape_names_combo(self, shape_names) -> None:
        for sname in shape_names:
            self._data.ims_d.delete_box.addItem(str(sname))

    def _add_ims_rois(self) -> None:

        shape_data, shape_names, shape_props, shape_text = self._generate_ims_rois()

        face_colors = []
        for i in range(len(shape_names)):
            if i > len(COLOR_HEXES) - 2:
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

        self._add_shape_names_combo(shape_names)

    def _collect_ims_res(self) -> float:
        if len(self._data.ims_d.res_info_input.text()) > 0:
            return float(self._data.ims_d.res_info_input.text())
        else:
            return 1.0

    def _add_ims_data(self) -> None:

        ims_res = self._collect_ims_res()

        if "IMS Pixel Map" in self.viewer.layers:
            self.viewer.layers.pop("IMS Pixel Map")
            self.viewer.layers.pop("IMS ROIs")
            self.viewer.layers.pop("IMS Fiducials")

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
            self._get_source_pts
        )

    def _get_target_pts(self, _) -> None:
        self._tform_c.tform_ctl.pt_table.add_point_data(
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
            self._tform_c.tform_ctl.tform_error.setText(
                f"{np.round(self.image_transformer.point_reg_error, 2)}"
            )
            if self.image_transformer.point_reg_error < self.ims_pixel_map.ims_res / 2:
                self._tform_c.tform_ctl.tform_error.setStyleSheet("color: green")
            else:
                self._tform_c.tform_ctl.tform_error.setStyleSheet("color: red")

    def _get_source_pts(self, _) -> None:

        self._tform_c.tform_ctl.pt_table.add_point_data(
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
            self._tform_c.tform_ctl.tform_error.setText(
                f"{np.round(self.image_transformer.point_reg_error, 3)}"
            )

    @thread_worker
    def _process_micro_data(self, file_path: str) -> Tuple[MicroRegImage, np.ndarray]:
        if Path(file_path).suffix.lower() == ".czi":
            microscopy_image = CziRegImage(file_path)
            micro_data = microscopy_image.get_dask_pyr()
        elif Path(file_path).suffix.lower() in TIFFFILE_EXTS:
            microscopy_image = TiffFileRegImage(file_path)
            micro_data = microscopy_image.get_dask_pyr()
        else:
            microscopy_image = MicroRegImage(file_path)
            dask_im = microscopy_image.pyr_levels_dask[1]
            if microscopy_image.is_rgb:
                micro_data = grayscale(dask_im)
            else:
                micro_data = []
                for i in range(dask_im.shape[0]):
                    im = sitk.GetImageFromArray(dask_im[i, :, :].compute())
                    im = sitk.RescaleIntensity(im)
                    im = sitk.GetArrayFromImage(im)
                    micro_data.append(im)

                micro_data = np.stack(micro_data)

        return microscopy_image, micro_data

    def _add_micro_data(
        self, data: Tuple[MicroRegImage, Union[np.ndarray, List[da.Array]]]
    ) -> None:

        self.microscopy_image = data[0]
        if self.microscopy_image.is_rgb:
            c_axis = None
        else:
            if isinstance(data[1], list):
                d = data[1][0]
            else:
                d = data[1]

            if len(d.shape) > 2:
                c_axis = 0
            else:
                c_axis = None

        if self.microscopy_image.is_rgb:
            cnames = self.microscopy_image.cnames[0]
        elif c_axis is None:
            cnames = self.microscopy_image.cnames[0]
        else:
            cnames = self.microscopy_image.cnames

        file_path = self.microscopy_image.image_filepath
        fp_name = Path(file_path).name
        fp_name_full = Path(file_path).as_posix()
        self.viewer.add_image(
            data[1],
            name=cnames,
            scale=(
                self.microscopy_image.base_layer_pixel_res,
                self.microscopy_image.base_layer_pixel_res,
            ),
            channel_axis=c_axis,
        )

        self._update_output_spacing(self.microscopy_image.base_layer_pixel_res)
        self.micro_image_names = self.microscopy_image.cnames
        self._data.micro_d.file_info_input.setText(fp_name)
        self._data.micro_d.file_info_input.setToolTip(fp_name_full)
        self._data.micro_d.res_info_input.setText(
            str(self.microscopy_image.base_layer_pixel_res)
        )
        self._data.micro_d.res_info_input.setText(
            str(self.microscopy_image.base_layer_pixel_res)
        )
        self._add_micro_fiducials()
        self.viewer.reset_view()

    def read_micro_data(self, file_path: Optional[str] = None) -> None:
        if not file_path:
            file_path = open_file_dialog(
                self,
                single=True,
                wd="",
                name="Open post-acquisition microscopy image",
                file_types="All Files (*);;Tiff files (*.tiff,*.tif);;czi files (*.czi)",
            )
        if file_path:
            pbr = progress(total=0)
            pbr.set_description("reading microscopy image")
            micro_reader_worker = self._process_micro_data(file_path)
            micro_reader_worker.returned.connect(self._add_micro_data)
            micro_reader_worker.start()
            micro_reader_worker.finished.connect(
                lambda: pbr.set_description("finished reading microscopy image")
            )
            micro_reader_worker.finished.connect(pbr.close)

    def _check_for_tforms(self) -> None:
        if self.image_transformer.affine_transform:
            self._save_c.save_ctl.save_data.setEnabled(True)
            self._tform_c.tform_ctl.run_transform.setEnabled(True)

    def _update_output_size(self) -> None:
        if self.ims_pixel_map:
            if len(self._data.micro_d.res_info_input.text()) > 0:
                micro_res = float(self._data.micro_d.res_info_input.text())
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

    def _set_pad_by(self, unit: str) -> None:
        if unit == "micron":
            if self._ims_c.ims_ctl.pixel_check.isChecked():
                self._ims_c.ims_ctl.pixel_check.setChecked(False)
            else:
                self._ims_c.ims_ctl.pixel_check.setChecked(True)
        else:
            if self._ims_c.ims_ctl.micron_check.isChecked():
                self._ims_c.ims_ctl.micron_check.setChecked(False)
            else:
                self._ims_c.ims_ctl.micron_check.setChecked(True)

        if self.ims_pixel_map:
            self._update_current_padding()

    def _update_current_padding(self) -> None:
        if self._ims_c.ims_ctl.micron_check.isChecked():
            self._ims_c.ims_ctl.cur_pad_left.setText(
                str(self.ims_pixel_map.padding_microns.get("x_left"))
            )
            self._ims_c.ims_ctl.cur_pad_right.setText(
                str(self.ims_pixel_map.padding_microns.get("x_right"))
            )
            self._ims_c.ims_ctl.cur_pad_top.setText(
                str(self.ims_pixel_map.padding_microns.get("y_top"))
            )
            self._ims_c.ims_ctl.cur_pad_bottom.setText(
                str(self.ims_pixel_map.padding_microns.get("y_bottom"))
            )
        else:
            self._ims_c.ims_ctl.cur_pad_left.setText(
                str(self.ims_pixel_map.padding.get("x_left"))
            )
            self._ims_c.ims_ctl.cur_pad_right.setText(
                str(self.ims_pixel_map.padding.get("x_right"))
            )
            self._ims_c.ims_ctl.cur_pad_top.setText(
                str(self.ims_pixel_map.padding.get("y_top"))
            )
            self._ims_c.ims_ctl.cur_pad_bottom.setText(
                str(self.ims_pixel_map.padding.get("y_bottom"))
            )

    def _pad_shapes(self):
        # update IMS ROIs
        shape_data, shape_names, shape_props, _ = self._generate_ims_rois(
            map_type="padded"
        )
        self.viewer.layers["IMS ROIs"].data = shape_data
        self.viewer.layers["IMS ROIs"].properties = shape_props
        self.viewer.layers["IMS ROIs"].text.values = shape_names

    def _get_pad_values(self) -> Tuple[int, int, int, int]:
        x_left = (
            int(self._ims_c.ims_ctl.pad_left.text())
            if len(self._ims_c.ims_ctl.pad_left.text()) > 0
            else 0
        )
        x_right = (
            int(self._ims_c.ims_ctl.pad_right.text())
            if len(self._ims_c.ims_ctl.pad_right.text()) > 0
            else 0
        )
        y_top = (
            int(self._ims_c.ims_ctl.pad_top.text())
            if len(self._ims_c.ims_ctl.pad_top.text()) > 0
            else 0
        )
        y_bottom = (
            int(self._ims_c.ims_ctl.pad_bottom.text())
            if len(self._ims_c.ims_ctl.pad_bottom.text()) > 0
            else 0
        )
        return x_left, x_right, y_top, y_bottom

    def _pad_ims_canvas(self) -> None:
        x_left, x_right, y_top, y_bottom = self._get_pad_values()

        ims_res = self._collect_ims_res()

        if self._ims_c.ims_ctl.micron_check.isChecked():
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
        self._pad_shapes()

        # update fiducial pts
        updated_ims_fids = deepcopy(self.viewer.layers["IMS Fiducials"].data)
        updated_ims_fids[:, 0] += y_top
        updated_ims_fids[:, 1] += x_left
        updated_ims_fids[updated_ims_fids < 0] = 0
        self.viewer.layers["IMS Fiducials"].data = updated_ims_fids

        self._update_current_padding()
        self._ims_c.ims_ctl.pad_left.setText("")
        self._ims_c.ims_ctl.pad_right.setText("")
        self._ims_c.ims_ctl.pad_top.setText("")
        self._ims_c.ims_ctl.pad_bottom.setText("")

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
                self._tform_c.tform_ctl.target_mode_combo.currentText()
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
        self._data.ims_d._reset()
        self._data.micro_d._reset()
        self._tform_c.tform_ctl._reset()

        # reset all data objects
        self.microscopy_image = None
        self.ims_pixel_map = None
        self.image_transformer = ImageTransform()
        self.output_size = None
        self.last_transform = np.eye(3)
        self.micro_image_names = None

        # start with certain buttons disabled
        self._save_c.save_ctl.save_data.setEnabled(False)
        self._tform_c.tform_ctl.run_transform.setEnabled(False)

        self._ims_c.ims_ctl.cur_pad_left.setText("0")
        self._ims_c.ims_ctl.cur_pad_right.setText("0")
        self._ims_c.ims_ctl.cur_pad_top.setText("0")
        self._ims_c.ims_ctl.cur_pad_bottom.setText("0")

        return

    def _get_save_info(self):
        d = SavePopUp.show_dialog(self, parent=self)

        if d.completed:
            project_data = d.project_data
        else:
            project_data = None

        return project_data

    def _generate_pmap_coords_and_meta(self, project_name: str) -> None:

        padding_ims_metadata = self.ims_pixel_map.prepare_pmap_metadata()

        pmap_coord_data = self.ims_pixel_map.prepare_pmap_dataframe()

        micro_res = float(self._data.micro_d.res_info_input.text())
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
        ims_res = float(self._data.ims_d.res_info_input.text())
        micro_res = float(self._data.micro_d.res_info_input.text())

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

    def _remove_ims_roi_in_viewer(self, roi_name):
        rm_idx = np.where(
            self.viewer.layers["IMS ROIs"].properties["name"].astype(str) == roi_name
        )[0]
        for idx in rm_idx:
            self.viewer.layers["IMS ROIs"].data.pop(int(idx))

        for k in self.viewer.layers["IMS ROIs"].properties:
            self.viewer.layers["IMS ROIs"].properties[k] = np.delete(
                self.viewer.layers["IMS ROIs"].properties[k], rm_idx, axis=0
            )

        return

    def _delete_ims_roi(self):
        roi_name = self._data.ims_d.delete_box.currentText()
        self.ims_pixel_map.delete_roi(roi_name, remove_padding=False)
        self.viewer.layers["IMS Pixel Map"].data = self.ims_pixel_map.pixelmap_padded
        shape_names = np.unique(self.ims_pixel_map.regions).astype(str)
        self._data.ims_d.delete_box.clear()
        self._add_shape_names_combo(shape_names)
        # self._remove_ims_roi_in_viewer(roi_name)
        self._pad_shapes()

    @thread_worker
    def _write_data(
        self, project_name: str, output_dir: str, output_filetype: str
    ) -> None:

        project_metadata, pmap_coord_data = self._generate_pmap_coords_and_meta(
            project_name
        )

        target_tform_modality = self._tform_c.tform_ctl.target_mode_combo.currentText()
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

        if target_tform_modality == "IMS":
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

    def save_data(self) -> None:

        project_data = self._get_save_info()

        if project_data:
            pbr = progress(total=0)
            pbr.set_description(f"Saving '{project_data.project_name}' project data")
            save_worker = self._write_data(
                project_data.project_name,
                project_data.output_dir,
                project_data.output_filetype,
            )
            save_worker.start()
            save_worker.finished.connect(
                lambda: pbr.set_description("Finished saving data")
            )
            save_worker.finished.connect(pbr.close)
            return
        else:
            error_message = QErrorMessage(self)
            error_message.showMessage("Project data not provided")
            return

    def _set_ims_res(self) -> None:
        if len(self._data.ims_d.res_info_input.text()) > 0:
            ims_res = float(self._data.ims_d.res_info_input.text())
            self.viewer.layers["IMS Pixel Map"].scale = (ims_res, ims_res)
            self.viewer.layers["IMS ROIs"].scale = (ims_res, ims_res)
            self.viewer.layers["IMS Fiducials"].scale = (ims_res, ims_res)
            self.viewer.layers["IMS Fiducials"].size = ims_res // 2
            if "Microscopy Fiducials" in self.viewer.layers:
                self.viewer.layers["Microscopy Fiducials"].size = ims_res // 2
            self.ims_pixel_map.ims_res = ims_res
            self.viewer.reset_view()
            self._update_output_size()

    def _set_micro_res(self) -> None:
        if len(self._data.micro_d.res_info_input.text()) > 0:
            micro_res = float(self._data.micro_d.res_info_input.text())
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
            if isinstance(self.viewer.layers[self.micro_image_names[0]].data, list):
                image_size = self.viewer.layers[self.micro_image_names[0]].data[0].shape
            else:
                image_size = self.viewer.layers[self.micro_image_names[0]].data.shape

            if self.microscopy_image.is_rgb:
                image_size = image_size[:2]
            else:
                image_size = image_size[1:]

            image_spacing = self.viewer.layers[self.micro_image_names[0]].scale
            cum_angle = angle + self._micro_rot
            if cum_angle > 360:
                cum_angle -= 360
            microscopy_transform = centered_transform(
                image_size, image_spacing, cum_angle
            )
            for micro_im in self.micro_image_names:
                self.viewer.layers[micro_im].affine = microscopy_transform
            self.viewer.layers["Microscopy Fiducials"].affine = microscopy_transform
            self._micro_rot += angle

        if modality == "ims" and self.ims_pixel_map:

            if len(self.viewer.layers["IMS Fiducials"].data) > 0:
                updated_fiducials = self.ims_pixel_map.rotate_coordinates(
                    rotation_angle=angle,
                    fiducial_pts=self.viewer.layers["IMS Fiducials"].data[:, [1, 0]],
                )
                self.viewer.layers["IMS Fiducials"].data = updated_fiducials[:, [1, 0]]
            else:
                self.ims_pixel_map.rotate_coordinates(
                    rotation_angle=angle,
                )

            self.viewer.layers[
                "IMS Pixel Map"
            ].data = self.ims_pixel_map.pixelmap_padded
            shape_data, *_ = self._generate_ims_rois()
            self._pad_shapes()
            self._ims_rot += angle
            self._update_output_size()

        return

    def reset_transform(self) -> None:
        for micro_im in self.micro_image_names:
            self.viewer.layers[micro_im].affine = np.eye(3)
        self.viewer.layers["Microscopy Fiducials"].affine = np.eye(3)
        self._micro_rot = 0

    def _ims_res_timing(self) -> None:
        """Wait until there are no changes for 0.5 second before making changes."""
        self._ims_res_timer.start(500)

    def _micro_res_timing(self) -> None:
        """Wait until there are no changes for 0.5 second before making changes."""
        self._micro_res_timer.start(500)


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return IMSMicroLink
