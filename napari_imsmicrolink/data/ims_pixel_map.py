from typing import Union, Tuple, List, Optional, Dict
from pathlib import Path
import sqlite3
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import cv2
from lxml import etree
import h5py
from napari_imsmicrolink.utils.points import apply_rotmat_points
from napari_imsmicrolink.utils.ims_coords import parse_tsf_coordinates


class PixelMapIMS:
    def __init__(
        self,
        data: Union[str, np.ndarray, Path],
        infer_regions: bool = True,
    ):

        """
        Container for IMS pixel coordinates for manipulation

        Parameters
        ----------
        data : n_pix x 3 (regions, x , y) np.ndarray or file path as string
        infer_regions : for imzML import, regions will be infered from non connected groups of pixels
        """
        if isinstance(data, (str, Path, np.ndarray)):
            data_imported = [data]
        else:
            data_imported = data

        self.data: List[Union[str, NDArray]] = data_imported
        self.ims_res: int = 1

        self.regions: Optional[NDArray] = None

        self.x_coords_orig: Optional[NDArray] = None
        self.y_coords_orig: Optional[NDArray] = None
        self.x_coords_min: Optional[NDArray] = None
        self.y_coords_min: Optional[NDArray] = None
        self.x_coords_pad: Optional[NDArray] = None
        self.y_coords_pad: Optional[NDArray] = None

        self.x_extent_pad: Optional[int] = None
        self.y_extent_pad: Optional[int] = None

        self._padding: Dict[str, int] = {
            "x_left": 0,
            "x_right": 0,
            "y_top": 0,
            "y_bottom": 0,
        }
        self.padding_microns: Dict[str, Union[int, float]] = {
            "x_left": 0,
            "x_right": 0,
            "y_top": 0,
            "y_bottom": 0,
        }

        for data in self.data:
            self.read_pixel_data(data, infer_regions=infer_regions)

        self._pixelmap_minimized: np.ndarray = self._make_pixel_map_at_ims(
            map_type="minimized", randomize=True
        )
        self.pixelmap_padded: NDArray = self._pixelmap_minimized

        self._shape_map_minimized = self._make_shape_map(map_type="minimized")

    def _check_bruker_sl(self, data_fp: str) -> bool:
        with open(data_fp) as f:
            first_line = f.readline()
        return any(e in first_line for e in ["flexImaging", "from R"])

    def _read_bruker_sl_rxy(
        self, data_fp: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        sl = pd.read_table(
            data_fp,
            header=None,
            skiprows=2,
            sep=" ",
            names=["X-pos", "Y-pos", "spot-name", "region"],
        )

        rxy = sl["spot-name"].str.split("X|Y", expand=True)

        # handle named regions
        regions = np.asarray(sl["region"])

        # TODO: defer to rxy regions over factorized if discrepancy exists
        x = np.asarray(rxy.iloc[:, 1], dtype=np.int32)
        y = np.asarray(rxy.iloc[:, 2], dtype=np.int32)

        return regions, x, y

    def _read_sqlite_rxy(
        self, data_fp: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        sqlite_db = sqlite3.connect(data_fp)

        c = sqlite_db.cursor()

        c.execute("SELECT RegionNumber, XIndexPos, YIndexPos FROM Spectra")

        rxy = np.array(c.fetchall())

        regions, x, y = rxy[:, 0], rxy[:, 1], rxy[:, 2]

        return regions, x, y

    def _read_h5(self, data_fp):
        with h5py.File(data_fp) as f:
            regions, x, y = (
                np.asarray(f["region"]),
                np.asarray(f["x"]),
                np.asarray(f["y"]),
            )

        return regions, x, y

    def _read_imzml_rxy(
        self, data_fp: str, infer_regions: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        elements = etree.iterparse(data_fp)

        coordinates = []
        for event, element in elements:
            if element.tag == "{http://psi.hupo.org/ms/mzml}spectrum":
                scan_elem = element.find(
                    "%sscanList/%sscan"
                    % ("{http://psi.hupo.org/ms/mzml}", "{http://psi.hupo.org/ms/mzml}")
                )
                x = scan_elem.find(
                    '%scvParam[@accession="IMS:1000050"]'
                    % "{http://psi.hupo.org/ms/mzml}"
                ).attrib["value"]
                y = scan_elem.find(
                    '%scvParam[@accession="IMS:1000051"]'
                    % "{http://psi.hupo.org/ms/mzml}"
                ).attrib["value"]
                coordinates.append((int(x), int(y)))

        rxy = np.column_stack(
            [
                np.zeros(len(coordinates), dtype=np.int32),
                np.asarray(coordinates, dtype=np.int32),
            ]
        )

        regions, x, y = rxy[:, 0], rxy[:, 1], rxy[:, 2]

        if infer_regions:
            pix_map_arr = np.zeros((np.max(y) + 1, np.max(x) + 1), dtype=np.uint8)
            pix_map_arr[y, x] = 255
            n_cc, cc = cv2.connectedComponents(pix_map_arr)

            cc_rs = []
            cc_xs = []
            cc_ys = []

            for roi in np.arange(1, n_cc, 1):
                cc_y, cc_x = np.where(cc == roi)
                cc_r = np.ones_like(cc_x, dtype=np.int32) * roi
                cc_rs.append(cc_r)
                cc_xs.append(cc_x)
                cc_ys.append(cc_y)

            cc_x = np.concatenate(cc_xs)
            cc_y = np.concatenate(cc_ys)
            cc_r = np.concatenate(cc_rs)

            infer_rxy = np.column_stack([cc_r, cc_x, cc_y])
            regions = np.asarray(
                pd.merge(pd.DataFrame(rxy), pd.DataFrame(infer_rxy), on=[1, 2])["0_y"]
            )

        return regions, x, y

    def _read_ims_microlink_csv(
        self, data_fp: str
    ) -> Tuple[
        pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series
    ]:
        allcoords = pd.read_csv(data_fp, comment="#")
        return (
            allcoords["regions"],
            allcoords["x_original"],
            allcoords["y_original"],
            allcoords["x_minimized"],
            allcoords["y_minimized"],
            allcoords["x_padded"],
            allcoords["y_padded"],
        )

    def read_pixel_data(
        self, data: Union[str, np.ndarray], infer_regions: bool
    ) -> None:
        data_round_trip = False
        if isinstance(data, np.ndarray):
            regions = data[:, 0]
            x = data[:, 1]
            y = data[:, 2]

        elif Path(data).suffix.lower() == ".txt":
            if self._check_bruker_sl(data) is False:
                raise ValueError(
                    "{} doesn't appear to be a bruker flexImaging spotlist".format(data)
                )
            regions, x, y = self._read_bruker_sl_rxy(data)

        elif Path(data).suffix.lower() == ".sqlite":

            regions, x, y = self._read_sqlite_rxy(data)

        elif Path(data).suffix.lower() == ".imzml":
            regions, x, y = self._read_imzml_rxy(str(data), infer_regions=infer_regions)

        elif Path(data).suffix.lower() == ".h5":
            regions, x, y = self._read_h5(str(data))

        elif Path(data).suffix.lower() == ".tsf":
            regions, x, y = parse_tsf_coordinates(str(data))

        elif Path(data).suffix.lower() == ".csv":
            (
                self.regions,
                self.x_coords_orig,
                self.y_coords_orig,
                self.x_coords_min,
                self.y_coords_min,
                self.x_coords_pad,
                self.y_coords_pad,
            ) = self._read_ims_microlink_csv(data)

            data_round_trip = True

        if data_round_trip is False:

            # assume there is no data if regions is None
            if self.regions is None:
                self.regions = regions
                self.x_coords_orig = x
                self.y_coords_orig = y
            else:
                offset_roi_no = int(np.max(self.regions) + 1)
                regions = regions + offset_roi_no

                self.regions = np.concatenate([self.regions, regions])
                self.x_coords_orig = np.concatenate([self.x_coords_orig, x])
                self.y_coords_orig = np.concatenate([self.y_coords_orig, y])

            self.x_coords_min = self.x_coords_orig - np.min(self.x_coords_orig)
            self.y_coords_min = self.y_coords_orig - np.min(self.y_coords_orig)

            self.x_coords_pad = self.x_coords_min
            self.y_coords_pad = self.y_coords_min

    def add_pixel_data(
        self,
        data: Union[str, np.ndarray],
        infer_regions: bool = True,
    ):
        if isinstance(data, (str, np.ndarray)):
            data_imported = [data]
        else:
            data_imported = data

        for data in data_imported:
            self.read_pixel_data(data, infer_regions=infer_regions)

        self.data.extend(data_imported)

        self._pixelmap_minimized: np.ndarray = self._make_pixel_map_at_ims(
            map_type="minimized", randomize=True
        )
        self.pixelmap_padded: NDArray = self._pixelmap_minimized
        self._shape_map_minimized = self._make_shape_map(map_type="minimized")

    def _get_xy_extents_coords(
        self, map_type: str = "minimized"
    ) -> Tuple[int, int, NDArray, NDArray]:
        if map_type == "minimized":
            y_extent = int(np.max(self.y_coords_min) + 1)
            x_extent = int(np.max(self.x_coords_min) + 1)
            y_coords = self.y_coords_min
            x_coords = self.x_coords_min

        elif map_type == "padded":
            y_extent = int(np.max(self.y_coords_pad) + 1)
            x_extent = int(np.max(self.x_coords_pad) + 1)
            y_coords = self.y_coords_pad
            x_coords = self.x_coords_pad

        elif map_type == "original":
            y_extent = int(np.max(self.y_coords_orig) + 1)
            x_extent = int(np.max(self.x_coords_orig) + 1)
            y_coords = self.y_coords_orig
            x_coords = self.x_coords_orig

        return y_extent, x_extent, y_coords, x_coords  # type:ignore

    def approx_polygon_contour(
        self, mask: np.ndarray, percent_arc_length: float = 0.01
    ) -> np.ndarray:
        """Approximate binary mask contours to polygon vertices using cv2.

        Parameters
        ----------
        mask : numpy.ndarray
            2-d numpy array of datatype np.uint8.
        percent_arc_length : float
            scaling of epsilon for polygon approximate vertices accuracy.
            maximum distance of new vertices from original.

        Returns
        -------
        numpy.ndarray
            returns an 2d array of vertices, rows: points, columns: y,x

        """

        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        if len(contours) > 1:
            contours = [contours[np.argmax([cnt.shape[0] for cnt in contours])]]

        epsilon = percent_arc_length * cv2.arcLength(contours[0], True)

        if len(contours[0]) > 1000:
            contours = cv2.approxPolyDP(contours[0], epsilon, True)
        elif len(contours[0]) == 1:
            ct1 = contours[0]
            ct2 = ct1 + 1
            contours = np.vstack([ct1, ct2])

        return np.squeeze(contours)[:, [1, 0]]

    def _approximate_roi(self, pix_map_in: np.ndarray, roi: int) -> np.ndarray:

        pix_map = np.zeros_like(pix_map_in)
        pix_map[pix_map_in == roi] = 255
        pix_map[pix_map_in != roi] = 0
        pix_map = pix_map.astype(np.uint8)

        while cv2.connectedComponents(pix_map)[0] > 2:
            pix_map = cv2.dilate(pix_map, np.ones((3, 3), np.uint8))

        return self.approx_polygon_contour(pix_map, 0.001)

    def _make_shape_map(
        self, map_type: str = "minimized"
    ) -> List[Tuple[str, np.ndarray]]:
        y_extent, x_extent, y_coords, x_coords = self._get_xy_extents_coords(
            map_type=map_type
        )

        region_names, region_indices = np.unique(self.regions, return_inverse=True)

        pix_map_arr = np.zeros((y_extent, x_extent), dtype=np.int32)

        pix_map_arr[y_coords, x_coords] = region_indices + 1

        ims_rois = [
            (region_name, self._approximate_roi(pix_map_arr, roi + 1))
            for region_name, roi in zip(region_names, np.unique(region_indices))
        ]

        return ims_rois

    def _make_pixel_map_at_ims(
        self, map_type: str = "minimized", randomize: bool = True
    ) -> np.ndarray:

        y_extent, x_extent, y_coords, x_coords = self._get_xy_extents_coords(map_type)

        pix_map_arr = np.zeros((y_extent, x_extent), dtype=np.uint8)

        if randomize is True:
            pix_map_arr[y_coords, x_coords] = np.random.randint(
                85, 255, len(y_coords), dtype=np.uint8
            )
        else:
            pix_map_arr[y_coords, x_coords] = 255

        return pix_map_arr

    def make_pixel_map_mz_fill(
        self, mz_vals: Union[list, np.ndarray], map_type: str = "minimized"
    ) -> np.ndarray:

        y_extent, x_extent, y_coords, x_coords = self._get_xy_extents_coords(map_type)

        pix_map_arr = np.zeros((y_extent, x_extent), dtype=np.float32)

        pix_map_arr[y_coords, x_coords] = mz_vals

        return pix_map_arr

    def delete_roi(self, roi_name: str, remove_padding: bool = True) -> None:
        roi_name_np = np.asarray(roi_name, dtype=self.regions.dtype)  # type:ignore
        non_roi_idx = self.regions != roi_name_np
        roi_idx = np.invert(non_roi_idx)

        self.regions = self.regions[non_roi_idx]  # type:ignore
        self.x_coords_orig = self.x_coords_orig[non_roi_idx]  # type:ignore
        self.y_coords_orig = self.y_coords_orig[non_roi_idx]  # type:ignore
        self.x_coords_min = self.x_coords_orig - np.min(self.x_coords_orig)
        self.y_coords_min = self.y_coords_orig - np.min(self.y_coords_orig)

        if remove_padding is True:
            self.x_coords_pad = self.x_coords_min
            self.y_coords_pad = self.y_coords_min
            self.x_extent_pad = np.max(self.x_coords_min) + 1
            self.y_extent_pad = np.max(self.y_coords_min) + 1
            self._pixelmap_minimized = self._make_pixel_map_at_ims(
                map_type="minimized", randomize=True
            )
            self.pixelmap_padded = self._pixelmap_minimized
        else:
            self.pixelmap_padded[
                self.y_coords_pad[roi_idx], self.x_coords_pad[roi_idx]  # type:ignore
            ] = 0
            self.x_coords_pad = self.x_coords_pad[non_roi_idx]  # type:ignore
            self.y_coords_pad = self.y_coords_pad[non_roi_idx]  # type:ignore

    @property
    def padding(self):
        return self._padding

    @padding.setter
    def padding(self, pad_values: Dict[str, int]) -> None:
        for k in pad_values.keys():
            self._padding[k] += pad_values[k]
            if self._padding[k] < 0:
                self._padding[k] = 0

        self._refresh_coords_after_pad()
        self._get_padding_microns()

    def _get_padding_microns(self) -> None:
        for k in self._padding.keys():
            self.padding_microns[k] = int(self._padding[k] * self.ims_res)

    def _refresh_coords_after_pad(self) -> None:

        self.pixelmap_padded = np.pad(
            self._pixelmap_minimized,
            (
                (self.padding.get("y_top"), self.padding.get("y_bottom")),
                (self.padding.get("x_left"), self.padding.get("x_right")),
            ),
        )

        self.x_coords_pad = self.x_coords_min + self.padding.get("x_left")
        self.y_coords_pad = self.y_coords_min + self.padding.get("y_top")

        self.y_extent_pad = self.pixelmap_padded.shape[0]
        self.x_extent_pad = self.pixelmap_padded.shape[1]

    def reset_padding(self) -> None:
        self.pixelmap_padded = self._pixelmap_minimized
        self.y_extent_pad = self.pixelmap_padded.shape[0]
        self.x_extent_pad = self.pixelmap_padded.shape[1]
        self._padding = {"x_left": 0, "x_right": 0, "y_top": 0, "y_bottom": 0}
        self._get_padding_microns()

    def prepare_pmap_metadata(self) -> Dict:
        return {
            "Pixel Map Datasets Files": self.data,
            "padding": {
                "x_left_padding (px)": self.padding["x_left"],
                "x_right_padding (px)": self.padding["x_right"],
                "y_top_padding (px)": self.padding["y_top"],
                "y_bottom_padding (px)": self.padding["y_bottom"],
                "x_left_padding (um)": self.padding_microns["x_left"],
                "x_right_padding (um)": self.padding_microns["x_right"],
                "y_top_padding (um)": self.padding_microns["y_top"],
                "y_bottom_padding (um)": self.padding_microns["y_bottom"],
            },
        }

    def prepare_pmap_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "regions": self.regions,
                "x_original": self.x_coords_orig,
                "y_original": self.y_coords_orig,
                "x_minimized": self.x_coords_min,
                "y_minimized": self.y_coords_min,
                "x_padded": self.x_coords_pad,
                "y_padded": self.y_coords_pad,
            }
        )

    def rotate_coordinates(
        self, rotation_angle: int, fiducial_pts: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:

        # translate center of mass to be near origin
        mean_x = np.max(self.x_coords_min) / 2
        mean_y = np.max(self.y_coords_min) / 2
        center_point = [mean_x, mean_y]

        # rotate around origin
        new_padding = dict()
        if rotation_angle in [90, -270]:
            rotmat = np.asarray([[0, 1], [-1, 0]])
            new_padding.update(
                {
                    "x_left": self._padding["y_top"],
                    "x_right": self._padding["y_bottom"],
                    "y_top": self._padding["x_right"],
                    "y_bottom": self._padding["x_left"],
                }
            )
            recenter_point = [mean_x, mean_y]

        elif rotation_angle in [-90, 270]:
            rotmat = np.asarray([[0, -1], [1, 0]])
            new_padding.update(
                {
                    "x_left": self._padding["y_bottom"],
                    "x_right": self._padding["y_top"],
                    "y_top": self._padding["x_left"],
                    "y_bottom": self._padding["x_right"],
                }
            )
            recenter_point = [mean_x, mean_y]

        elif rotation_angle in [-180, 180]:
            rotmat = np.asarray([[-1, 0], [0, -1]])
            new_padding.update(
                {
                    "x_left": self._padding["x_right"],
                    "x_right": self._padding["x_left"],
                    "y_top": self._padding["y_bottom"],
                    "y_bottom": self._padding["y_top"],
                }
            )
            # recenter_point = [mean_y, mean_x]
            recenter_point = [mean_x, mean_y]

        point_mat = np.column_stack([self.x_coords_min, self.y_coords_min])

        rotcoords = apply_rotmat_points(rotmat, point_mat, center_point, recenter_point)
        rotcoords = np.round(rotcoords).astype(np.uint32)

        self.x_coords_min = rotcoords[:, 0]
        self.y_coords_min = rotcoords[:, 1]
        self._pixelmap_minimized = self._make_pixel_map_at_ims(
            map_type="minimized", randomize=True
        )
        self.reset_padding()
        self.padding = new_padding
        self._shape_map_minimized = self._make_shape_map(map_type="minimized")

        if fiducial_pts is not None:
            return apply_rotmat_points(
                rotmat, fiducial_pts, center_point, recenter_point
            )
        else:
            return