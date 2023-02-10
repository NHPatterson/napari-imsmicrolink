from typing import Union, Tuple, Optional
from copy import deepcopy
import numpy as np
import SimpleITK as sitk


class ImageTransform:
    def __init__(self):
        self.output_size: Optional[Tuple[int, int]] = None
        self.output_spacing: Optional[Tuple[float, float]] = None
        self.source_pts: np.ndarray = np.empty((0, 2), dtype=np.float32)
        self.target_pts: np.ndarray = np.empty((0, 2), dtype=np.float32)

        self.affine_transform: Optional[sitk.AffineTransform] = None
        self.affine_np_mat_xy_um: Optional[np.ndarray] = None
        self.affine_np_mat_yx_um: Optional[np.ndarray] = None
        self.affine_np_mat_xy_px: Optional[np.ndarray] = None
        self.affine_np_mat_yx_px: Optional[np.ndarray] = None
        self.inverse_affine_transform: Optional[sitk.AffineTransform] = None
        self.inverse_affine_np_mat_xy_um: Optional[np.ndarray] = None
        self.inverse_affine_np_mat_yx_um: Optional[np.ndarray] = None
        self.inverse_affine_np_mat_xy_px: Optional[np.ndarray] = None
        self.inverse_affine_np_mat_yx_px: Optional[np.ndarray] = None

        self.point_reg_error: float = float("inf")

    def add_points(
        self,
        pts: np.ndarray,
        round: bool = False,
        src_or_tgt: str = "source",
        scaling: Union[float, int] = 1,
    ) -> None:
        pts = deepcopy(pts)
        if round:
            pts = np.round(pts)
        pts *= scaling

        if src_or_tgt == "source":
            self.source_pts = pts
        elif src_or_tgt == "target":
            self.target_pts = pts

        self._compute_transform()

    def _compute_transform(self) -> None:
        if (
            self.source_pts.shape[0] == self.target_pts.shape[0]
            and self.source_pts.shape[0] > 3
        ):
            target_pts_flat = [c for p in self.target_pts[:, [1, 0]] for c in p]
            source_pts_flat = [c for p in self.source_pts[:, [1, 0]] for c in p]

            self.affine_transform = sitk.AffineTransform(
                sitk.LandmarkBasedTransformInitializer(
                    sitk.AffineTransform(2), target_pts_flat, source_pts_flat
                )
            )
            self.inverse_affine_transform = sitk.AffineTransform(
                self.affine_transform.GetInverse()
            )
            self._compute_pt_euclidean_error()
            self._get_np_matrices()

    @staticmethod
    def convert_sitk_to_np_matrix(
        transform: sitk.AffineTransform,
        use_np_ordering: bool = False,
        n_dim: int = 3,
        to_px_idx: bool = False,
        output_spacing: Tuple[float, float] = (1, 1),
    ) -> np.ndarray:

        if use_np_ordering is True:
            order = slice(None, None, -1)
        else:
            order = slice(None, None, 1)

        # pull transform values
        rotmat = np.array(transform.GetMatrix()[order]).reshape(2, 2)
        center = np.array(transform.GetCenter()[order])
        translation = np.array(transform.GetTranslation()[order])

        if to_px_idx is True:
            phys_to_index = 1 / np.asarray(output_spacing[0]).astype(np.float64)
            center *= phys_to_index
            translation *= phys_to_index

        # construct matrix
        full_matrix = np.eye(n_dim)
        full_matrix[0:2, 0:2] = rotmat
        full_matrix[0:2, n_dim - 1] = -np.dot(rotmat, center) + translation + center

        return full_matrix

    def _get_np_matrices(self, n_dim: int = 3):

        self.affine_np_mat_yx_um = self.convert_sitk_to_np_matrix(
            self.affine_transform,
            use_np_ordering=True,
            n_dim=n_dim,
            to_px_idx=False,
            output_spacing=self.output_spacing,
        )
        self.affine_np_mat_xy_um = self.convert_sitk_to_np_matrix(
            self.affine_transform,
            use_np_ordering=False,
            n_dim=n_dim,
            to_px_idx=False,
            output_spacing=self.output_spacing,
        )

        self.affine_np_mat_yx_px = self.convert_sitk_to_np_matrix(
            self.affine_transform,
            use_np_ordering=True,
            n_dim=n_dim,
            to_px_idx=True,
            output_spacing=self.output_spacing,
        )
        self.affine_np_mat_xy_px = self.convert_sitk_to_np_matrix(
            self.affine_transform,
            use_np_ordering=False,
            n_dim=n_dim,
            to_px_idx=True,
            output_spacing=self.output_spacing,
        )
        self.inverse_affine_np_mat_yx_um = self.convert_sitk_to_np_matrix(
            self.inverse_affine_transform,
            use_np_ordering=True,
            n_dim=n_dim,
            to_px_idx=False,
            output_spacing=self.output_spacing,
        )
        self.inverse_affine_np_mat_yx_px = self.convert_sitk_to_np_matrix(
            self.inverse_affine_transform,
            use_np_ordering=True,
            n_dim=n_dim,
            to_px_idx=True,
            output_spacing=self.output_spacing,
        )
        self.inverse_affine_np_mat_xy_um = self.convert_sitk_to_np_matrix(
            self.inverse_affine_transform,
            use_np_ordering=False,
            n_dim=n_dim,
            to_px_idx=False,
            output_spacing=self.output_spacing,
        )
        self.inverse_affine_np_mat_xy_px = self.convert_sitk_to_np_matrix(
            self.inverse_affine_transform,
            use_np_ordering=False,
            n_dim=n_dim,
            to_px_idx=True,
            output_spacing=self.output_spacing,
        )

    def _compute_pt_euclidean_error(self):
        tformed_pts = []
        for pt in self.target_pts[:, [1, 0]]:
            tformed_pt = self.affine_transform.TransformPoint(pt)
            tformed_pts.append(tformed_pt[::-1])
        tformed_pts = np.asarray(tformed_pts)

        self.point_reg_error = float(
            np.sqrt(np.sum((self.source_pts - tformed_pts) ** 2))
        )

    @staticmethod
    def apply_transform_to_pts(
        point_set: np.ndarray, transform: sitk.AffineTransform, xy_order: str = "xy"
    ) -> np.ndarray:
        if xy_order == "yx":
            point_set = point_set[:, [1, 0]]

        transformed_pts = []
        for pt in point_set:
            transformed_pts.append(transform.TransformPoint(pt))
        transformed_pts = np.asarray(transformed_pts)

        return transformed_pts
