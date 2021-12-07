import pytest
import numpy as np
import SimpleITK as sitk
from napari_imsmicrolink.data.image_transform import ImageTransform


def test_ImageTransform_add_points():

    test_pts = np.array([[50.75, 100.0], [20.0, 10.0], [10.0, 50.0], [60.0, 20.0]])

    itfm = ImageTransform()
    itfm.output_spacing = (1, 1)
    itfm.add_points(test_pts, round=True, src_or_tgt="source", scaling=100)

    assert itfm.source_pts is not None
    assert itfm.source_pts[0, 0] == 5100.0
    assert itfm.source_pts[3, 0] == 6000.0

    itfm.add_points(test_pts, round=False, src_or_tgt="source", scaling=100)

    assert itfm.source_pts[0, 0] == 50.75 * 100
    assert itfm.source_pts[3, 0] == 60 * 100

    itfm.add_points(test_pts, round=False, src_or_tgt="target", scaling=100)

    assert itfm.target_pts is not None


def test_ImageTransform_compute_transform():
    source_pts = np.array(
        [
            [356.93356879, 6713.16214535],
            [6285.96351516, 11137.0624842],
            [15154.40947051, 7596.13949593],
            [6905.28155271, 936.74985065],
        ]
    )

    target_pts = np.array(
        [[500.0, 6250.0], [6400.0, 10700.0], [15300.0, 7200.0], [7100.0, 500.0]]
    )

    itfm = ImageTransform()
    itfm.output_spacing = (0.92, 0.92)
    itfm.add_points(source_pts, round=False, src_or_tgt="source", scaling=1)
    itfm.add_points(target_pts, round=True, src_or_tgt="target", scaling=1)

    assert itfm.affine_transform is not None
    assert itfm.affine_np_mat_xy_um is not None
    assert itfm.affine_np_mat_yx_um is not None
    assert itfm.affine_np_mat_xy_px is not None
    assert itfm.affine_np_mat_yx_px is not None
    assert itfm.inverse_affine_transform is not None
    assert itfm.inverse_affine_np_mat_xy_um is not None
    assert itfm.inverse_affine_np_mat_yx_um is not None
    assert itfm.inverse_affine_np_mat_xy_px is not None
    assert itfm.inverse_affine_np_mat_yx_px is not None
    assert itfm.point_reg_error < 10


def test_ImageTransform_apply_transform_pts():
    aff_tform = sitk.AffineTransform(2)
    aff_tform.SetMatrix([2, 0, 0, 2])
    pts = np.array([[1, 1], [2, 2]]).astype(float)

    tformed_pts = ImageTransform.apply_transform_to_pts(pts, aff_tform)
    scaled_pts = np.array([[2, 2], [4, 4]]).astype(np.double)
    np.testing.assert_array_equal(tformed_pts, scaled_pts)
