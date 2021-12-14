import pytest
import os
from pathlib import Path
import numpy as np
from napari_imsmicrolink.data.ims_pixel_map import PixelMapIMS


@pytest.mark.parametrize(
    "data_fp, ", ["bruker_spotlist.txt", "imzml_file.imzML", "peaks.sqlite"]
)
def test_PixelMapIMS_read(data_fp):

    HERE = os.path.dirname(__file__)
    data_fp = Path(HERE) / "_test_data" / data_fp

    pmap = PixelMapIMS(data_fp)

    assert len(pmap.data) == 1
    assert pmap.regions is not None
    assert pmap.x_coords_orig is not None
    assert pmap.y_coords_orig is not None
    assert pmap.x_coords_min is not None
    assert pmap.y_coords_min is not None
    assert pmap.x_coords_pad is not None
    assert pmap.y_coords_pad is not None


def test_PixelMapIMS_rotate():
    HERE = os.path.dirname(__file__)
    data_fp = Path(HERE) / "_test_data" / "bruker_spotlist.txt"

    pmap = PixelMapIMS(data_fp)
    pre_rot_x = np.copy(pmap.x_coords_min)
    pmap.rotate_coordinates(90)
    assert np.any(np.not_equal(pre_rot_x, pmap.x_coords_min))

    pmap.rotate_coordinates(-90)
    np.testing.assert_array_equal(pre_rot_x, pmap.x_coords_min)

    pmap.rotate_coordinates(-90)
    assert np.any(np.not_equal(pre_rot_x, pmap.x_coords_min))

    pmap.rotate_coordinates(-90)
    assert np.any(np.not_equal(pre_rot_x, pmap.x_coords_min))


def test_PixelMapIMS_delete_roi():
    HERE = os.path.dirname(__file__)
    data_fp = Path(HERE) / "_test_data" / "bruker_spotlist.txt"

    pmap = PixelMapIMS(data_fp)
    pre_delete_no_regions = len(np.unique(pmap.regions))
    pmap.delete_roi(2)

    assert len(np.unique(pmap.regions)) < pre_delete_no_regions


def test_PixelMapIMS_prepare_meta():
    HERE = os.path.dirname(__file__)
    data_fp = Path(HERE) / "_test_data" / "bruker_spotlist.txt"

    pmap = PixelMapIMS(data_fp)
    meta_dict = pmap.prepare_pmap_metadata()
    pmap_df = pmap.prepare_pmap_dataframe()
    df_cols = [
        "regions",
        "x_original",
        "y_original",
        "x_minimized",
        "y_minimized",
        "x_padded",
        "y_padded",
    ]
    assert "Pixel Map Datasets Files" in meta_dict.keys()
    assert "padding" in meta_dict.keys()
    assert np.all(pmap_df.columns == df_cols)


def test_PixelMapIMS_change_padding():
    HERE = os.path.dirname(__file__)
    data_fp = Path(HERE) / "_test_data" / "bruker_spotlist.txt"

    pmap = PixelMapIMS(data_fp)
    pmap.ims_res = 20
    pad_image_shape = pmap.pixelmap_padded.shape
    pmap.padding = {"x_left": 10, "x_right": 20, "y_top": 30, "y_bottom": 40}

    assert pmap.padding_microns["x_left"] == pmap.padding["x_left"] * pmap.ims_res
    assert pmap.padding_microns["y_bottom"] == pmap.padding["y_bottom"] * pmap.ims_res
    assert (
        pmap.pixelmap_padded.shape[0]
        == pad_image_shape[0] + pmap.padding["y_top"] + pmap.padding["y_bottom"]
    )
    assert (
        pmap.pixelmap_padded.shape[1]
        == pad_image_shape[1] + pmap.padding["x_left"] + pmap.padding["x_right"]
    )
