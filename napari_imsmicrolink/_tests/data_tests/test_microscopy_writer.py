import pytest
from pathlib import Path
import tempfile
import numpy as np
from tifffile import imwrite, imread, TiffFile
from ome_types import from_xml
import SimpleITK as sitk
from napari_imsmicrolink.data.tifffile_reader import TiffFileRegImage
from napari_imsmicrolink.data.microscopy_writer import OmeTiffWriter

@pytest.fixture(scope="session")
def data_out_dir(tmpdir_factory):
    out_dir = tmpdir_factory.mktemp("output")
    return out_dir


def test_OmeTiffWriter_rgb(data_out_dir):
    out_fp = Path(data_out_dir) / "test_rgb.tiff"
    rgb_im = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
    imwrite(out_fp, rgb_im, compression="deflate")

    mr = TiffFileRegImage(out_fp)
    writer = OmeTiffWriter(
        mr,
        "test_im",
        sitk.AffineTransform(2),
        (2048, 2048),
        (1, 1),
        512,
        Path(data_out_dir),
        compression="deflate",
    )

    writer.write_image()
    im_out_path = Path(data_out_dir) / "test_im.ome.tiff"
    with TiffFile(im_out_path) as tf:
        ome_metadata = from_xml(tf.ome_metadata)
    image_out = imread(im_out_path)

    assert np.array_equal(image_out, rgb_im)
    assert ome_metadata.images[0].pixels.channels[0].samples_per_pixel == 3
    assert ome_metadata.images[0].pixels.interleaved is True
    assert ome_metadata.images[0].pixels.size_c == 3
    assert ome_metadata.images[0].pixels.size_y == 2048
    assert ome_metadata.images[0].pixels.size_x == 2048

def test_OmeTiffWriter_mc(data_out_dir):
    out_fp = Path(data_out_dir) / "test_rgb.tiff"

    mc_im = np.random.randint(0, 255, (3, 2048, 2048), dtype=np.uint8)
    imwrite(out_fp, mc_im, compression="deflate")

    mr = TiffFileRegImage(out_fp)
    writer = OmeTiffWriter(
        mr,
        "test_im",
        sitk.AffineTransform(2),
        (2048, 2048),
        (1, 1),
        512,
        Path(data_out_dir),
    )

    writer.write_image()
    im_out_path = Path(data_out_dir) / "test_im.ome.tiff"
    with TiffFile(im_out_path) as tf:
        ome_metadata = from_xml(tf.ome_metadata)
    image_out = imread(im_out_path)

    assert np.array_equal(image_out, mc_im)
    assert ome_metadata.images[0].pixels.channels[0].samples_per_pixel == 1
    assert ome_metadata.images[0].pixels.interleaved is False
    assert ome_metadata.images[0].pixels.size_c == 3
    assert ome_metadata.images[0].pixels.size_y == 2048
    assert ome_metadata.images[0].pixels.size_x == 2048
