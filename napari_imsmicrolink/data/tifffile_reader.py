import warnings
from pathlib import Path
from tifffile import TiffFile
from napari_imsmicrolink.utils.image import (
    guess_rgb,
    get_tifffile_info,
    tf_zarr_read_single_ch,
    tifffile_to_dask,
)
from ome_types import from_xml
from napari_imsmicrolink.utils.tifffile_meta import (
    tifftag_xy_pixel_sizes,
    ometiff_xy_pixel_sizes,
    svs_xy_pixel_sizes,
    ometiff_ch_names,
)

TIFFFILE_EXTS = [".scn", ".ome.tiff", ".tif", ".tiff", ".svs", ".ndpi"]


class TiffFileRegImage:
    def __init__(self, image_fp):
        self.image_filepath = image_fp
        self.tf = TiffFile(self.image_filepath)
        self.reader = "tifffile"

        (self.im_dims, self.im_dtype, self.largest_series) = self._get_image_info()

        self.im_dims = tuple(self.im_dims)
        self.is_rgb = guess_rgb(self.im_dims)
        self.n_ch = self.im_dims[2] if self.is_rgb else self.im_dims[0]

        self.base_layer_pixel_res = self._get_im_res()
        self.cnames = self._get_ch_names()
        self.ccolors = None

        d_image = self.get_dask_pyr()
        if isinstance(d_image, list):
            self.pyr_levels_dask = {1: d_image[0]}
        else:
            self.pyr_levels_dask = {1: d_image}

        self.base_layer_idx = 0

    def get_dask_pyr(self):
        return tifffile_to_dask(self.image_filepath, self.largest_series)

    def _get_im_res(self):
        if Path(self.image_filepath).suffix.lower() in [".scn", ".ndpi"]:
            return tifftag_xy_pixel_sizes(
                self.tf,
                self.largest_series,
                0,
            )[0]
        elif Path(self.image_filepath).suffix.lower() in [".svs"]:
            return svs_xy_pixel_sizes(
                self.tf,
                self.largest_series,
                0,
            )[0]
        elif self.tf.ome_metadata:
            return ometiff_xy_pixel_sizes(
                from_xml(self.tf.ome_metadata),
                self.largest_series,
            )[0]
        else:
            try:
                return tifftag_xy_pixel_sizes(
                    self.tf,
                    self.largest_series,
                    0,
                )[0]
            except KeyError:
                warnings.warn(
                    "Unable to parse pixel resolution information from file"
                    " defaulting to 1"
                )
                return 1.0

    def _get_ch_names(self):
        if self.tf.ome_metadata:
            cnames = ometiff_ch_names(
                from_xml(self.tf.ome_metadata), self.largest_series
            )
        else:
            cnames = []
            if self.is_rgb:
                cnames.append("C01 - RGB")
            else:
                for idx, ch in enumerate(range(self.n_ch)):
                    cnames.append(f"C{str(idx + 1).zfill(2)}")

        return cnames

    def _get_image_info(self):
        if len(self.tf.series) > 1:
            warnings.warn(
                "The tiff contains multiple series, "
                "the largest series will be read by default"
            )

        im_dims, im_dtype, largest_series = get_tifffile_info(self.image_filepath)

        return im_dims, im_dtype, largest_series

    def read_single_channel(self, channel_idx: int):
        if channel_idx > (self.n_ch - 1):
            warnings.warn(
                "channel_idx exceeds number of channels, reading channel at channel_idx == 0"
            )
            channel_idx = 0
        image = tf_zarr_read_single_ch(self.image_filepath, channel_idx, self.is_rgb)
        return image
