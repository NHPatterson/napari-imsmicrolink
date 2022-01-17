from typing import Tuple
import numpy as np
from tifffile import xml2dict
import dask.array as da
import zarr
from napari_imsmicrolink.utils.czi import CziRegImageReader
from napari_imsmicrolink.utils.image import guess_rgb


class CziRegImage:
    def __init__(
        self,
        image,
    ):
        self.image_filepath = image
        self.czi = CziRegImageReader(self.image_filepath)

        (
            self.ch_dim_idx,
            self.y_dim_idx,
            self.x_dim_idx,
            self.im_dims,
            self.im_dtype,
        ) = self._get_image_info()

        self.im_dims = tuple(self.im_dims)
        self.is_rgb = guess_rgb(self.im_dims)
        self.n_ch = self.im_dims[2] if self.is_rgb else self.im_dims[0]

        czi_meta = xml2dict(self.czi.metadata())
        pixel_scaling_str = czi_meta["ImageDocument"]["Metadata"]["Scaling"]["Items"][
            "Distance"
        ][0]["Value"]
        pixel_scaling = float(pixel_scaling_str) * 1000000
        self.base_layer_pixel_res = pixel_scaling
        channels_meta = czi_meta["ImageDocument"]["Metadata"]["DisplaySetting"][
            "Channels"
        ]["Channel"]

        cnames = []
        for ch in channels_meta:
            cnames.append(ch.get("ShortName"))

        self.cnames = cnames

        ccolors = []
        for ch in channels_meta:
            ccolors.append(ch.get("Color"))

        self.ccolors = ccolors

        d_image = self._prepare_dask_image()
        self.pyr_levels_dask = {1: d_image}
        self.base_layer_idx = 0

    def _prepare_dask_image(self):
        ch_dim = self.im_dims[1:] if not self.is_rgb else self.im_dims[:2]
        chunks = ((1,) * self.n_ch, (ch_dim[0],), (ch_dim[1],))
        d_image = da.map_blocks(
            self.read_single_channel,
            chunks=chunks,
            dtype=self.im_dtype,
            meta=np.array((), dtype=self.im_dtype),
        )
        return d_image

    def get_dask_pyr(self):
        return self.czi.zarr_pyramidalize_czi(zarr.storage.TempStore())

    def _get_image_info(self):
        # if RGB need to get 0
        if self.czi.shape[-1] > 1:
            ch_dim_idx = self.czi.axes.index("0")
        else:
            ch_dim_idx = self.czi.axes.index("C")
        y_dim_idx = self.czi.axes.index("Y")
        x_dim_idx = self.czi.axes.index("X")
        if self.czi.shape[-1] > 1:
            im_dims = np.array(self.czi.shape)[[y_dim_idx, x_dim_idx, ch_dim_idx]]
        else:
            im_dims = np.array(self.czi.shape)[[ch_dim_idx, y_dim_idx, x_dim_idx]]

        im_dtype = self.czi.dtype

        return ch_dim_idx, y_dim_idx, x_dim_idx, im_dims, im_dtype

    def read_single_channel(self, block_id: Tuple[int, ...]):
        channel_idx = block_id[0]
        if self.is_rgb is False:
            image = self.czi.sub_asarray(
                channel_idx=[channel_idx],
            )
        else:
            image = self.czi.sub_asarray_rgb(channel_idx=[channel_idx], greyscale=False)

        return np.expand_dims(np.squeeze(image), axis=0)
