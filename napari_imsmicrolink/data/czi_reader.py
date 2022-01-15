from typing import Tuple
import multiprocessing
import warnings
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tifffile import create_output, xml2dict
from czifile import CziFile
import dask.array as da
import zarr
import cv2


class CziRegImageReader(CziFile):
    """
    Sub-class of CziFile with added functionality to only read certain channels
    """

    def sub_asarray(
        self,
        resize=True,
        order=0,
        out=None,
        max_workers=None,
        channel_idx=None,
        as_uint8=False,
        zarr_fp=None,
        ds_factor=2,
    ):

        """Return image data from file(s) as numpy array.

        Parameters
        ----------
        resize : bool
            If True (default), resize sub/supersampled subblock data.
        order : int
            The order of spline interpolation used to resize sub/supersampled
            subblock data. Default is 0 (nearest neighbor).
        out : numpy.ndarray, str, or file-like object; optional
            Buffer where image data will be saved.
            If numpy.ndarray, a writable array of compatible dtype and shape.
            If str or open file, the file name or file object used to
            create a memory-map to an array stored in a binary file on disk.
        max_workers : int
            Maximum number of threads to read and decode subblock data.
            By default up to half the CPU cores are used.
        channel_idx : int or list of int
            The indices of the channels to extract
        as_uint8 : bool
            byte-scale image data to np.uint8 data type

        Parameters
        ----------
        out:np.ndarray
            image read with selected parameters as np.ndarray
        """

        out_shape = list(self.shape)
        start = list(self.start)

        ch_dim_idx = self.axes.index("C")

        if channel_idx is not None:
            if isinstance(channel_idx, int):
                channel_idx = [channel_idx]

            if out_shape[ch_dim_idx] == 1:
                channel_idx = None

            else:
                out_shape[ch_dim_idx] = len(channel_idx)
                min_ch_seq = {}
                for idx, i in enumerate(channel_idx):
                    min_ch_seq.update({i: idx})

        if as_uint8 is True:
            out_dtype = np.uint8
        else:
            out_dtype = self.dtype

        if zarr_fp is not None:
            if ds_factor > 1:
                out_shape[3] = out_shape[3] // ds_factor
                out_shape[4] = out_shape[4] // ds_factor
                out_shape = tuple(out_shape)
                start[3] = start[3] // ds_factor
                start[4] = start[4] // ds_factor
            rgb_chunk = self.shape[-1] if self.shape[-1] > 2 else 1
            root = zarr.open_group(zarr_fp, mode="a")
            pyramid_seq = str(int(np.log2(ds_factor)))
            chunking = (1, 1, 1, 512, 512, rgb_chunk)
            out = root.create_dataset(
                pyramid_seq,
                shape=tuple(out_shape),
                chunks=chunking,
                dtype=out_dtype,
                overwrite=True,
            )

        elif out is None:
            out = create_output(None, tuple(out_shape), out_dtype)

        if max_workers is None:
            max_workers = multiprocessing.cpu_count() - 1

        def func(directory_entry, resize=resize, order=order, start=start, out=out):
            """Read, decode, and copy subblock data."""
            subblock = directory_entry.data_segment()
            dvstart = list(directory_entry.start)
            czi_c_idx = [de.dimension for de in subblock.dimension_entries].index("C")
            subblock_ch_idx = subblock.dimension_entries[czi_c_idx].start
            if channel_idx is not None:
                if subblock_ch_idx in channel_idx:
                    subblock.dimension_entries[czi_c_idx].start
                    tile = subblock.data(resize=resize, order=order)
                    dvstart[ch_dim_idx] = min_ch_seq.get(subblock_ch_idx)
                else:
                    return
            else:
                tile = subblock.data(resize=resize, order=order)

            if ds_factor > 1:
                tile = np.squeeze(tile)

                w = tile.shape[0] // ds_factor
                h = tile.shape[0] // ds_factor
                tile_ds = cv2.resize(
                    np.squeeze(tile), dsize=(w, h), interpolation=cv2.INTER_LINEAR
                )

                tile_ds = np.reshape(
                    tile_ds, (1, 1, 1, tile_ds.shape[0], tile_ds.shape[1], rgb_chunk)
                )
                tile = tile_ds
                dvstart[3] = dvstart[3] // ds_factor
                dvstart[4] = dvstart[4] // ds_factor

            if as_uint8 is True:
                tile = (tile / 256).astype("uint8")

            index = tuple(
                slice(i - j, i - j + k)
                for i, j, k in zip(tuple(dvstart), tuple(start), tile.shape)
            )

            try:
                out[index] = tile
            except ValueError as e:
                error = e
                corr_shape = (
                    str(error)
                    .split("shape ")[1]
                    .split(", got")[0]
                    .strip("(")
                    .strip(")")
                )
                corr_shape.split(",")
                cor_shape = tuple([slice(int(t)) for t in corr_shape.split(",")])
                tile = tile[cor_shape]
                index = tuple(
                    slice(i - j, i - j + k)
                    for i, j, k in zip(tuple(dvstart), tuple(start), tile.shape)
                )
                out[index] = tile

        if max_workers > 1:
            self._fh.lock = True
            with ThreadPoolExecutor(max_workers) as executor:
                executor.map(func, self.filtered_subblock_directory)
            self._fh.lock = None
        else:

            for idx, directory_entry in enumerate(self.filtered_subblock_directory):
                func(directory_entry)

        if hasattr(out, "flush"):
            out.flush()
        return out

    def sub_asarray_rgb(
        self,
        resize=True,
        order=0,
        out=None,
        max_workers=None,
        channel_idx=None,
        as_uint8=False,
        greyscale=False,
    ):

        """Return image data from file(s) as numpy array.

        Parameters
        ----------
        resize : bool
            If True (default), resize sub/supersampled subblock data.
        order : int
            The order of spline interpolation used to resize sub/supersampled
            subblock data. Default is 0 (nearest neighbor).
        out : numpy.ndarray, str, or file-like object; optional
            Buffer where image data will be saved.
            If numpy.ndarray, a writable array of compatible dtype and shape.
            If str or open file, the file name or file object used to
            create a memory-map to an array stored in a binary file on disk.
        max_workers : int
            Maximum number of threads to read and decode subblock data.
            By default up to half the CPU cores are used.
        channel_idx : int or list of int
            The indices of the channels to extract
        as_uint8 : bool
            byte-scale image data to np.uint8 data type

        Parameters
        ----------
        out:np.ndarray
            image read with selected parameters as np.ndarray
        """

        out_shape = list(self.shape)
        start = list(self.start)
        ch_dim_idx = self.axes.index("0")

        if channel_idx is not None:
            if isinstance(channel_idx, int):
                channel_idx = [channel_idx]
            out_shape[ch_dim_idx] = len(channel_idx)

        if greyscale is True:
            out_shape[ch_dim_idx] = 1

        if as_uint8 is True:
            out_dtype = np.uint8
        else:
            out_dtype = self.dtype

        if out is None:
            out = create_output(None, tuple(out_shape), out_dtype)

        if max_workers is None:
            max_workers = multiprocessing.cpu_count() - 1

        def func(directory_entry, resize=resize, order=order, start=start, out=out):
            """Read, decode, and copy subblock data."""
            subblock = directory_entry.data_segment()
            dvstart = list(directory_entry.start)
            tile = subblock.data(resize=resize, order=order)

            if greyscale is True:
                tile = czi_tile_grayscale(tile)

            if channel_idx is not None:
                tile = tile[:, :, :, :, :, channel_idx]

            index = tuple(
                slice(i - j, i - j + k)
                for i, j, k in zip(tuple(dvstart), tuple(start), tile.shape)
            )

            try:
                out[index] = tile
            except ValueError as e:
                warnings.warn(str(e))

        if max_workers > 1:
            self._fh.lock = True
            with ThreadPoolExecutor(max_workers) as executor:
                executor.map(func, self.filtered_subblock_directory)
            self._fh.lock = None
        else:
            for directory_entry in self.filtered_subblock_directory:
                func(directory_entry)

        if hasattr(out, "flush"):
            out.flush()
        return out

    def zarr_pyramidalize_czi(self, zarr_fp):
        dask_pyr = []
        root = zarr.open_group(zarr_fp, mode="a")

        root.attrs["axes_names"] = list(self.axes)
        root.attrs["orig_shape"] = list(self.shape)
        all_axes = list(self.axes)
        yx_dims = np.where(np.isin(all_axes, ["Y", "X"]) == 1)[0].tolist()
        yx_shape = np.array(self.shape[slice(yx_dims[0], yx_dims[1] + 1)])

        ds = 1
        while np.min(yx_shape) // 2 ** ds >= 512:
            ds += 1

        self.sub_asarray(
            zarr_fp=zarr_fp, resize=True, order=0, ds_factor=1, max_workers=1
        )
        zarray = da.squeeze(da.from_zarr(zarr.open(zarr_fp)[0]))
        dask_pyr.append(da.squeeze(zarray))
        for ds_factor in range(1, ds):
            zres = zarr.storage.TempStore()
            rgb_chunk = self.shape[-1] if self.shape[-1] > 2 else 1
            is_rgb = True if rgb_chunk > 1 else False

            sub_res_image = compute_sub_res(zarray, ds_factor, 512, is_rgb, self.dtype)

            da.to_zarr(sub_res_image, zres, component="0")

            dask_pyr.append(da.squeeze(da.from_zarr(zres, component="0")))

        return dask_pyr


def czi_tile_grayscale(rgb_image):
    """
    convert RGB image data to greyscale

    Parameters
    ----------
    rgb_image: np.ndarray
        image data
    Returns
    -------
    image:np.ndarray
        returns 8-bit greyscale image for 24-bit RGB image
    """
    result = (
        (rgb_image[..., 0] * 0.2125).astype(np.uint8)
        + (rgb_image[..., 1] * 0.7154).astype(np.uint8)
        + (rgb_image[..., 2] * 0.0721).astype(np.uint8)
    )

    return np.expand_dims(result, axis=-1)


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


def guess_rgb(shape):
    """
    Guess if the passed shape comes from rgb data.
    If last dim is 3 or 4 assume the data is rgb, including rgba.

    Parameters
    ----------
    shape : list of int
        Shape of the data that should be checked.

    Returns
    -------
    bool
        If data is rgb or not.
    """
    ndim = len(shape)
    last_dim = shape[-1]
    if ndim > 2 and last_dim < 5:
        rgb = True
    else:
        rgb = False

    return rgb


def compute_sub_res(zarray, ds_factor, tile_size, is_rgb, im_dtype):
    if is_rgb:
        resampling_axis = {1: 2 ** ds_factor, 1: 2 ** ds_factor, 2: 1}
        tiling = (tile_size, tile_size, 3)
    else:
        resampling_axis = {0: 1, 1: 2 ** ds_factor, 2: 2 ** ds_factor}
        tiling = (1, tile_size, tile_size)

    resampled_zarray_subres = da.coarsen(
        np.mean,
        zarray,
        resampling_axis,
        trim_excess=True,
    )
    resampled_zarray_subres = resampled_zarray_subres.astype(im_dtype)
    resampled_zarray_subres = resampled_zarray_subres.rechunk(tiling)

    return resampled_zarray_subres
