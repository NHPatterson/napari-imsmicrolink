from typing import Union, Tuple, Optional
from pathlib import Path
import numpy as np
import dask.array as da
import zarr.storage
from tifffile import TiffWriter
import cv2
import SimpleITK as sitk
import zarr
from napari_imsmicrolink.data.microscopy_reader import MicroRegImage
from napari_imsmicrolink.data.czi_reader import CziRegImage
from napari_imsmicrolink.utils.image import (
    get_pyramid_info,
    yield_tiles,
    compute_sub_res,
)
from napari_imsmicrolink.utils.ome import generate_ome

PathLike = Union[str, Path]

NP_DTYPES_TO_OME = {
    np.dtype("uint8"): "uint8",
    np.dtype("int8"): "int8",
    np.dtype("uint16"): "uint16",
    np.dtype("int16"): "int16",
    np.dtype("uint32"): "uint32",
    np.dtype("int32"): "int32",
    np.dtype("float32"): "float",
    np.dtype("double"): "double",
    np.dtype("float64"): "double",
}


class OmeTiffWriter:
    def __init__(
        self,
        microscopy_image: Union[MicroRegImage, CziRegImage],
        image_name: str,
        image_transform: sitk.AffineTransform,
        output_size: Tuple[int, int],
        output_spacing: Tuple[float, float],
        tile_size: int = 512,
        output_dir: PathLike = "",
        compression: Optional[str] = None,
    ):

        self.microscopy_image: Union[MicroRegImage, CziRegImage] = microscopy_image
        self.image_series_idx: int = self.microscopy_image.base_layer_idx
        self.dask_im: da.Array = self.microscopy_image.pyr_levels_dask[1]
        if len(self.dask_im.shape) < 3:
            self.dask_im = self.dask_im.reshape(
                (1, self.dask_im.shape[0], self.dask_im.shape[1])
            )
        self.n_ch: int = self.dask_im.shape[0]
        self.image_name: str = image_name
        self.image_transform: sitk.AffineTransform = image_transform
        self.output_size: Tuple[int, int] = output_size
        self.output_spacing: Tuple[float, float] = output_spacing
        self.is_rgb: bool = self.microscopy_image.is_rgb
        self.tile_size: int = tile_size
        self.output_dir: PathLike = output_dir
        self.compression = compression

    def _transform_plane(
        self,
        image: sitk.Image,
        transform: sitk.AffineTransform,
        output_size: Tuple[int, int],
        output_spacing: Tuple[float, float],
    ) -> sitk.Image:
        resampler = sitk.ResampleImageFilter()
        resampler.SetTransform(transform)
        resampler.SetOutputSpacing(output_spacing)
        resampler.SetSize(output_size)
        image = resampler.Execute(image)
        return image

    def write_image(self) -> None:

        pyr_levels, pyr_shapes = get_pyramid_info(
            self.output_size[1],
            self.output_size[0],
            self.dask_im.shape[0],
            self.tile_size,
        )
        n_pyr_levels = len(pyr_levels)

        output_file_name = Path(self.output_dir) / f"{self.image_name}.ome.tiff"
        try:
            self.microscopy_image.ome_metadata.images = [
                self.microscopy_image.ome_metadata.images[self.image_series_idx]
            ]
            self.microscopy_image.ome_metadata.images[
                0
            ].pixels.size_x = self.output_size[
                0
            ]  # type:ignore
            self.microscopy_image.ome_metadata.images[
                0
            ].pixels.size_y = self.output_size[
                1
            ]  # type:ignore
            omexml = self.microscopy_image.ome_metadata.to_xml().encode("utf-8")
            self.microscopy_image.ome_metadata.images[0].name = output_file_name.name

        except AttributeError:

            def get_n_ch_mc(shape):
                if len(shape) == 2:
                    return 1
                else:
                    return shape[0]

            size_c = (
                self.dask_im.shape[-1]
                if self.microscopy_image.is_rgb
                else get_n_ch_mc(self.dask_im.shape)
            )
            pixel_metadata = {
                "size_t": 1,
                "size_z": 1,
                "size_x": self.output_size[0],
                "size_y": self.output_size[1],
                "size_c": size_c,
                "dimension_order": "XYCZT",
                "type": NP_DTYPES_TO_OME[self.microscopy_image.im_dtype],
                "physical_size_x": self.output_spacing[0],
                "physical_size_y": self.output_spacing[1],
                "interleaved": self.is_rgb,
            }
            spp = self.dask_im.shape[-1] if self.microscopy_image.is_rgb else 1
            if self.microscopy_image.ccolors:
                ccolors = self.microscopy_image.ccolors
            else:
                ccolors = [-1 for _ in range(self.microscopy_image.n_ch)]

            channel_meta_list = [
                {"name": cn, "samples_per_pixel": spp, "color": cc}
                for cn, cc in zip(self.microscopy_image.cnames, ccolors)
            ]
            ome = generate_ome(self.image_name, pixel_metadata, channel_meta_list)
            omexml = ome.to_xml().encode("utf-8")

        subifds = n_pyr_levels - 1

        if self.compression:
            compression = self.compression
        else:
            compression = "jpeg" if self.microscopy_image.is_rgb else "deflate"

        with TiffWriter(output_file_name, bigtiff=True) as tif:
            rgb_stores = []
            for channel_idx in range(self.microscopy_image.n_ch):
                if self.microscopy_image.is_rgb:
                    image = self.dask_im[:, :, channel_idx].compute()
                else:
                    image = self.dask_im[channel_idx, :, :].compute()

                image = np.squeeze(image)
                image = sitk.GetImageFromArray(image)
                image.SetSpacing(self.output_spacing)

                image = self._transform_plane(
                    image, self.image_transform, self.output_size, self.output_spacing
                )

                if isinstance(image, sitk.Image):
                    image = sitk.GetArrayFromImage(image)

                options = dict(
                    tile=(self.tile_size, self.tile_size),
                    compression=compression,
                    photometric="rgb" if self.is_rgb else "minisblack",
                    metadata=None,
                )

                if not self.is_rgb:
                    # write OME-XML to the ImageDescription tag of the first page
                    description = omexml if channel_idx == 0 else None
                    # write channel data
                    print(f" writing channel {channel_idx} - shape: {image.shape}")
                    tif.write(
                        image,
                        subifds=subifds,
                        description=description,
                        **options,
                    )

                    for pyr_idx in range(1, n_pyr_levels):
                        resize_shape = (
                            pyr_levels[pyr_idx][0],
                            pyr_levels[pyr_idx][1],
                        )
                        image = cv2.resize(
                            image,
                            resize_shape,
                            cv2.INTER_LINEAR,
                        )
                        tif.write(image, **options, subfiletype=1)

                elif channel_idx < self.microscopy_image.n_ch:
                    rgb_temp_store = zarr.storage.TempStore()
                    root = zarr.open_group(rgb_temp_store, mode="a")
                    chunking = (512, 512)
                    out = root.create_dataset(
                        0,
                        shape=tuple(np.asarray(self.output_size)[::-1]),
                        chunks=chunking,
                        dtype=self.microscopy_image.im_dtype,
                        overwrite=True,
                    )
                    out[:] = image
                    rgb_stores.append(rgb_temp_store)

                    if channel_idx == self.microscopy_image.n_ch - 1:
                        rgb_stack = []
                        for z in rgb_stores:
                            rgb_stack.append(da.from_zarr(zarr.open(z)[0]))

                        rgb_interleaved = da.stack(rgb_stack, axis=2)

                        yx_shape = rgb_interleaved.shape[:2]
                        ds = 1
                        while np.min(yx_shape) // 2**ds >= 512:
                            ds += 1

                        # write rgb to ome.tiff file
                        sub_resolutions = []
                        for ds_factor in range(1, ds):
                            sub_res_image = compute_sub_res(
                                rgb_interleaved,
                                ds_factor,
                                512,
                                self.microscopy_image.is_rgb,
                                self.microscopy_image.im_dtype,
                            )
                            sub_resolutions.append(sub_res_image)

                        subifds = len(sub_resolutions)

                        # write OME-XML to the ImageDescription tag of the first page
                        description = omexml
                        tiles = yield_tiles(rgb_interleaved, 512, True)

                        # write channel data
                        tif.write(
                            tiles,
                            subifds=subifds,
                            description=description,
                            shape=rgb_interleaved.shape,
                            dtype=self.microscopy_image.im_dtype,
                            **options,
                        )

                        for sr in sub_resolutions:
                            sr_tiles = yield_tiles(sr, 512, True)
                            tif.write(
                                sr_tiles,
                                shape=sr.shape,
                                dtype=self.microscopy_image.im_dtype,
                                **options,
                                subfiletype=1,
                            )

                        for store in rgb_stores:
                            try:
                                store.clear()
                            except FileNotFoundError:
                                continue
