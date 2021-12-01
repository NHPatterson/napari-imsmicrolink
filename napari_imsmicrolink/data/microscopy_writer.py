from typing import Union, Tuple
from pathlib import Path
import numpy as np
import dask.array as da
from tifffile import TiffWriter
import cv2
import SimpleITK as sitk
from napari_imsmicrolink.data.microscopy_reader import MicroRegImage
from napari_imsmicrolink.utils.image import get_pyramid_info

PathLike = Union[str, Path]


class OmeTiffWriter:
    def __init__(
        self,
        microscopy_image: MicroRegImage,
        image_name: str,
        image_transform: sitk.AffineTransform,
        output_size: Tuple[int, int],
        output_spacing: Tuple[float, float],
        tile_size: int = 512,
        output_dir: PathLike = "",
    ):

        self.microscopy_image: MicroRegImage = microscopy_image
        self.image_series_idx: int = self.microscopy_image.base_layer_idx
        self.dask_im: da.Array = self.microscopy_image.pyr_levels_dask[1]
        self.n_ch: int = self.dask_im.shape[0]
        self.image_name: str = image_name
        self.image_transform: sitk.AffineTransform = image_transform
        self.output_size: Tuple[int, int] = output_size
        self.output_spacing: Tuple[float, float] = output_spacing
        self.is_rgb: bool = False
        self.tile_size: int = tile_size
        self.output_dir: PathLike = output_dir

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

        self.microscopy_image.ome_metadata.images = [
            self.microscopy_image.ome_metadata.images[self.image_series_idx]
        ]

        self.microscopy_image.ome_metadata.images[0].pixels.size_x = self.output_size[
            0
        ]  # type:ignore
        self.microscopy_image.ome_metadata.images[0].pixels.size_y = self.output_size[
            1
        ]  # type:ignore

        omexml = self.microscopy_image.ome_metadata.to_xml().encode("utf-8")

        subifds = n_pyr_levels - 1

        # compression = "jpeg" if self.reg_image.is_rgb else "deflate"
        compression = "deflate"

        output_file_name = Path(self.output_dir) / f"{self.image_name}.ome.tiff"

        self.microscopy_image.ome_metadata.images[0].name = output_file_name.name

        with TiffWriter(output_file_name, bigtiff=True) as tif:
            for channel_idx in range(self.n_ch):
                image = self.dask_im[channel_idx, :, :].compute()
                image = np.squeeze(image)
                image = sitk.GetImageFromArray(image)
                image.SetSpacing(self.output_spacing)

                image = self._transform_plane(
                    image, self.image_transform, self.output_size, self.output_spacing
                )

                # if self.is_rgb:
                #     rgb_im_data.append(image)
                # else:
                if isinstance(image, sitk.Image):
                    image = sitk.GetArrayFromImage(image)

                options = dict(
                    tile=(self.tile_size, self.tile_size),
                    compression=compression,
                    photometric="rgb" if self.is_rgb else "minisblack",
                    metadata=None,
                )
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
            #
            # if self.reg_image.is_rgb:
            #     rgb_im_data = sitk.Compose(rgb_im_data)
            #     rgb_im_data = sitk.GetArrayFromImage(rgb_im_data)
            #
            #     options = dict(
            #         tile=(self.tile_size, self.tile_size),
            #         compression=self.compression,
            #         photometric="rgb",
            #         metadata=None,
            #     )
            #     # write OME-XML to the ImageDescription tag of the first page
            #     description = self.omexml
            #
            #     # write channel data
            #     tif.write(
            #         rgb_im_data,
            #         subifds=self.subifds,
            #         description=description,
            #         **options,
            #     )
            #
            #     print(f"RGB shape: {rgb_im_data.shape}")
            #     if write_pyramid:
            #         for pyr_idx in range(1, self.n_pyr_levels):
            #             resize_shape = (
            #                 self.pyr_levels[pyr_idx][0],
            #                 self.pyr_levels[pyr_idx][1],
            #             )
            #             rgb_im_data = cv2.resize(
            #                 rgb_im_data,
            #                 resize_shape,
            #                 cv2.INTER_LINEAR,
            #             )
            #             tif.write(rgb_im_data, **options, subfiletype=1)
