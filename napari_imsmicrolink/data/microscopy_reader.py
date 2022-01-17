import os

os.environ[
    "BIOFORMATS_MEMORY"
] = "8g"  # need to source before bioformats is initialized
from typing import Union, List, Dict
from pathlib import Path
import numpy as np
import dask.array as da
from aicsimageio.readers.bioformats_reader import BioFile
from ome_types import OME
from napari_imsmicrolink.utils.image import guess_rgb

PathLike = Union[str, Path]


class MicroRegImage:
    def __init__(self, image_filepath: PathLike):
        self.image_filepath: PathLike = image_filepath
        with BioFile(self.image_filepath) as bf:
            self.n_scenes: int = len(bf.ome_metadata.images)
            self.ome_metadata: OME = bf.ome_metadata
        self.base_layer_pixel_res: float = 1
        self.base_layer_idx: int = 0
        self.cnames: List[str, ...] = []
        self.pyr_levels_dask: Dict[int, da.Array] = dict()
        self._find_pyramid()
        self._get_base_metadata()
        self.is_rgb = guess_rgb(self.pyr_levels_dask[1].shape)

    def _find_pyramid(self) -> None:
        yx_shapes = []
        for scene in range(self.n_scenes):
            with BioFile(self.image_filepath, scene) as bf:
                bf.set_series(scene)
                im_shape = bf.to_dask().shape
                if im_shape[-1] > 4:
                    yx_shapes.append((im_shape[-2], im_shape[-1]))
                else:
                    yx_shapes.append((im_shape[-3], im_shape[-2]))

                image_xy_dim = [i[-2] * i[-1] for i in yx_shapes]
                max_xy_idx = np.argmax(image_xy_dim)
                self.base_layer_idx = int(max_xy_idx)
                self.base_layer_pixel_res = bf.ome_metadata.images[
                    max_xy_idx
                ].pixels.physical_size_x

                if self.base_layer_pixel_res is None:
                    self.base_layer_pixel_res = 1

                diff_from_max = [
                    np.asarray(yx_shapes[max_xy_idx]) / np.asarray(i) for i in yx_shapes
                ]
                dim_size_comparison = [np.abs(i[1] - i[0]) for i in diff_from_max]
                pyramid_consistent = np.where(np.asarray(dim_size_comparison) < 0.5)[0]

            for pyr_idx in pyramid_consistent:
                with BioFile(
                    self.image_filepath,
                    pyr_idx,
                    dask_tiles=False,
                    tile_size=(4096, 4096),
                ) as bf:
                    dask_im = da.squeeze(bf.to_dask())
                    if len(dask_im.shape) != 3:
                        dask_im = dask_im.reshape(
                            (1, dask_im.shape[0], dask_im.shape[1])
                        )
                    ds = int(diff_from_max[pyr_idx][0])
                    self.pyr_levels_dask.update({ds: dask_im})

    def _get_base_metadata(self) -> None:
        self.cnames = [
            ch.name if ch.name else f"Unnamed Microscopy {idx}"
            for idx, ch in enumerate(
                self.ome_metadata.images[self.base_layer_idx].pixels.channels
            )
        ]
