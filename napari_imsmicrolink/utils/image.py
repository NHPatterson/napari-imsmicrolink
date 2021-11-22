from typing import Tuple, List, Union
import numpy as np


def calc_pyramid_levels(
    xy_final_shape: np.ndarray, tile_size: int
) -> List[Tuple[int, int]]:
    """
    Calculate number of pyramids for a given image dimension and tile size
    Stops when further downsampling would be smaller than tile_size.

    Parameters
    ----------
    xy_final_shape:np.ndarray
        final shape in xy order
    tile_size: int
        size of the tiles in the pyramidal layers
    Returns
    -------
    res_shapes:list
        list of tuples of the shapes of the downsampled images

    """
    res_shape = xy_final_shape[::-1]
    res_shapes = [(int(res_shape[0]), int(res_shape[1]))]

    while all(res_shape > tile_size):
        res_shape = res_shape // 2
        res_shapes.append((int(res_shape[0]), int(res_shape[1])))

    return res_shapes[:-1]


def get_pyramid_info(
    y_size: int, x_size: int, n_ch: int, tile_size: int
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int, int, int, int]]]:
    """
    Get pyramidal info for OME-tiff output

    Parameters
    ----------
    y_size: int
        y dimension of base layer
    x_size:int
        x dimension of base layer
    n_ch:int
        number of channels in the image
    tile_size:int
        tile size of the image

    Returns
    -------
    pyr_levels
        pyramidal levels
    pyr_shapes:
        OME-zarr pyramid shapes for all levels

    """
    yx_size = np.asarray([y_size, x_size], dtype=np.int32)
    pyr_levels = calc_pyramid_levels(yx_size, tile_size)
    pyr_shapes = [(1, n_ch, 1, int(pl[0]), int(pl[1])) for pl in pyr_levels]
    return pyr_levels, pyr_shapes


def centered_transform(
    image_size,
    image_spacing,
    rotation_angle,
) -> np.ndarray:
    angle = np.deg2rad(rotation_angle)

    sina = np.sin(angle)
    cosa = np.cos(angle)

    # build rot mat
    rot_mat = np.eye(3)
    rot_mat[0, 0] = cosa
    rot_mat[1, 1] = cosa
    rot_mat[1, 0] = sina
    rot_mat[0, 1] = -sina

    # recenter transform
    center_point = np.multiply(image_size, image_spacing) / 2
    center_point = np.append(center_point, 0)
    translation = center_point - np.dot(rot_mat, center_point)
    rot_mat[:2, 2] = translation[:2]

    return rot_mat
