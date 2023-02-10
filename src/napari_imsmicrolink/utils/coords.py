from typing import NamedTuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
import h5py


class CoordData(NamedTuple):
    regions: np.ndarray
    xy_original: np.ndarray
    xy_minimized: np.ndarray
    xy_padded: np.ndarray


class CoordDataMicro(NamedTuple):
    regions: np.ndarray
    xy_original: np.ndarray
    xy_minimized: np.ndarray
    xy_padded: np.ndarray
    xy_micro_ims_px: np.ndarray
    xy_micro_physical: np.ndarray
    xy_micro_px: np.ndarray


def pmap_coords_to_h5(pmap_data: pd.DataFrame, output_file: Union[str, Path]):

    includes_micro = "x_micro_ims_px" in pmap_data.columns

    if includes_micro:
        cdata = CoordDataMicro(
            regions=np.asarray(pmap_data["regions"]).astype("S"),
            xy_original=np.asarray(pmap_data[["x_original", "y_original"]]).astype(
                np.int64
            ),
            xy_minimized=np.asarray(pmap_data[["x_minimized", "y_minimized"]]).astype(
                np.int64
            ),
            xy_padded=np.asarray(pmap_data[["x_padded", "y_padded"]]).astype(np.int64),
            xy_micro_ims_px=np.asarray(
                pmap_data[["x_micro_ims_px", "y_micro_ims_px"]]
            ).astype(np.float64),
            xy_micro_physical=np.asarray(
                pmap_data[["x_micro_physical", "y_micro_physical"]]
            ).astype(np.float64),
            xy_micro_px=np.asarray(pmap_data[["x_micro_px", "y_micro_px"]]).astype(
                np.float64
            ),
        )
    else:
        cdata = CoordData(
            regions=np.asarray(pmap_data["regions"]).astype("S"),
            xy_original=np.asarray(pmap_data[["x_original", "y_original"]]).astype(
                np.int64
            ),
            xy_minimized=np.asarray(pmap_data[["x_minimized", "y_minimized"]]).astype(
                np.int64
            ),
            xy_padded=np.asarray(pmap_data[["x_padded", "y_padded"]]).astype(np.int64),
        )

    with h5py.File(output_file, "w") as f:
        for k, v in zip(cdata._fields, cdata):
            f.create_dataset(k, data=v, compression="gzip")
