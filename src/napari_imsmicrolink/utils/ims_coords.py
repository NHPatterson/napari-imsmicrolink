from typing import Tuple
import sqlite3
import numpy as np


def parse_tsf_coordinates(
    tsf_data_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    conn = sqlite3.connect(tsf_data_path)

    cursor = conn.cursor()
    cursor.execute(
        "SELECT Frame, RegionNumber, XIndexPos, YIndexPos FROM MaldiFrameInfo"
    )

    frame_index_position = np.array(cursor.fetchall())

    regions = frame_index_position[:, 1]
    x = frame_index_position[:, 2]
    y = frame_index_position[:, 3]

    return regions, x, y
