import numpy as np


def apply_rotmat_points(rotmat, point_mat, center_point, recenter_point):
    return np.dot(rotmat, (point_mat - center_point).T).T + recenter_point[::-1]
