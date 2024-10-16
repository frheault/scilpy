# -*- coding: utf-8 -*-
import numpy as np


def convert_dps_to_dpp(sft, keys, overwrite=False):
    """
    Copy the value of the data_per_streamline to each point of the
    streamline, as data_per_point. The dps key is removed and added as dpp key.

    Parameters
    ----------
    sft: StatefulTractogram
    keys: List[str], optional
        The list of dps keys to convert to dpp.
    overwrite: bool
        If true, allow continuing even if the key already existed as dpp.
    """
    for key in keys:
        if key not in sft.data_per_streamline:
            raise ValueError("Dps key {} not found!".format(key))
        if key in sft.data_per_point and not overwrite:
            raise ValueError("Dpp key {} already existed. Please allow "
                             "overwriting.".format(key))
        sft.data_per_point[key] = [[val]*len(s) for val, s in
                                   zip(sft.data_per_streamline[key],
                                       sft.streamlines)]
        del sft.data_per_streamline[key]

    return sft


def project_map_to_streamlines(sft, map_volume, endpoints_only=False):
    """
    Projects a map onto the points of streamlines. The result is a
    data_per_point.

    Parameters
    ----------
    sft: StatefulTractogram
        Input tractogram.
    map_volume: DataVolume
        Input map.
    endpoints_only: bool, optional
        If True, will only project the map_volume onto the endpoints of the
        streamlines (all values along streamlines set to zero). If False,
        will project the map_volume onto all points of the streamlines.

    Returns
    -------
    streamline_data: List
        The values that could now be associated to a data_per_point key.
        The map_volume projected to each point of the streamlines.
    """
    if len(map_volume.data.shape) == 4:
        dimension = map_volume.data.shape[3]
    else:
        dimension = 1

    streamline_data = []
    if endpoints_only:
        for s in sft.streamlines:
            p1_data = map_volume.get_value_at_coordinate(
                s[0][0], s[0][1], s[0][2],
                space=sft.space, origin=sft.origin)
            p2_data = map_volume.get_value_at_coordinate(
                s[-1][0], s[-1][1], s[-1][2],
                space=sft.space, origin=sft.origin)

            thisstreamline_data = np.ones((len(s), dimension)) * np.nan

            thisstreamline_data[0] = p1_data
            thisstreamline_data[-1] = p2_data
            thisstreamline_data = np.asarray(thisstreamline_data)

            streamline_data.append(
                np.reshape(thisstreamline_data,
                           (len(thisstreamline_data), dimension)))
    else:
        for s in sft.streamlines:
            thisstreamline_data = []
            for p in s:
                thisstreamline_data.append(map_volume.get_value_at_coordinate(
                    p[0], p[1], p[2], space=sft.space, origin=sft.origin))

            streamline_data.append(
                np.reshape(thisstreamline_data,
                           (len(thisstreamline_data), dimension)))

    return streamline_data


def project_dpp_to_map(sft, dpp_key, sum_lines=False, endpoints_only=False):
    """
    Saves the values of data_per_point keys to the underlying voxels. Averages
    the values of various streamlines in each voxel. Returns one map per key.
    The streamlines are not preprocessed here. You should probably first
    uncompress your streamlines to have smoother maps.

    Parameters
    ----------
    sft: StatefulTractogram
    dpp_key: str
        The data_per_point key to project to a map.
    sum_lines: bool
        Do not average values of streamlines that cross a same voxel; sum them
        instead.
    endpoints_only: bool
        If true, only project the streamline's endpoints.

    Returns
    -------
    the_map: np.ndarray
        The 3D resulting map.
    """
    sft.to_vox()

    # Using to_corner, if we simply floor the coordinates of the point, we find
    # the voxel where it is.
    sft.to_corner()

    # count: could also use compute_tract_counts_map.
    count = np.zeros(sft.dimensions)
    the_map = np.zeros(sft.dimensions)
    for s in range(len(sft)):
        if endpoints_only:
            points = [0, -1]
        else:
            points = range(len(sft.streamlines[s]))

        for p in points:
            x, y, z = sft.streamlines[s][p, :].astype(int)  # Or floor
            count[x, y, z] += 1
            the_map[x, y, z] += sft.data_per_point[dpp_key][s][p]

    if not sum_lines:
        count = np.maximum(count, 1e-6)  # Avoid division by 0
        the_map /= count

    return the_map


def perform_operation_on_dpp(op_name, sft, dpp_name, endpoints_only=False):
    """
    Peforms an operation on the data per point for all streamlines (mean, sum,
    min, max, correlation). The operation is applied on each point invidiually,
    and thus makes sense if the data_per_point at each point is a vector. The
    result is a new data_per_point.

    Parameters
    ----------
    op_name: str
        A callable that takes a list of streamline data per point (4D) and
        returns a list of streamline data per point.
    sft: StatefulTractogram
        The streamlines used in the operation.
    dpp_name: str
        The name of the data per point to be used in the operation.
    endpoints_only: bool
        If True, will only perform operation on endpoints. Values at other
        points will be set to NaN.

    Returns
    -------
    new_data_per_point: list
        The values that could now be associated to a new data_per_point key.
    """

    # Performing operation
    call_op = OPERATIONS[op_name]
    if endpoints_only:
        new_data_per_point = []
        for s in sft.data_per_point[dpp_name]:
            this_data_per_point = np.nan * np.ones((len(s), 1))
            this_data_per_point[0] = call_op(s[0])
            this_data_per_point[-1] = call_op(s[-1])
            new_data_per_point.append(
                np.reshape(this_data_per_point, (len(this_data_per_point), 1)))
    else:
        new_data_per_point = []
        for s in sft.data_per_point[dpp_name]:
            this_data_per_point = []
            for p in s:
                this_data_per_point.append(call_op(p))
            new_data_per_point.append(
                np.reshape(this_data_per_point, (len(this_data_per_point), 1)))

    return new_data_per_point


def perform_operation_dpp_to_dps(op_name, sft, dpp_name, endpoints_only=False):
    """
    Converts dpp to dps, using a chosen operation.

    Performs an operation across all data_per_points for each streamline (mean,
    sum, min, max, correlation). The result is a data_per_streamline.

    Parameters
    ----------
    op_name: str
        A callable that takes a list of streamline data per streamline and
        returns a list of data per streamline.
    sft: StatefulTractogram
        The streamlines used in the operation.
    dpp_name: str
        The name of the data per point to be used in the operation.
    endpoints_only: bool
        If True, will only perform operation on endpoints. Other points will be
        ignored in the operation.

    Returns
    -------
    new_data_per_streamline: list
        The values that could now be associated to a new data_per_streamline
        key.
    """
    # Performing operation
    call_op = OPERATIONS[op_name]
    if endpoints_only:
        new_data_per_streamline = []
        for s in sft.data_per_point[dpp_name]:
            start = s[0]
            end = s[-1]
            concat = np.concatenate((start[:], end[:]))
            new_data_per_streamline.append(call_op(concat))
    else:
        new_data_per_streamline = []
        for s in sft.data_per_point[dpp_name]:
            s_np = np.asarray(s)
            new_data_per_streamline.append(call_op(s_np))

    return new_data_per_streamline


def perform_pairwise_streamline_operation_on_endpoints(op_name, sft,
                                                       dpp_name='metric'):
    """Peforms an operation across endpoints for each streamline.

    Parameters
    ----------
    op_name: str
        A callable that takes a list of streamline data per streamline and
        returns a list of data per streamline.
    sft: StatefulTractogram
        The streamlines used in the operation.
    dpp_name: str
        The name of the data per point to be used in the operation.

    Returns
    -------
    new_sft: StatefulTractogram
        sft with data per streamline resulting from the operation.
    """
    # Performing operation
    call_op = OPERATIONS[op_name]
    new_data_per_streamline = []
    for s in sft.data_per_point[dpp_name]:
        new_data_per_streamline.append(call_op(s[0], s[-1])[0, 1])

    return new_data_per_streamline


def stream_mean(array):
    return np.squeeze(np.mean(array, axis=0))


def stream_sum(array):
    return np.squeeze(np.sum(array, axis=0))


def stream_min(array):
    return np.squeeze(np.min(array, axis=0))


def stream_max(array):
    return np.squeeze(np.max(array, axis=0))


def stream_correlation(array1, array2):
    return np.corrcoef(array1, array2)


OPERATIONS = {
    'mean': stream_mean,
    'sum': stream_sum,
    'min': stream_min,
    'max': stream_max,
    'correlation': stream_correlation,
}
