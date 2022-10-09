"""
This file contains functions related to policy and experiment evaluation.
"""
import numpy as np


def compute_mota_scores(num_gt_matrix, num_sectors):
    """
    Takes an n x 2 x 200 array and returns an n x num_sectors array with the
    computed MOTA score over 200 // num_sectors window sizes with that stride
    over the 200 frames.
    """
    assert np.issubdtype(num_gt_matrix.dtype, np.integer), \
        f"num_gt_matrix is not integer type ({num_gt_matrix.dtype})"
    assert num_gt_matrix.shape[1:] == (2, 200), num_gt_matrix.shape
    arr = num_gt_matrix.reshape(*num_gt_matrix.shape[:2], num_sectors, -1)
    # there should be no nans, infs, or negative values
    assert np.all(np.isfinite(arr) & (arr >= 0)), \
        "num_gt_matrix given has negative values"
    arr = np.cumsum(arr, axis=-1)
    cum_num = arr[:, 0, :, :]
    cum_gt = arr[:, 1, :, :]

    def og_mota(cum_num, cum_gt):
        mota_arr = 1 - cum_num / (cum_gt)  # n x num_sectors x sector_len
        og_mota = mota_arr[:, :, -1]
        return og_mota

    def method_2(cum_num, cum_gt):
        num = cum_num[:, :, -1]
        gt = cum_gt[:, :, -1]
        return np.where(
            gt != 0,  # if gt!=0
            1 - num / gt,  # use traditional method
            np.where(
                num != 0,  # elif gt!=0 and num!=0
                -3 - num / 100,  # range of [-inf, -3]
                np.ones_like(num)))  # else return 1

    # don't want to hear divide by zero warning repeatedly
    old_settings = np.seterr(divide='ignore', invalid='ignore')
    mota = og_mota(cum_num, cum_gt)
    np.seterr(**old_settings)
    # mota = method_2(cum_num, cum_gt)

    return mota * 100


def compute_mota_numerator(num_gt_matrix, num_sectors):
    """
    Takes an n x 2 x 200 array and returns an n x num_sectors array with the
    computed MOTA score over 200 // num_sectors window sizes with that stride
    over the 200 frames.
    """
    assert np.issubdtype(num_gt_matrix.dtype, np.integer), \
        f"num_gt_matrix is not integer type ({num_gt_matrix.dtype})"
    assert num_gt_matrix.shape[1:] == (2, 200), num_gt_matrix.shape
    arr = num_gt_matrix.reshape(*num_gt_matrix.shape[:2], num_sectors, -1)
    # there should be no nans, infs, or negative values
    assert np.all(np.isfinite(arr) & (arr >= 0)), \
        "num_gt_matrix given has negative values"
    arr = np.cumsum(arr, axis=-1)
    cum_num = arr[:, 0, :, :]

    return cum_num[:, :, -1]


def compute_other_tracking_metrics(fp_fn_matrix, num_sectors):
    """
    Takes an n x 5 x 200 array and returns an n x num_sectors array with the
    computed auxillary metric over 200 // num_sectors window sizes with that
    stride over the 200 frames.
    middle dim: FP, FN, IDS, motp_num, motp_denominator
    """

    assert fp_fn_matrix.shape[1:] == (5, 200), fp_fn_matrix.shape
    arr = fp_fn_matrix.reshape(*fp_fn_matrix.shape[:2], num_sectors, -1)
    # there should be no nans, infs, or negative values
    assert np.all(np.isfinite(arr) & (arr >= 0)), \
        "fp_fn_matrix given has negative values"

    arr = np.sum(arr, axis=-1)
    motp = arr[:, 3, :] / arr[:, 4, :]

    return arr[:, :3, :].astype('int64'), motp
