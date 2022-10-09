"""
Functionality related to collecting raw file data to organized tables
"""
import json
from pathlib import Path

import numpy as np

import pandas as pd

from tqdm import tqdm

from ad_config_search.utils import fix_pylot_profile  # , replace_first_line
from ad_config_search.utils import prepend_line_to_file


def get_fn_without_ext(path):
    """
    pathlib can't do .stub or .name when there's a period (.) in the fn
    """
    return ".".join(str(path).split("/")[-1].split(".")[:-1])


def split_run(run):
    """turns trial output file format back into dict of properties"""
    return {x.split("=")[0]: x.split("=")[1] for x in run.split("__")}


def get_detection_tracking_accuracies(csv_path,
                                      time_mode,
                                      anchor,
                                      end_frame_strategy,
                                      with_ts=False):
    """
    mode is one of ["sync", "timely"].

    with_ts indicates whether to also pass the timestamps in the video that
    correspond to the metric scores.

    Returns a dictionary
    {
        metric_name: (ts[if with_ts], all the scores in the trial)
    }
    """
    csv_path = Path(csv_path)
    prepend_line_to_file(
        csv_path, ("log_ts,simulator_ts,operator,anchor,end_frame_strategy"
                   ",extra_info,value"))
    df = pd.read_csv(csv_path)
    eval_dict = {}
    eval_df = df[
        (df["operator"] == "coco_detection_eval_operator_{}".format(time_mode))
        & (df["anchor"] == anchor)
        & (df["end_frame_strategy"] == end_frame_strategy)]
    for k in eval_df["extra_info"].unique():
        ts = (eval_df[eval_df["extra_info"] == k]
              ["simulator_ts"].values.astype(float))
        value = \
            eval_df[eval_df["extra_info"] == k]["value"].values.astype(float)
        if with_ts:
            eval_dict["D_" + str(k)] = (ts, value)
        else:
            eval_dict["D_" + str(k)] = value

    operator_name = f"tracking_eval_operator_{time_mode}"
    if time_mode == "timely":
        operator_name += f"_{end_frame_strategy}"
    eval_df = df[(df["operator"] == operator_name)
                 & (df["anchor"] == anchor)
                 & (df["end_frame_strategy"] == end_frame_strategy)]
    for k in eval_df["extra_info"].unique():
        ts = (eval_df[eval_df["extra_info"] == k]
              ["simulator_ts"].values.astype(float))
        value = \
            eval_df[eval_df["extra_info"] == k]["value"].values.astype(float)
        if with_ts:
            eval_dict["T_" + str(k)] = (ts, value)
        else:
            eval_dict["T_" + str(k)] = value
    return eval_dict


def get_latencies(json_path):
    fix_pylot_profile(json_path)
    with open(json_path, 'r') as f:
        data = json.load(f)
    return {
        tup: rows["dur"].values / 1000
        for tup, rows in pd.DataFrame(data).groupby(["name"])
    }


def get_binned_array(ts, values):
    """
    Given an array of timestamps and their corresponding values, creates
    a single array where each cell represents a timestamp, and places the
    values in the correct index. An index whose corresponding timestamp has
    no value is assigned nan.

    Right now implementation assumes that timestamps start at 0, and increase
    at increments of 100.
    """
    # ts can be out of order sometimes
    ts, values = zip(*sorted(list(zip(ts, values)), key=lambda x: x[0]))
    ts = np.array(ts)
    values = np.array(values)
    bincount = np.bincount(ts.astype(int) // 100)
    has_2_mask = np.arange(len(bincount))[bincount == 2]
    assert len(has_2_mask) <= 1, (len(has_2_mask))
    if len(has_2_mask) == 1:
        bincount[has_2_mask[0]] = 1
        assert has_2_mask[0]-1 >= 0 and bincount[has_2_mask[0]-1] == 0, \
            has_2_mask[0]
        bincount[has_2_mask[0] - 1] = 1
    bincount = bincount.astype(float)
    bincount[bincount == 0] = np.nan
    bincount[~np.isnan(bincount)] = values
    assert len(bincount) <= 201, (len(bincount))
    bincount = np.pad(bincount, (0, 201 - len(bincount)),
                      constant_values=np.nan)
    return bincount


def get_fp_fn(csv_path,
              time_mode,
              anchor,
              end_frame_strategy,
              from_mota=False):
    """
    Takes a csv_path containing logged trial data of a configuration running
    in an environment.

    Returns a 1 x 2 x 200 numpy array. The first row contains the numerator
    in the MOTA computation (see section 2.1.3 of
    https://cvhci.anthropomatik.kit.edu/\
        ~stiefel/papers/ECCV2006WorkshopCameraReady.pdf)
    per frame and the second row contains the denominator, g_t, the number of
    ground truth objects in the frame.

    While the numerator items for each frame were logged and are scraped in
    this function, g_t wasn't. As a result, we recover g_t from the MOTA
    score and the numerator reported in the csv file.

    The purpose of this function is to get the per-frame statistics so that
    the MOTA accumulation can afterward be freely computed via accumulation
    starting at any point in the scene. The mota score reported in the csv file
    is computed by accumulating over frames from the beginning of the video.
    This is not useful when we wish to just compute MOTA scores over segments
    of the video.
    """
    def forward_fill_nans(arr, including_start=True):
        if including_start and np.isnan(arr[0]):
            arr[0] = 0
        # taken from https://stackoverflow.com/a/41191127/1546071
        mask = np.isnan(arr)
        idx = np.where(~mask, np.arange(len(mask)), 0)
        np.maximum.accumulate(idx, axis=0, out=idx)
        out = arr[idx]
        return out

    def prep(arr, name):
        arr = forward_fill_nans(arr)
        rounded = np.round(arr)
        assert (np.all(arr >= 0)
                and np.all(np.abs(rounded - arr) < 1e-5)), (name, csv_path)
        return rounded.astype(int)

    dct = get_detection_tracking_accuracies(csv_path,
                                            time_mode,
                                            anchor,
                                            end_frame_strategy,
                                            with_ts=True)

    # forward fill the nans because tracking eval doesn't output when there
    # are no grouth truth and no predictions
    cum_misses = prep(get_binned_array(*dct["T_num_misses"]), "misses")
    cum_fpos = prep(get_binned_array(*dct["T_num_false_positives"]), "fpos")
    cum_nswitch = prep(get_binned_array(*dct["T_num_switches"]), "switches")

    cum_motp = forward_fill_nans(get_binned_array(*dct["T_motp"]))
    cum_num_objects = prep(get_binned_array(*dct["T_num_objects"]),
                           "num_objects")
    nswitch = np.diff(cum_nswitch)
    fpos = np.diff(cum_fpos)
    miss = np.diff(cum_misses)
    cum_num_matches = cum_num_objects - cum_nswitch - cum_misses
    cum_num_detections = cum_num_matches + cum_nswitch
    cum_motp_num = cum_num_detections * cum_motp
    motp_num = np.diff(cum_motp_num)
    num_detections = np.diff(cum_num_detections)

    np_array = np.array([[fpos, miss, nswitch, motp_num, num_detections]])
    return np_array


def get_gt_and_numerator(csv_path,
                         time_mode,
                         anchor,
                         end_frame_strategy,
                         from_mota=False):
    """
    Takes a csv_path containing logged trial data of a configuration running
    in an environment.

    Returns a 1 x 2 x 200 numpy array. The first row contains the numerator
    in the MOTA computation (see section 2.1.3 of
    https://cvhci.anthropomatik.kit.edu/\
        ~stiefel/papers/ECCV2006WorkshopCameraReady.pdf)
    per frame and the second row contains the denominator, g_t, the number of
    ground truth objects in the frame.

    While the numerator items for each frame were logged and are scraped in
    this function, g_t wasn't. As a result, we recover g_t from the MOTA
    score and the numerator reported in the csv file.

    The purpose of this function is to get the per-frame statistics so that
    the MOTA accumulation can afterward be freely computed via accumulation
    starting at any point in the scene. The mota score reported in the csv file
    is computed by accumulating over frames from the beginning of the video.
    This is not useful when we wish to just compute MOTA scores over segments
    of the video.
    """
    def forward_fill_nans(arr, including_start=True):
        if including_start and np.isnan(arr[0]):
            arr[0] = 0
        # taken from https://stackoverflow.com/a/41191127/1546071
        mask = np.isnan(arr)
        idx = np.where(~mask, np.arange(len(mask)), 0)
        np.maximum.accumulate(idx, axis=0, out=idx)
        out = arr[idx]
        return out

    def prep(arr, name):
        arr = forward_fill_nans(arr)
        rounded = np.round(arr)
        assert (np.all(arr >= 0)
                and np.all(np.abs(rounded - arr) < 1e-5)), (name, csv_path)
        return rounded.astype(int)

    dct = get_detection_tracking_accuracies(csv_path,
                                            time_mode,
                                            anchor,
                                            end_frame_strategy,
                                            with_ts=True)
    # forward fill the nans because tracking eval doesn't output when there
    # are no grouth truth and no predictions
    cum_misses = prep(get_binned_array(*dct["T_num_misses"]), "misses")
    cum_fpos = prep(get_binned_array(*dct["T_num_false_positives"]), "fpos")
    cum_nswitch = prep(get_binned_array(*dct["T_num_switches"]), "switches")
    cum_numerator = cum_misses + cum_fpos + cum_nswitch
    numerator = np.diff(cum_numerator)
    if from_mota:
        mota = forward_fill_nans(get_binned_array(*dct["T_mota"]), False)
        assert np.all((mota <= 100) | np.isnan(mota)), csv_path
        cum_gt_float = cum_numerator / ((100 - mota) + 1e-7) * 100
        cum_gt = np.round(cum_gt_float)
        # cum_gt_float should be basically integers in float form (or nan).
        # We assert this below.
        assert np.all((np.abs(cum_gt_float - cum_gt) < 1e-2)
                      | np.isnan(cum_gt_float)), ((cum_gt_float - cum_gt),
                                                  csv_path)
        cum_gt = forward_fill_nans(cum_gt)  # remove nans from division by mota
        cum_gt = cum_gt.astype(int)
        gt = np.diff(cum_gt)
    else:
        cum_gt = prep(get_binned_array(*dct["T_num_objects"]), "num_objects")
        gt = np.diff(cum_gt).astype(int)

    assert np.all(gt >= 0) and np.all(numerator >= 0), (gt, cum_gt, mota,
                                                        cum_numerator,
                                                        csv_path)
    return np.array([[numerator, gt]])


def shift_rows_back_one(df, scenario_column):
    def get_section_number(section_name):
        return int(section_name.split("-P")[-1].split("_")[0])

    total_slices = int(df[scenario_column].iloc[0].split("_")[-1])

    # get rid of last sections
    df_cut = df[df[scenario_column].map(
        lambda x: get_section_number(x) < total_slices - 1)]
    df_copy = df_cut.copy()

    def increment_section_number(section_name):
        first_split = section_name.split("-P")
        first_part = first_split[0]
        last_part = first_split[-1]
        last_part_split = last_part.split("_")
        new_section_number = int(last_part_split[0]) + 1
        assert total_slices == int(last_part_split[1]), section_name
        assert new_section_number < total_slices, section_name
        return first_part + f"-P{new_section_number}_{last_part_split[1]}"

    df_copy[scenario_column] = [
        increment_section_number(name)
        for name in tqdm(df_copy[scenario_column])
    ]
    return df_copy


def deprecated_data_csv_paths(base_bucket_path, extension="csv"):
    base_bucket_path = Path(base_bucket_path)
    loc1 = list(
        set(
            list((base_bucket_path /
                  "03-22-v11-v2/").glob(f"**/*.{extension}"))) -
        set(
            list((base_bucket_path /
                  "03-22-v11-v2/1616504995553").glob(f"**/*.{extension}"))))
    all_csvs = loc1
    return all_csvs


def max_age_extension(base_bucket_path, extension="csv"):
    """
    The location of configuration sweep results using higher max-age knob
    values (9, 11).
    """
    base_bucket_path = Path(base_bucket_path)
    repeats = [
        Path(base_bucket_path /
             '04-29-v15-higher-t-max-age/1619760191304-i-03827da7b69fa3c87'),
        Path(base_bucket_path /
             '04-29-v15-higher-t-max-age/1619760191595-i-08ada5fd7f803ddb4'),
        Path(base_bucket_path /
             '04-29-v15-higher-t-max-age/1619760592039-i-03827da7b69fa3c87'),
    ]
    repeats = [list(r.glob(f"**/*.{extension}")) for r in repeats]
    repeats = set(
        [j for i in repeats for j in i]
    )  # joining the lists together: https://stackoverflow.com/a/716761/1546071
    loc1 = list(
        set(
            list(
                Path(base_bucket_path / "04-29-v15-higher-t-max-age/").glob(
                    f"**/*.{extension}"))) - repeats)
    all_csvs = loc1
    return all_csvs


def full_config_space_paths(base_bucket_path, extension="csv"):
    """
    A function that gets the paths of all the Waymo config sweeps we've done.
    Each path is a directory, containing a csv, json, log, jsonl files that
    represent one config x video trial. This list of paths can then be used to
    curate the final config x knob x score dataset.
    """
    base_bucket_path = Path(base_bucket_path)
    repeats = [
        Path(base_bucket_path / '04-09-v12/1618009510077'),
        Path(base_bucket_path / '04-09-v12/1618009073297'),
        Path(base_bucket_path / '04-09-v12/1618009074553'),
        Path(base_bucket_path / '04-09-v12/1618009510401'),
        Path(base_bucket_path / '04-09-v12/1618009073515'),
        Path(base_bucket_path / '04-09-v12/1618009073559'),
        Path(base_bucket_path / '04-09-v12/1618009073351'),
        Path(base_bucket_path / '04-09-v12/1618009073210'),
        Path(base_bucket_path / '04-09-v12/1618009073524'),
        Path(base_bucket_path / '04-09-v12/1618009510058'),
        Path(base_bucket_path / '04-09-v12/1618009511093'),
        Path(base_bucket_path / '04-09-v12/1618009095032'),
        Path(base_bucket_path / '04-09-v12/1618009509847'),
        Path(base_bucket_path / '04-09-v12/1618009073399'),
        Path(base_bucket_path / '04-09-v12/1618009510036'),
        Path(base_bucket_path / '04-09-v12/1618009094595'),
        Path(base_bucket_path / '04-09-v12/1618009509882'),
        Path(base_bucket_path / '04-09-v12/1618009094785')
    ]
    repeats = [list(r.glob(f"**/*.{extension}")) for r in repeats]
    repeats = set(
        [j for i in repeats for j in i]
    )  # joining the lists together: https://stackoverflow.com/a/716761/1546071
    # waymo training 0-5, full configuration space
    loc1 = list(
        set(
            list(
                Path(base_bucket_path /
                     "04-09-v12/").glob(f"**/*.{extension}"))) - repeats)
    loc2 = list((Path(base_bucket_path) /
                 "08-02-waymo-6-11-full-v2").glob(f"**/*.{extension}"))
    all_csvs = loc1 + loc2
    return all_csvs


def data_csv_paths(base_bucket_path, extension="csv"):
    """
    A function that gets the paths of all the Waymo config sweeps we've done.
    Each path is a directory, containing a csv, json, log, jsonl files that
    represent one config x video trial. This list of paths can then be used to
    curate the final config x knob x score dataset.
    """
    base_bucket_path = Path(base_bucket_path)
    repeats = [
        Path(base_bucket_path / '04-09-v12/1618009510077'),
        Path(base_bucket_path / '04-09-v12/1618009073297'),
        Path(base_bucket_path / '04-09-v12/1618009074553'),
        Path(base_bucket_path / '04-09-v12/1618009510401'),
        Path(base_bucket_path / '04-09-v12/1618009073515'),
        Path(base_bucket_path / '04-09-v12/1618009073559'),
        Path(base_bucket_path / '04-09-v12/1618009073351'),
        Path(base_bucket_path / '04-09-v12/1618009073210'),
        Path(base_bucket_path / '04-09-v12/1618009073524'),
        Path(base_bucket_path / '04-09-v12/1618009510058'),
        Path(base_bucket_path / '04-09-v12/1618009511093'),
        Path(base_bucket_path / '04-09-v12/1618009095032'),
        Path(base_bucket_path / '04-09-v12/1618009509847'),
        Path(base_bucket_path / '04-09-v12/1618009073399'),
        Path(base_bucket_path / '04-09-v12/1618009510036'),
        Path(base_bucket_path / '04-09-v12/1618009094595'),
        Path(base_bucket_path / '04-09-v12/1618009509882'),
        Path(base_bucket_path / '04-09-v12/1618009094785')
    ]
    repeats = [list(r.glob(f"**/*.{extension}")) for r in repeats]
    repeats = set(
        [j for i in repeats for j in i]
    )  # joining the lists together: https://stackoverflow.com/a/716761/1546071
    # waymo training 0-5, full configuration space
    loc1 = list(
        set(
            list(
                Path(base_bucket_path /
                     "04-09-v12/").glob(f"**/*.{extension}"))) - repeats)
    # waymo training 6-32, validtion 0-6, pruned subset of configuration space
    loc2 = list(
        Path(base_bucket_path /
             "05-21-tracking-waymo-all").glob(f"**/*.{extension}"))
    # waymo validation 7, pruned subset of configuration space
    loc3 = list(
        Path(base_bucket_path / "05-23-waymo-val7").glob(f"**/*.{extension}"))
    loc4_repeat_dirs = [
        "1621840205045-i-08b238917d2ab1822",
        "1621842694835-i-08b238917d2ab1822",
        "1621842839758-i-08b238917d2ab1822",
        "1621845998216-i-08b238917d2ab1822",
        "1621851374494-i-08b238917d2ab1822",
        "1621854275396-i-08b238917d2ab1822",
        "1621858242711-i-0198e6e9a051be3a0",
        "1621859702364-i-08b238917d2ab1822",
        "1621859746803-i-08b238917d2ab1822",
        "1621860648130-i-08b238917d2ab1822",
        "1621860775489-i-08b238917d2ab1822",
        "1621861601454-i-0c00239a53cbf6df7",
        "1621862242745-i-08b238917d2ab1822",
        "1621863596136-i-01612065844909b4b",
        "1621863738844-i-01612065844909b4b",
        "1621864333426-i-08b238917d2ab1822",
        "1621864543573-i-08b238917d2ab1822",
        "1621865243544-i-08b238917d2ab1822",
        "1621866682364-i-00179dd7425f98cea",
        "1621866750843-i-08b238917d2ab1822",
        "1621867103241-i-08b238917d2ab1822",
        "1621868102409-i-0bab8a9e87061d080",
        "1621869376412-i-05f21598e07d37e77",
        "1621869950670-i-0a93c7320f478fccd",
        "1621870798819-i-0ca54f99ab34e4100",
        "1621872433700-i-0e26a12607ed3e37e",
        "1621873562720-i-00f3e6259151d0534",
        "1621875469397-i-08b238917d2ab1822",
        "1621876463272-i-08b238917d2ab1822",
        "1621877168861-i-08b238917d2ab1822",
        "1621877610628-i-09b93cea411290ba8",
        "1621877949446-i-0ab25683cc15929d6",
    ]

    loc4_repeats = [
        Path('../../ad-config-search/05-24-waymo-det-seq-inf-baseline/') / p
        for p in loc4_repeat_dirs
    ]
    loc4_repeats = [list(r.glob(f"**/*.{extension}")) for r in loc4_repeats]
    loc4_repeats = set(
        [j for i in loc4_repeats for j in i]
    )  # joining the lists together: https://stackoverflow.com/a/716761/1546071
    # waymo training 6-32, val 0-7 infinite mode (locs 2-3 are not infinite)
    loc4 = list(
        set(
            list(
                Path(base_bucket_path / "05-24-waymo-det-seq-inf-baseline").
                glob(f"**/*.{extension}"))) - set(loc4_repeats))
    all_csvs = loc1 + loc2 + loc3 + loc4 \
        # + max_age_extension(base_bucket_path, extension)

    return all_csvs
