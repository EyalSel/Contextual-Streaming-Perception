from itertools import chain, product
from pathlib import Path

import pandas as pd
from offline_pylot.detection_models import (datasets_dict,
                                            detection_model_dict,
                                            tracker_model_dict)
from tqdm import tqdm

DETECTION_MODELS = list(detection_model_dict.values())
DETECTION_MODELS = [
    detection_model_dict[k]
    for k in ["efficientdet-d1", "efficientdet-d3", "efficientdet-d5"]
]
DETECTION_MODELS = [
    detection_model_dict[k]
    for k in ["efficientdet-d1", "efficientdet-d3", "efficientdet-d5"]
]
DETECTION_MODELS = [
    detection_model_dict[k] for k in [
        # "efficientdet-d1", "efficientdet-d2",
        "efficientdet-d3",
        "efficientdet-d4",
        "efficientdet-d5",
        "efficientdet-d6",
        "efficientdet-d7",
        "efficientdet-d7x"
    ]
]
# DETECTION_MODELS = [detection_model_dict[k] for k in ["efficientdet-d1"]]
# DETECTION_MODELS = [DETECTION_MODELS[-1]]

# SCENARIO_RUNS = datasets_dict["waymo_cherrypick_v1"]
SCENARIO_RUNS = datasets_dict["waymo_0000_pl"]
SCENARIO_RUNS = datasets_dict["waymo_0001_pl"]
SCENARIO_RUNS = list(
    chain.from_iterable(
        [datasets_dict[f"waymo_{str(i).zfill(4)}_pl"] for i in range(32)] +
        [datasets_dict[f"waymo_validation_000{i}_pl"] for i in range(7)]))
# SCENARIO_RUNS = datasets_dict["waymo_T0_S0"]
# SCENARIO_RUNS = [datasets_dict["waymo_T4_S23"][0]]
# SCENARIO_RUNS = datasets_dict["argo_train_1"]
# SCENARIO_RUNS = [SCENARIO_RUNS[-1]]

# SCENARIO_RUNS = list(chain.from_iterable([
#     datasets_dict[f"waymo_00{str(i).zfill(2)}_pl"]
#     for i in range(6, 32)
# ] + [
#     datasets_dict[f"waymo_validation_000{i}_pl"]
#     for i in range(7)
# ]))

CONFIDENCE_THRESHOLDS = [
    # 0.1,
    0.4,
    # 0.5,
    # 0.6,
    # 0.7,
]

DETECTION_SEQUENCER_POLICIES = [
    # None,
    "infinite",
    "tail-aware",
    # "wait",
    # "eager"
]

detection_options = [
    DETECTION_MODELS, CONFIDENCE_THRESHOLDS, DETECTION_SEQUENCER_POLICIES
]

# TRACKING_MODELS = [tracker_model_dict[k] for k in ["sort", "da_siam_rpn_VOT", "deep_sort"]]
TRACKING_MODELS = [tracker_model_dict[k] for k in ["sort"]]

TRACKING_MIN_MATCH_IOUS = [
    0.1,
    # 0.2,
    # 0.3,
    # 0.5,
    # 0.75,
    # 0.9
]

TRACKING_MAX_AGE = [
    1,
    3,
    # 5,
    7
]

TRACKING_IGNORE_SHORT_HIST_LENGTHS = [
    1,
    # 2,
    # 3,
    # 5,
]

TRACKING_EVERY_NTH_DETECTIONS = [
    1,
    # 2,
    # 3,
    # 5,
]

tracking_options = [
    TRACKING_MODELS,
    TRACKING_MIN_MATCH_IOUS,
    TRACKING_MAX_AGE,
    TRACKING_IGNORE_SHORT_HIST_LENGTHS,
    TRACKING_EVERY_NTH_DETECTIONS,
]

TRACKING_NUM_STEPS = [10]

prediction_options = [TRACKING_NUM_STEPS]

# TOGGLES
experiment_toggles = {
    "save_frames": False,
    "cache_detection": True,
    "turn_on_tracking": True,
    "use_cached_detection": True,
    "eval_min_matching_iou": 0.5
}


def config_sweep(experiment_toggles):
    run_columns = ["run"]
    detection_columns = ["D-model", "D-conf", "D-seq-pol"]
    tracking_columns = [
        "T-model", "T-min-iou", "T-max-age", "T-ignore-len", "T-every-nth-det"
    ]
    if experiment_toggles["turn_on_tracking"]:
        knobs = [SCENARIO_RUNS] + detection_options + tracking_options
        columns = run_columns + detection_columns + tracking_columns
    else:
        knobs = [SCENARIO_RUNS] + detection_options
        columns = run_columns + detection_columns
        assert len(CONFIDENCE_THRESHOLDS) == 1 and CONFIDENCE_THRESHOLDS[
            0] == 0.1, "Trying to run detection caching with >0.1 threshold"
    import sys
    print(knobs, [len(x) for x in knobs], file=sys.stderr)
    all_data = product(*knobs)
    df = pd.DataFrame(columns=columns, data=all_data)
    return df.to_dict('records')


def configs_from_dataframe(df: pd.DataFrame):
    return [{
        "run":
        run,
        "T-ignore-len":
        t_ignore_len,
        **rest_of_config,
        "D-model":
        detection_model_dict[rest_of_config["D-model"]],
        "T-model":
        tracker_model_dict[rest_of_config["T-model"]],
    } for run, t_ignore_len, rest_of_config in product(*[
        SCENARIO_RUNS, TRACKING_IGNORE_SHORT_HIST_LENGTHS,
        df.to_dict('records')
    ])]


def specific_configs():
    df = pd.DataFrame({
        'run': [datasets_dict["waymo_T4_S23"][0]] * 2,
        'D-model': [
            detection_model_dict[k]
            for k in ['efficientdet-d2', 'efficientdet-d7x']
        ],
        'D-conf': [0.3, 0.3],
        'D-seq-pol': ['tail-aware', 'tail-aware'],
        "T-model": [tracker_model_dict["sort"]] * 2,
        'T-min-iou': [0.1, 0.1],
        'T-max-age': [7, 7],
        'T-every-nth-det': [1] * 2,
        "T-ignore-len": [1] * 2,
    })
    df = pd.DataFrame({
        'run': [datasets_dict["waymo_T4_S0"][0]] * 2,
        'D-model': [
            detection_model_dict[k]
            for k in ['efficientdet-d4', 'efficientdet-d7x']
        ],
        'D-conf': [0.3, 0.3],
        'D-seq-pol': ['tail-aware', 'tail-aware'],
        "T-model": [tracker_model_dict["sort"]] * 2,
        'T-min-iou': [0.1, 0.1],
        'T-max-age': [1, 1],
        'T-every-nth-det': [1] * 2,
        "T-ignore-len": [1] * 2,
    })
    df = pd.DataFrame({
        'run': [datasets_dict["waymo_T4_S24"][0]] * 2,
        'D-model': [
            detection_model_dict[k]
            for k in ['efficientdet-d4', 'efficientdet-d7x']
        ],
        'D-conf': [0.3, 0.3],
        'D-seq-pol': ['tail-aware', 'tail-aware'],
        "T-model": [tracker_model_dict["sort"]] * 2,
        'T-min-iou': [0.1, 0.1],
        'T-max-age': [1, 1],
        'T-every-nth-det': [1] * 2,
        "T-ignore-len": [1] * 2,
    })
    import sys
    with pd.option_context('display.max_rows', None, 'display.max_columns',
                           None):
        print(df, file=sys.stderr)
    return df.to_dict('records')


def logical_trial_config_to_command(trial_configs, experiment_toggles):
    if experiment_toggles["turn_on_tracking"]:
        (
            run,  # scenario
            det_model,
            det_conf_thres,
            det_seq_policy,  # det knobs
            track_model,
            track_min_iou,
            track_max_age,
            track_ignore_len,
            track_every_nth_det  # track knobs
        ) = [
            trial_configs[k] for k in [
                "run", "D-model", "D-conf", "D-seq-pol", "T-model",
                "T-min-iou", "T-max-age", "T-ignore-len", "T-every-nth-det"
            ]
        ]
    else:
        (
            run,  # scenario
            det_model,
            det_conf_thres,
            det_seq_policy,  # det knobs
        ) = [
            trial_configs[k] for k in [
                "run",
                "D-model",
                "D-conf",
                "D-seq-pol",
            ]
        ]
    camera_height, camera_width = run.resolution
    if experiment_toggles["turn_on_tracking"]:
        run_name = "{}__{}__{}__{}__{}__{}__{}__{}__{}__{}__{}__{}".format(
            "dataset={}".format(run.dataset), "run={}".format(
                run.sequence_name), "D-model={}".format(det_model.name),
            "res-h={}".format(camera_height), "res-w={}".format(camera_width),
            "D-conf={}".format(det_conf_thres),
            "D-seq-pol={}".format(det_seq_policy), "T-model={}".format(
                track_model.name), "T-min-iou={}".format(track_min_iou),
            "T-max-age={}".format(track_max_age),
            "T-ignore-len={}".format(track_ignore_len),
            "T-every-nth-det={}".format(track_every_nth_det))
    else:
        run_name = "{}__{}__{}__{}__{}__{}__{}".format(
            "dataset={}".format(run.dataset),
            "run={}".format(run.sequence_name),
            "D-model={}".format(det_model.name),
            "res-h={}".format(camera_height), "res-w={}".format(camera_width),
            "D-conf={}".format(det_conf_thres),
            "D-seq-pol={}".format(det_seq_policy))
    # The code is set up right now so that the run datastructure specifies where the detection cache directory
    # is, but not the specific file within the directory. This arbitration is done below this comment, because
    # it is the same place that chooses the name for file where a potential detection cache would be saved.
    # This doesn't necessarily need to be the case, and in fact should probably not be.
    cached_det_run_name = "{}__{}__{}__{}__{}".format(
        "dataset={}".format(run.dataset),
        "run={}".format(run.sequence_name),
        "D-model={}".format(det_model.name),
        "res-h={}".format(camera_height),
        "res-w={}".format(camera_width),
    )

    Path("sweep_files").mkdir(exist_ok=True)

    # SCENARIO FLAGS
    dataset_name_flag = "--dataset_name={}".format(run.dataset.split("_")[0])
    dataset_path_flag = "--dataset_path={}".format(run.path)
    dataset_label_map_flag = "--dataset_label_map={}".format(run.label_map)
    dataset_frame_interval_flag = "--dataset_frame_interval={}".format(
        run.frame_interval_ms)
    dataset_flags = "{} {} {} {}".format(dataset_name_flag, dataset_path_flag,
                                         dataset_label_map_flag,
                                         dataset_frame_interval_flag)

    # DETECTION MODEL FLAGS
    if experiment_toggles["use_cached_detection"]:
        assert run.cached_detection_path is not None, \
            "Indicated to use cached detection, but the dataset doesn't have a specified cached location"
        cached_detection_source_path_flag = \
            ("--cache_obstacle_detection_source_path={}/"
                "{}__detection_cache.jsonl").format(run.cached_detection_path, cached_det_run_name)
        detection_source_flags = cached_detection_source_path_flag
    else:
        det_model_path_flag = "--obstacle_detection_model_paths={}".format(
            det_model.path)
        detection_source_flags = det_model_path_flag

    # Used both when using cached source or running detection because
    # the flag is used used both by the detection model and the caching
    # to file operator for metadata logging purposes.
    det_model_name_flag = "--obstacle_detection_model_names={}".format(
        det_model.name)
    detection_source_flags = "{} {}".format(detection_source_flags,
                                            det_model_name_flag)

    det_model_label_map_flag = \
        "--path_coco_labels={}".format(det_model.label_map)
    camera_res_flags = "--camera_image_height={} --camera_image_width={}"\
        .format(camera_height, camera_width)

    detection_model_flags = "{} {} {}".format(detection_source_flags,
                                              det_model_label_map_flag,
                                              camera_res_flags)

    # DETECTION EVAL FLAGS
    coco_aggregate_eval_flags = ("--coco_detection_eval_lookback={} "
                                 "--coco_detection_eval_freq={}").format(
                                     1000 // run.frame_interval_ms,
                                     (100 // run.frame_interval_ms) * 2)
    detection_eval_flags = \
        "--evaluate_obstacle_detection {}".format(coco_aggregate_eval_flags)

    # FINAL DETECTION CONFIG FLAGS
    detection_confidence_threshold_flag = \
        "--obstacle_detection_min_score_threshold={}".format(det_conf_thres)
    if experiment_toggles["cache_detection"]:
        cache_detection = \
            ("--cache_obstacle_detection_destination_path=sweep_files/"
                "{}__detection_cache.jsonl").format(cached_det_run_name)
    else:
        cache_detection = ""
    if det_seq_policy is not None:
        det_seq_policy_flag = "--det_sequencer_policy={}".format(
            det_seq_policy)
    else:
        det_seq_policy_flag = ""
    detection_config_flags = \
        "--obstacle_detection {} {} {} {}".format(detection_model_flags,
                                                  detection_confidence_threshold_flag,
                                                  cache_detection,
                                                  det_seq_policy_flag)

    # TRACKING FLAGS
    if experiment_toggles["turn_on_tracking"]:
        tracker_type_flag = "--tracker_type={}".format(track_model.name)
        if track_model.path_flag_dict is not None:
            tracker_weight_flags = " ".join(
                [f"--{k}={v}" for k, v in track_model.path_flag_dict.items()])
        else:
            tracker_weight_flags = ""
        min_matching_iou_flag = "--min_matching_iou={}".format(track_min_iou)
        obstacle_track_max_age_flag = "--obstacle_track_max_age={}".format(
            track_max_age)
        ignore_obstacles_with_short_history_flag = \
            "--ignore_obstacles_with_short_history={}".format(track_ignore_len)
        track_every_nth_detection_flag = "--track_every_nth_detection={}".format(
            track_every_nth_det)
        track_eval_iou_flag = "--eval_min_matching_iou={}".format(
            experiment_toggles["eval_min_matching_iou"])
        tracking_config_flags = "--obstacle_tracking {} {} {} {} {} {} {} --evaluate_obstacle_tracking=true".format(
            tracker_type_flag, tracker_weight_flags, min_matching_iou_flag,
            obstacle_track_max_age_flag,
            ignore_obstacles_with_short_history_flag,
            track_every_nth_detection_flag, track_eval_iou_flag)
    else:
        tracking_config_flags = ""

    # OPTIONAL SAVE FRAME FLAGS
    if experiment_toggles["save_frames"]:
        Path("sweep_files/{}".format(run_name)).mkdir(exist_ok=True)
        detector_output_flag = "--log_detector_output=True"
        data_path_flag = "--data_path=sweep_files/{}".format(run_name)
        save_frames_flags = \
            "{} {}".format(detector_output_flag, data_path_flag)
    else:
        save_frames_flags = ""

    # OUTPUT FILES FLAGS
    profile_fn_flag = \
        "--profile_file_name=sweep_files/{}.json".format(run_name)
    csv_log_fn_flag = \
        "--csv_log_file_name=sweep_files/{}.csv".format(run_name)
    log_fn_flag = "--log_file_name=sweep_files/{}.log".format(run_name)
    result_file_flags = "{} {} {}".format(profile_fn_flag, csv_log_fn_flag,
                                          log_fn_flag)

    # MISCELLANY
    # Reducing the number of inflight frames reduces inference latency variance
    # when detection runs with an actual model. Otherwise it should be set as
    # high as possible to increase throughput.
    # The assumption is that actually running detection inference only happens
    # when caching the detection stage, i.e. seq-det-pol = infinite.
    # The number 10 is arbitrary and can be set higher, though it would further
    # increase contention. The number 2 is the minimum number of inflight
    # frames needed to make progress in infinite mode.
    num_inflight_frames = 10 if experiment_toggles[
        "use_cached_detection"] else 2
    misc_flags = f"--num_inflight_frames={num_inflight_frames}"

    flagfile_flag = "--flagfile=detection.conf"
    print("python run_offline_pipeline.py {} {} {} {} {} {} {} {}".format(
        flagfile_flag, dataset_flags, detection_config_flags,
        tracking_config_flags, detection_eval_flags, result_file_flags,
        save_frames_flags, misc_flags))


if __name__ == "__main__":
    # trials = specific_configs()
    trials = config_sweep(experiment_toggles)
    # trials = configs_from_dataframe(pd.read_pickle("condensed_space.pl"))
    for i, trial_configs in tqdm(enumerate(trials)):
        print(f"echo \">>>>>>>>>>>>>>>>>> {i} <<<<<<<<<<<<<<<<<<\"")
        logical_trial_config_to_command(trial_configs, experiment_toggles)
