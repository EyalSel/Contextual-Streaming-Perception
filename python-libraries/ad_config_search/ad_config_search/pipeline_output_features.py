import itertools
import pickle
from pathlib import Path

import jsonlines
import numpy as np

from ad_config_search.data_curation import get_fn_without_ext, split_run
from ad_config_search.utils import merge_dicts, unique_list


def num_bboxes_per_frame(frames):
    """
    Returns an array of size number of frames, with the total number of bboxes
    in each frame
    """
    return [len(f["obstacles"]) for f in frames]


def get_with_optional_key(d, k, k_reserve):
    if k in d:
        return d[k]
    else:
        return d[k_reserve]


def bbox_corners(bbox):
    xmn = get_with_optional_key(bbox, "xmin", "xmn")
    xmx = get_with_optional_key(bbox, "xmax", "xmx")
    ymn = get_with_optional_key(bbox, "ymin", "ymn")
    ymx = get_with_optional_key(bbox, "ymax", "ymx")
    return xmn, xmx, ymn, ymx


def to_sort_format(bbox):
    h = get_with_optional_key(bbox, "ymax", "ymx") - get_with_optional_key(
        bbox, "ymin", "ymn")
    w = get_with_optional_key(bbox, "xmax", "xmx") - get_with_optional_key(
        bbox, "xmin", "xmn")
    assert h > 0, h
    assert w > 0, w
    return {
        "c_x": get_with_optional_key(bbox, "xmin", "xmn") + w / 2,
        "c_y": get_with_optional_key(bbox, "ymin", "ymn") + h / 2,
        "ratio": w / h,
        "size": h * w
    }


def from_sort_format(bbox):
    assert bbox["size"] > 0, bbox["size"]
    assert bbox["ratio"] > 0, bbox["ratio"]
    w = np.sqrt(bbox["size"] * bbox["ratio"])
    h = bbox["size"] / w
    return {
        "xmin": bbox["c_x"] - w / 2,
        "xmax": bbox["c_x"] + w / 2,
        "ymin": bbox["c_y"] - h / 2,
        "ymax": bbox["c_y"] + h / 2
    }


def bbox_area(bbox):
    return (get_with_optional_key(bbox, "xmax", "xmx") - get_with_optional_key(
        bbox, "xmin", "xmn")) * (get_with_optional_key(bbox, "ymax", "ymx") -
                                 get_with_optional_key(bbox, "ymin", "ymn"))


# https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def bb_intersection_over_union(boxA, boxB):
    xmn_a, xmx_a, ymn_a, ymx_a = bbox_corners(boxA)
    xmn_b, xmx_b, ymn_b, ymx_b = bbox_corners(boxB)
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(xmn_a, xmn_b)
    yA = max(ymn_a, ymn_b)
    xB = min(xmx_a, xmx_b)
    yB = min(ymx_a, ymx_b)
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (xmx_a - xmn_a + 1) * (ymx_a - ymn_a + 1)
    boxBArea = (xmx_b - xmn_b + 1) * (ymx_b - ymn_b + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


# https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def bb_vertical_intersection_over_union(boxA, boxB):
    xmn_a, xmx_a, ymn_a, ymx_a = bbox_corners(boxA)
    xmn_b, xmx_b, ymn_b, ymx_b = bbox_corners(boxB)
    xmx_a = xmx_b
    xmn_a = xmn_b
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(xmn_a, xmn_b)
    yA = max(ymn_a, ymn_b)
    xB = min(xmx_a, xmx_b)
    yB = min(ymx_a, ymx_b)
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (xmx_a - xmn_a + 1) * (ymx_a - ymn_a + 1)
    boxBArea = (xmx_b - xmn_b + 1) * (ymx_b - ymn_b + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def bbox_iou_to_others_across_time(frames, frame_gap: int):
    assert frame_gap > 0, frame_gap
    instantenous_bbox_ious = []
    for start, end in zip(frames[:-frame_gap], frames[frame_gap:]):
        # verify there aren't multiple same ids in the frame
        start_ids = [o["id"] for o in start["obstacles"]]
        end_ids = [o["id"] for o in end["obstacles"]]
        assert len(set(start_ids)) == len(start_ids), \
            (len(set(start_ids)), len(start_ids))
        assert len(set(end_ids)) == len(end_ids), \
            (len(set(end_ids)), len(end_ids))
        start_id_dict = {o["id"]: o for o in start["obstacles"]}
        end_id_dict = {o["id"]: o for o in end["obstacles"]}
        dist_dict = {}

        for start_oid in start_id_dict:
            all_ious = [
                bb_intersection_over_union(start_id_dict[start_oid]["bbox"],
                                           end_id_dict[end_oid]["bbox"])
                for end_oid in end_id_dict if end_oid != start_oid
            ]
            dist_dict[start_oid] = 0 if len(all_ious) == 0 else np.max(
                all_ious)
        instantenous_bbox_ious.append(dist_dict)
    return instantenous_bbox_ious


def bbox_self_iou_across_time(frames, frame_gap: int):
    assert frame_gap > 0, frame_gap
    instantenous_bbox_ious = []
    for start, end in zip(frames[:-frame_gap], frames[frame_gap:]):
        # verify there aren't multiple same ids in the frame
        start_ids = [o["id"] for o in start["obstacles"]]
        end_ids = [o["id"] for o in end["obstacles"]]
        assert len(set(start_ids)) == len(start_ids), \
            (len(set(start_ids)), len(start_ids))
        assert len(set(end_ids)) == len(end_ids), \
            (len(set(end_ids)), len(end_ids))
        # convert from [{id, bbox}, ...] format to {id: bbox, ...} format
        start_id_dict = {o["id"]: o for o in start["obstacles"]}
        end_id_dict = {o["id"]: o for o in end["obstacles"]}
        shared_ids = set(start_ids).intersection(set(end_ids))
        dist_dict = {}

        for oid in shared_ids:
            dist_dict[oid] = bb_intersection_over_union(
                start_id_dict[oid]["bbox"], end_id_dict[oid]["bbox"])
        instantenous_bbox_ious.append(dist_dict)
    return instantenous_bbox_ious


def worst_trajectory_IOU(expected_bbox_trajectory, observed_bbox_trajectory):
    """
    expected_bbox_trajectory: list of bboxes over consecutive frames. Assumes
    all values are given (no Nones).
    observed_bbox_trajectory: list of bboxes over consecutive frames. None in
    the frames where the bbox no longer exists.
    """
    assert len(expected_bbox_trajectory) == len(observed_bbox_trajectory), (
        len(expected_bbox_trajectory), len(observed_bbox_trajectory))
    accumulated_IOUs = []
    for expected_bbox, observed_bbox in zip(expected_bbox_trajectory,
                                            observed_bbox_trajectory):
        if observed_bbox is None:
            return 0
        else:
            accumulated_IOUs.append(
                bb_intersection_over_union(expected_bbox, observed_bbox))
    return np.min(accumulated_IOUs)


def linearly_extrapolate_bbox_movement(bbox, v_x: float, v_y: float,
                                       v_size: float, steps: int):
    assert steps > 0, steps
    sort_format = to_sort_format(bbox)
    c_x, c_y, ratio, size = [
        sort_format[x] for x in ["c_x", "c_y", "ratio", "size"]
    ]

    bbox_trajectory_sort_format = [{
        "c_x":
        c_x + i * v_x,
        "c_y":
        c_y + i * v_y,
        "size":
        (size + max(i * v_size, -np.abs(v_size) *
                    (size // np.abs(v_size)), 1)) if v_size != 0 else size,
        "ratio":
        ratio
    } for i in range(steps)]
    return [from_sort_format(bbox) for bbox in bbox_trajectory_sort_format]


def infer_velocities(bbox_start, bbox_end, steps: int):
    assert steps > 0, steps
    bbox_start_sort_format = to_sort_format(bbox_start)
    bbox_end_sort_format = to_sort_format(bbox_end)
    c_x_0, c_y_0, size_0 = [
        bbox_start_sort_format[x] for x in ["c_x", "c_y", "size"]
    ]
    c_x_n, c_y_n, size_n = [
        bbox_end_sort_format[x] for x in ["c_x", "c_y", "size"]
    ]
    return {
        "v_x": (c_x_n - c_x_0) / steps,
        "v_y": (c_y_n - c_y_0) / steps,
        "v_size": (size_n - size_0) / steps
    }


def bbox_self_iou_across_time_linearly_projected(frames,
                                                 extrapolate_over_n: int):
    assert extrapolate_over_n > 1, extrapolate_over_n
    unique_ids_per_frame = [
        unique_list([o["id"] for o in frame["obstacles"]]) for frame in frames
    ]
    assert np.all(unique_ids_per_frame), unique_ids_per_frame
    frames = [{o["id"]: o for o in frame["obstacles"]} for frame in frames]
    from more_itertools import windowed
    all_worst_IOUs = []
    for window in windowed(frames, 2 * extrapolate_over_n):
        number_of_none_frames = np.sum([x is None for x in window])
        if number_of_none_frames > extrapolate_over_n - 2:
            continue
        if number_of_none_frames > 0:
            window = window[:-number_of_none_frames]
            number_of_none_frames = np.sum([x is None for x in window])
            assert number_of_none_frames == 0, number_of_none_frames
        frame_start = window[0]
        frame_end = window[extrapolate_over_n]
        shared_keys = set(frame_start.keys()).intersection(
            set(frame_end.keys()))
        expected_trajectories = {
            key: linearly_extrapolate_bbox_movement(
                bbox=frame_end[key]["bbox"],
                **infer_velocities(frame_start[key]["bbox"],
                                   frame_end[key]["bbox"],
                                   steps=extrapolate_over_n),
                steps=len(window) - extrapolate_over_n - 1)
            for key in shared_keys
        }
        observed_trajectories = {
            key: [
                frame.get(key, {"bbox": None})["bbox"]
                for frame in window[extrapolate_over_n + 1:]
            ]
            for key in shared_keys
        }
        worst_IOUs = [
            worst_trajectory_IOU(expected_trajectories[key],
                                 observed_trajectories[key])
            for key in shared_keys
        ]
        all_worst_IOUs.append(worst_IOUs)
    return np.concatenate(all_worst_IOUs) if len(all_worst_IOUs) > 0 else []


def bbox_vertical_self_iou_across_time(frames, frame_gap: int):
    assert frame_gap > 0, frame_gap
    instantenous_bbox_ious = []
    for start, end in zip(frames[:-frame_gap], frames[frame_gap:]):
        # verify there aren't multiple same ids in the frame
        start_ids = [o["id"] for o in start["obstacles"]]
        end_ids = [o["id"] for o in end["obstacles"]]
        assert len(set(start_ids)) == len(start_ids), \
            (len(set(start_ids)), len(start_ids))
        assert len(set(end_ids)) == len(end_ids), \
            (len(set(end_ids)), len(end_ids))
        # convert from [{id, bbox}, ...] format to {id: bbox, ...} format
        start_id_dict = {o["id"]: o for o in start["obstacles"]}
        end_id_dict = {o["id"]: o for o in end["obstacles"]}
        shared_ids = set(start_ids).intersection(set(end_ids))
        dist_dict = {}

        for oid in shared_ids:
            dist_dict[oid] = bb_vertical_intersection_over_union(
                start_id_dict[oid]["bbox"], end_id_dict[oid]["bbox"])
        instantenous_bbox_ious.append(dist_dict)
    return instantenous_bbox_ious


def bbox_speed_per_frame(frames):
    """
    returns array [{id: speed, ...}, ...] of size = (number of frames)-5
    where bbox speed is smoothed over 5 frames
    """
    instantenous_bbox_speed = []
    # average bbox frame speed 5 frames away (0.5s)
    for start, end in zip(frames[:-1], frames[1:]):
        # verify there aren't multiple same ids in the frame
        start_ids = [o["id"] for o in start["obstacles"]]
        end_ids = [o["id"] for o in end["obstacles"]]
        assert len(set(start_ids)) == len(start_ids), \
            (len(set(start_ids)), len(start_ids))
        assert len(set(end_ids)) == len(end_ids), \
            (len(set(end_ids)), len(end_ids))
        # convert from [{id, bbox}, ...] format to {id: bbox, ...} format
        start_id_dict = {o["id"]: o for o in start["obstacles"]}
        end_id_dict = {o["id"]: o for o in end["obstacles"]}
        shared_ids = set(start_ids).intersection(set(end_ids))
        dist_dict = {}

        def center(bbox):
            """ convert from top left anchor to center anchor """
            xmn, xmx, ymn, ymx = bbox_corners(bbox)
            return (
                xmn + (xmx - xmn) / 2,
                ymn + (ymx - ymn) / 2,
            )

        for oid in shared_ids:
            start_center_x, start_center_y = center(start_id_dict[oid]["bbox"])
            end_center_x, end_center_y = center(end_id_dict[oid]["bbox"])
            x_diff = end_center_x - start_center_x
            y_diff = end_center_y - start_center_y
            dist = np.sqrt(x_diff**2 + y_diff**2)
            dist_dict[oid] = dist
        instantenous_bbox_speed.append(dist_dict)
    return instantenous_bbox_speed


def bbox_sizes_per_frame(frames):
    """
    returns array [{id: size, ...}, ...] of size = number of frames
    """
    def size(bbox):
        xmn, xmx, ymn, ymx = bbox_corners(bbox)
        return (xmx - xmn) * (ymx - ymn)

    return [{o["id"]: size(o["bbox"])
             for o in frame["obstacles"]} for frame in frames]


def bboxes_longevity(frames):
    """
    Returns a list of bbox longevity in no particular order
    """
    ids_per_frame = [[o["id"] for o in frame["obstacles"]] for frame in frames]
    _, bbox_longevity = \
        np.unique(
            list(itertools.chain.from_iterable(ids_per_frame)),
            return_counts=True
        )
    return bbox_longevity


def bboxes_labels_per_frame(frames):
    """
    Returns the number of pedestrians and number of cars in each frame
    """
    all_dicts = [
        dict(
            zip(*np.unique([o["label"] for o in frame["obstacles"]],
                           return_counts=True)))
        if len(frame["obstacles"]) > 1 else {} for frame in frames
    ]
    return {
        f"num_label_{k}": np.mean([d.get(k, 0) for d in all_dicts])
        for k in ["vehicle", "pedestrian"]
    }


def nontemporal_per_frame_jsonl_bbox_info(frames):
    def id_dict_to_list(per_frame_id_dict):
        return [np.mean(list(d.values())) for d in per_frame_id_dict if d]

    assert len(frames) > 0, len(frames)
    result = {
        "num_bboxes":
        num_bboxes_per_frame(frames),
        "bbox_speed":
        id_dict_to_list(bbox_speed_per_frame(frames)),
        "bbox_self_iou_1frame":
        id_dict_to_list(bbox_self_iou_across_time(frames, frame_gap=1)),
        # "bbox_self_iou_2frame":
        # id_dict_to_list(bbox_self_iou_across_time(frames, frame_gap=2)),
        # "bbox_self_iou_3frame":
        # id_dict_to_list(bbox_self_iou_across_time(frames, frame_gap=3)),
        # "bbox_iou_to_others_1frame":
        # id_dict_to_list(bbox_iou_to_others_across_time(frames, frame_gap=1)),
        # "bbox_iou_to_others_2frame":
        # id_dict_to_list(bbox_iou_to_others_across_time(frames, frame_gap=2)),
        # "bbox_iou_to_others_3frame":
        # id_dict_to_list(bbox_iou_to_others_across_time(frames, frame_gap=3)),
        # "bbox_projected_self_iou_2frames":
        # bbox_self_iou_across_time_linearly_projected(frames, 2),
        # "bbox_projected_self_iou_3frames":
        # bbox_self_iou_across_time_linearly_projected(frames, 3),
        # "bbox_vertical_self_iou_1frame":
        # id_dict_to_list(bbox_vertical_self_iou_across_time(frames,
        #    frame_gap=1)),
        # "avg_bbox_size":
        # id_dict_to_list(bbox_sizes_per_frame(frames)),
        # longevity is across frames, so values are not per frame
        "bbox_longevity":
        bboxes_longevity(frames)
    }
    result.update(bboxes_labels_per_frame(frames))
    assert len(result["num_bboxes"]) == len(frames), \
        (len(result["num_bboxes"]), len(frames))
    return result


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_scenario_only_featurize(scenario_features_path, scenario_name):
    with open(scenario_features_path, 'rb') as f:
        scenario_only_features = pickle.load(f)
    index = scenario_only_features["scenarios"].index(scenario_name)
    vel = scenario_only_features["velocities"][index]
    context_info = scenario_only_features["context_infos"][index]
    return vel, context_info


def ego_vehicle_angle_change(vel_chunk, velocity_threshold=0.1):
    """
    This function works well if the chunks aren't very big.
     - Assumes vel_chuck is a [N x 3] numpy array, with columns v_x, v_y, v_z
    respectively. This function only focuses on x, y plane.
     - If there is no movement in the chunk then the function returns 0.
     - Otherwise, the function returns the difference in velocity angle
       between the first and last frames with nonzero speed, downscaled by the
       fraction of frames within this span (out of the chunk length).
    """
    assert len(vel_chunk) >= 1, "must have at least one frame velocity reading"
    vel_chunk = vel_chunk[:, :-1]  # getting rid of v_z axis
    vel_chunk[np.abs(vel_chunk) <
              velocity_threshold] = 0  # ignore really small velocities
    per_frame_is_car_moving = np.any(~np.isclose(vel_chunk, 0), axis=1)
    if not np.any(per_frame_is_car_moving):
        return 0
    # https://stackoverflow.com/a/16244044/1546071
    # argmax returns the first True it finds, or 0 if all items are False,
    # which is exactly what we want.
    first_frame_with_movement = np.argmax(per_frame_is_car_moving)
    last_frame_with_movement = len(vel_chunk) - np.argmax(
        per_frame_is_car_moving[::-1]) - 1
    assert first_frame_with_movement <= last_frame_with_movement
    vx_0, vy_0 = vel_chunk[first_frame_with_movement]
    vx_n, vy_n = vel_chunk[last_frame_with_movement]
    angle_0 = np.arctan2(vy_0, vx_0)
    angle_n = np.arctan2(vy_n, vx_n)
    # https://stackoverflow.com/a/7869457/1546071
    angle_diff = ((angle_n - angle_0) + np.pi) % (np.pi * 2) - np.pi
    # We only compare frames with nonzero speed, because when the vehicle is
    # still we can't infer its orientation. Therefore if there's just one
    # frame with nonzero speed we just ignore it and return 0.
    result = angle_diff * (last_frame_with_movement -
                           first_frame_with_movement + 1) / len(vel_chunk)
    # somewhat correcting outliers where the car swinging on its suspensions
    # causes the gyroscope velocity reads to make it look like the car is
    # moving in opposite directions in a very short period.
    result = np.clip(result, -0.5, 0.5)
    return result


def nontemporal_per_frame_scenario_only_features(scenario_features_path,
                                                 scenario_name, part_length_s):
    """
    Scenario features only attainable from the actual scenario file. Does not
    include any ground truth that wouldn't be available to the car.
    """
    vel, context_info = \
        get_scenario_only_featurize(scenario_features_path, scenario_name)
    result = []
    for i, vel_chunck in enumerate(chunks(vel, 10 * part_length_s)):
        ego_vehicle_speed = np.sqrt(vel_chunck[:, 0]**2 + vel_chunck[:, 1]**2 +
                                    vel_chunck[:, 2]**2)
        ego_vertical_movement = np.abs(vel_chunck[:, 2])
        ego_angle_change = ego_vehicle_angle_change(vel_chunck)
        features = {
            "avg_ego_speed": np.mean(ego_vehicle_speed),
            "90p_ego_speed": np.percentile(ego_vehicle_speed, 90),
            "10p_ego_speed": np.percentile(ego_vehicle_speed, 10),
            "ego_angle_change": ego_angle_change,
            "avg_ego_vertical_movement": np.mean(ego_vertical_movement),
            "90p_ego_vertical_movement": np.percentile(ego_vertical_movement,
                                                       90),
            "10p_ego_vertical_movement": np.percentile(ego_vertical_movement,
                                                       10),
            **context_info
        }
        result.append((i, features))
    return result


def aggregates_per_bbox_size_bin(frame_chunk):
    """
    Breaks the bboxes into size tiers, chosen emperically based on the
    detection power for efficient-det, then for each tier returns aggregate
    stats.
    """
    def filter_by_size(frames, size_low, size_high):
        import copy
        frames = copy.deepcopy(frames)
        for frame in frames:
            frame["obstacles"] = [
                o for o in frame["obstacles"]
                if bbox_area(o["bbox"]) >= size_low
                and bbox_area(o["bbox"]) < size_high
            ]
        return frames

    # values emperically selected basked on edet bbox detections on waymo.
    bins = [
        0,
        np.exp(6.5), 32**2,
        np.exp(7.3),
        np.exp(7.6),
        np.exp(7.85), np.inf
    ]
    bin_features_list = []
    index_start = 1
    for i, (bin_start, bin_end) in enumerate(
            zip(bins[index_start:-1], bins[index_start + 1:])):
        chunk = filter_by_size(frame_chunk, bin_start, bin_end)
        features = {
            f"bin_{i}-{k}": v
            for k, v in aggregate_bbox_stats(chunk).items()
        }
        bin_features_list.append(features)
    return merge_dicts(bin_features_list)


def aggregate_bbox_stats(frame_chunk):
    bbox_info = nontemporal_per_frame_jsonl_bbox_info(frame_chunk)
    if np.max(bbox_info["num_bboxes"]) != 0:
        features_part_1 = {
            "avg_bbox_longevity": np.mean(bbox_info["bbox_longevity"]),
            "90p_bbox_longevity": np.percentile(bbox_info["bbox_longevity"],
                                                90),
            "10p_bbox_longevity": np.percentile(bbox_info["bbox_longevity"],
                                                10),
            "avg_num_bboxes": np.mean(bbox_info["num_bboxes"]),
            "90p_num_bboxes": np.percentile(bbox_info["num_bboxes"], 90),
            "10p_num_bboxes": np.percentile(bbox_info["num_bboxes"], 10),
            # "avg_bbox_size": np.mean(bbox_info["avg_bbox_size"]),
            # "90p_bbox_size": np.percentile(bbox_info["avg_bbox_size"], 90),
            # "10p_bbox_size": np.percentile(bbox_info["avg_bbox_size"], 10),
        }
    else:
        features_part_1 = {
            "avg_bbox_longevity": 0,
            "90p_bbox_longevity": 0,
            "10p_bbox_longevity": 0,
            "avg_num_bboxes": 0,
            "90p_num_bboxes": 0,
            "10p_num_bboxes": 0,
            # "avg_bbox_size": 0,
            # "90p_bbox_size": 0,
            # "10p_bbox_size": 0,
        }

    # bbox speed may have no values even if there are bboxes
    # because bbox speed is computed over 5 frames. bboxes that have
    # shorter longevity than that are not included.
    statistics = [
        ("avg_", np.mean),
        ("90p_", lambda x: np.percentile(x, 90)),
        ("10p_", lambda x: np.percentile(x, 10)),
    ]
    more_features = [
        {(prefix + feature_name):
         fn(bbox_info[feature_name]) if len(bbox_info[feature_name]) else 0
         for prefix, fn in statistics}
        for feature_name in [
            "bbox_vertical_self_iou_1frame",
            "bbox_speed",
            "bbox_self_iou_1frame",
            "bbox_self_iou_2frame",
            "bbox_self_iou_3frame",
            # "bbox_iou_to_others_1frame", "bbox_iou_to_others_2frame",
            # "bbox_iou_to_others_3frame",
            # "bbox_projected_self_iou_2frames",
            # "bbox_projected_self_iou_3frames"
        ] if feature_name in bbox_info
    ]
    features = {
        **features_part_1,
        **merge_dicts(more_features),
        **{k: v
           for k, v in bbox_info.items() if k.startswith("num_label_")}
    }
    return features


def memoryless_featurize(jsonl_path, part_length_s,
                         features_from_chunk_fn_lst):
    """
    features_from_chunk_fn_lst is a list of functions that take just one
    frame chunk at a time.

    Returns a list of (i, features) where i is the index of the scenario
    section in the video and features is the feature dicts for that section,
    combined from the results of all the functions.
    Skips sections with no bounding boxes.
    """
    assert 20 % part_length_s == 0, \
        ("20s (the time length of each scenario in waymo) is not divisible by "
         "{}").format(part_length_s)
    if type(features_from_chunk_fn_lst) != list:
        features_from_chunk_fn_lst = [features_from_chunk_fn_lst]
    with jsonlines.open(jsonl_path) as reader:
        frames = list(reader)
    frames = frames[1:]  # getting rid of first metadata element
    result = []
    for i, frame_chunk in enumerate(chunks(frames, 10 * part_length_s)):
        all_feats = [fn(frame_chunk) for fn in features_from_chunk_fn_lst]
        features = merge_dicts(
            all_feats) if len(all_feats) > 1 else all_feats[0]
        result.append((i, features))
    return result


def get_final_features(jsonl_path, part_length_s):
    """
    Returns a row as a dict with all the information in a C' x E
    """
    run_name = split_run(
        get_fn_without_ext(jsonl_path).replace("__detection_cache", ""))["run"]
    # turns list of (key, value) tuples to dicts
    jsonl_features = dict(
        memoryless_featurize(jsonl_path, part_length_s, aggregate_bbox_stats))
    scenario_only_features = dict(
        nontemporal_per_frame_scenario_only_features(
            "../data/scenario_only_features.pq", run_name, part_length_s))
    csv_files = list(jsonl_path.parent.glob("*.csv"))
    assert len(csv_files) == 1, csv_files
    csv_file = csv_files[0]
    rows = []
    for i in sorted(list(jsonl_features.keys())):
        slice_name = run_name + "-P{}_{}".format(i, 20 // part_length_s)
        row = {
            **jsonl_features[i],  # E from env
            **scenario_only_features[i],  # E from pipeline
            "scenario_name": slice_name,
            **{
                f"pastC-{k}": v
                for k, v in split_run(get_fn_without_ext(csv_file)).items()
            }  # C'
        }
        rows.append(row)
    return rows


def get_gt_features(jsonl_path,
                    part_length_s,
                    scenario_only_feature_path=None):
    jsonl_path = Path(jsonl_path)
    run_name = jsonl_path.parent.stem + "-" + jsonl_path.stem
    # turns list of (key, value) tuples to dicts
    jsonl_features = dict(
        memoryless_featurize(jsonl_path, part_length_s,
                             aggregates_per_bbox_size_bin))
    scenario_only_feature_path = (scenario_only_feature_path
                                  or "../data/scenario_only_features.pq")
    scenario_only_features = dict(
        nontemporal_per_frame_scenario_only_features(
            scenario_only_feature_path, run_name, part_length_s))
    rows = []
    for i in sorted(list(jsonl_features.keys())):
        slice_name = run_name + "-P{}_{}".format(i, 20 // part_length_s)
        row = {
            **jsonl_features[i],  # E from env
            **scenario_only_features[i],  # E from ground truth
            "scenario_name":
            slice_name,
        }
        rows.append(row)
    return rows
