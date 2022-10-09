import pickle
from pathlib import Path

import imageio
import jsonlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from tqdm import tqdm

# Moved inside functions to allow erdos_env (using tf1.5) to import
# import tensorflow.compat.v1 as tf
# from waymo_open_dataset import dataset_pb2 as open_dataset


def frames_from_path(f_path, silent=True):
    """
    Returns list of frames from tfrecord path
    """
    import tensorflow.compat.v1 as tf
    from waymo_open_dataset import dataset_pb2 as open_dataset
    assert Path(f_path).suffix == ".tfrecord", f_path
    dataset = tf.data.TFRecordDataset(str(f_path), compression_type='')
    frames = []
    with_tqdm = tqdm(dataset) if not silent else dataset
    for data in with_tqdm:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        frames.append(frame)
    return frames


def get_frame_dict(frame, id_map, rbg=True):
    """
    uses gbr if rbg=False
    returns
    {
        CAMERA_POSITION: {
            "rbg_img": uint8 H, W, C image
            obstacles: [
                {"bbox": [xmin,xmax,ymin,ymax], "label": str, "id": int},
                ...
            ]
        }
    }
    """
    import tensorflow.compat.v1 as tf
    from waymo_open_dataset import dataset_pb2 as open_dataset
    result = {}
    for camera_image, frame_labels in zip(frame.images, frame.camera_labels):
        assert camera_image.name == frame_labels.name
        camera_name = open_dataset.CameraName.Name.Name(camera_image.name)
        camera_dict = {}

        def extract(label):
            given_labels = {
                label.TYPE_CYCLIST: "cyclist",
                label.TYPE_PEDESTRIAN: "pedestrian",
                label.TYPE_SIGN: "sign",
                label.TYPE_UNKNOWN: "unknown",
                label.TYPE_VEHICLE: "vehicle"
            }
            xmin = label.box.center_x - 0.5 * label.box.length
            ymin = label.box.center_y - 0.5 * label.box.width
            xmax = xmin + label.box.length
            ymax = ymin + label.box.width
            str_label = given_labels[label.type]
            id = id_map[label.id]
            return {
                "bbox": [xmin, xmax, ymin, ymax],
                "label": str_label,
                "id": id
            }

        img = tf.image.decode_jpeg(camera_image.image).numpy()  # rbg
        if rbg:
            camera_dict["img"] = img
        else:
            camera_dict["img"] = img[:, :, ::-1]

        camera_dict["obstacles"] = \
            [extract(label) for label in frame_labels.labels]
        result[camera_name] = camera_dict
    return result


def pl_to_jsonl_schema(pl_path, frame_gap=None):
    pl_path = Path(pl_path)
    jsonl_dest = pl_path.parent / (pl_path.stem.replace("_", "") + ".jsonl")
    if jsonl_dest.exists():
        print("the jsonl of {} exists already!".format(pl_path))
        return
    with open(pl_path, 'rb') as f:
        frames = pickle.load(f)

    def list_to_dict(bbox):
        xmin, xmax, ymin, ymax = bbox
        return {"xmn": xmin, "xmx": xmax, "ymn": ymin, "ymx": ymax}

    def prep(d, ts):
        obstacles = [
            {
                **o,  # bbox, label, id
                "bbox": list_to_dict(o["bbox"]),
                "confidence": 1.0
            } for o in d["obstacles"]
        ]
        return {
            "content_type": "ground_truth",
            "runtime": 0,
            "obstacles": obstacles,
            "timestamp": ts,
        }

    frame_dicts = [prep(d, i * frame_gap) for i, d in enumerate(frames)]

    metadata = {
        "content_type": "gt_metadata",
    }

    # Write the bounding boxes.
    with jsonlines.open(jsonl_dest, mode='a') as writer:
        writer.write(metadata)
        for frame_dict in frame_dicts:
            writer.write(frame_dict)


def tfrecord_to_jsonl_schema(tfrecord_path, just_front=True):
    tfrecord_path = Path(tfrecord_path)
    new_fn = tfrecord_path.parent.stem + "-" + tfrecord_path.stem.replace(
        "_", "") + ".jsonl"
    jsonl_dest = tfrecord_path.parent / new_fn
    if jsonl_dest.exists():
        print("the jsonl of {} exists already!".format(tfrecord_path))
        return
    frames = frames_from_path(tfrecord_path)
    id_map = simplify_ids(frames)

    def list_to_dict(bbox):
        xmin, xmax, ymin, ymax = bbox
        return {"xmn": xmin, "xmx": xmax, "ymn": ymin, "ymx": ymax}

    def prep(d, ts):
        obstacles = [
            {
                **o,  # bbox, label, id
                "bbox": list_to_dict(o["bbox"]),
                "confidence": 1.0
            } for o in d["FRONT"]["obstacles"]
        ]
        return {
            "content_type": "ground_truth",
            "runtime": 0,
            "obstacles": obstacles,
            "timestamp": ts,
        }

    frame_dicts = [
        prep(get_frame_dict(f, id_map, rbg=False), i * 100)
        for i, f in enumerate(frames)
    ]

    metadata = {
        "content_type": "gt_metadata",
    }

    # Write the bounding boxes.
    with jsonlines.open(jsonl_dest, mode='a') as writer:
        writer.write(metadata)
        for frame_dict in frame_dicts:
            writer.write(frame_dict)


def waymo_tfrecord_to_pl(tfrecord_path):
    tfrecord_path = Path(tfrecord_path)
    pl_dest = tfrecord_path.parent / (tfrecord_path.stem.replace("_", "") +
                                      ".pl")
    if pl_dest.exists():
        print("the pl of {} exists already!".format(tfrecord_path))
        return
    frames = frames_from_path(tfrecord_path)
    id_map = simplify_ids(frames)
    dicts = [get_frame_dict(f, id_map, rbg=False)["FRONT"] for f in frames]
    for d in dicts:
        d["center_camera_feed"] = d["img"]
        del d['img']
    with open(pl_dest, 'wb') as f:
        pickle.dump(dicts, f)


def draw_frame(frame,
               id_map,
               ignore_bbox_below=None,
               plot_label=False,
               all_cameras=False,
               show_ground_truth=True):
    """
    Plots all cameras on frame, including labels. if ignore_bbox_below is not
    None it doesn't draw bbox labels under the given area.
    """
    import tensorflow.compat.v1 as tf
    from waymo_open_dataset import dataset_pb2 as open_dataset
    fig = plt.figure(figsize=((25, 20) if all_cameras else None))
    for i, (camera_image,
            frame_labels) in enumerate(zip(frame.images, frame.camera_labels)):
        camera_name = open_dataset.CameraName.Name.Name(camera_image.name)
        if not all_cameras and camera_name != "FRONT":
            continue
        ax = plt.subplot(*([3, 3, i + 1] if all_cameras else [1, 1, 1]))
        assert camera_image.name == frame_labels.name

        # Show the camera image.
        ax.imshow(tf.image.decode_jpeg(camera_image.image), cmap=None)
        ax.set_title(camera_name)
        ax.grid(False)
        ax.axis('off')

        if not show_ground_truth:
            continue

        for label in frame_labels.labels:
            given_labels = {
                label.TYPE_CYCLIST: "cyclist",
                label.TYPE_PEDESTRIAN: "pedestrian",
                label.TYPE_SIGN: "sign",
                label.TYPE_UNKNOWN: "unknown",
                label.TYPE_VEHICLE: "vehicle"
            }
            xmin = label.box.center_x - 0.5 * label.box.length
            ymin = label.box.center_y - 0.5 * label.box.width
            if ignore_bbox_below and \
               label.box.length * label.box.width < ignore_bbox_below:
                continue
            bbox_area = label.box.length * label.box.width
            if bbox_area > 96**2:
                color = "red"
            elif bbox_area > 32**2:
                color = "orange"
            else:
                color = "green"
            ax.add_patch(
                patches.Rectangle(xy=(xmin, ymin),
                                  width=label.box.length,
                                  height=label.box.width,
                                  linewidth=1,
                                  edgecolor=color,
                                  facecolor='none'))
            if plot_label:
                ax.text(xmin,
                        ymin,
                        "{}, {}".format(given_labels[label.type],
                                        id_map[label.id]),
                        horizontalalignment="left",
                        verticalalignment="top",
                        color='red')
    return fig


def simplify_raw_dataset_fnames(path, ext="tfrecord", dry_run=True):
    """
    To be used in scripts to convert files in a directory from a raw hash name
    to a S#.ext
    """
    names = sorted([
        p.stem for p in Path(path).iterdir()
        if ("." in p.name and p.name.split(".")[1] == ext)
    ])
    name_map = {x: "S_{}".format(i) for i, x in enumerate(names)}
    if not dry_run:
        return name_map
    for old_name, new_name in name_map.items():
        (Path(path)/(old_name+f".{ext}")).\
            rename(Path(path)/(new_name+f".{ext}"))


def collect_bboxes(frames, just_front=True):
    """
    Returns list of bbox areas (w*h) for all frames
    """
    id_map = simplify_ids(frames)
    labels_per_frame = []
    from waymo_open_dataset import dataset_pb2 as open_dataset
    for frame in frames:
        labels_this_frame = {}
        for (camera_image, frame_labels) in (zip(frame.images,
                                                 frame.camera_labels)):
            camera_name = open_dataset.CameraName.Name.Name(camera_image.name)
            if just_front and camera_name != "FRONT":
                continue
            for label in frame_labels.labels:
                labels_this_frame[id_map[label.id]] = label
        labels_per_frame.append(labels_this_frame)
    return labels_per_frame


def get_velocity(frames):
    return np.array([[
        f.images[0].velocity.v_x, f.images[0].velocity.v_y,
        f.images[0].velocity.v_z
    ] for f in frames])


def get_context_info(frames):
    return {
        "time_of_day": frames[0].context.stats.time_of_day,
        "weather": frames[0].context.stats.weather,
        "location": frames[0].context.stats.location
    }


def collect_bbox_info_per_frame(frames):
    bboxes_per_frame = collect_bboxes(frames, just_front=True)
    num_bboxes_per_frame = \
        [len(frame_bboxes.keys()) for frame_bboxes in bboxes_per_frame]
    # average bbox frame speed 5 frames away (0.5s)
    instantenous_bbox_speed = []
    for start, end in zip(bboxes_per_frame[:-5], bboxes_per_frame[5:]):
        shared_keys = set(start.keys()).intersection(set(end.keys()))
        dist_dict = {}
        for k in shared_keys:
            x_diff = start[k].box.center_x - end[k].box.center_x
            y_diff = start[k].box.center_y - end[k].box.center_y
            dist = np.sqrt(x_diff**2 + y_diff**2)
            dist_dict[k] = dist
        instantenous_bbox_speed.append(dist_dict)
    # average bbox size
    avg_bbox_size_per_frame = [
        np.mean([
            label.box.length * label.box.width
            for label in frame_bboxes.values()
        ]) if len(frame_bboxes) > 0 else 0 for frame_bboxes in bboxes_per_frame
    ]
    # longevity
    import itertools
    ids_per_frame = [frame.keys() for frame in bboxes_per_frame]
    _, bbox_longevity = \
        np.unique(
            list(itertools.chain.from_iterable(ids_per_frame)),
            return_counts=True
        )
    return {
        "num_bboxes": num_bboxes_per_frame,
        "bbox_speed": instantenous_bbox_speed,
        "avg_bbox_size": avg_bbox_size_per_frame,
        "bbox_longevity": bbox_longevity
    }


def simplify_ids(frames):
    """
    Returns a dictionary that maps unique id hashes in waymo format to
    unique integers
    """
    unique_ids = set()
    for frame in frames:
        for frame_labels in frame.camera_labels:
            for label in frame_labels.labels:
                unique_ids.add(label.id)
    unique_ids = sorted(list(unique_ids))  # to make the id map deterministic
    return {unique_id: index for index, unique_id in enumerate(unique_ids)}


def img_from_fig(frame,
                 id_map,
                 ignore_bbox_below=None,
                 all_cameras=False,
                 show_ground_truth=True):
    """
    Used to save a gif, see below
    """
    fig = draw_frame(frame,
                     id_map,
                     ignore_bbox_below=ignore_bbox_below,
                     all_cameras=all_cameras,
                     show_ground_truth=show_ground_truth)
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    plt.close(fig)
    return image


def save_frames_to_gif(frames,
                       fname,
                       all_cameras=False,
                       show_ground_truth=True,
                       fps=10,
                       silent=False):
    """
    fname shouldn't include extension
    """
    id_map = simplify_ids(frames)
    with_tqdm = tqdm(frames) if not silent else frames
    images = [
        img_from_fig(frame,
                     id_map,
                     all_cameras=all_cameras,
                     show_ground_truth=show_ground_truth)
        for frame in with_tqdm
    ]
    imageio.mimsave('{}.gif'.format(fname), images, fps=fps)


def sync_to_timely_jsonl(jsonl_contents):
    import numpy as np

    metadata = jsonl_contents[0]
    data = jsonl_contents[1:]

    frame_gap = 100
    # verify jsonl
    timestamps = [x["timestamp"] for x in data]
    assert timestamps[0] == frame_gap
    assert np.all(np.diff(timestamps) == frame_gap)

    timestamp_obstacles_dict = {x["timestamp"]: x["obstacles"] for x in data}
    timestamp_runtime_list = [(x["timestamp"], x["runtime"]) for x in data]
    last_frame = timestamp_runtime_list[-1][0]
    final_tuples = {}

    import math

    def roundup(x):
        return int(math.ceil(x / float(frame_gap))) * frame_gap

    for timestamp, runtime in timestamp_runtime_list:
        target = roundup(timestamp + runtime)
        if timestamp in timestamp_obstacles_dict:
            final_tuples[target] = timestamp_obstacles_dict[timestamp]

    last_res = []
    for i in range(frame_gap, last_frame + frame_gap, frame_gap):
        if i not in final_tuples.keys():
            final_tuples[i] = last_res
        else:
            last_res = final_tuples[i]

    def make_entry(ts, obstacles):
        return {
            "content_type": "frame_prediction",
            "runtime": 0,
            "timestamp": ts,
            "obstacles": obstacles
        }

    final_data = [metadata] + [
        make_entry(timestamp, final_tuples[timestamp])
        for timestamp, _ in timestamp_runtime_list
    ]
    return final_data


def visualize_inference(run_path,
                        replayer_class,
                        det_cache_paths,
                        fps,
                        frame_range=None,
                        ground_boxes=True,
                        sync_to_timely=False,
                        confidence_threshold=0):
    """
    Not specific to Waymo, so should be placed in a separate location at
    some point.

    run_path: path to pickle file with scenario frames and ground truth
    replayer_class: Class that implements interface in dataset_replay.py
    det_cache_paths: Location of predictions jsonl file(s)
    fps: fps to run gif at, should match dataset fps
    frame_range: a list with frame indices to make into gif. Leave as None to
                 use all frames in scenario
    ground_boxes: toggle to show ground truth boxes
    sync_to_timely: toggle to interpret the prediction bounding boxes as sync
                    mode, therefore translating them to timely predictions
                    using the associated prediction latency
    confidence_threshold: omit predictions with confidence under given
                          threshold.
    """
    import jsonlines
    offline_replayer = replayer_class(run_path)
    if type(det_cache_paths) is not list:
        det_cache_paths = [det_cache_paths]
    det_cache_datas = []
    for det_cache_path in det_cache_paths:
        with jsonlines.open(det_cache_path) as reader:
            det_cache_data = list(reader)
        assert det_cache_data[0]["content_type"] == "run_metadata", \
            det_cache_path
        if sync_to_timely:
            det_cache_data = sync_to_timely_jsonl(det_cache_data)
            print(det_cache_data)
        det_cache_data.pop(0)
        det_cache_datas.append(det_cache_data)
    prediction_lengths = [len(lst) for lst in det_cache_datas]
    print(offline_replayer.total_num_frames(), prediction_lengths)
    length_used = min(
        offline_replayer.total_num_frames(),
        min(prediction_lengths, default=offline_replayer.total_num_frames()))
    images = []
    frame_range = frame_range or range(length_used)
    for i in tqdm(frame_range):
        txt_alignments = [("top", "left"), ("top", "right"),
                          ("bottom", "right"), ("bottom", "left")]
        colors = ["red", "orange", "blue", "green"]
        run_frame = offline_replayer.get_frame(i)
        fig = plt.figure(figsize=(30, 22.5))
        ax = plt.subplot(*[1, 1, 1])
        if ground_boxes:
            for obs in run_frame["obstacles"]:
                xmn, xmx, ymn, ymx = obs["bbox"]
            ax.add_patch(
                patches.Rectangle(xy=(xmn, ymn),
                                  width=(xmx - xmn),
                                  height=(ymx - ymn),
                                  linewidth=1,
                                  edgecolor=colors[0],
                                  facecolor='none'))
            v_align, h_align = txt_alignments[0]
        import matplotlib.patches as mpatches
        label_patches = []
        label_patches.append(
            mpatches.Patch(color=colors[0], label='ground truth'))
        for j, det_cache_data in enumerate(det_cache_datas):
            label_patches.append(
                mpatches.Patch(color=colors[j + 1], label=f'{j}'))
            entry = det_cache_data[i]
            assert entry["content_type"] == "frame_prediction"
            for obs in entry["obstacles"]:
                # won't work for tracker feed, because it doesn't preserve
                # confidence
                if obs["confidence"] >= confidence_threshold:
                    ax.add_patch(
                        patches.Rectangle(
                            xy=(obs["bbox"]["xmn"], obs["bbox"]["ymn"]),
                            width=(obs["bbox"]["xmx"] - obs["bbox"]["xmn"]),
                            height=(obs["bbox"]["ymx"] - obs["bbox"]["ymn"]),
                            linewidth=1,
                            edgecolor=colors[j + 1],
                            facecolor='none'))
                    if obs["id"] != -1:
                        v_align, h_align = txt_alignments[j + 1]
                        ax.text(obs["bbox"]["xmn"],
                                obs["bbox"]["ymn"],
                                "{}".format(obs["id"]),
                                horizontalalignment=h_align,
                                verticalalignment=v_align,
                                color=colors[j + 1],
                                fontsize=12)
        # BGR to RGB
        ax.imshow(run_frame["center_camera_feed"][:, :, ::-1], cmap=None)
        ax.grid(False)
        ax.axis('off')
        ax.legend(handles=label_patches)
        fig.canvas.draw()  # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
        plt.close(fig)
        images.append(image)
    imageio.mimsave('{}.gif'.format(Path(run_path).stem), images, fps=fps)
