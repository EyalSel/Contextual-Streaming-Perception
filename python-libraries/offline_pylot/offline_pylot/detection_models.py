from pathlib import Path

homedir = Path.home()  # in erdos3
# homedir = Path("/home/erdos/workspace")  # inside pylot container

pylot_label_map = homedir / "pylot/dependencies/models/pylot.names"
coco_label_map = homedir / "pylot/dependencies/models/coco.names"
waymo_label_map = homedir / "AD-config-search/dependencies/waymo.names"
argoverse_jpg_label_map = (homedir /
                           "AD-config-search/dependencies/argoverse-jpg.names")
argoverse_cuboid_label_map = (
    homedir / "AD-config-search/dependencies/argoverse-cuboid.names")

base_det_path = homedir / "pylot/dependencies/models/obstacle_detection"
base_track_path = homedir / "pylot/dependencies/models/tracking"

based_cached_data_path = Path("../data/")


class DetectionModel:
    def __init__(self, name, path, label_map):
        self.name = name
        self.path = path
        self.label_map = label_map

    def __str__(self):
        return f"DetectionModel({self.name})"


def edet_info_from_n(n):
    edet_path = base_det_path / "efficientdet"
    name = "efficientdet-d{}".format(n)
    return name, DetectionModel(name,
                                edet_path / name / "{}_frozen.pb".format(name),
                                coco_label_map)


edet_models = {
    k: v
    for k, v in (edet_info_from_n(i) for i in list(range(8)) + ["7x"])
}
pylot_det_models = {
    n:
    DetectionModel(n, base_det_path / "{}/frozen_inference_graph.pb".format(n),
                   pylot_label_map)
    for n in ["faster-rcnn", "ssdlite-mobilenet-v2"]
}

detection_model_dict = {**pylot_det_models, **edet_models}


class TrackerModel:
    def __init__(self, name, path_flag_dict, ignores_detection=False):
        self.name = name
        self.path_flag_dict = path_flag_dict
        self.ignores_detection = ignores_detection

    def __str__(self):
        return f"TrackerModel({self.name})"


tracker_model_dict = {
    "da_siam_rpn_VOT":
    TrackerModel("da_siam_rpn", {
        "da_siam_rpn_model_path":
        base_track_path / "DASiamRPN/SiamRPNVOT.model"
    }),
    "da_siam_rpn_OTB":
    TrackerModel("da_siam_rpn", {
        "da_siam_rpn_model_path":
        base_track_path / "DASiamRPN/SiamRPNOTB.model"
    }),
    "da_siam_rpn_BIG":
    TrackerModel("da_siam_rpn", {
        "da_siam_rpn_model_path":
        base_track_path / "DASiamRPN/SiamRPNBIG.model"
    }),
    "deep_sort":
    TrackerModel(
        "deep_sort", {
            "deep_sort_tracker_weights_path":
            base_track_path / "deep-sort/mars-small128.pb"
        }),
    "trained_deep_sort":
    TrackerModel(
        "deep_sort", {
            "deep_sort_tracker_weights_path":
            base_track_path / "deep-sort-trained/best_model_state-dict"
        }),
    "sort":
    TrackerModel("sort", None),
    "qd_track":
    TrackerModel(
        "qd_track",
        {
            "qd_track_model_path":
            homedir /
            "pylot/dependencies/models/tracking/qd_track/qdtrack-frcnn_r50_fpn_12e_bdd100k-13328aed.pth",  # noqa: E501
            "qd_track_config_path":
            homedir /
            "pylot/dependencies/qdtrack/configs/qdtrack-frcnn_r50_fpn_12e_bdd100k.py"  # noqa: E501
        },
        True)
}


class Run:
    def __init__(self,
                 label_map,
                 dataset,
                 path,
                 sequence_name,
                 frame_interval_ms,
                 resolution,
                 cached_detection_path=None):
        self.label_map = label_map
        self.path = path
        self.dataset = dataset
        self.sequence_name = sequence_name
        self.frame_interval_ms = frame_interval_ms
        self.cached_detection_path = cached_detection_path
        self.resolution = resolution

    def __str__(self):
        return f"Run({self.dataset}, {self.sequence_name})"


def offline_carla_run(path):
    sequence_name = "{}-{}".format(
        Path(path).parent.parent.stem,
        Path(path).parent.stem)
    return Run(pylot_label_map, "carla_v1", path, sequence_name, 1000,
               (1080, 1920))


carla_datasets = {
    "offline_carla": [
        offline_carla_run(p) for p in filter(
            lambda p: p.is_dir() and p.stem == "ClearNoon" and any(p.iterdir()
                                                                   ),
            Path("/data/ges/faster-rcnn-driving/training_data").rglob("*"))
    ]
}


def offline_waymo_run(path, dataset_name, cached_detection_path):
    sequence_name = Path(path).parent.stem + "-" + Path(path).stem
    return Run(waymo_label_map, dataset_name, path, sequence_name, 100,
               (1280, 1920), cached_detection_path)


def offline_argoverse_jpg_run(path, dataset_name, cached_detection_path):
    sequence_name = Path(path).parent.parent.stem + "-" + path.parent.stem
    return Run(argoverse_jpg_label_map, dataset_name, path, sequence_name, 33,
               (1200, 1920), cached_detection_path)


def offline_argoverse_cuboid_run(path, dataset_name, cached_detection_path):
    sequence_name = Path(path).stem
    return Run(argoverse_cuboid_label_map, dataset_name, path, sequence_name,
               33, (1200, 1920), cached_detection_path)


datasets_dir = Path("/data/ges")  # in erdos3
# datasets_dir = Path("/home/erdos/datasets")  # inside pylot container

waymo_0000_files = list(
    (datasets_dir / "waymo/training_0000").rglob("*.tfrecord"))
argoverse_jpg_train_1 = list(
    (datasets_dir / "argoverse-tracking/train1").glob("*/ring_front_center"))
argoverse_jpg_train_2 = list(
    (datasets_dir / "argoverse-tracking/train2").glob("*/ring_front_center"))
argoverse_cuboid_train_1 = list(
    (datasets_dir /
     "argo_cuboids/cuboid_argo1_2d_amodal_labels_100fr").glob("*.pl"))
argoverse_cuboid_train_2 = list(
    (datasets_dir /
     "argo_cuboids/cuboid_argo2_2d_amodal_labels_100fr").glob("*.pl"))
argoverse_cuboid_train_3 = list(
    (datasets_dir /
     "argo_cuboids/cuboid_argo3_2d_amodal_labels_100fr").glob("*.pl"))
argoverse_cuboid_train_4 = list(
    (datasets_dir /
     "argo_cuboids/cuboid_argo4_2d_amodal_labels_100fr").glob("*.pl"))
argoverse_cuboid_val = list(
    (datasets_dir /
     "argo_cuboids/cuboid_argoval_2d_amodal_labels_100fr/").glob("*.pl"))

waymo_cherrypicked_scenarios = {(k or "waymo_T{}_S{}".format(t, s)): [
    offline_waymo_run(
        datasets_dir / "waymo/training_000{}/S{}.pl".format(t, s),
        "waymo_000{}_pl".format(t), based_cached_data_path /
        "cached_waymo_train_000{}_detection_v100{}".format(
            t, "_v2" if t == 0 else ""))
]
                                for k, t, s in [
                                    ("waymo_desolate_scene", 5, 20),
                                    ("waymo_golden_gate_turning_left", 0, 17),
                                    (None, 4, 23),
                                    (None, 4, 0),
                                    (None, 4, 24),
                                ]}

waymo_datasets = {
    dset_name: [
        offline_waymo_run(f, dset_name, based_cached_data_path / cache_path)
        for f in files
    ]
    for dset_name, files, cache_path in
    [("waymo_0000", waymo_0000_files,
      "cached_waymo_train_0000_detection_v100")] + [(
          f"waymo_{str(i).zfill(4)}_pl",
          list((datasets_dir /
                f"waymo/training_{str(i).zfill(4)}").rglob("*.pl")),
          f"cached_waymo_train_{str(i).zfill(4)}_detection_v100",
      ) for i in range(0, 32)] + [(
          f"waymo_validation_{str(i).zfill(4)}_pl",
          list((datasets_dir /
                f"waymo/validation_{str(i).zfill(4)}").rglob("*.pl")),
          f"cached_waymo_validation_{str(i).zfill(4)}_detection_v100",
      ) for i in range(0, 7)]
}

argoverse_jpg_datasets = {
    dset_name: [
        offline_argoverse_jpg_run(f, dset_name,
                                  based_cached_data_path / cache_path)
        for f in files
    ]
    for dset_name, files, cache_path in [(
        "argo-jpg_train_1", argoverse_jpg_train_1,
        "cached_argo_train_1_detection_v100"),
                                         ("argo-jpg_train_2",
                                          argoverse_jpg_train_2,
                                          "argo_train_2_detection_sweep_v100")]
}

argoverse_cuboid_datasets = {
    dset_name: [
        offline_argoverse_cuboid_run(f, dset_name,
                                     based_cached_data_path / cache_path)
        for f in files
    ]
    for dset_name, files, cache_path in
    [("argo-cuboid_train_1", argoverse_cuboid_train_1,
      "cached_argo_cuboid_train_1_detection_v100"),
     ("argo-cuboid_train_2", argoverse_cuboid_train_2,
      "cached_argo_cuboid_train_2_detection_v100"),
     ("argo-cuboid_train_3", argoverse_cuboid_train_3,
      "cached_argo_cuboid_train_3_detection_v100"),
     ("argo-cuboid_train_4", argoverse_cuboid_train_4,
      "cached_argo_cuboid_train_4_detection_v100"),
     ("argo-cuboid_val", argoverse_cuboid_val,
      "cached_argo_cuboid_val_detection_v100")]
}

datasets_dict = {
    **carla_datasets,
    **waymo_datasets,
    **argoverse_cuboid_datasets,
    **argoverse_jpg_datasets,
    **waymo_cherrypicked_scenarios
}
