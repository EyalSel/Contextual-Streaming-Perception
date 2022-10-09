from pathlib import Path
import pickle
import json
import numpy as np
import cv2
import collections


class DatasetReplayer:
    """
    A module that takes care of loading and then returning frame information
    on command. Separate from the offline sensor operator because it's also
    useful in gif generation for visualization.
    """
    def __init__(self, data_path):
        """
        Called during initialization, allows child class to examine the
        given dataset path and set any class variables it needs.

        Returns the number of frames in this segment
        """
        raise NotImplementedError("To be implemented by child class")

    def total_num_frames(self):
        """
        Returns the total number of frames loaded
        """
        raise NotImplementedError("To be implemented by child class")

    def get_frame(self, frame_index):
        """
        Return dictionary:
            - "center_camera_feed": Image as numpy array BGR format, np.uint8
            - "obstacles": list of dictionaries
                - "bbox": [xmn,xmx,ymn,ymx]
                - "label"
                - "id": non-negative integer, or -1 if not supported
        """
        raise NotImplementedError("To be implemented by child class")


# class OfflineWaymoSensorV1(DatasetReplayer):

#     def __init__(self, data_path):
#         frames = frames_from_path(data_path)
#         id_map = simplify_ids(frames)
#         dicts = [get_frame_dict(f, id_map, rbg=False)["FRONT"]
#                  for f in frames]
#         for d in dicts:
#           d["center_camera_feed"] = d["img"]
#           del d['img']
#         self.all_data = dicts

#     def total_num_frames(self):
#        return len(self.all_data)

#     def get_frame(self, frame_index):
#         return self.all_data[frame_index]


class OfflineWaymoSensorV1_1(DatasetReplayer):
    """
    Same as version 1 except _child_index_dir is run offline first
    """
    def __init__(self, data_path):
        with open(data_path, 'rb') as handle:
            dicts = pickle.load(handle)
        self.all_data = dicts

    def total_num_frames(self):
        return len(self.all_data)

    def get_frame(self, frame_index):
        return self.all_data[frame_index]


class OfflineArgoverseSensorJPG(DatasetReplayer):
    def __init__(self, data_path):
        data_path = Path(data_path)
        sorted_image_files = sorted(list(data_path.glob("*.jpg")),
                                    key=lambda p: int(p.stem.split("_")[-1]))
        with open(
                data_path.parent.parent.parent.parent /
                "Argoverse-HD/argoverse_jpg_to_annotations.pl", 'rb') as f:
            jpg_to_annotations = pickle.load(
                f
            )  # prepped separately in notebooks/Argoverse-HD-examination.ipynb
        annotations_ids = [
            jpg_to_annotations[f.name] for f in sorted_image_files
        ]
        with open(
                data_path.parent.parent.parent.parent /
                "Argoverse-HD/annotations/htc_dconv2_ms_train.json", 'r') as f:
            json_file = json.load(f)
            self.category_id_to_name = [
                d["name"] for d in json_file["categories"]
            ]
            self.annotations = [[
                json_file["annotations"][id] for id in id_array
            ] for id_array in annotations_ids]
        from imageio import imread
        from tqdm import tqdm
        if (data_path / "cached_frames.npy").exists():
            print("Loading npy to memory...")
            import time
            start = time.time()
            self.loaded_image_files = np.load(data_path / "cached_frames.npy")
            print("Took {}s".format(time.time() - start))
        else:
            print("Loading {} jpg files to memory...".format(
                len(sorted_image_files)))
            self.loaded_image_files = [
                imread(f) for f in tqdm(sorted_image_files)
            ]  # RGB

    def total_num_frames(self):
        return len(self.loaded_image_files)

    def get_frame(self, frame_index):
        return {
            "center_camera_feed":
            self.loaded_image_files[frame_index][:, :, ::-1],
            "obstacles": [
                {
                    "bbox": [
                        ann["bbox"][0],  # xmn
                        ann["bbox"][0] + ann["bbox"][2],  # xmx
                        ann["bbox"][1],  # ymn
                        ann["bbox"][1] + ann["bbox"][3]  # ymx
                    ],
                    "label":
                    self.category_id_to_name[ann["category_id"]],
                    "id":
                    -1
                } for ann in self.annotations[frame_index]
            ]
        }


class OfflineArgoverseSensorCuboid(DatasetReplayer):
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        missing_frames = [
            i for i, a in enumerate(self.data) if a["obstacles"] is None
        ]
        # missing labels can be found either on the very beginning or the very
        # end of the video, or not at all, this is cumbersome code to trim
        # these parts
        if len(missing_frames) > 0 and missing_frames[0] == 0:
            # shifting indices down 1
            missing_frames = [x - 1 for x in missing_frames][1:]
            self.data = self.data[1:]
        if len(missing_frames) > 0:
            last_frame = len(self.data)
            for i in reversed(missing_frames):
                assert i == last_frame - 1, (
                    "there's a missing, nonconsecutive frame from the end!")
                last_frame -= 1
            self.data = self.data[:-len(missing_frames)]

    def total_num_frames(self):
        return len(self.data)

    def get_frame(self, frame_index):
        """
        Return dictionary:
            - "center_camera_feed": Image as numpy array BGR format, np.uint8
            - "obstacles": list of dictionaries
                - "bbox": [xmn,xmx,ymn,ymx]
                - "label"
                - "id": non-negative integer, or -1 if not supported
        """
        return {
            "center_camera_feed":
            self.data[frame_index]["center_camera_feed"][:, :, ::-1].astype(
                np.uint8),
            "obstacles": [{
                **d, 'label': d["label"].lower()
            } for d in self.data[frame_index]["obstacles"]]
        }


class OfflineCarlaSensor(DatasetReplayer):
    def __init__(self, data_path):
        # extract file paths from directory
        json_files = list(data_path.glob("*.json"))
        png_files = list(data_path.glob("*.png"))

        def get_file_number(path):
            # get fn, drop extension, split by '-' and get last value
            return int(path.stem.split("-")[-1])

        # verify some properties of the json and png file numbers
        json_numbers = sorted([get_file_number(p) for p in json_files])
        png_numbers = sorted([get_file_number(p) for p in png_files])
        # all unique
        assert np.array_equal(np.unique(json_numbers), json_numbers), \
            "json numbers not unique in directory {}".format(data_path)
        assert np.array_equal(np.unique(png_numbers), png_numbers), \
            "png numbers not unique in directory {}".format(data_path)
        only_in_json = list(set(json_numbers) - set(png_numbers))
        only_in_png = list(set(png_numbers) - set(json_numbers))
        assert len(only_in_json) == 0, \
            "nonzero set of fn numbers in json files in dir {}: {}".format(
                data_path, only_in_json)
        assert len(only_in_png) == 0, \
            "nonzero set of fn numbers in png files in dir {}: {}".format(
                data_path, only_in_png)

        zipped_paths = zip(sorted(json_files, key=get_file_number),
                           sorted(png_files, key=get_file_number))
        zipped_paths = list(zipped_paths)
        assert len(zipped_paths) > 0, len(zipped_paths)
        self.zipped_paths = zipped_paths

        self._logger.debug("Found {} json, png pairs in {}".format(
            len(self.zipped_paths), self.DATASET_PATH))

    def total_num_frames(self):
        return len(self.zipped_paths)

    def get_frame(self, frame_index):
        json_path, png_path = self.zipped_paths[frame_index]
        # Extract img to get dimensions for bbox label resizing
        img = cv2.imread(str(png_path)).astype(np.uint8)  # BGR

        def read_json(path):
            with open(path, 'r') as f:
                return json.load(f)

        def get_obst(row):
            # bbox_coords format: [[x_min,y_min],[x_max,y_max]]
            class_label, bbox_coords = row
            bbox_coords = np.array(bbox_coords).T
            arg_order = bbox_coords.reshape(-1)  # xmn,xmx,ymn,ymx
            xmn, xmx, ymn, ymx = arg_order
            assert xmn < xmx and ymn < ymx and xmn >= 0 and ymn >= 0, \
                (arg_order, self.zipped_paths[frame_index])
            return {"bbox": arg_order, "label": class_label, "id": -1}

        obstacles = [get_obst(r) for r in read_json(json_path)]
        return {"center_camera_feed": img, "obstacles": obstacles}


class UAVDT(DatasetReplayer):
    def __init__(self, data_path):
        data_path = Path(data_path)
        label_path = data_path / "gt_whole.txt"
        self.frame_index_map = collections.defaultdict(list)
        self.img_map = {}
        with open(label_path, 'r') as f:
            for line in f:
                frame_index, target_id, bbox_l, bbox_t, \
                    bbox_w, bbox_h, _, _, label = line.strip().split(',')
                frame_index, target_id, bbox_l, bbox_t, \
                    bbox_w, bbox_h, label = int(frame_index), \
                    int(target_id), int(bbox_l), int(bbox_t), \
                    int(bbox_w), int(bbox_h), int(label)
                bbox = [bbox_l, bbox_l + bbox_w, bbox_t, bbox_t + bbox_h]
                frame_label = {'bbox': bbox, 'label': label, 'id': target_id}
                self.frame_index_map[frame_index].append(frame_label)
                if frame_index not in self.img_map:
                    img_name = \
                        f"img{'0' * (6 - len(str(frame_index)))}{frame_index}"
                    img_path = data_path / (img_name + ".jpg")
                    img = cv2.imread(str(img_path)).astype(np.uint8)
                    self.img_map[frame_index] = img

    def total_num_frames(self):
        return len(self.frame_index_map)

    def get_frame(self, frame_index):
        return {
            'obstacles': self.frame_index_map[frame_index + 1],
            'center_camera_feed': self.img_map[frame_index + 1]
        }  # offset by 1
