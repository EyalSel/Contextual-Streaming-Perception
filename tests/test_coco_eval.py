from pathlib import Path

import numpy as np

import pandas as pd

from offline_pylot.coco_detection_eval_operator import \
    CocoDetectionScoringModule
from offline_pylot.dataset_replay import OfflineWaymoSensorV1_1

from pylot.perception.detection.obstacle import Obstacle
from pylot.perception.detection.utils import BoundingBox2D

from test_utils import CustomFlags

this_file_path = Path(__file__).parent.absolute()
waymo_label_map_path = this_file_path / "../scripts/waymo.names"
S0_labels_path = this_file_path / "S0_no-frames.pl"


# Extract img to get dimensions for bbox label resizing
def obst_list(frame_contents):

    # resize obstacles and put into object
    def process_obst(obst_info):
        # resize bbox label given image label
        xmn, xmx, ymn, ymx = obst_info["bbox"]

        assert xmn < xmx and ymn < ymx and xmn >= 0 and ymn >= 0,\
            obst_info["bbox"]
        return Obstacle(BoundingBox2D(xmn, xmx, ymn, ymx),
                        confidence=1.0,
                        label=obst_info["label"],
                        id=obst_info["id"])

    return [process_obst(o) for o in frame_contents["obstacles"]]


sensor = OfflineWaymoSensorV1_1(S0_labels_path)

obst_list_list = [
    obst_list(sensor.get_frame(i)) for i in range(sensor.total_num_frames())
]


def load_and_get_score(ground_truth, predictions, scoring_module):
    scoring_module.add_datapoint(predictions, ground_truth)
    return scoring_module.get_scores()


scoring_module = CocoDetectionScoringModule(
    CustomFlags({
        # because we're comparing to ground truth labels the predictions also
        # use the waymo label map
        "path_coco_labels": str(waymo_label_map_path),
        "dataset_label_map": str(waymo_label_map_path),
        "coco_detection_eval_lookback": 10,
        "coco_detection_eval_freq": 2,
        "camera_image_height": 1280,
        "camera_image_width": 1920,
    }))
sync = pd.DataFrame(
    list(
        filter(lambda x: x != {}, [
            load_and_get_score(obs1, obs2, scoring_module)
            for obs1, obs2 in zip(obst_list_list, obst_list_list)
        ])))


def test_coco_eval_sync():
    assert np.all(
        np.isclose(
            sync.drop(columns=[
                "coco_AR_IoU=0.50:0.95_area=all_maxDets=1",
                "coco_AR_IoU=0.50:0.95_area=all_maxDets=10"
            ]), -1) | np.isclose(
                sync.drop(columns=[
                    "coco_AR_IoU=0.50:0.95_area=all_maxDets=1",
                    "coco_AR_IoU=0.50:0.95_area=all_maxDets=10"
                ]), 1))


if __name__ == '__main__':
    test_coco_eval_sync()
