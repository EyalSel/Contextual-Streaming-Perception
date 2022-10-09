from pathlib import Path

import numpy as np

from offline_pylot.dataset_replay import OfflineWaymoSensorV1_1

import pandas as pd

from pylot.perception.detection.obstacle import Obstacle
from pylot.perception.detection.utils import BoundingBox2D
from pylot.perception.tracking.tracking_eval_operator import \
    TrackingScoringModule

from test_utils import CustomFlags

this_file_path = Path(__file__).parent.absolute()
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


scoring_module = TrackingScoringModule(
    CustomFlags({
        "tracking_metrics": [
            'num_misses', 'num_switches', 'num_false_positives', 'mota',
            'motp', 'mostly_tracked', 'mostly_lost', 'partially_tracked',
            'idf1', 'num_objects'
        ],
        "min_matching_iou":
        0.1,
    }))
sync = pd.DataFrame(
    list(
        filter(lambda x: x != {}, [
            load_and_get_score(obs1, obs2, scoring_module)
            for obs1, obs2 in zip(obst_list_list, obst_list_list)
        ])))


def test_tracking_eval_sync():
    assert np.all(sync[[
        "num_misses", "num_switches", "num_false_positives",
        "ratio_mostly_lost", "ratio_partially_tracked"
    ]] == 0)
    assert np.all(sync[["mota", "motp", "idf1"]] == 100)
    assert np.all(sync[["ratio_mostly_tracked"]] == 1)


scoring_module = TrackingScoringModule(
    CustomFlags({
        "tracking_metrics": [
            'num_misses', 'num_switches', 'num_false_positives', 'mota',
            'motp', 'mostly_tracked', 'mostly_lost', 'partially_tracked',
            'idf1', 'num_objects'
        ],
        "min_matching_iou":
        0.1,
    }))
delay_1 = pd.DataFrame(
    list(
        filter(lambda x: x != {}, [
            load_and_get_score(obs1, obs2, scoring_module)
            for obs1, obs2 in zip(obst_list_list[1:], obst_list_list[:-1])
        ])))


def test_tracking_eval_delay1():
    assert np.all(delay_1["mota"] == (
        1 - (delay_1["num_misses"] + delay_1["num_switches"] +
             delay_1["num_false_positives"]) / delay_1["num_objects"]) * 100)


if __name__ == '__main__':
    test_tracking_eval_delay1()
