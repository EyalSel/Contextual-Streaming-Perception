"""
A test intended to run on cached detection output (json + csv for each trial).
It tests that larger detection model architectures have higher average latency
and average accuracy.
"""

from pathlib import Path

import numpy as np

import pandas as pd

from tqdm import tqdm

from ad_config_search.data_curation import get_fn_without_ext, split_run
from ad_config_search.data_curation import (get_detection_tracking_accuracies,
                                            get_latencies)

result_path = Path("../scripts/S0_det_sweep")
jsons = list(result_path.glob("*.json"))
assert len(jsons) > 0, "Could not find any json files in the directory"

expected_csvs = [
    Path(json_path).parent / (get_fn_without_ext(json_path) + ".csv")
    for json_path in jsons
]
for expected_csv_path in expected_csvs:
    assert expected_csv_path.exists()

# The detection model uses a different column schema in the csv file. This is
# a (over) simple approach to get rid of these rows (counting the number of
# commas)
for csv in expected_csvs:
    with open(csv) as f:
        lines = f.readlines()
    lines = list(filter(lambda line: line.count(",") > 4, lines))
    with open(csv, "w") as f:
        f.writelines(lines)

json_file_index = pd.DataFrame([{
    **split_run(get_fn_without_ext(json_path)), "index":
    i
} for i, json_path in enumerate(jsons)])

detection_latencies = [
    np.mean(get_latencies(json_path)["efficientdet_operator.on_watermark"])
    for json_path in tqdm(jsons)
]
detection_accuracies = [
    np.mean(
        get_detection_tracking_accuracies(csv_path,
                                          time_mode="sync",
                                          anchor="end",
                                          end_frame_strategy="ceil")
        ["D_coco_AR_IoU=0.50:0.95_area=all_maxDets=100"])
    for csv_path in expected_csvs
]


def test_monotonically_increasing_accuracy():
    """
    Testing that accuracy increases monotonically as model architecture
    increases.
    """
    for tup, row in json_file_index.groupby(["run", "D-conf", "D-seq-pol"]):
        indices = np.array(row["D-model"].sort_values().index)
        accuracy_values = np.array(detection_accuracies)[indices]
        assert np.all(np.diff(accuracy_values) > 0), (
            tup, row["D-model"].sort_values().values, accuracy_values)


def test_monotonically_increasing_latency():
    """
    Testing that latency increases monotonically as model architecture
    increases.
    """
    for tup, row in json_file_index.groupby(["run", "D-conf", "D-seq-pol"]):
        indices = np.array(row["D-model"].sort_values().index)
        latency_values = np.array(detection_latencies)[indices]
        assert np.all(
            np.diff(latency_values) > 0), (tup,
                                           row["D-model"].sort_values().values,
                                           latency_values)
