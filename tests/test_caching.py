import os
from pathlib import Path

import erdos

from offline_pylot.detection_stream_caching import (ObstacleStreamLogger,
                                                    ObstacleStreamReplayer)

from pylot.perception.messages import ObstaclesMessage

from test_utils import (CustomSequenceSender, SequenceVerifier,
                        get_dummy_obstacle_message, make_operator_test,
                        single_operator_graph, start)

verbose = False
fn = "test_caching_output.jsonl"


def notest_graph(input_stream_content, expected_output_stream, FLAGS,
                 op_config, verbose):
    (input_stream, ) = erdos.connect(CustomSequenceSender,
                                     op_config("custom_sequence_sender"), [],
                                     input_stream_content)

    (output_stream, ) = erdos.connect(ObstacleStreamLogger,
                                      op_config("stream_logger", True),
                                      [input_stream], FLAGS)

    (stream_for_tracker_verify_finish, ) = erdos.connect(
        SequenceVerifier, op_config("sequence_verifier-stream_logger"),
        [output_stream], expected_output_stream, os.getpid(), verbose, "red")

    extract_stream = erdos.ExtractStream(stream_for_tracker_verify_finish)
    node_handle = erdos.run_async()
    return node_handle, extract_stream


def write_part(verbose):
    if Path(fn).exists():
        Path(fn).unlink()
    obstacle_stream = [
        f(t) for t in range(100, 800, 100) for f in
        (lambda t: get_dummy_obstacle_message(t, 60, confidence=t / 1000),
         lambda t: erdos.WatermarkMessage(erdos.Timestamp(coordinates=[t])))
    ] + [erdos.WatermarkMessage(erdos.Timestamp(is_top=True))]

    expected_sequenced_stream = [
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[t]))
        for t in range(100, 800, 100)
    ] + [erdos.WatermarkMessage(erdos.Timestamp(is_top=True))]
    return make_operator_test(
        "wait_runtime_60_test",
        notest_graph, {
            "cache_obstacle_detection_destination_path": fn,
            "obstacle_detection_model_names": "test",
            "camera_image_height": 1280,
            "camera_image_width": 1920,
        },
        obstacle_stream,
        expected_sequenced_stream,
        verbose=verbose)


def read_part(verbose):
    assert Path(fn).exists(), fn
    obstacle_stream = [
        f(t) for t in range(100, 800, 100) for f in
        (lambda t: get_dummy_obstacle_message(t, 60),
         lambda t: erdos.WatermarkMessage(erdos.Timestamp(coordinates=[t])))
    ] + [erdos.WatermarkMessage(erdos.Timestamp(is_top=True))]

    expected_sequenced_stream = [
        ObstaclesMessage(erdos.Timestamp(coordinates=[100]), [], runtime=60.0),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[100])),
        ObstaclesMessage(erdos.Timestamp(coordinates=[200]), [], runtime=60.0),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[200])),
        ObstaclesMessage(erdos.Timestamp(coordinates=[300]), [], runtime=60.0),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[300])),
        get_dummy_obstacle_message(400, 60.0, confidence=0.4),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[400])),
        get_dummy_obstacle_message(500, 60.0, confidence=0.5),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[500])),
        get_dummy_obstacle_message(600, 60.0, confidence=0.6),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[600])),
        get_dummy_obstacle_message(700, 60.0, confidence=0.7),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[700])),
        erdos.WatermarkMessage(erdos.Timestamp(is_top=True))
    ]

    return make_operator_test(
        "wait_runtime_60_test",
        single_operator_graph(ObstacleStreamReplayer), {
            "cache_obstacle_detection_source_path": fn,
            "obstacle_detection_model_names": "test_model",
            "obstacle_detection_min_score_threshold": 0.4,
            "camera_image_height": 1280,
            "camera_image_width": 1920,
        },
        obstacle_stream,
        expected_sequenced_stream,
        verbose=verbose)


def test_caching():
    start(write_part(verbose))
    start(read_part(verbose))


if __name__ == '__main__':
    test_caching()
