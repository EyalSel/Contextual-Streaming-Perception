"""
The unit tests in this file run both the sequencer and the tracker.
To aid debugging, the intermediate stream values sent from the sequencer to
the tracker are writte in the comment blocks below.

Tests:
sequencer: eager, runtime=160, reload freq=2, tracker runtime=10, reinit=15
sequencer: eager, runtime=160, reload freq=1, tracker runtime=10, reinit=15
sequencer: eager, runtime=160, reload freq=2, tracker runtime=300, reinit=15
"""

import os

import erdos

from offline_pylot.prediction_frame_sequencing import PredictionFrameSequencer

from pylot.perception.camera_frame import CameraFrame
from pylot.perception.messages import FrameMessage
from pylot.perception.tracking.object_tracker_operator import \
    ObjectTrackerOperator

from test_utils import (CustomSequenceSender, SequenceVerifier,
                        get_dummy_obstacle_message, identity_print_stream,
                        make_operator_test, start)

verbose = False


def make_tracker_op(track_runtime, reinit_runtime):
    class TestModeTracker(ObjectTrackerOperator):
        """
        Makes the runtime predetermined
        """
        def __init__(self, obstacles_stream, camera_stream,
                     time_to_decision_stream, obstacle_tracking_stream,
                     tracker_type, flags):
            super().__init__(obstacles_stream, camera_stream,
                             time_to_decision_stream, obstacle_tracking_stream,
                             tracker_type, flags)
            self.received_obstacles = None

        def _reinit_tracker(self, camera_frame, detected_obstacles):
            self.received_obstacles = detected_obstacles
            return reinit_runtime, None

        def _run_tracker(self, camera_frame):
            return (track_runtime, (True, self.received_obstacles))

    return TestModeTracker


def tracker_test_graph(camera_stream_length, track_runtime, reinit_runtime):
    def tracker_test_graph(input_stream_content, expected_output_stream, FLAGS,
                           op_config, verbose):
        camera_stream_content = [
            f(t) for t in range(100, camera_stream_length, 100)
            for f in (lambda t: FrameMessage(erdos.Timestamp(
                coordinates=[t]), CameraFrame(None, "BGR")), lambda t: erdos.
                      WatermarkMessage(erdos.Timestamp(coordinates=[t])))
        ] + [erdos.WatermarkMessage(erdos.Timestamp(is_top=True))]

        time_to_decision_stream_content = [
            f(t) for t in range(100, camera_stream_length, 100)
            for f in (lambda t: erdos.Message(erdos.Timestamp(coordinates=[t]),
                                              (100000, 100000)), lambda t:
                      erdos.WatermarkMessage(erdos.Timestamp(coordinates=[t])))
        ] + [erdos.WatermarkMessage(erdos.Timestamp(is_top=True))]

        (camera_stream, ) = erdos.connect(CustomSequenceSender,
                                          op_config("frame_sequence_sender"),
                                          [], camera_stream_content)

        (time_to_decision_stream, ) = erdos.connect(
            CustomSequenceSender, op_config("time_to_decision_stream"), [],
            time_to_decision_stream_content)

        (full_obstacle_stream, ) = erdos.connect(
            CustomSequenceSender, op_config("full_detection_sequence_sender"),
            [], input_stream_content)

        (sequenced_obstacle_stream, ) = erdos.connect(
            PredictionFrameSequencer, op_config("frame_sequencer", False),
            [full_obstacle_stream], FLAGS)

        if verbose:
            sequenced_obstacle_stream = identity_print_stream(
                "sequenced_obstacle", sequenced_obstacle_stream)

        (tracking_stream, ) = erdos.connect(
            make_tracker_op(track_runtime, reinit_runtime),
            op_config("tracker_operator"), [
                sequenced_obstacle_stream, camera_stream,
                time_to_decision_stream
            ], "sort", FLAGS)

        (stream_for_tracker_verify_finish, ) = erdos.connect(
            SequenceVerifier, op_config("sequence_verifier"),
            [tracking_stream], expected_output_stream, os.getpid(), verbose,
            "red")

        extract_stream = erdos.ExtractStream(stream_for_tracker_verify_finish)
        node_handle = erdos.run_async()
        return node_handle, extract_stream

    return tracker_test_graph


def eager_runtime_160_reload_freq_2_test(verbose):
    obstacle_stream = [
        f(t) for t in range(100, 900, 100) for f in
        (lambda t: get_dummy_obstacle_message(t, 160),
         lambda t: erdos.WatermarkMessage(erdos.Timestamp(coordinates=[t])))
    ] + [erdos.WatermarkMessage(erdos.Timestamp(is_top=True))]

    seq_pol = "eager"

    # expected_sequenced_stream = [
    #     # run starts at 100
    #     get_dummy_obstacle_message(100, 160, id_ts=100),
    #     erdos.WatermarkMessage(erdos.Timestamp(coordinates=[100])),
    #     # run ended at 260 and starts at 260 for ts=200
    #     get_dummy_obstacle_message(200, 220, id_ts=200),
    #     erdos.WatermarkMessage(erdos.Timestamp(coordinates=[200])),
    #     erdos.WatermarkMessage(erdos.Timestamp(coordinates=[300])),
    #     # run ended at 420 and starts at 420 for ts=400
    #     get_dummy_obstacle_message(400, 180, id_ts=400),
    #     erdos.WatermarkMessage(erdos.Timestamp(coordinates=[400])),
    #     # run ended at 580 and starts at 580 for ts=600
    #     get_dummy_obstacle_message(500, 240, id_ts=500),
    #     erdos.WatermarkMessage(erdos.Timestamp(coordinates=[500])),
    #     erdos.WatermarkMessage(erdos.Timestamp(coordinates=[600])),
    #     # run ended at 740 and starts at 740 for ts=800
    #     get_dummy_obstacle_message(700, 200, id_ts=700),
    #     erdos.WatermarkMessage(erdos.Timestamp(coordinates=[700])),
    #     erdos.WatermarkMessage(erdos.Timestamp(coordinates=[800])),
    #     erdos.WatermarkMessage(erdos.Timestamp(is_top=True))
    # ]

    track_runtime = 10
    reinit_runtime = 15
    reload_freq = 2

    expected_tracker_stream = [
        # run starts at 100
        # receive detection get_dummy_obstacle_message(100, 160, id_ts=100),
        get_dummy_obstacle_message(100, 185, id_ts=100),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[100])),
        # receive detection get_dummy_obstacle_message(200, 220, id_ts=200),
        # don't apply reinit
        get_dummy_obstacle_message(200, 95, id_ts=100),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[200])),
        # detection skips
        get_dummy_obstacle_message(300, 10, id_ts=100),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[300])),
        # receive detection get_dummy_obstacle_message(400, 180, id_ts=400),
        get_dummy_obstacle_message(400, 205, id_ts=400),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[400])),
        # receive detection get_dummy_obstacle_message(500, 240, id_ts=500),
        # don't apply reinit
        get_dummy_obstacle_message(500, 115, id_ts=400),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[500])),
        # detection skips
        get_dummy_obstacle_message(600, 25, id_ts=400),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[600])),
        # receive detection get_dummy_obstacle_message(700, 200, id_ts=700),
        get_dummy_obstacle_message(700, 225, id_ts=700),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[700])),
        # detection skips
        get_dummy_obstacle_message(800, 135, id_ts=700),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[800])),
        erdos.WatermarkMessage(erdos.Timestamp(is_top=True))
    ]
    return make_operator_test("eager_runtime_160_reload_freq_2_test",
                              tracker_test_graph(
                                  900,
                                  track_runtime=track_runtime,
                                  reinit_runtime=reinit_runtime), {
                                      "dataset_frame_interval": 100,
                                      "det_sequencer_policy": seq_pol,
                                      "obstacle_track_max_age": 1,
                                      "min_matching_iou": 0.1,
                                      "track_every_nth_detection": reload_freq
                                  },
                              obstacle_stream,
                              expected_tracker_stream,
                              verbose=verbose)


def eager_runtime_160_reload_freq_1_test(verbose):
    obstacle_stream = [
        f(t) for t in range(100, 900, 100) for f in
        (lambda t: get_dummy_obstacle_message(t, 160),
         lambda t: erdos.WatermarkMessage(erdos.Timestamp(coordinates=[t])))
    ] + [erdos.WatermarkMessage(erdos.Timestamp(is_top=True))]

    seq_pol = "eager"

    # expected_sequenced_stream = [
    #     # run starts at 100
    #     get_dummy_obstacle_message(100, 160, id_ts=100),
    #     erdos.WatermarkMessage(erdos.Timestamp(coordinates=[100])),
    #     # run ended at 260 and starts at 260 for ts=200
    #     get_dummy_obstacle_message(200, 220, id_ts=200),
    #     erdos.WatermarkMessage(erdos.Timestamp(coordinates=[200])),
    #     erdos.WatermarkMessage(erdos.Timestamp(coordinates=[300])),
    #     # run ended at 420 and starts at 420 for ts=400
    #     get_dummy_obstacle_message(400, 180, id_ts=400),
    #     erdos.WatermarkMessage(erdos.Timestamp(coordinates=[400])),
    #     # run ended at 580 and starts at 580 for ts=600
    #     get_dummy_obstacle_message(500, 240, id_ts=500),
    #     erdos.WatermarkMessage(erdos.Timestamp(coordinates=[500])),
    #     erdos.WatermarkMessage(erdos.Timestamp(coordinates=[600])),
    #     # run ended at 740 and starts at 740 for ts=800
    #     get_dummy_obstacle_message(700, 200, id_ts=700),
    #     erdos.WatermarkMessage(erdos.Timestamp(coordinates=[700])),
    #     erdos.WatermarkMessage(erdos.Timestamp(coordinates=[800])),
    #     erdos.WatermarkMessage(erdos.Timestamp(is_top=True))
    # ]

    track_runtime = 10
    reinit_runtime = 15
    reload_freq = 1

    expected_tracker_stream = [
        # run starts at 100
        # receive detection get_dummy_obstacle_message(100, 160, id_ts=100),
        get_dummy_obstacle_message(100, 185, id_ts=100),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[100])),
        # receive detection get_dummy_obstacle_message(200, 220, id_ts=200),
        get_dummy_obstacle_message(200, 245, id_ts=200),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[200])),
        # detection skips
        get_dummy_obstacle_message(300, 155, id_ts=200),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[300])),
        # receive detection get_dummy_obstacle_message(400, 180, id_ts=400),
        get_dummy_obstacle_message(400, 205, id_ts=400),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[400])),
        # receive detection get_dummy_obstacle_message(500, 240, id_ts=500),
        get_dummy_obstacle_message(500, 265, id_ts=500),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[500])),
        # detection skips
        get_dummy_obstacle_message(600, 175, id_ts=500),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[600])),
        # receive detection get_dummy_obstacle_message(700, 200, id_ts=700),
        get_dummy_obstacle_message(700, 225, id_ts=700),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[700])),
        # detection skips
        get_dummy_obstacle_message(800, 135, id_ts=700),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[800])),
        erdos.WatermarkMessage(erdos.Timestamp(is_top=True))
    ]
    return make_operator_test("eager_runtime_160_reload_freq_1_test",
                              tracker_test_graph(
                                  900,
                                  track_runtime=track_runtime,
                                  reinit_runtime=reinit_runtime), {
                                      "dataset_frame_interval": 100,
                                      "det_sequencer_policy": seq_pol,
                                      "obstacle_track_max_age": 1,
                                      "min_matching_iou": 0.1,
                                      "track_every_nth_detection": reload_freq
                                  },
                              obstacle_stream,
                              expected_tracker_stream,
                              verbose=verbose)


def eager_runtime_300_reload_freq_2_test(verbose):
    obstacle_stream = [
        f(t) for t in range(100, 900, 100) for f in
        (lambda t: get_dummy_obstacle_message(t, 160),
         lambda t: erdos.WatermarkMessage(erdos.Timestamp(coordinates=[t])))
    ] + [erdos.WatermarkMessage(erdos.Timestamp(is_top=True))]

    seq_pol = "eager"

    # expected_sequenced_stream = [
    #     # run starts at 100
    #     get_dummy_obstacle_message(100, 160, id_ts=100),
    #     erdos.WatermarkMessage(erdos.Timestamp(coordinates=[100])),
    #     # run ended at 260 and starts at 260 for ts=200
    #     get_dummy_obstacle_message(200, 220, id_ts=200),
    #     erdos.WatermarkMessage(erdos.Timestamp(coordinates=[200])),
    #     erdos.WatermarkMessage(erdos.Timestamp(coordinates=[300])),
    #     # run ended at 420 and starts at 420 for ts=400
    #     get_dummy_obstacle_message(400, 180, id_ts=400),
    #     erdos.WatermarkMessage(erdos.Timestamp(coordinates=[400])),
    #     # run ended at 580 and starts at 580 for ts=600
    #     get_dummy_obstacle_message(500, 240, id_ts=500),
    #     erdos.WatermarkMessage(erdos.Timestamp(coordinates=[500])),
    #     erdos.WatermarkMessage(erdos.Timestamp(coordinates=[600])),
    #     # run ended at 740 and starts at 740 for ts=800
    #     get_dummy_obstacle_message(700, 200, id_ts=700),
    #     erdos.WatermarkMessage(erdos.Timestamp(coordinates=[700])),
    #     erdos.WatermarkMessage(erdos.Timestamp(coordinates=[800])),
    #     erdos.WatermarkMessage(erdos.Timestamp(is_top=True))
    # ]

    track_runtime = 300
    reinit_runtime = 15
    reload_freq = 2

    expected_tracker_stream = [
        # run starts at 100
        # receive detection get_dummy_obstacle_message(100, 160, id_ts=100),
        get_dummy_obstacle_message(100, 475, id_ts=100),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[100])),
        # receive detection get_dummy_obstacle_message(200, 220, id_ts=200),
        # don't use detection
        get_dummy_obstacle_message(200, 675, id_ts=100),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[200])),
        # detection skips
        get_dummy_obstacle_message(300, 875, id_ts=100),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[300])),
        # receive detection get_dummy_obstacle_message(400, 180, id_ts=400),
        get_dummy_obstacle_message(400, 1090, id_ts=400),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[400])),
        # receive detection get_dummy_obstacle_message(500, 240, id_ts=500),
        # don't use detection
        get_dummy_obstacle_message(500, 1290, id_ts=400),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[500])),
        # detection skips
        get_dummy_obstacle_message(600, 1490, id_ts=400),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[600])),
        # receive detection get_dummy_obstacle_message(700, 200, id_ts=700),
        get_dummy_obstacle_message(700, 1705, id_ts=700),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[700])),
        # detection skips
        get_dummy_obstacle_message(800, 1905, id_ts=700),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[800])),
        erdos.WatermarkMessage(erdos.Timestamp(is_top=True))
    ]
    return make_operator_test("eager_runtime_300_reload_freq_1_test",
                              tracker_test_graph(
                                  900,
                                  track_runtime=track_runtime,
                                  reinit_runtime=reinit_runtime), {
                                      "dataset_frame_interval": 100,
                                      "det_sequencer_policy": seq_pol,
                                      "obstacle_track_max_age": 1,
                                      "min_matching_iou": 0.1,
                                      "track_every_nth_detection": reload_freq
                                  },
                              obstacle_stream,
                              expected_tracker_stream,
                              verbose=verbose)


def test_tracker_eager_runtime_160_reload_freq_1_test():
    start(eager_runtime_160_reload_freq_1_test(verbose))


def test_tracker_eager_runtime_160_reload_freq_2_test():
    start(eager_runtime_160_reload_freq_2_test(verbose))


def test_tracker_eager_runtime_300_reload_freq_2_test():
    start(eager_runtime_300_reload_freq_2_test(verbose))


if __name__ == "__main__":
    test_tracker_eager_runtime_160_reload_freq_1_test()
    test_tracker_eager_runtime_160_reload_freq_2_test()
    test_tracker_eager_runtime_300_reload_freq_2_test()
