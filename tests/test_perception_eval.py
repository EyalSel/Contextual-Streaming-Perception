"""
end_to_end_test:
        sequencer pol=eager
        track runtime=10
        track reinit=15
        reload freq=2
        evaluate_timely=True
        matching_policy="ceil"
changing_end_times: [where earlier start times can finish sooner]
        evaluate_timely=True
        matching_policy="ceil"
missing_start_times:
        evaluate_timely=True
        matching_policy="ceil"

changing_end_times_sync: [where earlier start times can finish sooner]
        evaluate_timely=False
        matching_policy="ceil"
missing_start_times_sync:
        evaluate_timely=False
        matching_policy="ceil"

end_to_end_test_round:
        sequencer pol=eager
        track runtime=10
        track reinit=15
        reload freq=2
        evaluate_timely=True
        matching_policy="round"


"""

import os

import erdos

from offline_pylot.prediction_frame_sequencing import PredictionFrameSequencer

from pylot.perception.base_perception_eval_operator import (
    BasePerceptionEvalOperator, ScoringModule)
from pylot.perception.camera_frame import CameraFrame
from pylot.perception.messages import FrameMessage

from test_tracker import make_tracker_op

from test_utils import (CustomSequenceSender, SequenceVerifier,
                        get_dummy_obstacle_message, identity_print_stream,
                        make_operator_test, start)

verbose = False


class UnitTestScoringModule(ScoringModule):
    def __init__(self, flags):
        pass


class UnitTestModeStreamEval(BasePerceptionEvalOperator):
    """
    Just tests the (frame_time, ground_time, start/end) matchings by sending
    them into the finished indicator stream. In order to maintain a
    pre-determined order this child class takes over the watermark sending
    and expects flow watermark to be turned off. For the purposes of testing
    this child class sends a watermark between every message, something which
    the parent class would not guarantee.
    """
    def __init__(self, obstacle_tracking_stream, ground_obstacles_stream,
                 finished_indicator_stream, evaluate_timely, matching_policy,
                 frame_gap, flags):
        super().__init__(obstacle_tracking_stream, ground_obstacles_stream,
                         finished_indicator_stream, evaluate_timely,
                         matching_policy, frame_gap, UnitTestScoringModule,
                         flags)
        self.finished_indicator_stream = finished_indicator_stream
        self.next_ts = self._frame_gap

    def on_watermark(self, timestamp, finished_indicator_stream):
        super().on_watermark(timestamp, finished_indicator_stream)
        if timestamp.is_top:
            self.finished_indicator_stream.send(
                erdos.WatermarkMessage(timestamp))

    def compute_accuracy(self, frame_time, ground_time, end_anchored):
        anchor = "end" if end_anchored else "start"
        assert self.get_ground_truth_at(ground_time) is not None, ground_time
        assert self.get_prediction_at(frame_time) is not None, frame_time
        self.finished_indicator_stream.send(
            erdos.Message(erdos.Timestamp(coordinates=[self.next_ts]),
                          (frame_time, ground_time, anchor)))
        self.finished_indicator_stream.send(
            erdos.WatermarkMessage(
                erdos.Timestamp(coordinates=[self.next_ts])))
        self.next_ts += self._frame_gap


def tracker_eval_test_graph(camera_stream_length, track_runtime,
                            reinit_runtime):
    def tracker_eval_test_graph(input_stream_content, expected_output_stream,
                                FLAGS, op_config, verbose):
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
            PredictionFrameSequencer, op_config("obstacle_sequencer", False),
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

        # flow watermark is off during testing
        (tracking_eval_stream, ) = erdos.connect(
            UnitTestModeStreamEval, op_config("tracking_eval_operator", False),
            [tracking_stream, full_obstacle_stream], FLAGS.evaluate_timely,
            FLAGS.matching_policy, FLAGS.dataset_frame_interval, FLAGS)

        (stream_for_tracker_verify_finish, ) = erdos.connect(
            SequenceVerifier, op_config("sequence_verifier"),
            [tracking_eval_stream], expected_output_stream, os.getpid(),
            verbose, "red")

        extract_stream = erdos.ExtractStream(stream_for_tracker_verify_finish)
        node_handle = erdos.run_async()
        return node_handle, extract_stream

    return tracker_eval_test_graph


def tracker_eval_test_graph_small(obstacle_stream_length):
    def tracker_eval_test_graph_small(input_stream_content,
                                      expected_output_stream, FLAGS, op_config,
                                      verbose):
        full_obstacle_contents = [
            f(t) for t in range(100, obstacle_stream_length, 100)
            for f in (lambda t: get_dummy_obstacle_message(t, 160), lambda t:
                      erdos.WatermarkMessage(erdos.Timestamp(coordinates=[t])))
        ] + [erdos.WatermarkMessage(erdos.Timestamp(is_top=True))]

        (full_obstacle_stream, ) = erdos.connect(
            CustomSequenceSender, op_config("full_detection_sequence_sender"),
            [], full_obstacle_contents)

        (tracking_stream, ) = erdos.connect(CustomSequenceSender,
                                            op_config("tracking_operator"), [],
                                            input_stream_content)

        # flow watermark is off during testing
        (tracking_eval_stream, ) = erdos.connect(
            UnitTestModeStreamEval, op_config("tracking_eval_operator", False),
            [tracking_stream, full_obstacle_stream], FLAGS.evaluate_timely,
            FLAGS.matching_policy, FLAGS.dataset_frame_interval, FLAGS)

        (stream_for_tracker_verify_finish, ) = erdos.connect(
            SequenceVerifier, op_config("sequence_verifier"),
            [tracking_eval_stream], expected_output_stream, os.getpid(),
            verbose, "red")

        extract_stream = erdos.ExtractStream(stream_for_tracker_verify_finish)
        node_handle = erdos.run_async()
        return node_handle, extract_stream

    return tracker_eval_test_graph_small


def make_expected_output(expected_output):
    return [
        f(i, tup) for i, tup in enumerate(expected_output)
        for f in (lambda i, tup: erdos.Message(
            erdos.Timestamp(coordinates=[(i + 1) * 100]), tup),
                  lambda i, tup: erdos.WatermarkMessage(
                      erdos.Timestamp(coordinates=[(i + 1) * 100])))
    ] + [erdos.WatermarkMessage(erdos.Timestamp(is_top=True))]


def end_to_end_test(verbose):
    dataset_frame_interval = 100

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

    # expected_tracker_stream = [
    #     # run starts at 100
    #     # receive detection get_dummy_obstacle_message(100, 160, id_ts=100),
    #     get_dummy_obstacle_message(100, 185, id_ts=100),
    #     erdos.WatermarkMessage(erdos.Timestamp(coordinates=[100])),
    #     # receive detection get_dummy_obstacle_message(200, 220, id_ts=200),
    #     # don't apply reinit
    #     get_dummy_obstacle_message(200, 95, id_ts=100),
    #     erdos.WatermarkMessage(erdos.Timestamp(coordinates=[200])),
    #     # detection skips
    #     get_dummy_obstacle_message(300, 10, id_ts=100),
    #     erdos.WatermarkMessage(erdos.Timestamp(coordinates=[300])),
    #     # receive detection get_dummy_obstacle_message(400, 180, id_ts=400),
    #     get_dummy_obstacle_message(400, 205, id_ts=400),
    #     erdos.WatermarkMessage(erdos.Timestamp(coordinates=[400])),
    #     # receive detection get_dummy_obstacle_message(500, 240, id_ts=500),
    #     # don't apply reinit
    #     get_dummy_obstacle_message(500, 115, id_ts=400),
    #     erdos.WatermarkMessage(erdos.Timestamp(coordinates=[500])),
    #     # detection skips
    #     get_dummy_obstacle_message(600, 25, id_ts=400),
    #     erdos.WatermarkMessage(erdos.Timestamp(coordinates=[600])),
    #     # receive detection get_dummy_obstacle_message(700, 200, id_ts=700),
    #     get_dummy_obstacle_message(700, 225, id_ts=700),
    #     erdos.WatermarkMessage(erdos.Timestamp(coordinates=[700])),
    #     # detection skips
    #     get_dummy_obstacle_message(800, 135, id_ts=700),
    #     erdos.WatermarkMessage(erdos.Timestamp(coordinates=[800])),
    #     erdos.WatermarkMessage(erdos.Timestamp(is_top=True))
    # ]

    evaluate_timely = True
    matching_policy = "ceil"

    expected_output = [
        (100, 300, "start"),
        (200, 300, "start"),
        (200, 300, "end"),
        (300, 400, "start"),
        (300, 400, "end"),
        (300, 500, "end"),
        (300, 600, "end"),
        (400, 700, "start"),
        (500, 700, "start"),
        (600, 700, "start"),
        (600, 700, "end"),
        (600, 800, "end"),
    ]

    expected_tracker_eval_stream = make_expected_output(expected_output)

    return make_operator_test(
        "end_to_end_test",
        tracker_eval_test_graph(900,
                                track_runtime=track_runtime,
                                reinit_runtime=reinit_runtime),
        {
            "dataset_frame_interval": dataset_frame_interval,
            "det_sequencer_policy": seq_pol,
            "obstacle_track_max_age": 1,
            "min_matching_iou": 0.1,
            "track_every_nth_detection": reload_freq,
            "evaluate_timely": evaluate_timely,
            "matching_policy": matching_policy
        },
        obstacle_stream,
        expected_tracker_eval_stream,
        verbose=verbose)


def changing_end_times(verbose):
    dataset_frame_interval = 100

    expected_tracker_stream = [
        get_dummy_obstacle_message(100, 500, id_ts=100),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[100])),
        get_dummy_obstacle_message(200, 1000, id_ts=100),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[200])),
        get_dummy_obstacle_message(300, 10, id_ts=100),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[300])),
        get_dummy_obstacle_message(400, 230, id_ts=400),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[400])),
        get_dummy_obstacle_message(500, 110, id_ts=400),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[500])),
        get_dummy_obstacle_message(600, 1000, id_ts=400),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[600])),
        get_dummy_obstacle_message(700, 1000, id_ts=700),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[700])),
        get_dummy_obstacle_message(800, 10, id_ts=700),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[800])),
        get_dummy_obstacle_message(900, 10, id_ts=700),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[900])),
        erdos.WatermarkMessage(erdos.Timestamp(is_top=True))
    ]

    evaluate_timely = True
    matching_policy = "ceil"

    expected_output = [
        (300, 400, "start"),
        (300, 400, "end"),
        (300, 500, "end"),
        (100, 600, "start"),
        (300, 600, "end"),
        (400, 700, "start"),
        (500, 700, "start"),
        (500, 700, "end"),
        (500, 800, "end"),
        (800, 900, "start"),
        (800, 900, "end"),
    ]

    expected_tracker_eval_stream = make_expected_output(expected_output)

    return make_operator_test(
        "changing_end_times",
        tracker_eval_test_graph_small(1000), {
            "dataset_frame_interval": dataset_frame_interval,
            "evaluate_timely": evaluate_timely,
            "matching_policy": matching_policy
        },
        expected_tracker_stream,
        expected_tracker_eval_stream,
        verbose=verbose)


def missing_start_times(verbose):
    dataset_frame_interval = 100

    expected_tracker_stream = [
        get_dummy_obstacle_message(100, 10, id_ts=100),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[100])),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[200])),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[300])),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[400])),
        get_dummy_obstacle_message(500, 1000, id_ts=400),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[500])),
        get_dummy_obstacle_message(600, 10, id_ts=400),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[600])),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[700])),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[800])),
        get_dummy_obstacle_message(900, 10, id_ts=700),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[900])),
        erdos.WatermarkMessage(erdos.Timestamp(is_top=True))
    ]

    evaluate_timely = True
    matching_policy = "ceil"

    expected_output = [
        (100, 200, "start"),
        (100, 200, "end"),
        (100, 300, "end"),
        (100, 400, "end"),
        (100, 500, "end"),
        (100, 600, "end"),
        (600, 700, "start"),
        (600, 700, "end"),
        (600, 800, "end"),
        (600, 900, "end"),
    ]

    expected_tracker_eval_stream = make_expected_output(expected_output)

    return make_operator_test(
        "missing_start_times",
        tracker_eval_test_graph_small(1000), {
            "dataset_frame_interval": dataset_frame_interval,
            "evaluate_timely": evaluate_timely,
            "matching_policy": matching_policy
        },
        expected_tracker_stream,
        expected_tracker_eval_stream,
        verbose=verbose)


def changing_end_times_sync(verbose):
    dataset_frame_interval = 100

    expected_tracker_stream = [
        get_dummy_obstacle_message(100, 500, id_ts=100),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[100])),
        get_dummy_obstacle_message(200, 1000, id_ts=100),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[200])),
        get_dummy_obstacle_message(300, 10, id_ts=100),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[300])),
        get_dummy_obstacle_message(400, 230, id_ts=400),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[400])),
        get_dummy_obstacle_message(500, 110, id_ts=400),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[500])),
        get_dummy_obstacle_message(600, 1000, id_ts=400),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[600])),
        get_dummy_obstacle_message(700, 1000, id_ts=700),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[700])),
        get_dummy_obstacle_message(800, 10, id_ts=700),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[800])),
        get_dummy_obstacle_message(900, 10, id_ts=700),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[900])),
        erdos.WatermarkMessage(erdos.Timestamp(is_top=True))
    ]

    evaluate_timely = False
    matching_policy = "ceil"

    expected_output = [
        (100, 100, "start"),
        (100, 100, "end"),
        (200, 200, "start"),
        (200, 200, "end"),
        (300, 300, "start"),
        (300, 300, "end"),
        (400, 400, "start"),
        (400, 400, "end"),
        (500, 500, "start"),
        (500, 500, "end"),
        (600, 600, "start"),
        (600, 600, "end"),
        (700, 700, "start"),
        (700, 700, "end"),
        (800, 800, "start"),
        (800, 800, "end"),
        (900, 900, "start"),
        (900, 900, "end"),
    ]

    expected_tracker_eval_stream = make_expected_output(expected_output)

    return make_operator_test(
        "changing_end_times_sync",
        tracker_eval_test_graph_small(1000), {
            "dataset_frame_interval": dataset_frame_interval,
            "evaluate_timely": evaluate_timely,
            "matching_policy": matching_policy
        },
        expected_tracker_stream,
        expected_tracker_eval_stream,
        verbose=verbose)


def missing_start_times_sync(verbose):
    dataset_frame_interval = 100

    expected_tracker_stream = [
        get_dummy_obstacle_message(100, 10, id_ts=100),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[100])),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[200])),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[300])),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[400])),
        get_dummy_obstacle_message(500, 1000, id_ts=400),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[500])),
        get_dummy_obstacle_message(600, 10, id_ts=400),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[600])),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[700])),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[800])),
        get_dummy_obstacle_message(900, 10, id_ts=700),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[900])),
        erdos.WatermarkMessage(erdos.Timestamp(is_top=True))
    ]

    evaluate_timely = False
    matching_policy = "ceil"

    expected_output = [
        (100, 100, "start"),
        (100, 100, "end"),
        (100, 200, "end"),
        (100, 300, "end"),
        (100, 400, "end"),
        (500, 500, "start"),
        (500, 500, "end"),
        (600, 600, "start"),
        (600, 600, "end"),
        (600, 700, "end"),
        (600, 800, "end"),
        (900, 900, "start"),
        (900, 900, "end"),
    ]

    expected_tracker_eval_stream = make_expected_output(expected_output)

    return make_operator_test(
        "missing_start_times_sync",
        tracker_eval_test_graph_small(1000), {
            "dataset_frame_interval": dataset_frame_interval,
            "evaluate_timely": evaluate_timely,
            "matching_policy": matching_policy
        },
        expected_tracker_stream,
        expected_tracker_eval_stream,
        verbose=verbose)


def end_to_end_test_round(verbose):
    dataset_frame_interval = 100

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

    # expected_tracker_stream = [
    #     # run starts at 100
    #     # receive detection get_dummy_obstacle_message(100, 160, id_ts=100),
    #     get_dummy_obstacle_message(100, 185, id_ts=100),
    #     erdos.WatermarkMessage(erdos.Timestamp(coordinates=[100])),
    #     # receive detection get_dummy_obstacle_message(200, 220, id_ts=200),
    #     # don't apply reinit
    #     get_dummy_obstacle_message(200, 95, id_ts=100),
    #     erdos.WatermarkMessage(erdos.Timestamp(coordinates=[200])),
    #     # detection skips
    #     get_dummy_obstacle_message(300, 10, id_ts=100),
    #     erdos.WatermarkMessage(erdos.Timestamp(coordinates=[300])),
    #     # receive detection get_dummy_obstacle_message(400, 180, id_ts=400),
    #     get_dummy_obstacle_message(400, 205, id_ts=400),
    #     erdos.WatermarkMessage(erdos.Timestamp(coordinates=[400])),
    #     # receive detection get_dummy_obstacle_message(500, 240, id_ts=500),
    #     # don't apply reinit
    #     get_dummy_obstacle_message(500, 115, id_ts=400),
    #     erdos.WatermarkMessage(erdos.Timestamp(coordinates=[500])),
    #     # detection skips
    #     get_dummy_obstacle_message(600, 25, id_ts=400),
    #     erdos.WatermarkMessage(erdos.Timestamp(coordinates=[600])),
    #     # receive detection get_dummy_obstacle_message(700, 200, id_ts=700),
    #     get_dummy_obstacle_message(700, 225, id_ts=700),
    #     erdos.WatermarkMessage(erdos.Timestamp(coordinates=[700])),
    #     # detection skips
    #     get_dummy_obstacle_message(800, 135, id_ts=700),
    #     erdos.WatermarkMessage(erdos.Timestamp(coordinates=[800])),
    #     erdos.WatermarkMessage(erdos.Timestamp(is_top=True))
    # ]

    evaluate_timely = True
    matching_policy = "round"

    expected_output = [
        (100, 300, "start"),
        (200, 300, "start"),
        (300, 300, "start"),
        (300, 300, "end"),
        (300, 400, "end"),
        (300, 500, "end"),
        (400, 600, "start"),
        (500, 600, "start"),
        (600, 600, "start"),
        (600, 600, "end"),
        (600, 700, "end"),
        (600, 800, "end"),
    ]

    expected_tracker_eval_stream = make_expected_output(expected_output)

    return make_operator_test(
        "end_to_end_test_round",
        tracker_eval_test_graph(900,
                                track_runtime=track_runtime,
                                reinit_runtime=reinit_runtime),
        {
            "dataset_frame_interval": dataset_frame_interval,
            "det_sequencer_policy": seq_pol,
            "obstacle_track_max_age": 1,
            "min_matching_iou": 0.1,
            "track_every_nth_detection": reload_freq,
            "evaluate_timely": evaluate_timely,
            "matching_policy": matching_policy
        },
        obstacle_stream,
        expected_tracker_eval_stream,
        verbose=verbose)


def test_perception_eval_changing_end_times():
    start(changing_end_times(verbose))


def test_perception_eval_end_to_end_test():
    start(end_to_end_test(verbose))


def test_perception_eval_missing_start_times():
    start(missing_start_times(verbose))


def test_perception_eval_changing_end_times_sync():
    start(changing_end_times_sync(verbose))


def test_perception_eval_missing_start_times_sync():
    start(missing_start_times_sync(verbose))


def test_perception_eval_end_to_end_test_round():
    start(end_to_end_test_round(verbose))


if __name__ == "__main__":
    test_perception_eval_changing_end_times()
    test_perception_eval_end_to_end_test()
    test_perception_eval_missing_start_times()
    test_perception_eval_changing_end_times_sync()
    test_perception_eval_missing_start_times_sync()
    test_perception_eval_end_to_end_test_round()
