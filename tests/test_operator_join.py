import os

import erdos

from offline_pylot.utils import join_streams

from test_utils import (CustomSequenceSender, SequenceVerifier,
                        get_dummy_obstacle_message, make_operator_test, start)

verbose = False


def join_streams_graph(input_stream_content, expected_output_stream, FLAGS,
                       op_config, verbose):
    input_streams = [
        erdos.connect(CustomSequenceSender,
                      op_config(f"custom_sequence_sender_{i}"), [],
                      input_stream_content[i])[0]
        for i in range(len(input_stream_content))
    ]

    (output_stream, ) = join_streams("join_stream", input_streams)

    (stream_for_tracker_verify_finish, ) = erdos.connect(
        SequenceVerifier, op_config("sequence_verifier-join_stream"),
        [output_stream], expected_output_stream, os.getpid(), verbose, "red")

    extract_stream = erdos.ExtractStream(stream_for_tracker_verify_finish)
    node_handle = erdos.run_async()
    return node_handle, extract_stream


def join_three_streams(verbose):
    input_stream_1 = [
        get_dummy_obstacle_message(100, 10),
    ] + [erdos.WatermarkMessage(erdos.Timestamp(is_top=True))]

    input_stream_2 = [
        get_dummy_obstacle_message(100, 10),
    ] + [erdos.WatermarkMessage(erdos.Timestamp(is_top=True))]

    input_stream_3 = [
        get_dummy_obstacle_message(100, 10),
    ] + [erdos.WatermarkMessage(erdos.Timestamp(is_top=True))]

    input_streams = [input_stream_1, input_stream_2, input_stream_3]

    expected_output_stream = [
        get_dummy_obstacle_message(100, 10),
        get_dummy_obstacle_message(100, 10),
        get_dummy_obstacle_message(100, 10),
        erdos.WatermarkMessage(erdos.Timestamp(is_top=True))
    ]

    return make_operator_test("join_three_streams",
                              join_streams_graph, {},
                              input_streams,
                              expected_output_stream,
                              verbose=verbose)


def join_two_streams(verbose):
    input_stream_1 = [
        get_dummy_obstacle_message(100, 10),
    ] + [erdos.WatermarkMessage(erdos.Timestamp(is_top=True))]

    input_stream_2 = [
        get_dummy_obstacle_message(100, 10),
    ] + [erdos.WatermarkMessage(erdos.Timestamp(is_top=True))]

    input_streams = [input_stream_1, input_stream_2]

    expected_output_stream = [
        get_dummy_obstacle_message(100, 10),
        get_dummy_obstacle_message(100, 10),
        erdos.WatermarkMessage(erdos.Timestamp(is_top=True))
    ]

    return make_operator_test("join_two_streams",
                              join_streams_graph, {},
                              input_streams,
                              expected_output_stream,
                              verbose=verbose)


def test_join_two_streams():
    start(join_two_streams(verbose))


def test_join_three_streams():
    start(join_three_streams(verbose))


if __name__ == '__main__':
    test_join_two_streams()
    test_join_three_streams()
