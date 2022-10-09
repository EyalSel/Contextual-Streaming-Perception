"""
Tests attempt to cover both sequencing strategies and various runtimes:
Test 1: wait, runtime=60
Test 2: wait, runtime=260
Test 3: eager, runtime=60
Test 4: eager, runtime=190
test 5: 2-GPU, runtime=80
test 6: 2-GPU, runtime=260
test 7: 3-GPU, runtime=20
test 8: 3-GPU, runtime=250
test 9: 3-GPU, custom runtimes
"""

import erdos

from offline_pylot.prediction_frame_sequencing import PredictionFrameSequencer

from test_utils import (get_dummy_obstacle_message, make_operator_test,
                        single_operator_graph, start)

verbose = False


def wait_runtime_60_test(verbose):
    obstacle_stream = [
        f(t) for t in range(100, 800, 100) for f in
        (lambda t: get_dummy_obstacle_message(t, 60),
         lambda t: erdos.WatermarkMessage(erdos.Timestamp(coordinates=[t])))
    ] + [erdos.WatermarkMessage(erdos.Timestamp(is_top=True))]

    expected_sequenced_stream = [
        f(t) for t in range(100, 800, 100) for f in
        (lambda t: get_dummy_obstacle_message(t, 60),
         lambda t: erdos.WatermarkMessage(erdos.Timestamp(coordinates=[t])))
    ] + [erdos.WatermarkMessage(erdos.Timestamp(is_top=True))]
    return make_operator_test("wait_runtime_60_test",
                              single_operator_graph(PredictionFrameSequencer),
                              {
                                  "dataset_frame_interval": 100,
                                  "det_sequencer_policy": "wait"
                              },
                              obstacle_stream,
                              expected_sequenced_stream,
                              verbose=verbose)


def wait_runtime_260_test(verbose):
    obstacle_stream = [
        f(t) for t in range(100, 900, 100) for f in
        (lambda t: get_dummy_obstacle_message(t, 260),
         lambda t: erdos.WatermarkMessage(erdos.Timestamp(coordinates=[t])))
    ] + [erdos.WatermarkMessage(erdos.Timestamp(is_top=True))]

    expected_sequenced_stream = [
        # run starts at 100
        get_dummy_obstacle_message(100, 260, id_ts=100),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[100])),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[200])),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[300])),
        # run ended at 360 and starts at 400
        get_dummy_obstacle_message(400, 260, id_ts=400),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[400])),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[500])),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[600])),
        # run ended at 660 and starts at 700
        get_dummy_obstacle_message(700, 260, id_ts=700),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[700])),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[800])),
        erdos.WatermarkMessage(erdos.Timestamp(is_top=True))
    ]
    return make_operator_test("wait_runtime_260_test",
                              single_operator_graph(PredictionFrameSequencer),
                              {
                                  "dataset_frame_interval": 100,
                                  "det_sequencer_policy": "wait"
                              },
                              obstacle_stream,
                              expected_sequenced_stream,
                              verbose=verbose)


def eager_runtime_60_test(verbose):
    obstacle_stream = [
        f(t) for t in range(100, 800, 100) for f in
        (lambda t: get_dummy_obstacle_message(t, 60),
         lambda t: erdos.WatermarkMessage(erdos.Timestamp(coordinates=[t])))
    ] + [erdos.WatermarkMessage(erdos.Timestamp(is_top=True))]

    expected_sequenced_stream = [
        f(t) for t in range(100, 800, 100) for f in
        (lambda t: get_dummy_obstacle_message(t, 60),
         lambda t: erdos.WatermarkMessage(erdos.Timestamp(coordinates=[t])))
    ] + [erdos.WatermarkMessage(erdos.Timestamp(is_top=True))]
    return make_operator_test("eager_runtime_60_test",
                              single_operator_graph(PredictionFrameSequencer),
                              {
                                  "dataset_frame_interval": 100,
                                  "det_sequencer_policy": "eager"
                              },
                              obstacle_stream,
                              expected_sequenced_stream,
                              verbose=verbose)


def eager_runtime_190_test(verbose):
    obstacle_stream = [
        f(t) for t in range(100, 1000, 100) for f in
        (lambda t: get_dummy_obstacle_message(t, 190),
         lambda t: erdos.WatermarkMessage(erdos.Timestamp(coordinates=[t])))
    ] + [erdos.WatermarkMessage(erdos.Timestamp(is_top=True))]

    expected_sequenced_stream = [
        # run starts at 100
        get_dummy_obstacle_message(100, 190, id_ts=100),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[100])),
        # run ended at 290 and starts at 290 for ts=200
        get_dummy_obstacle_message(200, 280, id_ts=200),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[200])),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[300])),
        # run ended at 480 and starts at 480 for ts=400
        get_dummy_obstacle_message(400, 270, id_ts=400),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[400])),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[500])),
        # run ended at 670 and starts at 670 for ts=600
        get_dummy_obstacle_message(600, 260, id_ts=600),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[600])),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[700])),
        # run ended at 860 and starts at 860 for ts=800
        get_dummy_obstacle_message(800, 250, id_ts=800),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[800])),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[900])),
        erdos.WatermarkMessage(erdos.Timestamp(is_top=True))
    ]
    return make_operator_test("eager_runtime_230_test",
                              single_operator_graph(PredictionFrameSequencer),
                              {
                                  "dataset_frame_interval": 100,
                                  "det_sequencer_policy": "eager"
                              },
                              obstacle_stream,
                              expected_sequenced_stream,
                              verbose=verbose)


def gpu_2_runtime_80_test(verbose):
    obstacle_stream = [
        f(t) for t in range(100, 800, 100) for f in
        (lambda t: get_dummy_obstacle_message(t, 80),
         lambda t: erdos.WatermarkMessage(erdos.Timestamp(coordinates=[t])))
    ] + [erdos.WatermarkMessage(erdos.Timestamp(is_top=True))]

    expected_sequenced_stream = [
        f(t) for t in range(100, 800, 100) for f in
        (lambda t: get_dummy_obstacle_message(t, 80),
         lambda t: erdos.WatermarkMessage(erdos.Timestamp(coordinates=[t])))
    ] + [erdos.WatermarkMessage(erdos.Timestamp(is_top=True))]
    return make_operator_test("gpu_2_runtime_80_test",
                              single_operator_graph(PredictionFrameSequencer),
                              {
                                  "dataset_frame_interval": 100,
                                  "det_sequencer_policy": "2-GPU"
                              },
                              obstacle_stream,
                              expected_sequenced_stream,
                              verbose=verbose)


def gpu_2_runtime_260_test(verbose):
    obstacle_stream = [
        f(t) for t in range(100, 900, 100) for f in
        (lambda t: get_dummy_obstacle_message(t, 260),
         lambda t: erdos.WatermarkMessage(erdos.Timestamp(coordinates=[t])))
    ] + [erdos.WatermarkMessage(erdos.Timestamp(is_top=True))]

    expected_sequenced_stream = [
        # GPU 1 starts at 100
        get_dummy_obstacle_message(100, 260, id_ts=100),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[100])),
        # GPU 2 starts at 200
        get_dummy_obstacle_message(200, 260, id_ts=200),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[200])),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[300])),
        # GPU 1 ended at 360 and starts at 400
        get_dummy_obstacle_message(400, 260, id_ts=400),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[400])),
        # GPU 2 ended at 460 and starts at 500
        get_dummy_obstacle_message(500, 260, id_ts=500),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[500])),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[600])),
        # GPU 1 ended at 660 and starts at 700
        get_dummy_obstacle_message(700, 260, id_ts=700),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[700])),
        # GPU 2 ended at 760 and starts at 800
        get_dummy_obstacle_message(800, 260, id_ts=800),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[800])),
        erdos.WatermarkMessage(erdos.Timestamp(is_top=True))
    ]
    return make_operator_test("wait_runtime_260_test",
                              single_operator_graph(PredictionFrameSequencer),
                              {
                                  "dataset_frame_interval": 100,
                                  "det_sequencer_policy": "2-GPU"
                              },
                              obstacle_stream,
                              expected_sequenced_stream,
                              verbose=verbose)


def gpu_3_runtime_20_test(verbose):
    obstacle_stream = [
        f(t) for t in range(100, 800, 100) for f in
        (lambda t: get_dummy_obstacle_message(t, 20),
         lambda t: erdos.WatermarkMessage(erdos.Timestamp(coordinates=[t])))
    ] + [erdos.WatermarkMessage(erdos.Timestamp(is_top=True))]

    expected_sequenced_stream = [
        f(t) for t in range(100, 800, 100) for f in
        (lambda t: get_dummy_obstacle_message(t, 20),
         lambda t: erdos.WatermarkMessage(erdos.Timestamp(coordinates=[t])))
    ] + [erdos.WatermarkMessage(erdos.Timestamp(is_top=True))]
    return make_operator_test("gpu_3_runtime_20_test",
                              single_operator_graph(PredictionFrameSequencer),
                              {
                                  "dataset_frame_interval": 100,
                                  "det_sequencer_policy": "3-GPU"
                              },
                              obstacle_stream,
                              expected_sequenced_stream,
                              verbose=verbose)


def gpu_3_runtime_250_test(verbose):
    obstacle_stream = [
        f(t) for t in range(100, 900, 100) for f in
        (lambda t: get_dummy_obstacle_message(t, 250),
         lambda t: erdos.WatermarkMessage(erdos.Timestamp(coordinates=[t])))
    ] + [erdos.WatermarkMessage(erdos.Timestamp(is_top=True))]

    expected_sequenced_stream = [
        # GPU 1 starts at 100
        get_dummy_obstacle_message(100, 250, id_ts=100),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[100])),
        # GPU 2 starts at 200
        get_dummy_obstacle_message(200, 250, id_ts=200),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[200])),
        # GPU 3 starts at 300
        get_dummy_obstacle_message(300, 250, id_ts=300),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[300])),
        # GPU 1 ended at 350 and starts at 400
        get_dummy_obstacle_message(400, 250, id_ts=400),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[400])),
        # GPU 2 ended at 450 and starts at 500
        get_dummy_obstacle_message(500, 250, id_ts=500),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[500])),
        # GPU 3 ended at 550 and starts at 600
        get_dummy_obstacle_message(600, 250, id_ts=600),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[600])),
        # GPU 1 ended at 650 and starts at 700
        get_dummy_obstacle_message(700, 250, id_ts=700),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[700])),
        # GPU 2 ended at 750 and starts at 800
        get_dummy_obstacle_message(800, 250, id_ts=800),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[800])),
        erdos.WatermarkMessage(erdos.Timestamp(is_top=True))
    ]
    return make_operator_test("wait_runtime_250_test",
                              single_operator_graph(PredictionFrameSequencer),
                              {
                                  "dataset_frame_interval": 100,
                                  "det_sequencer_policy": "3-GPU"
                              },
                              obstacle_stream,
                              expected_sequenced_stream,
                              verbose=verbose)


def gpu_3_custom_runtime_test(verbose):
    obstacle_stream = [
        get_dummy_obstacle_message(100, 450),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[100])),
        get_dummy_obstacle_message(200, 250),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[200])),
        get_dummy_obstacle_message(300, 50),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[300])),
        get_dummy_obstacle_message(400, 1000),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[400])),
        get_dummy_obstacle_message(500, 1000),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[500])),
        get_dummy_obstacle_message(600, 250),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[600])),
        get_dummy_obstacle_message(700, 250),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[700])),
        get_dummy_obstacle_message(800, 250),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[800])),
        get_dummy_obstacle_message(900, 250),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[900])),
    ] + [erdos.WatermarkMessage(erdos.Timestamp(is_top=True))]

    expected_sequenced_stream = [
        # GPU 1 starts at 100 and ends at 550 > GPU 3 which ends at 350
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[100])),
        # GPU 2 starts at 200 and ends at 450 > GPU 3 which ends at 350
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[200])),
        # GPU 3 starts at 300
        get_dummy_obstacle_message(300, 50, id_ts=300),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[300])),
        # GPU 3 ended at 350 and starts at 400, which ends at 1400 > GPU 1
        # which ends at 850
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[400])),
        # GPU 2 ended at 450 and starts at 500, which ends at 1500 > GPU 1
        # which ends at 850
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[500])),
        # GPU 1 ended at 550 and starts at 600, which ends at 850
        get_dummy_obstacle_message(600, 250, id_ts=600),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[600])),
        # All GPUs working
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[700])),
        # All GPUs working
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[800])),
        # GPU 1 ended at 850, and starts at 900
        get_dummy_obstacle_message(900, 250, id_ts=900),
        erdos.WatermarkMessage(erdos.Timestamp(coordinates=[900])),
        erdos.WatermarkMessage(erdos.Timestamp(is_top=True))
    ]
    return make_operator_test("gpu_3_custom_runtime_test",
                              single_operator_graph(PredictionFrameSequencer),
                              {
                                  "dataset_frame_interval": 100,
                                  "det_sequencer_policy": "3-GPU"
                              },
                              obstacle_stream,
                              expected_sequenced_stream,
                              verbose=verbose)


def test_sequencer_wait_runtime_60_test():
    start(wait_runtime_60_test(verbose))


def test_sequencer_wait_runtime_260_test():
    start(wait_runtime_260_test(verbose))


def test_sequencer_eager_runtime_60_test():
    start(eager_runtime_60_test(verbose))


def test_sequencer_eager_runtime_190_test():
    start(eager_runtime_190_test(verbose))


def test_sequencer_gpu_2_runtime_80_test():
    start(gpu_2_runtime_80_test(verbose))


def test_sequencer_gpu_2_runtime_260_test():
    start(gpu_2_runtime_260_test(verbose))


def test_sequencer_gpu_3_runtime_20_test():
    start(gpu_3_runtime_20_test(verbose))


def test_sequencer_gpu_3_runtime_250_test():
    start(gpu_3_runtime_250_test(verbose))


def test_sequencer_gpu_3_custom_runtime_test():
    start(gpu_3_custom_runtime_test(verbose))


if __name__ == "__main__":
    test_sequencer_wait_runtime_60_test()
    test_sequencer_wait_runtime_260_test()
    test_sequencer_eager_runtime_60_test()
    test_sequencer_eager_runtime_190_test()
    test_sequencer_gpu_2_runtime_80_test()
    test_sequencer_gpu_2_runtime_260_test()
    test_sequencer_gpu_3_runtime_20_test()
    test_sequencer_gpu_3_runtime_250_test()
    test_sequencer_gpu_3_custom_runtime_test()
