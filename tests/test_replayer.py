"""
Jointly tests the offline sensor and Waymo's dataset replayer.
"""

import os
import pickle

import erdos
from erdos.operators.map import Map

import numpy as np

from offline_pylot.offline_dataset_sensors import OfflineDatasetSensor

import pylot
from pylot.drivers.sensor_setup import RGBCameraSetup
from pylot.perception.camera_frame import CameraFrame
from pylot.perception.messages import FrameMessage

from test_utils import (SequenceVerifier, get_dummy_obstacle_message,
                        make_operator_test, split_stream, start)
"""
- "center_camera_feed": Image as numpy array BGR format, np.uint8
- "obstacles": list of dictionaries
    - "bbox": [xmn,xmx,ymn,ymx]
    - "label"
    - "id": non-negative integer, or -1 if not supported
"""


class CameraFrameV2(CameraFrame):
    """
    This child class just adds the frame contents to the repr function to make
    sure the expected frame contents are being sent, for testing purposes.
    """
    def __init__(self, *args):
        # python classes can apparently have only 1 init, so this workaround
        # is used to initialize from parent class instance or from traditional
        # init args.
        if len(args) == 1:
            parent_obj = args[0]
            super().__init__(parent_obj.frame, parent_obj.encoding,
                             parent_obj.camera_setup)
        else:
            [frame, encoding, camera_setup] = args
            super().__init__(frame, encoding, camera_setup)

    def __repr__(self):
        return 'CameraFrame(encoding: {}, camera_setup: {}, frame: {})'.format(
            self.encoding, self.camera_setup, self.frame)

    def __str__(self):
        return 'CameraFrame(encoding: {}, camera_setup: {}, frame: {})'.format(
            self.encoding, self.camera_setup, self.frame)


images = np.arange(16).reshape(1, 4, 4, 1).repeat(
    3, axis=0) + np.arange(3).reshape(3, 1, 1, 1)
obstacles = [[{
    "bbox": [0, 1, 0, 1],
    "label": "vehicle",
    "id": 75 * (i + 1),
}] for i in range(3)]

pickle_file_contents = [{
    "center_camera_feed": img,
    "obstacles": obsts
} for img, obsts in zip(images, obstacles)]

with open("replayer_waymo_pickle.pl", 'wb') as f:
    pickle.dump(pickle_file_contents, f)

name = "offline-carla-camera"
camera_image_width = 4
camera_image_height = 4
camera_fov = 90.0

CENTER_CAMERA_LOCATION = pylot.utils.Location(1.3, 0.0, 1.8)
transform = pylot.utils.Transform(CENTER_CAMERA_LOCATION,
                                  pylot.utils.Rotation(pitch=-15))
offline_camera_setup = RGBCameraSetup(name,
                                      camera_image_width,
                                      camera_image_height,
                                      transform,
                                      fov=camera_fov)

verbose = False


def specific_verify_graph(keyword):
    def tracker_test_graph(input_stream_content, expected_output_stream, FLAGS,
                           op_config, verbose):
        backwards_notify_stream = erdos.LoopStream()
        (center_camera_stream, perfect_obstacles_stream,
         time_to_decision_stream) = erdos.connect(
             OfflineDatasetSensor,
             op_config(offline_camera_setup.get_name() + "_operator"),
             [backwards_notify_stream], offline_camera_setup, None, FLAGS)

        center_camera_stream, stream_back = split_stream(
            "split_stream", center_camera_stream, 2)

        backwards_notify_stream.set(center_camera_stream)

        # map the center camera stream coming from the offline sensor to camera
        # frame v2 to explicitly test the frame contents.
        (center_camera_stream, ) = erdos.connect(
            Map,
            erdos.OperatorConfig(name="camera_to_v2"), [center_camera_stream],
            function=lambda x: FrameMessage(x.timestamp, CameraFrameV2(x.frame)
                                            )
        )  # convert to more descriptive printing version of test

        stream_to_test = {
            "camera": center_camera_stream,
            "gt": perfect_obstacles_stream,
            "ttd": time_to_decision_stream
        }[keyword]

        (stream_verify_finish, ) = erdos.connect(
            SequenceVerifier, op_config("sequence_verifier_camera",
                                        True), [stream_to_test],
            expected_output_stream, os.getpid(), verbose, "red")

        extract_stream = erdos.ExtractStream(stream_verify_finish)

        node_handle = erdos.run_async()
        return node_handle, extract_stream

    return tracker_test_graph


def replay_waymo_pl_test_camera(verbose):
    expected_camera_stream = [
        f(i, t) for i, t in enumerate(range(75, 300, 75)) for f in
        (lambda i, t: FrameMessage(
            erdos.Timestamp(coordinates=[t]),
            CameraFrameV2(images[i], "BGR", offline_camera_setup)),
         lambda i, t: erdos.WatermarkMessage(erdos.Timestamp(coordinates=[t])))
    ] + [erdos.WatermarkMessage(erdos.Timestamp(is_top=True))]

    return make_operator_test("replay_waymo_pl_test_camera",
                              specific_verify_graph("camera"), {
                                  "camera_image_width": camera_image_width,
                                  "camera_image_height": camera_image_height,
                                  "camera_fov": camera_fov,
                                  "dataset_path": "replayer_waymo_pickle.pl",
                                  "dataset_name": "waymo",
                                  "dataset_frame_interval": 75,
                                  "csv_log_file_name": "replayer_test.csv",
                              },
                              None,
                              expected_camera_stream,
                              verbose=verbose)


def replay_waymo_pl_test_gt(verbose):
    expected_gt_stream = [
        f(t) for t in range(75, 300, 75) for f in
        (lambda t: get_dummy_obstacle_message(t, 0, confidence=1.0),
         lambda t: erdos.WatermarkMessage(erdos.Timestamp(coordinates=[t])))
    ] + [erdos.WatermarkMessage(erdos.Timestamp(is_top=True))]

    return make_operator_test("replay_waymo_pl_test_gt",
                              specific_verify_graph("gt"), {
                                  "camera_image_width": camera_image_width,
                                  "camera_image_height": camera_image_height,
                                  "camera_fov": camera_fov,
                                  "dataset_path": "replayer_waymo_pickle.pl",
                                  "dataset_name": "waymo",
                                  "dataset_frame_interval": 75,
                                  "csv_log_file_name": "replayer_test.csv",
                              },
                              None,
                              expected_gt_stream,
                              verbose=verbose)


def replay_waymo_pl_test_ttd(verbose):
    expected_ttd_stream = [
        f(t) for t in range(75, 300, 75) for f in
        (lambda t: erdos.Message(erdos.Timestamp(coordinates=[t]),
                                 (100000, 100000)),
         lambda t: erdos.WatermarkMessage(erdos.Timestamp(coordinates=[t])))
    ] + [erdos.WatermarkMessage(erdos.Timestamp(is_top=True))]

    return make_operator_test("replay_waymo_pl_test_ttd",
                              specific_verify_graph("ttd"), {
                                  "camera_image_width": camera_image_width,
                                  "camera_image_height": camera_image_height,
                                  "camera_fov": camera_fov,
                                  "dataset_path": "replayer_waymo_pickle.pl",
                                  "dataset_name": "waymo",
                                  "dataset_frame_interval": 75,
                                  "csv_log_file_name": "replayer_test.csv",
                              },
                              None,
                              expected_ttd_stream,
                              verbose=verbose)


def test_replay_waymo_pl_test_camera():
    start(replay_waymo_pl_test_camera(verbose))


def test_replay_waymo_pl_test_gt():
    start(replay_waymo_pl_test_gt(verbose))


def test_replay_waymo_pl_test_ttd():
    start(replay_waymo_pl_test_ttd(verbose))


if __name__ == '__main__':
    test_replay_waymo_pl_test_camera()
    test_replay_waymo_pl_test_gt()
    test_replay_waymo_pl_test_ttd()
