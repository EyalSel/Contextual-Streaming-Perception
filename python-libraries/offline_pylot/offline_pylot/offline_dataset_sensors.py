import json
import os
import signal
import time
from pathlib import Path

import cv2
import erdos
from pylot.perception.camera_frame import CameraFrame
from pylot.perception.detection.obstacle import Obstacle
from pylot.perception.detection.utils import BoundingBox2D
from pylot.perception.messages import FrameMessage, ObstaclesMessage

from .dataset_replay import (OfflineArgoverseSensorCuboid,
                             OfflineArgoverseSensorJPG, OfflineCarlaSensor,
                             OfflineWaymoSensorV1_1)


class OfflineDatasetSensor(erdos.Operator):
    """
    parent_pid is used to interrupt the main erdos process once all of the
    frames have been processed. Otherwise the program will hang indefinitely.
    """
    def __init__(self, backwards_notify_stream, camera_stream,
                 ground_obstacles_stream, time_to_decision_stream,
                 camera_setup, parent_pid, flags):
        self._csv_logger = erdos.utils.setup_csv_logging(
            self.config.name + '-csv', self.config.csv_log_file_name)
        self._flags = flags
        self.DATASET_PATH = Path(self._flags.dataset_path)
        self._camera_stream = camera_stream
        self._backwards_notify_stream = backwards_notify_stream
        self._ground_obstacles_stream = ground_obstacles_stream
        self._time_to_decision_stream = time_to_decision_stream
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._parent_pid = parent_pid
        erdos.add_watermark_callback([self._backwards_notify_stream], [],
                                     self.on_watermark)

        self._camera_setup = camera_setup
        self.output_width = camera_setup.width
        self.output_height = camera_setup.height
        assert camera_setup.camera_type == "sensor.camera.rgb", \
            camera_setup.camera_type

        replayer_class = {
            "waymo": OfflineWaymoSensorV1_1,
            "carla": OfflineCarlaSensor,
            "argo-jpg": OfflineArgoverseSensorJPG,
            "argo-cuboid": OfflineArgoverseSensorCuboid
        }[self._flags.dataset_name]

        self.replayer = replayer_class(self.DATASET_PATH)

        self.num_frames = self.replayer.total_num_frames()
        self.num_frames_sent = 0
        self.frame_interval = self._flags.dataset_frame_interval

    def _child_index_dir(self, data_path):
        """
        Called during initialization, allows child class to examine the
        given dataset path and set any class variables it needs.

        Returns the number of frames in this segment
        """
        raise NotImplementedError("To be implemented by child class")

    def _child_get_frame(self, frame_index):
        """
        Return dictionary:
            - "center_camera_feed":
                Image as numpy array BGR format, np.uint8, HWC order
            - "obstacles": list of dictionaries
                - "bbox": [xmn,xmx,ymn,ymx]
                - "label"
                - "id": non-negative integer, or -1 if not supported
        """
        raise NotImplementedError("To be implemented by child class")

    @erdos.profile_method()
    def get_messages(self, index, timestamp):
        self._logger.debug("@{}: {} releasing sensor index {}".format(
            timestamp, self.config.name, index))
        frame_contents = self.replayer.get_frame(index)

        # Extract img to get dimensions for bbox label resizing
        img_h, img_w, c = frame_contents["center_camera_feed"].shape
        ratio_h = img_h / self.output_height
        ratio_w = img_w / self.output_width

        # resize obstacles and put into object
        def process_obst(obst_info):
            # resize bbox label given image label
            xmn, xmx, ymn, ymx = obst_info["bbox"]
            xmn, xmx = xmn / ratio_w, xmx / ratio_w
            ymn, ymx = ymn / ratio_h, ymx / ratio_h

            assert xmn < xmx and ymn < ymx and xmn >= 0 and ymn >= 0,\
                obst_info["bbox"]
            return Obstacle(BoundingBox2D(xmn, xmx, ymn, ymx),
                            confidence=1.0,
                            label=obst_info["label"],
                            id=obst_info["id"])

        def read_json(path):
            with open(path, 'r') as f:
                return json.load(f)

        ground_obstacles = [
            process_obst(o) for o in frame_contents["obstacles"]
        ]
        obst_message = ObstaclesMessage(timestamp, ground_obstacles)

        # resize img
        resized_img = cv2.resize(frame_contents["center_camera_feed"],
                                 dsize=(self.output_width, self.output_height),
                                 interpolation=cv2.INTER_CUBIC)
        # if there's just one channel cv2 resize returns no channels...
        if c == 1:
            resized_img = resized_img.reshape(*resized_img.shape, c)
        camera_frame = CameraFrame(resized_img, "BGR", self._camera_setup)
        camera_msg = FrameMessage(timestamp, camera_frame)
        return camera_msg, obst_message

    @staticmethod
    def connect(backwards_notify_stream):
        camera_stream = erdos.WriteStream()
        ground_obstacles_stream = erdos.WriteStream()
        time_to_decision_stream = erdos.WriteStream()
        return [
            camera_stream, ground_obstacles_stream, time_to_decision_stream
        ]

    def send_frame(self):
        print("Sending frame {}".format(self.num_frames_sent))
        if self.num_frames_sent >= self.num_frames:
            print("Trying to send more frames than there are!")
            return
        # Eyal: Not 100% clear on why we start with a non-zero timestamp
        timestamp = erdos.Timestamp(coordinates=[(self.num_frames_sent + 1) *
                                                 self.frame_interval])
        camera_msg, obst_message = self.get_messages(self.num_frames_sent,
                                                     timestamp)
        # 10s time to decision
        ttd_msg = erdos.Message(timestamp, (100000, 100000))
        self._camera_stream.send(camera_msg)
        self._camera_stream.send(erdos.WatermarkMessage(timestamp))
        self._ground_obstacles_stream.send(obst_message)
        self._ground_obstacles_stream.send(erdos.WatermarkMessage(timestamp))
        self._time_to_decision_stream.send(ttd_msg)
        self._time_to_decision_stream.send(erdos.WatermarkMessage(timestamp))
        self.num_frames_sent += 1
        if self.num_frames_sent == self.num_frames:
            print("Sending top frame watermark")
            top_watermark = erdos.WatermarkMessage(
                erdos.Timestamp(is_top=True))
            self._camera_stream.send(top_watermark)
            self._ground_obstacles_stream.send(top_watermark)
            self._time_to_decision_stream.send(top_watermark)

    def on_watermark(self, timestamp):
        if self.num_frames_sent < self.num_frames:
            self.send_frame()
        elif (timestamp.is_top or timestamp.coordinates[0] >=
              self.num_frames_sent * self.frame_interval):
            time.sleep(2)  # let the last frame go through the pipeline
            print(">>>>> DONE <<<<<")
            if self._parent_pid is None:
                print("Offline dataset sensor finished or got a top watermark "
                      "but received a parent ID = None and therefore can't "
                      "stop the system.")
            os.kill(self._parent_pid, signal.SIGINT)

    def run(self):
        from .utils import prepend_line_to_file
        while not Path(self._flags.csv_log_file_name).exists():
            print("Waiting for {} to get created...".format(
                self._flags.csv_log_file_name))
            time.sleep(0.5)
        prepend_line_to_file(self._flags.csv_log_file_name,
                             "log_ts,simulator_ts,operator,extra_info,value")
        # Some operators (e.g. sequencer) do some batching in the sense that
        # they hold onto some messages and then release them all at once.
        # In the case of the sequencer 2 frames can be held.
        max_in_flight = self._flags.num_inflight_frames
        print(f"Offline dataset sensor releasing {max_in_flight} frames to "
              "process inflight at once. If the pipeline hangs it means this "
              "value is too low.")
        for i in range(max_in_flight):
            self.send_frame()
