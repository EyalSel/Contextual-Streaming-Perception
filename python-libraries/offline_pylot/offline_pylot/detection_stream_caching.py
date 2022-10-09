import json

import erdos

import jsonlines

import numpy as np

from pylot.perception.detection.obstacle import Obstacle
from pylot.perception.detection.utils import BoundingBox2D
from pylot.perception.messages import ObstaclesMessage


class ObstacleStreamReplayer(erdos.Operator):
    """
    Uses the center_camera_stream as a clocking mechanism to know when to send
    the next obstacles.
    """
    def __init__(self, center_camera_stream, obstacle_stream, flags):
        self._flags = flags
        self._file_name = self._flags.cache_obstacle_detection_source_path
        self._obstacle_stream = obstacle_stream
        self._center_camera_stream = center_camera_stream
        erdos.add_watermark_callback([self._center_camera_stream], [],
                                     self.on_watermark)
        with jsonlines.open(self._file_name) as reader:
            self.data = list(reader)
        assert any([x["content_type"] == "run_metadata" for x in self.data])
        metadata = self.data.pop(0)
        assert metadata["content_type"] == "run_metadata", metadata
        assert all(
            [x["content_type"] == "frame_prediction" for x in self.data])
        print(("Detection cache reader found {} entries with the metadata {} "
               "in {}").format(len(self.data) - 1, metadata, self._file_name))
        self.num_obstacles_sent = 0

    @staticmethod
    def connect(center_camera_stream):
        """Connects the operator to other streams.

        Returns:
            :py:class:`erdos.WriteStream`: Stream on which the operator sends
            :py:class:`~pylot.perception.messages.ObstaclesMessage` messages.
        """
        obstacles_stream = erdos.WriteStream()
        return [obstacles_stream]

    def on_watermark(self, watermark_ts):
        if watermark_ts.is_top:
            self._obstacle_stream.send(
                erdos.WatermarkMessage(erdos.Timestamp(is_top=True)))
            return
        assert self.num_obstacles_sent < len(self.data), (
            "The dataset replayer is sending more watermarks than there are "
            "cached obstacle frames ({})").format(len(self.data))
        entry = self.data[self.num_obstacles_sent]
        watermark_ts = watermark_ts.coordinates[0]
        assert watermark_ts == entry["timestamp"], (
            "the watermark sent from the dataset replayer and the next "
            "watermark on the obstacle replayer don't match: {} vs {}").format(
                watermark_ts, entry["timestamp"])
        timestamp = erdos.Timestamp(coordinates=[entry["timestamp"]])

        def obstacle_form_summary(summary):
            return Obstacle(BoundingBox2D(float(summary["bbox"]["xmn"]),
                                          float(summary["bbox"]["xmx"]),
                                          float(summary["bbox"]["ymn"]),
                                          float(summary["bbox"]["ymx"])),
                            summary["confidence"],
                            summary["label"],
                            id=summary["id"],
                            detailed_label=summary["detailed_label"])

        obstacles = [obstacle_form_summary(s) for s in entry["obstacles"]]
        obstacles = list(
            filter(
                lambda o: o.confidence >= self._flags.
                obstacle_detection_min_score_threshold, obstacles))
        msg = ObstaclesMessage(timestamp, obstacles, entry["runtime"])
        self._obstacle_stream.send(msg)
        self._obstacle_stream.send(erdos.WatermarkMessage(timestamp))

        self.num_obstacles_sent += 1


class ObstacleStreamLogger(erdos.Operator):
    """Logs bounding boxes of obstacles to files.

    Args:
        obstacles_stream (:py:class:`erdos.ReadStream`): The stream on which
            :py:class:`~pylot.perception.messages.ObstaclesMessage` are
            received.
        flags (absl.flags): Object to be used to access absl flags.

    Attributes:
        _logger (:obj:`logging.Logger`): Instance to be used to log messages.
        _flags (absl.flags): Object to be used to access absl flags.
    """
    def __init__(self, obstacles_stream: erdos.ReadStream, confirm_stream,
                 flags):
        obstacles_stream.add_callback(self.on_obstacles_msg)
        erdos.add_watermark_callback([obstacles_stream], [confirm_stream],
                                     self.on_watermark)
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._flags = flags
        self._file_name = self._flags.cache_obstacle_detection_destination_path
        metadata = {
            "content_type": "run_metadata",
            "model_names": self._flags.obstacle_detection_model_names,
            "camera_image_height": self._flags.camera_image_height,
            "camera_image_width": self._flags.camera_image_width
        }
        with jsonlines.open(self._file_name,
                            mode='w',
                            dumps=np_compatible_dumps) as writer:
            writer.write(metadata)
            # detection configurations, run name
        self.bookkeeping = {}

    @staticmethod
    def connect(obstacles_stream: erdos.ReadStream):
        """Connects the operator to other streams.

        The operator receives an obstacles stream and does not write to any
        output stream.
        """
        confirm_stream = erdos.WriteStream()
        return [confirm_stream]

    def _summarize_obstacle(self, obstacle):
        mn = obstacle.bounding_box_2D.get_min_point()
        mx = obstacle.bounding_box_2D.get_max_point()
        return {
            "label": obstacle.label,
            "detailed_label": obstacle.detailed_label,
            "id": obstacle.id,
            "bbox": {
                "xmn": float(mn.x),
                "ymn": float(mn.y),
                "xmx": float(mx.x),
                "ymx": float(mx.y)
            },
            "confidence": float(obstacle.confidence)
        }

    def on_obstacles_msg(self, msg: erdos.Message):
        """Logs bounding boxes to files.

        Invoked upon the receipt of a msg on the obstacles stream.

        Args:
            msg (:py:class:`~pylot.perception.messages.ObstaclesMessage`):
                Received message.
        """
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self.config.name))
        obstacles = [self._summarize_obstacle(o) for o in msg.obstacles]
        assert len(msg.timestamp.coordinates) == 1
        frame_predictions = {
            "content_type": "frame_prediction",
            "runtime": float(msg.runtime),
            "obstacles": obstacles,
            "timestamp": msg.timestamp.coordinates[0]
        }
        self.bookkeeping[msg.timestamp] = frame_predictions

    def on_watermark(self, timestamp, confirm_stream):
        if timestamp.is_top:
            return
        frame_predictions = self.bookkeeping[timestamp]
        del self.bookkeeping[timestamp]
        # Write the bounding boxes.
        with jsonlines.open(self._file_name,
                            mode='a',
                            dumps=np_compatible_dumps) as writer:
            writer.write(frame_predictions)


# https://stackoverflow.com/a/47626762/1546071
# https://stackoverflow.com/a/12570040/1546071
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if type(obj).__module__ == np.__name__:
            return obj.item()
        return json.JSONEncoder.default(self, obj)


def np_compatible_dumps(obj):
    """
    For the case where the values in the stream are numpy objects
    (e.g. np.int8) this function automatically converts to native types.
    """
    return json.dumps(obj, cls=NumpyEncoder)
