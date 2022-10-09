import os
import signal

from absl import app, flags

from offline_pylot.coco_detection_eval_operator import \
    CocoDetectionEvalOperator
from offline_pylot.detection_stream_caching import ObstacleStreamLogger, \
    ObstacleStreamReplayer
from offline_pylot.offline_dataset_sensors import OfflineDatasetSensor
from offline_pylot.prediction_frame_sequencing import \
    PredictionFrameSequencer
from offline_pylot.utils import join_streams

import erdos

import pylot
import pylot.flags
import pylot.operator_creator
from pylot.drivers.sensor_setup import RGBCameraSetup

# import logging

# erdos.logger.setLevel(logging.DEBUG)

flags.DEFINE_string('dataset_path', None, 'Path to offline dataset')
flags.DEFINE_string('dataset_name', None,
                    'The name of the dataset, to know which replayer to use')
flags.DEFINE_string('dataset_label_map', None,
                    'Name of label map used in dataset')
flags.DEFINE_integer('dataset_frame_interval', None,
                     'ms interval between frames, used for timely mAP')
# Flags related to caching.
flags.DEFINE_string("cache_obstacle_detection_source_path", None,
                    ('Either this or obstacle_detection should be turned on. '
                     'To turn this on, please specify a path.'))
flags.DEFINE_string(
    "cache_obstacle_detection_destination_path", None,
    'If specified it saves the obstacle stream to the specified path')
# Flags related to mAP evaluation.
flags.DEFINE_integer('coco_detection_eval_lookback', None,
                     'Number of images from the past to aggregate')
flags.DEFINE_integer('coco_detection_eval_freq', None,
                     'Give rolling coco eval every this many images')
flags.DEFINE_enum('det_sequencer_policy', 'infinite',
                  ['infinite', 'eager', 'wait', 'tail-aware'],
                  'Policy used by sequencer. Leave None to disable sequencer')
flags.DEFINE_integer(
    'num_inflight_frames',
    default=1,
    help='The number of frames being processed in the pipeline concurrently')
# Flags related to logging.
flags.DEFINE_bool('log_rgb_camera', False,
                  'True to enable center camera RGB logging')
flags.DEFINE_bool('log_perfect_obstacles', False,
                  'True to enable obstacle perfect bounding box logging')
flags.DEFINE_bool('log_obstacles', False,
                  'True to enable obstacle bounding box logging')

for flag in [
        "dataset_path", "dataset_name", "dataset_label_map",
        "dataset_frame_interval"
]:
    flags.mark_flag_as_required(flag)


def detection_source_checker(flags_dict):
    return (flags_dict["obstacle_detection_model_paths"] !=
            flags_dict["cache_obstacle_detection_source_path"])


def component_and_eval_checker(flags_dict):
    return not ((flags_dict["evaluate_obstacle_detection"]
                 and not flags_dict["obstacle_detection"]) or
                (flags_dict["evaluate_obstacle_tracking"]
                 and not flags_dict["obstacle_tracking"]))


def coco_eval_flags_checker(flags_dict):
    coco_lookback = flags_dict["coco_detection_eval_lookback"]
    coco_freq = flags_dict["coco_detection_eval_freq"]
    synced_none_status = (coco_lookback is None and coco_freq is None) or \
        (coco_lookback is not None and coco_freq is not None)
    both_positive = coco_freq > 0 and coco_lookback > 0
    return synced_none_status and (coco_lookback is None or both_positive)


flags.register_multi_flags_validator(
    [
        "evaluate_obstacle_detection", "obstacle_detection",
        "evaluate_obstacle_tracking", "obstacle_tracking"
    ],
    component_and_eval_checker,
    message=("A component's eval flag cannot be turned on "
             "if the component itself is not turned on."))

flags.register_multi_flags_validator(
    ["obstacle_detection_model_paths", "cache_obstacle_detection_source_path"],
    detection_source_checker,
    message=("Exactly one of obstacle_detection_model_paths or"
             " cache_obstacle_detection_source_path should be on."))

flags.register_multi_flags_validator(
    ["coco_detection_eval_lookback", "coco_detection_eval_freq"],
    coco_eval_flags_checker,
    message=("Both flags should be None or not None together. "
             "If they are not None then they should be positive."))

FLAGS = flags.FLAGS

# copied from pylot.py
CENTER_CAMERA_LOCATION = pylot.utils.Location(1.3, 0.0, 1.8)


def add_dataset_replayer_operator():
    name = "offline-carla-camera"
    transform = pylot.utils.Transform(CENTER_CAMERA_LOCATION,
                                      pylot.utils.Rotation(pitch=-15))
    offline_camera_setup = RGBCameraSetup(name,
                                          FLAGS.camera_image_width,
                                          FLAGS.camera_image_height,
                                          transform,
                                          fov=FLAGS.camera_fov)
    op_config = erdos.OperatorConfig(name=offline_camera_setup.get_name() +
                                     '_operator',
                                     flow_watermarks=False,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    backwards_notify_stream = erdos.LoopStream()
    center_camera_stream, perfect_obstacles_stream, time_to_decision_stream = \
        erdos.connect(OfflineDatasetSensor, op_config,
                      [backwards_notify_stream], offline_camera_setup,
                      os.getpid(), FLAGS)
    return center_camera_stream, offline_camera_setup, \
        perfect_obstacles_stream, time_to_decision_stream, \
        backwards_notify_stream


def add_detection_replayer_operator(camera_stream):
    op_config = erdos.OperatorConfig(name="detection_cache_replay_operator",
                                     flow_watermarks=False,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    obstacle_streams = erdos.connect(ObstacleStreamReplayer, op_config,
                                     [camera_stream], FLAGS)
    return obstacle_streams[0]


def add_obstacle_detection(camera_stream, time_to_decision_stream):
    if FLAGS.cache_obstacle_detection_source_path is not None:
        obstacle_stream = add_detection_replayer_operator(camera_stream)
    else:
        if any('efficientdet' in model
               for model in FLAGS.obstacle_detection_model_names):
            obstacles_streams = \
                pylot.operator_creator.add_efficientdet_obstacle_detection(
                    camera_stream, time_to_decision_stream)
            obstacle_stream = obstacles_streams[0]
        else:
            obstacles_streams = pylot.operator_creator.add_obstacle_detection(
                camera_stream, time_to_decision_stream)
            obstacle_stream = obstacles_streams[0]
    return obstacle_stream


def add_obstacle_detection_eval(obstacle_stream, perfect_obstacles_stream):
    if FLAGS.coco_detection_eval_lookback is None:
        return pylot.operator_creator.add_detection_evaluation(
            obstacle_stream, perfect_obstacles_stream)
    else:
        op_config = erdos.OperatorConfig(
            name="coco_detection_eval_operator_timely",
            flow_watermarks=True,
            log_file_name=FLAGS.log_file_name,
            csv_log_file_name=FLAGS.csv_log_file_name,
            profile_file_name=FLAGS.profile_file_name)
        (det_timely_eval_finished_indicator_stream, ) = erdos.connect(
            CocoDetectionEvalOperator, op_config,
            [obstacle_stream, perfect_obstacles_stream], True, "ceil",
            FLAGS.dataset_frame_interval, FLAGS)
        op_config = erdos.OperatorConfig(
            name="coco_detection_eval_operator_sync",
            flow_watermarks=True,
            log_file_name=FLAGS.log_file_name,
            csv_log_file_name=FLAGS.csv_log_file_name,
            profile_file_name=FLAGS.profile_file_name)
        (det_sync_eval_finished_indicator_stream, ) = erdos.connect(
            CocoDetectionEvalOperator, op_config,
            [obstacle_stream, perfect_obstacles_stream], False, "ceil",
            FLAGS.dataset_frame_interval, FLAGS)
        return join_streams("det_eval_indicator_join", [
            det_timely_eval_finished_indicator_stream,
            det_sync_eval_finished_indicator_stream
        ])


def add_obstacle_tracking_eval(obstacles_wo_history_tracking_stream,
                               perfect_obstacles_stream):
    track_timely_ceil_eval_finished_indicator_stream = \
        pylot.operator_creator.add_tracking_evaluation(
            obstacles_wo_history_tracking_stream,
            perfect_obstacles_stream,
            evaluate_timely=True,
            matching_policy='ceil',
            frame_gap=FLAGS.dataset_frame_interval,
            name="tracking_eval_operator_timely_ceil")
    track_timely_round_eval_finished_indicator_stream = \
        pylot.operator_creator.add_tracking_evaluation(
            obstacles_wo_history_tracking_stream,
            perfect_obstacles_stream,
            evaluate_timely=True,
            matching_policy='round',
            frame_gap=FLAGS.dataset_frame_interval,
            name="tracking_eval_operator_timely_round")
    track_sync_eval_finished_indicator_stream = \
        pylot.operator_creator.add_tracking_evaluation(
            obstacles_wo_history_tracking_stream,
            perfect_obstacles_stream,
            evaluate_timely=False,
            frame_gap=FLAGS.dataset_frame_interval,
            name="tracking_eval_operator_sync")
    track_eval_finished_indicator_stream = join_streams(
        "track_eval_indicator_join", [
            track_timely_ceil_eval_finished_indicator_stream,
            track_timely_round_eval_finished_indicator_stream,
            track_sync_eval_finished_indicator_stream
        ])
    return track_eval_finished_indicator_stream


def add_obstacle_tracking(obstacle_stream, camera_stream, camera_setup,
                          time_to_decision_stream):
    if FLAGS.tracker_type == 'center_track':
        obstacles_wo_history_tracking_stream = \
            pylot.operator_creator.add_center_track_tracking(
                camera_stream, camera_setup)
    elif FLAGS.tracker_type == 'qd_track':
        obstacles_wo_history_tracking_stream = \
            pylot.operator_creator.add_qd_track_tracking(
                camera_stream, camera_setup)
    else:
        obstacles_wo_history_tracking_stream = \
            pylot.operator_creator.add_obstacle_tracking(
                obstacle_stream, camera_stream, time_to_decision_stream)

    # the component below converts 2d to 3d tracking
    # obstacles_tracking_stream = \
    #     pylot.operator_creator.add_obstacle_location_history(
    #         obstacles_wo_history_tracking_stream, depth_stream,
    #         pose_stream, center_camera_setup)
    return obstacles_wo_history_tracking_stream


def driver():
    # Offline dataset replayer.
    (center_camera_stream, center_camera_setup, perfect_obstacles_stream,
     time_to_decision_stream, backwards_notify_stream) = \
         add_dataset_replayer_operator()

    if FLAGS.log_rgb_camera:
        log_camera_indicator_stream = \
            pylot.operator_creator.add_camera_logging(
                center_camera_stream,
                'center_camera_logger_operator',
                'center-')

    if FLAGS.log_perfect_obstacles:
        pylot.operator_creator.add_bounding_box_logging(
            perfect_obstacles_stream)

    # Detection from online inference or from cached file.
    if FLAGS.obstacle_detection:
        obstacle_stream = add_obstacle_detection(center_camera_stream,
                                                 time_to_decision_stream)
        if FLAGS.log_obstacles:
            pylot.operator_creator.add_bounding_box_logging(obstacle_stream)

    # Add detection sequencer operator.
    op_config = erdos.OperatorConfig(name="detection_frame_sequencer",
                                     flow_watermarks=False,
                                     log_file_name=FLAGS.log_file_name,
                                     csv_log_file_name=FLAGS.csv_log_file_name,
                                     profile_file_name=FLAGS.profile_file_name)
    (sequenced_obstacle_stream, ) = erdos.connect(PredictionFrameSequencer,
                                                  op_config, [obstacle_stream],
                                                  FLAGS)

    if FLAGS.evaluate_obstacle_detection:
        det_eval_finished_indicator_stream = add_obstacle_detection_eval(
            sequenced_obstacle_stream, perfect_obstacles_stream)

    if FLAGS.obstacle_tracking:
        obstacles_wo_history_tracking_stream = add_obstacle_tracking(
            sequenced_obstacle_stream, center_camera_stream,
            center_camera_setup, time_to_decision_stream)
        if FLAGS.evaluate_obstacle_tracking:
            track_eval_finished_indicator_stream = add_obstacle_tracking_eval(
                obstacles_wo_history_tracking_stream, perfect_obstacles_stream)

    # Decide which stream to cache.
    if FLAGS.cache_obstacle_detection_destination_path is not None:
        if FLAGS.obstacle_tracking:
            stream_to_cache = obstacles_wo_history_tracking_stream
            print("Caching obstacles_wo_history_tracking_stream")
        else:
            stream_to_cache = sequenced_obstacle_stream
            print("Caching sequenced_obstacle_stream")
        op_config = erdos.OperatorConfig(
            name="detection_cache_operator",
            flow_watermarks=False,
            log_file_name=FLAGS.log_file_name,
            csv_log_file_name=FLAGS.csv_log_file_name,
            profile_file_name=FLAGS.profile_file_name)
        erdos.connect(ObstacleStreamLogger, op_config, [stream_to_cache],
                      FLAGS)

    # Logic to figure out which component clocks the dataset replayer.
    if FLAGS.obstacle_tracking:
        if FLAGS.evaluate_obstacle_tracking:
            backwards_notify_stream.set(track_eval_finished_indicator_stream)
            print("Tracking eval is clocking")
        else:
            backwards_notify_stream.set(obstacles_wo_history_tracking_stream)
            print("Tracking operator is clocking")
    elif FLAGS.obstacle_detection:
        if FLAGS.cache_obstacle_detection_source_path is not None:
            if FLAGS.log_rgb_camera:
                backwards_notify_stream.set(log_camera_indicator_stream)
                print("Camera logger is clocking")
            else:
                if FLAGS.evaluate_obstacle_detection:
                    backwards_notify_stream.set(
                        det_eval_finished_indicator_stream)
                    print("Detection eval is clocking")
                else:
                    backwards_notify_stream.set(sequenced_obstacle_stream)
                    print("Detection oeprator is clocking")
        # See same if elif sequence above, the if check below is meaningless.
        # We're forced to check caching first then the model because of flag
        # default issues.
        elif FLAGS.obstacle_detection_model_paths:
            if FLAGS.log_rgb_camera:
                backwards_notify_stream.set(log_camera_indicator_stream)
                print("Camera logger is clocking")
            elif FLAGS.evaluate_obstacle_detection:
                backwards_notify_stream.set(det_eval_finished_indicator_stream)
                print("Detection eval is clocking")
            else:
                backwards_notify_stream.set(obstacle_stream)
                print("Detection model is clocking")
        else:
            raise RuntimeError("Flag check failed: exactly one of "
                               "cache_obstacle_detection_source_path or "
                               "obstacle_detection_model_paths need to be on")
    else:
        if FLAGS.log_rgb_camera:
            backwards_notify_stream.set(log_camera_indicator_stream)
            print("Camera logger is clocking")
        else:
            raise RuntimeError(
                "The obstacle_detection flag should be turned on...")

    node_handle = erdos.run_async()
    return node_handle


def shutdown(sig, frame):
    raise KeyboardInterrupt


def shutdown_pylot(node_handle):
    node_handle.shutdown()


def main(args):
    node_handle = None
    try:
        node_handle = driver()
        signal.signal(signal.SIGINT, shutdown)
        node_handle.wait()
    except KeyboardInterrupt:
        shutdown_pylot(node_handle)
    except Exception:
        shutdown_pylot(node_handle)
        raise


if __name__ == "__main__":
    app.run(main)
