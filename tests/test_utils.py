import os
import signal
import sys
from pprint import pprint

import erdos

from pylot.perception.detection.obstacle import Obstacle
from pylot.perception.detection.utils import BoundingBox2D
from pylot.perception.messages import ObstaclesMessage

from termcolor import cprint


class IdentityPrintStream(erdos.Operator):
    def __init__(self, input_stream, output_stream):
        self.output_stream = output_stream
        erdos.add_watermark_callback([input_stream], [], self.on_watermark)
        input_stream.add_callback(self.on_msg)

    def on_msg(self, msg):
        print(f"msg on {self.config.name} stream: {msg}")
        self.output_stream.send(msg)

    def on_watermark(self, timestamp):
        print(f"watermark {self.config.name} stream: {timestamp}")

    @staticmethod
    def connect(input_stream):
        output_stream = erdos.WriteStream()
        return [output_stream]


def identity_print_stream(name, input_stream):
    return erdos.connect(IdentityPrintStream, erdos.OperatorConfig(name=name),
                         [input_stream])[0]


def split_stream(name, input_stream, n):
    class StreamSplitter(erdos.Operator):
        def __init__(self, input_stream, *output_streams):
            self.output_streams = output_streams
            erdos.add_watermark_callback([input_stream], output_streams,
                                         self.on_watermark)
            input_stream.add_callback(self.on_msg)

        @staticmethod
        def connect(input_stream):
            output_streams = [erdos.WriteStream() for i in range(n)]
            return output_streams

        def on_watermark(self, *output_streams):
            pass

        def on_msg(self, msg):
            for stream in self.output_streams:
                stream.send(msg)

    return erdos.connect(StreamSplitter, erdos.OperatorConfig(name=name),
                         [input_stream])


class CustomSequenceSender(erdos.Operator):
    def __init__(self, output_stream, output_stream_contents):
        self._output_stream_contents = output_stream_contents
        self.output_stream = output_stream

    @staticmethod
    def connect():
        output_stream = erdos.WriteStream()
        return [output_stream]

    def run(self):
        for msg in self._output_stream_contents:
            self.output_stream.send(msg)


def repr_or_str(obj):
    if type(obj).__repr__ is not object.__repr__:
        return repr(obj)
    else:
        return str(obj)


class SequenceVerifier(erdos.Operator):
    def __init__(self, input_stream, finished_indicator_stream,
                 expected_input_stream_contents, parent_pid, verbose, color):
        self.expected_input_stream_contents = expected_input_stream_contents
        self.finished_indicator_stream = finished_indicator_stream
        self.input_stream = input_stream
        self.parent_pid = parent_pid
        self.verbose = verbose
        self.color = color
        input_stream.add_callback(self.on_msg)
        erdos.add_watermark_callback([self.input_stream], [],
                                     self.on_watermark)
        self.next_index = 0
        cprint(self.config.name, self.color)

    @staticmethod
    def connect(input_stream):
        finished_indicator_stream = erdos.WriteStream()
        return [finished_indicator_stream]

    def increment_and_check_if_done(self):
        self.next_index += 1
        if self.next_index == len(self.expected_input_stream_contents):
            cprint("Finished all expected stream values", self.color)
        elif self.verbose:
            cprint(
                "not done, next is {}".format(
                    self.expected_input_stream_contents[self.next_index]),
                self.color)

    def kill_if_not_true(self, condition, message, timestamp):
        if not condition:
            attrs = ['bold'] if self.verbose else None
            cprint(f"{self.config.name}: {message}", self.color, attrs=attrs)
            # send a non-watermark message, indicating there's an error.
            # Data payload contains node name identifying the test.
            self.finished_indicator_stream.send(
                erdos.Message(timestamp, self.config.name))

    def on_watermark(self, timestamp):
        self.kill_if_not_true(
            self.next_index < len(self.expected_input_stream_contents),
            (f"Received more messages than expected: Watermark {timestamp}"),
            timestamp)
        expected_msg = self.expected_input_stream_contents[self.next_index]
        self.kill_if_not_true(
            isinstance(expected_msg, erdos.WatermarkMessage),
            (f"Expected {expected_msg}, but received a watermark with "
             f"timestamp {timestamp}"), timestamp)
        expected_timestamp = expected_msg.timestamp
        self.kill_if_not_true(
            expected_timestamp == timestamp,
            (f"Expected watermark timestamp {expected_timestamp} but "
             f"received {timestamp}"), timestamp)
        if self.verbose:
            cprint(f"Successfully matched Watermark {timestamp}", self.color)
        self.increment_and_check_if_done()

    def on_msg(self, msg):
        """
        Using repr for comparison because implement __eq__ has to be done
        recursively and it's too much work.
        """
        self.kill_if_not_true(
            self.next_index < len(self.expected_input_stream_contents),
            (f"Received more messages than expected: message {msg}"),
            msg.timestamp)
        expected_msg = self.expected_input_stream_contents[self.next_index]
        self.kill_if_not_true(
            repr_or_str(msg) == repr_or_str(expected_msg),
            (f"expected {repr_or_str(expected_msg)}, "
             f"but received {repr_or_str(msg)}"), msg.timestamp)
        if self.verbose:
            cprint((f"Successfully matched {repr_or_str(msg)} "
                    f"with {repr_or_str(expected_msg)}"), self.color)
        self.increment_and_check_if_done()


class CustomFlags:
    def __init__(self, custom_flags_dict):
        for k, v in custom_flags_dict.items():
            setattr(self, k, v)


def get_dummy_obstacle_message(ts, runtime, id_ts=None, confidence=0.0):
    """
    id_ts is used to tell apart obstacles that belong to different frames
    (i.e. if detection latency is long, ts would be higher than id_ts)
    """
    obstacles = [
        Obstacle(
            BoundingBox2D(0.0, 1.0, 0.0,
                          1.0),  # bbox is float in waymo whatever reasons
            confidence,
            "vehicle",  # gets filtered by the tracker for some reason
            id=id_ts or ts,
            detailed_label=str(runtime)  # also for uniqueness
        )
    ]
    return ObstaclesMessage(erdos.Timestamp(coordinates=[ts]),
                            obstacles,
                            runtime=runtime)


def single_operator_graph(operator):
    def fn(input_stream_content, expected_output_stream, FLAGS, op_config,
           verbose):
        (input_stream, ) = erdos.connect(CustomSequenceSender,
                                         op_config("custom_sequence_sender"),
                                         [], input_stream_content)

        (output_stream, ) = erdos.connect(
            operator, op_config("detection_frame_sequencer", False),
            [input_stream], FLAGS)

        (stream_for_tracker_verify_finish, ) = erdos.connect(
            SequenceVerifier, op_config("sequence_verifier-obstacle_stream"),
            [output_stream], expected_output_stream, os.getpid(), verbose,
            "red")

        extract_stream = erdos.ExtractStream(stream_for_tracker_verify_finish)
        node_handle = erdos.run_async()
        return node_handle, extract_stream

    return fn


def make_operator_test(test_name,
                       operator_graph,
                       extra_flags,
                       input_stream_content,
                       expected_output_stream,
                       verbose=False):
    """
    The operator_graph is a function that takes the following arguments:
        input_stream_content: forwarded from arg of same name of this function
        expected_output_stream: same as above
        FLAGS: all the flags used by the graph
        op_config: A function that takes operator name and returns the configs
        verbose: forwarded from arg of same nam eof this function
    """
    def driver():
        if verbose:
            print("input_stream_content")
            pprint(input_stream_content)
            print("expected_output_stream")
            pprint(expected_output_stream)

        FLAGS = CustomFlags({
            **extra_flags,
            "log_file_name": f"{test_name}_test.log",
            "csv_log_file_name": f"{test_name}_test.csv",
            "profile_file_name": f"{test_name}_test.json",
        })

        def op_config(name, flow_watermarks=True):
            return erdos.OperatorConfig(
                name=test_name + "-" + name,
                flow_watermarks=flow_watermarks,
                log_file_name=FLAGS.log_file_name,
                csv_log_file_name=FLAGS.csv_log_file_name,
                profile_file_name=FLAGS.profile_file_name)

        return operator_graph(input_stream_content, expected_output_stream,
                              FLAGS, op_config, verbose)

    return driver


def shutdown(sig, frame):
    raise KeyboardInterrupt


def shutdown_pylot(node_handle):
    node_handle.shutdown()


def start(test_driver):
    """
    Uses an ExtractStream to tell if the test succeeded. If no message is sent
    then the test is assumed to be succesful. Otherwise, the test is assumed
    to have failed and sys.exit is called.
    """
    node_handle = None
    success = False
    try:
        node_handle, finish_stream = test_driver()
        signal.signal(signal.SIGINT, shutdown)
        while True:
            msg = finish_stream.read()
            if msg.data is not None:
                print(f"Received test failed from {msg.data}")
                break
            if msg.is_top:
                success = True
                break
        node_handle.shutdown()
        erdos.reset()
    except KeyboardInterrupt:
        shutdown_pylot(node_handle)
    except Exception:
        shutdown_pylot(node_handle)
        raise
    if not success:
        sys.exit(1)
