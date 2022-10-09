import erdos

from pylot.perception.messages import ObstaclesMessage


class PredictionFrameSequencer(erdos.Operator):
    """
    At a high level, this component takes a stream of inferences per frame and
    simulates how the inference would have executed if only one can run at a
    time. This matters when inference run time is greater than the time gap
    between frames. When that happens this module returns the most recent,
    available prediction from a previous, stale frame.

    This component uses the "det_sequencer_policy" flag in {"eager", "wait",
    "tail-aware"} to choose whether to start running inference on the most
    recent frame straightaway, or sit idle and wait for a more fresh one.

    The component assumes that obstacles_stream has the following properties:
    1. Each message corresponds to a prediction ran on one frame. The timestamp
       of the messages correspond to the frame's "game time".
    2. The messages are in order of increasing frame timestamps (spaced out by
       flags.dataset_frame_interval milliseconds) without skipping.
    3. Each message has a runtime field corresponding to the inference runtime
       in milliseconds.

    The component also assumes that a watermark is sent for each frame
    gametime: frame_gap, 2*frame_gap ...

    The component sends messages in sequenced_obstacles_stream with the
    following properties:
    1. Each message corresponds to one frame. The timestamp of the messages
       correspond to the frame's "game time".
    2. The messages are in order of increasing frame timestamps (spaced out by
       flags.dataset_frame_interval milliseconds) BUT *may* include skipping!
    3. Each message corresponds to a timestamp that the sequencer determined
       the detection component would run inference on, and each skipped
       timestamp (with no corresponding message in the stream) is one that the
       sequencer chose to make the detection component skip. Each message
       contains obstacles resulting from inference done on the frame with the
       corresponding timestamp. The runtime is nonzero.
    4. Either way, a watermark message is sent at every timestamp.

    Note that because of the implementation of sequenced_obstacles_stream a
    batching behavior occurs whereby this module holds onto a few watermark
    messages before releasing them all at once. This can mess with a desirable
    self-clocking behavior in the pipeline if there are less messages in flight
    than what this module batches. The offline dataset sensor sends enough
    inflight messages to overcome this, but this value needs to be watched.
    """
    def __init__(self, obstacles_stream, sequenced_obstacles_stream, flags):
        self._flags = flags
        self._obstacles_stream = obstacles_stream
        erdos.add_watermark_callback([obstacles_stream],
                                     [sequenced_obstacles_stream],
                                     self.on_watermark)
        obstacles_stream.add_callback(self.on_obstacles)
        self._received_predictions = []
        self.frame_gap = self._flags.dataset_frame_interval
        self.policy = self._flags.det_sequencer_policy
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        if self.policy in ["infinite", "eager", "wait", "tail-aware"]:
            self.num_workers = 1
        elif self.policy.endswith("-GPU"):
            self.num_workers = int(self.policy.split("-")[0])
            assert self.num_workers > 1, (
                "if using a multi-GPU policy, the number of GPUS has to be > 1"
            )
        # Contains information about the prediction that's in progress.
        self.ongoing_inference = [None for _ in range(self.num_workers)]
        self.ongoing_inference_end_time = [
            None for _ in range(self.num_workers)
        ]
        self.not_sent_finished_inference = []
        # This is what's updated when inferences finish and what's sent for
        # frames arriving while the current inference is in progress.
        self.last_finished_inference = None
        self.latest_finished_inference_ts = None

    @staticmethod
    def connect(obstacles_stream: erdos.ReadStream):
        """Connects the operator to other streams.

        Returns:
            :py:class:`erdos.WriteStream`: Stream on which the operator sends
            :py:class:`~pylot.perception.messages.ObstaclesMessage` messages.
        """
        sequenced_obstacles_stream = erdos.WriteStream()
        return [sequenced_obstacles_stream]

    def on_obstacles(self, msg):
        assert msg.runtime > 0, "ObstacleMessage doesn't have runtime field"
        prediction = {
            "timestamp": msg.timestamp,
            "runtime": msg.runtime,
            "obstacles": msg.obstacles
        }
        self._received_predictions.append((msg.timestamp, prediction))

    def on_watermark(self, timestamp, sequenced_obstacles_stream):
        if timestamp.is_top:
            # Complete what's running inference, in increasing finish order.
            running_worker_ids = [
                idx for idx, inf in enumerate(self.ongoing_inference)
                if inf is not None
            ]
            increasing_finish_order = sorted(
                running_worker_ids,
                key=lambda idx: self.ongoing_inference_end_time[idx])
            for worker_id in increasing_finish_order:
                self.__complete_inference(sequenced_obstacles_stream,
                                          worker_id)
            self.__send_completed_timestamps(timestamp,
                                             sequenced_obstacles_stream)
            # Send the top watermark.
            sequenced_obstacles_stream.send(erdos.WatermarkMessage(timestamp))
            return
        current_time = timestamp.coordinates[0]
        if self.__get_any_free_worker() is not None:
            # Bootstrap first inference.
            self.__start_inference_on_frame(timestamp, current_time,
                                            self.__get_any_free_worker())
            return
        if self.policy == "infinite":
            # Complete what's running and start new inference.
            self.__complete_inference(sequenced_obstacles_stream, 0)
            self.__start_inference_on_frame(timestamp, current_time, 0)
        elif self.__get_earliest_ending_worker(current_time) is not None:
            # The inference finished running.
            worker_id = self.__get_earliest_ending_worker(current_time)
            previous_inference_end_time = self.ongoing_inference_end_time[
                worker_id]
            self.__complete_inference(sequenced_obstacles_stream, worker_id)
            # If inference time < frame_gap, the waiting is the best policy.
            if (self.last_finished_inference["timestamp"].coordinates[0]
                    == current_time - self.frame_gap or self.policy == "wait"
                    or self.policy.endswith("-GPU")):
                self.__start_inference_on_frame(timestamp, current_time,
                                                worker_id)
            elif self.policy == "eager":
                # Pretend inference started when last one finished.
                self.__start_inference_on_frame(
                    erdos.Timestamp(
                        coordinates=[current_time - self.frame_gap]),
                    previous_inference_end_time, worker_id)
            elif self.policy == "tail-aware":
                # From "towards streaming perception" paper, algorithm 1.
                estimated_runtime = self.last_finished_inference["runtime"]
                tao_s = previous_inference_end_time % self.frame_gap
                tao_s_plus_r = (previous_inference_end_time +
                                estimated_runtime) % self.frame_gap
                if tao_s_plus_r < tao_s:
                    self.__start_inference_on_frame(timestamp, current_time,
                                                    worker_id)
                else:
                    self.__start_inference_on_frame(
                        erdos.Timestamp(
                            coordinates=[current_time - self.frame_gap]),
                        previous_inference_end_time, worker_id)

    def __get_any_free_worker(self):
        for i in range(len(self.ongoing_inference)):
            if self.ongoing_inference[i] is None:
                return i
        return None

    def __get_earliest_ending_worker(self, current_time):
        assert all([
            x is not None for x in self.ongoing_inference
        ]), ("this function should not be called if there is a free worker")
        earliest_viable_end_time = None
        chosen_index = None
        for i, end_time in enumerate(self.ongoing_inference_end_time):
            assert end_time is not None, "missing end time entry"
            if (earliest_viable_end_time is None
                    or end_time < earliest_viable_end_time):
                earliest_viable_end_time = end_time
                chosen_index = i
        if earliest_viable_end_time < current_time:
            return chosen_index
        else:
            return None

    def __get_prediction_by_frame_timestamp(self, target_timestamp):
        for (timestamp, prediction) in self._received_predictions:
            if timestamp == target_timestamp:
                return prediction
        just_timestamps = [x[0] for x in self._received_predictions]
        raise RuntimeError("Could not find timestamp {} in list of {}".format(
            target_timestamp, just_timestamps))

    def __start_inference_on_frame(self, timestamp, release_time, worker_id):
        """Start inference on frame from timestamps at release_time."""
        assert self.ongoing_inference[worker_id] is None, (
            "Inference already running")
        assert self.ongoing_inference_end_time[worker_id] is None, (
            "Inference end time cell is in use")
        self.ongoing_inference[worker_id] = \
            self.__get_prediction_by_frame_timestamp(timestamp)
        self.ongoing_inference_end_time[worker_id] = release_time + \
            self.ongoing_inference[worker_id]["runtime"]

    def __complete_inference(self, sequenced_obstacles_stream, worker_id):
        """Finishes the currently "running" inference."""
        assert self.ongoing_inference[worker_id] is not None, (
            "No inference is running")
        assert self.ongoing_inference_end_time[worker_id] is not None, (
            "Inference end time cell not filled")
        self.last_finished_inference = self.ongoing_inference[worker_id]
        self.ongoing_inference[worker_id] = None
        # NOTE: The runtime includes the time the frame waited the
        # inference to be started (i.e., that time that the
        # sequencer introduced).
        runtime = (self.ongoing_inference_end_time[worker_id] -
                   self.last_finished_inference["timestamp"].coordinates[0])
        self.ongoing_inference_end_time[worker_id] = None
        if (self.latest_finished_inference_ts is None
                or self.latest_finished_inference_ts <
                self.last_finished_inference["timestamp"]):
            sequenced_obstacles_stream.send(
                ObstaclesMessage(self.last_finished_inference["timestamp"],
                                 self.last_finished_inference["obstacles"],
                                 runtime=runtime))
            self.__send_completed_timestamps(
                self.last_finished_inference["timestamp"],
                sequenced_obstacles_stream)
            self.latest_finished_inference_ts = self.last_finished_inference[
                "timestamp"]

    def __send_completed_timestamps(self, timestamp_ceiling,
                                    sequenced_obstacles_stream):
        """Sends watermarks for completed timestamps."""
        cutoff = None
        for i, (timestamp, _) in enumerate(self._received_predictions):
            if timestamp > timestamp_ceiling:
                cutoff = i
                break
            else:
                sequenced_obstacles_stream.send(
                    erdos.WatermarkMessage(timestamp))
        if cutoff is not None:
            self._received_predictions = self._received_predictions[cutoff:]
        else:
            self._received_predictions = []
