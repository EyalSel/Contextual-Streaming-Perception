import erdos


def prepend_line_to_file(file_path, line, repeatable=False):
    with open(file_path, 'r') as f:
        first_line = f.readline()[:-1]
    if first_line == line and not repeatable:
        return
    with open(file_path, 'r') as f:
        contents = f.read()
    with open(file_path, 'w') as f:
        f.write(line + "\n")
        f.write(contents)


def verify_keys_in_dict(required_keys, arg_dict):
    assert set(required_keys).issubset(set(arg_dict.keys())), \
            "one or more of {} not found in {}".format(required_keys, arg_dict)


def fix_pylot_profile(file_path, silent=True):
    with open(file_path, 'r') as f:
        contents = f.read()
    if contents[0] == "[":
        if not silent:
            print("The pylot_profile.json file seems to already be fixed")
        return
    with open(file_path, 'w') as f:
        f.write("[\n")
        f.write(contents[:-2])
        f.write("\n]")


def get_num_det_evals(flags):
    """
    A function that returns the number of detection evaluations finished
    """
    import pandas as pd
    df = pd.read_csv(flags.csv_log_file_name)
    num_eval_maps_found = \
        len(df.loc[(df["operator"] == "detection_eval_operator") &
                   (df["extra_info"] == "mAP")])
    num_coco_aggr_maps_found = \
        len(df.loc[(df["operator"] == "coco_detection_eval_operator") &
                   (df["extra_info"] ==
                    "coco_AP_IoU=0.50:0.95_area=all_maxDets=100")])
    coco_lookback = flags.coco_detection_eval_lookback
    coco_freq = flags.coco_detection_eval_freq
    num_coco_aggr_maps_found = \
        num_coco_aggr_maps_found + coco_lookback // coco_freq
    eval_count = num_eval_maps_found \
        if coco_freq is None else num_coco_aggr_maps_found * coco_freq
    return eval_count


def top_watermark_stream():
    class CustomSequenceSender(erdos.Operator):
        """
        Should be aggregated with the identical function from test_utils at
        some point
        """
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

    return erdos.connect(
        CustomSequenceSender,
        erdos.OperatorConfig(name="top_watermark_sender"), [],
        [erdos.WatermarkMessage(erdos.Timestamp(is_top=True))])[0]


def join_streams(name, input_streams):
    MAX_STREAMS = 3
    assert len(input_streams) > 1, "requires at least 2 streams"
    assert len(input_streams
               ) <= MAX_STREAMS, f"supports at most {MAX_STREAMS} streams"

    for i in range(MAX_STREAMS - len(input_streams)):
        input_streams.append(top_watermark_stream())

    class StreamJoin(erdos.Operator):
        def __init__(self, input_stream_1: erdos.ReadStream,
                     input_stream_2: erdos.ReadStream,
                     input_stream_3: erdos.ReadStream,
                     output_stream: erdos.WriteStream):
            self.output_stream = output_stream
            for stream in [input_stream_1, input_stream_2, input_stream_3]:
                stream.add_callback(self.on_msg)

        @staticmethod
        def connect(input_stream_1: erdos.ReadStream,
                    input_stream_2: erdos.ReadStream,
                    input_stream_3: erdos.ReadStream):
            output_stream = erdos.WriteStream()
            return [output_stream]

        def on_msg(self, msg):
            self.output_stream.send(msg)

    return erdos.connect(StreamJoin, erdos.OperatorConfig(name=name),
                         input_streams)[0]
