import pickle

from absl import app, flags

flags.DEFINE_string('experiment_path',
                    default=None,
                    required=True,
                    help="path to exp_dir experiment")
FLAGS = flags.FLAGS


def main(_):
    with open(FLAGS.experiment_path, 'rb') as f:
        exp_result = pickle.load(f)
    feature_importances = list(
        zip(exp_result.model_parameters.feature_importances_,
            list(exp_result.other_data_dump["column_schema"])))
    for feature_prefix in ["bin_0", "bin_1", "bin_2", "bin_3", "bin_4"]:
        print(
            feature_prefix,
            sum([
                score for score, name in feature_importances
                if name.startswith(feature_prefix)
            ]))

    for feature_suffix in [
            "bbox_longevity", "num_bboxes", "bbox_speed",
            "bbox_self_iou_1frame"
    ]:
        print(
            feature_suffix,
            sum([
                score for score, name in feature_importances
                if name.endswith(feature_suffix)
            ]))


if __name__ == '__main__':
    app.run(main)
