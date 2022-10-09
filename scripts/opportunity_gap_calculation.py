from absl import app, flags
from ad_config_search.configuration_space import ConfigurationSpace

flags.DEFINE_boolean("infinite_mode", None, "Whether to use infinite GPU mode")
flags.DEFINE_integer("num_sectors", None,
                     "The number of sectors per scenario to consider")
flags.DEFINE_enum(
    "time_mode",
    None, ["timely", "sync"],
    help="Timely = streaming evaluation mode, sync = offline evaluation mode")
FLAGS = flags.FLAGS


def main(_):
    configuration_space = ConfigurationSpace(FLAGS.time_mode,
                                             FLAGS.infinite_mode,
                                             FLAGS.num_sectors, ".")
    split = configuration_space.train_test_split(
        lambda scenarios: {
            "train": [x for x in scenarios if x.startswith("train")],
            "test": [x for x in scenarios if x.startswith("val")]
        })
    configuration_space_train = split["train"]
    configuration_space_test = split["test"]
    optimal_policy = configuration_space_test.oracle_policy()
    global_best_config = configuration_space_train.average_configuration()
    global_best_policy = {
        scenario: global_best_config
        for scenario in optimal_policy.keys()
    }
    mota_analysis = configuration_space_test.improvement_analysis(
        configuration_space_train, global_best_policy, other_metrics=False)
    print(mota_analysis[1])
    other_metrics_analysis = configuration_space_test.improvement_analysis(
        configuration_space_train, global_best_policy, other_metrics=True)
    print(other_metrics_analysis[1])


if __name__ == '__main__':
    app.run(main)
