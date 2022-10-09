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
    from ad_config_search.configuration_space import ConfigurationSpace

    space = ConfigurationSpace(time_mode=exp_result.time_mode,
                               infinite_mode=exp_result.infinite_mode,
                               num_sectors=exp_result.num_sectors,
                               working_directory=".")
    from ad_config_search.policy_utils import config_knobs

    space.slice_config_subset(
        exp_result.train_df[config_knobs].drop_duplicates())
    train_scenarios = exp_result.train_df.index.map(
        lambda x: x.split("-P")[0]).unique()
    test_scenarios = exp_result.test_df.index.map(
        lambda x: x.split("-P")[0]).unique()
    split = space.train_test_split(
        lambda scenarios: {
            "train": [x for x in scenarios if x in train_scenarios],
            "test": [x for x in scenarios if x in test_scenarios]
        })
    space_train = split["train"]
    space_test = split["test"]
    from ad_config_search.policy_utils import oracle_from_df

    learned_policy = oracle_from_df(exp_result.test_df)
    mota_result = space_test.improvement_analysis(space_train,
                                                  learned_policy,
                                                  other_metrics=False)
    other_metrics_result = space_test.improvement_analysis(space_train,
                                                           learned_policy,
                                                           other_metrics=True)
    print(mota_result[1])
    print(other_metrics_result[1])


if __name__ == '__main__':
    app.run(main)
