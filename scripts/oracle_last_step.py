import numpy as np
from ad_config_search.configuration_space import ConfigurationSpace
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_boolean("infinite_mode", None, "Whether to use infinite GPU mode")
flags.DEFINE_integer("num_sectors", None,
                     "The number of sectors per scenario to consider")


def main(_):
    space = ConfigurationSpace("timely", FLAGS.infinite_mode,
                               FLAGS.num_sectors, ".")
    split = space.train_test_split(
        lambda scenarios: {
            "train": [x for x in scenarios if "train" in x],
            "test": [x for x in scenarios if "val" in x]
        })
    train_space, test_space = split["train"], split["test"]
    oracle_policy = test_space.oracle_policy()

    # assign scenarios from before to now
    def increment(scenario):
        scenario_number = int(scenario.split("-P")[1].split("_")[0])
        assert scenario_number != (FLAGS.num_sectors - 1)
        return (scenario.split("-P")[0] +
                f"-P{scenario_number+1}_{FLAGS.num_sectors}")

    # set default to global static config
    incremented_oracle = {
        increment(k): v
        for k, v in oracle_policy.items()
        if int(k.split("-P")[1].split("_")[0]) < FLAGS.num_sectors - 1
    }
    policy_scores = test_space.policy_scores(
        incremented_oracle, default_config=train_space.average_configuration())
    policy_scores_other_metrics = test_space.policy_scores(
        incremented_oracle,
        default_config=train_space.average_configuration(),
        other_metrics=True)

    def extract(result_tuple):
        result = np.concatenate([
            np.squeeze(result_tuple[0]),
            np.squeeze(result_tuple[1]).reshape(1)
        ])
        return result

    def aggregate(score_matrix):
        keys = [("FP", np.sum), ("FN", np.sum), ("IDS", np.sum),
                ("MOTP", np.mean)]
        return {k: fn(score_matrix[:, i]) for i, (k, fn) in enumerate(keys)}

    policy_scores_other_metrics = {
        k: extract(v)
        for k, v in policy_scores_other_metrics.items()
    }
    policy_score_other_metrics = aggregate(
        np.array(list(policy_scores_other_metrics.values())))

    print(np.mean(list(policy_scores.values())))
    print(policy_score_other_metrics)


if __name__ == '__main__':
    app.run(main)
