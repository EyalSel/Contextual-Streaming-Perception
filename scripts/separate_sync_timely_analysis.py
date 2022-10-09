"""
This script evaluates a policy that combines two score predictors: one that
predicts the sync accuracy drop, and another that predicts the degradation. The
predicted scores are subtracted, and then a policy is created by argmaxing to
find the best predicted configuration in every scenario.

The flags sync_mode and degradation_mode specify what kind of policy to use to
predict these two values.
 - Optimal
 - model: A trained model using random_forest_prep.py, specified by the
   XXXX_model_path flags
 - global_static: For each configuration in the test set, it predicts the
   average score of that configuration on the training dataset. The argmax
   policy of this predictor is the global best policy.
"""

from pathlib import Path
import pickle

import pandas as pd
from absl import app, flags
from ad_config_search.configuration_space import ConfigurationSpace
from ad_config_search.data_processing_utils import cache_score_dataset
from ad_config_search.policy_utils import config_knobs, oracle_from_df
from ad_config_search.rforest_utils import (split_scenarios,
                                            waymo_official_train_val_test)
from icecream import ic

flags.DEFINE_enum(
    "sync_mode",
    default="model",
    enum_values=["optimal", "model", "global_static"],
    required=False,
    help="trained model predicting degradation with baseline subtraction")
flags.DEFINE_string(
    'sync_model_path',
    default=None,
    required=False,
    help="trained model predicting sync with baseline subtraction")
flags.DEFINE_string(
    'degradation_model_path',
    default=None,
    required=False,
    help="trained model predicting degradation with baseline subtraction")
flags.DEFINE_enum(
    "degradation_mode",
    default="model",
    enum_values=["optimal", "model", "global_static"],
    required=False,
    help="trained model predicting degradation with baseline subtraction")

flags.DEFINE_bool('infinite_mode',
                  default=False,
                  help="use infinite GPU mode or not")
flags.DEFINE_bool('remove_inf_nan',
                  default=True,
                  help="remove rows in the dataset with inf or nan scores")
flags.DEFINE_integer('num_sectors',
                     default=None,
                     required=True,
                     help="number of sectors to divide videos into")
flags.DEFINE_enum('dataset_split_strategy', None, ['waymo', 'random'],
                  "The strategy used to split the dataset to train and test")
flags.DEFINE_bool('save_policy', False, "whether to save the resulting policy")


def sync_model_check(flags_dict):
    return (flags_dict["sync_mode"] == "model") == (
        flags_dict["sync_model_path"] is not None)


def degradation_model_check(flags_dict):
    return (flags_dict["degradation_mode"] == "model") == (
        flags_dict["degradation_model_path"] is not None)


flags.register_multi_flags_validator(
    ['sync_mode', 'sync_model_path'],
    sync_model_check,
    message=("Either use perfect sync or provide sync model path"))

flags.register_multi_flags_validator(
    ['degradation_model_path', 'degradation_mode'],
    degradation_model_check,
    message=("Either use global static degradation or "
             "provide degradation model path"))

FLAGS = flags.FLAGS


def verify_exp(exp_path):
    """
    Verify that the experiment in this path was done with the same
    infinite_mode, num_sectors, remove_inf_nan, and dataset_split_strategy
    as the values passed to this evaluation script.
    """
    with open(exp_path, 'rb') as f:
        exp_result = pickle.load(f)
    for flag in [
            "infinite_mode", "num_sectors", "remove_inf_nan",
            "dataset_split_strategy"
    ]:
        flag_value = FLAGS.flag_values_dict()[flag]
        exp_value = exp_result.metadata[flag]
        assert (flag_value == exp_value), (
            f"{flag} doesn't match between experiment ({exp_value}) "
            f"{exp_path} and the given flag {flag_value}")


def subtract_baseline(df_config, train_scenarios, time_mode, infinite_mode):
    from ad_config_search.transformations import SubtractBaseline
    just_train_df = df_config[df_config["run"].map(
        lambda x: x.split("-P")[0]).isin(train_scenarios)]
    transform = SubtractBaseline(time_mode=time_mode,
                                 infinite_mode=infinite_mode)
    transform.fit(just_train_df)
    return transform.transform(df_config)


def prep_columns(df):
    """
    prepares df to be merged with another on config knob columns.
    In addition keeps score/run, and drops everything else.
    """
    df = df.reset_index()
    df = df.drop(columns=list(
        set(df.columns) - set(config_knobs) - {"T-model", "score", "run"}))
    return df


def prepare_predictor_scores(sync_or_degradation,
                             cached_train_test_scenarios=None):
    """
    sync_or_degradation is True if called for sync score and False for
    degradation.
    
    Returns a dataframe with the predicted scores on the test dataset. This
    function uses cached_train_test_scenarios if they are provided, otherwise
    it creates them.
    """
    time_mode, score = ("sync",
                        "mota") if sync_or_degradation else ("timely",
                                                             "degradation")
    config_scores = cache_score_dataset(time_mode=time_mode,
                                        infinite_mode=FLAGS.infinite_mode,
                                        num_sectors=FLAGS.num_sectors,
                                        score=score,
                                        remove_inf_nan=FLAGS.remove_inf_nan)
    if cached_train_test_scenarios is None:
        split_fn = {
            "waymo": waymo_official_train_val_test,
            "random":
            lambda x: split_scenarios(x, 1 / 3, FLAGS.dataset_split_seed)
        }[FLAGS.dataset_split_strategy]
        train_scenarios, test_scenarios = split_fn(
            config_scores["run"].map(lambda x: x.split("-P")[0]).unique())
    else:
        train_scenarios, test_scenarios = cached_train_test_scenarios
    config_scores = subtract_baseline(config_scores, train_scenarios, "timely",
                                      FLAGS.infinite_mode)
    test_df = config_scores[config_scores["run"].map(
        lambda x: x.split("-P")[0]).isin(test_scenarios)]
    if (sync_or_degradation and FLAGS.sync_mode == "global_static") or (
            not sync_or_degradation
            and FLAGS.degradation_mode == "global_static"):
        # Construct the prediction function that generates the global best
        # policy. See docstring at the top of the script for details.
        train_df = config_scores[config_scores["run"].map(
            lambda x: x.split("-P")[0]).isin(train_scenarios)]
        config_averages = {}
        for config, rows in train_df.groupby(config_knobs):
            config_averages[config] = rows["score"].mean()
        all_rows = []
        test_df = test_df.copy()
        for config, rows in test_df.groupby(config_knobs):
            rows["score"] = config_averages[config]
            all_rows.append(rows)
        test_df = pd.concat(all_rows)
    return test_df, (train_scenarios, test_scenarios)


def main(_):
    # First create a dataframe matching each scenario and configuration with
    # the their predicted scores (both sync and degradation)

    # Create the dataframe for degradation predictions
    # This block also initializes train_scenarios and test_scenarios, which
    # are used by the next block to make the sync predictions dataframe.
    # These train and test scenarios are assumed to be the same between the two
    # predictors, because dataset_split_strategy is the same throughout.
    if FLAGS.degradation_mode in ["global_static", "optimal"]:
        # Create a degradation dataset, split it to train and test, subtract
        # the baseline from all the scores
        (deg_exp_test_df, (train_scenarios,
                           test_scenarios)) = prepare_predictor_scores(
                               sync_or_degradation=False)
        ic(deg_exp_test_df)
    else:  # model
        verify_exp(FLAGS.degradation_model_path)
        with open(FLAGS.degradation_model_path, 'rb') as f:
            deg_exp = pickle.load(f)
        deg_exp_train_df = prep_columns(deg_exp.train_df)
        deg_exp_test_df = prep_columns(deg_exp.test_df)
        train_scenarios = deg_exp_train_df["run"].map(
            lambda x: x.split("-P")[0]).unique()
        test_scenarios = deg_exp_test_df["run"].map(
            lambda x: x.split("-P")[0]).unique()

    # Create the dataframe for sync predictions
    if FLAGS.sync_mode in ["global_static", "optimal"]:
        sync_exp_test_df, _ = prepare_predictor_scores(
            sync_or_degradation=True,
            cached_train_test_scenarios=(train_scenarios, test_scenarios))
    else:  # model
        verify_exp(FLAGS.sync_model_path)
        with open((FLAGS.sync_model_path), 'rb') as f:
            sync_exp = pickle.load(f)
        sync_exp_test_df = prep_columns(sync_exp.test_df)

    # Merge the predictions dataframes, subtract, and derive the argmax policy
    df = deg_exp_test_df.merge(
        sync_exp_test_df,
        how="inner",
        on=list(set(deg_exp_test_df.columns) - {"score"}),
        suffixes=("_deg", "_sync"),
    )
    df["score"] = df["score_sync"] - df["score_deg"]
    policy = oracle_from_df(df)

    # Maybe save the policy
    if FLAGS.save_policy:
        fn = "sync_timely_policy--"
        flags = [
            "degradation_mode", "sync_mode", "infinite_mode", "remove_inf_nan",
            "num_sectors", "dataset_split_strategy"
        ]
        flags_dict = FLAGS.flag_values_dict()
        fn_pieces = [f"{flag}={flags_dict[flag]}" for flag in flags]
        if FLAGS.degradation_mode == "model":
            path = Path(FLAGS.degradation_model_path).name.split("--")[0]
            fn_pieces.append(f"degradation_model_prefix={path}")
        if FLAGS.sync_mode == "model":
            path = Path(FLAGS.sync_model_path).name.split("--")[0]
            fn_pieces.append(f"sync_model_prefix={path}")
        fn = fn + "__".join(fn_pieces) + ".pl"
        with open(fn, 'wb') as f:
            pickle.dump(policy, f)

    # Evaluate the policy
    space = ConfigurationSpace("timely", FLAGS.infinite_mode,
                               FLAGS.num_sectors, ".")

    split = space.train_test_split(
        lambda scenarios: {
            "train": [x for x in scenarios if x in train_scenarios],
            "test": [x for x in scenarios if x in test_scenarios],
        })
    space_train = split["train"]
    space_test = split["test"]
    result = space_test.improvement_analysis(space_train,
                                             policy,
                                             other_metrics=False)
    print(result[1])
    print(result[2])


if __name__ == '__main__':
    app.run(main)
