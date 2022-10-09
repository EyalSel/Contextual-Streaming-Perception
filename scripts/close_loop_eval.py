"""
This script file takes as argument trained models in the form of ExpResult
(from policy_utils.py), and applies the trained model in a close loop where its
own configuration choice at time t is used to featurize the environment at time
t+1.

Present vs past-1-step dictates if the configuration choice at time t applies
immediately or is only considered to apply at time t+1.

This script assumes that the cached environment featurization from the
configuration choices is available.
"""
import pickle
from ad_config_search.configuration_space import ConfigurationSpace

import numpy as np
import pandas as pd
from ad_config_search.data_processing_utils import cache_features_from_pipeline
from ad_config_search.policy_utils import (ExpResult, config_knobs,
                                           highest_avg_config)
from ad_config_search.utils import contract_one_hot, get_rows, suppress_stdout
from tqdm import tqdm

from absl import app, flags

flags.DEFINE_string('experiment_path',
                    default=None,
                    required=True,
                    help="path to exp_dir experiment")

flags.DEFINE_enum('features_time',
                  default=None,
                  enum_values=["present", "past-1-step"],
                  required=True,
                  help="collect features from present, or past")
FLAGS = flags.FLAGS


def get_best_config(env: pd.DataFrame, exp_result: ExpResult, cached_model):
    all_configs = exp_result.test_df[config_knobs].drop_duplicates()
    all_configs = pd.get_dummies(all_configs, prefix_sep="__")
    # repeating env for each configuration
    env = pd.concat([env for _ in range(len(all_configs))])
    # concatenate columns of configuration and environment
    final_df = pd.concat(
        [all_configs.reset_index(drop=True),
         env.reset_index(drop=True)],
        axis=1)
    # get columns in the right order that the model is used to
    column_order = exp_result.other_data_dump["column_schema"]
    assert set(final_df.columns) == set(column_order), (set(final_df.columns) -
                                                        set(column_order),
                                                        set(column_order) -
                                                        set(final_df.columns))
    final_df = final_df[column_order]
    with suppress_stdout(stderr=True):
        scores = cached_model.predict(final_df)
        best_config = contract_one_hot(all_configs).iloc[np.argmax(scores)]
    return dict(best_config)


def main(_):
    # load C' x V -> E dataframe
    cve_df = cache_features_from_pipeline(1, True)
    cve_df["T-model"] = "sort"
    # change environment columns to one hot, so it includes all of the possible
    # labels choices
    categorical_object_cols = cve_df.select_dtypes(["object",
                                                    "category"]).columns
    categories_to_one_hot = set(categorical_object_cols) - set(cve_df.columns[
        cve_df.columns.str.startswith("pastC-")]) - {"scenario_name"}
    cve_df = pd.get_dummies(cve_df,
                            prefix_sep="__",
                            columns=categories_to_one_hot)
    # load policy
    with open(FLAGS.experiment_path, 'rb') as f:
        exp_result = pickle.load(f)

    # Find global static configuration from exp_result datastructure
    global_static_config = highest_avg_config(exp_result)

    # record scenario segments and full scenarios in test set
    test_scenarios_segments = exp_result.test_df.index.unique()
    test_scenarios = test_scenarios_segments.map(
        lambda x: x.split("-P")[0]).unique()

    policy_decisions = {}

    # caching so it doesn't have to be copied every time
    cached_model = exp_result.model_parameters

    for scenario in tqdm(test_scenarios):
        # get sorted list of segment names
        scenario_segment_names = sorted(
            cve_df[cve_df["scenario_name"].map(lambda x: x.split("-P")[0]) ==
                   scenario]["scenario_name"].unique(),
            key=lambda x: int(x.split("-P")[1].split("_")[0]))
        current_config = global_static_config
        for i, scenario_segment in enumerate(scenario_segment_names):
            if FLAGS.features_time == "past-1-step" and i > 0:
                # record current_config and segment in policy dictionary
                policy_decisions[scenario_segment] = current_config
            # if segment is in the missing ones from the policy test_df
            if scenario_segment not in test_scenarios_segments:
                current_config = global_static_config
            else:
                entry = {
                    "scenario_name": scenario_segment,
                    **{f"pastC-{k}": v
                       for k, v in current_config.items()}
                }
                rows = get_rows(cve_df, entry)
                assert len(rows) == 1, (entry, rows)
                raw_env = rows.drop(columns=[
                    c for c in rows.columns if c.startswith("pastC-")
                ] + ["scenario_name"])
                # feed into policy to get configuration, set current_config
                current_config = get_best_config(raw_env, exp_result,
                                                 cached_model)
            if FLAGS.features_time == "present":
                # record current_config and segment in policy dictionary
                policy_decisions[scenario_segment] = current_config

    experiment_name = FLAGS.experiment_path.split("--")[0]

    with open((f"{experiment_name}--close_loop_policy_decisions"
               f"__features_time={FLAGS.features_time}.pl"), 'wb') as f:
        pickle.dump(policy_decisions, f)

    # evaluate new policy dictionary
    space = ConfigurationSpace(exp_result.time_mode, exp_result.infinite_mode,
                               exp_result.num_sectors, ".")
    space.slice_config_subset(
        exp_result.train_df[config_knobs].drop_duplicates())

    def train_test_split(scenarios):
        train_scenarios = exp_result.train_df.index.map(
            lambda x: x.split("-P")[0]).unique()
        test_scenarios = exp_result.test_df.index.map(
            lambda x: x.split("-P")[0]).unique()
        assert set(scenarios) == set(train_scenarios).union(
            set(test_scenarios)), set(scenarios) / (set(train_scenarios).union(
                set(test_scenarios)))
        return {"train": train_scenarios, "test": test_scenarios}

    d = space.train_test_split(train_test_split)
    train_space, test_space = d["train"], d["test"]
    num_gt_analysis_results = test_space.improvement_analysis(
        train_space, policy_decisions, other_metrics=False)
    with open((f"{experiment_name}--close_loop_num_gt_analysis_result"
               f"__features_time={FLAGS.features_time}.pl"), 'wb') as f:
        pickle.dump(num_gt_analysis_results, f)
    other_metrics_analysis_results = test_space.improvement_analysis(
        train_space, policy_decisions, other_metrics=True)
    with open((f"{experiment_name}--close_loop_other_metrics_analysis_result"
               f"__features_time={FLAGS.features_time}.pl"), 'wb') as f:
        pickle.dump(other_metrics_analysis_results, f)


if __name__ == '__main__':
    app.run(main)
