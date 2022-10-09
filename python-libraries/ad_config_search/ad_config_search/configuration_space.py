import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ad_config_search.data_processing_utils import cache_score_dataset
from ad_config_search.policy_utils import (config_knobs,
                                           evaluate_fine_grained_policy,
                                           get_average_config,
                                           get_oracle_policy)
from ad_config_search.utils import run_under_working_dir


class ConfigurationSpace:
    def __init__(self, time_mode: str, infinite_mode: bool, num_sectors: int,
                 working_directory):
        self.time_mode = time_mode
        self.infinite_mode = infinite_mode
        self.num_sectors = num_sectors
        self.working_directory = Path(working_directory).resolve()
        with run_under_working_dir(self.working_directory):
            self.score_df = cache_score_dataset(time_mode, infinite_mode,
                                                num_sectors)

    def slice_config_subset(self, config_subset: pd.DataFrame):
        assert set(config_subset.columns).issubset(
            set(config_knobs)), "config_subset has unexpected columns"
        assert not np.any(config_subset.duplicated(
            config_subset.columns)), "config subset has duplicate rows"
        self.score_df = self.score_df.merge(config_subset,
                                            on=list(config_subset.columns))

    def slice_config_subset_from_file(self, configuration_subset_path):
        with open(configuration_subset_path, 'rb') as f:
            configuration_subset = pickle.load(f)
        for k, v in configuration_subset.items():
            assert k in config_knobs, k
            config_space_knob_values = self.score_df[k].unique()
            assert np.all([
                knob_value in config_space_knob_values for knob_value in v
            ]), config_space_knob_values
        for k, v in configuration_subset.items():
            self.score_df = self.score_df[self.score_df[k].isin(v)]

    @property
    def configs(self):
        return self.score_df[config_knobs].drop_duplicates().sort_values(
            by=config_knobs).copy()

    @property
    def df(self):
        return self.score_df.copy()

    @property
    def scenarios(self):
        return list(
            self.score_df["run"].map(lambda x: x.split("-P")[0]).unique())

    def slice_scenario_subset(self, scenario_list: list):
        # Requires a non empty unique list, contains full scenario names,
        # which all exist in this configuration space
        assert len(scenario_list) > 0, len(scenario_list)
        assert len(np.unique(scenario_list)) == len(
            scenario_list), "Scenario list has repeated scenarios"
        assert "-P" not in scenario_list[
            0], "scenario names should not have segment values in them"
        full_scenario_list = self.score_df["run"].map(
            lambda x: x.split("-P")[0])
        assert set(list(full_scenario_list)).issuperset(scenario_list), (
            "Some of the scenario list meant to be a subset are not in "
            "the configuration space")
        self.score_df = self.score_df[full_scenario_list.isin(scenario_list)]

    def train_test_split(self, split_fn) -> tuple:
        split_result = split_fn(self.scenarios)
        train_scenarios, test_scenarios = [
            split_result[x] for x in ["train", "test"]
        ]
        assert len(set(train_scenarios).intersection(
            test_scenarios)) == 0, "train and test scenarios have overlap"
        # make a copy of score_df, slice each by scenario list
        train_space = self.__class__(time_mode=self.time_mode,
                                     infinite_mode=self.infinite_mode,
                                     num_sectors=self.num_sectors,
                                     working_directory=self.working_directory)
        test_space = self.__class__(time_mode=self.time_mode,
                                    infinite_mode=self.infinite_mode,
                                    num_sectors=self.num_sectors,
                                    working_directory=self.working_directory)
        train_space.score_df = self.score_df.copy()
        test_space.score_df = self.score_df.copy()
        train_space.slice_scenario_subset(train_scenarios)
        test_space.slice_scenario_subset(test_scenarios)
        return {"train": train_space, "test": test_space}

    def average_configuration(self) -> dict:
        with run_under_working_dir(self.working_directory):
            return get_average_config(
                train_scenarios=self.score_df["run"].unique(),
                time_mode=self.time_mode,
                infinite_mode=self.infinite_mode,
                config_subset=self.configs)

    def static_policy_scores(self, config: dict, other_metrics=False) -> dict:
        assert len(config) > 0, config
        assert np.all([k in config_knobs for k in config.keys()
                       ]), f"some key in config {config} is not a config knob"
        static_policy = {k: config for k in self.score_df["run"].unique()}
        with run_under_working_dir(self.working_directory):
            return evaluate_fine_grained_policy(static_policy,
                                                time_mode=self.time_mode,
                                                default_config=config,
                                                other_metrics=other_metrics)

    def oracle_policy(self) -> dict:
        with run_under_working_dir(self.working_directory):
            oracle_policy = get_oracle_policy(time_mode=self.time_mode,
                                              infinite_mode=self.infinite_mode,
                                              num_sectors=self.num_sectors,
                                              config_subset=self.configs)
        scenarios = self.scenarios
        return {
            k: v
            for k, v in oracle_policy.items() if k.split("-P")[0] in scenarios
        }

    def policy_scores(self,
                      policy_dict: dict,
                      default_config: dict = None,
                      other_metrics=False) -> int:
        # requires nonempty policy dict, whose scenarios and configurations
        # exist in this configuration space, and where the default
        # configuration is in the configuration space
        assert len(policy_dict) > 0, policy_dict
        assert set([k.split("-P")[0] for k in policy_dict.keys()]).issubset(
            list(self.scenarios)
        ), "policy given has scenarios not in this configuration space"
        configs_in_this_space = self.configs.to_dict('records')
        assert default_config is None or (
            default_config in configs_in_this_space), (
                "default config not in this configuration space")
        assert np.all([
            v in configs_in_this_space for v in policy_dict.values()
        ]), "policy_dict contains configurations not in this space"
        with run_under_working_dir(self.working_directory):
            return evaluate_fine_grained_policy(policy_dict,
                                                time_mode=self.time_mode,
                                                default_config=default_config,
                                                other_metrics=other_metrics)

    def improvement_analysis(self,
                             training_configuration_space,
                             policy_dict,
                             other_metrics=False) -> int:
        assert (
            training_configuration_space.time_mode == self.time_mode and
            training_configuration_space.infinite_mode == self.infinite_mode
            and training_configuration_space.num_sectors == self.num_sectors
            and training_configuration_space.working_directory
            == self.working_directory
        ), (f"Configuration spaces {self} and "
            f"{training_configuration_space} don't match on basic parameters")
        assert training_configuration_space.configs.sort_values(
            by=config_knobs).reset_index(drop=True).equals(
                self.configs.sort_values(by=config_knobs).reset_index(
                    drop=True)
            ), ("configurations of this space and the training space is "
                "not the same")
        assert len(
            set(list(training_configuration_space.scenarios)).intersection(
                list(self.scenarios))) == 0, (
                    "training configuration space and this one have "
                    "intersection scenarios")
        # go through motions of getting everything's score dicts
        train_avg_config = training_configuration_space.average_configuration()
        static_policy_scores = self.static_policy_scores(
            train_avg_config, other_metrics=other_metrics)
        policy_scores = self.policy_scores(policy_dict=policy_dict,
                                           default_config=train_avg_config,
                                           other_metrics=other_metrics)
        oracle_policy = self.oracle_policy()
        oracle_scores = self.policy_scores(policy_dict=oracle_policy,
                                           default_config=None,
                                           other_metrics=other_metrics)
        scores_tuple = (static_policy_scores, policy_scores, oracle_scores)
        if other_metrics:

            def extract(result_tuple):
                result = np.concatenate([
                    np.squeeze(result_tuple[0]),
                    np.squeeze(result_tuple[1]).reshape(1)
                ])
                return result

            def aggregate(score_matrix):
                keys = [("FP", np.sum), ("FN", np.sum), ("IDS", np.sum),
                        ("MOTP", np.mean)]
                return {
                    k: fn(score_matrix[:, i])
                    for i, (k, fn) in enumerate(keys)
                }

            oracle_scores = {k: extract(v) for k, v in oracle_scores.items()}
            policy_scores = {k: extract(v) for k, v in policy_scores.items()}
            static_scores = {
                k: extract(v)
                for k, v in static_policy_scores.items()
            }
            for name, d in zip(["oracle", "learned policy", "global static"],
                               [oracle_scores, policy_scores, static_scores]):
                nan_scenarios = [k for k, v in d.items() if np.isnan(v).any()]
                if len(nan_scenarios) > 0:
                    print(">>>>>>> Deleting nan scenarios from " + name +
                          str(nan_scenarios))
                    for k in nan_scenarios:
                        del d[k]

            oracle_diff = np.array([
                (oracle_scores[k] - static_scores[k])
                for k in set.intersection(set(static_scores.keys()),
                                          set(oracle_scores.keys()))
            ])
            oracle_improvement = aggregate(oracle_diff)
            policy_diff = np.array([
                (policy_scores[k] - static_scores[k])
                for k in set.intersection(set(static_scores.keys()),
                                          set(policy_scores.keys()))
            ])
            policy_improvement = aggregate(policy_diff)

            static_score = aggregate(np.array(list(static_scores.values())))
            policy_score = aggregate(np.array(list(policy_scores.values())))
            oracle_score = aggregate(np.array(list(oracle_scores.values())))

            return (oracle_scores, policy_scores,
                    static_scores), (static_score, policy_score,
                                     oracle_score), (policy_improvement,
                                                     oracle_improvement)
        else:
            policy_improvement = np.mean([
                policy_scores[k] - static_policy_scores[k]
                for k in static_policy_scores.keys()
            ])
            oracle_improvement = np.mean([
                oracle_scores[k] - static_policy_scores[k]
                for k in static_policy_scores.keys()
            ])
            return scores_tuple, tuple([
                np.mean(list(d.values())) for d in scores_tuple
            ]), (policy_improvement, oracle_improvement)

    def __repr__(self) -> str:
        return (f"ConfigurationSpace(time_mode={self.time_mode}, "
                f"infinite_mode={self.infinite_mode}, "
                f"num_sectors={self.num_sectors}, "
                f"working_directory={self.working_directory})")
