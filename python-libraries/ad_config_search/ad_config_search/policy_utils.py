"""
Functionality related to transformations and processing of
environment -> config policies
"""
import json
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from ad_config_search.data_processing_utils import (cache_num_gt,
                                                    cache_score_dataset)
from ad_config_search.evaluation_utils import (compute_mota_scores,
                                               compute_other_tracking_metrics)
from ad_config_search.utils import contract_one_hot, get_rows

config_knobs = [
    'D-model', 'D-conf', 'D-seq-pol', 'T-min-iou', 'T-max-age',
    'T-every-nth-det'
]


class ExpResult:
    """
    The primary API to do policy wrangling from a model.
    """
    def __init__(self,
                 train_df: pd.DataFrame,
                 test_df: pd.DataFrame,
                 time_mode: str,
                 infinite_mode: bool,
                 num_sectors: int,
                 model_parameters: Any = None,
                 metadata: dict = {},
                 other_data_dump=None):
        """
        train_df:
            - has config_knobs columns for the configurations
            - an index which is the fine-grained scenario name, index name
              should be "run"
            - score column representing ground truth score
        test_df:
            - has config_knobs columns for the configurations
            - an index which is the fine-grained scenario name, index name
              should be "run"
            - score column representing predicted score (the scores just have
                to maintain order with each scenario slice)
        time_mode: one of timely or sync
        num_sectors: the whole system's been written with Waymo's 20s videos
                     in mind, so this variable is useful for deriving the
                     oracle/global static configuration.
        infinite_mode: indicating whether infinite gpus were used
        model_parameters: Optional, any format. A place for book-keeping of
                          model weights associated with an experiment.
        metadata: Other metadata used in running the experiment. Useful for
                  plotting experiments that control for one or more of these
                  values.
        other_data_dump: Information not used for comparing and controlling
                         experiments like in the metadata argument, but still
                         needed to reproduce parts of the model execution.
        """
        self._train_df = train_df
        self._test_df = test_df
        assert time_mode in ["timely", "sync"], time_mode
        self._time_mode = time_mode
        self._infinite_mode = infinite_mode
        self._num_sectors = num_sectors
        self._model_parameters = model_parameters
        self._metadata = metadata
        self._other_data_dump = other_data_dump

    @property
    def train_df(self):
        return self._train_df.copy()

    @property
    def test_df(self):
        return self._test_df.copy()

    @property
    def time_mode(self):
        return self._time_mode

    @property
    def infinite_mode(self):
        return self._infinite_mode

    @property
    def num_sectors(self):
        return self._num_sectors

    @property
    def model_parameters(self):
        import copy
        return copy.deepcopy(self._model_parameters)

    @property
    def metadata(self):
        import copy
        return copy.deepcopy(self._metadata)

    @property
    def other_data_dump(self):
        import copy
        return copy.deepcopy(self._other_data_dump)


def greedy_condense_space(df_configs, base_policy, k):
    """
    A configuration space condensation function. Tries to apply configurations
    in the base policy in decreasing popularity to every scenario. A candidate
    configuration is deemed an acceptable match for a scenario if it's within
    the top k spots for it in terms of score.

    This function assumes that there's at least one configuration used that is
    within the top-k for every scenario. The oracle configuration is an example
    of a policy that satisfies this condition.

    This function simulates the rich-get-richer scheme, with the underlying
    assumption that the most popular configurations are probably within the
    top-k of other scenarios as well.
    """

    # https://stackoverflow.com/a/47626762/1546071
    # https://stackoverflow.com/a/12570040/1546071
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if type(obj).__module__ == np.__name__:
                return obj.item()
            return json.JSONEncoder.default(self, obj)

    base_policy_str = {
        k: json.dumps(v, cls=NumpyEncoder)
        for k, v in base_policy.items()
    }
    oracle_configs, oracle_config_counts = \
        np.unique(list(base_policy_str.values()), return_counts=True)
    new_policy = {}
    performance_drop = {}

    for scenario, rows in tqdm(df_configs.groupby(by=["run"])):
        found_config = False
        top_k = rows.nlargest(k, columns=["score"], keep='all')
        # some scenarios are just empty frames, making mota score nan or -inf.
        # nlargest is dumb and completely ignores rows with scores = nan
        # (even ranking score = -inf) as higher, which contradicts both the
        # philosphy of nans (spread and propagate) and our notion of
        # configuration goodness (nan is better because -inf represents false
        # positives).
        # To get around this we catch these cases explicitly.
        if len(top_k) == 0 and np.all(np.isnan(top_k["score"])):
            top_k = rows
        elif (len(top_k) > 0 and top_k.iloc[0]["score"] == -np.inf
              and np.any(np.isnan(top_k["score"]))):
            if k == 1:
                top_k = rows[rows["score"] == np.nan]
            else:
                top_k = rows
        for c_str in oracle_configs[np.argsort(oracle_config_counts)[::-1]]:
            found_rows = get_rows(top_k, json.loads(c_str))
            if len(found_rows) > 0:
                assert len(found_rows) == 1
                new_policy[scenario] = json.loads(c_str)
                base_policy_score = \
                    get_rows(
                        rows,
                        json.loads(base_policy_str[scenario])
                    ).iloc[[0]]["score"].values[0]
                found_row_score = found_rows.iloc[[0]]["score"].values[0]
                # To quiet the runtime error of doing -np.inf - (-np.inf)
                if base_policy_score == found_row_score:
                    performance_drop[scenario] = 0
                else:
                    performance_drop[scenario] = \
                        base_policy_score - found_row_score
                found_config = True
                break
        assert found_config, scenario
    return new_policy, performance_drop


def oracle_from_df(configs_scores, config_subset=None):
    if config_subset is not None:
        assert set(config_subset.columns).issubset(
            set(config_knobs)), "config_subset has unexpected columns"
        assert not np.any(config_subset.duplicated(
            config_subset.columns)), "config subset has duplicate rows"
        configs_scores = configs_scores.merge(config_subset,
                                              on=list(config_subset.columns))

    final_policy = {}
    for scenario, rows in configs_scores.groupby(["run"]):
        best_row = rows.iloc[rows["score"].argmax()]
        best_config = {k: best_row[k] for k in config_knobs}
        final_policy[scenario] = best_config
    return final_policy


def get_oracle_policy(time_mode: str,
                      infinite_mode: bool,
                      num_sectors: int,
                      config_subset=None):
    """
    Assumes column called "run" and column called "score"
    config_subset is a dataframe with the subset of the configs to be
    considered. Its columns are a subset of the config names
    """
    configs_scores = cache_score_dataset(time_mode,
                                         infinite_mode,
                                         num_sectors,
                                         score="mota-num",
                                         remove_inf_nan=True)
    return oracle_from_df(configs_scores, config_subset)


def get_condensed_policy(time_mode: str, infinite_mode: bool, num_sectors: int,
                         k: int):
    oracle_policy = get_oracle_policy(time_mode=time_mode,
                                      infinite_mode=infinite_mode,
                                      num_sectors=num_sectors)
    configs_scores = cache_score_dataset(time_mode=time_mode,
                                         infinite_mode=infinite_mode,
                                         num_sectors=num_sectors,
                                         score="mota",
                                         remove_inf_nan=True)
    return greedy_condense_space(configs_scores, oracle_policy, k=k)[0]


def create_sequential_policy(scenario, starting_configuration, df_predictions):
    """
    Assumes there's columns "scenario_names", "prediction".
    Creates a sequence of configurations that a greedy policy would choose to
    use in arg:scenario to optimize the next step's score using the scores in
    df_predictions.
    """
    num_segments = max(df_predictions["scenario_name"].map(
        lambda x: int(x.split("-P")[-1].split("_")[1])))
    final_policy = {f"{scenario}-P0_{num_segments}": starting_configuration}
    previous_config = starting_configuration
    for i in range(1, num_segments):
        current_scenario = f"{scenario}-P{i}_{num_segments}"
        pastC_config = {f"pastC-{k}": v for k, v in previous_config.items()}
        filter_columns = {**pastC_config, "scenario_name": current_scenario}
        rows = get_rows(df_predictions, filter_columns)
        if len(rows) > 0:
            best_row = rows.iloc[rows["prediction"].argmax()]
            best_config = {k: best_row[k] for k in config_knobs}
        else:
            print(f"{current_scenario} doesn't exist in the data fed!")
            best_config = previous_config
        final_policy[current_scenario] = best_config
        previous_config = best_config
    return final_policy


def evaluate_fine_grained_policy(policy_map,
                                 time_mode,
                                 default_config=None,
                                 zero_first_n_frames=3,
                                 debug_desc="",
                                 other_metrics=False):
    """
    Default config is used when a particular scenario segment doesn't have
    a matching config. This function will complain when this happens and
    impute the default config. If None is provided the function will fail.

    For info on the zero_first_n_frames arg, see docstring of cache_num_gt.
    """
    sample_key = list(policy_map.keys())[0]
    assert "-P" in sample_key, \
        f"policy passed is not per segment: {sample_key}"
    all_configs = list(policy_map.values())
    num_segments = int(sample_key.split("-P")[1].split("_")[1])
    all_scenarios = [x.split("-P")[0] for x in policy_map.keys()]
    all_segment_indices = [
        int(x.split("-P")[1].split("_")[0]) for x in policy_map.keys()
    ]
    # use pandas dataframe for access to groupby functionality
    df_index = pd.DataFrame({
        "scenario": all_scenarios,
        "segment_index": all_segment_indices,
        "configs_index": range(len(all_scenarios))
    })
    policy_scores = {}
    num_gt_matrix_index, num_gt_matrix = cache_num_gt(
        time_mode,
        zero_first_n_frames=zero_first_n_frames,
        other_metrics=other_metrics)
    for scenario, rows in tqdm(
            df_index.groupby(["scenario"]),
            desc=f"evaluate_fine_grained_policy: {debug_desc}"):
        rows = rows.sort_values(by=["segment_index"], ascending=True)
        found_indices = list(rows["segment_index"])
        config_indices = list(rows["configs_index"])
        if len(rows) != num_segments:
            complaint = f"only found {found_indices} of scenario {scenario}"
            if default_config is None:
                raise Exception(complaint)
            else:
                print(complaint)
        config_sequence = [None] * num_segments
        for segment_index, config_index in zip(found_indices, config_indices):
            config_sequence[segment_index] = all_configs[config_index]
        for i in range(num_segments):
            config_sequence[i] = config_sequence[i] or default_config
        mota_score = evaluate_configuration_sequence(
            config_sequence,
            scenario,
            time_mode,
            cached=(num_gt_matrix_index, num_gt_matrix),
            other_metrics=other_metrics)
        policy_scores[scenario] = mota_score

    return policy_scores


def po_past_scores(exp_dict):
    best_average_config = highest_avg_config(exp_dict)

    test_copy = exp_dict["test"].copy()
    test_copy["scenario_name"] = test_copy.index
    test_copy["prediction"] = exp_dict["model"].predict(exp_dict["test"])
    test_copy = contract_one_hot(test_copy)

    assert len(
        get_rows(test_copy, best_average_config)
    ) > 0, "best average config from train set not found in test set"

    final_policy = {}
    # iterate over full scenarios
    full_scenarios = \
        test_copy["scenario_name"].map(lambda x: x.split("-P")[0]).unique()
    for scenario in tqdm(full_scenarios, desc="po_past_scores"):
        scenario_policy = create_sequential_policy(scenario,
                                                   best_average_config,
                                                   test_copy)
        final_policy.update(scenario_policy)

    return evaluate_fine_grained_policy(
        final_policy,
        time_mode=exp_dict["metadata"]["time_mode"],
        debug_desc="po_past")


def highest_avg_config(exp_result: ExpResult):
    """
    Computes only over training dataset as it should be.
    Only focuses on the configuration knobs listed at the top of the file and
    the scenario columns for the sake of computing the average score. All
    other columns are ignored.
    """
    train_copy = exp_result.train_df
    # only taking the configurations in the training dataset, in case training
    # was done on a subset of the configurations (not the full space).
    # essentially ignore other columns, because config_knobs+run are repeated
    # when they were joined with environment conditions before training.
    training_configs = train_copy.drop_duplicates(
        subset=config_knobs)[config_knobs]
    best_average_config = get_average_config(
        train_copy.index.unique(),
        time_mode=exp_result.time_mode,
        infinite_mode=exp_result.infinite_mode,
        config_subset=training_configs)
    return best_average_config


def gt_scores(exp_result: ExpResult, other_metrics=False):
    '''
        Argmax on regression to get best policy
    '''
    final_policy = oracle_from_df(exp_result.test_df)
    avg_config = highest_avg_config(exp_result)
    return evaluate_fine_grained_policy(final_policy,
                                        time_mode=exp_result.time_mode,
                                        default_config=avg_config,
                                        other_metrics=other_metrics)


def po_present_scores(exp_dict):
    test_copy = exp_dict["test"].copy()
    test_copy["scenario_name"] = test_copy.index
    test_copy["prediction"] = exp_dict["model"].predict(exp_dict["test"])
    test_copy = contract_one_hot(test_copy)

    final_min_max_policy = {}
    final_max_max_policy = {}
    for scenario, scenario_rows in tqdm(test_copy.groupby(["scenario_name"])):
        best_config_and_score = []
        past_knobs = [f'pastC-{k}' for k in config_knobs]
        for config_tup, config_rows in scenario_rows.groupby(past_knobs):
            best_row = config_rows.iloc[config_rows["prediction"].argmax()]
            best_config = {k: best_row[k] for k in config_knobs}
            best_score = best_row["prediction"]
            best_config_and_score.append((best_config, best_score))
        sorted_config_scores = \
            sorted(best_config_and_score, key=lambda x: x[1])
        final_min_max_policy[scenario] = sorted_config_scores[0][0]
        final_max_max_policy[scenario] = sorted_config_scores[-1][0]

    avg_config = highest_avg_config(exp_dict)
    min_max_policy_scores = evaluate_fine_grained_policy(
        final_min_max_policy,
        time_mode=exp_dict["metadata"]["time_mode"],
        default_config=avg_config)
    max_max_policy_scores = evaluate_fine_grained_policy(
        final_max_max_policy,
        time_mode=exp_dict["metadata"]["time_mode"],
        default_config=avg_config)
    return min_max_policy_scores, max_max_policy_scores


def get_average_config(train_scenarios,
                       time_mode: str,
                       infinite_mode: bool,
                       config_subset: pd.DataFrame = None):
    df_config_1sector = cache_score_dataset(time_mode,
                                            infinite_mode,
                                            num_sectors=1,
                                            score="mota",
                                            remove_inf_nan=True)
    assert np.all(df_config_1sector["run"].apply(
        lambda x: x.split("-P")[1].split("_")[1]) == "1"), (
            "average configuration should be evaluated on scores "
            "assigned to entire videos")

    if config_subset is not None:
        assert set(config_subset.columns).issubset(
            set(config_knobs)), "config_subset has unexpected columns"
        assert not np.any(config_subset.duplicated(
            config_subset.columns)), "config subset has duplicate rows"
        df_config_1sector = df_config_1sector.merge(config_subset,
                                                    on=list(
                                                        config_subset.columns))
    whole_train_scenarios = np.unique(
        [s.split("-P")[0] for s in train_scenarios])
    df_config_1sector = df_config_1sector[df_config_1sector["run"].map(
        lambda x: x.split("-P")[0]).isin(whole_train_scenarios)]

    def remove_bad_values(array):
        return array[np.isfinite(array)]

    assert not df_config_1sector.duplicated(config_knobs + ["run"]).any(
    ), "duplicate configuration entries found in score dataframe"

    config_score = [
        (config_tup, np.mean(remove_bad_values(rows["score"])))
        for config_tup, rows in df_config_1sector.groupby(config_knobs)
    ]
    best_config_tup = sorted(config_score, key=lambda x: x[1])[-1][0]
    avg_config = dict(zip(config_knobs, best_config_tup))
    return avg_config


def average_config_score(
    train_scenarios,
    test_scenarios,
    time_mode: str,
    infinite_mode: bool,
):
    """
    df_config_1sector: The dataframe that maps from configuration to score
    """
    avg_config = get_average_config(train_scenarios,
                                    time_mode=time_mode,
                                    infinite_mode=infinite_mode)
    policy = {k: avg_config for k in test_scenarios}
    return evaluate_fine_grained_policy(policy,
                                        time_mode=time_mode,
                                        default_config=avg_config)


def evaluate_configuration_sequence(config_sequence,
                                    scenario,
                                    time_mode,
                                    cached=None,
                                    other_metrics=False):
    """
    config_sequence is a list of configurations applied in a sequence at
    sequential, window-sized stride, fractions of the video.
    """
    # # # collect gt_num sequence
    # decide on split of num_gt_array based on number of configs in
    # config_sequence

    num_gt_matrix_index, num_gt_matrix = cached or cache_num_gt(
        time_mode, other_metrics=other_metrics)
    num_frames = num_gt_matrix.shape[-1]
    array_split = \
        np.arange(0, num_frames+1, num_frames // len(config_sequence))
    num_gt_array_segments = []
    for config, (start, end) in zip(config_sequence,
                                    zip(array_split[:-1], array_split[1:])):
        rows = get_rows(num_gt_matrix_index, {**config, "run": scenario})
        assert len(rows) == 1, (len(rows), config, scenario)
        num_gt_array = num_gt_matrix[rows.index[0]]
        num_gt_array_segment = num_gt_array[:, start:end]
        num_gt_array_segments.append(num_gt_array_segment)
    final_num_gt_array = np.concatenate(num_gt_array_segments, axis=1)
    # # # accumulate to get mota score
    if other_metrics:
        metrics = compute_other_tracking_metrics(
            np.expand_dims(final_num_gt_array, axis=0), 1)
        return metrics
    else:
        mota = compute_mota_scores(np.expand_dims(final_num_gt_array, axis=0),
                                   1)
        mota = mota[0, 0]
        return mota
