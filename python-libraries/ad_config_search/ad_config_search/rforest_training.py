import pickle
import time

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
# from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from tune_sklearn import TuneSearchCV

from ad_config_search.rforest_utils import ScenarioAwareCVSplitter
from ad_config_search.policy_utils import ExpResult
from ad_config_search.utils import contract_one_hot

RSEED = 50


def train_test_split_across_scenarios(original_df: pd.DataFrame,
                                      train_scenarios: list,
                                      test_scenarios: list):
    """
    original_df is a pandas DataFrame that should have the following two
    columns:
      - "scenario_name" (no quotes) that indicates the scenario of each
      datapoint (used to split dataset across scenarios)
      - "score" (no quotes) that indicates the label of each datapoint
    """
    df = original_df.copy()

    # report stats on number of rows per scenario
    counts = df["scenario_name"].value_counts()
    stats_str = ("min: {}  | p10: {}  | p50:  {}"
                 "  | p90:  {}  | max:  {}").format(np.min(counts),
                                                    np.percentile(counts, 10),
                                                    np.percentile(counts, 50),
                                                    np.percentile(counts, 90),
                                                    np.max(counts))
    print("Stats of # of rows per scenario_name. Important to achieve desired "
          "train/test split.")
    print(stats_str)

    labels = np.array(df.pop('score'))

    train = df[df["scenario_name"].isin(train_scenarios)]
    train_labels = labels[df["scenario_name"].isin(train_scenarios)]
    test = df[df["scenario_name"].isin(test_scenarios)]
    test_labels = labels[df["scenario_name"].isin(test_scenarios)]

    train = train.drop(columns=["scenario_name"])
    test = test.drop(columns=["scenario_name"])

    return train, train_labels, test, test_labels


def hyperparam_search(train: pd.DataFrame,
                      train_labels: list,
                      num_iterations: int,
                      seed=RSEED):
    """
    Random logspace search, k=5 fold cross validation, 20 trials.
    """
    model = RandomForestRegressor(n_estimators=1600,
                                  random_state=seed,
                                  max_features='sqrt',
                                  n_jobs=-1,
                                  verbose=1)
    print("num iterations: {}".format(num_iterations))
    clf = TuneSearchCV(
        model,
        {
            "max_depth": randint(1, 25),
            "max_features": randint(2, 24),
            "n_estimators": [200, 400],  # 800, 1600, 2400],
            "min_impurity_decrease": loguniform(0.00005, 0.05, scale=2)
        },
        cv=ScenarioAwareCVSplitter(n_splits=5, shuffle=True,
                                   random_state=seed),
        random_state=seed,
        search_optimization="random",
        # n_iter=num_iterations,
        n_trials=num_iterations,
        n_jobs=1)
    print(clf)

    # an attempt to get a ray backend for joblib... Using tune-sklearn instead.
    # import joblib
    # from ray.util.joblib import register_ray
    # register_ray()
    # with joblib.parallel_backend('ray'):

    clf.fit(train, train_labels)

    print(f"Best score: {clf.best_score_}")
    print(f"Best parameters found: {clf.best_params_}")
    return clf


class MetadataV1:
    """
    configs_dict has all of the config names and values for the experiment

    fn_configs is a list of the configs that would show up in the saved file's
    name.
    """
    def __init__(self, configs_dict: dict, fn_configs: list):
        self.configs_dict = configs_dict
        self.fn_configs = fn_configs
        not_in_configs = list(
            filter(lambda x: x not in configs_dict.keys(), fn_configs))
        print(configs_dict.keys())
        assert len(not_in_configs) == 0, not_in_configs


def get_stats(model, train: pd.DataFrame, train_labels: list,
              test: pd.DataFrame, test_labels: list):
    """
    Computes statistics of trained model
    """
    # FEATURE IMPORTANCE
    features = train.columns
    sorted_feature_importance = \
        sorted(zip(features, model.feature_importances_),
               key=(lambda f: f[1]),
               reverse=True)
    print("Feature importance")
    print(sorted_feature_importance)

    # FOREST SIZE
    n_nodes = []
    max_depths = []

    # Stats about the trees in random forest
    for ind_tree in model.estimators_:
        n_nodes.append(ind_tree.tree_.node_count)
        max_depths.append(ind_tree.tree_.max_depth)

    tree_avg_num_nodes = int(np.mean(n_nodes))
    tree_avg_max_depth = int(np.mean(max_depths))

    print(f'Average number of nodes {tree_avg_num_nodes}')
    print(f'Average maximum depth {tree_avg_max_depth}')

    # Training predictions (to demonstrate overfitting)
    train_score = model.score(train, train_labels)
    print("train score", train_score)

    # Testing predictions (to determine performance)
    test_score = model.score(test, test_labels)
    print("test score", test_score)

    cached_model_stats = {
        "train-score": train_score,
        "test-score": test_score,
        "tree-avg-num-nodes": tree_avg_num_nodes,
        "tree-avg-max-depth": tree_avg_max_depth,
        "sorted-feature-importance": sorted_feature_importance
    }
    return cached_model_stats


def save_experiment(model, train: pd.DataFrame, train_labels: list,
                    test: pd.DataFrame, test_labels: list,
                    metadata: MetadataV1, cached_model_stats: dict):
    """
    Saves all the model, the data needed to reproduce the training, and cached
    stats in a pl file in the current directory.
    """
    # SAVING MODEL
    dictionary = {
        "model": model,
        "train": train,
        "train_labels": train_labels,
        "test": test,
        "test_labels": test_labels,
        "metadata": metadata.configs_dict,
        "cached_model_stats": cached_model_stats
    }

    # BUILDING EXPERIMENT FILE NAME
    fname_dict = {k: metadata.configs_dict[k] for k in metadata.fn_configs}
    test_score = cached_model_stats["test-score"]
    fname_dict = {**fname_dict, "te": np.round(test_score, 2)}
    fname = "__".join(f"{k}={v}" for k, v in fname_dict.items())
    exp_time = int(time.time() * 1e3)
    fname = f"exp_result_{exp_time}--" + fname
    with open(f"{fname}.pl", 'wb') as f:
        pickle.dump(exp_result_from_rforest_exp_dict(dictionary), f)

    return dictionary


def exp_result_from_rforest_exp_dict(exp_dict):
    """
    The column schema that the model expects as vectorized input is recorded in
    other_data_dump.
    """
    train = exp_dict["train"].copy()
    train = contract_one_hot(train)
    train.index.name = "run"
    train["score"] = exp_dict["train_labels"]
    test = exp_dict["test"].copy()
    test = contract_one_hot(test)
    test.index.name = "run"
    test["score"] = exp_dict["model"].predict(exp_dict["test"])
    return ExpResult(
        train,
        test,
        time_mode=exp_dict["metadata"]["time_mode"],
        infinite_mode=exp_dict["metadata"]["infinite_mode"],
        num_sectors=exp_dict["metadata"]["num_sectors"],
        model_parameters=exp_dict["model"],
        metadata=exp_dict["metadata"],
        other_data_dump={"column_schema": exp_dict["train"].columns})


def full_training_process_v1(original_df: pd.DataFrame, train_scenarios: list,
                             test_scenarios: list, metadata: MetadataV1,
                             model_seed):
    train, train_labels, test, test_labels = \
        train_test_split_across_scenarios(original_df,
                                          train_scenarios, test_scenarios)

    search_result = hyperparam_search(
        train,
        train_labels,
        num_iterations=metadata.configs_dict["hp_search_iters"],
        seed=model_seed)

    model = search_result.best_estimator_

    stats = get_stats(model, train, train_labels, test, test_labels)

    return save_experiment(model, train, train_labels, test, test_labels,
                           metadata, stats)
