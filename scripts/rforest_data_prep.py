"""
Random forest + hand-picked features experiment script. At the moment only
supports MOTA score regression + argmax. The driving video is broken into
equal sized segments and the model has to predict the score of a candidate
configuration on the segment.
Background:
 * The model learns to map E x C -> score, where E is environment
   representation and C is candidate configuration.
 * The hand-picked features are certain aggregate statistics about the bbox
evolution throughout the video (average box size, average change in bbox size,
avg bbox frame lifespan, etc)

Key experiment vars:
> features_source
gt = hand-picked features derived using ground-truth labels of video
po = hand-picked features derived using the pipeline's previous output. We
     train so that the model receives (C' x E), both the pipeline configuraiton
     when it generated the bounding boxes, and E, the hand-picked features
     derived from said bounding box predictions. C' is what the "pastC-{}"
     features represent.
po-global-static = like po, except that the previous previous configuration
                   used to derive the features is always the global static
                   configuration (of the train dataset), regardless of what
                   configuration was actually picked. As a result, in this mode
                   we drop C', since it's always the same.

> feature time
present: features are derived from aggregating bbox stats of the same video
         segment on which the configuration choice is evaluated.
past-1-step: features derived from the video segment coming before the one
             that the model is evaluated on. This is the realistic setting
             in a car. It is also limited because we don't use the past further
             back.

> separate_sync_timely_mode
Changes the behavior of the target score to be trained on. When False, all
flags behave as expected. When True, if time_mode = "timely", the target score
becomes the degradation. If time_mode = "sync" the target score becomes the
offline accuracy drop. This mode is currently only supported in gt, present
settings.


"""

import json

import numpy as np
import pandas as pd
from absl import app, flags
from ad_config_search.data_curation import shift_rows_back_one
from ad_config_search.data_processing_utils import (
    cache_features_from_pipeline, cache_score_dataset, get_ground_truth)
from ad_config_search.policy_utils import (config_knobs, get_average_config,
                                           get_condensed_policy)
from ad_config_search.rforest_training import (MetadataV1,
                                               full_training_process_v1)
from ad_config_search.rforest_utils import (split_scenarios,
                                            waymo_official_train_val_test)
from ad_config_search.utils import get_rows
from icecream import ic
from tqdm import tqdm

flags.DEFINE_bool(
    'subtract_baseline', False,
    ("change task to predict score difference from global static "
     "configuration, instead of predicting the global score"))
flags.DEFINE_enum('stride_strategy',
                  'uniform_window_skip', ['uniform_window_skip'],
                  help="when the policy chooses to run")
flags.DEFINE_integer('num_sectors',
                     default=None,
                     required=True,
                     help="number of sectors to divide videos into")
flags.DEFINE_enum('features_source',
                  default=None,
                  enum_values=["po", "gt", "po-global-static"],
                  required=True,
                  help="how features are derived")
flags.DEFINE_enum('features_time',
                  default=None,
                  enum_values=["present", "past-1-step"],
                  required=True,
                  help="collect features from present, or past")
flags.DEFINE_enum('time_mode',
                  default='timely',
                  enum_values=['timely', 'sync'],
                  help="sync or timely")
flags.DEFINE_bool('infinite_mode',
                  default=False,
                  help="use infinite GPU mode or not")
flags.DEFINE_bool('remove_inf_nan',
                  default=True,
                  help="remove rows in the dataset with inf or nan scores")
flags.DEFINE_bool(
    'separate_sync_timely_mode',
    default=False,
    help="implements the changes needed for the separate prediction experiment"
)
flags.DEFINE_enum(
    'condensing_strategy',
    default=None,
    enum_values=["greedy_v1_k=10"],
    help="whether and how to condense the configuration space of the dataset")
flags.DEFINE_multi_integer(
    'score_clip_range',
    default=[],
    required=False,
    help=("two values representing the minimum and maximum value "
          "to clip dataset scores to"))
flags.DEFINE_multi_enum(
    'knob_subset',
    default=config_knobs,
    enum_values=config_knobs,
    help=("subset of knobs to optimize. The rest are frozen using "
          "config_freeze_strategy."))
flags.DEFINE_enum('dataset_split_strategy', None, ['waymo', 'random'],
                  "The strategy used to split the dataset to train and test")
flags.DEFINE_float('dataset_split_seed',
                   default=0.43,
                   help="seed to control random train-test split")
flags.DEFINE_enum(
    'config_freeze_strategy',
    default=None,
    enum_values=['uniform_random', 'train_static_config'],
    help=("the configuration freezing strategy for knobs not considered "
          "for optimization. Can either be random or the global static "
          "configuration derived from the training dataset."))
flags.DEFINE_float('config_freeze_seed',
                   default=0.43,
                   help="seed to control configuration freeze if it's on")
flags.DEFINE_integer('model_seed',
                     default=50,
                     help="seed to control model training")
flags.DEFINE_integer('hp_search_iters',
                     default=20,
                     help="number of iterations to run hp search on")


def positive_hp_search_iters(flags_dict):
    return flags_dict['hp_search_iters'] > 0


def really_simple_config_past(flags_dict):
    return (flags_dict['features_source'] != 'po-global-static') or (
        flags_dict['config_freeze_strategy'] is None
        and flags_dict['knob_subset'] == config_knobs
        and flags_dict['condensing_strategy'] is None)


def check_positive_num_sectors(flags_dict):
    return flags_dict['num_sectors'] > 0


def validate_clip_range(flags_dict):
    if len(flags_dict["score_clip_range"]) == 0:
        return True
    elif len(flags_dict["score_clip_range"]) != 2:
        return False
    else:
        return flags_dict["score_clip_range"][1] > flags_dict[
            "score_clip_range"][0]


def validate_separate_sync_timely_mode(flags_dict):
    if flags_dict["separate_sync_timely_mode"]:
        return flags_dict["features_source"] == "gt" and flags[
            "features_time"] == "present"
    else:
        return True


flags.register_multi_flags_validator(
    ['hp_search_iters'],
    positive_hp_search_iters,
    message=("number of hp search iterations must be positive"))

flags.register_multi_flags_validator(
    [
        'features_source', 'config_freeze_strategy', 'knob_subset',
        'condensing_strategy'
    ],
    really_simple_config_past,
    message=("Don't support configuration space manipulation while using "
             "past pipeline output fixed to global static configuration"))

flags.register_multi_flags_validator(
    ["score_clip_range"],
    validate_clip_range,
    message=("if specified, clip range should be exactly two values, "
             "the second bigger than the first"))

flags.register_multi_flags_validator(["num_sectors"],
                                     check_positive_num_sectors,
                                     message=("num_sectors must be positive"))

flags.register_multi_flags_validator(
    ["time_mode", "separate_sync_timely_mode", "features_time"],
    validate_separate_sync_timely_mode,
    message=("separate_sync_timely_mode is only supported when "
             "for present, gt settings."))

FLAGS = flags.FLAGS


def filter_df_by_configs(df, configs, set_to_past=False):
    """
    see docstring at top of file for explanation.
    """
    if set_to_past:
        configs = \
            [{"pastC-{}".format(k): v for k, v in c.items()} for c in configs]
    return pd.concat([get_rows(df, c) for c in tqdm(configs)])


def get_oracle_k_10_configs(configs_scores, train_scenarios):
    scenario_groupby = configs_scores.groupby(
        [configs_scores["run"].map(lambda x: x.split("-P")[0])])
    configs_scores_train = \
        pd.concat([rows for scenario, rows in scenario_groupby
                   if scenario in train_scenarios])
    oracle_k_10_policy = get_condensed_policy(configs_scores_train, k=10)
    oracle_k_10_configs = \
        [json.loads(s) for s in
         np.unique([json.dumps(c) for c in oracle_k_10_policy.values()])]
    return oracle_k_10_configs


def po_global_static_config_features(chunk_size, train_scenarios):
    po_features = cache_features_from_pipeline(
        chunk_size, infinite_mode=FLAGS.infinite_mode)
    time_mode = ("timely"
                 if FLAGS.separate_sync_timely_mode else FLAGS.time_mode)
    avg_policy = get_average_config(train_scenarios,
                                    time_mode=time_mode,
                                    infinite_mode=FLAGS.infinite_mode,
                                    config_subset=None)
    avg_policy_rows = get_rows(
        po_features, {f"pastC-{k}": v
                      for k, v in avg_policy.items()})
    avg_policy_rows = avg_policy_rows.drop(
        columns=[c for c in avg_policy_rows.columns if c.startswith("pastC-")])
    return avg_policy_rows


def get_experiment_data():
    chunk_size = 20 // FLAGS.num_sectors
    # TRAIN TEST SPLIT
    if FLAGS.separate_sync_timely_mode and FLAGS.time_mode == "timely":
        config_scores = cache_score_dataset(
            time_mode="timely",
            infinite_mode=FLAGS.infinite_mode,
            num_sectors=FLAGS.num_sectors,
            score="degradation",
            remove_inf_nan=FLAGS.remove_inf_nan)
    else:
        config_scores = cache_score_dataset(
            time_mode=FLAGS.time_mode,
            infinite_mode=FLAGS.infinite_mode,
            num_sectors=FLAGS.num_sectors,
            score="mota",
            remove_inf_nan=FLAGS.remove_inf_nan)
    config_scores = clip_scores(config_scores)
    split_fn = {
        "waymo": waymo_official_train_val_test,
        "random": lambda x: split_scenarios(x, 1 / 3, FLAGS.dataset_split_seed)
    }[FLAGS.dataset_split_strategy]
    train_scenarios, test_scenarios = split_fn(
        config_scores["run"].map(lambda x: x.split("-P")[0]).unique())
    # CHOOSE ENVIRONMENT FEATURES
    env_features = {
        "po":
        lambda: cache_features_from_pipeline(
            chunk_size, infinite_mode=FLAGS.infinite_mode),
        "po-global-static":
        lambda: po_global_static_config_features(chunk_size, train_scenarios),
        "gt":
        lambda: get_ground_truth(chunk_size)
    }[FLAGS.features_source]
    env_features = env_features()
    env_feature_list = sorted(list(env_features.columns))
    # SHIFT BACK FEATURES IF NECESSARY
    if FLAGS.features_time == "past-1-step":
        env_features = shift_rows_back_one(env_features, "scenario_name")
    # CONDENSE CONFIGURATION SPACE
    frozen_config, config_scores, env_features = freeze_knobs(
        config_scores, env_features, train_scenarios)
    if FLAGS.condensing_strategy == "greedy_v1_k=10":
        oracle_k_10_configs = \
            get_oracle_k_10_configs(config_scores, train_scenarios)
        ic(len(oracle_k_10_configs))
        config_scores = filter_df_by_configs(config_scores,
                                             oracle_k_10_configs)
        # in addition to pruning the configuration space from the scores df
        # above, we also prune the configuration used in the pipeline-output
        # derived environment dataframe (C' x E), see top-of-file docstring.
        if FLAGS.features_source == "po":
            env_features = filter_df_by_configs(env_features,
                                                oracle_k_10_configs,
                                                set_to_past=True)
    if FLAGS.subtract_baseline:
        config_scores = subtract_baseline(config_scores, train_scenarios)
    # JOIN SCORE AND ENV FEATURES TOGETHER
    ic(config_scores)
    ic(env_features)
    joined_df = config_scores.set_index('run').join(
        env_features.set_index('scenario_name'),
        lsuffix="_config_df",
        rsuffix="_scenario_df",
        how="inner")
    joined_df = pd.get_dummies(joined_df, prefix_sep="__")
    joined_df["scenario_name"] = \
        joined_df.index.map(lambda x: "-".join(x.split("-")[:2]))
    return {
        "joined_df": joined_df,
        "train_scenarios": train_scenarios,
        "test_scenarios": test_scenarios,
        "frozen_config": frozen_config,
        "env_feature_list": env_feature_list
    }


def clip_scores(config_scores):
    if FLAGS.score_clip_range is not None:
        config_scores["score"] = config_scores["score"].clip(
            *FLAGS.score_clip_range)
    return config_scores


def subtract_baseline(df_config, train_scenarios):
    from ad_config_search.transformations import SubtractBaseline
    just_train_df = df_config[df_config["run"].map(
        lambda x: x.split("-P")[0]).isin(train_scenarios)]
    time_mode = ("timely"
                 if FLAGS.separate_sync_timely_mode else FLAGS.time_mode)
    transform = SubtractBaseline(time_mode, FLAGS.infinite_mode)
    transform.fit(just_train_df)
    return transform.transform(df_config)


def freeze_knobs(config_scores, env_features, train_scenarios):
    import random
    knobs_to_freeze = sorted(list(set(config_knobs) - set(FLAGS.knob_subset)))
    if len(knobs_to_freeze) == 0:
        assert FLAGS.config_freeze_strategy is None, \
            FLAGS.config_freeze_strategy
        return None, config_scores, env_features
    if FLAGS.config_freeze_strategy == "uniform_random":
        random.seed(FLAGS.config_freeze_seed)
        # first generating and then deleting unwanted knobs in order to ensure
        # that a seed will correspond to the same random point in the
        # configuration. Otherwise random will produce different results
        # because it is not run in exactly the same way, depending on the
        # choice of knob subset to freeze.
        frozen_config = {
            k: random.choice(config_scores[k].unique())
            for k in config_knobs
        }
    elif FLAGS.config_freeze_strategy == "train_static_config":
        frozen_config = get_average_config(train_scenarios,
                                           time_mode=FLAGS.time_mode,
                                           infinite_mode=FLAGS.infinite_mode)
    frozen_config = {
        k: v
        for k, v in frozen_config.items() if k in knobs_to_freeze
    }
    small_df_configs_5sectors = get_rows(config_scores, frozen_config)
    ic(frozen_config)
    if FLAGS.features_source == "po":
        ic({k: env_features[f"pastC-{k}"].unique() for k in config_knobs})
        env_features = get_rows(
            env_features, {f"pastC-{k}": v
                           for k, v in frozen_config.items()})

    return frozen_config, small_df_configs_5sectors, env_features


def main(argv):
    # execute only if run as a script
    experiment_data = get_experiment_data()
    joined_df = experiment_data["joined_df"]
    train_scenarios = experiment_data["train_scenarios"]
    test_scenarios = experiment_data["test_scenarios"]
    frozen_config = experiment_data["frozen_config"]
    env_feature_list = experiment_data["env_feature_list"]
    ic(joined_df)
    full_feature_name = {
        "po": "pipeline-output-v1",
        "gt": "ground-truth-v4",
        "po-global-static": "po-global-static"
    }[FLAGS.features_source]
    flags_dict = FLAGS.flag_values_dict()
    flags_dict.update({
        "window_length":
        20 // FLAGS.num_sectors,
        "model_type":
        "RF",
        "frozen_config":
        frozen_config,
        "full_feature_name":
        full_feature_name,
        "feature-list":
        env_feature_list,
        "seeds":
        (FLAGS.dataset_split_seed, FLAGS.model_seed, FLAGS.config_freeze_seed),
    })
    # converting lists to tuples so they're hashable, needed for automatic
    # plotting
    flags_dict = {
        k: tuple(v) if type(v) == list else v
        for k, v in flags_dict.items()
    }
    ic(flags_dict)
    fn_configs = [
        "model_type", "window_length", "time_mode", "infinite_mode",
        "features_time", "features_source", "subtract_baseline",
    ]
    if FLAGS.separate_sync_timely_mode:
        fn_configs.append("separate_sync_timely_mode")
    metadata = MetadataV1(flags_dict,
                          fn_configs=fn_configs)
    full_training_process_v1(joined_df, train_scenarios, test_scenarios,
                             metadata, FLAGS.model_seed)


if __name__ == '__main__':
    app.run(main)
