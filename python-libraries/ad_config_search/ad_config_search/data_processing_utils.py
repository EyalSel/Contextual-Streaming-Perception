"""
This file contains functionality to derive, caching where possible, the dataset
components used for downstream training. The intention is for this file to be
the API between the ML code and the raw data, rather than direct interaction
with the rest of the utils files.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import ray
from tqdm import tqdm

from ad_config_search.data_curation import (data_csv_paths, get_fn_without_ext,
                                            get_fp_fn, get_gt_and_numerator,
                                            split_run)
from ad_config_search.evaluation_utils import (compute_mota_numerator,
                                               compute_mota_scores)
from ad_config_search.pipeline_output_features import get_final_features
from ad_config_search.utils import ray_map, ray_map_np


def cache_num_gt(time_mode="timely",
                 force_redo=False,
                 zero_first_n_frames=3,
                 other_metrics=False):
    """
    possible mode: timely, sync
    force_redo = True means to do recomputation even if files exist
    returns index df, numpy matrix

    zero_first_n_frames specifies the number of frames in the beginning of the
    video sequence for which to set the score values (along the last axis,
    video time) to zero. This is done to overcome the situation where detection
    models of varying latencies begin recording predictions at different
    starting points in the video. E.g. if the frame gap is 100ms, a 30ms
    detection model will begin giving outputs at the second frame and
    beyond, while a 250ms detection model will only begin giving outputs at
    the fourth frame. Because there are scores available for frames 2-3 in the
    faster model but none for the slower model, it could potentially create a
    situation where in sync mode the faster model gets higher video scores.
    To overcome this edge case, we simply zero out the first few frames
    uniformly to begin considering scores at a uniform point in the video.
    """
    if other_metrics:
        metric_name = "other_metrics"
    else:
        metric_name = "num_gt"
    target_index_fn, target_matrix_name = {
        "timely": (f"{metric_name}_matrix_index.pl", f"{metric_name}_matrix"),
        "sync":
        (f"sync_{metric_name}_matrix_index.pl", f"sync_{metric_name}_matrix")
    }[time_mode]

    if Path(target_index_fn).exists() and not force_redo:
        print(f"Using cached cache_num_gt with time_mode {time_mode}")
        with open(target_index_fn, 'rb') as f:
            num_gt_index = pickle.load(f)
        num_gt_matrix = np.load(target_matrix_name + ".npy")
        num_gt_matrix[:, :, :zero_first_n_frames] = 0
        return num_gt_index, num_gt_matrix

    all_csvs = data_csv_paths("../../ad-config-search/")

    df_configs = pd.DataFrame([{
        **split_run(get_fn_without_ext(x)), "index": i
    } for i, x in enumerate(all_csvs)])

    df_configs = df_configs.drop(columns=["T-ignore-len"])
    df_configs = df_configs.drop(columns=["res-h"])
    df_configs = df_configs.drop(columns=["res-w"])
    df_configs = df_configs.drop(columns=["dataset"])

    df_configs["T-max-age"] = df_configs["T-max-age"].astype(int)
    df_configs["T-every-nth-det"] = df_configs["T-every-nth-det"].astype(int)
    df_configs["T-min-iou"] = df_configs["T-min-iou"].astype(float)
    df_configs["D-conf"] = df_configs["D-conf"].astype(float)
    df_configs["D-seq-pol"] = df_configs["D-seq-pol"].astype("category")
    df_configs["D-model"] = df_configs["D-model"].astype("category")
    df_configs = df_configs.set_index("index")

    with open(target_index_fn, 'wb') as f:
        pickle.dump(df_configs, f)

    num_gt_index = df_configs

    ray.shutdown()
    ray.init(_redis_password="234k1234jlk2j34lkj34kj3kfjkfj",
             _temp_dir="/tmp/melih_is_using_the_default_ray_dir")

    def sync_gt_numerator(csv_path):
        return get_gt_and_numerator(csv_path, "sync", "end", "ceil")

    if other_metrics:
        scraper = get_fp_fn
    else:
        scraper = get_gt_and_numerator
    gen_fn = {
        "timely": lambda csv_path: scraper(csv_path, "timely", "end", "ceil"),
        "sync": lambda csv_path: scraper(csv_path, "sync", "end", "ceil")
    }[time_mode]
    num_gt_matrix = ray_map_np(all_csvs, gen_fn, 120)
    num_gt_matrix = np.squeeze(num_gt_matrix)

    assert np.all(np.isfinite(num_gt_matrix))
    assert np.all(num_gt_matrix >= 0)

    np.save(target_matrix_name, num_gt_matrix)

    num_gt_matrix[:, :, :zero_first_n_frames] = 0

    return num_gt_index, num_gt_matrix


def cache_score_dataset(time_mode,
                        infinite_mode,
                        num_sectors,
                        score="mota",
                        remove_inf_nan=True,
                        force=False):
    if score == "degradation":
        assert time_mode == "timely", (
            "score of type degradation requires timely mode")
        sync_scores = cache_score_dataset(time_mode="sync",
                                          infinite_mode=infinite_mode,
                                          num_sectors=num_sectors,
                                          score="mota",
                                          remove_inf_nan=remove_inf_nan)
        timely_scores = cache_score_dataset(time_mode="timely",
                                            infinite_mode=infinite_mode,
                                            num_sectors=num_sectors,
                                            score="mota",
                                            remove_inf_nan=remove_inf_nan)
        config_scores = sync_scores.merge(
            timely_scores,
            how="inner",
            on=list(set(sync_scores.columns) - {"score"}),
            suffixes=("_sync", "_timely"))
        config_scores["score"] = (config_scores["score_sync"] -
                                  config_scores["score_timely"])
        config_scores = config_scores.drop(
            columns=["score_sync", "score_timely"])
        return config_scores

    score_dataset_fn = (
        f"df_config_{time_mode}_{score}_{num_sectors}sectors.pl")
    if Path(score_dataset_fn).exists() and not force:
        print(f"Using cached score dataset with\n"
              f"time_mode: {time_mode}\n"
              f"infinite_mode: {infinite_mode}\n"
              f"num_sectors: {num_sectors}\n"
              f"score: {score}")
        with open(score_dataset_fn, 'rb') as f:
            df_config = pickle.load(f)
    else:
        num_gt_matrix_index, num_gt_matrix = cache_num_gt(time_mode=time_mode,
                                                          force_redo=False)
        df_config = make_score_dataframe(num_gt_matrix_index,
                                         num_gt_matrix,
                                         num_sectors,
                                         score,
                                         zero_first_n_frames=3)
        with open(score_dataset_fn, 'wb') as f:
            pickle.dump(df_config, f)

    df_config["D-conf"] = df_config["D-conf"].astype(float)
    df_config["T-min-iou"] = df_config["T-min-iou"].astype(float)
    df_config["T-max-age"] = df_config["T-max-age"].astype(int)
    df_config["T-every-nth-det"] = df_config["T-every-nth-det"].astype(int)
    df_config["score"] = df_config["score"].astype(float)
    if remove_inf_nan:
        df_config = df_config[~df_config["score"].isin([-np.inf, np.nan])]

    if infinite_mode:
        df_config = df_config[df_config["D-seq-pol"] == "infinite"]
    else:
        df_config = df_config[df_config["D-seq-pol"] != "infinite"]

    return df_config


def make_score_dataframe(df_index,
                         num_gt_matrix,
                         num_sectors,
                         score="mota",
                         zero_first_n_frames=None):
    """
    df_index: Dataframe with configuration, scenario, and an index pointing
    to the right num_gt_matrix row
    num_gt_matrix: an n x 2 x 200 matrix with the numerator and g_t values
    per frame, where n is configuration x scenario
    num_sectors: the number of sectors to divide the video segments into.

    Returns another dataframe with configuration, scenario sector, MOTA score
    columns
    """
    if zero_first_n_frames is not None:
        assert zero_first_n_frames > 0, "if value is given it must be > 0"
        num_gt_matrix[:, :, :zero_first_n_frames] = 0
    scores = {
        "mota": lambda: compute_mota_scores(num_gt_matrix, num_sectors),
        "mota-num": lambda: -compute_mota_numerator(num_gt_matrix, num_sectors)
    }[score]
    scores = scores()
    new_rows = []
    for index, row in tqdm(df_index.iterrows(), total=len(df_index)):
        for i, score in enumerate(scores[index]):
            new_rows.append({
                **dict(row), "run":
                row["run"] + "-P{}_{}".format(i, num_sectors),
                "score":
                score
            })
    result = pd.DataFrame(new_rows)
    for scenario, rows in result.groupby(["run"]):
        if np.any(rows["score"].isin([np.nan, -np.inf])):
            assert np.all(rows["score"].isin([-np.inf, np.nan])), scenario
    return result


def cache_features_from_pipeline(sector_length_s, infinite_mode):
    features_fn = f"features_from_pipeline_{sector_length_s}s_chunks.pl"
    if Path(features_fn).exists():
        print("Using cached pipeline output features with\n"
              f"sector_length_s: {sector_length_s}\n"
              f"infinite_mode: {infinite_mode}")
        features = pd.read_pickle(features_fn)
    else:
        all_jsonls = data_csv_paths("../../ad-config-search/", "jsonl")
        all_jsonls = [
            jsonl for jsonl in tqdm(all_jsonls)
            if len(list(jsonl.parent.glob("*.csv"))) == 1
        ]

        ray.shutdown()
        ray.init(_redis_password="234k1234jlk2j34lkj34kj3kfjkfj",
                 _temp_dir="/tmp/melih_is_using_the_default_ray_dir")

        all_rows = ray_map(
            all_jsonls,
            lambda jsonl: get_final_features(jsonl, sector_length_s), 120)

        import itertools
        all_rows = list(itertools.chain.from_iterable(all_rows))

        features = pd.DataFrame(all_rows)

        features = features.drop(columns=[
            "pastC-dataset", "pastC-run", "pastC-res-h", "pastC-res-w",
            "pastC-T-ignore-len", "pastC-T-model"
        ])
        for column in [
                'weather', 'location', 'time_of_day', 'scenario_name',
                'pastC-D-model', 'pastC-D-conf', 'pastC-D-seq-pol',
                'pastC-T-min-iou', 'pastC-T-max-age', 'pastC-T-every-nth-det'
        ]:
            features[column] = features[column].astype("category")

        # weather column is all the same
        features = features.drop(columns=["weather"])

        features.to_pickle(features_fn, compression=None)

    features["pastC-D-conf"] = features["pastC-D-conf"].astype(float)
    features["pastC-T-min-iou"] = features["pastC-T-min-iou"].astype(float)
    features["pastC-T-max-age"] = features["pastC-T-max-age"].astype(int)
    features["pastC-T-every-nth-det"] = features[
        "pastC-T-every-nth-det"].astype(int)

    if infinite_mode:
        features = features[features["pastC-D-seq-pol"] == "infinite"]
    else:
        features = features[features["pastC-D-seq-pol"] != "infinite"]

    return features


def get_ground_truth(sector_length_s):
    path = f"../data/scenario_features_v5_fine_{sector_length_s}s.csv"
    scenario_feats_df = pd.read_csv(path)
    scenario_feats_df = scenario_feats_df.loc[:, ~scenario_feats_df.columns.
                                              str.contains('^Unnamed')]
    # weather column is all the same
    scenario_feats_df = scenario_feats_df.drop(columns=["weather"])
    scenario_feats_df["scenario_name"] = scenario_feats_df[
        "scenario_name"].apply(lambda x: x.replace("-S_", "-S"))
    return scenario_feats_df


def gather_results():
    models_paths = list(Path(".").glob("model_type=RF*"))
    model_dicts = []
    for p in models_paths:
        with open(p, 'rb') as f:
            model_dicts.append(pickle.load(f))
    file_df = pd.DataFrame([{
        **m["metadata"], "index": i
    } for i, m in enumerate(model_dicts)])
    file_df = file_df.set_index(file_df["index"]).drop(columns=["index"])
    return file_df, model_dicts
