import pickle
from pathlib import Path

import numpy as np
from absl import app, flags
from ad_config_search.policy_utils import (ExpResult,
                                           evaluate_fine_grained_policy,
                                           get_oracle_policy, gt_scores,
                                           highest_avg_config)

flags.DEFINE_bool('other_metrics',
                  default=False,
                  help="Whether to get MOTA or the other metrics")

FLAGS = flags.FLAGS


def extract_trained_model_result(exp_result: ExpResult, index):
    """
    Returns average_score, policy_score, oracle_score, label related to the
    trained model given in exp_result.
    """
    from icecream import ic
    ic(index)
    fn = None
    label = None
    fn = gt_scores
    label = {k: v for k, v in exp_result.metadata.items()}
    print(label)
    # optim_score = round(exp_result.optim_score, 2)
    optim_score = None
    label = {**label, "te": optim_score}
    avg_config = highest_avg_config(exp_result)
    avg_policy = {k: avg_config for k in exp_result.test_df.index.unique()}
    average_score = evaluate_fine_grained_policy(
        avg_policy,
        time_mode=exp_result.time_mode,
        default_config=avg_config,
        other_metrics=FLAGS.other_metrics)
    if "classification" in exp_result.metadata and exp_result.metadata[
            "classification"]:
        from ad_config_search.policy_utils import config_knobs
        learned_policy = {
            run: dict(row[config_knobs])
            for run, row in exp_result.test_df.iterrows()
        }
        policy_score = evaluate_fine_grained_policy(
            learned_policy,
            time_mode=exp_result.time_mode,
            default_config=avg_config,
            other_metrics=FLAGS.other_metrics)
    else:
        policy_score = fn(exp_result, other_metrics=FLAGS.other_metrics)

    oracle_policy = get_oracle_policy(time_mode=exp_result.time_mode,
                                      infinite_mode=exp_result.infinite_mode,
                                      num_sectors=exp_result.num_sectors,
                                      config_subset=None)
    oracle_policy_scores = evaluate_fine_grained_policy(
        oracle_policy,
        time_mode=exp_result.time_mode,
        other_metrics=FLAGS.other_metrics)

    return average_score, policy_score, oracle_policy_scores, label


def decompose_other_metrics(policy_score):
    from more_itertools import unzip
    fp_fn_sw_scores, motp_scores = unzip(list(policy_score.values()))
    fp_fn_sw_scores, motp_scores = np.array(
        list(fp_fn_sw_scores)), list(motp_scores)
    fp_fn_sw_scores = np.squeeze(fp_fn_sw_scores)
    fp_fn_sw_scores = np.sum(fp_fn_sw_scores, axis=0)
    fp_score, fn_score, sw_score = fp_fn_sw_scores
    return {
        "fp": fp_score,
        "fn": fn_score,
        "sw": sw_score,
        "motp": np.mean(motp_scores)
    }


def main(argv):
    all_labels = []
    for i, p in list(enumerate(Path("results").glob("**/*.pl"))):
        with open(p, 'rb') as f:
            exp_result = pickle.load(f)
        (average_score, policy_score, oracle_score,
         label) = extract_trained_model_result(exp_result, i)
        if FLAGS.other_metrics:
            average_score = decompose_other_metrics(average_score)
            policy_score = decompose_other_metrics(policy_score)
            oracle_score = decompose_other_metrics(oracle_score)
        else:
            average_score = np.mean(list(average_score.values()))
            policy_score = np.mean(list(policy_score.values()))
            oracle_score = np.mean(list(oracle_score.values()))
        print(p)
        label["baseline_score"] = average_score
        label["policy_score"] = policy_score
        label["oracle_score"] = oracle_score
        all_labels.append(label)
        print("baseline_score", average_score)
        print("policy_score", policy_score)
        print("oracle_score", oracle_score)
    fname = ("all_labels_other_metrics.pl"
             if FLAGS.other_metrics else "all_labels.pl")
    with open(fname, 'wb') as f:
        pickle.dump(all_labels, f)


if __name__ == "__main__":
    app.run(main)
