import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib as mpl


def compare_policy_scores(baseline_policy_legend_triples, metric):
    """
    Plots policies as cdf of improvement per scenario over a baseline policy.

    baseline_policy_legend_triples is a list of triple of the form:
    (baseline_scores, policy_scores, legend)

    baseline_scores and policy_scores are dictionaries mapping from a scenario
    to a score. They share the same set of keys.

    metric: The metric whose score is used to compare the policies'
    configuration choice per environment, e.g. mota, ratio_mostly_tracked, etc

    legends: The label names of the cdf plots of a policy in the figure.
    """
    for baseline, policy, legend in baseline_policy_legend_triples:
        legend = ", ".join([f"{k}: {v}" for k, v in legend.items()])
        assert set(baseline.keys()) == set(policy.keys()), (
            f"baseline and policy under legend {legend} don't have the same"
            " set of scenarios")
        improvement_matrix = [policy[k] - baseline[k] for k in baseline.keys()]
        sns.ecdfplot(improvement_matrix, label=legend)
    plt.xlabel(f"{metric} improvement")
    plt.legend()


def general_compare_policy_scores_violin(baseline_policy_legend_triples,
                                         metric,
                                         not_controlled=[]):
    """
    Plots policies as cdf of improvement per scenario over a baseline policy.

    baseline_policy_legend_triples is a list of triple of the form:
    (baseline_scores, policy_scores, legend)

    baseline_scores and policy_scores are dictionaries mapping from a scenario
    to a score. They share the same set of keys.

    metric: The metric whose score is used to compare the policies'
    configuration choice per environment, e.g. mota, ratio_mostly_tracked, etc

    legends: The label names of the cdf plots of a policy in the figure.
    """
    rows = []
    y_column = f"{metric} improvement"
    for baseline, policy, legend in baseline_policy_legend_triples:
        assert set(baseline.keys()) == set(policy.keys()), (
            f"baseline and policy under legend {legend} don't have the same"
            " set of scenarios")
        new_rows = [{
            y_column: policy[k] - baseline[k],
            **legend
        } for k in baseline.keys()]
        rows.extend(new_rows)
    df = pd.DataFrame(rows)
    from functools import reduce
    if len(not_controlled) > 0:
        x_column = ", ".join(not_controlled)
        df["not_controlled"] = reduce(
            (lambda x, y: x + ", " + y),
            [df[c].astype(str) for c in not_controlled])
    else:
        df["not_controlled"] = "True"
        x_column = "not_controlled"
    exp_df = df[~df["special"].notnull()]
    #  global settings
    global_fontsize = 20
    mpl.rcParams.update(mpl.rcParamsDefault)
    params = {
        'legend.fontsize': global_fontsize,
        'axes.labelsize': global_fontsize,
        'xtick.labelsize': global_fontsize,
        'ytick.labelsize': global_fontsize,
        'font.family': 'sans-serif',
        'font.serif': ['Helvetica'],
        #           'font.weight': 'bold',
        #           'axes.labelweight': 'bold'
    }
    plt.rcParams.update(params)
    ###
    fig, (ax1, ax2) = plt.subplots(1,
                                   2,
                                   sharey=True,
                                   figsize=(20, 10),
                                   gridspec_kw={'width_ratios': [3, 1]})
    g1 = sns.violinplot(ax=ax1,
                        data=exp_df,
                        x=x_column,
                        y=y_column,
                        order=sorted(exp_df[x_column].unique()),
                        cut=0,
                        inner="quartile",
                        split=False,
                        linewidth=2,
                        palette="pastel")
    sns.pointplot(ax=ax1,
                  data=exp_df,
                  x=x_column,
                  y=y_column,
                  order=sorted(exp_df[x_column].unique()),
                  join=False,
                  ci="sd",
                  linewidth=2,
                  dodge=0.4,
                  markers="s",
                  palette="dark")
    g1.axhline(y=0, color="red", label="baseline")
    ax1.minorticks_on()
    ax1.tick_params(axis='both',
                    which='both',
                    labelsize=global_fontsize,
                    reset=True,
                    top=False,
                    right=False)
    ax1.grid(b=True, which='both')
    ax1.legend(frameon=False)
    g2 = sns.violinplot(ax=ax2,
                        data=df,
                        x="special",
                        y=y_column,
                        cut=0,
                        inner="quartile",
                        split=False,
                        linewidth=2,
                        label="Oracle",
                        palette="pastel",
                        marker="s")
    sns.pointplot(ax=ax2,
                  data=df,
                  x="special",
                  y=y_column,
                  join=False,
                  ci="sd",
                  linewidth=2,
                  palette="dark")
    g2.axhline(y=0, color="red", label="baseline")
    ax2.minorticks_on()
    ax2.tick_params(axis='both',
                    which='both',
                    labelsize=global_fontsize,
                    reset=True,
                    top=False,
                    right=False)
    ax2.grid(b=True, which='both')
    ax2.legend(frameon=False)
    g2.tick_params(left=True)
    return fig


def compare_policy_scores_violin(baseline_policy_legend_triples, metric):
    """
    Plots policies as cdf of improvement per scenario over a baseline policy.

    baseline_policy_legend_triples is a list of triple of the form:
    (baseline_scores, policy_scores, legend)

    baseline_scores and policy_scores are dictionaries mapping from a scenario
    to a score. They share the same set of keys.

    metric: The metric whose score is used to compare the policies'
    configuration choice per environment, e.g. mota, ratio_mostly_tracked, etc

    legends: The label names of the cdf plots of a policy in the figure.
    """
    rows = []
    for baseline, policy, legend in baseline_policy_legend_triples:
        assert set(baseline.keys()) == set(policy.keys()), (
            f"baseline and policy under legend {legend} don't have the same"
            " set of scenarios")
        if isinstance(legend, str):
            legend = {"special": legend}
        new_rows = [{
            f"{metric} improvement": policy[k] - baseline[k],
            **legend
        } for k in baseline.keys()]
        rows.extend(new_rows)
    df = pd.DataFrame(rows)
    x_column = "full_feature_name"
    hue_column = "features_time"
    fig, (ax1, ax2) = plt.subplots(1,
                                   2,
                                   sharey=True,
                                   figsize=(20, 10),
                                   gridspec_kw={'width_ratios': [3, 1]})
    g1 = sns.violinplot(
        ax=ax1,
        data=df,
        x=x_column,
        y=f"{metric} improvement",
        hue=hue_column,
        # hue_order=sorted(df[hue_column].unique()),
        # order=sorted(df[x_column].unique()),
        cut=0,
        inner="quartile",
        split=False,
        linewidth=2,
        palette="pastel")
    sns.pointplot(
        ax=ax1,
        data=df,
        x=x_column,
        y=f"{metric} improvement",
        hue=hue_column,
        # hue_order=sorted(df[hue_column].unique()),
        # order=sorted(df[x_column].unique()),
        join=False,
        ci="sd",
        linewidth=2,
        dodge=0.4,
        markers="s",
        palette="dark")
    g1.axhline(y=0, color="red", label="baseline")
    ax1.tick_params(axis='both',
                    which='both',
                    labelsize="large",
                    reset=True,
                    top=False,
                    right=False)
    ax1.legend(frameon=False)
    g2 = sns.violinplot(ax=ax2,
                        data=df,
                        x="special",
                        y=f"{metric} improvement",
                        cut=0,
                        inner="quartile",
                        split=False,
                        linewidth=2,
                        label="Oracle",
                        palette="pastel",
                        marker="s")
    sns.pointplot(ax=ax2,
                  data=df,
                  x="special",
                  y=f"{metric} improvement",
                  join=False,
                  ci="sd",
                  linewidth=2,
                  palette="dark")
    g2.axhline(y=0, color="red", label="baseline")
    ax2.tick_params(axis='both',
                    which='both',
                    labelsize="large",
                    reset=True,
                    top=False,
                    right=False)
    ax2.legend(frameon=False)
    g2.tick_params(left=True)
