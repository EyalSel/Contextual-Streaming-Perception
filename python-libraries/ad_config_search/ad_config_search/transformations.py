"""
This file contains implementations of dataset preprocessing (e.g.
normalization, trimming, etc) to be run prior to training of AD-config-search
models. They are abstractly called transformations, closely following
SKLearn's API (https://scikit-learn.org/stable/data_transforms.html). We
implement our own version to maintain certain invariants described below.

In our implementation we always assume the data is in pandas DataFrame form.
All of these transformations are assumed to preserve the index and row order
in the dataframe. This is needed to keep track of outputs. Furthermore, the
transformations never alter the dataframe object in place, they always return
a modified copy.

These preprocessing transformations can be stacked using a MultiTransformation
instance.
"""

import json

import numpy as np
import pandas as pd
from ad_config_search.policy_utils import greedy_condense_space, oracle_from_df
from ad_config_search.utils import get_rows
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm


class Transformation:
    """
    The abstract parent class to be extended with implementation for the _fit
    and _transform functions. The fit and transform functions are to be left
    alone, they take care of boilerplate input copying and guard against
    multiple fit calls.

    All subclasses must run super().__init__() first thing in their init fn
    """
    def __init__(self):
        self.fitted = False

    # Do not extend this function
    def fit(self, config_df: pd.DataFrame) -> None:
        assert not self.fitted, "Trying to fit transformation twice"
        self.fitted = True
        self._fit(config_df.copy())

    def _fit(self, config_df: pd.DataFrame) -> None:
        raise NotImplementedError("To be implemented by child class")

    # Do not extend this function
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.fitted, "Trying to transform without fitting!"
        return self._transform(df.copy())

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("To be implemented by child class")


class MultiTransformation(Transformation):
    """
    Combines multiple unfitted transformation objects.
    """
    def __init__(self, transformations: list):
        super().__init__()
        for i, transform in enumerate(transformations):
            assert not transform.fitted, (
                f"Transformation of index {i} is already fitted!")
        self.transformations = transformations

    def _fit(self, df: pd.DataFrame):
        for t in self.transformations:
            t.fit(df)
            df = t.transform(df)

    def _transform(self, df: pd.DataFrame):
        for t in self.transformations:
            df = t.transform(df)
        return df


class ScoreClipping(Transformation):
    def __init__(self, clip_range):
        super().__init__()
        self.clip_range = clip_range

    def _fit(self, scores_df: pd.DataFrame):
        pass

    def _transform(self, scores_df: pd.DataFrame):
        scores_df["score"] = scores_df["score"].clip(*self.clip_range)
        return scores_df


class ScoreNormalization(Transformation):
    def __init__(self):
        super().__init__()
        self.mean_std = None

    def _fit(self, df: pd.DataFrame):
        self.mean_std = (df["score"].mean(), df["score"].std())

    def _transform(self, df: pd.DataFrame):
        score_mean, score_std = self.mean_std
        df["score"] = (df["score"] - score_mean) / score_std
        return df


class SubtractBaseline(Transformation):
    def __init__(self, time_mode, infinite_mode):
        super().__init__()
        self.time_mode = time_mode
        self.infinite_mode = infinite_mode
        self.baseline_configuration = {}

    def _fit(self, scores_df: pd.DataFrame):
        from ad_config_search.policy_utils import get_average_config
        baseline_config = get_average_config(
            scores_df["run"].map(lambda x: x.split("-P")[0]).unique(),
            self.time_mode, self.infinite_mode)
        self.baseline_configuration = baseline_config

    def _transform(self, scores_df: pd.DataFrame):
        from ad_config_search.utils import get_rows
        all_rows = []
        for video_segment, rows in tqdm(scores_df.groupby("run")):
            baseline_config_row = get_rows(rows, self.baseline_configuration)
            assert len(baseline_config_row) == 1, len(baseline_config_row)
            baseline_config_score = baseline_config_row.iloc[0]["score"]
            rows["score"] = rows["score"] - baseline_config_score
            all_rows.append(rows)
        return pd.concat(all_rows)


class EnvironmentVectorization(Transformation):
    """
    When hand-picked features are used, this transformation preprocesses them
    similarily to ConfigVectorization below. Categorical features are one-hot
    encoded, numerical features are normalized, with mean 0 and std of 1.
    """
    def __init__(self):
        super().__init__()
        self.transform_dict = {}

    def _fit(self, env_feats):
        nums = sorted(list(env_feats.dtypes[env_feats.dtypes == float].index))
        cats = list(env_feats.dtypes[env_feats.dtypes != float].index)
        cats = list(set(cats) - set([
            "scenario_name"
        ]))  # remove scenario name from categorical columns to be encoded
        self.transform_dict["num_columns"] = nums
        # NUMERICAL
        from sklearn.preprocessing import StandardScaler
        num_scaler = StandardScaler()
        num_scaler.fit(env_feats[nums])
        self.transform_dict["num_scaler"] = num_scaler
        # CATEGORICAL
        self.transform_dict["cat_encoder"] = {}
        for c in cats:
            encoder = MyLabelBinarizer()
            encoder.fit(env_feats[c])
            self.transform_dict["cat_encoder"][c] = encoder

    def _transform(self, env_feats: pd.DataFrame):
        assert ("num_scaler" in self.transform_dict and "cat_encoder"
                in self.transform_dict), self.transform_dict.keys()

        nums = sorted(list(env_feats.dtypes[env_feats.dtypes == float].index))
        cats = list(env_feats.dtypes[env_feats.dtypes != float].index)
        cats = list(set(cats) - set([
            "scenario_name"
        ]))  # remove scenario name from categorical columns to be encoded

        assert nums == self.transform_dict["num_columns"], (
            "num columns don't match between fitted df and transformed df",
            nums, self.transform_dict["num_columns"])
        transform_dict_cat_cols = self.transform_dict["cat_encoder"].keys()
        assert set(cats) == set(transform_dict_cat_cols), (
            ("Mistmatch between categorical columns intended for one-hot "
             "transformation and the columns given by the transform_dict"),
            cats, transform_dict_cat_cols)

        env_feats[nums] = self.transform_dict["num_scaler"].transform(
            env_feats[nums])

        for c, encoder in self.transform_dict["cat_encoder"].items():
            ohe_df = pd.DataFrame(
                encoder.transform(env_feats[c]),
                columns=[f"{c}__{label}" for label in encoder.classes_],
                index=env_feats.index)
            env_feats = pd.concat([env_feats, ohe_df], axis=1).drop([c],
                                                                    axis=1)

        return env_feats


class MyLabelBinarizer(LabelBinarizer):
    """
    https://stackoverflow.com/a/31948178/1546071
    """
    def transform(self, y):
        Y = super().transform(y)
        if self.y_type_ == 'binary' and len(self.classes_) == 2:
            return np.hstack((Y, 1 - Y))
        else:
            return Y

    def inverse_transform(self, Y, threshold=None):
        if self.y_type_ == 'binary' and len(self.classes_) == 2:
            return super().inverse_transform(Y[:, 0], threshold)
        else:
            return super().inverse_transform(Y, threshold)


class ConfigVectorization(Transformation):
    """
    Example
    ---------------------------------------------
        D-model               lastdigit OR one-hot [d_model_numerical]
        D-conf                              [-1,1]
        D-seq-pol                          one-hot
        T-model                            one-hot
        T-min-iou                           [-1,1]
        T-max-age (linear scale)            [-1,1]
        T-every-nth-det                     [-1,1]
    """
    def __init__(self, d_model_numerical=False, num_range=(-1, 1)):
        super().__init__()
        self.d_model_numerical = d_model_numerical
        self.num_range = num_range
        self.transform_dict = {}

    def _verify_and_prep_df(self, config_df: pd.DataFrame):
        cats = ["D-seq-pol", "T-model"]
        nums = ["T-min-iou", "T-max-age", "D-conf", "T-every-nth-det"]
        if self.d_model_numerical:

            def map_to_digit(x):
                if x[-1] == 'x':
                    return 8
                return int(x[-1])

            config_df['D-model'] = config_df['D-model'].apply(map_to_digit)
            nums.append("D-model")
        else:
            cats.append("D-model")

        assert np.all(
            config_df.dtypes.loc[cats] ==
            object), "some config columns expected to be categorical are not"
        assert np.all((config_df.dtypes.loc[nums] == float)
                      | (config_df.dtypes.loc[nums] == int)
                      ), "some config columns expected to be float are not"
        return config_df, cats, nums

    def _fit(self, config_df: pd.DataFrame):
        config_df, cats, nums = self._verify_and_prep_df(config_df)
        # NUMERICAL
        from sklearn.preprocessing import MinMaxScaler
        num_scaler = MinMaxScaler(feature_range=self.num_range)
        num_scaler.fit(config_df[nums])
        self.transform_dict["num_scaler"] = num_scaler
        # CATEGORICAL
        self.transform_dict["cat_encoder"] = {}
        for c in cats:
            encoder = MyLabelBinarizer()
            encoder.fit(config_df[c])
            self.transform_dict["cat_encoder"][c] = encoder

    def _transform(self, config_df: pd.DataFrame):
        assert not np.any(config_df.isnull().values)
        if self.transform_dict == {}:
            raise Exception("Fit first before transforming!")
        config_df, cats, nums = self._verify_and_prep_df(config_df)

        assert ("num_scaler" in self.transform_dict and "cat_encoder"
                in self.transform_dict), self.transform_dict.keys()

        transform_dict_cat_cols = self.transform_dict["cat_encoder"].keys()
        assert set(cats) == set(transform_dict_cat_cols), (
            ("Mistmatch between categorical columns intended for one-hot "
             "transformation and the columns given by the transform_dict"),
            cats, transform_dict_cat_cols)

        config_df[nums] = self.transform_dict.get("num_scaler").transform(
            config_df[nums])

        for c, encoder in self.transform_dict["cat_encoder"].items():
            assert not np.any(config_df.isnull().values)
            ohe_df = pd.DataFrame(
                encoder.transform(config_df[c]),
                columns=[f"{c}__{label}" for label in encoder.classes_],
                index=config_df.index)
            config_df = pd.concat([config_df, ohe_df], axis=1).drop([c],
                                                                    axis=1)

        return config_df


class ConfigSpacePruning(Transformation):
    def __init__(self, k=1):
        super().__init__()
        self.k = k
        self.pruned_config_space = None

    def _fit(self, config_df):
        oracle_policy = oracle_from_df(config_df)
        condensed_oracle_policy = greedy_condense_space(config_df,
                                                        oracle_policy,
                                                        k=self.k)[0]
        self.pruned_config_space = [
            json.loads(x) for x in np.unique(
                [json.dumps(x) for x in condensed_oracle_policy.values()])
        ]

    def _transform(self, config_df):
        scores_df_condensed_space = pd.concat(
            [get_rows(config_df, c) for c in self.pruned_config_space])
        return scores_df_condensed_space
