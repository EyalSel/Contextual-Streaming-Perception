import numpy as np
from tqdm import tqdm
import random


def get_prediction_dist(model, data):
    """
    Returns N x T array where N is number of test points and T is number of
    trees in forest.
    """
    return (np.array(
        [ind_tree.predict(data) for ind_tree in tqdm(model.estimators_)]))


def split_equally(lst, num_pieces):
    """
    Splits a list into pieces as equal a size as possible.
    Essentially takes any % leftovers from the last piece and distributes as
    equally as possible among the other pieces
    """
    assert len(lst) > 0 and num_pieces > 0, (len(lst), num_pieces)
    assert len(lst) >= num_pieces, (len(lst), num_pieces)
    base_piece_size = len(lst) // num_pieces
    leftover = len(lst) % num_pieces
    # the first few pieces with the extra modulo piece
    for i in range(0, leftover * (base_piece_size + 1), base_piece_size + 1):
        yield list(lst[i:i + (base_piece_size + 1)])
    # the rest of the pieces
    for i in range(leftover * (base_piece_size + 1), len(lst),
                   base_piece_size):
        yield list(lst[i:i + base_piece_size])


class ScenarioAwareCVSplitter:
    """
    An implementation of https://scikit-learn.org/stable/glossary.html#term-cross-validation-generator  # noqa: E501
    Assumes X argument in split function is a dataframe with index equal to the
    scenario value of the datapoint.
    """
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        unique_scenarios = list(X.index.unique())
        if self.shuffle:
            random.shuffle(unique_scenarios, random=None)
        for scenario_list in split_equally(unique_scenarios, self.n_splits):
            all_indices = np.arange(len(X))
            test_mask = X.index.isin(scenario_list)
            yield (all_indices[~test_mask], all_indices[test_mask])


def split_scenarios(scenarios, ratio_test, seed=0.43):
    unique_scenarios = sorted(list(np.unique(scenarios)))
    random.shuffle(unique_scenarios, random=lambda: seed)
    num_test_boundary = int(len(unique_scenarios) * ratio_test)
    train_scenarios = unique_scenarios[num_test_boundary:]
    test_scenarios = unique_scenarios[:num_test_boundary]
    return train_scenarios, test_scenarios


def waymo_official_train_val_test(scenario_segment_list):
    unique_scenarios = sorted(list(np.unique(scenario_segment_list)))
    train = [x for x in unique_scenarios if x.startswith("train")]
    val = [x for x in unique_scenarios if x.startswith("validation")]
    return train, val
