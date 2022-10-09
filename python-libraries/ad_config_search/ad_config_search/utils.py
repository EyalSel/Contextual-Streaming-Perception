from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull

import numpy as np
import pandas as pd
from tqdm import tqdm


def prepend_line_to_file(file_path, line, repeatable=False):
    with open(file_path, 'r') as f:
        first_line = f.readline()[:-1]
    if first_line == line and not repeatable:
        return
    with open(file_path, 'r') as f:
        contents = f.read()
    with open(file_path, 'w') as f:
        f.write(line + "\n")
        f.write(contents)


def replace_first_line(file_path, current_line, new_line):
    with open(file_path) as f:
        lines = f.readlines()
    first_line = lines[0][:-1]
    if first_line == current_line:
        lines[0] = new_line + "\n"
        with open(file_path, "w") as f:
            f.writelines(lines)
    else:
        assert first_line == new_line, (file_path, current_line, new_line,
                                        first_line)


def fix_pylot_profile(file_path, silent=True):
    with open(file_path, 'r') as f:
        contents = f.read()
    if contents[0] == "[":
        if not silent:
            print("The pylot_profile.json file seems to already be fixed")
        return
    with open(file_path, 'w') as f:
        f.write("[\n")
        f.write(contents[:-2])
        f.write("\n]")


def verify_keys_in_dict(required_keys, arg_dict):
    assert set(required_keys).issubset(set(arg_dict.keys())), \
            "one or more of {} not found in {}".format(required_keys, arg_dict)


def ray_map(lst, fn, n):
    """
    Helper function to run fn on every element of lst in parallel on n workers.
    Requires ray, so make sure a ray cluster is up before calling this.
    """
    import ray  # placed inside because ray sets the pickle version in

    # https://github.com/ray-project/ray/blob/871cde989a45341129aa9bcd7b59115020ef13d2/python/ray/cloudpickle/compat.py
    # If ray is imported before skelarn, it sets the pickle version to 5,
    # which bones the joblib hyperparameter search functionality of scikit
    # learn's training.
    assert len(lst) > 1, len(lst)

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    @ray.remote
    def remote_version(arg_lst, load_bar=False):
        def try_with_except(arg):
            try:
                return fn(arg)
            except Exception as e:
                print(f"Got exception below for input: {arg}")
                raise e

        if load_bar:
            arg_lst = tqdm(arg_lst)
        result = [try_with_except(arg) for arg in arg_lst]
        return result

    pieces = chunks(lst, len(lst) // n)
    oids = [remote_version.remote(p, i == 0) for i, p in enumerate(pieces)]
    list_of_lists = ray.get(oids)

    import itertools
    merged = list(itertools.chain.from_iterable(list_of_lists))
    return merged


def ray_map_np(lst, fn, n):
    """
    Helper function to run fn on every element of lst in parallel on n workers.
    Requires ray, so make sure a ray cluster is up before calling this.

    Assumes that the output of the function is a numpy array, and therefore
    concatenates the outputs along axis=0 (output size must be the same).
    """
    import ray  # placed inside because ray sets the pickle version in

    # https://github.com/ray-project/ray/blob/871cde989a45341129aa9bcd7b59115020ef13d2/python/ray/cloudpickle/compat.py
    # If ray is imported before skelarn, it sets the pickle version to 5,
    # which bones the joblib hyperparameter search functionality of scikit
    # learn's training.
    assert len(lst) > 1, len(lst)

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    @ray.remote
    def remote_version(arg_lst, load_bar=False):
        def try_with_except(arg):
            try:
                return fn(arg)
            except Exception as e:
                print(f"Got exception below for input: {arg}")
                raise e

        if load_bar:
            arg_lst = tqdm(arg_lst)
        result = np.array([try_with_except(arg) for arg in arg_lst])
        return result

    pieces = chunks(lst, len(lst) // n)
    oids = [remote_version.remote(p, i == 0) for i, p in enumerate(pieces)]
    list_of_lists = ray.get(oids)

    merged = np.concatenate(list_of_lists, axis=0)
    return merged


def get_rows(df: pd.DataFrame, dict_columns):
    """
    given dict_columns={col_name: col_value...}, finds all rows in df with
    those conditions met.
    """
    from functools import reduce
    return df[reduce(lambda x, y: x & y,
                     [df[k] == v for k, v in dict_columns.items()])]


def contract_one_hot(df_original, delimiter="__"):
    """
    Turns one hot columns named '<original_col_name>__<col_option>'
    into '<original_col_name>' with category options. Returns a copy of the
    original df.
    """
    df = df_original.copy()
    # preserving column order after contraction
    # https://stackoverflow.com/a/17016257/1546071
    column_order = [c.split(delimiter)[0] for c in df.columns]
    column_order = list(dict.fromkeys(column_order))
    parent_columns = \
        set([c.split(delimiter)[0] for c in df.columns if delimiter in c])
    for pc in tqdm(parent_columns, desc="contract_one_hot"):
        ccs = list(filter(lambda cc: cc.startswith(pc), df.columns))
        rows = df[ccs]
        rows.columns = [cc.split(delimiter)[1] for cc in ccs]
        df[pc] = rows.idxmax(axis=1)
        df = df.drop(columns=ccs)
    return df[column_order]


# https://stackoverflow.com/a/52442331/1546071
@contextmanager
def suppress_stdout(stderr=False):
    """A context manager that redirects stdout to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stdout(fnull) as out:
            if stderr:
                with redirect_stderr(fnull) as err:
                    yield (out, err)
            else:
                yield (out)


@contextmanager
def run_under_working_dir(dir):
    import os
    current_dir = os.getcwd()
    os.chdir(dir)
    try:
        yield
    finally:
        os.chdir(current_dir)


def merge_dicts(dict_list: list, raise_on_duplicate_keys=True):
    """
    Returns a merged dictionary. If raise_on_duplicate_keys is false then
    arbitrarily chooses one of the repeated keys.
    """
    assert len(dict_list) > 1, len(dict_list)
    if raise_on_duplicate_keys:
        from more_itertools import flatten
        all_keys_append = list(flatten([list(d.keys()) for d in dict_list]))
        from collections import Counter
        duplicates = [k for k, v in Counter(all_keys_append).items() if v > 1]
        if len(duplicates) > 0:
            raise Exception(f"duplicates keys {duplicates}")
    result = {}
    for d in dict_list:
        result.update(d)
    return result


def unique_list(lst):
    return len(lst) == len(set(lst))
