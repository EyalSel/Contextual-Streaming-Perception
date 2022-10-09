from pathlib import Path
"""
The overall purpose of this module is to map from the labels outputted by the
model (meaning the labels used in the model's training dataset), and the ground
truth labels used by the dataset containing the AD scenario that we're
evaluating on (which could be entirely different).

This module handles the complexities involved in reconciling this dataset
difference, including issues like:
- One set simply has labels that the other does not
- One set has more abstract labels than the other e.g. ("vehicle" in one set
    and "car", "truck" in the other). In this case we're forced to treat all
    the concrete labels as their abstract equivalent.
"""


def merge(pairs):
    """
    Takes list of pairs, at most one of the items in the pairs can be a list.
    Example to illustrate what this function does:
    input: [([a, b], w), (c, [x, y]), (d, z)]
    returns:
        forward {a: w, b: w, c: c, d: d}
        backward {w: w, x: c, y: c, z: d}
        merged_label_list [w, c, d]
    """
    forward, backward = {}, {}
    merged_label_list = []
    print(pairs)
    for f, b in pairs:
        assert type(f) != list or type(
            b) != list, "both {} and {} are lists".format(f, b)
        if type(f) is list:
            forward.update({sub_f: b for sub_f in f})
            backward[b] = b
            merged_label_list.append(b)
        elif type(b) is list:
            backward.update({sub_b: f for sub_b in b})
            forward[f] = f
            merged_label_list.append(f)
        else:
            forward[f] = f
            backward[b] = f
            merged_label_list.append(f)
    return merged_label_list, (forward, backward)


def merge_label_maps(model_map_path, dataset_map_path):
    """
    Arguments:
        Path to DSET_NAME.names files, listing all the labels used in the
        scenario dataset in a separate line. One is for the dataset that
        contains the AD scenario being evaluate, and one is the label map used
        to train the model.
    Returns:
      - A list of merged labels
      - A dictionary mapping from model map labels to merged label list.
        If a label from the model map isn't a key, then it's not in the merged
        label map.
      - Same as item 2 but for dataset map
    """
    if model_map_path == dataset_map_path:
        # Simple scenario where paths are the same.
        with open(model_map_path, 'r') as f:
            label_list = [line.strip() for line in f.readlines()]
        identity = {x: x for x in label_list}
        return label_list, identity, identity

    model_map = Path(model_map_path).stem
    dataset_map = Path(dataset_map_path).stem
    model_conversion_found = list(
        filter(lambda d: d["from"] == model_map and dataset_map in d,
               conversion_dict))
    dataset_conversion_found = list(
        filter(lambda d: d["from"] == dataset_map and model_map in d,
               conversion_dict))
    total_found = len(model_conversion_found) + len(dataset_conversion_found)
    # Must be in exactly one or the other.
    assert total_found == 1, (model_map, dataset_map, total_found)

    model_to_dataset = model_conversion_found != []
    # Get conversion from one of those.
    conversion = model_conversion_found + dataset_conversion_found
    conversion = conversion[0][dataset_map if model_to_dataset else model_map]
    merged_label_list, fn_pair = merge(conversion)
    from_model, from_dataset = fn_pair if model_to_dataset else reversed(
        fn_pair)
    return merged_label_list, from_model, from_dataset


conversion_dict = [{
    "from":
    "waymo",
    "coco": [
        ("cyclist", "bicycle"),
        ("pedestrian", "person"),
        ("sign", "stop sign"),
        ("vehicle", ["car", "motorcycle", "bus", "truck"]),
    ],
    "pylot": [
        ("cyclist", "bicycle"),
        ("pedestrian", "person"),
        ("sign", ["speed limit 30", "speed limit 60", "speed limit 90"]),
        ("vehicle", ["car", "motorcycle"]),
    ]
}, {
    "from":
    "pylot",
    "coco": [
        ("car", "car"),
        ("person", "person"),
        ("motorcycle", "motorcycle"),
        ("bicycle", "bicycle"),
    ]
}, {
    "from":
    "argoverse-jpg",
    "pylot": [
        ("person", "person"),
        ("bicycle", "bicycle"),
        ("car", "car"),
        ("motorcycle", "motorcycle"),
    ],
    "coco": [
        ("person", "person"),
        ("bicycle", "bicycle"),
        ("car", "car"),
        ("motorcycle", "motorcycle"),
        ("bus", "bus"),
        ("truck", "truck"),
        ("traffic light", "traffic light"),
        ("stop sign", "stop sign"),
    ]
}, {
    "from":
    "argoverse-cuboid",
    "pylot": [
        ("pedestrian", "person"),
        ("vehicle", ["car", "motorcycle"]),
    ],
    "coco": [
        ("pedestrian", "person"),
        ("vehicle", ["car", "motorcycle", "bus", "truck"]),
    ]
}]
