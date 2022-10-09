from pathlib import Path
from imageio import imread
from tqdm import tqdm
import numpy as np


def simplify_dir_names(path):
    names = sorted([p.stem for p in Path(path).iterdir()])
    return {x: "S_{}".format(i) for i, x in enumerate(names)}


def cache_as_npy(data_path):
    sorted_image_files = sorted(list(data_path.glob("*.jpg")),
                                key=lambda p: int(p.stem.split("_")[-1]))
    print("Loading {} jpg files to memory...".format(len(sorted_image_files)))
    loaded_image_files = [imread(f) for f in tqdm(sorted_image_files)]  # RGB
    np.save(data_path / "cached_frames", loaded_image_files)
