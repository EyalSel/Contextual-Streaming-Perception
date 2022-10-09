from offline_pylot.dataset_replay import OfflineWaymoSensorV1_1
from ad_config_search.waymo_utils import visualize_inference
import argparse
import os
from pathlib import Path
# Import before waymo_utils so that matplotlib doesn't require a display
import matplotlib as mpl

mpl.use('Agg')
parser = argparse.ArgumentParser(
    description='Visualize cached waymo jsonl as a gif.')
parser.add_argument('dset_sector', type=int)
parser.add_argument('scenario_number', type=int)
parser.add_argument('--erdos', action='store_true')
parser.add_argument('--files', nargs='+', required=False)
args = parser.parse_args()
if args.files is None:
    p = sorted(list(Path(".").glob("*.jsonl")), key=os.path.getmtime)
    assert p, "Didn't find any jsonl files in this directory"
else:
    p = args.files
print(p)
if args.erdos:
    base_path = Path("/data/ges/waymo/")
else:
    base_path = Path("/home/erdos/datasets/waymo/")
pl_path = base_path / "training_000{}/S{}.pl".format(args.dset_sector,
                                                     args.scenario_number)
if not pl_path.exists():
    from ad_config_search.waymo_utils import waymo_tfrecord_to_pl
    waymo_tfrecord_to_pl(base_path / "training_000{}/S_{}.tfrecord".format(
        args.dset_sector, args.scenario_number))
visualize_inference(pl_path, OfflineWaymoSensorV1_1, p, 10)
