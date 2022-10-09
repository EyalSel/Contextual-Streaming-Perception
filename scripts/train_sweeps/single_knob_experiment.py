import itertools
from ad_config_search.evaluation_utils import config_knobs

features_time_mode_list = ["present", "past-1-step"]

features_source_list = ["gt", "po"]

knob_list = config_knobs

config_freeze_strategies = ["train_static_config", "uniform_random"]

# config_freeze_seeds = [0.43, 0.44, 0.45, 0.46, 0.47]
config_freeze_seeds = []

other_flags = ("--time_mode=timely "
               "--infinite_mode=false "
               "--remove_inf_nan=true "
               "--score_clip_range=-100 "
               "--score_clip_range=100 "
               "--num_sectors=5 ")


def main():
    for i, (knob, features_source, features_time,
            config_freeze_strategy) in enumerate(
                itertools.product(knob_list, features_source_list,
                                  features_time_mode_list,
                                  config_freeze_strategies)):
        if config_freeze_strategy == "uniform_random":
            for config_freeze_seed in config_freeze_seeds:
                print(f"echo \">>>>>>>>>>>>>>>>>> {i} <<<<<<<<<<<<<<<<<<\"")
                print(f"python rforest_data_prep.py "
                      f"--knob_subset={knob} "
                      f"--features_source={features_source} "
                      f"--features_time={features_time} "
                      f"--config_freeze_seed={config_freeze_seed} "
                      f"--config_freeze_strategy={config_freeze_strategy} "
                      f"{other_flags} ")
        elif config_freeze_strategy == "train_static_config":
            print(f"echo \">>>>>>>>>>>>>>>>>> {i} <<<<<<<<<<<<<<<<<<\"")
            print(f"python rforest_data_prep.py "
                  f"--knob_subset={knob} "
                  f"--features_source={features_source} "
                  f"--features_time={features_time} "
                  f"--config_freeze_strategy={config_freeze_strategy} "
                  f"{other_flags} ")


if __name__ == '__main__':
    main()
