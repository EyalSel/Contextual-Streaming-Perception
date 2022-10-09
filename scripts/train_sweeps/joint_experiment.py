import itertools

features_time_mode_list = ["present", "past-1-step"]

features_source_list = ["gt", "po"]

time_modes = ["sync", "timely"]

infinite_modes = ["false", "true"]

num_sectors_list = [5]

other_flags = \
    "--remove_inf_nan=true --score_clip_range=-100 --score_clip_range=100"


def main():
    for i, (num_sectors, infinite_mode, time_mode, features_source,
            features_time) in enumerate(
                itertools.product(num_sectors_list, infinite_modes, time_modes,
                                  features_source_list,
                                  features_time_mode_list)):
        if features_source == "po":
            config_condense_flag = "--condensing_strategy=greedy_v1_k=10"
        else:
            config_condense_flag = ""
        print(f"echo \">>>>>>>>>>>>>>>>>> {i} <<<<<<<<<<<<<<<<<<\"")
        print(f"python rforest_data_prep.py "
              f"--num_sectors={num_sectors} "
              f"--infinite_mode={infinite_mode} "
              f"--time_mode={time_mode} "
              f"--features_source={features_source} "
              f"--features_time={features_time} "
              f"{config_condense_flag} "
              f"{other_flags}")


if __name__ == '__main__':
    main()
