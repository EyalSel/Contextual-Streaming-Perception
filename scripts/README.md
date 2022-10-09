## Setting up the repo

 - (Recommended in a new conda environment) run: `cd python-libraries/ad-config-search && pip install -e .`
 - To run on argo use branch <TODO>. To run Waymo use branch <TODO>.
 - Copy files from [here](https://drive.google.com/drive/u/1/folders/1eNr6FfUv7v9_SvFe3qN45mPx51WvTuhT) to the following locations in the repository:
   - Copy `dataset/*.pl` and `dataset/*.npy` files in the link to the `scripts/` directory
   - Copy `dataset/engineered-features/po/features_from_pipeline_1s_chunks.pl` to the `scripts/` directory
   - Copy `dataset/engineered-features/po/scenario_features_v5_fine_1s.csv` to the `data/` directory


## To get the results in Table 1
```bash
## Leave the configuration for the dataset of choice
# Waymo
NUM_SECTORS="20"
INF_MODE="true"

# Argo
NUM_SECTORS="30"
INF_MODE="false"

# Computes and outputs the numbers
python opportunity_gap_calculation.py --time_mode=timely --infinite_mode=$(INF_MODE) --num_sectors=$(NUM_SECTORS)
python opportunity_gap_calculation.py --time_mode=sync --infinite_mode=$(INF_MODE) --num_sectors=$(NUM_SECTORS)
```


## To get the results in Table 2

```bash
# Waymo
NUM_SECTORS="20"
INF_MODE="true"
SPLIT_STRAT="waymo"

# delta S optimal, delta D optimal
python separate_sync_timely_analysis.py --infinite_mode=true --num_sectors=20 --remove_inf_nan=true --dataset_split_strategy=waymo --degradation_mode=optimal --sync_mode=optimal 

# delta S optimal, delta D global-best
python separate_sync_timely_analysis.py --infinite_mode=true --num_sectors=20 --remove_inf_nan=true --dataset_split_strategy=waymo --degradation_mode=global_static --sync_mode=optimal

# delta S global-best, delta D optimal
python separate_sync_timely_analysis.py --infinite_mode=true --num_sectors=20 --remove_inf_nan=true --dataset_split_strategy=waymo --degradation_mode=optimal --sync_mode=global_static

# delta S global-best, delta D global-best
python separate_sync_timely_analysis.py --infinite_mode=true --num_sectors=20 --remove_inf_nan=true --dataset_split_strategy=waymo --degradation_mode=global_static --sync_mode=global_static
```


## To get the results in Table 3

```bash
## Leave the configuration for the dataset of choice
# Waymo
NUM_SECTORS="20"
INF_MODE="true"
SPLIT_STRAT="waymo"

# Argo
NUM_SECTORS="30"
INF_MODE="false"
SPLIT_STRAT="argo"

## Train the Octopus policies
# Ground truth from current segment
python rforest_data_prep.py --flagfile=flagfile.txt --num_sectors=$(NUM_SECTORS) --subtract_baseline=true --time_mode=timely --infinite_mode=$(INF_MODE) --dataset_split_strategy=$(SPLIT_STRAT) --features_time=present --features_source=gt --hp_search_iters=40

# Ground truth from previous segment
python rforest_data_prep.py --flagfile=flagfile.txt --num_sectors=$(NUM_SECTORS) --subtract_baseline=true --time_mode=timely --infinite_mode=$(INF_MODE) --dataset_split_strategy=$(SPLIT_STRAT) --features_time=past-1-step --features_source=gt --hp_search_iters=40

# Pipeline output from previous segment
python rforest_data_prep.py --flagfile=flagfile.txt --num_sectors=$(NUM_SECTORS) --subtract_baseline=true --time_mode=timely --infinite_mode=$(INF_MODE) --dataset_split_strategy=$(SPLIT_STRAT) --features_time=past-1-step --features_source=po-global-static --hp_search_iters=40

## Get the numbers
# Global static score, oracle, and a ground truth policy (either previous or current segment) score
python analyze_exp_in_current_workdir.py --experiment_path=<PATH-TO-TRAINED-POLICY>

# Policy using the pipeline output from previous segment
python close_loop_eval.py --features_time=past-1-step --experiment_path=<PATH-TO-TRAINED-POLICY>

# Oracle from previous segment
python oracle_last_step.py --infinite_mode=$(INF_MODE) --num_sectors=$(NUM_SECTORS)
```


## To get the results in Table 4a
```bash
# Get the numbers from a trained policy
python rforest_feature_importance.py --experiment_path=<PATH-TO-TRAINED-POLICY>
```


## To get the results in Table 4b
```bash
NUM_SECTORS="20"
INF_MODE="true"
SPLIT_STRAT="waymo"

# Ablating out T-max-age knob
python rforest_data_prep.py --flagfile=flagfile.txt --num_sectors=$(NUM_SECTORS) --subtract_baseline=true --time_mode=timely --infinite_mode=$(INF_MODE) --dataset_split_strategy=$(SPLIT_STRAT) --features_time=present --features_source=gt --hp_search_iters=40 --knob_subset=D-model --config_freeze_strategy=train_static_config
# Ablating out D-model knob
python rforest_data_prep.py --flagfile=flagfile.txt --num_sectors=$(NUM_SECTORS) --subtract_baseline=true --time_mode=timely --infinite_mode=$(INF_MODE) --dataset_split_strategy=$(SPLIT_STRAT) --features_time=present --features_source=gt --hp_search_iters=40 --knob_subset=T-max-age --config_freeze_strategy=train_static_config
```