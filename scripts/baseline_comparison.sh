# Example trained model paths
# SYNC_MODEL="exp_result_1646648876124--model_type=RF__window_length=1__time_mode=sync__infinite_mode=True__features_time=present__features_source=gt__subtract_baseline=True__separate_sync_timely_mode=True__te=0.3.pl"
# DEG_MODEL="exp_result_1646651297830--model_type=RF__window_length=1__time_mode=timely__infinite_mode=True__features_time=present__features_source=gt__subtract_baseline=True__separate_sync_timely_mode=True__te=0.6.pl"

echo ">>>>>>>>>>>>>>>>> sync learned degradation learned <<<<<<<<<<<<<<<<<<<<<<<"
python separate_sync_timely_analysis.py \
--infinite_mode=true --num_sectors=20 --remove_inf_nan=true \
--dataset_split_strategy=waymo \
--degradation_mode=model --sync_mode=model \
--degradation_model_path="$DEG_MODEL" \
--sync_model_path="$SYNC_MODEL"

echo ">>>>>>>>>>>>>>>>> sync optimal degradation learned <<<<<<<<<<<<<<<<<<<<<<<"
python separate_sync_timely_analysis.py \
--infinite_mode=true --num_sectors=20 --remove_inf_nan=true \
--dataset_split_strategy=waymo \
--degradation_mode=model --sync_mode=optimal \
--degradation_model_path="$DEG_MODEL"

echo ">>>>>>>>>>>>>>>>> sync learned degradation optimal <<<<<<<<<<<<<<<<<<<<<<<"
python separate_sync_timely_analysis.py \
--infinite_mode=true --num_sectors=20 --remove_inf_nan=true \
--dataset_split_strategy=waymo \
--degradation_mode=optimal --sync_mode=model \
--sync_model_path="$SYNC_MODEL"


#######

echo ">>>>>>>>>>>>>>>>> sync optimal degradation optimal <<<<<<<<<<<<<<<<<<<<<<<"
python separate_sync_timely_analysis.py \
--infinite_mode=true --num_sectors=20 --remove_inf_nan=true \
--dataset_split_strategy=waymo \
--degradation_mode=optimal --sync_mode=optimal 

echo ">>>>>>>>>>>>>>>>> sync optimal degradation global static <<<<<<<<<<<<<<<<<<<<<<<"
python separate_sync_timely_analysis.py \
--infinite_mode=true --num_sectors=20 --remove_inf_nan=true \
--dataset_split_strategy=waymo \
--degradation_mode=global_static --sync_mode=optimal

echo ">>>>>>>>>>>>>>>>> sync global static degradation global static <<<<<<<<<<<<<<<<<<<<<<<"
python separate_sync_timely_analysis.py \
--infinite_mode=true --num_sectors=20 --remove_inf_nan=true \
--dataset_split_strategy=waymo \
--degradation_mode=global_static --sync_mode=global_static

echo ">>>>>>>>>>>>>>>>> sync global static degradation optimal <<<<<<<<<<<<<<<<<<<<<<<"
python separate_sync_timely_analysis.py \
--infinite_mode=true --num_sectors=20 --remove_inf_nan=true \
--dataset_split_strategy=waymo \
--degradation_mode=optimal --sync_mode=global_static