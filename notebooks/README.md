condensing-config-space.ipynb
- Computes and plots how many top k configurations are necessary to match the
  oracle policy.

empty-frame-mask.ipynb
- Plots a frame heatmap indicating presence or absence of obstacles.

model-latencies-importance-analysis.ipynb
- Analysis of the effect of the low frame rate on the configuration space.
- Shows that some models are never useful because they're penalized too much
  due to the low frame rate.

model-runtimes-plot.ipynb
- Plots histograms of runtime for each detection model. 

policy-comparison.ipynb
- Generating policy comparison plots.

RForest-confidence.ipynb
- Preliminary attempts to extract a confidence value from RForest predictions

scrape-env-conditions-fine-grain.ipynb, scrape-env-conditions.ipynb
- A notebook-script to generate the ground-truth environment features.

scrape-env-conditions-pipeline-output.ipynb
- A notebook-script to generate the pipeline output per scenario, to be used
  as past scenario features during training.

argo-examination.ipynb
- Profiles loading time of images as pickle, npy...
- Converts scenarios to GIFs.
- Examining labels.

timely-ground-truth-across-time.ipynb
- Plots a timeline of timely mAP of a perfect detector. The results are shown for
  different scenarios and latencies.

depreceated/detection
- Older notebooks to visualize the effect of detection-only knobs
- carla-mAP-eval.ipynb
  - Computes and plots the sync mAP for different models on the CARLA dataset.
  - Splits the data across different towns, and shows that the accuracy spread
  in town 3 is greater than in town 1 and 2. 

deprecated/RForest_training
- Notebook files used to write and debug training manually before consolidation into python

deprecated/carla-dataset-bbox-size-split.ipynb
- Computes the split of small, medium and large bounding boxes of the CARLA dataset.

deprecated/plots-for-course-writeup.ipynb
- Plots the importance of metaparameters, environment context features, and detection models.

deprecated/prep-waymo-train-dir.ipynb
- Transforms the Waymo scenario data into the format required for training/replaying.
