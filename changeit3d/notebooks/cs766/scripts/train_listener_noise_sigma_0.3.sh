#!/bin/bash
python /home/shared/changeit3d/changeit3d/notebooks/cs766/exp_c_listener_training.py \
--batch-size=128 \
--weight-decay=1e-4 \
--run-name="ablation_model_1_sigma3" \
--model-type='ablation_model_one' \
--shapetalk-file="home/shared/data/noise_added_to_point_clouds/chair_noiseSigma3_exp1.csv"
