import numpy as np
import os
import pandas as pd
# from mayavi import mlab
from changeit3d.utils.visualization import visualize_point_clouds_3d_v2
import matplotlib.pyplot as plt


# Noise Base Point Cloud Path
noisy_base_path = "data/noise_added_to_point_clouds/"
result_path = "data/generation_results/noisy_point_clouds_v2"
pc_base_path = "data/shapetalk_dataset/shapetalk/point_clouds/scaled_to_align_rendering"
vis_path = "visualizations/exp_b_noisy"


# Target Point Cloud Path
target_pc_path = ""
category = "chair"

# Sigma List
sigma_list = ['0.00', '0.01',  '0.02', '0.03']
sigma_str = [str(sigma).split(".")[1] for sigma in sigma_list]


# Result Dictionary
result_dict = {}
for i, sigma in enumerate(sigma_list):
    sigma_result_path = os.path.join(result_path, "pcae_dataset_noisy_pc_{}_{}.npy".format(sigma, category))  
    csv_path = os.path.join(noisy_base_path, "{}_noiseSigma{}_exp1.csv".format(category, sigma_str[i]))

    sigma_df = pd.read_csv(csv_path)
    sigma_result_dict = np.load(sigma_result_path, allow_pickle=True).item()

    result_dict[sigma] = {}
    result_dict[sigma]["df"] = sigma_df
    result_dict[sigma]["result_dict"]  = sigma_result_dict
    result_len = len(sigma_df)