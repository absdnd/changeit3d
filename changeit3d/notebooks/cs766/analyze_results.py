import numpy as np
import os
import pandas as pd
from changeit3d.evaluation.generic_metrics import chamfer_dists
from tabulate import tabulate
import torch

# from helper import load_pretrained_pc_ae, read_saved_args, ShapeTalkCustomShapeDataset, generate_notebook_args
torch.cuda.empty_cache()
# Directory 
target_directory = "/home/shared/data/shapetalk_dataset/shapetalk/point_clouds/scaled_to_align_rendering"
result_dir = "/home/shared/data/generation_results"

table1 = []

# Experiment 1
exp_type = "noisy_point_clouds_v2"
folder_name = os.path.join(result_dir, exp_type)
# categories = ["lamp", "bottle", "mug"]
# noise_list = ["0.01", "0.02", "0.03"]
categories = ["lamp", "bottle", "mug"]
noise_list = ["0.003", "0.005", "0.008","0.01", "0.02", "0.025", "0.03"]

for category in categories:
    for noise_value in noise_list:
        # noise_dec = noise_value.split(".")[1]
        # sub_script = f"Sigma{noise_dec}"
        result_file = os.path.join(folder_name, f"pcae_dataset_noisy_pc_{noise_value}_{category}.npy")
        results = np.load(result_file, allow_pickle=True).item()
        csv_path = results['csv_path']

        df = pd.read_csv(csv_path)
        target_files = df['target_uid'].apply(lambda x: os.path.join(target_directory, x + ".npz")).tolist()
        target_pcs = [np.load(file)['pointcloud'] for file in target_files]

        target_pcs = np.array(target_pcs)
        rand_ids = np.random.choice(target_pcs.shape[1], 100, replace=False)
        target_pcs = target_pcs[:, rand_ids]

        transformed_shapes = results['recons'][0]
        b_size = 4 # You can adjust this value based on your memory constraints

        average_chamfer, _ = chamfer_dists(target_pcs, transformed_shapes, b_size)
        table1.append([category, noise_value, average_chamfer])

print(tabulate(table1, headers=["Category", "Noise Value", "Average Chamfer Distance"]))

table2 = []

# Experiment 2
exp_type = "language" # language or noisy_point_clouds 
folder_name = os.path.join(result_dir, exp_type)
categories = ["lamp", "chair", "vase"]
exp_category = ["baseline", "modified_lang"]

for category in categories:
    for exp in exp_category:
        result_file = os.path.join(folder_name, f"pcae_dataset_{category}_{exp}.npy")
        results = np.load(result_file, allow_pickle=True).item()
        csv_path = results['csv_path']
    
        df = pd.read_csv(csv_path)
        target_files = df['target_uid'].apply(lambda x: os.path.join(target_directory, x + ".npz")).tolist()
        target_pcs = [np.load(file)['pointcloud'] for file in target_files]

        transformed_shapes = results['recons'][1]
        b_size = 4  # You can adjust this value based on your memory constraints

        average_chamfer, _ = chamfer_dists(target_pcs, transformed_shapes, b_size)
        table2.append([category, exp, average_chamfer])
 
print(tabulate(table2, headers=["Category", "Experiment", "Average Chamfer Distance"]))

