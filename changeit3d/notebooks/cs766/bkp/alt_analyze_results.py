import numpy as np
import os
import pandas as pd
import torch

from changeit3d.evaluation.generic_metrics import chamfer_dists
from changeit3d.in_out.basics import create_logger

# from helper import load_pretrained_pc_ae, read_saved_args, ShapeTalkCustomShapeDataset, generate_notebook_args

# Directory 
target_directory = "/home/shared/changeit3d/changeit3d/data/shapetalk/point_clouds/scaled_to_align_rendering/"
result_dir = "/home/shared/changeit3d/data/generation_results/"

logger = create_logger("/home/shared/changeit3d/logs/log.txt")

# # Experiment 1
# exp_type = "noisy_point_clouds"
# folder_name = os.path.join(result_dir, exp_type)
# categories = ["lamp", "bottle", "mug"]
# noise_list = ["0.01", "0.02", "0.03"]

# for category in categories:
#     for noise_value in noise_list:
#         # noise_dec = noise_value.split(".")[1]
#         # sub_script = f"Sigma{noise_dec}"
#         result_file = os.path.join(folder_name, f"pcae_dataset_noisy_pc_{noise_value}_{category}.npy")
#         results = np.load(result_file, allow_pickle=True).item()
#         csv_path = results['csv_path']

#         df = pd.read_csv(csv_path)
#         target_files = df['target_uid'].apply(lambda x: os.path.join(target_directory, x + ".npz")).tolist()
#         target_pcs = [np.load(file)['pointcloud'] for file in target_files]

#         transformed_shapes = results['recons'][1]




#         # TODO: 
#         ## Read in csv 
#         ## Compute CD between target and generated point clouds from results['recons']

transformed_shapes_list = []
gt_pcs_list = []
gt_classes_list = []

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

        transformed_shapes_list.append(transformed_shapes)
        gt_pcs_list.append(target_pcs)
        gt_classes_list.append(df['target_object_class'].tolist())  # Assuming 'class' is the column containing class labels
        


transformed_shapes = np.concatenate(transformed_shapes_list, axis=0)
gt_pcs = np.concatenate(gt_pcs_list, axis=0)
gt_classes = np.concatenate(gt_classes_list, axis=0)


        # TODO: 
        ## Read in csv 
        ## Compute CD between target and generated point clouds from results['recons']

gpu_id = "0"  # GPU ID obtained from nvidia-smi

@torch.no_grad()
def compute_chamfer(transformed_shapes, gt_pcs, gt_classes):
    device = torch.device("cuda:" + gpu_id)
    scale_chamfer_by = 1000
    batch_size_for_cd = 2048
    for cmp_with in [gt_pcs]:
        holistic_cd_mu, holistic_cds = chamfer_dists(cmp_with, transformed_shapes,
                                                    bsize=min(len(transformed_shapes), batch_size_for_cd), 
                                                    device=device)
        torch.cuda.empty_cache()
        score = round(holistic_cd_mu * scale_chamfer_by, 3)
        score_per_class = (pd.concat([gt_classes, pd.DataFrame(holistic_cds.tolist())], axis=1).groupby('shape_class').mean()*scale_chamfer_by).round(3)
        score_per_class = score_per_class.reset_index().rename(columns={0: 'holistic-chamfer'})
        
        logger.info(f"Chamfer Distance (all pairs), Average: {score}")    
        logger.info(f"Chamfer Distance (all pairs), Average, per class:")
        logger.info(score_per_class)
        