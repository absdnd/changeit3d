import numpy as np
import os
import pandas as pd
# from mayavi import mlab
from changeit3d.utils.visualization import visualize_point_clouds_3d_v2
import matplotlib.pyplot as plt
import ast 

noisy_base_path = "data/noise_added_to_point_clouds/"
result_path = "data/generation_results/noisy_point_clouds_v2"
pc_base_path = "data/shapetalk_dataset/shapetalk/point_clouds/scaled_to_align_rendering"
vis_path = "visualizations/exp_b_noisy"

target_pc_path = ""
category = "chair"

sigma_list = ['0.00', '0.01',  '0.02', '0.03']
sigma_str = [str(sigma).split(".")[1] for sigma in sigma_list]

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


for index in range(result_len):
    
    src_pt_clouds = {}
    modified_pt_clouds = {}

    stacked_image = []
    for sigma in sigma_list:
        sigma_df = result_dict[sigma]["df"]
        target_uid = sigma_df.iloc[index]['target_uid']
        source_uid = sigma_df.iloc[index]['source_uid']

        row = sigma_df.iloc[index]
        sigma_recons = result_dict[sigma]["result_dict"]['recons'][0][index]

        if sigma == '0.00':
            source_pc_path = os.path.join(pc_base_path, source_uid) + ".npz"
        else: 
            source_pc_path = os.path.join(noisy_base_path, row['source_uid'])
        
        with np.load(source_pc_path) as data:   
            sigma_source_pc = data['pointcloud']
        
        target_pc_path = os.path.join(pc_base_path, target_uid) + ".npz"
        with np.load(target_pc_path) as data:
            target_pc = data['pointcloud']

        language_instruction = ast.literal_eval(row['tokens'])
        labguage_instruction = " ".join(language_instruction)
        fig = visualize_point_clouds_3d_v2([sigma_source_pc, sigma_recons, target_pc], title_lst=["Source", "Recons", "Distractor", "Target"], vis_axis_order=[0, 2, 1], fig_title="{}".format(language_instruction))
        stacked_image.append(fig)


    fig, ax = plt.subplots(len(sigma_list), 1, figsize=(10, 10))
    for i, image in enumerate(stacked_image):
        ax[i].imshow(image)
        ax[i].set_ylabel(f"Sigma = {sigma_list[i]}")
        # Don't show axes 
        ax[i].axis('off')

    # 
    os.makedirs(os.path.join(vis_path, category), exist_ok=True)
    plt.savefig(os.path.join(vis_path, category, f"{index}.png"))
    plt.clf()
