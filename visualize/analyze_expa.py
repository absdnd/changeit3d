import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import sys
from PIL import Image
sys.path.append('/home/shared/changeit3d')
# from changeit3d.utils.visualization import visualize_point_clouds_3d_v2

def visualize_point_clouds_3d_v2(
        pcl_lst, 
        title_lst=None, 
        vis_axis_order=[0, 2, 1], 
        fig_title=None,
        x_axis_label="",
        y_axis_label="",
    ):
    if title_lst is None:
        title_lst = [""] * len(pcl_lst)
    
    
    fig = plt.figure(figsize=(3.5 * len(pcl_lst), 4))
    # Turn of grid but show Axes labels
    # plt.grid(False)
    # plt.axes('off')
    # plt.axis('off')
    if fig_title is not None:
        plt.title(fig_title)
    plt.xticks([])
    plt.yticks([])

    # Increase font size
    plt.xlabel(x_axis_label, size=16)
    plt.ylabel(y_axis_label, size=16)
    for idx, (pts, title) in enumerate(zip(pcl_lst, title_lst)):
        ax1 = fig.add_subplot(1, len(pcl_lst), 1 + idx, projection='3d')
        ax1.set_title(title) 
        # ax1.view_init(elev=10, azim=245)      
        # Don't show axes
        ax1.axis('off')
        ax1.scatter(pts[:, vis_axis_order[0]], pts[:, vis_axis_order[1]], pts[:, vis_axis_order[2]], s=2)
    fig.canvas.draw()

    # grab the pixel buffer and dump it into a numpy array
    res = np.array(fig.canvas.renderer._renderer)
    res = np.transpose(res, (2, 0, 1))

    plt.tight_layout()
    plt.close()
    res = Image.fromarray(res[:3].transpose(1,2,0))
    return res

# data_drive = '/home/shared/data'
# shape_talk_file = f'{data_drive}/shapetalk_dataset/shapetalk/language/shapetalk_preprocessed_public_version_0.csv'
# vocab_file = f'{data_drive}/shapetalk_dataset/shapetalk/language/vocabulary.pkl'
# log_dir =  f'{data_drive}/trained_listener/'
# random_seed = 2022

# shape_latent_encoder = 'pcae'  
# top_pretrained_dir = f'{data_drive}/pretrained'
# latent_codes_file = f'{top_pretrained_dir}/shape_latents/{shape_latent_encoder}_latent_codes.pkl'


target_directory = "/home/shared/data/shapetalk_dataset/shapetalk/point_clouds/scaled_to_align_rendering"
result_dir = "/home/shared/data/generation_results"

exp_type = "language"  # language or noisy_point_clouds
folder_name = os.path.join(result_dir, exp_type)
categories = ["lamp", "chair", "vase", "mug", "bottle"]
exp_category = ["baseline", "modified_lang"]

# Create a directory to save the visualizations if it doesn't exist
viz_dir = "/home/shared/visualizations/exp_a_lang"
os.makedirs(viz_dir, exist_ok=True)


for category in categories:
    category_cnt = 0
    df_dict = {}  # Dictionary to hold dataframes for both experiments
    point_clouds_dict = {}  # Dictionary to hold point clouds for both experiments

    # Load results for both experiment types
    for exp in exp_category:
        result_file = os.path.join(folder_name, f"pcae_dataset_{category}_{exp}.npy")
        results = np.load(result_file, allow_pickle=True).item()
        csv_path = results['csv_path']
        df = pd.read_csv(csv_path)
        df_dict[exp] = df
        point_clouds_dict[exp] = []

        # Process point clouds for each experiment
        for i in range(len(df)):
            source_file = os.path.join(target_directory, df.loc[i, 'source_uid'] + ".npz")
            target_file = os.path.join(target_directory, df.loc[i, 'target_uid'] + ".npz")
            transformed_shape = results['recons'][1][i]

            with np.load(source_file) as data:
                source_pc = data['pointcloud']
            with np.load(target_file) as data:
                target_pc = data['pointcloud']
            
            point_clouds_dict[exp].append((source_pc, transformed_shape, target_pc))

    # Create visualizations for each index and combine them into a single image
    for i in range(len(df_dict['baseline'])):  # Assume equal length for simplicity
        combined_images = []
        for exp in exp_category:
            point_clouds = point_clouds_dict[exp][i]
            prompt = " ".join(ast.literal_eval(df_dict[exp].loc[i, 'tokens']))
            if exp == "baseline":
                y_axis_label = "Baseline"
            else: 
                y_axis_label = "Modified Lang"
            img = visualize_point_clouds_3d_v2(point_clouds, 
                                               title_lst=["Source", "Modified", "Target"],
                                               vis_axis_order=[0, 2, 1],
                                               fig_title= '"' + prompt + '"',
                                               y_axis_label=y_axis_label)
            combined_images.append(img)
        
        total_width = max(img.width for img in combined_images)
        total_height = sum(img.height for img in combined_images)
        new_img = Image.new('RGB', (total_width, total_height))
        
        y_offset = 0
        for img in combined_images:
            new_img.paste(img, (0, y_offset))
            y_offset += img.height
        
        print(f"Creating visualization for {category} index {i}... Path: {viz_dir}")
        # Save the combined visualization as an image directly using PIL
        save_path = os.path.join(viz_dir, f"{category}_index_{i}.png")
        category_cnt += 1
        new_img.save(save_path)