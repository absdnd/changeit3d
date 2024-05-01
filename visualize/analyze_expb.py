import numpy as np
import os
import pandas as pd
# from mayavi import mlab
from changeit3d.utils.visualization import visualize_point_clouds_3d_v2
import matplotlib.pyplot as plt
import ast 
from PIL import Image


# Increase font size
plt.rcParams.update({'font.size': 14})
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

noisy_base_path = "/home/shared/data/noise_added_to_point_clouds/"
result_path = "/home/shared/data/generation_results/noisy_point_clouds_v2"
pc_base_path = "/home/shared/data/shapetalk_dataset/shapetalk/point_clouds/scaled_to_align_rendering"
vis_path = "/home/shared/visualizations/exp_b_noisy"

target_pc_path = ""
categories_list = ["chair", "lamp", "mug", "vase"]
sigma_list = ['0.00', '0.01',  '0.02', '0.03']
max_img_per_category = 20

for category in categories_list: 
    category_cnt = 0

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
            language_instruction = " ".join(language_instruction)
            fig = visualize_point_clouds_3d_v2([
                sigma_source_pc, sigma_recons, target_pc], 
                title_lst=["Source", "Modified",  "Target"], 
                vis_axis_order=[0, 2, 1], 
                fig_title="{}".format('"' + language_instruction + '"'), 
                y_axis_label="Sigma = {}".format(sigma)
            )
            stacked_image.append(fig)


        # PIL Image stacking # 
        new_im = Image.new('RGB', (stacked_image[0].width, stacked_image[0].height * len(sigma_list)))
        y_offset = 0
        for im in stacked_image:
            new_im.paste(im, (0, y_offset))
            y_offset += im.size[1]  
        
    # fig, ax = plt.subplots(len(sigma_list), 1, figsize=(10, 10))
    # for i, image in enumerate(stacked_image):
    #     ax[i].imshow(image)
    #     ax[i].set_ylabel(f"Sigma = {sigma_list[i]}")
    #     # Don't show axes 
    #     ax[i].axis('off')

        print("Saving image: ", index)
        category_cnt += 1
        if category_cnt > max_img_per_category: 
            break
        new_im.save(os.path.join(vis_path, f"{category}_{index}.png"))
        
