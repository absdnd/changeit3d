import torch
import numpy as np
import os.path as osp
from functools import partial


import os
from changeit3d.in_out.changeit3d_net import prepare_input_data
from changeit3d.in_out.language_contrastive_dataset import LanguageContrastiveDataset
from changeit3d.in_out.pointcloud import pc_loader_from_npz, uniform_subsample
from changeit3d.in_out.basics import pickle_data
from changeit3d.in_out.basics import create_logger
from changeit3d.in_out.arguments import parse_evaluate_changeit3d_arguments

from changeit3d.utils.basics import parallel_apply
from changeit3d.models.model_descriptions import load_pretrained_changeit3d_net
from changeit3d.models.model_descriptions import load_pretrained_pc_ae

from changeit3d.notebooks.cs766.helper import pc_ae_transform_point_clouds, get_transformed_latent_code
# from changeit3d.evaluation.auxiliary import pc_ae_transform_point_clouds, sgf_transform_point_clouds
from changeit3d.external_tools.sgf.loader import initialize_and_load_sgf
from changeit3d.utils.visualization import visualize_point_clouds_3d_v2, plot_3d_point_cloud
from changeit3d.in_out.arguments import parse_evaluate_changeit3d_arguments
from changeit3d.in_out.changeit3d_net import prepare_input_data
from changeit3d.notebooks.cs766.helper import (
    describe_pc_ae, 
    load_pretrained_pc_ae, 
    read_saved_args, 
    ShapeTalkCustomShapeDataset, 
    generate_notebook_args
)
import pandas as pd
from torch.utils.data import Dataset

from changeit3d.language.vocabulary import Vocabulary
from ast import literal_eval
import os

# Top Data Directory # 
top_data_dir = "/home/shared/data/"
top_train_dir = "/home/shared/data/"
shape_generator_type = "pcae"
shape_talk_file = f'{top_data_dir}/shapetalk/language/shapetalk_preprocessed_public_version_0.csv'
vocab_file = f'{top_data_dir}/shapetalk/language/vocabulary.pkl'
top_pc_dir = f'{top_data_dir}/shapetalk_dataset/shapetalk/point_clouds/scaled_to_align_rendering'
pretrained_oracle_listener = f'{top_train_dir}/pretrained/listeners/oracle_listener/all_shapetalk_classes/rs_2023/listener_dgcnn_based/ablation1/best_model.pkl'
pretrained_shape_classifier =  f'{top_train_dir}/pretrained/pc_classifiers/rs_2022/all_shapetalk_classes/best_model.pkl'
shape_part_classifiers_top_dir = f'{top_train_dir}/pretrained/part_predictors/shapenet_core_based'
latent_codes_file = f'{top_train_dir}/pretrained/shape_latents/{shape_generator_type}_latent_codes.pkl'

output_folder = f"{top_data_dir}/generation_results/noisy_point_clouds_v3"
os.makedirs(output_folder, exist_ok=True)

noisy_pc_file_dict = {
    'public_version': f'{top_data_dir}/shapetalk/language/shapetalk_preprocessed_public_version_0.csv',        
}

sigma_list = ['0.1', '0.2', '0.00', '0.01', '0.02', '0.03', '0.04', '0.05', '0.025', '0.035', '0.045', '0.055', '0.003', '0.005', '0.008']
# categories = ['mug', 'lamp', 'bottle', 'chair']
categories = ['vase']


# Point cloud data paths
pc_data_path = f"{top_data_dir}/100pc/"
csv_base_path = f"{top_data_dir}/100pc/noiseCSVs/"

target_base_dir = f"{top_data_dir}/shapetalk_dataset/shapetalk/point_clouds/scaled_to_align_rendering"
csv_out_path = f"{top_data_dir}/../noisy_data.csv"
exp_type = "noisy_pc"

def load_custom_df(shape_talk_file, args):
    df = pd.read_csv(shape_talk_file)
    df.tokens_encoded = df.tokens_encoded.apply(literal_eval)
    vocab = Vocabulary.load(args.vocab_file)

    if hasattr(args, "add_shape_glot") and args.add_shape_glot:
        raise NotImplementedError('Not in public version')
        df = add_sg_to_snt(df, vocab, args.split_file)

    df = df.assign(target=df.target_uid)
    df = df.assign(distractor_1=df.source_uid)
    return df

def load_pretrained_nets(args):
    pc_ae, pc_ae_args = load_pretrained_pc_ae(args.pretrained_shape_generator)
    
    c3d_net, best_epoch, c3d_args = load_pretrained_changeit3d_net(args.pretrained_changeit3d, shape_latent_dim, vocab)
    return pc_ae, c3d_net

def run_exp_b(
        args, 
        sigma, 
        category, 
        device='cuda:0', 
        shape_talk_file=None
    ):
    if shape_talk_file is None:
        shape_talk_file = csv_base_path + "/"  + category + "_noise" + "Sigma" + sigma.split('.')[1] + "_exp1.csv"
    else: 
        shape_talk_file = shape_talk_file
    cur_df = load_custom_df(shape_talk_file, args)

    cur_dataset = ShapeTalkCustomShapeDataset(
                                      cur_df,
                                      base_data_dir=pc_data_path,
                                      target_base_dir=target_base_dir,
                                      to_stimulus_func=None,
                                      n_distractors=1,
                                      shuffle_items=False)  # important, source (distractor) now is first

    print(f"Running {sigma} on the Category: {category}, Dataset Size: {len(cur_dataset)}")

    cur_dataloader = torch.utils.data.DataLoader(dataset=cur_dataset,
                                            batch_size=8,
                                            shuffle=False,
                                            num_workers=1,
                                            worker_init_fn=lambda _ : np.random.seed(args.random_seed))



    transformation_results = pc_ae_transform_point_clouds(
        pc_ae,
        c3d_net,
        cur_dataloader,
        stimulus_index=0,
        scales=[0, 1],  # use "0" to get also the simple reconstruction of the decoder (no edit)
        device=device,
        encode_stimulus=True
    )

    # Transformation Results # 
    transformation_results["csv_path"] = shape_talk_file
    np.save(os.path.join(output_folder, f"{args.shape_generator_type}_dataset_{exp_type}_{sigma}_{category}.npy"), transformation_results)
    

if __name__ == "__main__":

    print("Using Shape Generator Type: ", shape_generator_type)

    pretrained_shape_generator = f'{top_train_dir}/pretrained/pc_autoencoders/pointnet/rs_2022/points_4096/all_classes/scaled_to_align_rendering/08-07-2022-22-23-42/best_model.pt'
    selected_ablation = 'decoupling_mag_direction/idpen_0.05_sc_True/'
    pretrained_changeit3d = f'{top_train_dir}/pretrained/changers/{shape_generator_type}_based/all_shapetalk_classes/{selected_ablation}/best_model.pt'

    notebook_args = generate_notebook_args(
        shape_talk_file,
        latent_codes_file,
        vocab_file,
        pretrained_changeit3d,
        top_pc_dir,
        shape_generator_type,
        pretrained_oracle_listener,
        pretrained_shape_classifier,
        shape_part_classifiers_top_dir,
        pretrained_shape_generator, 
    )

    args = parse_evaluate_changeit3d_arguments(notebook_args)
    logger = create_logger(args.log_dir)
    _, shape_to_latent_code, shape_latent_dim, vocab = prepare_input_data(args, logger)

    pc_ae, c3d_net = load_pretrained_nets(args)
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.gpu_id))
    else:
        device = torch.device('cpu')

    pc_ae = pc_ae.to(device)
    pc_ae = pc_ae.eval()
    c3d_net = c3d_net.to(device)

    def to_stimulus_func(x):
        return shape_to_latent_code[x]

    for category in categories:
        for sigma in sigma_list:
            run_exp_b(args, sigma, category)

        # run_exp_b(
        #     args, 
        #     "00", 
        #     category, 
        #     shape_talk_file=f"/home/shared/data/noise_added_to_point_clouds/{category}_baseline_exp1.csv"   
        # )