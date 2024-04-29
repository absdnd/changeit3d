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

from changeit3d.evaluation.auxiliary import pc_ae_transform_point_clouds, sgf_transform_point_clouds
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

# Constants being written here. # 
top_data_dir = "/home/shared/data/"
top_train_dir = "/home/shared/data/"
shape_generator_type = "pcae"
shape_talk_file = f'{top_data_dir}/shapetalk/language/shapetalk_preprocessed_public_version_0.csv'
vocab_file = f'{top_data_dir}/shapetalk/language/vocabulary.pkl'
top_pc_dir = f'{top_data_dir}/shapetalk/point_clouds/scaled_to_align_rendering'
pretrained_oracle_listener = f'{top_train_dir}/pretrained/listeners/oracle_listener/all_shapetalk_classes/rs_2023/listener_dgcnn_based/ablation1/best_model.pkl'
pretrained_shape_classifier =  f'{top_train_dir}/pretrained/pc_classifiers/rs_2022/all_shapetalk_classes/best_model.pkl'
shape_part_classifiers_top_dir = f'{top_train_dir}/pretrained/part_predictors/shapenet_core_based'
latent_codes_file = f'{top_train_dir}/pretrained/shape_latents/{shape_generator_type}_latent_codes.pkl'
output_folder = f"{top_data_dir}/generation_results/language/"


lang_file_dict = {
    'lamp_baseline': f'{top_data_dir}/experiment2/lamp_baseline.csv',        
    'lamp_modified_lang': f'{top_data_dir}/experiment2/lamp_modified_language.csv',
    'chair_baseline': f'{top_data_dir}/experiment2/chair_baseline.csv',
    'chair_modified_lang': f'{top_data_dir}/experiment2/chair_modified_language.csv',
    'vase_baseline': f'{top_data_dir}/experiment2/vase_baseline.csv',
    'vase_modified_lang': f'{top_data_dir}/experiment2/vase_modified_language.csv',
}


batch_size = 4

def preprocess_and_load_df(args, df_file_name):
    cur_df = pd.read_csv(df_file_name)

    cur_df.tokens_encoded = cur_df.tokens_encoded.apply(literal_eval)
    vocab = Vocabulary.load(args.vocab_file)

    if hasattr(args, "add_shape_glot") and args.add_shape_glot:
        raise NotImplementedError('Not in public version')
        cur_df = add_sg_to_snt(cur_df, vocab, args.split_file)

    cur_df = cur_df.assign(target=cur_df.target_uid)
    cur_df = cur_df.assign(distractor_1=cur_df.source_uid)
    return cur_df

# Running Part A: Removing specific instances. # 
def run_part_a(lang_file_dict, output_folder, batch_size, args):
    
    for lang_exp_type, exp_file_name in lang_file_dict.items():
        cur_df = preprocess_and_load_df(args, exp_file_name)
        cur_df.reset_index(inplace=True, drop=True)

        msg = 'Restricting to class(es) {}. Total utterances: {}'.format(args.restrict_shape_class, len(cur_df))
        cur_dataset = ShapeTalkCustomShapeDataset(cur_df,
                                        to_stimulus_func,
                                        n_distractors=1,
                                        shuffle_items=False)  # important, source (distractor) now is first

        print(f"Processing {lang_exp_type}...with {len(cur_dataset)} samples.")
        cur_dataloader = torch.utils.data.DataLoader(dataset=cur_dataset,
                                            batch_size=8,
                                            shuffle=False,
                                            num_workers=4,
                                            worker_init_fn=lambda _ : np.random.seed(args.random_seed))


        transformation_results = pc_ae_transform_point_clouds(pc_ae,
                                                                c3d_net,
                                                                cur_dataloader,
                                                                stimulus_index=0,
                                                                scales=[0, 1],  # use "0" to get also the simple reconstruction of the decoder (no edit)
                                                                device=device)

        transformation_results["csv_path"] = exp_file_name
        np.save(os.path.join(output_folder, f"{args.shape_generator_type}_dataset_{lang_exp_type}.npy"), transformation_results)



def load_pretrained_nets():
    pc_ae, pc_ae_args = load_pretrained_pc_ae(args.pretrained_shape_generator)
    

    c3d_net, best_epoch, c3d_args = load_pretrained_changeit3d_net(args.pretrained_changeit3d, shape_latent_dim, vocab)
    return pc_ae, c3d_net

if __name__ == "__main__":

    # Pretrained Shape Generator #
    pretrained_shape_generator = f'{top_train_dir}/pretrained/pc_autoencoders/pointnet/rs_2022/points_4096/all_classes/scaled_to_align_rendering/08-07-2022-22-23-42/best_model.pt'
    selected_ablation = 'decoupling_mag_direction/idpen_0.05_sc_True/'
    pretrained_changeit3d = f'{top_train_dir}/pretrained/changers/{shape_generator_type}_based/all_shapetalk_classes/{selected_ablation}/best_model.pt'

    notebook_arguments = generate_notebook_args(
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
    args = parse_evaluate_changeit3d_arguments(notebook_arguments)
    logger = create_logger(args.log_dir)
    _, shape_to_latent_code, shape_latent_dim, vocab = prepare_input_data(args, logger)

    pc_ae, c3d_net = load_pretrained_nets()

    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.gpu_id))
    else:
        device = torch.device('cpu')

    pc_ae = pc_ae.to(device)
    pc_ae = pc_ae.eval()
    c3d_net = c3d_net.to(device)
    
    
    def to_stimulus_func(x):
        return shape_to_latent_code[x]

    
    run_part_a(lang_file_dict, output_folder, batch_size, args)
    