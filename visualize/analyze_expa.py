import numpy as np
import os
import pandas as pd
from changeit3d.utils.visualization import visualize_point_clouds_3d_v2
import matplotlib.pyplot as plt
import ast 

data_drive = '/home/shared/data'
shape_talk_file = f'{data_drive}/shapetalk_dataset/shapetalk/language/shapetalk_preprocessed_public_version_0.csv'
vocab_file = f'{data_drive}/shapetalk_dataset/shapetalk/language/vocabulary.pkl'
log_dir =  f'{data_drive}/trained_listener/'
random_seed = 2022

shape_latent_encoder = 'pcae'  
top_pretrained_dir = f'{data_drive}/pretrained'
latent_codes_file = f'{top_pretrained_dir}/shape_latents/{shape_latent_encoder}_latent_codes.pkl'

