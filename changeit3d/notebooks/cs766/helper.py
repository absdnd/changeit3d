from argparse import ArgumentParser
import json
from changeit3d.models.point_net import PointNet
from changeit3d.models.mlp import MLP
from changeit3d.models.pointcloud_autoencoder import PointcloudAutoencoder
from changeit3d.in_out.basics import load_state_dicts
from changeit3d.language.vocabulary import Vocabulary
from torch.utils.data import Dataset
import os.path as osp
import torch
import pandas as pd
from ast import literal_eval
import numpy as np
import uuid
from changeit3d.in_out.language_contrastive_dataset import LanguageContrastiveDataset



from collections import defaultdict

@torch.no_grad()
def pc_ae_transform_point_clouds(
    pc_ae,
    direction_finder,
    data_loader,
    stimulus_index,
    scales=[1],
    device="cuda",
    encode_stimulus = False
  ):
    """

    Args:
        pc_ae:
        direction_finder:
        data_loader:
        stimulus_index: stimulus location in data_loader (e.g., is it the first or the second shape)
        scales: use them to multiply the edit_latent before you add it to the original latent, this way you can *manually* boost or attenuate the edit's effect
        device:

    Returns:
        Let's assume that:
            1) the input language/shape-dataset concerns the transformation for N shape-language pairs.
            2) the latent space is L-dimensional

        then,

         A dictionary carrying the following items:
            'z_codes' -> dict, carrying the updated *final* latent codes. The keys are the input magnitudes. Each value is
                an N x L numpy array.
            'recons' -> dict, the N decoded/reconstructed point-clouds. The keys are input magnitudes. Each value is
                an N x PC-points x 3 numpy array.
            'tokens' -> list of lists, the N sentences used to create the transformations
            'edit_latents' -> N x L numpy array, the latents corresponding to the edits (before adding them to each input)
            'magnitudes' -> N x 1 numpy array, carrying the magnitudes the editing network guessed for each input.
    """

    results = get_transformed_latent_code(pc_ae, direction_finder, data_loader, stimulus_index, scales=scales, device=device, encode_stimulus = encode_stimulus)

    pc_ae.eval()
    all_recons = defaultdict(list)

    for key, val in results['z_codes'].items():
        recons = pc_ae.decoder(torch.from_numpy(val).to(device))
        recons = recons.view([len(recons), -1, 3]).cpu()
        all_recons[key].append(recons)

    for key in all_recons:
        all_recons[key] = torch.cat(all_recons[key]).numpy()

    results['recons'] = all_recons
    return results

@torch.no_grad()
def get_transformed_latent_code(pc_ae, direction_finder, data_loader, stimulus_index, scales=[1], device="cuda", encode_stimulus = False):
    """  Extract transformation for a given latent code based on LatentDirectionFinder
    """
    direction_finder.eval()

    all_z_codes = defaultdict(list)
    all_tokens = []
    all_edit_latents = []
    all_magnitudes = []

    # Enumerating the batches of data loader #
    for batch in data_loader:
        t = batch['tokens'].to(device)

        # Should I encode the stimulus function? #
        if encode_stimulus:
          batch_size, n_distractors, num_points, d = batch['pc_stimulus'].shape
          ae_input = batch['pc_stimulus'].view(-1, num_points, d).permute(0, 2, 1).to(device)
          batch['stimulus'] = pc_ae.encoder(ae_input.float())
          batch['stimulus'] = batch['stimulus'].view(batch_size, n_distractors, -1)

        s = batch['stimulus'][:, stimulus_index].to(device)
        edit_latent, guessed_mag = direction_finder(t, s)
        # print(edit_latent, guessed_mag)

        for scale in scales:
            if scale == 0:  # no transformation, just return the input/starting latent code
                transformed = s
            else:
                transformed = s + scale * edit_latent

            all_z_codes[scale].append(transformed.cpu())
        all_edit_latents.append(edit_latent.cpu())
        all_magnitudes.append(guessed_mag.cpu())

        all_tokens.extend(t.tolist())


    all_edit_latents = torch.cat(all_edit_latents).numpy()
    all_magnitudes = torch.cat(all_magnitudes).numpy()

    for scale in scales:
        all_z_codes[scale] = torch.cat(all_z_codes[scale]).numpy()

    results = {'z_codes': all_z_codes,
               'tokens': all_tokens,
               'edit_latents': all_edit_latents,
               'magnitudes': all_magnitudes}

    return results

def load_custom_df(shape_talk_file, args):
    df = pd.read_csv(shape_talk_file)
    df.tokens_encoded = df.tokens_encoded.apply(literal_eval)
    vocab = Vocabulary.load(args.vocab_file)

    if hasattr(args, "add_shape_glot") and args.add_shape_glot:
        raise NotImplementedError('Not in public version')
        df = add_sg_to_snt(df, vocab, args.split_file)

    df = df.assign(target=df.target_uid)
    df = df.assign(distractor_1=df.source_uid)

    # if len(args.restrict_shape_class) > 0:
        # mask = df.target_object_class.isin(set(args.restrict_shape_class))
        # df = df[mask].copy()
        # df.reset_index(inplace=True, drop=True)

    return df


# def generate_config(
#     listner_model,
#     vocab,
#     shape_latent_dim,
#     criterion,
#     log_dir,
#     shape_latent_encoder,
#     **kwargs
# ):
#   config = {
#     "model_type": listner_model,
#     "vocab_size": len(vocab),
#     "latent_dim": shape_latent_dim,
#     "criterion": criterion,
#     "log_dir": log_dir,
#     "modality": shape_latent_encoder
#   }
#   config.update(kwargs)
#   return config


def generate_config(
    model_type, 
    **kwargs
):
  return kwargs.update({"model_type": model_type})


def extract_data_df(args, shape_to_latent_code, logger):
    df = pd.read_csv(args.shape_talk_file)
    df.tokens_encoded = df.tokens_encoded.apply(literal_eval)
    vocab = Vocabulary.load(args.vocab_file)

    if args.add_shape_glot:
        raise NotImplementedError('left out of public code')

    # constraint training in language of particular classes
    if len(args.restrict_shape_class) > 0:
        mask = df.target_object_class.isin(set(args.restrict_shape_class))
        df = df[mask].copy()
        df.reset_index(inplace=True, drop=True)
        logger.info('Restricting to class(es) {}. Total utterances: {}'.format(args.restrict_shape_class, len(df)))

    assert df.target_uid.apply(lambda x: x in shape_to_latent_code).all(), 'all loaded stimuli must have a latent code'
    assert df.source_uid.apply(lambda x: x in shape_to_latent_code).all(), 'all loaded stimuli must have a latent code'
    df = df.assign(target=df.target_uid)
    df = df.assign(distractor_1=df.source_uid)
    return df





def generate_dataloaders(args, df, to_stimulus_func):
    dataloaders = dict()
    for split in ['train', 'val', 'test']:
        ndf = df[df.listening_split == split].copy()
        ndf.reset_index(inplace=True, drop=True)
        seed = None if split == 'train' else args.random_seed
        batch_size = args.batch_size if split == 'train' else 2 * args.batch_size
        shuffle_items = split == 'train'

        dataset = LanguageContrastiveDataset(ndf,
                                            to_stimulus_func,
                                            n_distractors=1,
                                            shuffle_items=shuffle_items)

        dataloaders[split] = torch.utils.data.DataLoader(dataset=dataset,
                                                        batch_size=args.batch_size,
                                                        shuffle=shuffle_items,
                                                        num_workers=args.num_workers,
                                                        worker_init_fn=lambda x: np.random.seed(seed))
    return dataloaders
def generate_notebook_args(
    shape_talk_file, 
    latent_codes_file, 
    vocab_file, 
    pretrained_changeit3d=None,
    top_pc_dir=None,
    shape_generator_type=None,
    pretrained_oracle_listener=None,
    pretrained_shape_classifier=None,
    shape_part_classifiers_top_dir=None,
    pretrained_shape_generator=None,
    sub_sample_dataset=None,
    do_training=None,
    pretrained_model_file=None,
    save_analysis_results=None,
    restrict_shape_class=None,
    shape_latent_encoder=None,
    weight_decay=None,
    log_dir=None,
    random_seed=None,
    train_patience=None,
    use_timestamp=None
): 
    notebook_arguments = []
    notebook_arguments.extend(['-shape_talk_file', shape_talk_file])
    notebook_arguments.extend(['-latent_codes_file', latent_codes_file])
    notebook_arguments.extend(['-vocab_file', vocab_file])
    if pretrained_changeit3d is not None:
        notebook_arguments.extend(['-pretrained_changeit3d', pretrained_changeit3d])
    if top_pc_dir is not None:
        notebook_arguments.extend(['-top_pc_dir', top_pc_dir])
    if shape_generator_type is not None:
        notebook_arguments.extend(['--shape_generator_type', shape_generator_type])
    if pretrained_oracle_listener is not None:
        notebook_arguments.extend(['--pretrained_oracle_listener', pretrained_oracle_listener])
    if pretrained_shape_classifier is not None:
        notebook_arguments.extend(['--pretrained_shape_classifier', pretrained_shape_classifier])
    if shape_part_classifiers_top_dir is not None:
        notebook_arguments.extend(['--shape_part_classifiers_top_dir', shape_part_classifiers_top_dir])
    if train_patience is not None: 
        notebook_arguments.extend(['--train_patience', train_patience])
    if use_timestamp is not None:
        notebook_arguments.extend(['--use_timestamp', use_timestamp])
    # notebook_arguments.extend(['-pretrained_changeit3d', pretrained_changeit3d])
    # notebook_arguments.extend(['-top_pc_dir', top_pc_dir])
    # notebook_arguments.extend(['--shape_generator_type', shape_generator_type])
    # notebook_arguments.extend(['--pretrained_oracle_listener', pretrained_oracle_listener])
    # notebook_arguments.extend(['--pretrained_shape_classifier', pretrained_shape_classifier])
    # notebook_arguments.extend(['--shape_part_classifiers_top_dir', shape_part_classifiers_top_dir])
    

    if do_training is not None:
        notebook_arguments.extend(['--do_training', do_training])
    if pretrained_model_file is not None:
        notebook_arguments.extend(['--pretrained_model_file', pretrained_model_file])
    if save_analysis_results is not None:
        notebook_arguments.extend(['--save_analysis_results', save_analysis_results])
    if restrict_shape_class is not None:
        restrict_args = ['--restrict_shape_class'] + restrict_shape_class
        notebook_arguments.extend(restrict_args)
    if weight_decay is not None:
        notebook_arguments.extend(['--weight_decay', weight_decay])
    if log_dir is not None:
        notebook_arguments.extend(['--log_dir', log_dir])
    if random_seed is not None:
        notebook_arguments.extend(['--random_seed', random_seed])
    if shape_latent_encoder is not None:
        notebook_arguments.extend(['--shape_latent_encoder', shape_latent_encoder])

    if pretrained_shape_generator is not None:
        notebook_arguments.extend(['--pretrained_shape_generator', pretrained_shape_generator])


    # if 'sub_sample_dataset' in  locals():
        # notebook_arguments.extend(['--sub_sample_dataset', sub_sample_dataset])

    return notebook_arguments
class ShapeTalkCustomShapeDataset(Dataset):
  def __init__(self, data_frame, to_stimulus_func=None, shuffle_items=False, n_distractors=2,
                 shape_to_latent_code=None, encoder=None, base_data_dir = None, target_base_dir = None):
        """
        Args:
            data_frame:
            to_stimulus_func:
            shuffle_items:
            n_distractors:
            shape_to_latent_code:
        """
        super(ShapeTalkCustomShapeDataset, self).__init__()
        self.df = data_frame
        self.shuffle_items = shuffle_items
        self.n_distractors = n_distractors
        self.to_stimulus_func = to_stimulus_func
        self.shape_to_latent_code = shape_to_latent_code
        self.encoder = encoder
        self.base_data_dir = base_data_dir
        self.target_base_dir = target_base_dir
        print(self.df.keys())
        print("Number of Distractors", self.n_distractors)



  def __getitem__(self, index):
      row = self.df.iloc[index]
      tokens = row['tokens_encoded']
      tokens = np.array(tokens).T  # todo do via collate.

      item_ids = []
      
      for i in range(1, self.n_distractors + 1):
          item_ids.append(row[f'distractor_{i}'])

      item_ids.append(row['target'])  # now target is last.
      item_ids = np.array(item_ids)
      n_items = len(item_ids)
      label = n_items - 1


      if self.shuffle_items:
          idx = np.arange(n_items)
          np.random.shuffle(idx)
          item_ids = item_ids[idx]
          label = np.where(idx == label)[0][0]

      res = dict()
      res['tokens'] = tokens
      res['label'] = label
      res['index'] = index
      res['pc_stimulus'] = index

      if self.to_stimulus_func is None:
          res['pc_stimulus'] = []
          for i, x in enumerate(item_ids):
            path_depth = x.count("/")
            if path_depth == 3:
                item_class, item_dataset, noise_value, item_code = x.split("/")
                test_pc_path = self.base_data_dir + "/" + item_class + "/" + item_dataset + "/" + noise_value + "/" + item_code

            # Comparing the Path Depth of ShapeTalk # 
            elif path_depth == 2:
                item_class, item_dataset, item_code = x.split("/")
                test_pc_path = self.target_base_dir + "/" + item_class + "/" + item_dataset + "/" + item_code + ".npz"
                         
            else: 
                raise ValueError("Invalid Path Depth")
            
            res['pc_stimulus'].append(np.load(test_pc_path)['pointcloud'])
        
          res['pc_stimulus'] = np.stack(res['pc_stimulus'])
          res['stimulus'] = index
      else:
          # print("Using Stimulus", self.to_stimulus_func)
          res['stimulus'] = []
          for x in item_ids:
              res['stimulus'].append(self.to_stimulus_func(x))
          res['stimulus'] = np.stack(res['stimulus'])
      return res

  def __len__(self):
      return len(self.df)

# Read Saved Arguments # 
def read_saved_args(config_file, override_or_add_args=None, verbose=False):
    """
    :param config_file: json file containing arguments
    :param override_args: dict e.g., {'gpu': '0'} will set the resulting arg.gpu to be 0
    :param verbose:
    :return:
    """
    parser = ArgumentParser()
    args = parser.parse_args([])
    with open(config_file, 'r') as f_in:
        args.__dict__ = json.load(f_in)

    if override_or_add_args is not None:
        for key, val in override_or_add_args.items():
            args.__setattr__(key, val)


    return args

def describe_pc_ae(args):
    # Make an AE.
    if args.encoder_net == 'pointnet':
        ae_encoder = PointNet(init_feat_dim=3, conv_dims=args.encoder_conv_layers)
        encoder_latent_dim = args.encoder_conv_layers[-1]
    else:
        raise NotImplementedError()

    if args.decoder_net == 'mlp':
        ae_decoder = MLP(in_feat_dims=encoder_latent_dim,
                         out_channels=args.decoder_fc_neurons + [args.n_pc_points * 3],
                         b_norm=False)

    model = PointcloudAutoencoder(ae_encoder, ae_decoder)
    return model



def load_pretrained_pc_ae(model_file):
    config_file = osp.join(osp.dirname(model_file), 'config.json.txt')
    pc_ae_args = read_saved_args(config_file)
    pc_ae = describe_pc_ae(pc_ae_args)

    # if osp.join(pc_ae_args.log_dir, 'best_model.pt') != osp.abspath(model_file):
        # warnings.warn("The saved best_model.pt in the corresponding log_dir is not equal to the one requested.")

    if torch.cuda.is_available():
      best_epoch = load_state_dicts(model_file, model=pc_ae)
    else:
      best_epoch = load_state_dicts(model_file, model=pc_ae, map_location='cpu')
    print(f'Pretrained PC-AE is loaded at epoch {best_epoch}.')
    return pc_ae, pc_ae_args