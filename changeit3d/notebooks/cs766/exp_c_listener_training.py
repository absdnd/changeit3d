import torch
import numpy as np
import pandas as pd
import os.path as osp
from torch import nn
from torch import optim
from ast import literal_eval
import sys
sys.path.append('/home/shared/changeit3d')
from changeit3d.in_out.basics import (unpickle_data, create_logger, pickle_data, torch_save_model, save_state_dicts, load_state_dicts)

from changeit3d.in_out.arguments import parse_train_test_latent_listener_arguments
from changeit3d.in_out.language_contrastive_dataset import LanguageContrastiveDataset
from changeit3d.language.vocabulary import Vocabulary
from changeit3d.models.listening_oriented import ablation_model_one, ablation_model_two
from changeit3d.models.listening_oriented import single_epoch_train, evaluate_listener
from torch.utils.data import Dataset
from changeit3d.notebooks.cs766.helper import (
    pc_ae_transform_point_clouds, 
    get_transformed_latent_code,
    generate_notebook_args, 
    extract_data_df,
    generate_dataloaders,
    generate_config
)
import wandb
import os

data_drive = '/home/shared/data'
shape_talk_file = f'{data_drive}/shapetalk_dataset/shapetalk/language/shapetalk_preprocessed_public_version_0.csv'
vocab_file = f'{data_drive}/shapetalk_dataset/shapetalk/language/vocabulary.pkl'
log_dir =  f'{data_drive}/trained_listener/'
random_seed = 2022

# Shape Latent Code used for evaluation # 
shape_latent_encoder = 'pcae'  
top_pretrained_dir = f'{data_drive}/pretrained'
latent_codes_file = f'{top_pretrained_dir}/shape_latents/{shape_latent_encoder}_latent_codes.pkl'

weight_decay = 1e-3

evaluation_results = dict()  

def find_latest_timestamp(log_dir):
    ckpt_path = ""
    print("Log Dir: ", log_dir)
    for path in os.listdir(log_dir):
        # Processing time-stamps 
        if len(path.split('-'))  == 6:
            if "best_model.pt" in os.listdir(log_dir + "/" + path):
                ckpt_path = path + "/best_model.pt"
        else: 
            if path == "best_model.pt":
                ckpt_path = "best_model.pt"
    return ckpt_path

# Evaluate the Listener Model # 
def exp_c_eval_listener(args, df, model_type, shape_latent_encoder, model_params, run_name = None):

    def to_stimulus_func(x):
        return shape_to_latent_code[x]
    
    dataloaders = generate_dataloaders(args, df, to_stimulus_func=to_stimulus_func)
    
    shape_to_latent_code = next(unpickle_data(args.latent_codes_file))
    shape_latent_dim = len(list(shape_to_latent_code.values())[0])
    latest_timestamp = find_latest_timestamp(args.log_dir)

    if model_type == 'ablation_model_one':
        model = ablation_model_one(vocab, shape_latent_dim)

    elif model_type  == 'ablation_model_two':
        model = ablation_model_two(vocab, shape_latent_dim)

    model_path = args.log_dir + "/" + latest_timestamp


    device = torch.device("cuda:" + str(args.gpu_id))
    model = model.to(device)

    epoch = load_state_dicts(model_path, model=model)
    logger.info(f"Model loaded from epoch {epoch}: Evaluating now...")


    for split in ['test', 'val']:
        evaluation_results[split] = dict()
        res = evaluate_listener(model, dataloaders[split], device=device, return_logits=True)        
        evaluation_results[split]['accuracy'] = res['accuracy']

    # Performance Metrics #
    with open(osp.join(args.log_dir, 'eval_result.txt'), 'w') as f:
        f.write("Evaluation Results\n")
        for split in ['test', 'val']:
            f.write(f"{split} : {evaluation_results[split]['accuracy']}\n")

    np.save(osp.join(args.log_dir, 'eval_results.npy'), evaluation_results)


def exp_c_train_listener(args, df, model_type, shape_latent_encoder, model_params={}, run_name=None):

    def to_stimulus_func(x):
      return shape_to_latent_code[x]

    epochs_val_not_improved = best_test_accuracy = best_val_accuracy = 0
    start_epoch = 1
    checkpoint_file = osp.join(args.log_dir, 'best_model.pt')
    criterion = nn.CrossEntropyLoss()

    shape_to_latent_code = next(unpickle_data(args.latent_codes_file))
    shape_latent_dim = len(list(shape_to_latent_code.values())[0])

    df = extract_data_df(args)
    dataloaders = generate_dataloaders(args, df, to_stimulus_func=to_stimulus_func)

    if model_type == 'ablation_model_one':
        model = ablation_model_one(vocab, shape_latent_dim)

    elif model_type  == 'ablation_model_two':
        model = ablation_model_two(vocab, shape_latent_dim)
    
    else:
        import importlib
        model_module = importlib.import_module(f'changeit3d.cs766.models.{model_type}')
        model = model_module(vocab, shape_latent_dim)

    device = torch.device("cuda:" + str(args.gpu_id))
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                            factor=0.5, patience=args.lr_patience,
                                                            verbose=True, min_lr=5e-7)

    config = generate_config(
        model_type, 
        **model_params
      )

    if run_name is not None:
        wandb.init(
            project = 'changeit3d', 
            entity='sentiment-analyzer', 
            name=run_name, 
            config=config
        )

    for epoch in range(start_epoch + 1, start_epoch + args.max_train_epochs + 1):
        np.random.seed()
        train_results = single_epoch_train(model,
                                      dataloaders['train'],
                                      criterion, optimizer, device=device)
        train_acc = train_results['accuracy']
        train_loss = train_results['entropy_loss']

        logger.info(f"@epoch-{epoch} train {train_acc:.3f}")
        wandb.log(data = {"train/acc": train_acc, "train/loss": train_loss}, step = epoch)

        for split in ['val', 'test']:
            epoch_accuracy = evaluate_listener(model, dataloaders[split], device=device)['accuracy']

            if split == 'val':
                lr_scheduler.step(epoch_accuracy)

                # If best accuracy is obtained #
                if epoch_accuracy > best_val_accuracy:
                    epochs_val_not_improved = 0
                    best_val_accuracy = epoch_accuracy
                    save_state_dicts(checkpoint_file, epoch=epoch, model=model,
                                    optimizer=optimizer, lr_scheduler=lr_scheduler)
                else:
                    epochs_val_not_improved += 1

            wandb.log(data = {f"{split}/acc": epoch_accuracy}, step = epoch)
            logger.info("{} {:.3f}".format(split, epoch_accuracy))

            if split == 'test' and epochs_val_not_improved == 0:
                best_test_accuracy = epoch_accuracy

        if epochs_val_not_improved == 0:
            logger.info("* validation accuracy improved *")

        logger.info("\nbest test accuracy {:.3f}".format(best_test_accuracy))

        if epochs_val_not_improved == args.train_patience:
            logger.warning(
                f'Validation loss did not improve for {epochs_val_not_improved} consecutive epochs. Training is stopped.')
            break

    logger.info('Training is done!')
    best_epoch = load_state_dicts(checkpoint_file, model=model)
    logger.info(f'per-validation optimal epoch {best_epoch}')
    test_acc = evaluate_listener(model, dataloaders['test'], device=device, return_logits=True)['accuracy']
    logger.info(f'(verifying) test accuracy at that epoch is : {test_acc}')

    checkpoint_pkl_file = checkpoint_file = osp.join(args.log_dir, 'best_model.pkl')
    torch_save_model(model, checkpoint_pkl_file)


def extract_data_df(args):
    df = pd.read_csv(args.shape_talk_file)
    df.tokens_encoded = df.tokens_encoded.apply(literal_eval)
    vocab = Vocabulary.load(args.vocab_file)

    if len(args.restrict_shape_class) > 0:
        mask = df.target_object_class.isin(set(args.restrict_shape_class))
        df = df[mask].copy()
        df.reset_index(inplace=True, drop=True)


    df = df.assign(target=df.target_uid)
    df = df.assign(distractor_1=df.source_uid)
    return df


def get_random_id():
    import random, string
    return ''.join(random.choice(string.ascii_uppercase) for _ in range(3)) + ''.join(random.choice(string.digits) for _ in range(3))


def get_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Train a listener model')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--init-lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--shape-classes', type=str, default=None, help = 'comma separated list of shape classes')
    parser.add_argument('--model-type', type=str, default='ablation_model_one', help = 'ablation_model_one or ablation_model_two')
    parser.add_argument('--run-name', type=str, default=None, help = 'run name for wandb')
    parser.add_argument('--shape_talk_file', type=str, default=shape_talk_file)
    parser.add_argument('--run-mode', default='train', choices=['train', 'eval'], help='train or eval')
    train_args, _ = parser.parse_known_args()
    return train_args

if __name__ == "__main__":
    
    # Train Arguments from Parser # 
    train_args = get_parser()
    if train_args.shape_classes is None:
        restrict_shape_class = None
    else: 
        restrict_shape_class = train_args.shape_classes.split(',')

    run_log_dir =  log_dir + "/" + train_args.run_name
    notebook_args = generate_notebook_args(
        shape_talk_file=train_args.shape_talk_file,
        vocab_file=vocab_file,
        latent_codes_file=latent_codes_file,
        log_dir=run_log_dir,
        weight_decay=str(weight_decay),
        do_training='true',
        pretrained_model_file=f'{top_pretrained_dir}/listeners/all_shapetalk_classes/rs_2022/single_utter/transformer_based/latent_pcae_based/best_model.pt',
        save_analysis_results='true',
        restrict_shape_class=restrict_shape_class, 
        train_patience='25',
        )
    
    notebook_args.extend(['--use_timestamp', 'false'])
    args = parse_train_test_latent_listener_arguments(notebook_args)

    args.batch_size = train_args.batch_size
    args.weight_decay = train_args.weight_decay

    if train_args.run_name is None: 
        train_args.run_name = get_random_id()


    vocab = Vocabulary.load(args.vocab_file)
    os.makedirs(args.log_dir, exist_ok=True)
    logger = create_logger(args.log_dir)
    model_params = vars(args)

    df = extract_data_df(args)

    if train_args.run_mode == 'train':
        exp_c_train_listener(
            args,
            df, 
            train_args.model_type, 
            'pcae',
            model_params=model_params,
            run_name = train_args.run_name
        )
    
    else: 
        exp_c_eval_listener(
            args,
            df, 
            train_args.model_type, 
            'pcae',
            model_params=model_params,
            run_name = train_args.run_name
        )


    
    # if len(args.restrict_shape_class) > 0:
    #     mask = df.target_object_class.isin(set(args.restrict_shape_class))
    #     df = df[mask].copy()
    #     df.reset_index(inplace=True, drop=True)


    # df = df.assign(target=df.target_uid)
    # df = df.assign(distractor_1=df.source_uid)

    

    # all_listener_models = ['ablation_model_one', 'ablation_model_two']
    # latent_encoders = ['pcae', 'imnet', 'resnet101']

    # batch_size_list = [512, 2048]
    # weight_decay_list = [0.001, 0.01]

    # for listener_model in all_listener_models:
    #     run_name = listener_model + "_" + get_random_id()
    #     cnt = 0
    #     for batch_size in batch_size_list:
    #         for weight_decay in weight_decay_list:
    #             cur_run_name = run_name + "_" + str(cnt)
    #             cnt += 1
    #             args.batch_size = batch_size
    #             args.weight_decay = weight_decay
    #             args.log_dir = log_dir + "/" + cur_run_name 
    #             model_params = vars(args)

    #             os.makedirs(args.log_dir, exist_ok=True)
    #             logger = create_logger(args.log_dir)

    #             exp_c_train_listener(args, listener_model, 'pcae', model_params=model_params, run_name = cur_run_name)