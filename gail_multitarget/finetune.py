import sys

sys.path.append("..")
import os
import hydra
import torch
import logging
import wandb
import pickle
import pandas as pd
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from utils import set_random_seed, predict_model, tokens_struct, selfies_tokens_struct
from model import SmilesGAILModel
from rewardfunc import RewardFunction
from gail_trainerloop import GAILTrainerLoop

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    config = OmegaConf.create(OmegaConf.to_yaml(cfg))
    output_path = os.getcwd()
    original_path = hydra.utils.get_original_cwd()
    root_path = '/'.join(original_path.split('/')[:-1])
    task_config = config.train.gail
    # set cuda
    if torch.cuda.is_available():
        dvc_id = 0
        device = f'cuda:{dvc_id}'
    else:
        device = 'cpu'
    set_random_seed(task_config.random_seed)
    config['device'] = device
    if task_config.use_selfies:
        vocab = selfies_tokens_struct()
        drd2_path = os.path.join(root_path, config.target.drd2.train_sf_path)
        htr1a_path = os.path.join(root_path, config.target.htr1a.train_sf_path)
        model_name = 'MLE_best_generator.pt'
    else:
        vocab = tokens_struct()
        drd2_path = os.path.join(root_path, config.target.drd2.train_path)
        htr1a_path = os.path.join(root_path, config.target.htr1a.train_path)
        model_name = 'Prior_DLGN.ckpt'

    drd2_train_data = pd.read_csv(drd2_path, index_col=False, header=None)
    htr1a_train_data = pd.read_csv(htr1a_path, index_col=False, header=None)

    drd2_valid_path = os.path.join(root_path, config.target.drd2.valid_path)
    htr1a_valid_path = os.path.join(root_path, config.target.htr1a.valid_path)
    drd2_valid_data = pd.read_csv(drd2_valid_path, index_col=False, header=None)
    htr1a_valid_data = pd.read_csv(htr1a_valid_path, index_col=False, header=None)

    drd2 = predict_model(os.path.join(root_path, f'data/DLGN/SVM/best_drd2_svm.m'))
    htr1a = predict_model(os.path.join(root_path, f'data/DLGN/SVM/best_htr1a_svm.m'))
    model = SmilesGAILModel(config, vocab)
    generator_path = os.path.join(original_path, task_config.model_path + f'{model_name}')
    model.generator.load_model(generator_path, device)
    model.generator.train()
    reward_func = RewardFunction(vocab, model.discriminator, device)
    if task_config.use_wandb:
        wandb.init(project="multitarget", name='drd2_htr1a_aug_Da_Dlr1e5_Glr3e5_sf')
        wandb.log({'drd2_dataset_num': len(drd2_train_data), 'htr1a_dataset_num': len(htr1a_train_data)})
    trainer = GAILTrainerLoop(task_config, vocab, model=model,
                              reward_func=reward_func, device=device,
                              dataset1=drd2_train_data, dataset2=htr1a_train_data,
                              valid_dataset1=drd2_valid_data, valid_dataset2=htr1a_valid_data)
    trainer.train(os.path.join(output_path, task_config.save_path), drd2, htr1a)


if __name__ == "__main__":
    main()
