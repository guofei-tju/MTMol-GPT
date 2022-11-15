#!/usr/bin/env python
import argparse
import os

from multitarget_gail.utils import selfies_tokens_struct

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import hydra
import torch
from omegaconf import OmegaConf, DictConfig
from torch.optim import Adam

from toolkits.data import GeneratorData
from models.model_MCMG import transformer_RL
from toolkits.Optim import ScheduledOptim
from toolkits.utils import tokens_struct, text_save


@hydra.main(config_path="./", config_name="config_pretrain")
def train_prior(cfg: DictConfig):
    """Trains the Prior decodertf"""

    config = OmegaConf.create(OmegaConf.to_yaml(cfg))

    # change the current work dir
    os.chdir('../../../')

    # set cuda
    if torch.cuda.is_available():
        dvc_id = 0
        use_cuda = True
        device_set = f'cuda:{dvc_id}'
    else:
        device_set = 'cpu'
        use_cuda = None

    data_path = './data/DLGN/Chembl24_sf.txt'

    # Create a Dataset from a SMILES file
    if config.use_selfies:
        tokstr = selfies_tokens_struct()
    else:
        tokstr = tokens_struct()

    token_list = tokstr.get_default_tokens()

    gen_data = GeneratorData(training_data_path=data_path, tokens=token_list, delimiter='\t',
                             batch_size=128, max_len=config.AgentNet.max_seq_length, pad_symbol=token_list[0],
                             cols_to_read=[0], keep_header=True, use_cuda=use_cuda)

    Prior = transformer_RL(tokstr, config.AgentNet, device_set)

    optim = ScheduledOptim(Adam(Prior.decodertf.parameters(), betas=(0.9, 0.98), eps=1e-09),
                           config.AgentNet.d_model * 8, config.AgentNet.n_warmup_steps)

    losses = Prior.fit(gen_data, n_iterations=15000, optim=optim, save_prior_path=config.save_prior_path)
    text_save('./results/loss/GAIL_selfies_loss.txt', losses)

    torch.cuda.empty_cache()

    return losses


if __name__ == "__main__":
    train_prior()
