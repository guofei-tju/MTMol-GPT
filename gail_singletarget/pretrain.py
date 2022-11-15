import os
import sys

sys.path.append("..")
import hydra
import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from moses import CharVocab

from model import Generator
from utils import set_random_seed, selfies_tokens_struct, tokens_struct

os.environ["CUDA_VISIBLE_DEVICES"] = '2'


def train(config, generator, vocab, dataloader, epochs, save_path):
    device = config.device
    generator.model.train()
    generator.model.to(device)

    min_loss = 10000
    epoch_mean_loss = 0
    best_model_path = os.path.join(save_path, f'saved_g/MLE_best_generator.pt')
    print('Generator pretraining begin')
    pad = vocab.pad
    min_losses = []
    for epoch in range(epochs):
        model_path = os.path.join(save_path, f'saved_g/MLE_generator_{epoch}.pt')
        epoch_losses = []
        loop = tqdm(enumerate(dataloader), total=len(dataloader))
        for step, batch in loop:
            batch_inp = {}
            batch_idx = [torch.tensor(vocab.encode(string, add_bos=True, add_eos=True), dtype=torch.long,
                                      device=device) for string in batch]
            batch_inp['index'] = pad_sequence([t[:-1] for t in batch_idx], batch_first=True, padding_value=pad)
            batch_inp['label'] = pad_sequence([t[1:] for t in batch_idx], batch_first=True, padding_value=pad)
            batch_inp['lens'] = torch.tensor([len(t) - 1 for t in batch_idx],
                                             dtype=torch.long, device=device)

            batch_inp['prop'] = []
            loss = generator.train_batch_smiles(batch_inp)
            epoch_losses.append(loss)
            loop.set_description(f'Epoch [{epoch + 1}/{epochs}]')
            epoch_mean_loss = np.mean(epoch_losses)
            loop.set_postfix(loss=loss, mean_loss=epoch_mean_loss)

        log_path = os.path.join(save_path, f'saved_g/MLE_epoch_{epoch}_loss.txt')
        np.savetxt(log_path, epoch_losses, fmt='%.4f')
        if epoch_mean_loss < min_loss:
            min_loss = epoch_mean_loss
            min_losses.append(min_loss)
            generator.save_model(best_model_path)
            print(f'Model saved    last loss:{min_loss}')
        generator.save_model(model_path)
        log_min_loss_path = os.path.join(save_path, f'saved_g/MLE_epoch_loss.txt')
        np.savetxt(log_min_loss_path, min_losses, fmt='%.4f')

    print('-------------------------generator pretraining finished')


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    config = OmegaConf.create(OmegaConf.to_yaml(cfg))
    original_path = hydra.utils.get_original_cwd()
    root_path = '/'.join(original_path.split('/')[:-1])
    task_config = config.train.pretrain
    save_path = os.path.join(original_path, task_config.save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # set cuda
    if torch.cuda.is_available():
        dvc_id = 0
        use_cuda = True
        device = f'cuda:{dvc_id}'
    else:
        device = 'cpu'
        use_cuda = None
    config['device'] = device
    set_random_seed(task_config.random_seed)

    if config.train.gail.use_selfies:
        gen_data_path = os.path.join(root_path, config.chembl_sf_data_path)
        train_data = pd.read_csv(gen_data_path)
        train_loader = DataLoader(list(train_data['selfies']), shuffle=True, pin_memory=True,
                                  batch_size=task_config.batch_size, drop_last=True)
        vocab = selfies_tokens_struct()
    else:
        gen_data_path = os.path.join(root_path, config.chembl_sm_data_path)
        train_data = pd.read_csv(gen_data_path)
        train_loader = DataLoader(list(train_data), shuffle=True, pin_memory=True,
                                  batch_size=task_config.batch_size, drop_last=True)
        vocab = tokens_struct()

    model_config = config.model.transformers
    generator = Generator(model_config, vocab, device=device)

    train(config, generator, vocab, train_loader, task_config.epochs, save_path)


if __name__ == "__main__":
    main()
