import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
import pickle
import hydra
import torch
import logging
import numpy as np
# import moses
import selfies as sf

from omegaconf import DictConfig, OmegaConf
from utils import set_random_seed, tokens_struct, selfies_tokens_struct
from model import Generator
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    config = OmegaConf.create(OmegaConf.to_yaml(cfg))
    output_path = os.getcwd()
    original_path = hydra.utils.get_original_cwd()
    root_path = '/'.join(original_path.split('/')[:-1])
    model_comfig = config.model.transformers
    task_config = config.generation.gail
    # set cuda
    if torch.cuda.is_available():
        dvc_id = 0
        device_set = f'cuda:{dvc_id}'
    else:
        device_set = 'cpu'

    set_random_seed(task_config.random_seed)
    if config.train.gail.use_selfies:
        vocab = selfies_tokens_struct()
    else:
        vocab = tokens_struct()

    save_path = os.path.join(original_path, task_config.save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    iteration = task_config.generate_num // task_config.batch_size
    model_root = os.path.join(original_path, task_config.model_path)
    for e in range(0, 1001, 5):
        generator = Generator(model_comfig, vocab, device=device_set)
        model_path = os.path.join(model_root, f'fine_tuning_generator_{e}.pt')
        generator.load_model(model_path, device_set)
        generator.model.eval()
        loop = tqdm(range(iteration))
        gen_samples = []

        with torch.no_grad():
            for i in loop:
                sequences = generator.generate(task_config.batch_size, task_config.max_length)
                if config.train.gail.use_selfies:
                    sm = []
                    for sequence in sequences:
                        sm.append(sf.decoder(sequence))
                    gen_samples += sm
                else:
                    gen_samples += sequences
        np.savetxt(save_path + f'finetune_gail_num{task_config.generate_num}_e{e}.txt', gen_samples, fmt='%s')


if __name__ == "__main__":
    main()
