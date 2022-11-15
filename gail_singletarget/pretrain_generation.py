import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
import pickle
import hydra
import torch
import logging
import numpy as np
import pandas as pd
import moses
from omegaconf import DictConfig, OmegaConf
from utils import set_random_seed, tokens_struct
from model import Generator
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    config = OmegaConf.create(OmegaConf.to_yaml(cfg))
    output_path = os.getcwd()
    original_path = hydra.utils.get_original_cwd()
    root_path = '/'.join(original_path.split('/')[:-1])
    model_comfig = config.model.transformers
    task_config = config.generation.pretrain
    # set cuda
    if torch.cuda.is_available():
        dvc_id = 0
        device = f'cuda:{dvc_id}'
    else:
        device = 'cpu'
    set_random_seed(task_config.random_seed)

    drd2_data = pd.read_csv(os.path.join(root_path, config.target.drd2.test_path),
                            names=['smiles'],
                            header=None)
    htr1a_data = pd.read_csv(os.path.join(root_path, config.target.htr1a.test_path),
                             names=['smiles'],
                             header=None)
    psy_data = pd.read_csv(os.path.join(root_path, config.target.psy.path),
                           names=['smiles'],
                           header=None)

    vocab = tokens_struct()

    save_path = os.path.join(original_path, task_config.save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    iteration = task_config.generate_num // task_config.batch_size
    generator = Generator(model_comfig, vocab, device=device)
    generator_path = os.path.join(original_path, task_config.model_path + f'Prior_DLGN.ckpt')
    generator.load_model(generator_path, device)
    generator.model.eval()
    loop = tqdm(range(iteration))
    gen_samples = []
    with torch.no_grad():
        for i in loop:
            sequence = generator.generate(task_config.batch_size, task_config.max_length)
            gen_samples += sequence
    np.savetxt(save_path + f'/pretrain_generation_num{task_config.generate_num}.txt', gen_samples, fmt='%s')
    results = moses.get_all_metrics(gen_samples, test=list(drd2_data['smiles']),
                                    test_scaffolds=list(htr1a_data['smiles']), train=list(psy_data['smiles']),
                                    device=device)
    df = pd.DataFrame([results])
    df.to_csv(save_path + f'/moses_metrics.csv', index=False)


if __name__ == "__main__":
    main()
