import os

import moses
import numpy as np
import torch
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from matplotlib import pyplot as plt
import seaborn as sns
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger

os.environ["CUDA_VISIBLE_DEVICES"] = '2'


from utils import canonical_smiles
from utils import get_mol, logP, SA, QED, NP, weight, predict_model


def plot_prop(data1, data2, name, file_name, legends=None, lines=[]):
    ax = sns.kdeplot(data1, shade=True)
    sns.kdeplot(data2, shade=True, ax=ax)
    ax.set_xlabel(name)
    for line in lines:
        ax.axvline(line, color='r')
    if legends is not None:
        ax.legend(legends, loc="upper right")
    plt.savefig(file_name, dpi=400)
    plt.show()


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    config = OmegaConf.create(OmegaConf.to_yaml(cfg))
    output_path = os.getcwd()
    original_path = hydra.utils.get_original_cwd()
    root_path = '/'.join(original_path.split('/')[:-1])
    task_config = config.generation.gail
    save_path = os.path.join(original_path, task_config.save_path)
    if torch.cuda.is_available():
        dvc_id = 0
        device = f'cuda:{dvc_id}'
    else:
        device = 'cpu'
    config['device'] = device
    drd2_success = []
    htr1a_success = []
    both_success = []

    drd2_data = pd.read_csv(os.path.join(root_path, config.target.drd2.test_path),
                            names=['smiles'],
                            header=None)
    htr1a_data = pd.read_csv(os.path.join(root_path, config.target.htr1a.test_path),
                             names=['smiles'],
                             header=None)
    psy_data = pd.read_csv(os.path.join(root_path, config.target.psy.path),
                           names=['smiles'],
                           header=None)
    evl_list = [('drd2', predict_model(os.path.join(root_path, f'data/DLGN/SVM/best_drd2_svm.m'))),
                ('htr1a', predict_model(os.path.join(root_path, f'data/DLGN/SVM/best_htr1a_svm.m')))]
    df = pd.DataFrame()
    for epoch in range(0, 1001, 5):
        RDLogger.DisableLog('rdApp.*')
        results_path = save_path + f'/finetune_gail_num{task_config.generate_num}_e{epoch}.txt'
        if not os.path.isfile(results_path):
            continue
        che_data = pd.read_csv(results_path,
                               names=['smiles'],
                               header=None)

        new_smiles, valid_vec, valid_smiles = canonical_smiles(list(che_data['smiles']), sanitize=True,
                                                               throw_warning=False)

        samples = pd.DataFrame(new_smiles, columns=['smiles']).sample(10000, ignore_index=True)
        for name, func in evl_list:
            if name in ['drd2', 'htr1a']:
                samples[name] = func(list(samples['smiles']))
        samples['both'] = (samples['drd2'] >= 0.5) & (samples['htr1a'] >= 0.5)
        drd2_success.append((samples['drd2'] >= 0.5).sum() / 10000)
        htr1a_success.append((samples['htr1a'] >= 0.5).sum() / 10000)
        both_success.append(samples['both'].sum() / 10000)
        print(
            f'drd2:{drd2_success[-1]}, htr1a_success:{htr1a_success[-1]}, both_success:{both_success[-1]}')
        results = moses.get_all_metrics(list(che_data['smiles']), test=list(drd2_data['smiles']),
                                        test_scaffolds=list(htr1a_data['smiles']), train=list(psy_data['smiles']),
                                        device=device)
        results['drd2'] = drd2_success[-1]
        results['htr1a'] = htr1a_success[-1]
        results['dual'] = both_success[-1]
        results['epoch'] = epoch

        df = df.append([results])
        df.to_csv(save_path + f'moses_metrics.csv', index=False)
    col = df.columns.tolist()
    col = col[-1:] + col[:-1]
    df = df[col]
    df.to_csv(save_path + f'moses_metrics.csv', index=False)


if __name__ == "__main__":
    main()
