import os

import numpy as np
import pandas as pd
import seaborn as sns

from tqdm import tqdm
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, QED
from rdkit.Chem import Draw, Crippen

from utils import sascorer
from utils.data_utils import canonical
from operator import itemgetter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def drawing_Mol(SMILES, save_path):
    invalid_num = 0
    for i, smiles in tqdm(enumerate(SMILES)):
        mol = Chem.MolFromSmiles(smiles)
        # drawing
        if mol != None:
            # 将SMILES生成图片
            Draw.MolToFile(mol, save_path + 'smile%d.png' % i, size=(600, 600))
            # 将SMILES格式转换成SDF格式
            w = Chem.SDWriter(save_path + 'smile%d.sdf' % i)
            # 加上坐标
            AllChem.Compute2DCoords(mol)
            w.write(mol)
            w.flush()
        else:
            invalid_num += 1

    valid_percent = 1 - invalid_num / (i + 1)

    print("valid is %f" % valid_percent)


def verify_sequence(smile):
    mol = Chem.MolFromSmiles(smile)
    return smile != '' and mol is not None and mol.GetNumAtoms() > 1


def Estimate_logP(smiles_list):
    logPs = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logPs.append(0)
        else:
            try:
                logPs.append(Chem.Crippen.MolLogP(mol))
            except:
                logPs.append(0)
    return logPs


def Estimate_QED(smiles_list):
    QEDS = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            QEDS.append(0)
        else:
            try:
                QEDS.append(QED.qed(mol))
            except:
                QEDS.append(0)
    return QEDS


def Estimate_SA(smiles_list):
    SAs = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            SAs.append(0)
        else:
            SAs.append(sascorer.calculateScore(mol))
    return SAs


def plot_hist(prop, prediction, save_path, color, iter, title_flag=0):
    prediction = np.array(prediction)
    # percentage_in_threshold = np.sum((prediction >= 1.0) &
    #                                  (prediction <= 4.0)) / len(prediction)
    # print("Percentage of predictions within drug-like region:", percentage_in_threshold)
    if title_flag == 0:
        save_flag = 'tra'
        title_name = f'Distribution of predicted {prop} for generated molecules'
    elif title_flag == 1:
        save_flag = 'pre'
        title_name = f'Distribution of predicted {prop} for pretrained molecules'
    ax = sns.kdeplot(prediction, shade=True, color=color, label=f'{iter}')
    # ax = sns.kdeplot(prediction, hue="size", label=f'{iter}', fill=True, palette="crest", alpha=0.5)
    # ax.set_xlim(0, 1)
    ax.set(xlabel=f'{prop}',
           title=f'')
    ax.set_xlabel(xlabel=f'{prop}', fontsize=14)
    ax.set_ylabel(ylabel='Density', fontsize=14)
    ax.legend(loc="upper left", fontsize=12)

    # fig format
    plt.savefig(save_path + f'Distribution{prop} at {iter}_{save_flag}.jpg')
    # pdf format
    plt.savefig(save_path + f'Distribution{prop} at {iter}_{save_flag}.pdf', format='pdf')
    # plt.clf()
    return 0


target1 = 'DRD2'
target2 = 'HTR1A'

# age_colors = sns.color_palette("bright",10)
age_colors = (sns.color_palette("pastel", 9))
# plot the generation SMILES
# file_name = f'finetune_drd2_num30000_e{iter}'
# SMILES = pd.read_csv(smiles_path + file_name + '.txt', names=['smiles'],
#                      header=None)
# _, _, sanitized = canonical(SMILES.values.reshape(-1), sanitize=True, throw_warning=False)
# unique_smiles = list(np.unique(sanitized))
#
# # random
# indices = np.arange(len(unique_smiles))
# np.random.shuffle(indices)

# plot the CHEMBL24 dataset
Chembl24_path = 'data/DLGN/Chembl24.txt'
Chembl24_smiles = pd.read_csv(Chembl24_path, header=None)
_, _, Chembl24_sanitized = canonical(Chembl24_smiles.values.reshape(-1), sanitize=True, throw_warning=False)
Chembl24_unique_smiles = list(np.unique(Chembl24_sanitized))

# plot the target specific SMILES
target1_path = f'data/DLGN/{target1.upper()}_test.txt'
target1_smiles = pd.read_csv(target1_path, header=None)
_, _, target1_sanitized = canonical(target1_smiles.values.reshape(-1), sanitize=True, throw_warning=False)
target1_unique_smiles = list(np.unique(target1_sanitized))

target2_path = f'data/DLGN/{target2.upper()}_test.txt'
target2_smiles = pd.read_csv(target2_path, header=None)
_, _, target2_sanitized = canonical(target2_smiles.values.reshape(-1), sanitize=True, throw_warning=False)
target2_unique_smiles = list(np.unique(target2_sanitized))

sm_GAIL_name = 'MolGAIL'
sf_GAIL_name = 'SF-MolGAIL'

sm_iter = 30
sm_file_name = f'finetune_gail_num30000_e{sm_iter}'
sm_smiles_path = 'data/MTMol-GPT/'
sm_save_path = 'data/MTMol-GPT/'

sf_iter = 190
sf_file_name = f'finetune_gail_num30000_e{sf_iter}'
sf_smiles_path = 'data/SF-MTMol-GPT/'
sf_save_path = 'data/SF-MTMol-GPT/'

# DLGN
DLGN_name = './dlgn_generation_num_30000.txt'
if not os.path.exists(sm_save_path):
    os.makedirs(sm_save_path)

if not os.path.exists(sf_save_path):
    os.makedirs(sf_save_path)

np.random.seed(0)
print('processing smiles...')
## sm
# trained smiles
SMILES = pd.read_csv(sm_smiles_path + sm_file_name + '.txt', names=['smiles'], header=None)
_, _, sanitized = canonical(SMILES.values.reshape(-1), sanitize=True, throw_warning=False)
unique_sm_smiles = list(np.unique(sanitized))
unique_sm_indices = np.arange(len(unique_sm_smiles))
np.random.shuffle(unique_sm_indices)
unique_sm_smiles = itemgetter(*unique_sm_indices[:10000])(unique_sm_smiles)

## sf
# trained selfies
SMILES = pd.read_csv(sf_smiles_path + sf_file_name + '.txt', names=['smiles'],
                     header=None)
_, _, sanitized = canonical(SMILES.values.reshape(-1), sanitize=True, throw_warning=False)
unique_sf_smiles = list(np.unique(sanitized))
unique_sf_indices = np.arange(len(unique_sf_smiles))
np.random.shuffle(unique_sf_indices)
unique_sf_smiles = itemgetter(*unique_sf_indices[:10000])(unique_sf_smiles)

prop = 'logP'
# with Pool(processes=16) as pool:
#     Chembl24_prop = list(tqdm(pool.imap(Estimate_logP, Chembl24_unique_smiles), total=len(Chembl24_unique_smiles)))
# with Pool(processes=16) as pool:
#     pre_sm_SMILES_prop = list(tqdm(pool.imap(Estimate_logP, unique_pre_sm_smiles), total=len(unique_pre_sm_smiles)))
# with Pool(processes=16) as pool:
#     pre_sf_SMILES_prop = list(tqdm(pool.imap(Estimate_logP, unique_pre_sf_smiles), total=len(unique_pre_sf_smiles)))
# with Pool(processes=16) as pool:
#     sm_SMILES_prop = list(tqdm(pool.imap(Estimate_logP, unique_sm_smiles), total=len(unique_sm_smiles)))
# with Pool(processes=16) as pool:
#     sf_SMILES_prop = list(tqdm(pool.imap(Estimate_logP, unique_sf_smiles), total=len(unique_sf_smiles)))
# with Pool(processes=16) as pool:
#     unique_DLGN_smiles = list(tqdm(pool.imap(Estimate_logP, unique_DLGN_smiles), total=len(unique_DLGN_smiles)))

Chembl24_prop = Estimate_logP(Chembl24_unique_smiles)
# pre_sm_SMILES_prop = Estimate_logP(unique_pre_sm_smiles)
# pre_sf_SMILES_prop = Estimate_logP(unique_pre_sf_smiles)
target1_prop = Estimate_logP(target1_unique_smiles)
target2_prop = Estimate_logP(target2_unique_smiles)
sm_SMILES_prop = Estimate_logP(unique_sm_smiles)
sf_SMILES_prop = Estimate_logP(unique_sf_smiles)

plt.clf()
plot_hist(prop, Chembl24_prop, sm_save_path, color=age_colors[4], iter='Chembl24')
# plot_hist(prop, pre_sm_SMILES_prop, sm_save_path, color=age_colors[2], iter=f'{sm_GAIL_name}', title_flag=1)
# plot_hist(prop, pre_sf_SMILES_prop, sm_save_path, color=age_colors[3], iter=f'{sf_GAIL_name}', title_flag=1)
# plot_hist(prop, unique_DLGN_smiles, sm_save_path, color=age_colors[3], iter=f'{sf_GAIL_name}', title_flag=1)

plt.clf()
plot_hist(prop, target1_prop, sm_save_path, color=age_colors[0], iter=target1)
plot_hist(prop, target2_prop, sm_save_path, color=age_colors[1], iter=target2)
plot_hist(prop, sm_SMILES_prop, sm_save_path, color=age_colors[2], iter=f'{sm_GAIL_name}')
plot_hist(prop, sf_SMILES_prop, sm_save_path, color=age_colors[3], iter=f'{sf_GAIL_name}')
print('LogP calculation finished...')

prop = 'QED'
Chembl24_prop = Estimate_QED(Chembl24_unique_smiles)
# pre_sm_SMILES_prop = Estimate_QED(unique_pre_sm_smiles)
# pre_sf_SMILES_prop = Estimate_QED(unique_pre_sf_smiles)

target1_prop = Estimate_QED(target1_unique_smiles)
target2_prop = Estimate_QED(target2_unique_smiles)
sm_SMILES_prop = Estimate_QED(unique_sm_smiles)
sf_SMILES_prop = Estimate_QED(unique_sf_smiles)

plt.clf()
plot_hist(prop, Chembl24_prop, sm_save_path, color=age_colors[4], iter='Chembl24')
# plot_hist(prop, pre_sm_SMILES_prop, sm_save_path, color=age_colors[2], iter=f'{sm_GAIL_name}', title_flag=1)
# plot_hist(prop, pre_sf_SMILES_prop, sm_save_path, color=age_colors[3], iter=f'{sf_GAIL_name}', title_flag=1)

plt.clf()
plot_hist(prop, target1_prop, sm_save_path, color=age_colors[0], iter=target1)
plot_hist(prop, target2_prop, sm_save_path, color=age_colors[1], iter=target2)
plot_hist(prop, sm_SMILES_prop, sm_save_path, color=age_colors[2], iter=f'{sm_GAIL_name}')
plot_hist(prop, sf_SMILES_prop, sm_save_path, color=age_colors[3], iter=f'{sf_GAIL_name}')
print('QED calculation finished...')

prop = 'SA'
Chembl24_prop = Estimate_SA(Chembl24_unique_smiles)
# pre_sm_SMILES_prop = Estimate_SA(unique_pre_sm_smiles)
# pre_sf_SMILES_prop = Estimate_SA(unique_pre_sf_smiles)

target1_prop = Estimate_SA(target1_unique_smiles)
target2_prop = Estimate_SA(target2_unique_smiles)
sm_SMILES_prop = Estimate_SA(unique_sm_smiles)
sf_SMILES_prop = Estimate_SA(unique_sf_smiles)

plt.clf()
plot_hist(prop, Chembl24_prop, sm_save_path, color=age_colors[4], iter='Chembl24')
# plot_hist(prop, pre_sm_SMILES_prop, sm_save_path, color=age_colors[2], iter=f'{sm_GAIL_name}', title_flag=1)
# plot_hist(prop, pre_sf_SMILES_prop, sm_save_path, color=age_colors[3], iter=f'{sf_GAIL_name}', title_flag=1)

plt.clf()
plot_hist(prop, target1_prop, sm_save_path, color=age_colors[0], iter=target1)
plot_hist(prop, target2_prop, sm_save_path, color=age_colors[1], iter=target2)
plot_hist(prop, sm_SMILES_prop, sm_save_path, color=age_colors[2], iter=f'{sm_GAIL_name}')
plot_hist(prop, sf_SMILES_prop, sm_save_path, color=age_colors[3], iter=f'{sf_GAIL_name}')
print('SA calculation finished...')
