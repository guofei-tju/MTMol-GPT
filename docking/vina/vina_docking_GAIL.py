import os
import sys
import warnings

from tqdm import tqdm

sys.path.append("../../..")
import pandas as pd
from rdkit.Chem import AllChem, rdmolfiles
from rdkit import Chem, RDLogger


def canonical(smiles, sanitize=True, throw_warning=False):
    """
    Takes list of SMILES strings and returns list of their canonical SMILES.

    Parameters
    ----------
    smiles: list
        list of SMILES strings to convert into canonical format

    sanitize: bool (default True)
        parameter specifying whether to sanitize SMILES or not.
            For definition of sanitized SMILES check
            http://www.rdkit.org/docs/api/rdkit.Chem.rdmolops-module.html#SanitizeMol

    throw_warning: bool (default False)
        parameter specifying whether warnings will be thrown if a SMILES is
        invalid

    Returns
    -------
    new_smiles: list
        list of canonical SMILES and NaNs if SMILES string is invalid or
        unsanitized (when sanitize is True)

    When sanitize is True the function is analogous to:
        sanitize_smiles(smiles, canonical=True).
    """
    new_smiles = []
    valid_vec = []
    valid_sm = []
    for sm in smiles:
        RDLogger.DisableLog('rdApp.*')
        try:
            mol = Chem.MolFromSmiles(sm, sanitize=sanitize)
            new_smiles.append(Chem.MolToSmiles(mol))
            valid_vec.append(1)
            valid_sm.append(Chem.MolToSmiles(mol))
        except:
            if throw_warning:
                warnings.warn(sm + ' can not be canonized: invalid '
                                   'SMILES string!', UserWarning)
            new_smiles.append('')
            valid_vec.append(0)
    return new_smiles, valid_vec, valid_sm


path = '.'
itera = 190
file_name = 'gail'
receptor1 = '6LUQ_H'
receptor2 = '7E2Y_H'
# file_name = 'DLGN_generation_30000'
# file_name = 'pretrain_generation_num30000'

num = 1000
file_index = f'_e{itera}_{receptor1}_{receptor2}_sample{num}_size25'
smiles_path = os.path.join(path, 'ligand')
smiles = pd.read_csv(os.path.join(smiles_path, f'data/finetune_{file_name}_num30000_sf_e{itera}_fi.txt'), names=['smiles'], header=None)
new_smiles, valid_vec, valid_smiles = canonical(list(smiles['smiles']), sanitize=True,
                                                throw_warning=False)
samples = pd.DataFrame(valid_smiles, columns=['smiles']).sample(frac=1, ignore_index=True)
samples_path = os.path.join(smiles_path, f'{file_name}{file_index}')
if not os.path.exists(samples_path):
    os.makedirs(samples_path)

receptor_path = os.path.join(path, 'receptor')
if not os.path.isfile(os.path.join(receptor_path, receptor1 + '.pdbqt')):
    os.system(f"prepare_receptor -r {receptor_path}/{receptor1}.pdb -o {receptor_path}/{receptor1}.pdbqt")
if not os.path.isfile(os.path.join(receptor_path, receptor2 + '.pdbqt')):
    os.system(f"prepare_receptor -r {receptor_path}/{receptor2}.pdb -o {receptor_path}/{receptor2}.pdbqt")

results = []
# df = pd.read_csv(os.path.join(smiles_path, 'DRD2_active_6LUQ_H_sample1000_size25_affinity_0803.csv'))
# df_smiles = list(df.smiles)
for i in tqdm(range(len(samples))):
    smile = samples.smiles[i]
    # if smile in df_smiles:
    #     continue
    mol = Chem.AddHs(Chem.MolFromSmiles(smile))
    p = AllChem.EmbedMolecule(mol, randomSeed=1)
    if p == -1:
        continue
    print(smile)
    AllChem.MMFFOptimizeMolecule(mol)
    Chem.MolToMolFile(mol, f"{smiles_path}/{file_name}{file_index}/ligand_{i}.sdf")
    os.system(
        f'mk_prepare_ligand.py -i {smiles_path}/{file_name}{file_index}/ligand_{i}.sdf -o {smiles_path}/{file_name}{file_index}/ligand_{i}.pdbqt')
    os.system(
        f'vina --receptor {receptor_path}/{receptor1}.pdbqt --ligand {smiles_path}/{file_name}{file_index}/ligand_{i}.pdbqt --config {receptor_path}/{receptor1}_receptor_vina_box.txt --exhaustiveness 32 --seed 1 --out {smiles_path}/{file_name}{file_index}/ligand_{i}_{receptor1}_ad4_out.pdbqt >> {smiles_path}/{file_name}{file_index}/ligand_{i}_{receptor1}_log.txt')
    os.system(
        f'vina --receptor {receptor_path}/{receptor2}.pdbqt --ligand {smiles_path}/{file_name}{file_index}/ligand_{i}.pdbqt --config {receptor_path}/{receptor2}_receptor_vina_box.txt --exhaustiveness 32 --seed 1 --out {smiles_path}/{file_name}{file_index}/ligand_{i}_{receptor2}_ad4_out.pdbqt >> {smiles_path}/{file_name}{file_index}/ligand_{i}_{receptor2}_log.txt')

    log_file1 = open(f'{smiles_path}/{file_name}{file_index}/ligand_{i}_{receptor1}_log.txt', "r")
    data1 = log_file1.readlines()
    log_file1.close()

    log_file2 = open(f'{smiles_path}/{file_name}{file_index}/ligand_{i}_{receptor2}_log.txt', "r")
    data2 = log_file2.readlines()
    log_file2.close()

    for (j1, j2) in zip(data1, data2):
        if len(j1.split()) >= 2 and j1.split()[0] == "1":
            affinity1 = j1.split()[1]
            print(str(i) + f"**_{receptor1}_**" + affinity1)
        if len(j2.split()) >= 2 and j2.split()[0] == "1":
            affinity2 = j2.split()[1]
            print(str(i) + f"**_{receptor2}_**" + affinity2)
    results.append([i, smile, affinity1, affinity2])
    r = pd.DataFrame(results, columns=['index', 'smiles', f'affinity_{receptor1}-{file_name}',
                                       f'affinity_{receptor2}-{file_name}'])
    r.to_csv(f'{smiles_path}/{file_name}{file_index}/{file_name}{file_index}_affinity.csv', index=False)
    if len(results) == num:
        break
