import pandas as pd
import selfies as sf

# read the whole dataset
from rdkit import Chem

pd_dataset_smi = pd.read_csv('./data/DLGN/DRD2_train.txt', names=['smiles'], header=None)

# encoding
none_list = []
for i in range(pd_dataset_smi.shape[0]):
    if Chem.MolFromSmiles(pd_dataset_smi['smiles'][i]) is None:
        none_list.append(i)

pd_dataset_smi = pd_dataset_smi.drop(none_list)
smiles_list = pd_dataset_smi['smiles'].tolist()
encoded_selfies = []
for i in smiles_list:
    encoded_selfies.append(sf.encoder(i))

# create alphabet
alphabet = sf.get_alphabet_from_selfies(encoded_selfies)
alphabet = list(sorted(alphabet))
pad_to_len = max(sf.len_selfies(s) for s in encoded_selfies)
symbol_to_idx = {s: i for i, s in enumerate(alphabet)}

# save
r = pd.DataFrame(encoded_selfies, columns=['selfies'])
r.to_csv('./data/DLGN/DRD2_train_sf.txt', index=False, header=None)
