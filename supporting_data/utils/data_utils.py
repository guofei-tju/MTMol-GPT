import pickle
import re

import joblib
import torch
import numpy as np
import warnings
import random
import pandas as pd
import csv

import time
import math


from rdkit.Chem import AllChem as Chem
import rdkit.Chem.QED as QED
import selfies as sf


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def smiles_to_tensor(smiles, device):
    smiles = list(smiles)
    _, valid_vec = canonical_smiles(smiles)
    valid_vec = torch.tensor(valid_vec).view(-1, 1).float().to(device)
    smiles, _ = pad_sequences(smiles)
    inp, _ = seq2tensor(smiles, tokens=get_default_tokens())
    inp = torch.from_numpy(inp).long().to(device)
    return inp, valid_vec


def seq_to_smiles(seqs, voc):
    """Takes an output sequence from the RNN and returns the
       corresponding SMILES."""
    smiles = []
    for seq in seqs.cpu().numpy():
        smiles.append(voc.decode(seq))
    return smiles


def read_smiles_from_file(filename, unique=True, add_start_end_tokens=False):
    """
    Reads SMILES from file. File must contain one SMILES string per line
    with \n token in the end of the line.

    Args:
        filename (str): path to the file
        unique (bool): return only unique SMILES

    Returns:
        smiles (list): list of SMILES strings from specified file.
        success (bool): defines whether operation was successfully completed or not.

    If 'unique=True' this list contains only unique copies.
    """
    f = open(filename, 'r')
    molecules = []
    for line in f:
        if add_start_end_tokens:
            molecules.append('<' + line[:-1] + '>')
        else:
            molecules.append(line[:-1])
    if unique:
        molecules = list(set(molecules))
    else:
        molecules = list(molecules)
    f.close()
    return molecules, f.closed




def seq2tensor2(seqs, tokens, flip=False):
    tensor = torch.zeros((len(seqs), len(seqs[0])))
    for i in range(len(seqs)):
        for j in range(len(seqs[i])):
            if seqs[i][j] in tokens:
                tensor[i, j] = tokens.index(seqs[i][j])
            else:
                tokens = tokens + [seqs[i][j]]
                tensor[i, j] = tokens.index(seqs[i][j])
    if flip:
        tensor = np.flip(tensor, axis=1).copy()
    return tensor, tokens


def seq2tensor(seqs, tokens):
    pad_to_len = max(sf.len_selfies(s) for s in seqs)
    symbol_to_idx = {s: i for i, s in enumerate(tokens)}
    tensor = torch.zeros((len(seqs), pad_to_len))
    for i, seq in enumerate(seqs):
        label, _ = sf.selfies_to_encoding(
            selfies=seq,
            vocab_stoi=symbol_to_idx,
            pad_to_len=pad_to_len,
            enc_type="both"
        )
        tensor[i, :] = label
    return tensor

def simple_moving_average(previous_values, new_value, ma_window_size=10):
    value_ma = np.sum(previous_values[-(ma_window_size - 1):]) + new_value
    value_ma = value_ma / (len(previous_values[-(ma_window_size - 1):]) + 1)
    return value_ma


# def init_stack(batch_size, stack_width, stack_depth, dvc='cpu'):
#     return torch.zeros(batch_size, stack_depth, stack_width).to(dvc)
#
#
# def init_hidden(num_layers, batch_size, hidden_size, num_dir=1, dvc='cpu'):
#     return torch.zeros(num_layers * num_dir, batch_size, hidden_size).to(dvc)
#
#
# def init_cell(num_layers, batch_size, hidden_size, num_dir=1, dvc='cpu'):
#     return init_hidden(num_layers, batch_size, hidden_size, num_dir, dvc)


def Variable(tensor, device_set):
    """Wrapper for torch.autograd.Variable that also accepts
       numpy arrays directly and automatically assigns it to
       the GPU. Be aware in case some operations are better
       left to the CPU."""
    device = torch.device(device_set if torch.cuda.is_available() else 'cpu')

    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).to(device)
    return torch.autograd.Variable(tensor)


def decrease_learning_rate(optimizer, decrease_by=0.01):
    """Multiplies the learning rate of the optimizer by 1 - decrease_by"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= (1 - decrease_by)


def fraction_valid_smiles(smiles):
    """Takes a list of SMILES and returns fraction valid."""
    i = 0
    for smile in smiles:
        if Chem.MolFromSmiles(smile):
            i += 1
    return i / len(smiles)


def unique(arr):
    # Finds unique rows in arr and return their indices
    arr = arr.cpu().numpy()
    arr_ = np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    _, idxs = np.unique(arr_, return_index=True)
    if torch.cuda.is_available():
        return torch.LongTensor(np.sort(idxs)).cuda()
    return torch.LongTensor(np.sort(idxs))


def canonical_smiles(smiles, sanitize=True, throw_warning=False):
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
    for sm in smiles:
        try:
            mol = Chem.MolFromSmiles(sm, sanitize=sanitize)
            new_smiles.append(Chem.MolToSmiles(mol))
            valid_vec.append(1)
        except:
            if throw_warning:
                warnings.warn(sm + ' can not be canonized: invalid '
                                   'SMILES string!', UserWarning)
            new_smiles.append('')
            valid_vec.append(0)
    return new_smiles, valid_vec


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


def pad_sequences(seqs, max_length=None, pad_symbol=' '):
    '''
    Pad and align the SMILES strings
    :param seqs: SMILES strings
    :param max_length: the Padding length of SMILES strings
    :param pad_symbol: Populate symbol
    :return:
    :seq: aligned SMILES strings
    :lengths: length of SMILES strings
    '''
    if max_length is None:
        max_length = -1
        for seq in seqs:
            max_length = max(max_length, len(seq))
    lengths = []
    for i in range(len(seqs)):
        cur_len = len(seqs[i])
        lengths.append(cur_len)
        seqs[i] = seqs[i] + pad_symbol * (max_length - cur_len)
    return seqs, lengths


def parse_optimizer(config, model):
    """
    Creates an optimizer for the given model using the argumentes specified in
    hparams.

    Arguments:
    -----------
    :param hparams: Hyperparameters config_torchfly
    :param model: An nn.Module object
    :return: a torch.optim object
    """
    # optimizer configuration
    optimizer = {
        "adadelta": torch.optim.Adadelta,
        "adagrad": torch.optim.Adagrad,
        "adam": torch.optim.Adam,
        "rmsprop": torch.optim.RMSprop,
        "adamax": torch.optim.Adamax,
        "rprop": torch.optim.Rprop,
        "sgd": torch.optim.SGD,
    }.get(config.optimizer.lower(), None)
    assert optimizer is not None, "{} optimizer could not be found"

    # filter optimizer arguments
    optim_kwargs = dict()
    optim_key = config.optimizer
    for k, v in config.items():
        if "optimizer__" in k:
            attribute_tup = k.split("__")
            if optim_key == attribute_tup[1] or attribute_tup[1] == "global":
                optim_kwargs[attribute_tup[2]] = v

    # optimizer = optimizer(model.parameters(), **optim_kwargs)

    optimizer = optimizer(filter(lambda p: p.requires_grad, model.parameters()), **optim_kwargs)
    return optimizer


def read_smi_file(filename, unique=True, add_start_end_tokens=False):
    """
    Reads SMILES from file. File must contain one SMILES string per line
    with \n token in the end of the line.

    Args:
        filename (str): path to the file
        unique (bool): return only unique SMILES

    Returns:
        smiles (list): list of SMILES strings from specified file.
        success (bool): defines whether operation was successfully completed or not.

    If 'unique=True' this list contains only unique copies.
    """
    f = open(filename, 'r')
    molecules = []
    for line in f:
        if add_start_end_tokens:
            molecules.append('<' + line[:-1] + '>')
        else:
            molecules.append(line[:-1])
    if unique:
        molecules = list(set(molecules))
    else:
        molecules = list(molecules)
    f.close()
    return molecules, f.closed


def tokenize(smiles, tokens=None):
    """
    Returns list of unique tokens, token-2-index dictionary and number of
    unique tokens from the list of SMILES

    Parameters
    ----------
        smiles: list
            list of SMILES strings to tokenize.

        tokens: list, str (default None)
            list of unique tokens

    Returns
    -------
        tokens: list
            list of unique tokens/SMILES alphabet.

        token2idx: dict
            dictionary mapping token to its index.

        num_tokens: int
            number of unique tokens.
    """
    if tokens is None:
        tokens = list(set(''.join(smiles)))
        tokens = list(np.sort(tokens))
        tokens = ''.join(tokens)
    token2idx = dict((token, i) for i, token in enumerate(tokens))
    num_tokens = len(tokens)
    return tokens, token2idx, num_tokens


def read_object_property_file(path, delimiter=',', cols_to_read=[0, 1],
                              keep_header=False, **kwargs):
    f = open(path, 'r', encoding='utf-8-sig')
    reader = csv.reader(f, delimiter=delimiter)
    data_full = np.array(list(reader))
    if keep_header:
        start_position = 0
    else:
        start_position = 1
    assert len(data_full) > start_position

    data = [[] for _ in range(len(cols_to_read))]
    for i in range(len(cols_to_read)):
        col = cols_to_read[i]
        data[i] = data_full[start_position:, col]
    f.close()

    if len(cols_to_read) == 1:
        data = data[0]
    return data


def text_save(filename, data):  # filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename, 'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("save the loss successfully!")


def get_default_tokens():
    """Default SMILES tokens"""
    tokens = [' ', '<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
              '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
              '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r']
    return tokens


class tokens_struct():
    def __init__(self):
        self.tokens = [' ', '<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
                       '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
                       '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r']

        self.tokens_length = len(self.tokens)

        self.tokens_vocab = dict(zip(self.tokens, range(len(self.tokens))))
        self.reversed_tokens_vocab = {v: k for k, v in self.tokens_vocab.items()}

    def get_default_tokens(self):
        """Default SMILES tokens"""
        return self.tokens

    def get_tokens_length(self):
        """Default SMILES tokens length"""
        return self.tokens_length

    def tokenize(self, smiles):
        """Takes a SMILES and return a list of characters/tokens"""
        regex = '(\[[^\[\]]{1,6}\])'
        smiles = replace_halogen(smiles)
        char_list = re.split(regex, smiles)
        tokenized = []
        for char in char_list:
            if char.startswith('['):
                tokenized.append(char)
            else:
                chars = [unit for unit in char]
                [tokenized.append(unit) for unit in chars]
        return tokenized

    def encode(self, char_list):
        """Takes a list of characters (eg '[NH]') and encodes to array of indices"""
        smiles_matrix = np.zeros(len(char_list), dtype=np.float32)
        for i, char in enumerate(char_list):
            smiles_matrix[i] = self.tokens_vocab[char]
        return smiles_matrix

    def decode(self, matrix):
        """Takes an array of indices and returns the corresponding SMILES"""
        chars = []
        for i in matrix:
            if i == self.tokens_vocab['>']: break
            if i == self.tokens_vocab['<']: continue
            chars.append(self.reversed_tokens_vocab[i])
        smiles = "".join(chars)
        # smiles = smiles.replace("L", "Cl").replace("R", "Br")
        return smiles

    def pad_sequence(self, sentence, sen_len=140, pad_index=0):
        # 将每个sentence变成一样的长度
        if len(sentence) > sen_len:
            sentence_tensor = torch.FloatTensor(sentence[:sen_len])
        else:
            sentence_tensor = torch.ones(sen_len) * pad_index
            sentence_tensor[:len(sentence)] = torch.FloatTensor(sentence)
        assert len(sentence_tensor) == sen_len
        return sentence_tensor


class tokens_struct2():
    def __init__(self):
        self.tokens = [' ', '<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
                       '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
                       '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r']

        self.tokens_length = len(self.tokens)
        self.tokens_vocab = dict(zip(self.tokens, range(len(self.tokens))))
        self.reversed_tokens_vocab = {v: k for k, v in self.tokens_vocab.items()}

    @property
    def bos(self):
        return self.tokens_vocab['<']

    @property
    def eos(self):
        return self.tokens_vocab['>']

    @property
    def pad(self):
        return self.tokens_vocab[' ']

    def get_default_tokens(self):
        """Default SMILES tokens"""
        return self.tokens

    def get_tokens_length(self):
        """Default SMILES tokens length"""
        return self.tokens_length

    def encode(self, char_list, add_bos=False, add_eos=False):
        """Takes a list of characters (eg '[NH]') and encodes to array of indices"""
        smiles_matrix = np.zeros(len(char_list), dtype=np.float32)
        for i, char in enumerate(char_list):
            smiles_matrix[i] = self.tokens_vocab[char]
        if add_bos:
            smiles_matrix = np.insert(smiles_matrix, 0, self.bos)
        if add_eos:
            smiles_matrix = np.append(smiles_matrix, self.eos)
        return smiles_matrix

    def decode(self, matrix):
        """Takes an array of indices and returns the corresponding SMILES"""
        chars = []
        for i in matrix:
            if i == self.tokens_vocab['>']: break
            if i == self.tokens_vocab['<']: continue
            chars.append(self.reversed_tokens_vocab[i])
        smiles = "".join(chars)
        return smiles

    def pad_sequence(self, sentence, sen_len=140, pad_index=0):
        # 将每个sentence变成一样的长度
        if len(sentence) > sen_len:
            sentence_tensor = torch.FloatTensor(sentence[:sen_len])
        else:
            sentence_tensor = torch.ones(sen_len) * pad_index
            sentence_tensor[:len(sentence)] = torch.FloatTensor(sentence)
        assert len(sentence_tensor) == sen_len
        return sentence_tensor


def replace_halogen(string):
    """Regex to replace Br and Cl with single letters"""
    br = re.compile('Br')
    cl = re.compile('Cl')
    string = br.sub('R', string)
    string = cl.sub('L', string)

    return string


class predict_model:
    """Scores based on an ECFP classifier for activity."""

    def __init__(self, path):
        with open(path, "rb") as f:
            self.clf = pickle.load(f)

    def __call__(self, smiles_list):
        from rdkit import Chem
        fps = []
        mask = []
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            mask.append(int(mol is not None))
            fp = self.fingerprints_from_mol(mol) if mol else np.zeros((1, 2048))
            fps.append(fp)

        fps = np.concatenate(fps, axis=0)
        scores = self.clf.predict_proba(fps)[:, 1]
        scores = scores * np.array(mask)
        return np.float32(scores)

    @classmethod
    def fingerprints_from_mol(cls, mol):  # use ECFP4
        from rdkit.Chem import DataStructs, AllChem
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features.reshape(1, -1)
