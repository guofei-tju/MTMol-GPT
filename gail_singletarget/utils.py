import re
import torch
import numpy as np
import random
import warnings
import pickle
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from rdkit import RDLogger
import selfies as sf

RDLogger.DisableLog('rdApp.*')


class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Multiplies the learning rate defined in the optimizer by a dynamic variable determined by the current step.
        Linearly increases the multiplicative variable from 0. to 1. over `warmup_steps` training steps.
        Linearly decreases the multiplicative variable from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """

    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


def logP(mol):
    from rdkit.Chem import Crippen
    """
    Computes RDKit's logP
    """
    return Crippen.MolLogP(mol)


def SA(mol):
    """
    Computes RDKit's Synthetic Accessibility score
    """
    from moses.metrics.SA_Score import sascorer
    return sascorer.calculateScore(mol)


def NP(mol):
    """
    Computes RDKit's Natural Product-likeness score
    """
    from moses.metrics.NP_Score import npscorer
    return npscorer.scoreMol(mol)


def QED(mol):
    """
    Computes RDKit's QED score
    """
    from rdkit.Chem.QED import qed
    return qed(mol)


def weight(mol):
    """
    Computes molecular weight for given molecule.
    Returns float,
    """
    from rdkit.Chem import Descriptors

    return Descriptors.MolWt(mol)


def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    from rdkit import Chem
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol


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


class gsk3_model:
    """Scores based on an ECFP classifier for activity."""

    def __init__(self, path):
        with open(path, "rb") as f:
            self.clf = pickle.load(f)

    def __call__(self, smiles_list):
        from rdkit import Chem
        fps = []
        mask = []
        for i, smiles in enumerate(smiles_list):
            mol = smiles
            if isinstance(smiles, str):
                mol = Chem.MolFromSmiles(smiles)
                mask.append(int(mol is not None))
            fp = gsk3_model.fingerprints_from_mol(mol) if mol else np.zeros((1, 2048))
            fps.append(fp)

        fps = np.concatenate(fps, axis=0)
        scores = self.clf.predict_proba(fps)[:, 1]
        scores = scores * np.array(mask)
        return np.float32(scores)

    @classmethod
    def fingerprints_from_mol(cls, mol):  # use ECFP4
        from rdkit.Chem import DataStructs, AllChem
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features.reshape(1, -1)


class jnk3_model:
    """Scores based on an ECFP classifier for activity."""

    def __init__(self, path):
        with open(path, "rb") as f:
            self.clf = pickle.load(f)

    def __call__(self, smiles_list):
        from rdkit import Chem
        fps = []
        mask = []
        if not isinstance(smiles_list, list):
            smiles_list = [smiles_list]
        for i, smiles in enumerate(smiles_list):
            mol = smiles
            if isinstance(smiles, str):
                mol = Chem.MolFromSmiles(smiles)
            mask.append(int(mol is not None))
            fp = jnk3_model.fingerprints_from_mol(mol) if mol else np.zeros((1, 2048))
            fps.append(fp)

        fps = np.concatenate(fps, axis=0)
        scores = self.clf.predict_proba(fps)[:, 1]
        scores = scores * np.array(mask)
        return np.float32(scores)

    @classmethod
    def fingerprints_from_mol(cls, mol):  # use ECFP4
        from rdkit.Chem import DataStructs, AllChem
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features.reshape(1, -1)


def set_random_seed(seed=None):
    if seed is not None and not (isinstance(seed, int) and 0 <= seed):
        raise ValueError('Seed must be a non-negative integer or omitted, not {}'.format(seed))

    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def get_prop_random(prop, batchsize, max=None, min=None):
    if max is not None and min is not None:
        return torch.rand(batchsize) * (max - min) + min
    if prop.lower() == 'logp':  # [-10, 16]
        return torch.rand(batchsize) * 26 - 10
    elif prop.lower() == 'qed':  # [0,1]
        return torch.rand(batchsize)
    elif prop.lower() == 'sa':  # [1,10]
        return torch.rand(batchsize) * 9 + 1


def fingerprint(smiles, radius=2, nbits=2048):
    """
    Generates fingerprint for SMILES
    If smiles is invalid, returns None
    Returns fingerprint bits
    Parameters:
        smiles: SMILES string
    """
    from rdkit.Chem import AllChem
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol=mol, radius=radius, nBits=nbits)
    return fingerprint


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
    from rdkit.Chem import AllChem as Chem
    new_smiles = []
    valid_vec = []
    valid_smiles = []
    for sm in smiles:
        try:
            mol = Chem.MolFromSmiles(sm, sanitize=sanitize)
            new_smiles.append(Chem.MolToSmiles(mol))
            valid_smiles.append(sm)
            valid_vec.append(1)
        except:
            if throw_warning:
                warnings.warn(sm + ' can not be canonized: invalid '
                                   'SMILES string!', UserWarning)
    return new_smiles, valid_vec, valid_smiles


def Variable(tensor, device):
    """Wrapper for torch.autograd.Variable that also accepts
       numpy arrays directly and automatically assigns it to
       the GPU. Be aware in case some operations are better
       left to the CPU."""
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).to(device)
    return torch.autograd.Variable(tensor)


def lazy_collate(batch):
    return batch


def get_dataloader(file_path, batch_size):
    data, read_successful = read_smiles_from_file(file_path)
    assert read_successful
    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lazy_collate,
        drop_last=True
    )
    return dataloader


class tokens_struct():
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


class selfies_tokens_struct():
    def __init__(self):
        self.tokens = ['[nop]', '<', '>', '[#Branch1]', '[#Branch2]', '[#C-1]', '[#C]', '[#N+1]', '[#N]', '[=Branch1]',
                       '[=Branch2]', '[=C-1]', '[=C]', '[=N+1]', '[=N-1]', '[=N]', '[=O+1]', '[=O]', '[=Ring1]',
                       '[=Ring2]', '[=S+1]', '[=SH1]', '[=S]', '[B-1]', '[B]', '[Br+1]', '[Br]', '[Branch1]',
                       '[Branch2]', '[C-1]', '[CH0]', '[CH1+1]', '[CH2+1]', '[CH2-1]', '[C]', '[Cl+1]', '[Cl]', '[F+1]',
                       '[F]', '[H]', '[I]', '[N+1]', '[N-1]', '[NH1]', '[N]', '[O+1]', '[O-1]', '[OH0]', '[O]', '[P]',
                       '[Ring1]', '[Ring2]', '[S+1]', '[SH1]', '[S]']

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
        return self.tokens_vocab['[nop]']

    def get_default_tokens(self):
        """Default SMILES tokens"""
        return self.tokens

    def get_tokens_length(self):
        """Default SMILES tokens length"""
        return self.tokens_length

    def encode(self, char_list, add_bos=False, add_eos=False):
        """Takes a list of characters (eg '[NH]') and encodes to array of indices"""
        label, _ = sf.selfies_to_encoding(
            selfies=char_list,
            vocab_stoi=self.tokens_vocab,
            enc_type="both"
        )
        smiles_matrix = np.array(label, dtype=np.float32)
        if add_bos:
            smiles_matrix = np.insert(smiles_matrix, 0, self.bos)
        if add_eos:
            smiles_matrix = np.append(smiles_matrix, self.eos)
        return smiles_matrix

    def seq2tensor(self, seqs):
        pad_to_len = max(sf.len_selfies(s) for s in seqs)
        tensor = torch.zeros((len(seqs), pad_to_len))
        for i, seq in enumerate(seqs):
            label, _ = sf.selfies_to_encoding(
                selfies=seq,
                vocab_stoi=self.tokens_vocab,
                pad_to_len=pad_to_len,
                enc_type="both"
            )
            tensor[i, :] = label
        return tensor

    def seq2list(self, seqs):
        results = []
        for i, seq in enumerate(seqs):
            label, _ = sf.selfies_to_encoding(
                selfies=seq,
                vocab_stoi=self.tokens_vocab,
                enc_type="both"
            )
            results.append(label)
        return results

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

    def convertSmiles(self, smiles):
        encoded_selfies = []
        for i in smiles:
            encoded_selfies.append(sf.encoder(i))
        return encoded_selfies


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


def parse_optimizer(hparams, model):
    """
    Creates an optimizer for the given model using the argumentes specified in
    hparams.

    Arguments:
    -----------
    :param hparams: Hyperparameters dictionary
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
    }.get(hparams["optimizer"].lower(), None)
    assert optimizer is not None, "{} optimizer could not be found"

    # filter optimizer arguments
    optim_kwargs = dict()
    optim_key = hparams["optimizer"]
    for k, v in hparams.items():
        if "optimizer__" in k:
            attribute_tup = k.split("__")
            if optim_key == attribute_tup[1] or attribute_tup[1] == "global":
                optim_kwargs[attribute_tup[2]] = v

    # optimizer = optimizer(model.parameters(), **optim_kwargs)

    optimizer = optimizer(filter(lambda p: p.requires_grad, model.parameters()), **optim_kwargs)
    return optimizer
    tokens = [' ', '<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
              '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
              '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n']


def Smiles2Selfies():
    import selfies as sf
    '''Translation    between    SELFIES and SMILES    representations:'''
    benzene = "c1ccccc1"
    # SMILES -> SELFIES -> SMILES translation
    try:
        benzene_sf = sf.encoder(benzene)  # [C][=C][C][=C][C][=C][Ring1][=Branch1]
        benzene_smi = sf.decoder(benzene_sf)  # C1=CC=CC=C1
    except sf.EncoderError:
        pass  # sf.encoder error!
    except sf.DecoderError:
        pass  # sf.decoder error!

    len_benzene = sf.len_selfies(benzene_sf)  # 8

    symbols_benzene = list(sf.split_selfies(benzene_sf))
    # ['[C]', '[=C]', '[C]', '[=C]', '[C]', '[=C]', '[Ring1]', '[=Branch1]']

    '''Integer and one-hot encoding SELFIES'''
    dataset = ["[C][O][C]", "[F][C][F]", "[O][=O]", "[C][C][O][C][C]"]
    alphabet = sf.get_alphabet_from_selfies(dataset)
    alphabet.add("[nop]")  # [nop] is a special padding symbol
    alphabet = list(sorted(alphabet))  # ['[=O]', '[C]', '[F]', '[O]', '[nop]']

    pad_to_len = max(sf.len_selfies(s) for s in dataset)  # 5
    symbol_to_idx = {s: i for i, s in enumerate(alphabet)}

    dimethyl_ether = dataset[0]  # [C][O][C]

    label, one_hot = sf.selfies_to_encoding(
        selfies=dimethyl_ether,
        vocab_stoi=symbol_to_idx,
        pad_to_len=pad_to_len,
        enc_type="both"
    )
    # label = [1, 3, 1, 4, 4]
    # one_hot = [[0, 1, 0, 0, 0], [0, 0, 0, 1, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1]]
