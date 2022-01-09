import re
from typing import Dict

import numpy as np
import torch.nn as nn

zinc_list = [
    "7",
    "6",
    "o",
    "]",
    "3",
    "s",
    "(",
    "-",
    "S",
    "/",
    "B",
    "4",
    "[",
    ")",
    "#",
    "I",
    "l",
    "O",
    "H",
    "c",
    "1",
    "@",
    "=",
    "n",
    "P",
    "8",
    "C",
    "2",
    "F",
    "5",
    "r",
    "N",
    "+",
    "\\",
    " ",
]
char2id = dict((c, i) for i, c in enumerate(zinc_list))
id2char = dict((i, c) for i, c in enumerate(zinc_list))


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def pad_smile(string: str, max_len: int, padding: str = "right") -> str:
    if len(string) <= max_len:
        if padding == "right":
            return string + " " * (max_len - len(string))
        elif padding == "left":
            return " " * (max_len - len(string)) + string
        elif padding == "none":
            return string


def smiles_to_hot(
    smiles: list,
    max_len: int = 120,
    padding: str = "right",
    char_to_id: Dict[str, int] = char2id,
) -> np.ndarray:
    """smiles list into one-hot tensors.

    :param smiles: SMILES list
    :param max_len: max length of the number of atoms
    :param padding: types of padding: ('right' or 'left' or None)
    :param char_indices: dictionary of SMILES characters
    :return: one-hot matrix (Batch × MAX_LEN × len(dict))
    """
    smiles = [pad_smile(smi, max_len, padding) for smi in smiles]
    hot_x = np.zeros((len(smiles), max_len, len(char_to_id)), dtype=np.float32)
    for list_id, smile in enumerate(smiles):
        for char_id, char in enumerate(smile):
            try:
                hot_x[list_id, char_id, char_to_id[char]] = 1
            except KeyError as e:
                print("ERROR: Check chars file. Bad SMILES:", smile)
                raise e
    return hot_x


def smiles_to_hot2(
    smiles: str, max_len: int = 120, char_to_id: Dict[str, int] = char2id
) -> np.ndarray:
    hot_x = np.zeros((max_len, len(char_to_id)), dtype=np.float32)
    for node_id, char in enumerate(smiles):
        hot_x[node_id, char_to_id[char]] = 1
    return hot_x


def hot_to_smiles(hot_x: np.ndarray, id2char: Dict[int, str] = id2char) -> list:
    """one hot list to SMILES list.

    :param hot_x: smiles one hot (id, max_len, node_dict)
    :param id2char: map from node id to smiles char
    :return: smiles list
    """
    smiles = ["".join([id2char[np.argmax(j)] for j in x]) for x in hot_x]
    smiles = [re.sub(" ", "", smi) for smi in smiles]  # paddingを消す
    return smiles


def smiles2one_hot_chars(smi_list: list) -> list:
    """obtain character in SMILES.
    :param smi_list: SMILES list
    :return: Char list
    """
    char_lists = [list(smi) for smi in smi_list]
    chars = list(set([char for sub_list in char_lists for char in sub_list]))
    chars.append(" ")

    return chars


class Repeat(nn.Module):
    def __init__(self, rep):
        super(Repeat, self).__init__()

        self.rep = rep

    def forward(self, x):
        size = tuple(x.size())
        size = (size[0], 1) + size[1:]
        x_expanded = x.view(*size)
        n = [1 for _ in size]
        n[1] = self.rep
        return x_expanded.repeat(*n)


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        # (samples * timesteps, input_size)
        x_reshape = x.contiguous().view(-1, x.size(-1))
        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            # (samples, timesteps, output_size)
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        else:
            # (timesteps, samples, output_size)
            y = y.view(-1, x.size(1), y.size(-1))

        return y


if __name__ == "__main__":
    import pandas as pd

    def test_recon():
        df = pd.read_table("./data/train.txt", header=None).iloc[0:100]
        df.columns = ["smiles"]

        hot_x = smiles_to_hot(df.smiles)
        s = hot_to_smiles(hot_x, id2char)
        print(s)

    test_recon()
