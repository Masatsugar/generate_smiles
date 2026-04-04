import numpy as np
import torch

from models.customs.chemutils import (
    Repeat,
    TimeDistributed,
    char2id,
    dotdict,
    hot_to_smiles,
    id2char,
    pad_smile,
    smiles_to_hot,
)


class TestDotdict:
    def test_attribute_access(self):
        d = dotdict({"a": 1, "b": 2})
        assert d.a == 1
        assert d.b == 2

    def test_attribute_set(self):
        d = dotdict()
        d.x = 42
        assert d["x"] == 42

    def test_attribute_delete(self):
        d = dotdict({"a": 1})
        del d.a
        assert "a" not in d


class TestPadSmile:
    def test_right_padding(self):
        result = pad_smile("CC", 5, "right")
        assert result == "CC   "
        assert len(result) == 5

    def test_left_padding(self):
        result = pad_smile("CC", 5, "left")
        assert result == "   CC"
        assert len(result) == 5

    def test_no_padding_needed(self):
        result = pad_smile("CCCCC", 5, "right")
        assert result == "CCCCC"

    def test_none_padding(self):
        result = pad_smile("CC", 5, "none")
        assert result == "CC"


class TestSmilesHotEncoding:
    def test_smiles_to_hot_shape(self):
        smiles = ["CC", "CCC"]
        hot = smiles_to_hot(smiles, max_len=10)
        assert hot.shape == (2, 10, len(char2id))

    def test_smiles_to_hot_dtype(self):
        hot = smiles_to_hot(["CC"], max_len=10)
        assert hot.dtype == np.float32

    def test_one_hot_sums_to_one(self):
        hot = smiles_to_hot(["CC"], max_len=5)
        # Each position should have exactly one 1 (the character or padding)
        for pos in range(5):
            assert hot[0, pos].sum() == 1.0

    def test_roundtrip(self):
        smiles = ["CC", "CCC"]
        hot = smiles_to_hot(smiles, max_len=10)
        recovered = hot_to_smiles(hot, id2char)
        assert recovered[0] == "CC"
        assert recovered[1] == "CCC"


class TestRepeat:
    def test_output_shape(self):
        repeat = Repeat(5)
        x = torch.randn(2, 16)
        out = repeat(x)
        assert out.shape == (2, 5, 16)

    def test_values_repeated(self):
        repeat = Repeat(3)
        x = torch.tensor([[1.0, 2.0]])
        out = repeat(x)
        for i in range(3):
            assert torch.equal(out[0, i], x[0])


class TestTimeDistributed:
    def test_output_shape(self):
        linear = torch.nn.Linear(10, 5)
        td = TimeDistributed(linear)
        x = torch.randn(2, 3, 10)  # batch=2, seq=3, features=10
        out = td(x)
        assert out.shape == (2, 3, 5)

    def test_2d_input_passthrough(self):
        linear = torch.nn.Linear(10, 5)
        td = TimeDistributed(linear)
        x = torch.randn(2, 10)
        out = td(x)
        assert out.shape == (2, 5)
