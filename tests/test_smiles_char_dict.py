import torch

from models.smiles_char_dict import SmilesCharDictionary


class TestSmilesCharDictionary:
    def setup_method(self):
        self.sd = SmilesCharDictionary()

    def test_char_count(self):
        assert self.sd.get_char_num() == 47

    def test_special_tokens(self):
        assert self.sd.PAD == " "
        assert self.sd.BEGIN == "Q"
        assert self.sd.END == "\n"

    def test_special_indices(self):
        assert self.sd.pad_idx == 0
        assert self.sd.begin_idx == 1
        assert self.sd.end_idx == 2

    def test_encode_multichar_tokens(self):
        assert self.sd.encode("Br") == "Y"
        assert self.sd.encode("Cl") == "X"
        assert self.sd.encode("Si") == "A"
        assert self.sd.encode("Se") == "Z"
        assert self.sd.encode("@@") == "R"

    def test_decode_multichar_tokens(self):
        assert self.sd.decode("Y") == "Br"
        assert self.sd.decode("X") == "Cl"
        assert self.sd.decode("A") == "Si"
        assert self.sd.decode("Z") == "Se"
        assert self.sd.decode("R") == "@@"

    def test_encode_decode_roundtrip(self):
        smiles = "c1ccc(Br)cc1"
        encoded = self.sd.encode(smiles)
        decoded = self.sd.decode(encoded)
        assert decoded == smiles

    def test_allowed_valid(self):
        assert self.sd.allowed("c1ccccc1") is True
        assert self.sd.allowed("CC(=O)O") is True

    def test_allowed_forbidden(self):
        assert self.sd.allowed("c1ccc(Au)cc1") is False

    def test_idx_char_bijection(self):
        for char, idx in self.sd.char_idx.items():
            assert self.sd.idx_char[idx] == char

    def test_matrix_to_smiles(self):
        # Encode "CC\n" as indices
        indices = [
            self.sd.char_idx["C"],
            self.sd.char_idx["C"],
            self.sd.char_idx[self.sd.END],
        ]
        tensor = torch.tensor([indices])
        result = self.sd.matrix_to_smiles(tensor)
        assert len(result) == 1
        assert result[0] == "CC"

    def test_matrix_to_smiles_with_encoded_tokens(self):
        # "Y" decodes to "Br"
        indices = [
            self.sd.char_idx["C"],
            self.sd.char_idx["Y"],
            self.sd.char_idx[self.sd.END],
        ]
        tensor = torch.tensor([indices])
        result = self.sd.matrix_to_smiles(tensor)
        assert result[0] == "CBr"
