"""
Guacamol: SMILES LSTM
https://github.com/BenevolentAI/guacamol_baselines
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Distribution

from models.smiles_char_dict import SmilesCharDictionary


def rnn_start_token_vector(batch_size, device):
    sd = SmilesCharDictionary()
    return torch.LongTensor(batch_size, 1).fill_(sd.begin_idx).to(device)


class SmilesRnn(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, n_layers, rnn_dropout
    ) -> None:
        """
            Basic RNN language model for SMILES

        Args:
            input_size: number of input symbols
            hidden_size: number of hidden units
            output_size: number of output symbols
            n_layers: number of hidden layers
            rnn_dropout: recurrent dropout
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.rnn_dropout = rnn_dropout

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)

        self.rnn = nn.LSTM(
            hidden_size,
            hidden_size,
            batch_first=True,
            num_layers=n_layers,
            dropout=rnn_dropout,
        )
        self.init_weights()

    def init_weights(self):
        # encoder / decoder
        nn.init.xavier_uniform_(self.encoder.weight)

        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.constant_(self.decoder.bias, 0)

        # RNN
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)
                # LSTM remember gate bias should be initialised to 1
                # https://github.com/pytorch/pytorch/issues/750
                r_gate = param[int(0.25 * len(param)) : int(0.5 * len(param))]
                nn.init.constant_(r_gate, 1)

    def forward(self, x, hidden):
        embeds = self.encoder(x)
        output, hidden = self.rnn(embeds, hidden)
        output = self.decoder(output)
        return output, hidden

    def init_hidden(self, bsz, device):
        # LSTM has two hidden states...
        return (
            torch.zeros(self.n_layers, bsz, self.hidden_size).to(device),
            torch.zeros(self.n_layers, bsz, self.hidden_size).to(device),
        )

    @property
    def config(self):
        return dict(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            n_layers=self.n_layers,
            rnn_dropout=self.rnn_dropout,
        )


class SmilesRnnActorCritic(nn.Module):
    def __init__(self, smiles_rnn) -> None:
        """
            Creates an Actor-Critic model from a Smiles RNN Language model

        Args:
            smiles_rnn: a SmilesRnn object
        """
        super().__init__()

        self.smiles_rnn = smiles_rnn
        self.critic_decoder = nn.Linear(self.smiles_rnn.hidden_size, 1)
        self.init_weights()

    def init_weights(self):
        # critic_decoder
        nn.init.xavier_uniform_(self.critic_decoder.weight)
        nn.init.constant_(self.critic_decoder.bias, 0)

    def forward(self, x, hidden):
        embeds = self.smiles_rnn.encoder(x)
        output, hidden = self.smiles_rnn.rnn(embeds, hidden)
        actor_output = self.smiles_rnn.decoder(output)
        critic_output = self.critic_decoder(output)
        return actor_output, critic_output, hidden


def load_model(model_definition, model_weights, device, copy_to_cpu=True):
    """

    Args:
        model_definition: path to model json
        model_weights: path to model weights
        device: cuda or cpu
        copy_to_cpu: bool

    Returns: an RNN model

    """
    json_in = open(model_definition).read()
    raw_dict = json.loads(json_in)
    model = SmilesRnn(**raw_dict)
    map_location = lambda storage, loc: storage if copy_to_cpu else None
    model.load_state_dict(torch.load(model_weights, map_location))
    return model.to(device)


def test_action(smiles_rnn, distribution_cls):
    max_batch_size = 64
    max_seq_length = 100
    num_samples = 64

    def _sample_batch(batch_size):
        actions = torch.zeros((batch_size, max_seq_length), dtype=torch.long).to(device)
        hidden = smiles_rnn.init_hidden(batch_size, device)
        inp = rnn_start_token_vector(batch_size, device)
        # print(actions.shape)
        for char in range(max_seq_length):
            output, hidden = smiles_rnn(inp, hidden)
            prob = F.softmax(output, dim=2)
            distribution = distribution_cls(probs=prob)
            action = distribution.sample()
            actions[:, char] = action.squeeze()
            inp = action
        # print(f"{output.shape}, hidden1={hidden[0].shape} hidden2={hidden[1].shape}")
        return actions

    number_batches = (num_samples + max_batch_size - 1) // max_batch_size
    remaining_samples = num_samples

    print(num_samples, max_seq_length)
    actions = torch.LongTensor(num_samples, max_seq_length).to(device)
    batch_start = 0
    for i in range(number_batches):
        batch_size = min(max_batch_size, remaining_samples)
        batch_end = batch_start + batch_size
        # print(f"start={batch_start}, end={batch_end}")
        actions[batch_start:batch_end, :] = _sample_batch(batch_size)
        batch_start += batch_size
        remaining_samples -= batch_size

    return actions


if __name__ == "__main__":
    model_weights = "./models/pretrained/model_final_0.473.pt"
    model_def = Path(model_weights).with_suffix(".json")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_def, model_weights=model_weights, device=device)

    sd = SmilesCharDictionary()
    actions = test_action(model, Categorical)
    smiles = sd.matrix_to_smiles(actions)
