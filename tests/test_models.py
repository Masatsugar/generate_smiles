import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from models.customs.chem_gan import Discriminator, Generator
from models.customs.chem_vae import ChemVAE, Decoder, Encoder, loss_function
from models.rnn_model import SmilesRnn, SmilesRnnActorCritic, rnn_start_token_vector


class TestSmilesRnn:
    def setup_method(self):
        self.model = SmilesRnn(
            input_size=47, hidden_size=64, output_size=47, n_layers=2, rnn_dropout=0.2
        )
        self.device = "cpu"

    def test_init_hidden(self):
        h, c = self.model.init_hidden(4, self.device)
        assert h.shape == (2, 4, 64)
        assert c.shape == (2, 4, 64)

    def test_forward(self):
        batch_size = 4
        seq_len = 10
        x = torch.randint(0, 47, (batch_size, seq_len))
        hidden = self.model.init_hidden(batch_size, self.device)
        output, new_hidden = self.model(x, hidden)
        assert output.shape == (batch_size, seq_len, 47)
        assert new_hidden[0].shape == (2, batch_size, 64)

    def test_config(self):
        cfg = self.model.config
        assert cfg["input_size"] == 47
        assert cfg["hidden_size"] == 64
        assert cfg["output_size"] == 47
        assert cfg["n_layers"] == 2

    def test_sampling(self):
        batch_size = 2
        seq_len = 5
        hidden = self.model.init_hidden(batch_size, self.device)
        inp = rnn_start_token_vector(batch_size, self.device)
        actions = []
        for _ in range(seq_len):
            output, hidden = self.model(inp, hidden)
            prob = F.softmax(output, dim=2)
            dist = Categorical(probs=prob)
            action = dist.sample()
            actions.append(action.squeeze())
            inp = action
        assert len(actions) == seq_len


class TestSmilesRnnActorCritic:
    def test_forward(self):
        rnn = SmilesRnn(input_size=47, hidden_size=64, output_size=47, n_layers=2, rnn_dropout=0.2)
        model = SmilesRnnActorCritic(rnn)
        batch_size = 4
        seq_len = 10
        x = torch.randint(0, 47, (batch_size, seq_len))
        hidden = rnn.init_hidden(batch_size, "cpu")
        actor_out, critic_out, new_hidden = model(x, hidden)
        assert actor_out.shape == (batch_size, seq_len, 47)
        assert critic_out.shape == (batch_size, seq_len, 1)


class TestEncoder:
    def test_forward(self):
        encoder = Encoder(input=120, hidden_dim=64, c=35)
        x = torch.randn(2, 120, 35)
        mu, logvar = encoder(x)
        assert mu.shape == (2, 64)
        assert logvar.shape == (2, 64)


class TestDecoder:
    def test_forward(self):
        decoder = Decoder(encode_dim=64, o=120, char=35)
        z = torch.randn(2, 64)
        out = decoder(z)
        assert out.shape == (2, 120, 35)


class TestChemVAE:
    def setup_method(self):
        self.vae = ChemVAE(input_dim=120, hidden_dim=64, max_length=120, dict_dim=35)

    def test_forward(self):
        x = torch.randn(2, 120, 35)
        recon, mu, logvar = self.vae(x)
        assert recon.shape == (2, 120, 35)
        assert mu.shape == (2, 64)
        assert logvar.shape == (2, 64)

    def test_reparameterize(self):
        mu = torch.zeros(2, 64)
        logvar = torch.zeros(2, 64)
        z = self.vae.reparameterize(mu, logvar)
        assert z.shape == (2, 64)

    def test_loss_function(self):
        x = torch.rand(2, 120, 35)
        # Normalize to make it a valid target for BCE
        x = x / x.sum(dim=-1, keepdim=True)
        recon_x = torch.rand(2, 120, 35)
        recon_x = recon_x / recon_x.sum(dim=-1, keepdim=True)
        mu = torch.randn(2, 64)
        logvar = torch.randn(2, 64)
        loss = loss_function(recon_x, x, mu, logvar)
        assert loss.item() > 0


class TestGenerator:
    def test_forward(self):
        gen = Generator(latent_dim=100, o=120, char=35)
        z = torch.randn(2, 100)
        out = gen(z)
        assert out.shape == (2, 120, 35)


class TestDiscriminator:
    def test_forward(self):
        disc = Discriminator(input_img=120, char=35)
        x = torch.randn(2, 120, 35)
        out = disc(x)
        assert out.shape == (2, 1)
        # Output should be between 0 and 1 (sigmoid)
        assert (out >= 0).all() and (out <= 1).all()
