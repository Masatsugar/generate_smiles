import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from collections import namedtuple
import pandas as pd
import numpy as np
from chemutils import *


class Encoder(nn.Module):
    def __init__(self, input=120, hidden_dim=264, c=35):
        super(Encoder, self).__init__()
        # Encoder
        self.conv1 = nn.Conv1d(input, out_channels=9, kernel_size=9)  # (batch, out_channels, len)
        self.bn1 = nn.BatchNorm1d(9)
        self.conv2 = nn.Conv1d(9, out_channels=9, kernel_size=9)  # (batch, out_channels, len)
        self.bn2 = nn.BatchNorm1d(9)
        self.conv3 = nn.Conv1d(9, out_channels=10, kernel_size=11)  # (batch, out_channels, len)
        self.bn3 = nn.BatchNorm1d(10)

        self.fc1 = nn.Linear((c - 29 + 3) * 10, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, hidden_dim)
        self.fc22 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.flatten().view(x.shape[0], -1)
        x = self.fc1(x)
        h1 = F.relu(self.fc21(x))
        h2 = F.relu(self.fc22(x))
        return h1, h2


class Decoder(nn.Module):
    def __init__(self, encode_dim=264, o=120, char=35):
        super(Decoder, self).__init__()
        self.fc3 = nn.Linear(encode_dim, encode_dim)
        self.repeat_vector = Repeat(o)
        # self.embedding = nn.Embedding(hidden_dim, hidden_dim)
        self.gru = nn.GRU(encode_dim, 500, 4, batch_first=True)  # (B, Seq, Feature)
        self.seq = TimeDistributed(nn.Sequential(
            nn.Linear(500, char),
            nn.Softmax(dim=1)
        ))

    def forward(self, z):
        out = F.relu(self.fc3(z))
        out = self.repeat_vector(out)
        out, h = self.gru(out)
        out = self.seq(out)
        return out


class ChemVAE(nn.Module):
    def __init__(self, input_dim=120, hidden_dim=264, max_length=120, dict_dim=35):
        super(ChemVAE, self).__init__()
        # Encoder
        self.encoder = Encoder(input_dim, hidden_dim, dict_dim)
        self.decoder = Decoder(hidden_dim, max_length, dict_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = -0.5 * torch.mean(1. + logvar - mu ** 2. - torch.exp(logvar))
    return BCE + KLD


def train_model(train_loader, model, optimizer, print_every=200):
    model.train()
    for t, (x, label) in enumerate(train_loader):
        x_var = Variable(x).cuda()
        recon_batch, mu, logvar = model(x_var)
        loss = loss_function(recon_batch, x_var, mu, logvar)
        if (t + 1) % print_every == 0:
            print('t = %d, loss = %.4f' % (t + 1, loss.data.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate_model(val_loader, model):
    model.eval()
    avg_val_loss = 0.
    for t, (x, y) in enumerate(val_loader):
        x_var = Variable(x).cuda()
        recon_batch, mu, logvar = model(x_var)
        loss = loss_function(recon_batch, x_var, mu, logvar)
        avg_val_loss += loss.data
    avg_val_loss /= t
    print('average validation loss: %.4f' % avg_val_loss.item())
    return avg_val_loss.item()


# data configuration
df = pd.read_table("./data/train.txt", header=None)
df.columns = ['smiles']
smiles = np.array(df.smiles[0:100])
smiles = ['C(C)C', 'CFCC', 'C1cccc1c', 'CC(=O)']

opt = namedtuple('opt', field_names=['n_epochs', 'batch_size', 'lr', 'b1', 'b2'
                                                                           'length', 'dict_size', 'channels',
                                     'sample_interval'])

opt = opt(n_epochs=10, batch_size=1, lr=0.002, b1=0.5, b2=0.999, n_cpu=2,
          latent_dim=100, length=120, dict_size=35,
          channels=1, sample_interval=300)

vae = ChemVAE()
cuda = True if torch.cuda.is_available() else False

if cuda:
    vae.cuda()

one_hot = smiles_to_hot(smiles)
data_train = torch.tensor(one_hot)
train = TensorDataset(data_train, torch.zeros(data_train.size()[0]))
train_loader = DataLoader(train, batch_size=opt.batch_size, shuffle=True)

data_val = data_train
val = TensorDataset(data_val, torch.zeros(data_val.size()[0]))
val_loader = DataLoader(val, batch_size=opt.batch_size, shuffle=True)
optimizer = optim.Adam(vae.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

best_loss = 1E6
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', min_lr=1E-5)
for epoch in range(opt.epochs):
    print(f'Epoch {epoch}')
    train_model(train_loader, vae, optimizer)
    avg_val_loss = validate_model(val_loader, vae)
    scheduler.step(avg_val_loss, epoch)
    is_best = avg_val_loss < best_loss
    if is_best:
        best_loss = avg_val_loss


def test_recon(vae, tes):
    vae.eval()
    mu, logvar = vae.encoder(tes.cuda())
    z = vae.reparameterize(mu, logvar)
    out = vae.decoder(z[0:10, :])
    cand = out.detach().cpu().numpy()
    cands = hot_to_smiles(cand, id2char)
    print(cands)


test_recon(vae, data_train)
