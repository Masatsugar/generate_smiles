# Vanilla GAN
# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from chemutils import *
import pandas as pd

from collections import namedtuple

opt = namedtuple('opt', field_names=['n_epochs', 'batch_size', 'lr', 'b1', 'b2', 'n_cpu', 'latent_dim',
                                     'length', 'dict_size', 'channels', 'sample_interval'])

opt = opt(n_epochs=200, batch_size=1, lr=0.002, b1=0.5, b2=0.999, n_cpu=2,
          latent_dim=100, length=120, dict_size=35,
          channels=1, sample_interval=300)

data_shape = (opt.batch_size, opt.length, opt.dict_size)

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self, o=120, char=35):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 128)
        )
        self.repeat_vector = Repeat(o)
        # self.embedding = nn.Embedding(hidden_dim, hidden_dim)
        self.gru = nn.GRU(128, 200, 3, batch_first=True)  # (B, Seq, Feature)
        self.seq = TimeDistributed(nn.Sequential(
            nn.Linear(200, char),
            nn.Softmax(dim=1)
        ))

    def forward(self, z):
        out = self.model(z)
        out = self.repeat_vector(out)
        out, h = self.gru(out)
        out = self.seq(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_img=120, char=35):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv1d(input_img, out_channels=9, kernel_size=4)  # (batch, out_channels, len)
        self.bn1 = nn.BatchNorm1d(9)
        self.conv2 = nn.Conv1d(9, out_channels=9, kernel_size=4)  # (batch, out_channels, len)
        self.bn2 = nn.BatchNorm1d(9)
        self.fc1 = nn.Linear((char - 8 + 2) * 9, 512)

        self.model = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.flatten().view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        validity = self.model(x)
        return validity


adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader

df = pd.read_table("./data/train.txt", header=None)
df.columns = ['smiles']
smiles = np.array(df.smiles[0:100])
smiles = ['C(C)C', 'CFCC', 'C1cccc1c', 'CC(=O)']

one_hot = smiles_to_hot(smiles)
data_train = torch.tensor(one_hot)
train = TensorDataset(data_train, torch.zeros(data_train.size()[0]))
dataloader = DataLoader(train, batch_size=opt.batch_size, shuffle=True)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        #  Train Generator
        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        #  Train Discriminator
        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(f"Epoch [{epoch}/{opt.n_epochs}] [Batch {i}/{len(dataloader)}] \
        [D loss:{d_loss.item():4f}] [G loss:{g_loss.item():4f}]")

        batches_done = epoch * len(dataloader) + i
