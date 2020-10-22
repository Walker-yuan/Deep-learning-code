import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import time

# Hyper-parameters
latent_size = 100
num_epochs = 100
batch_size = 128
learning_rate = 0.0002
DOWNLOAD_MNIST = False

nc = 1
ndf = 32

nz = 100
ngf = 32

# Image processing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,),  # 3 for RGB channels
                         std=(0.3087,))])

# MNIST dataset
mnist = torchvision.datasets.MNIST(root='./data/',
                                   train=True,
                                   transform=transform,
                                   download=DOWNLOAD_MNIST)


# Data loader
data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                          batch_size=batch_size,
                                          shuffle=True)

# Discriminator
D = nn.Sequential(

    nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
    nn.BatchNorm2d(ndf * 2),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
    nn.BatchNorm2d(ndf * 4),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
    nn.Sigmoid()
)

# Generator
G = nn.Sequential(
    nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
    nn.BatchNorm2d(ngf * 4),
    nn.ReLU(True),

    nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, bias=False),
    nn.BatchNorm2d(ngf * 2),
    nn.ReLU(True),

    nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
    nn.BatchNorm2d(ngf),
    nn.ReLU(True),

    nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
    nn.Tanh()
)



# Binary cross entropy loss and optimizer
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate)
g_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate)



def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()


# Statistics to be saved
d_losses = np.zeros(num_epochs)
g_losses = np.zeros(num_epochs)
real_scores = np.zeros(num_epochs)
fake_scores = np.zeros(num_epochs)

# Start training
r, c = 5, 5
total_step = len(data_loader)
noise = torch.randn(r * c, latent_size, 1, 1)
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        time_start = time.time()
        bs = images.shape[0]
        images = Variable(images)
        # Create the labels which are later used as input for the BCE loss
        real_labels = torch.ones(bs, 1)
        real_labels = Variable(real_labels)
        fake_labels = torch.zeros(bs, 1)
        fake_labels = Variable(fake_labels)

        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #

        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs.mean().item()

        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        z = torch.randn(bs, latent_size, 1, 1)
        z = Variable(z)
        fake_images = G(z)
        outputs = D(fake_images).view(-1, 1).squeeze(1)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs.mean().item()

        # Backprop and optimize
        # If D is trained so well, then don't update
        d_loss = d_loss_real + d_loss_fake
        reset_grad()
        d_loss.backward()
        d_optimizer.step()
        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #

        # Compute loss with fake images
        z = torch.randn(bs, latent_size, 1, 1)
        z = Variable(z)
        fake_images = G(z)
        outputs = D(fake_images).view(-1, 1).squeeze(1)

        # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
        # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
        g_loss = criterion(outputs, real_labels)

        # Backprop and optimize
        # if G is trained so well, then don't update
        reset_grad()
        g_loss.backward()
        g_optimizer.step()
        # =================================================================== #
        #                          Update Statistics                          #
        # =================================================================== #
        d_losses[epoch] = d_loss.data
        g_losses[epoch] = g_loss.data
        real_scores[epoch] = real_score
        fake_scores[epoch] = fake_score
        time_end = time.time()

        if (i + 1) % 200 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}, time: {:.2f}'
                  .format(epoch, num_epochs, i + 1, total_step, d_loss.data, g_loss.data,
                          real_score, fake_score, time_end-time_start))

        gen_imgs = G(noise)
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :].detach().view(28, 28), cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()