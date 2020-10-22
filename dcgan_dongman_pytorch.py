import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import time
import numpy

# Hyper-parameters
from torchvision.utils import make_grid

latent_size = 100
num_epochs = 100
batch_size = 128
learning_rate = 0.0002
DOWNLOAD_MNIST = False
beta1 = 0.5
imageSize = 96

nc = 3
ndf = 64

nz = 100
ngf = 64

# Image processing
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize([96, 96]),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = torchvision.datasets.ImageFolder('./face/faces', transform=transforms)

data_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)
# Discriminator
D = nn.Sequential(
# layer5输出尺寸 ncx96x96
    nn.Conv2d(nc, ndf, 5, 3, 1, bias=False),
    nn.BatchNorm2d(ndf),
    nn.LeakyReLU(0.2, inplace=True),
# layer4输出尺寸(ngf)x32x32
    nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
    nn.BatchNorm2d(ndf * 2),
    nn.LeakyReLU(0.2, inplace=True),
# layer3输出尺寸(ngf*2)x16x16
    nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
    nn.BatchNorm2d(ndf * 4),
    nn.LeakyReLU(0.2, inplace=True),
# layer2输出尺寸(ngf*4)x8x8
    nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
    nn.BatchNorm2d(ndf * 8),
    nn.LeakyReLU(0.2, inplace=True),
# layer1输入的是一个nzx1x1的随机噪声, 输出尺寸(ngf*8)x4x4
    nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
    nn.Sigmoid()

)

# Generator
G = nn.Sequential(
# layer1输入的是一个nzx1x1的随机噪声, 输出尺寸(ngf*8)x4x4
    nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
    nn.BatchNorm2d(ngf * 8),
    nn.ReLU(True),

# layer2输出尺寸(ngf*4)x8x8
    nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
    nn.BatchNorm2d(ngf * 4),
    nn.ReLU(True),

# layer3输出尺寸(ngf*2)x16x16
    nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
    nn.BatchNorm2d(ngf * 2),
    nn.ReLU(True),

# layer4输出尺寸(ngf)x32x32
    nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
    nn.BatchNorm2d(ngf),
    nn.ReLU(True),

# layer5输出尺寸 ncx96x96
    nn.ConvTranspose2d(ngf, nc, 5, 3, 1, bias=False),
    nn.Tanh()
)



# Binary cross entropy loss and optimizer
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=(beta1, 0.999))
g_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=(beta1, 0.999))



def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()


# Statistics to be saved
d_losses = np.zeros(num_epochs)
g_losses = np.zeros(num_epochs)
real_scores = np.zeros(num_epochs)
fake_scores = np.zeros(num_epochs)

# Start training
total_step = len(data_loader)
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

    r, c = 5, 5
    noise = torch.randn(r * c, latent_size, 1, 1)
    gen_imgs = G(noise)
    gen_imgs = make_grid(gen_imgs.data,
                                 nrow=5, normalize=True)
    fig= plt.figure(figsize=(6, 6))
    plt.imshow(gen_imgs.permute(1, 2, 0).numpy())
    fig.savefig("image_face/%d.png" % epoch)
    plt.close()