import argparse
import os
import numpy as np
import math
import itertools
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images_gan_small", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=6, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=48, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=60, help="high res. image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available()

hr_shape = (opt.hr_height, opt.hr_width)

# Initialize generator and discriminator
G1 = GeneratorResNet1()
G2 = GeneratorResNet2()
D = Discriminator(input_shape=(opt.channels, *hr_shape))
feature_extractor = FeatureExtractor()

# Set feature extractor to inference mode
feature_extractor.eval()

# Losses
MSE = torch.nn.MSELoss() #L2
L1 = torch.nn.L1Loss()
tv_loss = TVLoss()

if cuda:
    G1 = G1.cuda()
    G2 = G2.cuda()
    D = D.cuda()
    feature_extractor = feature_extractor.cuda()


    MSE = MSE.cuda()
    L1 = L1.cuda()
    tv_loss = tv_loss.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(itertools.chain(G1.parameters(), G2.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

dataloader = DataLoader(
    ImageDataset(),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

# ----------
#  Training
# ----------

for epoch in range(opt.epoch, opt.n_epochs): #0 200
    for i, imgs in enumerate(dataloader):


        D.train()

        # Configure model input
        img_blur = Variable(imgs["blur"].type(Tensor))
        img_lr = Variable(imgs["lr"].type(Tensor))
        img_hr = Variable(imgs["hr"].type(Tensor)) 


        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((img_lr.size(0), *D.output_shape))), requires_grad=False) 
        fake = Variable(Tensor(np.zeros((img_lr.size(0), *D.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        G1.train()
        G2.train()


        optimizer_G.zero_grad()


        gen_db = G1(img_blur) 
        loss_IMG1 = MSE(gen_db,img_lr)
        loss_identity = L1(G1(img_lr),img_lr)

        loss1 = loss_IMG1 + 10 * loss_identity
        print(loss_identity)
        print(loss_IMG1)


        gen_hr = G2(gen_db)
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(img_hr)

        loss_IMG2 = MSE(gen_hr,img_hr)
        loss_GAN2 = MSE(D(gen_hr), valid)
        loss_content = L1(gen_features, real_features.detach())
        loss_tv = tv_loss(gen_hr)

        loss2 = loss_IMG2 + 0.006 * loss_content + 1e-3 * loss_GAN2 + 2e-8 * loss_tv

        loss = loss1 + loss2

        loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss of real and fake images
        loss_real = MSE(D(img_hr), valid)
        loss_fake = MSE(D(gen_hr.detach()), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        
        sys.stdout.write(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G1 loss: %f] [G2 loss: %f] "
            % (epoch, opt.n_epochs, i, len(dataloader), loss_D.item(), loss1.item(), loss2.item())
        )
        
        print('')

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            # Save image grid with upsampled inputs and SRGAN outputs
            img_hrr = nn.functional.interpolate(img_lr, scale_factor=4)
            gen_hrr = make_grid(gen_hr, nrow=2, normalize=True)
            img_hrr = make_grid(img_hrr, nrow=2, normalize=True)
            img_grid = torch.cat((img_hrr, gen_hrr), -1)

            save_image(img_grid, "images/%d.png" % batches_done, normalize=False)

    if opt.checkpoint_interval != -1 and epoch % 10 == 0:
        # Save model checkpoints
        torch.save(G1.state_dict(), "saved_models/G1_%d.pth" % epoch)
        torch.save(G2.state_dict(), "saved_models/G2_%d.pth" % epoch)
        torch.save(D.state_dict(), "saved_models/D_%d.pth" % epoch)
        torch.save(feature_extractor.state_dict(), "saved_models/feature_extractor_%d.pth" % epoch)

















