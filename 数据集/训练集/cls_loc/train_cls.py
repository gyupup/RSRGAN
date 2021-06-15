"""
单独训练cls分类
"""


import argparse
import os
import numpy as np
import math
import itertools
import sys
import cv2

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import _fast_rcnn_loc_loss,bbox2loc
from models import *
from datasets import *
#from utils import _fast_rcnn_loc_loss,bbox2loc

import torch.nn as nn
import torch.nn.functional as F
import torch

#os.makedirs("images_gan_r", exist_ok=True)
#os.makedirs("saved_models", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=600, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=6, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=80, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=80, help="high res. image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available else "cpu")

# Initialize generator and discriminator
feature_extractor = FeatureExtractor()
cls = Cls()
loc = Loc()

criterion_cls = nn.BCEWithLogitsLoss()

if cuda:
    feature_extractor = feature_extractor.cuda()
    cls = cls.cuda()
    loc = loc.cuda()
    criterion_cls = criterion_cls.cuda()

optimizer_feter = torch.optim.Adam(feature_extractor.parameters(),lr=opt.lr,betas=(opt.b1,opt.b2))
optimizer_Cls = torch.optim.Adam(cls.parameters(),lr=opt.lr,betas=(opt.b1,opt.b2))
optimizer_Loc = torch.optim.Adam(loc.parameters(),lr=0.00001,betas=(opt.b1,opt.b2))

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

        optimizer_Cls.zero_grad()
        optimizer_Loc.zero_grad()
        optimizer_feter.zero_grad()

        # Configure model input
        imgs_hr = Variable(imgs["hr"].type(Tensor)) #1 1 100 100
        '''
        img_hrr = make_grid(imgs_hr, nrow=2, normalize=True)
        save_image(img_hrr, "temp/%d.png" % i, normalize=False)
        '''
        label = Variable(imgs["label"].type(Tensor)) #1 1
        #print(label)

        bbox = imgs["bbox"].numpy()#32 1 4
        bbox = bbox.reshape((bbox.shape[0],4))#32 4
        #print(bbox)
        gt_loc = bbox2loc(bbox)
        gt_loc = torch.from_numpy(gt_loc)
        gt_loc = Variable(gt_loc.type(Tensor))

        '''
        c1, c2 = (int(bbox[0][1]), int(bbox[0][0])), (int(bbox[0][3]), int(bbox[0][2]))
        imgs_hr1 = imgs_hr.view(80,80,1).cpu().numpy()
        cv2.rectangle(imgs_hr1, c1, c2, color=(255,255,255), thickness=1, lineType=cv2.LINE_AA)
        cv2.imwrite('./temp1/'+str(i)+'.png',imgs_hr1*255)
        '''



 
        features = feature_extractor(imgs_hr)

        feature_cls = cls(features)
        loss_cls = criterion_cls(feature_cls,label)

        features_loc = loc(features)
        loss_loc = _fast_rcnn_loc_loss(features_loc,gt_loc,label,1.)

        print(loss_cls.data)
        print(loss_loc.data)

        loss = loss_cls + loss_loc

        print(epoch)



        # Total loss

        loss.backward()
        optimizer_feter.step()
        optimizer_Cls.step()
        optimizer_Loc.step()


    if opt.checkpoint_interval != -1 and epoch % 10 == 0:
        # Save model checkpoints

        torch.save(cls.state_dict(), "saved_models/cls_%d.pth" % epoch)
        torch.save(feature_extractor.state_dict(), "saved_models/feter_%d.pth" % epoch)
        torch.save(loc.state_dict(),"saved_models/loc_%d.pth" % epoch)















