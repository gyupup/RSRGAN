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
import glob

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import *
from utils import *
#from utils import _fast_rcnn_loc_loss,bbox2loc

import torch.nn as nn
import torch.nn.functional as F
import torch

cuda = torch.cuda.is_available()

device = torch.device("cuda" if torch.cuda.is_available else "cpu")


mean = np.array([0.5])
std = np.array([0.5])


# Initialize generator and discriminator
'''
discriminator = Discriminator(input_shape=(1, *hr_shape))
model_dict = discriminator.state_dict()

pretrained_dict = torch.load("./D_460.pth")
pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}

model_dict.update(pretrained_dict)
discriminator.load_state_dict(model_dict)
discriminator.to(device).eval()
'''


feature_extractor = FeatureExtractor()
feature_extractor.load_state_dict(torch.load("./saved_models/feter_590.pth"))
feature_extractor.to(device).eval()

model_cls = Cls()
model_cls.load_state_dict(torch.load("./saved_models/cls_590.pth"))
model_cls.to(device).eval()


model_loc = Loc()
model_loc.load_state_dict(torch.load("./saved_models/loc_590.pth"))
model_loc.to(device).eval()



Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
hr_transform = transforms.Compose(
            [
                #transforms.Resize((hr_height, hr_width), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )


#file = os.listdir('./test_loc')
file = glob.glob('./test_loc/'+'*.*')

for img_name in file:
    img = Image.open(img_name).convert('L')
    img1 = hr_transform(img)
    img1 = img1.view(1,1,80,80).type(Tensor)
    feature = feature_extractor(img1)
    feature_loc = model_loc(feature)
    bbox = loc2bbox(feature_loc)

    print(bbox)

    c1, c2 = (int(bbox[0][1]), int(bbox[0][0])), (int(bbox[0][3]), int(bbox[0][2]))
    img1 = img1.view(80,80,1).cpu().numpy()



    img11 = cv2.imread(img_name,0)
    cv2.rectangle(img11, c1, c2, color=(255,255,255), thickness=1, lineType=cv2.LINE_AA)
    cv2.imwrite('./temp2/'+img_name[11:],img11)













'''
img1 = hr_transform(img)
img1 = img1.view(1,1,80,80).type(Tensor)
feature = feature_extractor(img1)
feature_cls = model_cls(feature)
S = nn.Sigmoid()
cls1 = S(feature_cls)
print(cls1)

feature_loc = model_loc(feature)
bbox = loc2bbox(feature_loc)
c1, c2 = (int(bbox[0][1]), int(bbox[0][0])), (int(bbox[0][3]), int(bbox[0][2]))
img1 = img1.view(80,80,1).cpu().numpy()

#cv2.rectangle(img1, c1, c2, color=(255,255,255), thickness=1, lineType=cv2.LINE_AA)
cv2.imwrite('./temp2/'+'195_1'+'.png',img1*255)






img_ = Image.open("./195_34.png").convert('L')
img2 = hr_transform(img_)
img2 = img2.view(1,1,80,80).type(Tensor)
feature1 = feature_extractor(img2)
feature_cls1 = model_cls(feature1)
S = nn.Sigmoid()
cls2 = S(feature_cls1)
print(cls2)

feature_loc1 = model_loc(feature1)

print(feature_loc1)
bbox1 = loc2bbox(feature_loc1)
print(bbox1)
c11, c22 = (int(bbox1[0][1]), int(bbox1[0][0])), (int(bbox1[0][3]), int(bbox1[0][2]))

img2 = img2.view(80,80,1).cpu().numpy()
cv2.rectangle(img2, c11, c22, color=(255,255,255), thickness=1, lineType=cv2.LINE_AA)
cv2.imwrite('./temp2/'+'28_34'+'.png',img2*255)


loc_ = bbox2loc(bbox1)
print(loc_)






'''

















