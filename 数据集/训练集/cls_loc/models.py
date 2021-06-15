import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
import math
import pprint

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        list1 = list(vgg19_model.features.children())[:18]
        list1[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.feature_extractor = nn.Sequential(*list1)

    def forward(self, img):

            feature = self.feature_extractor(img)# 1 256 25 25
            return feature



class Cls(nn.Module):
    def __init__(self,):
        super(Cls,self).__init__()

        list2 = [nn.Conv2d(256, 128,kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(128, 0.8),nn.ReLU(inplace=True),]
			#nn.Linear(128*25*25, 1,kernel_size=1, stride=1,padding=0),nn.Sigmoid()]
        self.feature_cls = nn.Sequential(*list2)

        l1 = [nn.Linear(128*20*20,1)]#,nn.BatchNorm1d(1, 0.8)]
        self.feature_cls_l1 = nn.Sequential(*l1)


    def forward(self,img_feature):
        batch_size, _, _, _ = img_feature.shape
        cls = self.feature_cls(img_feature)
        cls = cls.view(batch_size,-1)

        cls = self.feature_cls_l1(cls)

        #img_feature = img_feature.view(batch_size,-1)

        return cls

class Loc(nn.Module):
    def __init__(self,):
        super(Loc,self).__init__()

        list3 = [nn.Conv2d(256, 128,3,1,1),nn.BatchNorm2d(128,0.8),nn.ReLU(inplace=True)]#,nn.Linear(256, 4),nn.Sigmoid()]
        self.feature_loc = nn.Sequential(*list3)

        l2 = [nn.Linear(128*20*20,4)]
        self.feature_loc_l2 = nn.Sequential(*l2)

    def forward(self,img_feature):
        
        batch_size, _, _, _ = img_feature.shape
        #img_feature = img_feature.view(batch_size,-1)
        loc = self.feature_loc(img_feature)
        loc = loc.view(batch_size,-1)
        loc = self.feature_loc_l2(loc)
        return loc
	












