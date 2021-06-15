import torch
import numpy as np
import os
import time
import random
import cv2

from RCN.Restnet50 import RestNet50
from models import GeneratorResNet1
from srgan.models import GeneratorResNet,FeatureExtractor,Cls,Loc
from PIL import Image
from torchvision.utils import save_image,make_grid
import matplotlib.pyplot as plt
from torchvision.ops import nms
from RCN.utils import anchor_base_RCN,anchor_base_SRGAN,heap_sort
from srgan.utils import loc2bbox
import torchvision.transforms as transforms
import torch.nn as nn

#GPU
device = torch.device("cuda" if torch.cuda.is_available else "cpu")

#加载RCN
model_RCN = RestNet50()
model_RCN.load_state_dict(torch.load("./RCN/result/best.pt"))
model_RCN.to(device).eval()

#加载db_GAN
model_dbgan = GeneratorResNet1((1,25,25),9)
model_dbgan.load_state_dict(torch.load("./saved_models/G_AB_168.pth" ))
model_dbgan.to(device).eval()

#加载SRGAN
model_SRGAN = GeneratorResNet()
model_SRGAN.load_state_dict(torch.load("./srgan/saved_models/generator_160.pth"))
model_SRGAN.to(device).eval()

#加载特征提取器
model_feature = FeatureExtractor()
model_feature.load_state_dict(torch.load("./srgan/saved_models/feature_extractor_190.pth"))
model_feature.to(device).eval()

#加载cls识别器
model_cls = Cls()
model_cls.load_state_dict(torch.load("./srgan/saved_models/cls_190.pth"))
model_cls.to(device).eval()


#加载边框回归
model_loc = Loc()
model_loc.load_state_dict(torch.load("./srgan/saved_models/loc_190.pth"))
model_loc.to(device).eval()



#测试图片文件夹
test_dir = './test1_dir'
test_img = os.listdir(test_dir)
test_img.sort()

mean = np.array([0.5])
std = np.array([0.5])

for img_name in test_img:
	with torch.no_grad():
		image = Image.open(os.path.join(test_dir,img_name)).convert('L')
		transform = transforms.Compose([
               # transforms.Resize((100 // 4, 100 // 4), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
		img_nm = transform(image)
		#print(img_nm.shape)
		img_nm = img_nm.view(1,1,25,25).type(torch.cuda.FloatTensor)


		img_db = model_dbgan(img_nm)


		



		gen_db = make_grid(img_db, nrow=1, normalize=True)
		save_image(gen_db, "tttt/%s.png" % img_name, normalize=False)


		img_hr = model_SRGAN(img_db)#4 1 192 192

		gen_hr = make_grid(img_hr, nrow=1, normalize=True)
		save_image(gen_hr, "ttttttt/%s.png" % img_name, normalize=False)
		'''





		feature1 = model_feature(img_hr)# 36 256 25 25

		#tongguo cls
		cls1 = model_cls(feature1) 
		#S = nn.Sigmoid()
		#cls1 = S(cls1) #* 10000
		print(cls1)
		cls2 = (cls1>0.1).nonzero()[:,0]


		feature2 = feature1[cls2]
		loc1 = model_loc(feature2)
		loc1 = loc1.cpu().numpy()
		loc2 = loc2bbox(loc1) 
		print(loc2)
		#loc回归后索引到原图坐标
		index1 = cls2 % 4



		c1, c2 = (int(loc2[0][1]), int(loc2[0][0])), (int(loc2[0][3]), int(loc2[0][2]))
		print(c1,c2)

		image_crop_t = img_hr[cls2].view(100,100).cpu().numpy()
		print(image_crop_t.shape)
		cv2.rectangle(image_crop_t, c1, c2, color=(67,0,12), thickness=1, lineType=cv2.LINE_AA)
		cv2.imshow('67',image_crop_t)
		cv2.waitKey(0)




		'''






















