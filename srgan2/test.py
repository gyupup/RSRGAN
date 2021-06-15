import torch
import numpy as np
import os
import time
import random
import cv2

from models import GeneratorResNet1,GeneratorResNet2
from PIL import Image
from torchvision.utils import save_image,make_grid
import matplotlib.pyplot as plt
from torchvision.ops import nms
import torchvision.transforms as transforms
import torch.nn as nn

#GPU
device = torch.device("cuda" if torch.cuda.is_available else "cpu")

#加载DBGAN
d = 460
model_DBGAN = GeneratorResNet1()
model_DBGAN.load_state_dict(torch.load("./saved_models/G1_%d.pth" %d))
model_DBGAN.to(device).eval()

#加载SRGAN
d = 460
model_SRGAN = GeneratorResNet2()
model_SRGAN.load_state_dict(torch.load("./saved_models/G2_%d.pth" %d))
model_SRGAN.to(device).eval()

#测试图片文件夹
test_dir = '/home/nvidia/RCN+GAN/srgan2/plane1'
test_img = os.listdir(test_dir)
test_img.sort()

mean = np.array([0.5])
std = np.array([0.5])

for img_name in test_img:
	with torch.no_grad():
		image = Image.open(os.path.join(test_dir,img_name)).convert('L')

		width,height = image.size

		transform = transforms.Compose([
                #transforms.Resize((height // 4, width // 4), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
		img = transform(image)
		_,height,width = img.shape
		img = img.view(1,1,height,width).type(torch.cuda.FloatTensor)

		img_db = model_DBGAN(img) 
		img_hr = model_SRGAN(img_db)

		gen_hr = make_grid(img_hr, nrow=1, normalize=True)
		save_image(gen_hr, "test_result/%s.png" % img_name, normalize=False)
		

		
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















