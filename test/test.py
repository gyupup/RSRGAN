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
from RCN.utils import anchor_base_RCN,anchor_base_SRGAN,heap_sort,anchor_base_front
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




d = 150
#加载SRGAN
model_SRGAN = GeneratorResNet()
model_SRGAN.load_state_dict(torch.load("./srgan/saved_models/generator_%d.pth" % d))
model_SRGAN.to(device).eval()

'''
#加载特征提取器
model_feature = FeatureExtractor()
model_feature.load_state_dict(torch.load("./srgan/saved_models/feature_extractor_190.pth"))
model_feature.to(device).eval()

#加载cls识别器
model_cls = Cls()
model_cls.load_state_dict(torch.load("./srgan/saved_models/cls_180.pth"))
model_cls.to(device).eval()


#加载边框回归
model_loc = Loc()
model_loc.load_state_dict(torch.load("./srgan/saved_models/loc_190.pth"))
model_loc.to(device).eval()
'''

#anchor_base
anchor1 = anchor_base_RCN()
anchor2 = anchor_base_SRGAN()
anchor3 = anchor_base_front()





#测试图片文件夹
test_dir = './test_dir'
test_img = os.listdir(test_dir)
test_img.sort()

mean = np.array([0.5])
std = np.array([0.5])

for img_name in test_img:
	with torch.no_grad():
		#从测试图片名字中提取数字
		num = [x for x in img_name if x.isdigit()]
		num = "".join(num)
		num = int(num)

		#读取测试图片，转换成tensor
		image = Image.open(os.path.join(test_dir,img_name)).convert('L')	
		img_y = np.array(image)	
		img = np.asarray(image,dtype=np.float32)
		img = torch.from_numpy(img)
		img = img.view(1,1,512,640).to(device)

		#model_RCN得到分数序列
		feature = model_RCN(img)
		scores = feature.permute(0, 2, 3, 1).contiguous() 
		#save_image(scores[:,:,:,1],'./RCN/tezhengtu/1.png', normalize=False)
		scores = scores.view(-1,2)[:,1]

		#取分数前100
		scores_index = torch.argsort(- scores)[0:100] #分数前100的索引    这个排序浪费太多的时间了  
		scores = scores[scores_index]

		#提取对应的anchor
		anchor_ = anchor1[scores_index]

		#nms处理
		nms_index = nms(anchor_.cuda(),scores,0.1)#nms 返回nms处理之后的索引(按分数降序排列)
		anchor_ = anchor_[nms_index]
		scores = scores[nms_index]

		#nms处理后取阈值  0.1
		thres_index = scores > 0.1
		top_anchor = anchor_[thres_index]

		#目标可能区域个数
		n_ = top_anchor.shape[0]

		#坐标转换
		top_anchor = top_anchor.numpy()
		top_anchor = top_anchor[:,[1,0,3,2]]

		#图像裁剪和拼接
		image_list = list()
		image_list.append(image.crop(top_anchor[0]))
		image_concat = np.array(image.crop(top_anchor[0]))

		for i in range(1,n_):
			image_crop = image.crop(top_anchor[i])#裁剪
			image_list.append(image_crop)
			image_concat = np.concatenate((image_concat,image.crop(top_anchor[i])),axis=1)

		#保存拼接后的图像
		im = Image.fromarray(image_concat)
		im.save('./result/rcn_result/' + img_name)

		#封装tensor
		img0 = torch.empty((n_,48,48))
		transform = transforms.Compose([#transforms.Resize((48, 48), Image.BICUBIC),
				transforms.ToTensor(),
				transforms.Normalize(mean, std),])
		for i,img in enumerate(image_list):
			img_nm = transform(img)
			img0[i] = img_nm
		img0 = img0.view(n_,1,48,48).type(torch.cuda.FloatTensor)
		
		'''
		img1 = torch.empty((9,n_,1,25,))
		for j,anchor in  enumerate(anchor1):
			img1[j] = img0[:,:,anchor[0]:anchor[2],anchor[1]:anchor[3]] #n_ 1 25 25
		img1 = img1.view(9*n_,1,100,100).type(torch.cuda.FloatTensor) #36 1 100 100
		'''

		

		img1 = model_dbgan(img0)


		for i in range(n_):
			img11 = make_grid(img1[i], nrow=1, normalize=False)
			save_image(img11, "./result/dbgan_result/%d_%d.png" % (num,i), normalize=False)


		
		
		#通过SRGAN提高分辨率
		img2 = model_SRGAN(img1)#4 1 192 192

		for i in range(n_):
			img22 = make_grid(img2[i], nrow=1, normalize=False)
			save_image(img22, "./result/srgan_result/%d_%d.png" % (num,i), normalize=False)



		#拆分成4*9 1 100 100
		#img1 = torch.empty((9,n_,1,100,100))
		#for j,anchor in  enumerate(anchor2):
			#img1[j] = img_hr[:,:,anchor[0]:anchor[2],anchor[1]:anchor[3]] #9 25 25
		#img1 = img1.view(9*n_crop,1,100,100).type(torch.cuda.FloatTensor) #36 1 100 100

		#for k in range(36):
			#img_k = make_grid(img1[k], nrow=1, normalize=True)
			#save_image(img_k,"./result/%s_%d.png"%(img_name,k),normalize=False)


		'''


		#提取特征
		#feature_extractor提取特征
		feature1 = model_feature(img1)# 36 256 25 25

		#通过 cls
		cls1 = model_cls(feature1) 
		#S = nn.Sigmoid()
		#cls1 = S(cls1) #* 10000
		print(cls1)
		
		max_cls = cls1.argmax()
		print(max_cls)


		#cls2 = (cls1>24).nonzero()[:,0]
		#print(cls2)

		feature2 = feature1[max_cls].view(1,256,25,25)
		#print(feature2.shape)


		#是目标的loc回归
		loc1 = model_loc(feature2)
		loc1 = loc1.cpu().numpy()
		loc2 = loc2bbox(loc1)
		#print(loc2)
		#loc回归后索引到原图坐标




		c1, c2 = (int(loc2[0][1]), int(loc2[0][0])), (int(loc2[0][3]), int(loc2[0][2]))
		print(c1,c2)

		image_crop_t = img1[max_cls]
		image_crop_t = make_grid(image_crop_t, nrow=1, normalize=True)[0]

		image_crop_t = image_crop_t.view(100,100,1).cpu().numpy()
		cv2.rectangle(image_crop_t, c1, c2, color=(100,100,200), thickness=1, lineType=cv2.LINE_AA)
		#cv2.imshow('555',image_crop_t)
		#cv2.waitKey(0)

		cv2.imwrite('./result/'+str(num)+'_hr.png',image_crop_t*255)




		#映射到192,192

		index1 = max_cls / 4  #2
		index1_1 = index1 / 3 #0
		index1_2 = index1 % 3 #2
		print(index1_1)
		print(index1_2)

		c1_192 = (c1[0] + index1_2.item() * 44 , c1[1] + index1_1.item() * 44)
		c2_192 = (c2[0] + index1_2.item() * 44 , c2[1] + index1_1.item() * 44)
		print(c1_192,c2_192)




		#映射到48,48

		c1_48 = (int(c1_192[0]/4) , int(c1_192[1]/4))
		c2_48 = (int(c2_192[0]/4) , int(c2_192[1]/4))
		print(c1_48,c2_48)

		
		#映射到512,640
		
		index2 = (max_cls % 4).item()
		print(top_anchor[3])
		c1_y = (int(c1_48[0] + top_anchor[index2][0]) , int(c1_48[1] + top_anchor[index2][1]))
		c2_y = (int(c2_48[0] + top_anchor[index2][0]) , int(c2_48[1] + top_anchor[index2][1]))
		print(c1_y,c2_y)

		
		cv2.rectangle(img_y, c1_y, c2_y, color=(255,255,255), thickness=1, lineType=cv2.LINE_AA)
		#cv2.imshow('666666',img_y)
		#cv2.waitKey(0)
		cv2.imwrite('./result/'+str(num)+'_taget.png',img_y)





		'''












