import torch
import numpy as np
import os
import time
import random

from Restnet50 import RestNet50
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.ops import nms


device = torch.device("cuda" if torch.cuda.is_available else "cpu")

model = RestNet50()
model.load_state_dict(torch.load("./result/best.pt"))
model.to(device).eval()

test_dir = './dataset/test_img'
test_img = os.listdir(test_dir)
test_img.sort()

#anchor
anchor_base = np.array([[0,0,25,25]])
#_, _, height, width = features.shape
height = 244
width = 308
feat_stride = 2

shift_x = torch.arange(0, width * feat_stride, feat_stride)
shift_y = torch.arange(0, height * feat_stride, feat_stride)
shift_x, shift_y = np.meshgrid(shift_x, shift_y)
shift = np.stack((shift_y.ravel(), shift_x.ravel(),shift_y.ravel(), shift_x.ravel()), axis=1)#[0 0 0 0] [0 4 0 4]

anchor = shift + anchor_base #18240 4
anchor = anchor.astype(np.float32)
anchor = torch.from_numpy(anchor) #18240 4




for img_name in test_img:
	a = time.time()
	image = Image.open(os.path.join(test_dir,img_name)).convert('L')
	img = np.asarray(image,dtype=np.float32)
	#image.close()
	img = torch.from_numpy(img)
	img = img.view(1,1,512,640).to(device)
	feature = model(img)

	scores = feature.permute(0, 2, 3, 1).contiguous() 
	scores = scores.view(-1,2)[:,1] #1 120*152 2     1 120*152


	#取分数前100
	scores_index = torch.argsort(- scores)[0:100] #分数前100的索引
	anchor_ = anchor[scores_index]
	scores = scores[scores_index]

	#nms处理
	nms_index = nms(anchor_.cuda(),scores,0.1)#nms 返回nms处理之后的索引(按分数降序排列)
	print(nms_index.shape)
	anchor_ = anchor_[nms_index]
	scores = scores[nms_index]

	#nms处理后取前9
	top_anchor = anchor_[:9]

	#坐标转换
	top_anchor = top_anchor.numpy()
	top_anchor = top_anchor[:,[1,0,3,2]]

	#图像裁剪
	image_crop = list()
	for i in range(9):
		image_crop.append(image.crop(top_anchor[i]))
		#image_crop = image_crop.append(image_)

	random.shuffle(image_crop)
	img_concat1 = np.concatenate((image_crop[0],image_crop[1],image_crop[2]),axis=1)
	img_concat2 = np.concatenate((image_crop[3],image_crop[4],image_crop[5]),axis=1)
	img_concat3 = np.concatenate((image_crop[6],image_crop[7],image_crop[8]),axis=1)
	img_concat = np.concatenate((img_concat1,img_concat2,img_concat3),axis=0)
	#plt.imshow(image2,cmap ='gray')
	#plt.show()
	im = Image.fromarray(img_concat)
	im.save('./image_crop/' + img_name)

	image.close()




















