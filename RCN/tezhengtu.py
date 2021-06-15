import argparse
import glob
import cv2
import torch
import numpy as np
import os
import time

from Restnet50 import *  # set ONNX_EXPORT in models.py
import sys



import matplotlib.pyplot as plt
from torchvision.utils import save_image





def viz(module, input):
	print(input[0].shape)
	x = input[0][0]
	#最多显示4张图
	print(x.size())
	min_num = np.minimum(4, x.size()[0])
	
	for i in range(min_num):
		#plt.subplot(1, 4, i+1)
		y = x[i].cpu() * 255
		plt.imshow(y,cmap ='gray')
		plt.savefig('./tezhengtu/'+str(i)+'.jpg')
		plt.show()








def main():
	device = torch.device('cuda:0')
	model = RestNet50()
	model.load_state_dict(torch.load("./result/best.pt"))
	model.to(device).eval()




	pic = cv2.imread('./958.png')
	img0 = pic.copy()       #512,640,3
	#img = letterbox(img0, new_shape=(640, 512))[0]   #变成32的倍数
	img = img0
	# Convert
	img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x512x640
	img = np.ascontiguousarray(img)  #返回一个内存连续的数组
	img = torch.from_numpy(img).to(device)
	img = img.float()  # uint8 to fp16/32

	img = img[0]
	img = img.unsqueeze(0)

	img /= 255.0  # 0 - 255 to 0.0 - 1.0
	if img.ndimension() == 3:
		img = img.unsqueeze(0)  # (1,3,512,640)

	

	for name, m in model.named_modules():
		if isinstance(m, torch.nn.Conv2d):
			m.register_forward_pre_hook(viz)


	with torch.no_grad():
		model(img)




if __name__ == '__main__':
	main()
	

























