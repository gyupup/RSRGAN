from data import Dataset
from Restnet50 import RestNet50
from utils import optimizer,if_init,anchor_base_RCN

import torch
import numpy as np
from torch.utils import data
from pprint import pprint
#from torchsummary import summary
from torch.nn import functional 
import cv2
import time

#cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#data
dataset=Dataset('train.txt')
dataloader = data.DataLoader(dataset , batch_size=1 , pin_memory=True,shuffle=True)

test_dataset=Dataset('test.txt')
test_dataloader = data.DataLoader(test_dataset , batch_size=1 , pin_memory=True,shuffle=True)

#model
model=RestNet50()
model.cuda()
#summary(model,(1,512,640))

#opt
epochs = 30
lr=0.0001
best = 0
n=1

#optimizer
optimizer1 = optimizer(lr=lr,model=model)

#anchor
anchor = anchor_base_RCN()

for epoch in range(epochs):
	for ii, (img, bbox,) in enumerate(dataloader):
		#img tensor 1 1 520 640
		#bbox tensor 1 1 4
		#label1 tensor 1 1

		img = img.to(device)
		n, c, h, w = img.shape  #1 1 512 640

		optimizer1.zero_grad()
		model.train()
		features = model(img) #1 2 57 73
		

		#scores
		scores = features.permute(0, 2, 3, 1).contiguous() #1 57 73 2
		#softmax_scores = functional.softmax(scores, dim=3)

		#提取出前景概率
		#fg_scores = softmax_scores[:, :, :, 1].contiguous() #1 120 152
		#fg_scores = fg_scores.view(n, -1) #1 120*152

		#前景和背景得分
		scores = scores.view(n, -1, 2)#1 57*73 2


		#label
		label = np.empty(shape=(n,len(anchor)))
		label.fill(-1)

		label1 = if_init(anchor,bbox)
		S = (bbox[:,:,2]-bbox[:,:,0])*(bbox[:,:,3]-bbox[:,:,1]) #1 1
		#label1 = torch.true_divide(label1, S)
		label1 = label1/S

		for i in range(n):
			pos_index = np.where(label1[i] == 1)[0]
			neg_index = np.where(label1[i] == 0)[0]
			disable_index = np.random.choice(neg_index, size=len(pos_index), replace=False)

			label[i][pos_index] = 1
			label[i][disable_index] = 0

		label = label.astype(np.int64)
		label = torch.from_numpy(label).to(device) #1 18240
	

		#loss
		loss = functional.cross_entropy(scores[0], label[0], ignore_index=-1) #1batch
		loss.backward()
		optimizer1.step()
		print(ii,'ok')


	print("Epoch %d/%d    loss: %f" % (epoch, epochs-1,  loss.item()))
	if True :
		model.eval()
		model_score = 0

		with torch.no_grad():
			for img, bbox in test_dataloader:
				img = img.to(device)
				test_features = model(img)
				del(img)

				test_scores = test_features.permute(0, 2, 3, 1).contiguous() #1 120 152 2
				del(test_features)

				#softmax_scores = functional.softmax(test_scores, dim=3)
				#fg_scores = softmax_scores[:, :, :, 1].contiguous() #1 120 152
				#fg_scores = fg_scores.view( -1) #1 120*152

				test_scores = test_scores.view(n,-1,2)[:,:,1] #1 120*152 2     1 120*152
				top10_index = torch.argsort(- test_scores[0])[0:10]
				top10_anchor = anchor[top10_index]
				del(test_scores)


				img_score = if_init(top10_anchor,bbox)
				S = (bbox[:,:,2]-bbox[:,:,0])*(bbox[:,:,3]-bbox[:,:,1]) #1 1
				#img_score = torch.true_divide(img_score, S)
				img_score = img_score/S
				img_score = torch.sum(img_score == 1)
				#img_score = torch.true_divide(img_score, 10)
				img_score = (img_score.numpy())/10
				model_score = model_score + img_score
				print(img_score)

			model_score = model_score / len(test_dataloader)
			model_score = model_score.item()
			print(model_score)
			if model_score > best:
				best = model_score
				torch.save(model.state_dict(),'./result/best.pt')
	
	





























