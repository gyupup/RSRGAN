import torch
import numpy as np
import os
import time
import random
import cv2

from RCN.Restnet50 import RestNet50
from models import FeatureExtractor,Cls,Loc
from srgan2.models import GeneratorResNet1,GeneratorResNet2
from PIL import Image
from torchvision.utils import save_image,make_grid
import matplotlib.pyplot as plt
from torchvision.ops import nms
from utils import *
import torchvision.transforms as transforms
import torch.nn as nn
import xml.etree.ElementTree as ET


#GPU
device = torch.device("cuda" if torch.cuda.is_available else "cpu")

#加载RCN
model_RCN = RestNet50()
model_RCN.load_state_dict(torch.load("./RCN/result/best.pt"))
model_RCN.to(device).eval()



#加载db_GAN
d = 460
model_dbgan = GeneratorResNet1()
model_dbgan.load_state_dict(torch.load("./srgan2/saved_models/G1_%d.pth" %d ))
model_dbgan.to(device).eval()

#加载SRGAN
model_SRGAN = GeneratorResNet2()
model_SRGAN.load_state_dict(torch.load("./srgan2/saved_models/G2_%d.pth" % d))
model_SRGAN.to(device).eval()

d = 300
#加载特征提取器
feature_extractor = FeatureExtractor()
feature_extractor.load_state_dict(torch.load("./srgan2/saved_models/feter_%d.pth" %d))
feature_extractor.to(device).eval()

#加载cls识别器
model_cls = Cls()
model_cls.load_state_dict(torch.load("./srgan2/saved_models/cls_%d.pth" %d))
model_cls.to(device).eval()




d= 490
#加载边框回归
model_loc = Loc()
model_loc.load_state_dict(torch.load("./srgan2/saved_models/loc_%d.pth" %d))
model_loc.to(device).eval()


#anchor_base
anchor1 = anchor_base_RCN()
anchor2 = anchor_base_SRGAN()


mean = np.array([0.5])
std = np.array([0.5])

iouv = torch.linspace(0.5, 0.95, 10).to(device)
#iouv = iouv[0].view(1)

iouv = torch.tensor([0.1]).to(device)
niou = iouv.numel()
stats = []
def normalize1(tensor):
	b = tensor.shape[0]

	min1 = torch.min(tensor.reshape(b,-1),1)[0]
	max1 = torch.max(tensor.reshape(b,-1),1)[0]

	min1 =min1.reshape(b,1,1,1).expand(tensor.shape)
	max1 =max1.reshape(b,1,1,1).expand(tensor.shape)
	return (tensor-min1)/(max1-min1)

def normalize2(tensor,mean,std):
	tensor.sub_(mean[0]).div_(std[0])
	return tensor

def box_iou(box1, box2):

	def box_area(box):
		return (box[2] - box[0]) * (box[3] - box[1])

	area1 = box_area(box1.t())
	area2 = box_area(box2.t())

	inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)

	return inter / (area1[:, None]+ area2 - inter)  # iou = inter / (area1 + area2 - inter)

def compute_ap(recall, precision): #(390)  (390)  r[0.0025,0.005,...,0.936]  p[1.0,1.0,...,0.93333]


	# Append sentinel values to beginning and end
	mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))

	mpre = np.concatenate(([0.], precision, [0.]))


	# Compute the precision envelope
	mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))#np.maximum.accumulate  计算累计最大值

	#print(mrec)
	#print(mpre)


	# Integrate area under curve
	method = 'interp'  # methods: 'continuous', 'interp'
	if method == 'interp':
		x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
		ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate     mrec和mpre画曲线 ，然后x横坐标，在图像上找对应的输出，返回对应的y坐标   trapz积分
	else:  # 'continuous'
		i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
		ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

	return ap


#测试图片文件夹
test_dir = './test/image'
test_img = os.listdir(test_dir)
test_img.sort()


for img_name in test_img:
	with torch.no_grad():
		#从测试图片名字中提取数字
		num = [x for x in img_name if x.isdigit()]
		num = "".join(num)
		num = int(num)
		print(num)

		#读取测试图片，转换成tensor
		image = Image.open(os.path.join(test_dir,img_name)).convert('L')
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
		thres_index = scores > 0.9
		top_anchor = anchor_[thres_index]

		#目标可能区域个数
		n_ = top_anchor.shape[0]
		if n_ == 0:
			continue

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
		im.save('./test/result/rcn_result/' + img_name)

		#封装tensor
		img0 = torch.empty((n_,48,48))
		transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize(mean, std),])
		for i,img in enumerate(image_list):
			img_nm = transform(img)
			img0[i] = img_nm
		img0 = img0.view(n_,1,48,48).type(torch.cuda.FloatTensor)

		#deblur
		img1 = torch.empty((16,n_,1,20,20))
		for j,anchor in  enumerate(anchor2):
			img1[j] = img0[:,:,anchor[0]:anchor[2],anchor[1]:anchor[3]] #n_ 1 25 25
		img1 = img1.view(16*n_,1,20,20).type(torch.cuda.FloatTensor) #36 1 100 100
		img2 = model_dbgan(img1)

		#通过SRGAN提高分辨率
		img3 = model_SRGAN(img2)# 36 1 80 80
		'''
		for i in range(16*n_):
			img22 = make_grid(img3[i], nrow=1, normalize=True)
			save_image(img22, "./test/result/srgan_result/%d_%d.png" % (num,i), normalize=False)
		'''
		#归一化
		img4 = normalize1(img3)
		img5 = normalize2(img4,mean,std)

		#Discriminator提取特征
		feature = feature_extractor(img5)# 36 256 20 20

		#通过 cls
		cls = model_cls(feature) 
		S = nn.Sigmoid()
		cls = S(cls)

		#index1每张图最大值在第几块索引，index2第几张图有目标索引
		cls1 = cls.reshape(16,n_)
		[max_img,index1] = torch.max(cls1,axis=0)
		index2 = (max_img>0.00000001).nonzero()

		if index2.shape[0] == 0:
			continue
		#根据索引找到feature，通过loc
		a = index1[index2]
		b = a*n_ + index2 
		feature_loc = torch.squeeze(feature[b],axis = 1)
		loc = model_loc(feature_loc)

		#转换为bbox
		loc1 = loc.cpu().numpy()
		bbox1 = loc2bbox(loc1)

		#回归到原图坐标，index1是48*48中第几个块有最大值，，index2第几张图有目标,
		#index2索引top_anchor,index3索引anchor2
		index3 = index1[index2].cpu().numpy()
		index2 = index2.cpu().numpy()

		#转换到20×20
		bbox2 = bbox1/4 

		#转换到48*48
		bbox_48 = np.squeeze(anchor2[index3],axis=1)
		bbox3 = bbox2 + np.tile(bbox_48[:,:2],2)

		#转换到640*512
		bbox_640512 = np.squeeze(top_anchor[index2],axis=1) #已经转换好了坐标是(w1,h1,w2,h2)	
		bbox4 = bbox3[:,[1,0,3,2]] + np.tile(bbox_640512[:,:2],2)
		bbox4 = np.around(bbox4).astype(np.int32)

		#获取实际坐标label
		gt_bbox = []
		label = []
		anno = ET.parse(os.path.join('test', 'anno1',img_name[:-4]+'.xml'))
		for obj in anno.findall('object'):
			bndbox_ = obj.find('bndbox')
			bbox_ = [int(bndbox_.find(tag).text) - 1  for tag in ('ymin','xmin','ymax','xmax')]
			gt_bbox.append(bbox_)
			label.append([1])

		#计算每个预测框匹配的最大iou，并返回在label中的索引
		pred = torch.tensor(bbox4[:,[1,0,3,2]],dtype=torch.float32).to(device)
		gt_bbox = torch.tensor(gt_bbox,dtype=torch.float32).to(device)
		correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)

		ious,i= box_iou(pred, gt_bbox).max(1)


		for j in (ious > iouv[0]).nonzero():
			correct[j] = ious[j] > iouv  # iou_thres is 1xn

		conf = torch.squeeze(cls[b],axis = 1).cpu()
		correct = correct.cpu()
		stats.append((correct, conf,label))












stats = [np.concatenate(x, 0) for x in zip(*stats)]
correct = stats[0]#  每个框True False标签  

conf = stats[1]
conf = conf.reshape(conf.shape[0],)
label = stats[2]

i = np.argsort(-conf)

correct, conf = correct[i], conf[i]


pr_score = 0.3
s = [1, 1]  # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)
ap, p, r = np.zeros(s), np.zeros(s), np.zeros(s)

n_p = conf.shape[0]#预测的框的数量
n_gt = label.sum()#实际的目标数量

fpc = (1 - correct).cumsum(0)  #检测到的错误的个数  递增    查1的个数
tpc = correct.cumsum(0)   #检测到的正确的个数  递增

#Recall
recall = tpc / (n_gt + 1e-16)  # recall curve



r[0] = np.interp(-pr_score, -conf, recall[:, 0])  # r at pr_score, negative x, xp because xp decreases

# Precision
precision = tpc / (tpc + fpc)  # precision curve
p[0] = np.interp(-pr_score, -conf, precision[:, 0])  # p at pr_score




for j in range(correct.shape[1]):
	ap[0, j] = compute_ap(recall[:, j], precision[:, j])

f1 = 2 * p * r / (p + r + 1e-16)
mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()



# Plot
fig, ax = plt.subplots(1, 1, figsize=(5, 5))    #共1×3个图像  返回值fig为总图像，ax为子图像的列表
ax.plot(recall, precision)
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_xlim(0, 1.01)   #x轴的上下限
ax.set_ylim(0, 1.01)   #y轴的上下限
#fig.tight_layout()    #自动调整图像，使他充满整个图像
fig.savefig('PR_curve.png', dpi=300)  #dpi跟可视化相关


np.savez('list1.npz',r=recall,p=precision)



print(map,mf1)












