import torch
import numpy as np
import os
from Restnet50 import RestNet50
from PIL import Image
from torchvision.ops import nms
import xml.etree.ElementTree as ET



device = torch.device("cuda" if torch.cuda.is_available else "cpu")

model = RestNet50()
model.load_state_dict(torch.load("./result/best.pt"))
model.to(device).eval()

test_dir = './dataset/images'
test_img = os.listdir(test_dir)
test_img.sort()

#anchor
anchor_base = np.array([[0,0,48,48]])
#_, _, height, width = features.shape
height = 60
width = 76
feat_stride = 8

shift_x = torch.arange(0, width * feat_stride, feat_stride)
shift_y = torch.arange(0, height * feat_stride, feat_stride)
shift_x, shift_y = np.meshgrid(shift_x, shift_y)
shift = np.stack((shift_y.ravel(), shift_x.ravel(),shift_y.ravel(), shift_x.ravel()), axis=1)#[0 0 0 0] [0 4 0 4]

anchor = shift + anchor_base #18240 4
anchor = anchor.astype(np.float32)
anchor1 = torch.from_numpy(anchor) #18240 4


for img_name in test_img:
	#从测试图片名字中提取数字
	num = [x for x in img_name if x.isdigit()]
	num = "".join(num)
	num = int(num)

	#读取测试图片，转换成tensor
	image = Image.open(os.path.join(test_dir,img_name)).convert('L')
	#img_y = np.array(image)
	img = np.asarray(image,dtype=np.float32)
	img = torch.from_numpy(img)
	img = img.view(1,1,512,640).to(device)

	#model_RCN得到分数序列
	feature = model(img)
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

	for i in range(n_):
		image_crop = image.crop(top_anchor[i])#裁剪
		image_crop.save('./tiqu/' + str(num)+'_'+str(i)+'.png')
	














































