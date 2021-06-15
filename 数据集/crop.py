import os
import glob
import cv2
import xml.etree.ElementTree as ET



files = glob.glob("./训练集/RCN/image" + "/*.*")
print(files)

for file_name in files:
	num = [x for x in file_name if x.isdigit()]
	num = "".join(num)
	num = int(num)

	print(num)


	image = cv2.imread(file_name)

	annotations = os.path.join('./训练集/RCN/RCN_anno',str(num) + '.xml')
	anno = ET.parse(annotations)
	for obj in anno.findall('object'):
		bndbox = obj.find('bndbox')
		bbox = [int(bndbox.find(tag).text) - 1 for tag in ('ymin','xmin','ymax','xmax')]#(h1,w1,h2,w2)

		h = bbox[2] - bbox[0]
		w = bbox[3] - bbox[1]


		if h>20 or w>20:
			continue

		up = int((20-h)/2)
		down = 20 - h - up
		left = int((20-w)/2)
		right = 20 - w - left



		image_crop = image[bbox[0]-up:bbox[2]+down,bbox[1]-left:bbox[3]+right]  #h w

	

		cv2.imwrite("./训练集/cls_loc/plane1/"+str(num)+".png",image_crop)
























