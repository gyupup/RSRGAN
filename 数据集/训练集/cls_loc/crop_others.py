import os
import glob
import cv2
import xml.etree.ElementTree as ET
import numpy as np



files = glob.glob("./others/"+"*.*")

for i,image_name in enumerate(files):

	num = [i for i in image_name if i.isdigit()]
	num = "".join(num)
	num = int(num)

	img = cv2.imread(image_name,0)
	img = np.array(img)
	index = np.unravel_index(img.argmax(),img.shape)

	up = index[0]-10
	down = index[0] + 10
	left = index[1] - 10
	right = index[1] + 10

	if index[0]<10:
		up = 0
		down = 20
	elif index[0] >38:
		up = 28
		down = 48

	if index[1]<10:
		left = 0
		right = 20
	elif index[1]>37:
		left = 28
		right = 48
	
	print(up,down,left,right)


	image_crop= img[up:down,left:right]
	cv2.imwrite('./others1/'+str(i)+'.png',image_crop)










