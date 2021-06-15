import os
import glob
import cv2
import xml.etree.ElementTree as ET


files = glob.glob('./test_loc/'+'*.*')
print(files)
for i,image_name in enumerate(files):
	os.rename(image_name,'./test_loc/'+str(i+642)+'.png')











