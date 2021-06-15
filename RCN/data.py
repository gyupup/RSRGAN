import os
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class Dataset:
    def __init__(self,txt_dir):
        self.data_dir='./dataset'

        self.id_list_file = os.path.join(self.data_dir,txt_dir)
        id_file = open(self.id_list_file)
        self.id_list = [id_.strip() for id_ in id_file]
        id_file.close()

        #self.label_file = os.path.join(self.data_dir,'label.txt')
        #self.label = [label_.strip() for label_ in open(self.label_file)]
   
    def __getitem__(self, idx):
        id_ = self.id_list[idx]
        bbox=list()
        label=list()

        img_file=os.path.join(self.data_dir,'images',id_ + '.png')
        image=Image.open(img_file).convert('L')#  'RGB'
        img = np.asarray(image, dtype=np.float32)
        #plt.imshow(img,cmap ='gray')
        #plt.show()
        image.close()

        anno = ET.parse(
            os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))#解析xml
        for obj in anno.findall('object'):

            bndbox_ = obj.find('bndbox')
            bbox_ = [int(bndbox_.find(tag).text) - 1  
                     for tag in ('ymin','xmin','ymax','xmax')]
            bbox.append(bbox_)

            #name_=obj.find('name').text.lower().strip()
            #label.append(self.label.index(name_))

        bbox = np.stack(bbox).astype(np.float32)#stack 增加一个维度
        #label = np.stack(label).astype(np.int32)
        transform=transforms.ToTensor()
        img=transform(img)

        return img, bbox.copy()#, label.copy()

    def __len__(self):
        return len(self.id_list)
