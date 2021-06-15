import glob
import random
import os
import numpy as np
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import time
from torchvision.utils import save_image, make_grid

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.5])
std = np.array([0.5])


class ImageDataset(Dataset):
    def __init__(self):
        hr_height, hr_width = 48,60
        # Transforms for low resolution images and high resolution images

        self.hr_transform = transforms.Compose(
            [
                #transforms.Resize((hr_height, hr_width), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.files = glob.glob("./plane2" + "/*.*")


    def __getitem__(self, index):
        file_name = self.files[index % len(self.files)]
        #print(file_name[9:])
        img = Image.open(file_name).convert('L')
        img_hr = self.hr_transform(img)


        num = [x for x in file_name if x.isdigit()]
        num = "".join(num)
        num = int(num)

        bbox=list()
        label=list()
        
        annotations = os.path.join('./plane_anno', file_name[9:-4]+'.xml')


        if (not os.path.exists(annotations)):
            bbox.append([0,0,0,0])
            label.append(0)
        else:
            anno = ET.parse(annotations)#解析xml
            for obj in anno.findall('object'):

                bndbox_ = obj.find('bndbox')
                bbox_ = [int(bndbox_.find(tag).text) - 1  
                         for tag in ('ymin','xmin','ymax','xmax')]
                bbox.append(bbox_)

                name_=obj.find('name').text.lower().strip()
                label.append(1)

        bbox = np.stack(bbox).astype(np.float32)#stack 增加一个维度
        label = np.stack(label).astype(np.int32)

        return {"hr": img_hr, "bbox": bbox, "label": label}
        


	
      
    def __len__(self):
        return len(self.files)




















