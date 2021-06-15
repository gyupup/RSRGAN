import glob
import random
import os
import numpy as np
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFilter
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
        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height // 4, hr_width // 4), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                #transforms.Resize((hr_height, hr_width), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.files = glob.glob("./images_gan_small" + "/*.*")


    def __getitem__(self, index):
        file_name = self.files[index % len(self.files)]
        img = Image.open(file_name).convert('L')
  
        img_hr = self.hr_transform(img)
        img_lr = self.lr_transform(img)

        num = [x for x in file_name if x.isdigit()]
        num = "".join(num)
        num = int(num)

        img_blur = Image.open("./blur/"+str(num)+".png").convert('L')
        img_blur = self.hr_transform(img_blur)


        return {"lr":img_lr,"hr":img_hr,"blur":img_blur}

	
      
    def __len__(self):
        return len(self.files)




















