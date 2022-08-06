import os
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

IMAGENET_MEAN_1 = np.array([0.485, 0.456, 0.406])  # normalizacija jer je vgg19 pretrenirana na ImageNet-u
IMAGENET_STD_1 = np.array([0.229, 0.224, 0.225])


class ImageDataSetTrain(Dataset):
    def __init__(self, img_size, total_images = None):
        self.img_size = img_size
        #project_path = os.path.dirname(__file__)
        #project_path = os.path.abspath(project_path)
        #self.data_root = os.path.join(project_path,"data","img_align_celeba")
        #self.image_paths = glob.glob(os.path.join(self.data_root,"*","*.jpg"))  #PROVERITI DA LI SU SLIKE JPG FORMATA
        self.image_paths = glob.glob("data/img_align_celeba" + "/*.*")
        self.image_paths = self.image_paths[:total_images]
        self.lr_transforms = transforms.Compose(
            [
                transforms.Resize((32,32), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN_1, IMAGENET_STD_1)
            ]
        )
        self.hr_transforms = transforms.Compose(
            [
                transforms.Resize((128,128), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN_1,IMAGENET_STD_1)
            ]
        )
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        lr = self.lr_transforms(image)
        hr = self.hr_transforms(image)
            
        return {"lr": lr, "hr": hr}

