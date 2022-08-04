import os
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision

class FacesDataSet(Dataset):
    def __init__(self, img_size, crop_size=128, total_images=None):
        assert img_size <= crop_size <= 250
        self.img_size = img_size
        self.crop_size = crop_size
        project_path = os.path.dirname('../data/mirflickr')
        self.data_root = project_path
        self.image_paths = glob.glob(os.path.join(self.data_root, "**", "*.jpg"))
        self.image_paths = self.image_paths[:total_images]
        self.transforms = transforms.Compose(
            [
                transforms.RandomCrop((self.crop_size, self.crop_size)),
                transforms.Resize(self.img_size),
                transforms.ToTensor()
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path =  self.image_paths[index]
        image = Image.open(image_path)
        return self.transforms(image)
    
    def getimage(self, index):
        image_path =  self.image_paths[index]
        image = Image.open(image_path)
        return self.transforms(image)

if __name__ == "__main__":
    data = FacesDataSet(256, 128, 100)
    for index in range(100):
        img = (data.getimage(index)*256).type(torch.uint8)
        torchvision.io.write_png(img, "C:/Users/psiml8/Desktop/projekat/data_out/im" + str(index+1) + ".png")