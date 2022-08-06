from turtle import shape
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
import numpy as np


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample= None):
        super(ResidualBlock,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU() #proveriti da li treba neka druga vrednost parametra
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        res = x
        y = self.conv1(x)
        y = self.conv2(y)
        y+=res
        #out = torch.add(y, res)
        return y

class UpsampleBlock(nn.Module):
    def __init__(self, channels):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, 256, kernel_size=3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )
    def forward(self,x):
        return self.upsample(x)


class Generator(nn.Module):
    def __init__(self,res_numbers=16):
        """
        
        """       
        super(Generator,self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 9, stride = 1, padding = 4),
            nn.PReLU()
        )
            
        resBlocks = []
        #residualni blokovi->16 treba da ih ima po originalnom radu
        for _ in range(16):
            resBlocks.append(ResidualBlock(64, 64, 1))
        self.resBlocks = nn.Sequential(*resBlocks)
        
        #self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        #self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64)
        )

        # self.Conv3 = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
        #     nn.PixelShuffle(2),
        #     nn.PRelu()
        # )

        UpSampleBlocks = []
        for _ in range(2):
            UpSampleBlocks.append(UpsampleBlock(64))
        self.UpSampleBlocks = nn.Sequential(*UpSampleBlocks)

        self.conv3 = nn.Conv2d(64,3,kernel_size=9,stride=1,padding = 4)

    
    def forward(self,x):
        y1 = self.conv1(x)      
        
        y = self.resBlocks(y1)        
        y2 = self.conv2(y)        
        y = torch.add(y1,y2)
        y = self.UpSampleBlocks(y)
        y = self.conv3(y)
        
        return torch.clamp_(y, 0.0, 1.0)

class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DiscBlock,self).__init__()
        self.discblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride = stride, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
    def forward(self,x):
        return self.discblock(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3),
            nn.LeakyReLU(0.2)
        )

        disBlocks = []
        disBlocks.append(DiscBlock(64,64,2))
        disBlocks.append(DiscBlock(64,128,1))
        disBlocks.append(DiscBlock(128,128,2))
        disBlocks.append(DiscBlock(128,256,1))
        disBlocks.append(DiscBlock(256,256,2))
        disBlocks.append(DiscBlock(256,512,1))
        disBlocks.append(DiscBlock(512,512,2))

        self.disBlocks = nn.Sequential(*disBlocks)
        
        self.finalBlock = nn.Sequential(
            nn.Linear(512*8*8, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024,1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        y = self.conv1(x)
        y = self.disBlocks(y)
        y = torch.flatten(y, start_dim=1)
        y = self.finalBlock(y)
        return y


class feature_extractor(nn.Module):
    def __init__(self):
        super(feature_extractor,self).__init__()
        model = models.vgg19(True)
        self.feature_extractor = nn.Sequential(*list(model.features.children())[:18])

    def forward(self, image):
        return self.feature_extractor(image)
