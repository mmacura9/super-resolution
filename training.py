import re
from tkinter import Variable
import os
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import Generator, Discriminator, feature_extractor
from dataset import ImageDataSetTrain


os.makedirs("images1", exist_ok=True)
os.makedirs("saved_models1", exist_ok=True)

IMG_SIZE = 128
CHANNELS = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUDA = torch.cuda.is_available()
MAX_SUMMARY_IMAGES = 4
LR = 1e-4
EPOCHS = 30
BATCH_SIZE = 4
NUM_WORKRES = 1
CLIP_VALUE = 1e-2

assert MAX_SUMMARY_IMAGES <= BATCH_SIZE
#Tensor = torch.cuda.FloatTensor if CUDA else torch.Tensor

def train():
    summary_writer = SummaryWriter()

    data_set = ImageDataSetTrain(IMG_SIZE, total_images=5000)
    total_iterations = len(data_set) // BATCH_SIZE
    data_loader = DataLoader(data_set, 
                            batch_size=BATCH_SIZE, 
                            shuffle = True, 
                            num_workers=NUM_WORKRES)

    generator = Generator()
    discriminator = Discriminator()
    feature_extract = feature_extractor()
    feature_extract.eval()

    gan_loss = torch.nn.MSELoss()
    content_loss = torch.nn.L1Loss() 

    generator.to(DEVICE)    
    discriminator.to(DEVICE)
    gan_loss.to(DEVICE)
    content_loss.to(DEVICE)
    feature_extract.to(DEVICE)

    optimizer_g = torch.optim.RMSprop(generator.parameters(),lr=LR)
    optimizer_d = torch.optim.RMSprop(discriminator.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        if epoch < 10:
            c_times = 100
        else:
            c_times = 5

        for i, real_image in tqdm(enumerate(data_loader), total = total_iterations, desc=f"Epoch: {epoch}", unit="batches"):
            global_step = epoch * total_iterations + i

            
            lr_img = torch.autograd.Variable(real_image["lr"])
            hr_img = torch.autograd.Variable(real_image["hr"])
            lr_img = lr_img.to(DEVICE)
            hr_img = hr_img.to(DEVICE)
            valid = torch.ones(4, 1).to(DEVICE)
            fake = torch.zeros(4, 1).to(DEVICE)

            fake_img = generator(lr_img)

            optimizer_d.zero_grad()
            loss_or = gan_loss(discriminator(hr_img), valid)
            loss_new = gan_loss(discriminator(fake_img.detach()), fake)

            loss = (loss_or+loss_new)/2

            summary_writer.add_scalar("Critic loss", loss, global_step)
            loss.backward()
            optimizer_d.step()

            for p in discriminator.parameters():
                p.data.clamp_(-CLIP_VALUE,CLIP_VALUE)

            if i % c_times==0:
                optimizer_g.zero_grad()
                gen_img = generator(lr_img)
                loss1 = 1e-3*gan_loss(discriminator(gen_img), valid)
                fake_f = feature_extract(gen_img)
                real_f = feature_extract(hr_img)
                loss2 = content_loss(fake_f,real_f.detach())
                lossG = loss1*1e-3+loss2

                

                summary_writer.add_scalar("Generator loss", lossG, global_step)

                lossG.backward()
                optimizer_g.step()

                summary_writer.add_images("Generated images", gen_img[:MAX_SUMMARY_IMAGES], global_step)
            

if __name__ =="__main__":
    train()
