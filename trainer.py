%matplotlib inline
%load_ext autoreload
%autoreload 2

import os,sys
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)



# normalize = transforms.Normalize(mean=[0.22337735, 0.21054482, 0.07122155], std=[0.3154441 , 0.24751204, 0.17515115])
#normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5 , 0.5, 0.5])
trans = transforms.Compose([
   transforms.CenterCrop((300,200)), transforms.ToTensor(),normalize,
])

data_train='CityScapes_Colored/train'
data_valid='CityScapes_Colored/val'
train_set = Pick_images(data_train,trans)
val_set = Pick_images(data_valid,trans)

image_datasets = {
    'train': train_set, 'val': val_set
}

batch_size = 4

dataloaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
}

dataset_sizes = {
    x: len(image_datasets[x]) for x in image_datasets.keys()
}

dataset_sizes


### 

import datetime
import tqdm
import torch.optim as optim
from torch.optim import lr_scheduler
import time

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))



def train_model(model, optimizer, scheduler, criterion ,num_epochs=1):
    losses=[]
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        since = time.time()
        
        # Each epoch has a training and validation phase
        for phase in ['train',"val"]:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                    
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

                epoch_samples = 0
            i=-1
            for images, masks in tqdm.tqdm(dataloaders[phase]):
                
                i+=1
                images = images.to(device)
                masks = masks.to(device)             

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(masks)
                    loss0=criterion(outputs.cuda(),images,0)
                    loss1=criterion(outputs.cuda(),images,1)
                    loss2=criterion(outputs.cuda(),images,2)
                    
                    if (loss0<loss1):
                        if(loss0<loss2):
                            loss=loss0
                        else :
                                loss=loss2
                    else :
                        if(loss1<loss2):
                            loss=loss1
                        else:
                            loss=loss2

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        batch_samples = images.size(0)

                        batch_loss = loss / batch_samples
                        
                        # Write in tensorboard
                        losses.append(batch_loss)
                        
                        ## Put loss on the tensorboard
                        writer.add_scalar('training loss',batch_loss,i+epoch*len(dataloaders["train"])) 
                        if (i%(len(dataloaders["train"])-1)==0): 
                            ## Build output image 
                            mm = np.argmax(masks[0].cpu(), 0)
                            stk = np.stack([mm,mm,mm])
                            msk = torch.tensor(stk)
                            img_grid = torchvision.utils.make_grid([(msk.float()).to(device), outputs[0], images[0]])              ## Put images (mask,output,original image) on the tensorboard
                            writer.add_image("predictions/b_%s"%(i+epoch*len(dataloaders["train"])), img_grid)

                    if phase=='val':
                        batch_samples = images.size(0)

                        batch_loss = loss / batch_samples
                        
                        # Write in tensorboard
        
                        writer.add_scalar('validation loss',batch_loss,i+epoch*len(dataloaders["val"]))

                        
        if(epoch%10==0):
            torch.save(model,"./unet_%i"%epoch) ## save model
    return model , losses


### Launch a training
from loss import *

model = UNet()
model = model.to(device)

criterion = VGGPerceptualLoss() ## Loss fonction 
# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model.parameters(), lr=1e-1) ### Decay

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

model,losses= train_model(model, optimizer_ft, exp_lr_scheduler,criterion, num_epochs=40)