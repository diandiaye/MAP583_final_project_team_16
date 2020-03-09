from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets, models
import torch.nn.functional as F 
import torch
from PIL import Image

from utils import one_hot


import os

class Pick_images(Dataset):
    def __init__(self,data_dir, transform=None):
        self.input_images = datasets.ImageFolder(os.path.join(data_dir,"imgs"))
        self.target_masks = datasets.ImageFolder(os.path.join(data_dir,"masks"))     
        self.transform = transform
    
    def __len__(self):
        return len(self.input_images)
    
    def __getitem__(self, idx):  
        image = self.input_images[idx][0]
        image = transforms.Compose([ transforms.CenterCrop((300,200)),transforms.ToTensor(),])(image)
        
        
        
        mask = self.target_masks[idx][0]
        if self.transform :
            mask = self.transform(mask)
        
#        mask =torch.tensor(np.array(mask)[:,:,0]) ## Use for onehoted encoding
        
#         mask = one_hot(mask).permute(2, 0, 1) 
#         mask = Image.fromarray(mask.numpy())
#         mask = transforms.Compose([transforms.ToTensor(),])(mask)
#         import ipdb;ipdb.set_trace()
        
        return image, mask
